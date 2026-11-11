"""`pack-video-window`: each night, ages out camera video older than the history window.

The system keeps video like an NVR: a fixed window (`video_window.days`, 30) of footage, and nothing older. Only the
video goes. What was derived from it (detections, spans, captions, transcripts, the journal) stays; the retention
pass below condenses the dense per-frame rows of each day past the window once its day page is written. Audio is
kept, except silent audio past the window.

Every removal is an age-out, never a purge by range: a range purge cascades over provenance and would delete every
detection built from the video, while an age-out removes only the chunks it names (and their absorbed frames' rows),
records their spans as aged out so a reader can say the footage expired, and tells the tier worker to delete the
chunk files. Ages count from when the broker committed a payload, so an import of old media gets the whole window from
its import.

Each run walks every `*.video` stream in commit order from where the last run stopped (a per-stream mark saved under
`paths.state`), up to the newest commit before the cutoff, and ages that out in batches. A night touches about a day of
video per camera.

Journal evidence first. Before it ages anything out, a run promotes the moments the journal's day pages cite
(memory/journal_evidence.py): a short clip, a still, and the caption of each, kept after the video goes. It
runs in the same process ahead of the purge, so a cited moment is always promoted before its video is removed,
including by the disk floor's early hours. A failed promotion is logged as an error and the purge goes on: a full
disk is the worse failure.

Disk floor. If free space on the recordings volume is below `video_window.free_floor_percent`, the run goes on past the
cutoff an hour at a time from the oldest video still kept, until free space is back above the floor, at most
`MAX_EARLY_HOURS` hours a run and never into the newest `MIN_KEEP_S` of video. Each early hour is logged as a warning.

Then the retention pass (`retention.py`): for each ended day past the window whose day page is written, it
condenses the dense coverage and device sightings, drops the per-frame rows, and ages out the audio chunks no speech
touches. A dry run skips it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import continuum_py
from continuum_py import Client

from liberational import config, continuum_link, json_state, logs
from liberational.continuum import to_ns

log = logs.get("video-window")

DAY_S = 86400.0
HOUR_S = 3600.0
#: Ages out are recorded as one span where the gap between two removed payloads is at most this.
AGED_GAP_S = 2.0
#: Seconds of video the disk floor never takes: the newest day stays whatever the disk says.
MIN_KEEP_S = DAY_S
#: Hours of early age-out one run may do for the floor.
MAX_EARLY_HOURS = 48
#: Seconds to wait after an early hour for the tier worker to delete its files before free space is read again.
FREE_SETTLE_S = 30.0
MARKS = "video_window.json"
PAGE = 1000

#: Ages out ids committed before a cutoff (UTC ns) with a reason; returns how many went (a dry run counts).
AgeOut = Callable[[list[str], int, str], Awaitable[int]]
#: Free fraction of the recordings volume, 0 to 1.
Free = Callable[[], float]


@dataclass(frozen=True)
class Settings:
    """`[video_window]` in config/defaults.toml."""

    days: float
    free_floor_percent: float
    batch: int

    @classmethod
    def load(cls) -> Settings:
        return cls(
            days=float(config.get("video_window.days")),
            free_floor_percent=float(config.get("video_window.free_floor_percent")),
            batch=int(config.get("video_window.batch")),
        )


def is_video(stream: str) -> bool:
    return stream.endswith(".video")


class Run:
    """One night's pass over the video streams."""

    def __init__(
        self,
        client: Client,
        state: Path,
        settings: Settings,
        now: float,
        age_out: AgeOut,
        free: Free,
        *,
        dry: bool = False,
        settle: Callable[[], Awaitable[None]] | None = None,
        promote: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        self.client, self.s, self.now, self.age_out, self.free = client, settings, now, age_out, free
        self.promote = promote  # journal evidence, run (in a thread) before anything is aged out
        self.dry = dry  # a dry run counts: it moves no mark
        self.settle = settle or (lambda: asyncio.sleep(FREE_SETTLE_S))
        self.path = state / MARKS
        # A stream's mark is an int sequence the broker takes, so anything else in a hand-edited or truncated file is
        # dropped rather than fed to query_by_seq (and the whole nightly age-out killed). Marks are rewritten by _save.
        saved = json_state.read(self.path, {})
        self.marks: dict[str, int] = {
            str(stream): mark
            for stream, mark in (saved.items() if isinstance(saved, dict) else ())
            if isinstance(stream, str) and isinstance(mark, int) and not isinstance(mark, bool)
        }
        self.counts: dict[str, int] = {}

    def _save(self) -> None:
        if self.dry:
            return
        json_state.write(self.path, self.marks, indent=1, sort_keys=True)

    async def run(self) -> dict[str, int]:
        if self.promote is not None:
            try:
                promoted = await asyncio.to_thread(self.promote)
                self.counts.update({f"promoted_{k}": v for k, v in promoted.items() if v})
            except Exception:
                log.exception("promoting journal evidence failed; aging out anyway")
        streams = sorted(s["name"] for s in await self.client.list_streams() if is_video(s["name"]))
        cutoff = self.now - self.s.days * DAY_S
        await self.upto(streams, cutoff, "older than the video window")
        floor = self.s.free_floor_percent / 100
        early = 0
        while (free := self.free()) < floor and early < MAX_EARLY_HOURS:
            # From the oldest video still kept, so a disk that fills before the window does still frees space.
            oldest = await self.oldest(streams, cutoff)
            if oldest is None or max(cutoff, oldest) + HOUR_S > self.now - MIN_KEEP_S:
                break
            cutoff = max(cutoff, oldest) + HOUR_S
            early += 1
            removed = await self.upto(streams, cutoff, "below the disk floor")
            log.warning(
                "free space %.1f%% is below the %.0f%% floor: aged out %d video payloads up to %s, early",
                free * 100,
                self.s.free_floor_percent,
                removed,
                time.strftime("%Y-%m-%d %H:%M", time.localtime(cutoff)),
            )
            if not self.dry:
                await self.settle()
        if early:
            self.counts["early_hours"] = early
        return self.counts

    async def oldest(self, streams: Sequence[str], after: float) -> float | None:
        """When the oldest video payload still kept past the streams' marks was committed (UTC seconds, by the
        broker's commit clock, the clock `upto` cuts by), or None. A payload's own capture time can be anything: an
        import is backdated, a camera with a bad clock is future-dated. `after` is a time it was committed after."""
        seqs = []
        for stream in streams:
            page = await self.client.query_by_seq(stream, after_seq=self.marks.get(stream, 0), max_items=1)
            seqs += [item["commit_seq"] for item in page["items"]]
        if not seqs:
            return None
        oldest, lo, hi = min(seqs), after, self.now
        if await self.client.commit_seq_at(to_ns(hi)) < oldest:
            return None  # committed after now by the broker's clock: nothing the floor may take
        # commit_seq_at only grows with time, so bisect for the first second by which the oldest was committed.
        while hi - lo > 1.0:
            mid = (lo + hi) / 2
            if await self.client.commit_seq_at(to_ns(mid)) >= oldest:
                hi = mid
            else:
                lo = mid
        return hi

    async def upto(self, streams: Sequence[str], cutoff: float, why: str) -> int:
        """Ages out every payload on `streams` committed before `cutoff` and after each stream's mark."""
        fence = await self.client.commit_seq_at(to_ns(cutoff))
        removed = 0
        for stream in streams:
            after = self.marks.get(stream, 0)
            while after < fence:
                page = await self.client.query_by_seq(stream, after_seq=after, max_items=PAGE)
                within = [item for item in page["items"] if item["commit_seq"] <= fence]
                ids = [str(continuum_py.decode_header(item["header"])["id"]) for item in within]
                for i in range(0, len(ids), self.s.batch):
                    n = await self.age_out(ids[i : i + self.s.batch], to_ns(cutoff), f"video {why}: {stream}")
                    self.counts[stream] = self.counts.get(stream, 0) + n
                    removed += n
                more = page.get("truncated") and len(within) == len(page["items"]) and within
                # Sequences are global, so once this stream has nothing left by the fence, the fence is its mark.
                after = within[-1]["commit_seq"] if more else fence
                self.marks[stream] = after
                self._save()
        return removed


async def silent_audio(
    client: Client, state: Path, day: str, committed_before_ns: int, age_out: AgeOut
) -> dict[str, int]:
    """Ages out `day`'s audio chunks that no speech touches (retention.py); returns how many went per stream."""
    from liberational import retention
    from liberational import segment_log as S
    from liberational.memory_log import day_bounds

    a, b = day_bounds(day)
    margin = retention.SPEECH_MARGIN_S
    segments = S.at(state)
    utterances = segments.range(S.UTTERANCES, a - margin, b + margin)
    # Only the seconds the gate heard speech, from three columns: a day is about 100,000 gate rows.
    gates = segments.table(S.SPEECH_ACTIVITY, a - margin, b + margin, ("ts_ns", "stream", "in_speech"))
    activity = [
        {"ts_start": S.secs(ts), "stream": stream, "in_speech": True}
        for c in gates.values()
        for ts, stream, heard in zip(c["ts_ns"].tolist(), c["stream"], c["in_speech"].tolist(), strict=True)
        if heard
    ]
    counts: dict[str, int] = {}
    for stream in sorted(s["name"] for s in await client.list_streams() if s["name"].endswith(".audio")):
        page = await client.query(stream, to_ns(a), to_ns(b), all=True)
        chunks = []
        for item in page["items"]:
            h = continuum_py.decode_header(item["header"])
            chunks.append((str(h["id"]), int(h["start_ns"]) / 1e9, int(h["end_ns"]) / 1e9))
        ids = retention.silent(chunks, retention.speech_spans(utterances, activity, stream))
        n = 0
        for i in range(0, len(ids), PAGE):
            n += await age_out(ids[i : i + PAGE], committed_before_ns, f"silent audio past the window: {stream}")
        if chunks:
            counts[f"{stream} chunks"] = len(chunks)
            counts[f"{stream} silent dropped"] = n
    return counts


def free_fraction(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / usage.total


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-video-window`: ages out camera video older than the window (see the module docstring)."""
    parser = argparse.ArgumentParser(prog="pack-video-window", description=main.__doc__)
    parser.add_argument("--dry-run", action="store_true", help="count what would go; remove nothing")
    parser.add_argument("--socket", type=Path, default=None, help="broker socket (default: paths.state/continuum)")
    args = parser.parse_args(argv)
    logs.setup()
    if not config.get("video_window.enabled"):
        log.info("the video window is off ([video_window] enabled)")
        return
    state = Path(config.path("paths.state"))
    recordings = state / "continuum"
    settings = Settings.load()
    now = time.time()

    def promote() -> dict[str, int]:
        from liberational.memory import journal_evidence

        return journal_evidence.promote_window(state, now, settings.days)

    days: list[str] = []
    kept: dict[str, dict[str, int]] = {}
    marks = None
    if not args.dry_run:  # retention: the segment side first, outside the broker session below
        from liberational import retention
        from liberational import segment_log as S

        marks = retention.Marks.load(state)
        segments = S.root(state)
        store = retention.Store(segments)
        try:
            days = retention.due(
                retention.first_day(segments), marks.days, now, settings.days, retention.day_written(state)
            )
            for day in days:
                kept[day] = retention.segments_day(store, day)
        except Exception:  # retention must never stop the video window: the day is done again next night
            log.exception("retention: a day's segments failed; the video window goes on")
            days = [d for d in days if d in kept]

    async def go() -> dict[str, int]:
        socket = str(args.socket or continuum_link.socket_path())
        async with await Client.connect(socket, client_name="video-window") as client:

            async def age_out(ids: list[str], before_ns: int, reason: str) -> int:
                if args.dry_run:
                    return len(ids)
                if not ids:
                    return 0
                done = await client.age_out(ids, before_ns, reason=reason, detail="{}", aged_gap_ns=to_ns(AGED_GAP_S))
                return sum(n for _, _, n in done["removed"])

            run = Run(
                client,
                state,
                settings,
                now,
                age_out,
                lambda: free_fraction(recordings),
                dry=args.dry_run,
                promote=None if args.dry_run else promote,
            )
            counts = await run.run()
            for day in days:
                kept[day] |= await silent_audio(client, state, day, to_ns(now - settings.days * DAY_S), age_out)
            return counts

    began = time.monotonic()
    counts = asyncio.run(go())
    verb = "would age out" if args.dry_run else "aged out"
    log.info("%s %s in %.0f s", verb, json.dumps(counts, sort_keys=True), time.monotonic() - began)
    if marks is not None:
        from liberational import retention

        for day in days:
            retention.report(day, kept[day])
            marks.done(day, kept[day])
