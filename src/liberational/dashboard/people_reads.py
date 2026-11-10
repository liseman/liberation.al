"""The People tab's reads: the list of people, a person's card and summary, the Inbox's unnamed list, and why a
line or a face has its name (moved out of the server, #61).

GET /people                 everyone, with the cards most likely opened next read ahead
GET /people/<name>          a person's summary
GET /people/<name>/card     a person's contact card
GET /people/references      a person's reference media, and GET /people/reference-media/<id> one file
GET /inbox/unnamed          the faces, voices, and lines waiting for a name
GET /identity/why           why a line or a recorded face has its name
GET /health                 the dashboard is up, and how far the People warm-up has got
"""

import collections
import copy
import datetime as dt
import multiprocessing
import os
import pathlib
import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from types import EllipsisType
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import logs, memtrim, speech_log, vision_log, voice_turn
from liberational.dashboard import (
    call_log,
    context,
    devreload,
    inbox_state,
    known_voices,
    live_status,
    people_list,
    photo_files,
    routes,
    transcript_lines,
    unnamed_groups,
    voice_clones,
)
from liberational.dashboard.context import SSE_CLIENTS, log
from liberational.dashboard.unnamed_groups import CARD_SHOTS, CARD_VIDEOS, CARD_VOICES
from liberational.memory import remembered
from liberational.people import card_prefs, face_shots, registry, transcript_view, video_thumbs
from liberational.see.faces import VISIT_GAP_S
from liberational.vision import views

try:
    import numpy as np
except ImportError:
    np = None

if TYPE_CHECKING:
    from liberational.dashboard.server import H

UNNAMED_FACES, UNNAMED_VOICES, UNNAMED_VIDEOS = 24, 12, 12  # how many of each the unnamed list offers


def mtime(p: pathlib.Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


def lines_near(e: dict[str, Any], n: int = 200) -> list[dict[str, Any]]:
    """The n lines stored on each side of e in the transcript, where another mic's copy of it is: written a
    moment apart live, or together by a re-transcription."""
    key = registry.key_of(e)
    after: collections.deque[dict[str, Any]] = collections.deque(maxlen=n)
    before: list[dict[str, Any]] = []
    found = False
    for x in transcript_lines.iter_back():
        if not found:
            if isinstance(x.get("ts_start"), int | float) and registry.key_of(x) == key:
                found = True
            else:
                after.append(x)
            continue
        before.append(x)
        if len(before) >= n:
            break
    return [*after, *before] if found else []


MEMORY_FILE = context.STATE / "memory.jsonl"


def _ready_videos(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    revision = photo_files._shot_revision()
    saved = video_thumbs.saved(context.STATE, revision)
    ready, pending, unavailable = [], 0, 0
    for row in rows:
        status, thumb = unnamed_groups._video_status(row, revision, saved)
        if status == "ready":
            ready.append({**row, "thumb": thumb})
        pending += status == "pending"
        unavailable += status == "unavailable"
    return ready, pending, unavailable


def unknown_groups_view() -> tuple[int, dict[str, Any]]:
    """(code, reply) for GET /faces/unknown/groups: Live's "Who's this?" rows (vision/face_guess.py), best guess
    first, how many, and the groups put aside (so Live leaves them out when they come into view). Kept (_Fresh): new
    sightings come in within GROUPS_REFRESH_S, the owner's own changes at once."""
    return unnamed_groups._ready_face_groups(*unnamed_groups._FACE_GROUPS.get())


def unknown_voices_view() -> tuple[int, dict[str, Any]]:
    """(code, reply) for GET /voices/unknown/groups: the unnamed voices waiting for a name as "Who's this?" rows
    (people/voice_guess.py), best guess first, each with its most useful clips to play, and how many. The voices
    /people lists as unknown.voices (unnamed, a line left to play, not put aside) that have a clip worth playing, the
    24 heard most recently. Guessed against the named voices that are people, not voice clones made from one uploaded
    recording (the live matcher leaves those out too). Live polls this, and it reads the whole transcript,
    so the rows are kept (_Fresh): a line heard since is taken in in the background, the owner's own changes at
    once."""
    return unnamed_groups._VOICE_GROUPS.get()


class Starting(Exception):
    """A route can't answer yet but will shortly (the people list's first read): the handler answers 503 with
    `starting`, which the frontend waits out."""


PEOPLE_WAIT_S = 2.0  # a /people request waits this long for its read, then answers from the last one


# While only speech moved the key, the last roster read is refreshed at most this often. A read is about a
# second of Python holding the GIL, and every request after a speech commit started one (every 30 to 60 s). A name,
# a contact, a skip, or a purge still reads at once: it moves `fresh`.
PEOPLE_REFRESH_S = 120.0


class _PeopleReads:
    """Share only overlapping roster computations. A read runs on its own thread, so a request that can't wait for
    it (a cold start's 7 to 30 s) answers from the last roster read, marked `stale`, or raises Starting when
    there is none yet; the read goes on and the next request has it.

    `fresh` is the part of the key a person's own change moves (a name given, a contact added). While it matches the
    last read, a request answers from that read at once and one read at a time refreshes it in the background (a
    later request starts the next), for new speech and for what the key doesn't cover, such as new unknown faces.
    Speech moves the key every few seconds, and relabeling the transcript for every request made each one wait.
    A changed `fresh` still waits for its own read, so an edit shows in the answer to it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pending: dict[tuple[Any, ...], Future[dict[str, Any]]] = {}
        self.last: dict[str, Any] | None = None
        self.last_fresh: tuple[Any, ...] | None = None
        self.cold_read: Future[dict[str, Any]] | None = None  # the shared read before any has finished (`cold`)
        self.last_at = 0.0  # time.monotonic() when the last read finished

    def get(
        self,
        key: tuple[Any, ...],
        read: Callable[[], dict[str, Any]],
        wait_s: float | None = None,
        fresh: tuple[Any, ...] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            last = self.last if fresh is not None and self.last_fresh == fresh else None
            future = self.pending.get(key)
            recent = last is not None and time.monotonic() - self.last_at < PEOPLE_REFRESH_S
            if future is None and (last is None or not self.pending) and not recent:
                future = self.pending[key] = Future()
                threading.Thread(
                    target=self._read, args=(key, fresh, future, read), name="people-read", daemon=True
                ).start()
        if last is not None:
            return copy.deepcopy(last)
        assert future is not None  # only a reusable last read skips starting one
        try:
            return copy.deepcopy(future.result(wait_s))
        except TimeoutError:
            if future.done():
                raise  # the read itself timed out (a broker timeout): not ours to hide
            with self.lock:
                last = self.last
            if last is None:
                raise Starting("The people list is still loading.") from None
            return {**copy.deepcopy(last), "stale": True}

    def peek(self, fresh: tuple[Any, ...]) -> dict[str, Any] | None:
        """The last read while `fresh` still matches it, starting no refresh: for a page that shows one row of it (a
        person's), which the list's own requests keep fresh. Shared, not copied: read it, never change it."""
        with self.lock:
            return self.last if self.last is not None and self.last_fresh == fresh else None

    def cold(self, read: Callable[[], dict[str, Any]], wait_s: float) -> dict[str, Any] | None:
        """Before any read has finished (a cold start), answer without working out the key: building it
        reads the registry, which the start's warm-ups keep busy for seconds, so "still loading" took 6 to 10 s.
        One background `read` is shared; it answers when it finishes within `wait_s`, else Starting. None once a
        read has finished: the key's normal path applies from then on."""
        with self.lock:
            if self.last is not None:
                return None
            if self.cold_read is None or self.cold_read.done() and self.cold_read.exception() is not None:
                future: Future[dict[str, Any]] = Future()
                self.cold_read = future

                def run() -> None:
                    try:
                        future.set_result(read())
                    except BaseException as error:
                        future.set_exception(error)

                threading.Thread(target=run, name="people-cold", daemon=True).start()
            future = self.cold_read
        try:
            return copy.deepcopy(future.result(wait_s))
        except TimeoutError:
            if future.done():
                raise  # the read itself timed out (a broker timeout): not ours to hide
            raise Starting("The people list is still loading.") from None

    def _read(
        self,
        key: tuple[Any, ...],
        fresh: tuple[Any, ...] | None,
        future: Future[dict[str, Any]],
        read: Callable[[], dict[str, Any]],
    ) -> None:
        # Out of `pending` before the result is set: a request that sees it done must not join a finished read.
        try:
            value = read()
        except BaseException as error:
            with self.lock:
                del self.pending[key]
            future.set_exception(error)
        else:
            with self.lock:
                del self.pending[key]
                self.last, self.last_fresh, self.last_at = value, fresh, time.monotonic()
            future.set_result(value)


_PEOPLE_READS = _PeopleReads()


def people_view(wait_s: float | None | EllipsisType = ..., *, cold: bool = False) -> dict[str, Any]:
    """GET /people. While nobody's names, contacts, labels, or skips have changed and nothing was purged since the
    last read, it answers from that read and refreshes it in the background; otherwise it waits up to `wait_s`
    (PEOPLE_WAIT_S when left out, None waits it out) for a read of the changed inputs, then answers from the last one
    (_PeopleReads). With `cold` (the plain GET only), before any read has finished it answers without working out
    the key (`_PeopleReads.cold`); a change's own answer (a name, a skip, a removal) never does, so it can't be the
    roster from before the change."""
    wait = PEOPLE_WAIT_S if isinstance(wait_s, EllipsisType) else wait_s
    if cold and wait is not None and (got := _PEOPLE_READS.cold(lambda: people_view(wait_s=None), wait)) is not None:
        return got
    commits, purge = speech_log.shared().mark()
    fresh = _people_fresh(purge)
    return _PEOPLE_READS.get((*fresh, commits), _people_view_worked_out, wait, fresh)


_people_executor: ProcessPoolExecutor | None = None


_people_lock = threading.Lock()


def _people_view_job() -> dict[str, Any]:
    """The people list's full read, in its worker process (_people_view_worked_out)."""
    if multiprocessing.parent_process() is not None:
        cfg.load.cache_clear()  # a setting saved since: the dashboard rereads its pack.toml, and so does this
        photo_files._low_priority(10)
        memtrim.start()  # what a refresh freed goes back to the system
    return _people_view()


def _people_view_worked_out() -> dict[str, Any]:
    """The people list's full read (_people_view), worked out in one process of its own, as the face and
    voice groups are. It relabels the retained speech for the roster's counts and the unnamed voices' clips: 1.6 to 4 s
    of pure Python every PEOPLE_REFRESH_S while a page polls /people, holding the interpreter lock every request
    needs. _PeopleReads stays in the dashboard, so a person's own change still waits for its own read. A worker that
    died, or ran past GROUPS_WORKER_TIMEOUT_S, is replaced on the next read, and this one fails as a read does."""
    global _people_executor
    if not unnamed_groups._groups_in_worker:
        return _people_view()
    with _people_lock:
        if _people_executor is None:
            _people_executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        executor = _people_executor
    try:
        return executor.submit(_people_view_job).result(timeout=unnamed_groups.GROUPS_WORKER_TIMEOUT_S)
    except (BrokenProcessPool, TimeoutError) as error:
        with _people_lock:
            if _people_executor is executor:
                _people_executor = None
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            process.terminate()  # one still running past the timeout
        executor.shutdown(wait=False, cancel_futures=True)
        log("people: the people list worker stopped:", repr(error), level=logs.WARN)
        raise RuntimeError("The people list could not be read.") from error


def _people_fresh(purge: object) -> tuple[Any, ...]:
    """The part of the people list's key a person's own change moves (_PeopleReads)."""
    # A purge is fresh too: what someone deleted never comes back from the last read.
    return (
        unnamed_groups._voices_key(),
        unnamed_groups._faces_owner_key(groups=False),
        purge,
        str(people_list.CONTACTS_FILE),  # which state this is: a different one never answers from another's read
        mtime(people_list.CONTACTS_FILE),
        mtime(people_list.PEOPLE_ADDED),
        mtime(inbox_state.INBOX_SKIPS),
        cfg.owner_name(),
        cfg.owner_label(),
        cfg.assistant_name(),
    )


def _people_view() -> dict[str, Any]:
    """GET /people: the roster, plus voices and faces nobody has named yet (skipped ones left out)."""
    raw, view = speech_log.retained(), transcript_lines._view()
    skips = inbox_state.inbox_skips()
    sp = known_voices._speakers()["speakers"]
    # Only voices with a line left to play: the backfill and refine move lines to named people, which can leave an
    # unnamed voice with nothing behind it.
    audible = unnamed_groups.speakers_with_audio(raw, view)
    voices = [
        {"id": s["id"], "label": s["label"], "n": s.get("n", 0), "last_seen": s.get("last_seen")}
        for s in sp
        if people_list.VOICE_GROUP.fullmatch((s.get("label") or "").strip())
        and s["id"] in audible
        and s["id"] not in skips["voice"]
    ]
    try:
        faces = [
            {**g, "label": f"Face {g['id']}"}
            for g in people_list._faces_mod().unknown_groups()
            if g["count"] >= 2 and g["id"] not in skips["face"]
        ][:24]
    except Exception:
        faces = []
    voices.sort(key=lambda v: -(v["last_seen"] or 0))
    _photos_from_scene()
    try:
        from liberational.people import face_photos

        face_photos.refresh_async()  # photos for unnamed faces cut from camera frames, in the background
    except Exception as ex:
        log("face photos: not refreshed:", logs.reason(ex), level=logs.WARN)
    for f in faces:
        f["photo"] = _photo_url(photo_files._unknown_photo(f["id"]), f"/faces/unknown/{f['id']}.jpg")
    clips, total = unnamed_groups.voice_clips(voices, raw=raw, view=view)
    return {
        "people": people_list.people_roster(raw, view, sp),
        "unknown": {"voices": voices, "faces": faces, "voice_clips": clips, "voice_clip_total": total},
    }


MIN_PHOTO_PX = 24  # smaller faces (frame pixels) are usually photos on a wall, not someone in the room


def _photo_url(path: pathlib.Path, url: str) -> str | None:
    try:
        return f"{url}?v={path.stat().st_mtime_ns}"
    except OSError:
        return None


def _frame_box(
    frame: Any, bbox_norm: Sequence[float], metadata: dict[str, Any] | None = None
) -> tuple[float, float, float, float]:
    """A box given as shares of the frame, in the frame's pixels."""
    h, w = frame.shape[:2]
    from liberational.see.frames import source_box_to_decoded

    inside = face_shots.in_frame(bbox_norm)  # a face at the frame's edge has a box past it
    if inside is None:
        raise ValueError("The box is outside the frame")
    box = source_box_to_decoded(inside, metadata) if metadata is not None else inside
    if box is None:
        raise ValueError("The source frame geometry is unavailable")
    x1, y1, x2, y2 = (float(v) for v in box)
    return x1 * w, y1 * h, x2 * w, y2 * h


def _crop(cam: str, ts: float, bbox_norm: Sequence[float], dest: pathlib.Path, size: int = 192) -> bool:
    """Square face crop of a face found on `cam` at `ts` (its box as shares of the frame), cut from that frame on the
    vision log and saved as a JPEG. False when the frame is gone, its geometry is unknown, or the face is too small."""
    from PIL import Image

    try:
        evidence = vision_log.at(context.STATE).frame_evidence(cam, ts)
        frame = evidence.image if evidence is not None else None
    except Exception as e:
        log("faces: the frame couldn't be read:", e, level=logs.WARN)
        return False
    if evidence is None or frame is None:
        return False
    try:
        x1, y1, x2, y2 = _frame_box(frame, bbox_norm, evidence.metadata)
    except ValueError:
        return False  # the box can't be placed on this frame

    side = min(x2 - x1, y2 - y1)
    if side < MIN_PHOTO_PX:
        return False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
    img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))  # ty: ignore[unresolved-attribute]  # numpy is a declared dependency
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("." + dest.name)
    img.crop(box).resize((size, size), Image.Resampling.LANCZOS).save(tmp, "JPEG", quality=88)
    os.replace(tmp, dest)
    return True


_scene_photos = {"t": 0.0}


def _living_name(view: transcript_view.View, name: str) -> bool:
    """Is `name` what a living person is called now, or a name the registry never had (a face tagged before the
    backfill)? False for the old name of someone renamed, and for someone forgotten."""
    if (person := view.person_at(name)) is not None:
        return registry.norm(person.display_name) == registry.norm(name)
    return not view.forgotten_at(name)


def _photos_from_scene() -> None:
    """Crops for faces in view now: named people with no photo yet, and unnamed groups (refreshed each minute). Only
    faces found in the last 30 seconds count, so a face named for someone deleted or renamed since gets no photo."""
    if time.time() - _scene_photos["t"] < 10:
        return
    _scene_photos["t"] = time.time()
    now, done, view = time.time(), set(), None
    for f in reversed(people_list._live_faces(30, now)):
        n, bbox = f["label"].strip(), f.get("bbox_norm")
        if not bbox or people_list._norm(n) in done or people_list._norm(n) == "unknown":
            continue
        done.add(people_list._norm(n))
        cam = str(f.get("cam") or "")
        try:
            if people_list.FACE_GROUP.fullmatch(people_list._norm(n)):
                p = photo_files._unknown_photo(people_list._norm(n).split()[1])
                if not p.exists() or now - p.stat().st_mtime > 60:
                    _crop(cam, f["ts"], bbox, p)
            elif not photo_files._photos(n):
                view = view or transcript_lines._view()
                if _living_name(view, n):
                    _crop(cam, f["ts"], bbox, photo_files._photo_dir(n) / f"{int(now * 1000)}.jpg")
        except OSError as ex:
            log("people: photo crop failed:", ex, level=logs.ERROR)


def _roster_row(name: str) -> dict[str, Any] | None:
    """Their row of the people list, from the list's last read: opening a person reads the roster GET /people already
    answered from, the way a pet's page reads the Pets list, instead of building a roster of its own (2 to 10 s on a
    busy machine), and starts no refresh of it. After a change to names or contacts, it waits for the list's
    own read of the change (people_view); before any read has finished, it builds one."""
    last = _PEOPLE_READS.peek(_people_fresh(speech_log.shared().mark()[1]))
    if last is not None:
        row = next((p for p in last["people"] if people_list._norm(p["name"]) == people_list._norm(name)), None)
        return copy.deepcopy(row)
    try:
        people = people_view()["people"]
    except Starting:
        people = people_list.people_roster()
    return next((p for p in people if people_list._norm(p["name"]) == people_list._norm(name)), None)


def person_summary(name: str) -> dict[str, Any] | None:
    """GET /people/<name>: their row of the people list (voices, faces, number, role, last heard and seen), how many
    lines they said (in the room and on calls), and what memory notes say about them."""
    if not call_log._safe_name(name):
        return None
    me = _roster_row(name)
    who = {
        k: me[k]
        for k in (
            "voices",
            "faces",
            "number",
            "owner",
            "enrolled",
            "last_heard",
            "last_seen",
            "last_seen_pending",
            "last_seen_error",
            "role",
            "organization",
            "aliases",
        )
        if me and k in me
    }
    if not me or not me.get("lines"):
        return {"name": me["name"] if me else name, "lines": 0, **who}
    memory = [text for text in remembered.items(MEMORY_FILE) if people_list._norm(name) in text.lower()]
    return {
        **who,
        "name": me["name"],
        "lines": me["lines"],
        "room_lines": me.get("room", 0),
        "call_lines": me.get("calls", 0),
        "memory": memory[-8:],
        "note": f"Built only from transcripts and notes {cfg.assistant_name()} has. Public-web lookups are not run automatically.",
    }


def _what(cam: str, start: float, end: float, name: str | None = None) -> str | None:
    """A short label for what happens in a clip ("Sitting on a chair, hand over mouth, glasses on."), from the
    camera's caption, with no subject and no name (memory/clip_labels.py). Only a cached one: an uncached clip is
    queued for a background rewrite and gets None now, so the card never waits on the model."""
    from liberational.memory import clip_labels

    try:
        return clip_labels.at(context.STATE).get(cam, start, end, name)
    except Exception as ex:  # a label is a nicety: the card shows the time and place without one
        log("card: clip label failed:", repr(ex), level=logs.WARN)
        return None


def _review_link(name: str, ts: float) -> str:
    """Review at that moment with the person picked, in the hash Review's url.ts reads."""
    q = urllib.parse.urlencode(
        {"person": name, "day": dt.datetime.fromtimestamp(ts).date().isoformat(), "t": f"{ts:.2f}"},
        quote_via=urllib.parse.quote,
    )
    return "/review#" + q


def _recent_voice(
    name: str,
    limit: int = CARD_VOICES,
    skip: Collection[str] = (),
    *,
    raw: list[dict[str, Any]] | None = None,
    view: transcript_view.View | None = None,
    named_lines: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The person's newest room lines with words, newest first: [{line, ts_start, text, mic}]. line is the line key
    (registry.key_of); the frontend plays it by ts_start until /clip takes line keys."""
    nm, out = people_list._norm(name), []
    if named_lines is None:
        lines = raw[-people_list.ROOM_TAIL :] if raw is not None else transcript_lines.heard_tail(people_list.ROOM_TAIL)
        view = view if view is not None else transcript_lines._view()
        named_lines = view.named(lines)
    for e in reversed(named_lines):
        if (
            people_list._norm(e.get("speaker") or "") == nm
            and (e.get("text") or "").strip()
            and isinstance(e.get("ts_start"), int | float)
            and registry.key_of(e) not in skip
        ):
            device = e.get("device")
            out.append(
                {
                    "line": registry.key_of(e),
                    "ts_start": e["ts_start"],
                    "text": e["text"].strip(),
                    "mic": device if isinstance(device, str) and device.strip() else None,
                }
            )
            if len(out) >= limit:
                break
    return out


def person_card(name: str) -> dict[str, Any] | None:
    """GET /people/<name>/card: the contact card's data (contract in the docstring at the top of this file), or None
    for a name that can't be one."""
    from liberational.vision import sightings

    if not call_log._safe_name(name):
        return None
    # Share raw speech and one fresh identity snapshot across roster and media readers. Cache only raw rows:
    # retained() follows commits and purges, while a new View sees renames without waiting for new speech.
    raw, view = speech_log.retained(), transcript_lines._view()
    named_lines = view.named(raw[-people_list.ROOM_TAIL :])  # for recent voices
    saved_shots = face_shots.saved(context.STATE, revision=photo_files._shot_revision())
    me = _roster_row(name)
    name = me["name"] if me else name.strip()
    prefs = card_prefs.get(context.STATE, name)
    ready, shots_pending, shots_unavailable = people_list._ready_faces(
        [r for r in sightings.ranked(name, context.STATE) if r["id"] not in prefs["hidden"]["shot"]], saved_shots
    )
    hidden_videos = set(prefs["hidden"]["video"])
    named = views.names()
    videos = []
    for v in sightings.recent_visits(name, CARD_VIDEOS + len(hidden_videos), context.STATE):
        if _video_id(v["cam"], v["start"]) in hidden_videos or len(videos) >= CARD_VIDEOS:
            continue
        thumb = "/people/video-thumb.jpg?" + urllib.parse.urlencode(
            {"cam": v["cam"], "start": f"{v['start']:.3f}", "end": f"{v['end']:.3f}"}
        )
        videos.append(
            {
                "id": _video_id(v["cam"], v["start"]),
                "cam": v["cam"],
                "camera_name": named.get(v["cam"], v["cam"]),
                "start": v["start"],
                "end": v["end"],
                "duration_s": round(v["end"] - v["start"], 1),
                "thumb": thumb,
                "what": _what(v["cam"], v["start"], v["end"], name),
                "review": _review_link(name, v["start"]),
            }
        )
    videos, videos_pending, videos_unavailable = _ready_videos(videos)
    seen = me.get("last_seen") if me else None
    cam = me.get("last_seen_cam") if me else None
    return {
        "name": name,
        "photo": people_list.main_photo(name, ready=ready, saved_shots=saved_shots),
        "shots": [
            people_list._shot(r)
            for r in sightings.best_spread(
                name, CARD_SHOTS, context.STATE, skip=prefs["hidden"]["shot"], ready={r["id"] for r in ready}
            )
        ],
        **({"shots_pending": shots_pending} if shots_pending else {}),
        **({"shots_unavailable": shots_unavailable} if shots_unavailable else {}),
        "videos": videos,
        **({"videos_pending": videos_pending} if videos_pending else {}),
        **({"videos_unavailable": videos_unavailable} if videos_unavailable else {}),
        "voices": _recent_voice(name, skip=prefs["hidden"]["voice"], named_lines=named_lines),
        "last_seen": {"ts": seen, "cam": cam, "camera_name": me.get("last_seen_camera_name") if me else None}
        if seen
        else None,
        "here_now": me.get("here_now") if me else None,
    }


CARD_REFRESH_S = 5.0  # a card older than this is answered as is and read again in the background


CARD_PENDING_REFRESH_S = 1.0  # sooner while crops or thumbnails are still being made, so progress shows on each poll


CARD_WARM = 8  # cards read ahead when the people list is served: the first this many people in it


class _CardReads:
    """The last card read per person, so opening one answers from it at once, the way the people list does
    (_PeopleReads). A card older than CARD_REFRESH_S is answered as is, marked `refreshing` so the page asks again, and
    read again in the background, one read per
    person at a time. A card is read on the request only the first time, or when its key moved: names, contacts, or a
    purge (the people list's fresh key), a forget (the segment store's revision), or card_prefs.json. Every People
    POST clears them all (CARD_EDITS), and a read that started before a clear is never kept. When GET /people is
    served, the first CARD_WARM people's cards that aren't cached under the current key are read in the background, so
    the first open is a cached one too. A cached card isn't read again ahead however old it is: its open answers from
    it, marked `refreshing`, and reads it behind (reading the first eight again every 5 minutes, whether or not
    anyone opened them, was about 2% of a core while a page polled the list)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last: dict[str, tuple[tuple[Any, ...], float, dict[str, Any]]] = {}  # norm name: (key, read at, card)
        self.reading: set[str] = set()
        self.epoch = 0  # moved by clear(): a read that began before it is dropped
        self.warming = False

    def get(self, name: str) -> dict[str, Any] | None:
        if not call_log._safe_name(name):
            return None
        k, key, now = people_list._norm(name), _card_key(), time.time()
        with self.lock:
            hit = self.last.get(k)
        if hit is not None and hit[0] == key:
            pending = hit[2].get("shots_pending") or hit[2].get("videos_pending")
            old = now - hit[1] >= (CARD_PENDING_REFRESH_S if pending else CARD_REFRESH_S)
            if old:
                self._refresh(name)
            with self.lock:
                reading = k in self.reading
            card = copy.deepcopy(hit[2])
            # An older card is answered as is, marked so the page asks again for the one read behind it (a
            # complete card polled no more, and showed the old one for the whole visit).
            return {**card, "refreshing": True} if old or reading else card
        return self._read(name)

    def _read(self, name: str) -> dict[str, Any] | None:
        with self.lock:
            epoch = self.epoch
        key, at = _card_key(), time.time()
        card = person_card(name)
        if card is not None:
            with self.lock:
                if self.epoch == epoch:
                    self.last[people_list._norm(name)] = (key, at, copy.deepcopy(card))
        return card

    def _refresh(self, name: str) -> None:
        k = people_list._norm(name)
        with self.lock:
            if k in self.reading:
                return
            self.reading.add(k)

        def run() -> None:
            try:
                self._read(name)
            except Exception as ex:
                log("card: background read failed:", repr(ex), level=logs.WARN)
            finally:
                with self.lock:
                    self.reading.discard(k)

        threading.Thread(target=run, name="card-read", daemon=True).start()

    def warm(self, names: list[str]) -> None:
        """Read the cards of these people that aren't cached under the current key, one at a time, in the
        background; a warm already running is left to finish."""
        with self.lock:
            if self.warming:
                return
            self.warming = True

        def run() -> None:
            try:
                for name in names:
                    key = _card_key()
                    with self.lock:
                        hit = self.last.get(people_list._norm(name))
                    if hit is None or hit[0] != key:
                        self._read(name)
            except Exception as ex:
                log("card: warm-up failed:", repr(ex), level=logs.WARN)
            finally:
                with self.lock:
                    self.warming = False

        threading.Thread(target=run, name="card-warm", daemon=True).start()

    def clear(self) -> None:
        with self.lock:
            self.last.clear()
            self.epoch += 1


def _card_key() -> tuple[Any, ...]:
    """What a cached card must still match to be answered from (_CardReads)."""
    return (
        _people_fresh(speech_log.shared().mark()[1]),
        photo_files._shot_revision(),
        mtime(context.STATE / card_prefs.PREFS),
    )


_CARD_READS = _CardReads()


def _video_id(cam: str, start: float) -> str:
    """A card or picker video's id: its camera and start, "cam0@1793212345.50"."""
    return f"{cam}@{start:.2f}"


def _voice_letters(i: int) -> str:
    """0 -> "A", 25 -> "Z", 26 -> "AA": the letter an unnamed voice is called by in the pickers."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


def why_name_view(q: Mapping[str, str]) -> tuple[dict[str, Any], int]:
    """(reply, status) for GET /identity/why: ?line=<key> (or ?ts_start=) for a transcript line, ?shot=<sighting id>
    for a recorded face (dashboard/why_name.py)."""
    from liberational.dashboard import why_name

    if (shot := q.get("shot")) is not None:
        if not photo_files.SIGHTING_ID.fullmatch(shot):
            return {"error": "shot is a sighting id"}, 400
        row = why_name.shot_row(context.STATE, shot)
        if row is None:
            return {"error": "no such shot"}, 404
        reg = None
        if people_list._has_log():
            with registry.safely("why this name"):
                reg = registry.at(context.STATE)
        return why_name.for_shot(row, reg), 200
    if q.get("line") is None and q.get("ts_start") is None:
        return {"error": "send line (a line key) or shot (a sighting id)"}, 400
    try:
        e = _stored_line(q.get("ts_start"), q.get("line"))
    except AmbiguousLine as ex:
        return {"error": str(ex)}, 409
    except TypeError, ValueError:
        return {"error": "ts_start is a time"}, 400
    if e is None:
        return {"error": "no such line"}, 404
    return why_name.for_line(e, transcript_lines._view(), lines_near(e)), 200


def _newest_face(f: Mapping[str, Any]) -> tuple[float, int]:
    """Sort key for the unnamed faces: newest seen first, then most seen; never seen (no shot left) last."""
    seen = f.get("last_seen")
    return -(float(seen) if isinstance(seen, int | float) else 0.0), -int(f.get("count") or 0)


def _newest_lines() -> dict[int, dict[str, Any]]:
    """{voice id: its newest line with audio, as read through the event log}, from one backward read of the transcript
    (what speakers_with_audio and latest_for_speaker would take one read per voice for)."""
    out: dict[int, dict[str, Any]] = {}
    # Share the retained speech cache and resolve voices in one batch, rather than one registry call per line.
    for e in reversed(
        transcript_lines._view().individual_lines(e for e in speech_log.retained() if transcript_lines.has_audio(e))
    ):
        if transcript_lines.has_audio(e) and (sid := e.get("speaker_id")) is not None and sid not in out:
            out[sid] = e
    return out


def people_unnamed() -> dict[str, Any]:
    """GET /inbox/unnamed: what nobody is named for yet, for People's "pick what's theirs" (contract at the top).

    Failed face history leaves face groups and voice samples usable, with an `incomplete` message for Retry.
    """
    from liberational.people import face_photos
    from liberational.vision import sightings

    skips = inbox_state.inbox_skips()
    try:
        groups = [
            g for g in people_list._faces_mod().unknown_groups() if g["count"] >= 2 and g["id"] not in skips["face"]
        ]
    except Exception:
        groups = []
    # Both readers need the complete month, including named faces that exclude mixed camera runs.
    # Current names and corrections are projected afresh for each request.
    now = time.time()
    incomplete = None
    try:
        rows = sightings.faces(now - sightings.SHOT_DAYS * 86400, now + 1, state=context.STATE)
    except Exception as ex:
        log("people: unnamed face history failed:", repr(ex), level=logs.WARN)
        rows = []
        incomplete = "Could not load face photos and video clips."
    try:
        recent = face_photos.recent_by_group(state=context.STATE, rows=rows)
    except Exception as ex:
        log("people: unnamed face shots failed:", repr(ex), level=logs.WARN)
        recent = {}
    faces, named, shots_pending, shots_unavailable, saved_shots = (
        [],
        views.names(),
        0,
        0,
        face_shots.saved(context.STATE, revision=photo_files._shot_revision()),
    )
    for g in groups:
        seen = recent.get(g["id"], [])
        ready, count, gone = people_list._ready_faces(seen, saved_shots)
        shots_pending += count
        shots_unavailable += gone
        photos = [f"/faces/unknown/{g['id']}.jpg"] if photo_files._unknown_photo(g["id"]).exists() else []
        photos += [f"/faces/shot/{s['id']}.jpg" for s in ready]
        faces.append(
            {
                "id": g["id"],
                "photos": photos[:4],
                "count": g["count"],
                "last_seen": seen[0]["ts"] if seen else None,
                "cam": seen[0]["cam"] if seen else None,
                "camera_name": named.get(seen[0]["cam"], seen[0]["cam"]) if seen else None,
                **({"shots_pending": count} if count else {}),
                **({"shots_unavailable": gone} if gone else {}),
            }
        )
    faces.sort(key=_newest_face)
    newest = _newest_lines()
    unnamed = [
        s
        for s in known_voices._speakers()["speakers"]
        if people_list.VOICE_GROUP.fullmatch((s.get("label") or "").strip())
        and s["id"] in newest
        and s["id"] not in skips["voice"]
    ]
    unnamed.sort(key=lambda s: -(s.get("last_seen") or 0))
    voices = []
    for i, s in enumerate(unnamed[:UNNAMED_VOICES]):
        e = newest[s["id"]]
        sample = None
        if e and isinstance(e.get("ts_start"), int | float):
            sample = {"line": registry.key_of(e), "ts_start": e["ts_start"], "text": (e.get("text") or "").strip()}
        voices.append(
            {
                "id": s["id"],
                "label": f"Unnamed voice {_voice_letters(i)}",
                "lines": s.get("n", 0),
                "last_heard": s.get("last_seen"),
                "sample": sample,
            }
        )
    try:
        clips = sightings.unnamed_clips(VISIT_GAP_S, UNNAMED_VIDEOS, context.STATE, rows=rows)
    except Exception as ex:
        log("people: unnamed clips failed:", repr(ex), level=logs.WARN)
        clips = []
    videos = [
        {
            "id": k["id"],
            "cam": k["cam"],
            "camera_name": named.get(k["cam"], k["cam"]),
            "start": k["start"],
            "end": k["end"],
            "duration_s": round(k["end"] - k["start"], 1),
            "thumb": f"/faces/shot/{k['thumb']}.jpg" if k["thumb"] is not None and k["thumb"] in saved_shots else None,
            "what": _what(k["cam"], k["start"], k["end"]),
        }
        for k in clips
    ]
    videos, videos_pending, videos_unavailable = _ready_videos(videos)
    result = {"faces": faces[:UNNAMED_FACES], "voices": voices, "videos": videos}
    if videos_pending:
        result["videos_pending"] = videos_pending
    if videos_unavailable:
        result["videos_unavailable"] = videos_unavailable
    if shots_pending:
        result["shots_pending"] = shots_pending
    if shots_unavailable:
        result["shots_unavailable"] = shots_unavailable
    return {**result, "incomplete": incomplete} if incomplete else result


class AmbiguousLine(LookupError):
    """A time alone that names lines from more than one microphone: the caller has to send the line's key."""


AMBIGUOUS = "more than one line starts then; send its line"


def _stored_line(ts: object = None, line: object = None) -> dict[str, Any] | None:
    """A transcript line as stored: by its key (`line`, registry.key_of), else by its ts_start to the millisecond.
    Raises AmbiguousLine when the time alone matches lines from two microphones, and ValueError for a bad time."""
    entries = transcript_lines.heard_lines()
    entries = [e for e in entries if isinstance(e.get("ts_start"), int | float) and not isinstance(e["ts_start"], bool)]
    if isinstance(line, str) and line:
        return next((e for e in entries if registry.key_of(e) == line), None)
    t = float(ts)  # ty: ignore[invalid-argument-type]  # a bad value is the caller's ValueError or TypeError
    hits = [e for e in entries if abs(float(e["ts_start"]) - t) < 0.0015]
    if len({registry.key_of(e) for e in hits}) > 1:
        raise AmbiguousLine(AMBIGUOUS)
    return hits[0] if hits else None


@routes.get_when(
    "/people/references, /people/reference-media/<id>",
    lambda p: p == "/people/references" or p.startswith("/people/reference-media/"),
)
def _get_people_references(h: H) -> None:
    p = h.path.split("?")[0]
    from liberational.people import references

    try:
        if p == "/people/references":
            return h._json(
                {
                    "references": references.listing(
                        context.STATE,
                        urllib.parse.parse_qs(urllib.parse.urlsplit(h.path).query).get("person", [""])[0],
                    )
                }
            )
        token = p.removeprefix("/people/reference-media/")
        meta = references.metadata(context.STATE, token)
        return h._media(
            references.entry(context.STATE, token) / meta["file"], meta["mime"], {"X-Content-Type-Options": "nosniff"}
        )
    except (ValueError, OSError) as exc:
        return h._json({"error": str(exc)}, 400)


# not under /people/, where it would be the page of someone named "unnamed"
@routes.get("/inbox/unnamed")
def _get_inbox_unnamed(h: H) -> None:
    return h._json(people_unnamed())


@routes.get("/people")
def _get_people(h: H) -> None:
    view = people_view(cold=True)
    _CARD_READS.warm([str(person["name"]) for person in view.get("people", [])[:CARD_WARM]])
    return h._json(view)


@routes.get_re(r"/people/([^/]+)/card")
def _get_people_x_card(h: H, m: re.Match[str]) -> None:
    card = _CARD_READS.get(urllib.parse.unquote(m.group(1)))
    return h._json(card or {"error": "bad name"}, 200 if card else 400)


# why a line or a recorded face has its name; read only
@routes.get("/identity/why")
def _get_identity_why(h: H) -> None:
    return h._json(*why_name_view(h._query()))


@routes.get_when("/people/<name>", lambda p: p.startswith("/people/"))
def _get_people_summary(h: H) -> None:
    p = h.path.split("?")[0]
    from urllib.parse import unquote

    s2 = person_summary(unquote(p[len("/people/") :]))
    return h._json(s2 or {"error": "bad name"}, 200 if s2 else 400)


@routes.get("/health")
def _get_health(h: H) -> None:
    return h._json(
        {
            "ok": True,
            "ts": time.time(),
            "clients": SSE_CLIENTS[0],
            "boot": devreload.BOOT,
            "people_warmup": _people_warm_status(),
        }
    )


_PEOPLE_WARM_LOCK = threading.Lock()


_PEOPLE_WARMED = threading.Event()  # the people warm-up finished (done or incomplete)


_PEOPLE_WARM_STATUS: dict[str, Any] = {"state": "pending", "stage": None, "elapsed_s": None, "errors": []}


def _people_warm_status() -> dict[str, Any]:
    """Cheap readiness metadata for /health, without triggering any data or model read."""
    with _PEOPLE_WARM_LOCK:
        return copy.deepcopy(_PEOPLE_WARM_STATUS)


def _set_people_warm(
    state: str, stage: str | None, *, elapsed_s: float | None = None, errors: Sequence[str] = ()
) -> None:
    with _PEOPLE_WARM_LOCK:
        _PEOPLE_WARM_STATUS.update(state=state, stage=stage, elapsed_s=elapsed_s, errors=list(errors))


def _warm_people() -> None:
    """Warm first-page dependencies after serving starts, at low priority and through the bounded voice gate.

    Header inputs go first, then roster/clone/voice dependencies and raw inbox history. Face groups go last so their
    short refresh interval starts near completion. A successful read alone is not ready: raw windows must have
    published, and groups must have published for their current owner. Retry an unpublished snapshot once, then
    report incomplete rather than claim the first page is warm.
    This only warms readers/imports; it never builds a model or keeps a permanent roster response.
    """
    photo_files._low_priority(19)
    started = time.monotonic()
    errors: list[str] = []
    stages: tuple[tuple[str, Callable[[], Any], Callable[[], bool] | None], ...] = (
        ("microphones", live_status.mic_status, None),
        ("scene", live_status.scene_entries, None),
        ("cameras", live_status.camera_status, None),
        ("roster", lambda: people_view(wait_s=None), None),
        ("clones", voice_clones.clones_list, None),
        ("voice groups", unknown_voices_view, unnamed_groups._VOICE_GROUPS.ready),
        ("face groups", unknown_groups_view, unnamed_groups._FACE_GROUPS.ready),
    )
    for name, read, ready in stages:
        try:
            for attempt in range(2):
                _set_people_warm("waiting", name)
                voice_turn.wait_quiet()
                _set_people_warm("running", name)
                result = read()
                if isinstance(result, tuple) and result[0] != 200:
                    errors.append(f"{name}: HTTP {result[0]}")
                    break
                if isinstance(result, dict) and result.get("error"):
                    errors.append(f"{name}: unavailable")
                    break
                if result is not False and (ready is None or ready()):
                    break
                if attempt == 1:
                    errors.append(f"{name}: snapshot not published")
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}")
    elapsed = time.monotonic() - started
    _set_people_warm("incomplete" if errors else "done", None, elapsed_s=round(elapsed, 3), errors=errors)
    _PEOPLE_WARMED.set()
    if errors:
        log("people: warm-up incomplete:", "; ".join(errors), level=logs.WARN)
    else:
        log(f"people: warmed in {elapsed:.1f} s", level=logs.INFO)
