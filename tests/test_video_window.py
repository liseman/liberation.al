"""The nightly video window (src/liberational/video_window.py): which video goes, when, and what stays."""

import asyncio
from pathlib import Path
from typing import Any, cast

import continuum_py
from continuum_py import Client

from liberational import video_window as V
from liberational.continuum import to_ns

NOW = 1_800_000_000.0
DAY = 86400.0
SETTINGS = V.Settings(days=30, free_floor_percent=15, batch=2)


def uid(n: int) -> str:
    return f"00000000-0000-7000-8000-{n:012d}"


class FakeBroker:
    """Payloads per stream, each with a commit time and optionally its own capture time (else the same); global
    commit sequences in commit order, as the broker's."""

    def __init__(self, payloads: dict[str, list[tuple[str, float]]], captured: dict[str, float] | None = None):
        self.payloads = payloads
        ordered = sorted((at, i) for rows in payloads.values() for i, at in rows)
        self.seq = {i: n for n, (_, i) in enumerate(ordered, 1)}
        self.at = {i: at for at, i in ordered}
        self.captured = captured or {}
        self.aged: list[tuple[str, int, list[str]]] = []

    async def list_streams(self) -> list[dict[str, Any]]:
        return [{"name": name} for name in self.payloads]

    async def commit_seq_at(self, at_ns: int) -> int:
        return max((n for i, n in self.seq.items() if to_ns(self.at[i]) <= at_ns), default=0)

    async def query_by_seq(self, stream: str, *, after_seq: int = 0, max_items: int = 0) -> dict[str, Any]:
        rows = sorted((self.seq[i], i) for i, _ in self.payloads[stream] if self.seq[i] > after_seq)
        page = rows[:max_items] if max_items else rows
        items = [{"commit_seq": n, "header": header(i, self.captured.get(i, self.at[i]))} for n, i in page]
        return {"items": items, "truncated": len(rows) > len(page), "next_after": page[-1][0] if page else 0}

    async def age_out(self, ids: list[str], before_ns: int, reason: str) -> int:
        self.aged.append((reason, before_ns, list(ids)))
        for rows in self.payloads.values():
            rows[:] = [r for r in rows if r[0] not in ids]
        return len(ids)

    def ids(self) -> list[str]:
        return [i for _, _, batch in self.aged for i in batch]


def header(i: str, at: float) -> bytes:
    return continuum_py.encode_header(
        {
            "id": i,
            "stream_id": 1,
            "start_ns": to_ns(at),
            "end_ns": to_ns(at),
            "layer": "Layer1Raw",
            "format": "EncodedVideo",
            "shape_name": "video",
            "parents": [],
            "subjects": [],
            "seq": 0,
        }
    )


async def nothing() -> None:
    pass


def run(broker: FakeBroker, state: Path, *, now: float = NOW, free=lambda: 0.5, dry: bool = False) -> dict[str, int]:
    r = V.Run(cast(Client, broker), state, SETTINGS, now, broker.age_out, free, dry=dry, settle=nothing)
    return asyncio.run(r.run())


def test_only_video_older_than_the_window_goes(tmp_path):
    broker = FakeBroker(
        {
            "patio.video": [(uid(1), NOW - 40 * DAY), (uid(2), NOW - 31 * DAY), (uid(3), NOW - 29 * DAY)],
            "kitchen.video": [(uid(4), NOW - 35 * DAY)],
            "wave3.audio": [(uid(5), NOW - 40 * DAY)],
            "pose.body": [(uid(6), NOW - 40 * DAY)],
        }
    )
    counts = run(broker, tmp_path)
    assert sorted(broker.ids()) == [uid(1), uid(2), uid(4)]
    assert counts == {"patio.video": 2, "kitchen.video": 1}
    assert {before for _, before, _ in broker.aged} == {to_ns(NOW - 30 * DAY)}


def test_a_later_run_picks_up_where_the_last_stopped(tmp_path):
    broker = FakeBroker({"patio.video": [(uid(1), NOW - 31 * DAY), (uid(2), NOW - 29.5 * DAY), (uid(3), NOW - DAY)]})
    run(broker, tmp_path)
    assert broker.ids() == [uid(1)]
    run(broker, tmp_path, now=NOW + DAY)
    assert broker.ids() == [uid(1), uid(2)]
    # The marks were saved, so a run with nothing newly old asks for nothing.
    run(broker, tmp_path, now=NOW + DAY)
    assert broker.ids() == [uid(1), uid(2)]


def test_a_stream_longer_than_a_page_is_read_to_the_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "PAGE", 2)
    old = [(uid(n), NOW - 40 * DAY + n) for n in range(1, 6)]
    broker = FakeBroker({"patio.video": [*old, (uid(9), NOW - DAY)]})
    run(broker, tmp_path)
    assert broker.ids() == [i for i, _ in old]


def test_below_the_floor_the_oldest_hours_go_early_but_never_the_newest_day(tmp_path):
    hours = [(uid(n), NOW - 30 * DAY + n * 7200) for n in range(1, 4)]  # inside the window, two hours apart
    broker = FakeBroker({"patio.video": [*hours, (uid(9), NOW - 0.5 * DAY)]})
    free = iter([0.10, 0.10, 0.20, 0.20])  # two early hours bring free space back over 15%
    counts = run(broker, tmp_path, free=lambda: next(free))
    assert broker.ids() == [uid(1), uid(2)]
    assert counts["early_hours"] == 2

    broker = FakeBroker({"patio.video": [(uid(9), NOW - 0.5 * DAY)]})
    run(broker, tmp_path / "full", free=lambda: 0.01)
    assert broker.ids() == []


def test_a_disk_that_fills_before_the_window_does_frees_its_oldest_video(tmp_path):
    young = [(uid(n), NOW - 10 * DAY + n * 600) for n in range(1, 4)]  # ten days old: nothing past the window
    broker = FakeBroker({"patio.video": [*young, (uid(9), NOW - 0.5 * DAY)]})
    free = iter([0.01, 0.20])
    counts = run(broker, tmp_path, free=lambda: next(free))
    assert broker.ids() == [i for i, _ in young]
    assert counts["early_hours"] == 1


def test_the_floor_goes_by_when_video_was_committed_not_its_own_clock(tmp_path):
    # A camera with its clock a day ahead, and an import of footage from years ago: both committed ten days ago.
    future, backdated = uid(1), uid(2)
    broker = FakeBroker(
        {"patio.video": [(future, NOW - 10 * DAY)], "ingest.video": [(backdated, NOW - 10 * DAY + 60)]},
        captured={future: NOW + DAY, backdated: NOW - 5 * 365 * DAY},
    )
    free = iter([0.01, 0.20])
    counts = run(broker, tmp_path, free=lambda: next(free))
    assert sorted(broker.ids()) == [future, backdated]
    assert counts["early_hours"] == 1


def test_a_dry_run_counts_and_moves_no_mark(tmp_path):
    broker = FakeBroker({"patio.video": [(uid(1), NOW - 31 * DAY)]})
    counted: list[list[str]] = []

    async def count(ids: list[str], before_ns: int, reason: str) -> int:
        counted.append(ids)
        return len(ids)

    r = V.Run(cast(Client, broker), tmp_path, SETTINGS, NOW, count, lambda: 0.5, dry=True, settle=nothing)
    assert asyncio.run(r.run()) == {"patio.video": 1}
    assert not (tmp_path / V.MARKS).exists()
    assert broker.aged == []


def test_journal_evidence_is_promoted_before_anything_is_aged_out(tmp_path):
    broker = FakeBroker({"patio.video": [(uid(1), NOW - 40 * DAY)]})
    order = []

    async def age_out(ids, before_ns, reason):
        order.append("age out")
        return await broker.age_out(ids, before_ns, reason)

    def promote():
        order.append("promote")
        assert broker.ids() == []  # nothing has gone yet
        return {"clip": 3, "still": 1, "gone": 0}

    r = V.Run(cast(Client, broker), tmp_path, SETTINGS, NOW, age_out, lambda: 0.5, settle=nothing, promote=promote)
    counts = asyncio.run(r.run())
    assert order == ["promote", "age out"]
    assert counts == {"promoted_clip": 3, "promoted_still": 1, "patio.video": 1}


def test_a_failed_promotion_still_lets_the_window_age_video_out(tmp_path, caplog):
    broker = FakeBroker({"patio.video": [(uid(1), NOW - 40 * DAY)]})

    def promote():
        raise OSError("no space for clips")

    r = V.Run(
        cast(Client, broker), tmp_path, SETTINGS, NOW, broker.age_out, lambda: 0.5, settle=nothing, promote=promote
    )
    assert asyncio.run(r.run()) == {"patio.video": 1}
    assert "promoting journal evidence failed" in caplog.text
