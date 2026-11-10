"""memory/reports.py: day aggregates for the Reports tab, over a fake Sources (made-up people, no broker)."""

import asyncio
import datetime as dt
import logging
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from typing import ClassVar

import pytest

from liberational import config, segment_log
from liberational.memory import reports
from liberational.memory.reports import Seen
from liberational.people import registry


@pytest.fixture(autouse=True)
def pacific(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def at(day, hh, mm=0):
    return time.mktime(time.strptime(f"{day} {hh:02d}:{mm:02d}", "%Y-%m-%d %H:%M"))


class FakeSources:
    def __init__(
        self,
        seen=(),
        people=None,
        video=(),
        audio=(),
        first=None,
        events=(),
        act_cov=(),
        visits=(),
        zone_cov=(),
        actions=(),
        segments=None,
    ):
        self.items = list(seen)
        self.spans, self.act_cov, self.visits, self.zone_cov = list(events), list(act_cov), list(visits), list(zone_cov)
        self.names = {"p-alice": "Alice", "p-dana": "Dana"} if people is None else people
        self.video, self.audio = dict(video), dict(audio)  # day: set of recorded hours
        self.first = first
        self.reads = []
        self.mark = "labels-1"
        self.hour_reads = 0
        self.during_seen = None  # called while a day is being read
        self.actions, self.segments = list(actions), segments or {}

    def people(self):
        return self.names

    def seen(self, start, end):
        self.reads.append((start, end))
        if self.during_seen:
            self.during_seen()
        return [s for s in self.items if s.end >= start and s.start < end]

    def identity_mark(self):
        return self.mark

    def recorded_hours(self, kind, day):
        self.hour_reads += 1
        hours = (self.video if kind == "video" else self.audio).get(day, set())
        return [h in hours for h in range(24)]

    def first_day(self):
        return self.first

    def events(self, start, end):
        return [e for e in self.spans if start <= e["start"] < end]

    def activity_coverage(self, start, end):
        return [(cam, max(a, start), min(b, end)) for cam, a, b in self.act_cov if b > start and a < end]

    def zone_visits(self, start, end):
        return [v for v in self.visits if v["ts_end"] >= start and v["ts_start"] < end]

    def zone_coverage(self, start, end):
        return [
            {**c, "ts_start": max(c["ts_start"], start), "ts_end": min(c["ts_end"], end)}
            for c in self.zone_cov
            if c["ts_end"] > start and c["ts_start"] < end
        ]

    def action_spans(self, start, end):
        return [s for s in self.actions if s["end"] > start and s["start"] < end]

    def segment_rows(self, stream, start, end):
        return [
            r
            for r in self.segments.get(stream, [])
            if float(r.get("ts_end", r.get("end", r.get("start", 0)))) >= start
            and float(r.get("ts_start", r.get("start", 0))) < end
        ]


DAY = "2026-10-20"


def test_hour_bounds_on_clock_change_days():
    spring = reports.hour_bounds("2026-03-08")
    assert len(spring) == 24
    assert spring[2][1] - spring[2][0] == 0  # 02:00 never happened
    assert spring[1][1] - spring[1][0] == 3600
    fall = reports.hour_bounds("2026-11-01")
    assert fall[1][1] - fall[1][0] == 7200  # 01:00 happened twice
    assert fall[23][1] == reports.day_bounds("2026-11-01")[1]


def test_whos_around_splits_a_visit_across_hours_and_marks_unrecorded_hours_as_gaps():
    src = FakeSources([Seen("p-alice", at(DAY, 10, 30), at(DAY, 11, 30), heard=False)], video={DAY: {9, 10, 11}})
    body = reports.WhosAround().day(src, DAY)
    assert body["people"]["p-alice"][10] == 1800
    assert body["people"]["p-alice"][11] == 1800
    out = reports.WhosAround().assemble({DAY: body}, src, None)
    hours = out["days"][0]["hours"]
    assert hours[8] is None  # nothing recorded: a gap, not a zero
    assert hours[9] == {"t": at(DAY, 9), "who": []}  # recorded, nobody in view: a real zero
    assert hours[10]["who"] == [["p-alice", 30.0]]
    assert out["people"] == [{"id": "p-alice", "name": "Alice", "hours": 1.0}]


def test_whos_around_counts_an_hour_with_someone_in_it_as_covered_even_without_a_recording():
    src = FakeSources([Seen("p-dana", at(DAY, 3), at(DAY, 3, 10), heard=False)])
    body = reports.WhosAround().day(src, DAY)
    assert body["cov"][3] and not body["cov"][4]


def test_whos_around_leaves_out_someone_the_timeline_no_longer_knows():
    src = FakeSources([Seen("p-gone", at(DAY, 10), at(DAY, 11), heard=False)], video={DAY: {10}})
    body = reports.WhosAround().day(src, DAY)
    src.names = {"p-alice": "Alice"}
    out = reports.WhosAround().assemble({DAY: body}, src, None)
    assert out["days"][0]["hours"][10]["who"] == []
    assert out["people"] == []


def test_whos_around_for_one_person_keeps_everyone_in_the_roster():
    src = FakeSources(
        [
            Seen("p-alice", at(DAY, 10), at(DAY, 10, 20), heard=False),
            Seen("p-dana", at(DAY, 10), at(DAY, 10, 40), heard=False),
        ],
        video={DAY: {10}},
    )
    body = reports.WhosAround().day(src, DAY)
    out = reports.WhosAround().assemble({DAY: body}, src, "p-dana")
    assert out["days"][0]["hours"][10]["who"] == [["p-dana", 40.0]]
    assert [p["name"] for p in out["people"]] == ["Dana", "Alice"]


def line(who, day, hh, mm, ss, text, length=3):
    start = at(day, hh, mm) + ss
    return Seen(who, start, start + length, heard=True, text=text)


def test_conversations_counts_exchanges_lines_and_words():
    src = FakeSources(
        [
            line("p-alice", DAY, 9, 0, 0, "Dana, the garden tomatoes are ripe"),
            line("p-dana", DAY, 9, 0, 20, "Great, tomatoes for dinner then"),
            line("p-alice", DAY, 9, 30, 0, "garden hose is leaking"),  # half an hour later: no exchange
            Seen("p-alice", at(DAY, 9), at(DAY, 9, 5), heard=False),  # a face visit is not a line
        ],
        audio={DAY: {9}},
    )
    body = reports.Conversations().day(src, DAY)
    assert body["lines"] == {"p-alice": [0] * 9 + [2] + [0] * 14, "p-dana": [0] * 9 + [1] + [0] * 14}
    assert body["pairs"] == {"p-alice|p-dana": [1, at(DAY, 9)]}
    assert body["words"]["tomatoes"] == 2 and body["words"]["garden"] == 2
    assert "dana" not in body["words"] and "the" not in body["words"]
    out = reports.Conversations().assemble({DAY: body}, src, None)
    assert [p["name"] for p in out["people"]] == ["Alice", "Dana"]
    assert out["people"][0]["by_hour"][9] == {"n": 2, "day": DAY, "t": at(DAY, 9)}
    assert out["pairs"][0]["n"] == 1
    assert out["daily"] == [{"day": DAY, "lines": 3, "t": at(DAY, 9)}]


def test_conversations_day_with_no_audio_is_a_gap_not_zero():
    src = FakeSources()
    body = reports.Conversations().day(src, DAY)
    out = reports.Conversations().assemble({DAY: body}, src, None)
    assert out["daily"] == [{"day": DAY, "lines": None, "t": None}]


def test_conversations_week_without_any_coverage_is_none_for_every_word():
    src = FakeSources([line("p-alice", DAY, 9, 0, 0, "tomatoes tomatoes")], audio={DAY: {9}})
    days = {DAY: reports.Conversations().day(src, DAY), "2026-10-28": reports.Conversations().day(src, "2026-10-28")}
    out = reports.Conversations().assemble(days, src, None)
    assert [w["n"] for w in out["words"][0]["weeks"]] == [2, None]


@pytest.mark.parametrize(
    ("purges", "dropped"),
    [
        pytest.param([{"range": [int(at(DAY, 12) * 1e9), int(at(DAY, 13) * 1e9)]}], {DAY}, id="range"),
        pytest.param([{"kind": "Entity", "range": None}], None, id="person"),
        pytest.param([{"kind": "Ids", "range": None, "id_count": 3}], None, id="ids"),
        pytest.param([{"range": [0, int(at(DAY, 13) * 1e9)]}], {"2026-10-18", "2026-10-19", DAY}, id="clipped"),
        pytest.param([{"range": [0, int(at("2026-10-17", 13) * 1e9)]}], set(), id="before-the-window"),
        pytest.param([{"range": [int(at(DAY, 1) * 1e9), int(at("2026-11-30", 1) * 1e9)]}], {DAY}, id="after-it"),
        pytest.param(
            [{"range": [0, int(at(DAY, 13) * 1e9)], "streams": ["worker.checkpoint.a.parts"]}], set(), id="worker"
        ),
        pytest.param(
            [
                {
                    "range": [int(at(DAY, 12) * 1e9), int(at(DAY, 13) * 1e9)],
                    "streams": ["worker.checkpoint.a.parts", "speech.utterance"],
                }
            ],
            {DAY},
            id="one-stream-read",
        ),
        pytest.param(
            [{"range": [int(at("2026-10-18", 12) * 1e9), int(at("2026-10-18", 13) * 1e9)]}]
            + [{"range": [0, int(at(DAY, 13) * 1e9)], "streams": ["worker.checkpoint.a.parts"]}] * 5000,
            {"2026-10-18"},
            id="many-unread",
        ),
    ],
)
def test_purged_days(purges, dropped):
    assert reports.purged_days(purges, ("2026-10-18", DAY)) == dropped


def make(tmp_path, src, now, purges=None):
    return reports.Reports(
        reports.Store(tmp_path / "reports.sqlite3"), src, now=lambda: now, purges=purges, background=False
    )


def test_view_lists_days_still_to_read_then_shows_them_once_built(tmp_path):
    src = FakeSources([Seen("p-alice", at(DAY, 10), at(DAY, 11), heard=False)], video={DAY: {10}}, first="2026-10-19")
    view = make(tmp_path, src, at(DAY, 12))
    first = view.view("whos_around")
    assert first["pending"] == [DAY, "2026-10-19"]
    assert first["days"] == []
    view._queue.clear()
    view.build("whos_around", DAY)
    view.build("whos_around", "2026-10-19")
    second = view.view("whos_around")
    assert second["pending"] == []
    assert [d["day"] for d in second["days"]] == ["2026-10-19", DAY]


def test_a_settled_day_is_not_read_again_but_today_is_once_its_copy_is_old(tmp_path):
    src = FakeSources(first="2026-10-19")
    now = at(DAY, 12)
    view = make(tmp_path, src, now)
    view.build("whos_around", "2026-10-19")
    view.build("whos_around", DAY)
    view.now = lambda: now + reports.FRESH_S + 1
    out = view.view("whos_around")
    assert out["pending"] == [] and out["refreshing"] == [DAY]


class DropLog(reports.Store):
    def __init__(self, path):
        super().__init__(path)
        self.dropped = []

    def drop(self, days=None):
        days = None if days is None else list(days)
        self.dropped.append(days)
        super().drop(days)


def built(tmp_path, purge):
    """Two days built, then one new purge: (the view after it, the days each drop was given)."""
    store = DropLog(tmp_path / "reports.sqlite3")
    pending = [([], "purge-1")]
    view = reports.Reports(
        store, FakeSources(first="2026-10-19"), now=lambda: at(DAY, 12) + 2 * 86400, purges=lambda after: pending[0]
    )
    view.background = False
    view.view("whos_around")
    view._queue.clear()
    for d in ("2026-10-19", DAY):
        view.build("whos_around", d)
    pending[0] = ([purge], "purge-2")
    return view.view("whos_around"), store.dropped


@pytest.mark.parametrize(
    "streams",
    [
        pytest.param(["worker.checkpoint.d88d.parts", "worker.checkpoint.d88d.manifest"], id="worker-checkpoint"),
    ],
)
def test_a_purge_of_streams_no_report_reads_keeps_every_stored_day(tmp_path, streams):
    out, dropped = built(tmp_path, {"kind": "Range", "range": (0, int(at(DAY, 12) * 1e9)), "streams": streams})
    assert out["pending"] == ["2026-10-22", "2026-10-21"]  # never built; the two built days stay
    assert dropped == []


def test_a_purge_since_1970_drops_only_the_days_reports_cover(tmp_path):
    purge = {"kind": "Range", "range": (0, int(at(DAY, 12) * 1e9)), "streams": ["face.visit", "worker.x"]}
    out, dropped = built(tmp_path, purge)
    assert out["pending"] == ["2026-10-22", "2026-10-21", DAY, "2026-10-19"]
    assert [sorted(d) for d in dropped] == [["2026-10-19", DAY]]  # from the first recorded day, not 1970


def test_a_purge_of_a_range_rereads_only_its_days(tmp_path):
    src = FakeSources(first="2026-10-19")
    pending = [([], "purge-1")]
    view = make(tmp_path, src, at(DAY, 12) + 2 * 86400, purges=lambda after: pending[0])
    view.view("whos_around")  # the first purge id is only remembered
    view._queue.clear()
    for d in ("2026-10-19", DAY):
        view.build("whos_around", d)
    pending[0] = ([{"range": [int(at(DAY, 1) * 1e9), int(at(DAY, 2) * 1e9)]}], "purge-2")
    out = view.view("whos_around")
    assert DAY in out["pending"] and "2026-10-19" not in out["pending"]


class FakeBroker:
    def __init__(self, streams, purges):
        self.streams, self.purges = streams, purges
        self.stream_lists = 0
        self.purge_reads = []

    async def list_streams(self):
        self.stream_lists += 1
        return [{"stream_id": i, "name": n, "kind": "Derived"} for i, n in self.streams.items()]

    async def list_purges(self, *, after=None, limit=0):
        from continuum_py import BrokerError

        self.purge_reads.append(after)
        ids = [p["purge_id"] for p in self.purges]
        if after is not None and after not in ids:
            raise BrokerError("NotFound", "purge cursor is no longer present")
        start = ids.index(after) + 1 if after in ids else 0
        return self.purges[start : start + (limit or 256)]


def purge(n, stream_ids):
    return {"purge_id": f"p{n}", "kind": "Range", "range": (0, 10**18), "stream_ids": stream_ids, "id_count": 0}


@pytest.fixture
def live_broker(monkeypatch):
    from liberational import vision_log

    broker = FakeBroker({3: "face.visit", 68: "worker.checkpoint.a.parts"}, [purge(1, [68]), purge(2, [3, 68])])

    class Link:
        def call(self, go):
            return asyncio.run(go(broker))

    monkeypatch.setattr(vision_log, "at", lambda state: type("Log", (), {"link": Link()})())
    return broker


def test_live_purges_name_their_streams(live_broker, tmp_path):
    broker = live_broker
    read = reports.live_purges(tmp_path)
    found, newest = read(None)
    assert [p["streams"] for p in found] == [["worker.checkpoint.a.parts"], ["face.visit", "worker.checkpoint.a.parts"]]
    assert newest == "p2"
    broker.purges.append(purge(3, [3, 99]))  # a stream registered since the names were read
    broker.streams[99] = "zone.visit"
    found, newest = read("p2")
    assert [p["streams"] for p in found] == [["face.visit", "zone.visit"]]
    assert broker.stream_lists == 2  # read again only for the name it didn't know


def test_missing_purge_cursor_retires_persistently_and_invalidates_cached_reports_once(live_broker, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    src = FakeSources(first=DAY)
    read = reports.live_purges(tmp_path)
    view = make(tmp_path, src, at(DAY, 12) + 2 * 86400, purges=read)
    view.build("whos_around", DAY)
    view.store.set_meta("purge", "retired")

    view.view("whos_around")
    assert view.store.meta("purge") is None
    assert view.store.days("whos_around", reports.WhosAround.version) == {}

    # A new reader/process must not need an in-memory list of already warned-about cursors.
    resumed = make(tmp_path, src, at(DAY, 12) + 2 * 86400, purges=reports.live_purges(tmp_path))
    resumed.view("whos_around")
    resumed.view("whos_around")
    assert live_broker.purge_reads == ["retired", None, "p2", "p2"]
    assert resumed.store.meta("purge") == "p2"
    assert len(caplog.records) == 1

    resumed.build("whos_around", DAY)
    live_broker.purges.append({**purge(3, [3]), "range": (int(at(DAY, 0) * 1e9), int(at(DAY, 23) * 1e9))})
    resumed.view("whos_around")
    assert resumed.store.days("whos_around", reports.WhosAround.version) == {}
    assert resumed.store.meta("purge") == "p3"
    assert len(caplog.records) == 1


def test_day_being_built_when_purge_cursor_disappears_is_not_published(live_broker, tmp_path):
    src = FakeSources([line("p-alice", DAY, 9, 0, 0, "secret garden")], audio={DAY: {9}}, first=DAY)
    view = make(tmp_path, src, at(DAY, 12), purges=reports.live_purges(tmp_path))
    view.store.set_meta("purge", "retired")

    def cursor_disappeared():
        src.during_seen = None
        view._check_purges()

    src.during_seen = cursor_disappeared
    assert view.build("conversations", DAY) is False
    assert view.store.days("conversations", reports.Conversations.version) == {}
    assert view.store.meta("purge") is None
    assert view.build("conversations", DAY) is True


@pytest.mark.parametrize(
    ("cursor", "error_args"),
    [
        ("p1", ("Unavailable", "broker starting")),
        ("p1", ("Conflict", "NotFound mentioned in unrelated detail")),
        (None, ("NotFound", "no cursor supplied")),
    ],
)
def test_other_purge_errors_keep_the_cursor_and_retry(live_broker, tmp_path, monkeypatch, caplog, cursor, error_args):
    from continuum_py import BrokerError

    caplog.set_level(logging.WARNING)
    error = BrokerError(*error_args)
    calls = []

    async def refused(*, after=None, limit=0):
        calls.append(after)
        raise error

    monkeypatch.setattr(live_broker, "list_purges", refused)
    view = make(tmp_path, FakeSources(first=DAY), at(DAY, 12), purges=reports.live_purges(tmp_path))
    view.build("whos_around", DAY)
    view.store.set_meta("purge", cursor)
    stored = view.store.days("whos_around", reports.WhosAround.version)
    view.view("whos_around")
    view.view("whos_around")
    assert calls == [cursor, cursor]
    assert view.store.meta("purge") == cursor
    assert view.store.days("whos_around", reports.WhosAround.version) == stored
    assert len(caplog.records) == 2


def test_stream_lookup_not_found_does_not_retire_a_valid_purge_cursor(live_broker, tmp_path, monkeypatch, caplog):
    from continuum_py import BrokerError

    caplog.set_level(logging.WARNING)

    async def missing_streams():
        raise BrokerError("NotFound", "stream lookup failed")

    monkeypatch.setattr(live_broker, "list_streams", missing_streams)
    view = make(tmp_path, FakeSources(first=DAY), at(DAY, 12), purges=reports.live_purges(tmp_path))
    view.store.set_meta("purge", "p1")
    view.view("whos_around")
    view.view("whos_around")
    assert live_broker.purge_reads == ["p1", "p2", "p1", "p2"]
    assert view.store.meta("purge") == "p1"
    assert len(caplog.records) == 2


def test_layout_survives_and_gains_new_built_ins_once(tmp_path):
    from liberational.dashboard import reports as page

    store = reports.Store(tmp_path / "reports.sqlite3")
    assert [r["kind"] for r in page.layout(store)] == list(page.BUILT_IN)
    page.save_layout(store, [{"id": "conversations", "kind": "conversations", "arg": None, "title": "Talk"}])
    assert [r["title"] for r in page.layout(store)] == ["Talk"]  # a removed built-in stays removed
    older = [k for k in page.BUILT_IN if k != "whos_around"]  # saved before Who's around existed
    store.set_meta("layout", {"reports": store.meta("layout")["reports"], "known": older})
    assert [r["kind"] for r in page.layout(store)] == ["conversations", "whos_around"]  # newer than the save


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param([{"id": "a", "kind": "nope", "arg": None, "title": "x"}], id="unknown-kind"),
        pytest.param([{"id": "a", "kind": "conversations", "arg": None, "title": "x"}] * 2, id="duplicate-id"),
        pytest.param([{"id": "a", "kind": "conversations", "arg": None, "title": " "}], id="blank-title"),
        pytest.param({"id": "a"}, id="not-a-list"),
    ],
)
def test_clean_layout_refuses(raw):
    from liberational.dashboard import reports as page

    with pytest.raises(ValueError):
        page.clean_layout(raw)


def visit(n, start, end, label="cat", kind="animal", zone="z1", name="litter box"):
    return {
        "id": f"v{n}",
        "ts_start": start,
        "ts_end": end,
        "dwell_s": end - start,
        "zone": zone,
        "zone_name": name,
        "label": label,
        "subject_kind": kind,
        "ended": "exited",
    }


def test_activity_counts_events_and_joined_zone_visits_over_their_coverage():
    t = at(DAY, 14)
    src = FakeSources(
        events=[{"name": "pushups", "start": t + 60}, {"name": "pushups", "start": t + 600}],
        act_cov=[("cam0", t, t + 3600)],
        # One cat visit split by its track: 8 s, a 5 s gap, 8 s. Joined it is 21 s, so it counts once.
        visits=[
            visit(1, t + 100, t + 108),
            visit(2, t + 113, t + 121),
            visit(3, t + 900, t + 905, label="person", kind="person"),
        ],
        zone_cov=[{"subject_kind": "animal", "ts_start": t, "ts_end": t + 1800}],
    )
    body = reports.Activity().day(src, DAY)
    assert body["events"] == {"pushups": [0] * 14 + [2] + [0] * 9}
    assert body["zones"] == {"litter box": [0] * 14 + [1] + [0] * 9}  # the 5 s person visit is passing by
    assert body["pets"] == {"cat": [0] * 14 + [1] + [0] * 9}
    assert body["event_cov"][14] and not body["event_cov"][15]


def test_activity_typical_day_is_the_median_of_covered_days_and_skips_gaps():
    days = {}
    for d, n in (("2026-10-20", 2), ("2026-10-21", 4), ("2026-10-22", 9)):
        t = at(d, 9)
        src = FakeSources(events=[{"name": "fridge", "start": t + i} for i in range(n)], act_cov=[("cam0", t, t + 60)])
        days[d] = reports.Activity().day(src, d)
    days["2026-10-23"] = reports.Activity().day(FakeSources(), "2026-10-23")  # nothing looked at: a gap
    out = reports.Activity().assemble(days, FakeSources(), None)
    fridge = out["events"][0]
    assert [d["n"] for d in fridge["days"]] == [2, 4, 9, None]
    assert fridge["typical"] == 4
    assert out["covered_days"]["events"] == 3


def test_activity_focused_on_one_zone_shows_only_that_zone():
    t = at(DAY, 9)
    src = FakeSources(
        visits=[visit(1, t, t + 30), visit(2, t + 100, t + 140, zone="z2", name="food bowl")],
        zone_cov=[{"subject_kind": "animal", "ts_start": t, "ts_end": t + 600}],
    )
    out = reports.Activity().assemble({DAY: reports.Activity().day(src, DAY)}, src, "zone:food bowl")
    assert [z["name"] for z in out["zones"]] == ["food bowl"]
    assert out["events"] == [] and out["pets"] == []


def test_suggestions_come_only_from_data_and_skip_what_is_on_the_page(tmp_path):
    from liberational.dashboard import reports as page

    t = at(DAY, 9)
    src = FakeSources(
        [Seen("p-alice", t, t + 7200, heard=False), line("p-dana", DAY, 9, 0, 0, "tomatoes tomatoes")],
        video={DAY: {9, 10}},
        audio={DAY: {9}},
        events=[{"name": "pushups", "start": t}],
        act_cov=[("cam0", t, t + 60)],
        first=DAY,
    )
    view = make(tmp_path, src, at(DAY, 23))
    for kind in ("whos_around", "conversations", "activity"):
        view.build(kind, DAY)
    page.save_layout(view.store, [{"id": "mine", "kind": "activity", "arg": "event:pushups", "title": "Pushups"}])
    # Pushups are on the page, and a person's presence is Who's around's filter, not a card of its own.
    assert page.suggestions(view) == []


def test_suggestions_rank_by_days_of_data_and_never_suggest_words(tmp_path):
    from liberational.dashboard import reports as page

    day2 = "2026-10-21"
    junk = "fucking alright long new three second count five again anyway nine"
    t1, t2 = at(DAY, 9), at(day2, 9)
    src = FakeSources(
        [
            Seen("p-alice", t1, t1 + 7200, heard=False),
            Seen("p-alice", t2, t2 + 3600, heard=False),
            # Dana swears a lot but is only heard for 2.5 minutes: the old word-count suggestions led with this.
            *[line("p-dana", DAY, 10, i // 20, (i % 20) * 3, junk) for i in range(50)],
        ],
        video={DAY: {9, 10}, day2: {9}},
        audio={DAY: {10}},
        events=[{"name": "fridge", "start": t1 + i} for i in range(4)],
        act_cov=[("cam0", t1, t1 + 60)],
        visits=[visit(1, t1, t1 + 120), visit(2, t2, t2 + 120)],
        zone_cov=[
            {"subject_kind": "animal", "ts_start": t1, "ts_end": t1 + 600},
            {"subject_kind": "animal", "ts_start": t2, "ts_end": t2 + 600},
        ],
        first=DAY,
    )
    view = make(tmp_path, src, at(day2, 23))
    for kind in ("whos_around", "conversations", "activity"):
        for d in (DAY, day2):
            view.build(kind, d)
    # Without the Activity card on the page, so its lead event (fridge, the only one) is suggested too.
    page.save_layout(view.store, [{"id": "conversations", "kind": "conversations", "arg": None, "title": "Talk"}])
    found = page.suggestions(view)
    # Two days of data before one; at equal days, people, then pets, events, and zones.
    assert [s["title"] for s in found] == [
        "A cat's day",
        "Visits to litter box",
        "fridge, day by day",
    ]
    assert all(s["kind"] != "conversations" for s in found)
    assert found[2]["why"] == "fridge: seen 4 times on 1 day, with a typical day"


def test_a_day_read_across_a_purge_is_not_stored_but_read_again(tmp_path):
    src = FakeSources([line("p-alice", DAY, 9, 0, 0, "secret garden")], audio={DAY: {9}}, first=DAY)
    purges = [([], "purge-1")]
    view = make(tmp_path, src, at(DAY, 12) + 2 * 86400, purges=lambda after: purges[0])
    view.view("conversations")
    view._queue.clear()

    def purge_now():
        purges[0] = ([{"range": [int(at(DAY, 9) * 1e9), int(at(DAY, 10) * 1e9)]}], "purge-2")
        src.during_seen = None
        view._check_purges()

    src.during_seen = purge_now
    assert view.build("conversations", DAY) is False
    assert view.store.days("conversations", 1) == {}
    assert ("conversations", DAY) in view._queue


def test_a_labeling_rereads_the_reports_about_people_and_keeps_showing_them(tmp_path):
    src = FakeSources([Seen("p-alice", at(DAY, 10), at(DAY, 11), heard=False)], video={DAY: {10}}, first=DAY)
    view = make(tmp_path, src, at(DAY, 23))
    view.view("whos_around")
    view.view("activity")
    view._queue.clear()
    for kind in ("whos_around", "activity"):
        view.build(kind, DAY)
    src.mark = "labels-2"
    out = view.view("whos_around")
    assert out["refreshing"] == [DAY] and out["pending"] == []
    assert len(out["days"]) == 1
    assert view.view("activity")["refreshing"] == [DAY]  # shared mark also covers owner activity evidence


def test_a_settled_days_recorded_hours_are_asked_for_once(tmp_path):
    src = FakeSources(video={DAY: {10}}, first=DAY)
    view = make(tmp_path, src, at(DAY, 12) + 2 * 86400)
    view.build("whos_around", DAY)
    asked = src.hour_reads
    view.build("whos_around", DAY)
    assert src.hour_reads == asked


def test_walkbys_keep_camera_coverage_separate_and_link_to_the_first_real_crossing():
    t = at(DAY, 9, 15)
    src = FakeSources(
        events=[
            {"kind": "crossing", "name": "crossing", "camera": "cam-a", "start": t},
            {"kind": "crossing", "name": "crossing", "camera": "cam-a", "start": t + 60},
            {"kind": "read", "name": "read", "camera": "cam-a", "start": t},
            {"kind": "crossing", "name": "crossing", "camera": "cam-b", "start": at(DAY, 12, 5)},
        ],
        act_cov=[("cam-a", at(DAY, 8), at(DAY, 10)), ("cam-b", at(DAY, 12), at(DAY, 13))],
    )
    report = reports.Walkbys()
    out = report.assemble({DAY: report.day(src, DAY)}, src, None)
    a, b = out["cameras"]
    assert a["total"] == 2
    assert a["days"][0]["hours"][8] == {"n": 0, "t": None}
    assert a["days"][0]["hours"][9] == {"n": 2, "t": t}
    assert a["days"][0]["hours"][12] is None
    assert b["days"][0]["hours"][9] is None
    assert b["days"][0]["hours"][12] == {"n": 1, "t": at(DAY, 12, 5)}
    assert report.assemble({DAY: report.day(src, DAY)}, src, "cam-b")["cameras"] == [b]


def test_person_activity_unions_overlapping_model_windows_and_preserves_unverified_and_unknown_hours():
    t = at(DAY, 10, 30)
    src = FakeSources(
        actions=[
            {"person": "p-alice", "kind": "read", "start": t, "end": t + 3600, "verified": True},
            {"person": "p-alice", "kind": "read", "start": t + 600, "end": t + 1800, "verified": False},
            {"person": None, "kind": "read", "start": t, "end": t + 3600, "verified": False},
            {"person": "p-gone", "kind": "sleep", "start": t, "end": t + 3600, "verified": True},
        ],
        segments={segment_log.ACTION_COVERAGE: [{"ts_start": at(DAY, 9), "ts_end": at(DAY, 12)}]},
    )
    report = reports.PersonActivity()
    body = report.day(src, DAY)
    out = report.assemble({DAY: body}, src, None)
    assert len(out["people"]) == 1
    series = out["people"][0]["series"][0]
    assert series["total_min"] == 60
    assert series["verified"] is False
    hours = series["days"][0]["hours"]
    assert hours[8] is None
    assert hours[9] is None  # producer coverage has no person/kind evaluation assertion
    assert hours[10] == {"minutes": 30, "t": t}
    assert hours[11] == {"minutes": 30, "t": at(DAY, 11)}
    src.names = {"p-dana": "Dana"}
    assert report.assemble({DAY: body}, src, None)["people"] == []


def test_devices_count_distinct_fresh_identifiers_and_keep_partial_or_inactive_coverage_unknown():
    t = at(DAY, 10, 5)
    src = FakeSources(
        segments={
            segment_log.NET_OBSERVATIONS: [
                {"start": t, "end": t, "identifier": "phone-a", "connected": True},
                {"start": t + 60, "end": t + 60, "identifier": "phone-a", "connected": True},
                {"start": t + 120, "end": t + 120, "identifier": "phone-b", "connected": True},
                {"start": t, "end": t, "identifier": "cached", "connected": None},
                {"start": t, "end": t, "identifier": "offline", "connected": False},
            ],
            segment_log.NET_INVENTORY: [
                {"start": at(DAY, 9), "end": at(DAY, 9), "complete": False, "authoritative": True},
                {"start": at(DAY, 11), "end": at(DAY, 11), "complete": True, "authoritative": True},
            ],
            segment_log.BLE_COVERAGE: [
                {"start": t, "end": t + 300, "absence_is_not_inferred": True, "source": {"active_sensors": 0}},
            ],
        }
    )
    report = reports.Devices()
    out = report.assemble({DAY: report.day(src, DAY)}, src, None)
    assert [s["key"] for s in out["sources"]] == ["network"]
    hours = out["sources"][0]["days"][0]["hours"]
    assert hours[9] is None
    assert hours[10] == {"n": 2, "t": t}
    assert hours[11] == {"n": 0, "t": None}
    assert out["sources"][0]["total"] == 2


def test_bluetooth_run_counts_real_endpoints_across_hours_without_filling_the_gap():
    start, end = at(DAY, 9, 59), at(DAY, 11)
    src = FakeSources(segments={segment_log.BLE_SIGHTINGS: [{"start": start, "end": end, "identifier": "tag-a"}]})
    report = reports.Devices()
    out = report.assemble({DAY: report.day(src, DAY)}, src, None)
    hours = out["sources"][0]["days"][0]["hours"]
    assert hours[9] == {"n": 1, "t": start}
    assert hours[10] is None
    assert hours[11] == {"n": 1, "t": end}


def test_suggestions_offer_new_kinds_only_when_their_aggregates_have_positive_observations(tmp_path):
    from liberational.dashboard import reports as page

    t = at(DAY, 10)
    src = FakeSources(
        first=DAY,
        events=[{"kind": "crossing", "name": "crossing", "camera": "cam-a", "start": t}],
        actions=[{"person": "p-alice", "kind": "read", "start": t, "end": t + 600, "verified": False}],
        segments={segment_log.NET_OBSERVATIONS: [{"start": t, "end": t, "identifier": "phone-a", "connected": True}]},
    )
    view = make(tmp_path, src, at(DAY, 23))
    view.build("walkbys", DAY)
    view.build("person_activity", DAY)
    view.build("devices", DAY)
    found = page.suggestions(view)
    # Walk-bys and people's activity are one card each, with a filter.
    assert [(s["id"], s["kind"], s["arg"]) for s in found] == [
        ("walkbys", "walkbys", None),
        ("person_activity", "person_activity", None),
        ("devices:network", "devices", "network"),
    ]
    assert found[0]["title"] == "Walk-bys by camera" and "cam-a" in found[0]["why"]
    assert found[1]["title"] == "Activity by person" and "Alice" in found[1]["why"]
    assert "unverified model spans" in found[1]["why"]
    chosen = [{k: v for k, v in s.items() if k != "why"} for s in found]
    page.save_layout(view.store, chosen)
    assert page.suggestions(view) == []


def test_live_action_source_filters_walkbys_replaced_and_owner_rejected_spans(tmp_path, vision):
    from liberational.memory import events

    t = at(DAY, 10)
    person = registry.at(tmp_path).ensure("Alice", "test")
    vision.add(segment_log.WALKBY_SPANS, ts_start=t, ts_end=t, cam="cam-a", kind="crossing", camera_moved=False)
    old = vision.add(
        segment_log.ACTION_SPANS,
        ts_start=t,
        ts_end=t + 30,
        cam="cam-a",
        kind="read",
        person=person,
        candidate=True,
        camera_moved=False,
    )
    replacement = vision.add(
        segment_log.ACTION_SPANS,
        ts_start=t,
        ts_end=t + 60,
        cam="cam-a",
        kind="read",
        person=person,
        replaces=old["id"],
        candidate=True,
        camera_moved=False,
    )
    wrong = vision.add(
        segment_log.ACTION_SPANS,
        ts_start=t + 60,
        ts_end=t + 90,
        cam="cam-a",
        kind="sleep",
        person=person,
        candidate=True,
        camera_moved=False,
    )
    events.feedback_activity(tmp_path, events.FeedbackActivity(wrong["id"], t + 60, "wrong"))
    rows = reports.LiveSources(tmp_path).action_spans(t, t + 120)
    assert [(r["id"], r["verified"], r["person"]) for r in rows] == [(replacement["id"], False, person)]


def test_walkby_coverage_does_not_treat_model_action_frames_as_evaluated_crossings(tmp_path, vision):
    t = at(DAY, 10)
    vision.add(segment_log.WALKBY_SPANS, ts_start=t, ts_end=t, cam="cam-a", kind="crossing", camera_moved=False)
    vision.add(segment_log.ACTION_COVERAGE, ts_start=at(DAY, 9), ts_end=at(DAY, 11), cam="cam-a", model="action-model")
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=at(DAY, 11), ts_end=at(DAY, 12), cam="cam-a", frames=30)
    src, report = reports.LiveSources(tmp_path), reports.Walkbys()
    out = report.assemble({DAY: report.day(src, DAY)}, src, None)
    hours = out["cameras"][0]["days"][0]["hours"]
    assert hours[9] is None
    assert hours[10] == {"n": 1, "t": t}
    assert hours[11] == {"n": 0, "t": None}


def test_person_activity_coverage_from_other_models_or_cameras_never_becomes_a_zero():
    tomorrow = "2026-10-21"
    t = at(DAY, 10)
    src = FakeSources(
        actions=[{"person": "p-alice", "kind": "read", "start": t, "end": t + 60, "verified": False}],
        segments={
            segment_log.ACTION_COVERAGE: [
                {"ts_start": at(DAY, 9), "ts_end": at(DAY, 12), "cam": "cam-other", "model": "skeleton-only"},
                {"ts_start": at(tomorrow, 9), "ts_end": at(tomorrow, 12), "cam": "cam-other", "model": "skeleton-only"},
            ]
        },
    )
    report = reports.PersonActivity()
    out = report.assemble({DAY: report.day(src, DAY), tomorrow: report.day(src, tomorrow)}, src, None)
    rows = out["people"][0]["series"][0]["days"]
    assert rows[0]["hours"][9] is None
    assert rows[0]["hours"][10] == {"minutes": 1, "t": t}
    assert rows[1]["minutes"] is None
    assert rows[1]["hours"] == [None] * 24


def test_device_only_history_discovers_the_first_local_day_from_segment_hour_files(tmp_path, vision):
    # No video or timeline at all. The UTC midnight file is the preceding local day in Pacific time.
    network = segment_log.root(tmp_path) / segment_log.NET_OBSERVATIONS / "router"
    network.mkdir(parents=True)
    (network / "2026102000-0000-s0-1.arrows").touch()
    (network / "2026102100-0000-s1.arrows").touch()
    (network / "2026010100-not-a-part.arrows").touch()
    assert reports.LiveSources(tmp_path).first_day() == "2026-10-19"
    radio = segment_log.root(tmp_path) / segment_log.BLE_SIGHTINGS / "ble"
    radio.mkdir(parents=True)
    (radio / "2026101808-0000-s0-1-v2.arrows").touch()
    src = reports.LiveSources(tmp_path)
    assert src.first_day() == "2026-10-18"
    view = reports.Reports(reports.Store(tmp_path / "reports.sqlite3"), src, background=False, now=lambda: at(DAY, 23))
    assert view.window() == ("2026-10-18", DAY)


@pytest.mark.parametrize(
    "kind,stream,field",
    [
        ("walkbys", segment_log.WALKBY_SPANS, "cameras"),
        ("person_activity", segment_log.ACTION_SPANS, "people"),
        ("activity", segment_log.WALKBY_SPANS, "events"),
    ],
)
def test_settled_reports_rebuild_after_owner_rejects_activity_evidence(tmp_path, vision, kind, stream, field):
    from liberational.memory import events

    t = at(DAY, 10)
    reg = registry.at(tmp_path)
    person = reg.ensure("Alice", "test")
    source = vision.add(
        stream,
        ts_start=t,
        ts_end=t + 60,
        cam="cam-a",
        person=person,
        kind="crossing" if stream == segment_log.WALKBY_SPANS else "read",
        camera_moved=False,
        candidate=stream != segment_log.WALKBY_SPANS,
    )
    vision.video.append(("cam-a", t, t + 60))
    src = reports.LiveSources(tmp_path)
    view = reports.Reports(
        reports.Store(tmp_path / "reports.sqlite3"), src, background=False, now=lambda: at(DAY, 23) + 3 * 86400
    )
    view.view(kind)  # establish the current owner-evidence mark
    view.build(kind, DAY)
    assert view.store.days(kind, 1)[DAY][0] is True
    assert len(view.view(kind)[field]) == 1
    mark = reg.mark()
    events.feedback_activity(tmp_path, events.FeedbackActivity(source["id"], t, "wrong"))
    assert reg.mark() == mark  # activity ratings are knowledge rows, not registry events
    assert DAY in view.view(kind)["refreshing"]
    view.drain()
    assert view.view(kind)[field] == []


def test_settled_person_activity_resolves_merged_ids_before_unioning_source_windows(tmp_path, vision):
    t = at(DAY, 10)
    reg = registry.at(tmp_path)
    old, survivor = reg.ensure("Dana", "test"), reg.ensure("Alice", "test")
    vision.add(
        segment_log.ACTION_SPANS,
        ts_start=t,
        ts_end=t + 60,
        cam="cam-a",
        person=old,
        kind="read",
        candidate=True,
        camera_moved=False,
    )
    vision.add(
        segment_log.ACTION_SPANS,
        ts_start=t + 30,
        ts_end=t + 90,
        cam="cam-a",
        person=survivor,
        kind="read",
        candidate=True,
        camera_moved=False,
    )
    vision.video.append(("cam-a", t, t + 90))
    view = reports.Reports(
        reports.Store(tmp_path / "reports.sqlite3"),
        reports.LiveSources(tmp_path),
        background=False,
        now=lambda: at(DAY, 23) + 3 * 86400,
    )
    view.view("person_activity")
    view.build("person_activity", DAY)
    assert len(view.view("person_activity")["people"]) == 2
    reg.merge("Dana", "Alice", "test")
    assert DAY in view.view("person_activity")["refreshing"]
    view.drain()
    people = view.view("person_activity")["people"]
    assert [(p["id"], p["name"]) for p in people] == [(survivor, "Alice")]
    assert people[0]["series"][0]["total_min"] == 1.5  # overlap unioned, not 1 + 1 minutes


@pytest.mark.parametrize(
    "settings",
    [
        {"events": {"answer_counts": False}},
        {"activity": {"promotion_min_ratings": 12}},
        {"activity": {"promotion_min_precision": 0.9}},
    ],
)
def test_activity_promotion_settings_change_the_cached_evidence_mark(tmp_path, settings):
    src = reports.LiveSources(tmp_path)
    before = src.identity_mark()
    with config.override(settings):
        assert src.identity_mark() != before


# ---- the worker process ----


class FakePool:
    """A ProcessPoolExecutor stand-in: each submit answers from `answers` in turn (an exception is raised), and the
    calls it was given are kept."""

    made: ClassVar[list[FakePool]] = []

    def __init__(self, answers):
        self.answers, self.calls, self.shut = answers, [], False
        FakePool.made.append(self)

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        future = Future()
        future.set_result(answer)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.shut = True


@pytest.fixture
def pools(monkeypatch):
    answers = []
    FakePool.made = []
    monkeypatch.setattr(reports, "ProcessPoolExecutor", lambda **_: FakePool(answers))
    monkeypatch.setattr(reports, "_executor", None)
    monkeypatch.setattr(reports, "READ_IN_WORKER", True)
    return answers


def test_with_the_worker_a_day_is_read_there_and_stored_here(tmp_path, pools):
    view = make(tmp_path, FakeSources(first=DAY), at(DAY, 12))
    view.state = tmp_path
    body = {"hours": [None] * 24, "people": {}}
    pools.append(body)
    assert view.build("whos_around", DAY)
    assert FakePool.made[0].calls == [(reports._read_day, (str(tmp_path), "whos_around", DAY))]
    report = reports.REPORTS["whos_around"]
    assert view.store.days(report.name, report.version)[DAY][2] == body


def test_a_worker_that_died_fails_that_day_and_the_next_read_starts_another(tmp_path, pools):
    view = make(tmp_path, FakeSources(first=DAY), at(DAY, 12))
    view.state = tmp_path
    pools += [BrokenProcessPool("killed"), {"hours": [None] * 24, "people": {}}]
    with pytest.raises(BrokenProcessPool):
        view.build("whos_around", DAY)
    assert FakePool.made[0].shut
    assert view.build("whos_around", DAY)
    assert len(FakePool.made) == 2


def test_without_the_worker_a_day_is_read_in_process(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "READ_IN_WORKER", False)
    monkeypatch.setattr(reports, "ProcessPoolExecutor", lambda **_: pytest.fail("no worker in tests"))
    view = make(tmp_path, FakeSources(first=DAY), at(DAY, 12))
    view.state = tmp_path
    assert view.build("whos_around", DAY)


def test_activity_and_walkbys_share_one_read_of_a_days_coverage(tmp_path, monkeypatch):
    from liberational.memory import events

    reads = []

    def coverage(state, start, end):
        reads.append((start, end))
        return [("cam0", start, end)]

    monkeypatch.setattr(events, "coverage", coverage)
    src = reports.LiveSources(tmp_path)
    lo, hi = at(DAY, 0), at(DAY, 23, 59)
    assert src.activity_coverage(lo, hi) == [("cam0", lo, hi)]
    assert src.activity_coverage(lo, hi) == [("cam0", lo, hi)]
    assert reads == [(lo, hi)]
    monkeypatch.setattr(reports, "COVERAGE_KEEP_S", 0.0)  # once it's older than that, it's read again
    src.activity_coverage(lo, hi)
    assert reads == [(lo, hi), (lo, hi)]


def test_the_event_activity_leads_with_is_not_suggested_while_activity_is_on_the_page(tmp_path):
    from liberational.dashboard import reports as page

    t = at(DAY, 9)
    src = FakeSources(
        [],
        video={DAY: {9}},
        events=[*({"name": "walked past", "start": t + i} for i in range(5)), {"name": "fridge", "start": t + 9}],
        act_cov=[("cam0", t, t + 60)],
        first=DAY,
    )
    view = make(tmp_path, src, at(DAY, 23))
    view.build("activity", DAY)
    events = [s["title"] for s in page.suggestions(view) if s["arg"].startswith("event:")]
    assert events == ["fridge, day by day"]  # the default page has Activity, which leads with "walked past"


def test_zone_coverage_is_each_kinds_stretches_joined_and_cut_to_the_window(tmp_path, vision):
    t = dt.datetime(2026, 10, 20, 9, tzinfo=dt.UTC).timestamp()
    for a, b, kind, cam in (
        (0, 60, "person", "cam0"),
        (60, 120, "person", "cam0"),
        (30, 90, "animal", "cam0"),
        (0, 30, "animal", "cam1"),
    ):
        vision.add(
            segment_log.ZONE_COVERAGE,
            ts_start=t + a,
            ts_end=t + b,
            cam=cam,
            subject_kind=kind,
            definition="d",
            zones=["z1"],
            frames=10,
        )
    got = reports.LiveSources(tmp_path).zone_coverage(t + 10, t + 100)
    assert [(c["subject_kind"], c["cam"], c["ts_start"] - t, c["ts_end"] - t) for c in got] == [
        ("person", "cam0", 10, 100),
        ("animal", "cam1", 10, 30),
        ("animal", "cam0", 30, 90),
    ]
