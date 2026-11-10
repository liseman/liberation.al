"""Retention past the window (#2809): what's condensed, what's dropped, what's kept, and the day-page gate."""

import asyncio
from typing import Any, cast

import pytest

from liberational import retention as R
from liberational import segment_log as S
from liberational import video_window as V
from liberational.continuum import to_ns
from liberational.memory_log import day_bounds

DAY = "2026-09-01"
A, B = day_bounds(DAY)
HOUR = 3600.0


def cover(t, end=None, **kw):
    return {
        "id": S.new_id(),
        "ts_start": t,
        "ts_end": t + 0.9 if end is None else end,
        "cam": "cam0",
        "frames": 5,
        **kw,
    }


def seen(t, **kw):
    r = {
        "id": S.new_id(),
        "source": "unifi",
        "start": t,
        "end": t,
        "ts_start": t,
        "inventory_id": S.new_id(),
        "observer": "local-network",
        "identifier_kind": "mac_address",
        "identifier": "AA:BB:CC:00:00:01",
        "mac": "AA:BB:CC:00:00:01",
        "hostname": "phone",
        "display_name": None,
        "ip": "10.0.0.5",
        "vendor": None,
        "ap_id": "AP:1",
        "ap_name": None,
        "connection_type": "wireless",
        "connected": True,
    }
    return {**r, **kw}


def test_coverage_seconds_become_one_interval_with_their_frames_summed():
    t = A + 100
    got = R.merge_coverage([cover(t), cover(t + 1), cover(t + 2.5)])
    assert [(r["ts_start"], r["ts_end"], r["frames"]) for r in got] == [(t, t + 3.4, 15)]


@pytest.mark.parametrize(
    "second",
    [
        pytest.param(cover(A + 100 + 4.0), id="a-gap-over-two-seconds"),
        pytest.param(cover(A + 101, model="other"), id="another-model"),
    ],
)
def test_coverage_with_a_gap_or_different_fields_stays_apart(second):
    first = cover(A + 100, model="m")
    if "model" not in second:
        second = {**second, "model": "m"}
    assert len(R.merge_coverage([first, second])) == 2


def test_an_interval_never_crosses_a_utc_hour():
    top = (A // HOUR + 2) * HOUR
    got = R.merge_coverage([cover(top - 1), cover(top)])
    assert [r["ts_start"] for r in got] == [top - 1, top]


def test_a_devices_polls_become_a_visit_until_it_moves_or_goes_quiet():
    t = A + 100
    polls = [seen(t), seen(t + 30), seen(t + 60), seen(t + 90, ap_id="AP:2"), seen(t + 400, ap_id="AP:2")]
    got = R.merge_sightings(polls, R._net_key)
    assert [(r["start"], r["end"], r["ap_id"]) for r in got] == [
        (t, t + 60, "AP:1"),
        (t + 90, t + 90, "AP:2"),  # a new access point is a new visit
        (t + 400, t + 400, "AP:2"),  # over a minute later: another
    ]


def test_audio_near_speech_is_kept_and_the_rest_is_silent():
    t = A + 1000
    utterances = [
        {
            "ts_start": t + 20,
            "ts_end": t + 22,
            "audio": {"stream": "mic.audio", "start_ns": to_ns(t + 20), "end_ns": to_ns(t + 22)},
        },
        {
            "ts_start": t + 200,
            "ts_end": t + 205,
            "audio": {"stream": "other.audio", "start_ns": to_ns(t + 200), "end_ns": to_ns(t + 205)},
        },
    ]
    activity = [
        {"ts_start": t + 100, "stream": "mic.audio", "in_speech": True},
        {"ts_start": t + 160, "stream": "mic.audio", "in_speech": False},
    ]
    speech = R.speech_spans(utterances, activity, "mic.audio")
    chunks = [
        ("before", t, t + 16.0),  # ends 1 s inside the margin before the utterance
        ("quiet", t + 30, t + 45),
        ("gate", t + 90, t + 105),
        ("not-speech", t + 150, t + 165),
        ("other-mic", t + 195, t + 210),  # speech, but on another microphone
    ]
    assert R.silent(chunks, speech) == ["quiet", "not-speech", "other-mic"]


def test_due_days_are_past_the_window_have_a_page_and_arent_done():
    now = day_bounds("2026-10-20")[0]
    written = {"2026-09-01", "2026-09-03", "2026-09-19", "2026-09-20"}
    got = R.due("2026-09-01", {"2026-09-01"}, now, 30, written.__contains__)
    assert got == ["2026-09-03", "2026-09-19"]  # 09-02 has no page; 09-20 ends inside the window


def test_a_run_takes_at_most_a_week_of_days():
    now = day_bounds("2026-12-01")[0]
    got = R.due("2026-09-01", (), now, 30, lambda day: True)
    assert got == [f"2026-09-0{d}" for d in range(1, 8)]


def test_an_empty_store_has_no_days():
    assert R.due(None, (), A + 90 * 86400, 30, lambda day: True) == []


def test_finished_days_are_recorded_and_read_back(tmp_path):
    marks = R.Marks.load(tmp_path)
    marks.done(DAY, {"face.match bytes": 10})
    assert R.Marks.load(tmp_path).days == {DAY: {"face.match bytes": 10}}


@pytest.fixture
def store(tmp_path):
    pytest.importorskip("continuum_py").SegmentWriter  # noqa: B018 (the native store must be built)
    return tmp_path / "segments"


def write(base, stream, records):
    writers = S.Writers(base)
    writers.append(stream, "cam0", records)
    writers.sync()
    writers.close()


def face(stream, t):
    rows = {
        S.FACE_VISITS: {"name": "Dana", "best_sim": 0.8, "samples": 3, "shots": "[]"},
    }
    return {"id": S.new_id(), "ts_start": t, "ts_end": t + 5, "cam": "cam0", **rows[stream]}


def test_a_day_is_condensed_its_per_frame_rows_dropped_and_its_spans_kept(store):
    t = A + 5000
    write(store, S.WALKBY_COVERAGE, [cover(t + i) for i in range(10)])
    write(store, S.WALKBY_COVERAGE, [cover(B + 50)])  # the next day stays as it is
    write(store, S.FACE_VISITS, [face(S.FACE_VISITS, t)])
    write(
        store,
        S.FACE_SIGNALS,
        [
            {
                "id": S.new_id(),
                "ts_start": t,
                "ts_end": t,
                "cam": "cam0",
                "match_id": S.new_id(),
                "head": [0.1, 0.2, 0.3, 0.4, 0.5],
                "frontal": 0.9,
                "norm": 1.0,
            }
        ],
    )
    counts = R.segments_day(R.Store(store), DAY)
    log = S.Log(store)
    got = [(r["ts_start"], r["ts_end"], r["frames"]) for r in log.range(S.WALKBY_COVERAGE, A, B + HOUR)]
    assert got == [(pytest.approx(t), pytest.approx(t + 9.9), 50), (pytest.approx(B + 50), pytest.approx(B + 50.9), 5)]
    assert log.range(S.FACE_SIGNALS, A, B) == []
    assert len(log.range(S.FACE_VISITS, A, B)) == 1
    assert counts[f"{S.WALKBY_COVERAGE} rows"] == 10 and counts[f"{S.WALKBY_COVERAGE} kept"] == 1

    R.segments_day(R.Store(store), DAY)  # a day done again (a run that stopped before recording it)
    assert len(log.range(S.WALKBY_COVERAGE, A, B)) == 1, "its condensed rows aren't written twice"


class Broker:
    """Audio chunks per stream, as the broker's query gives them, and what was aged out."""

    def __init__(self, chunks: dict[str, list[tuple[str, float, float]]]):
        self.chunks = chunks
        self.aged: list[list[str]] = []

    async def list_streams(self) -> list[dict[str, Any]]:
        return [{"name": name} for name in self.chunks] + [{"name": "cam0.video"}]

    async def query(self, stream: str, start_ns: int, end_ns: int, *, all: bool = True) -> dict[str, Any]:
        del all
        items = [
            {"header": header(i, a, b)} for i, a, b in self.chunks[stream] if to_ns(a) < end_ns and to_ns(b) > start_ns
        ]
        return {"items": items}


def header(i: str, a: float, b: float) -> bytes:
    return pytest.importorskip("continuum_py").encode_header(
        {
            "id": i,
            "stream_id": 7,
            "start_ns": to_ns(a),
            "end_ns": to_ns(b),
            "layer": "Layer1Raw",
            "format": "RawBytes",
            "shape_name": "PCM_F32LEx1",
            "parents": [],
            "subjects": [],
            "seq": 0,
        }
    )


def test_a_days_silent_audio_chunks_are_aged_out_and_speech_stays(tmp_path, vision):
    t = A + 1000
    quiet, spoken = "00000000-0000-7000-8000-000000000001", "00000000-0000-7000-8000-000000000002"
    broker = Broker({"mic.audio": [(quiet, t, t + 15), (spoken, t + 15, t + 30)]})
    vision.add(
        S.UTTERANCES,
        ts_start=t + 20,
        ts_end=t + 24,
        device="mic",
        audio={"stream": "mic.audio", "start_ns": to_ns(t + 20), "end_ns": to_ns(t + 24)},
    )
    aged: list[tuple[list[str], int]] = []

    async def age_out(ids: list[str], before_ns: int, reason: str) -> int:
        aged.append((ids, before_ns))
        assert "silent audio" in reason
        return len(ids)

    counts = asyncio.run(V.silent_audio(cast(Any, broker), tmp_path, DAY, to_ns(B), age_out))
    assert aged == [([quiet], to_ns(B))]
    assert counts == {"mic.audio chunks": 2, "mic.audio silent dropped": 1}
