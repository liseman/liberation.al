"""The speech log (src/liberational/speech_log.py): lines written to the segment store read back as v1's lines, the
process-local history behind since and mark, live and durable following, and the broker purge marks."""

import asyncio
import math
import threading
import uuid
from pathlib import Path

import continuum_py
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from liberational import segment_log, speech_log
from liberational.listen import revisions
from liberational.people import transcript_view

T = 1_800_000_000.0
UTTERANCE = {
    "ts_start": T + 0.25,
    "ts_end": T + 2.5,
    "device": "wave-3",
    "speaker": "Speaker 4",
    "speaker_id": 4,
    "speaker_sim": 0.81,
    "text": "turn the lights off",
    "lat_ms": {"asr": 120, "emb": 9},
    "audio": {"stream": "wave-3.audio", "start_ns": 1_800_000_000_250_000_000, "end_ns": 1_800_000_002_500_000_000},
}


@pytest.fixture
def store(tmp_path: Path) -> Path:
    pytest.importorskip("continuum_py").SegmentFollowers  # noqa: B018 (the native store must be built)
    return tmp_path / "segments"


def write(store: Path, slot: str, *lines: tuple[str, dict]) -> list[str]:
    writer = speech_log.Writer(slot, store)
    try:
        ids = [writer.write(stream, line) for stream, line in lines]
        writer.sync()
        return ids
    finally:
        writer.close()


def test_rejected_turn_is_hidden_without_rewriting_raw_rows_and_undo_restores_it(store):
    write(
        store,
        "test",
        (speech_log.UTTERANCES, UTTERANCE),
        (speech_log.UTTERANCES, {**UTTERANCE, "ts_start": T + 5, "ts_end": T + 8, "text": "keep this line"}),
    )
    raw = speech_log.lines(T, T + 10, base=store)
    write(
        store,
        "test-activity",
        (
            speech_log.ACTIVITY,
            {
                "ts_start": T + 0.25,
                "ts_end": T + 2.5,
                "device": "wave-3",
                "stream": "wave-3.audio",
                "in_speech": True,
                "level_dbfs": -20.0,
                "threshold_dbfs": -40.0,
            },
        ),
    )
    activity = speech_log.lines(T, T + 10, (speech_log.ACTIVITY,), base=store)
    token = revisions.reject_lines(store.parent, raw[:1])
    assert [row["text"] for row in speech_log.lines(T, T + 10, base=store)] == ["keep this line"]
    assert [row["text"] for row in transcript_view.View(store.parent).lines(raw)] == ["keep this line"]
    assert segment_log.Log(store).range(speech_log.UTTERANCES, T, T + 10) == raw
    assert speech_log.lines(T, T + 10, (speech_log.ACTIVITY,), base=store) == activity
    assert revisions.undo_correction(store.parent, token) == ["wave-3@1800000000.250"]
    assert speech_log.lines(T, T + 10, base=store) == raw


def test_one_durable_undo_restores_whole_rejected_turn(store):
    write(
        store,
        "test",
        (speech_log.UTTERANCES, UTTERANCE),
        (speech_log.UTTERANCES, {**UTTERANCE, "ts_start": T + 5, "ts_end": T + 8}),
    )
    raw = speech_log.lines(T, T + 10, base=store)
    before = revisions.review_revision(store.parent, T, T + 10)
    token = revisions.reject_lines(store.parent, raw)
    assert revisions.reject_lines(store.parent, raw) == token
    assert revisions.review_revision(store.parent, T, T + 10) > before
    assert revisions.filter_lines(store.parent, raw) == []
    revisions.undo_correction(store.parent, token)
    assert revisions.filter_lines(store.parent, raw) == raw
    with pytest.raises(ValueError, match="already"):
        revisions.undo_correction(store.parent, token)


def close(a: object, b: object) -> bool:
    """Equal lines, with times to the microsecond and gate levels to float32."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, float) and not isinstance(b, bool):
        return isinstance(b, int | float) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-4)
    return a == b


# ---- The codecs: a line survives its typed columns

times = st.floats(min_value=T, max_value=T + 86400, allow_nan=False)
names = st.text(alphabet="abcdefghij ", min_size=1, max_size=12)


def optional(key: str, values: st.SearchStrategy) -> st.SearchStrategy[dict]:
    return st.one_of(st.just({}), values.map(lambda v: {key: v}))


@st.composite
def utterances(draw: st.DrawFn) -> dict:
    start = draw(times)
    line: dict[str, object] = {"ts_start": start, "ts_end": start + draw(st.floats(0, 30))}
    line |= {"device": draw(names), "text": draw(names)}
    line |= draw(optional("speaker", names)) | draw(optional("speaker_id", st.integers(1, 10_000)))
    line |= draw(optional("speaker_sim", st.integers(0, 1000).map(lambda n: n / 1000)))
    steps = draw(st.dictionaries(st.sampled_from(segment_log.LATENCIES), st.integers(0, 60_000)))
    line |= {"lat_ms": steps} if steps else {}
    line |= draw(optional("password", st.just(True)))
    if draw(st.booleans()):
        a, b = sorted(draw(st.lists(st.integers(0, 2**62), min_size=2, max_size=2)))
        line["audio"] = {"stream": "mic.audio", "start_ns": a, "end_ns": b}
    return line


@st.composite
def replies(draw: st.DrawFn) -> dict:
    start = draw(times)
    line = {"ts_start": start, "ts_end": start + draw(st.floats(0, 30)), "text": draw(names), "assistant": True}
    for key in ("speaker", "reply_to", "route", "error"):
        line |= draw(optional(key, names))
    return line


@st.composite
def gate_states(draw: st.DrawFn) -> dict:
    line = {"ts_start": draw(times), "device": draw(names), "stream": "mic.audio", "in_speech": draw(st.booleans())}
    for key in ("level_dbfs", "threshold_dbfs"):
        line |= draw(optional(key, st.integers(-1200, 0).map(lambda n: n / 10)))
    return line


LINES = {speech_log.UTTERANCES: utterances(), speech_log.REPLIES: replies(), speech_log.ACTIVITY: gate_states()}


@pytest.mark.parametrize("stream", sorted(LINES))
@settings(max_examples=80, deadline=None)
@given(data=st.data())
def test_a_line_survives_its_typed_columns(stream: str, data: st.DataObject) -> None:
    line = data.draw(LINES[stream])
    if stream != speech_log.ACTIVITY:
        line["id"] = str(uuid.uuid7())
    spec = segment_log.STREAMS[stream]
    row = spec.to_row(line)
    assert set(row) == {name for name, _ in spec.schema}, "the row is exactly the schema's columns"
    assert close(spec.to_record(row, "slot"), line)


# ---- Writing and reading, through the native store


def test_an_utterance_reads_back_as_the_line_it_was(store: Path) -> None:
    (line_id,) = write(store, "wave-3", (speech_log.UTTERANCES, UTTERANCE))
    (back,) = speech_log.lines(T, T + 10, base=store)
    assert close(back, {**UTTERANCE, "id": line_id})


def test_an_exact_span_is_the_lines_span_not_its_seconds(store: Path) -> None:
    writer = speech_log.Writer("wave-3", store)
    span = (speech_log.to_ns(T + 5), speech_log.to_ns(T + 6))
    writer.write(speech_log.UTTERANCES, {"ts_start": T, "ts_end": T, "text": "hi"}, span_ns=span)
    writer.close()
    assert speech_log.lines(T - 1, T + 1, base=store) == []
    (back,) = speech_log.lines(T + 4, T + 10, base=store)
    assert (back["ts_start"], back["ts_end"]) == pytest.approx((T + 5, T + 6), abs=1e-6)


def test_keys_a_stream_doesnt_carry_are_refused_not_dropped(store: Path) -> None:
    writer = speech_log.Writer("wave-3", store)
    try:
        with pytest.raises(ValueError, match=r"don't carry \['mood'\]"):
            writer.write(speech_log.UTTERANCES, {"ts_start": T, "text": "hi", "mood": "glad"})
    finally:
        writer.close()


def test_the_log_reads_utterances_and_replies_in_time_order_across_slots(store: Path) -> None:
    write(store, "Mic A", (speech_log.UTTERANCES, {"ts_start": T + 100, "ts_end": T + 101, "text": "what time is it"}))
    write(store, "voice", (speech_log.REPLIES, {"ts_start": T + 101.5, "ts_end": T + 102, "text": "3 pm"}))
    write(
        store,
        "Mic B",
        (speech_log.UTTERANCES, {"ts_start": T + 99, "ts_end": T + 99.5, "text": "hey bob"}),
        (speech_log.UTTERANCES, {"ts_start": T + 200, "ts_end": T + 201, "text": "later"}),
    )
    both = speech_log.lines(T + 98, T + 150, (speech_log.UTTERANCES, speech_log.REPLIES), base=store)
    assert [(line["text"], line.get("assistant", False)) for line in both] == [
        ("hey bob", False),
        ("what time is it", False),
        ("3 pm", True),
    ]
    assert [line["text"] for line in speech_log.lines(T + 98, T + 150, base=store)] == ["hey bob", "what time is it"]


def test_a_line_that_started_before_the_window_is_found_in_it(store: Path) -> None:
    write(store, "mic", (speech_log.UTTERANCES, {"ts_start": T, "ts_end": T + 20, "text": "a long story"}))
    assert [line["text"] for line in speech_log.lines(T + 10, T + 11, base=store)] == ["a long story"]
    assert speech_log.lines(T + 21, T + 30, base=store) == []


def test_a_second_row_of_the_same_id_is_left_out(store: Path) -> None:
    line = {"id": str(uuid.uuid7()), "ts_start": T, "ts_end": T + 1, "text": "first pass"}
    write(store, "ingest-1", (speech_log.UTTERANCES, line), (speech_log.UTTERANCES, {**line, "text": "second pass"}))
    log = speech_log.SpeechLog(segments=store)
    try:
        assert [x["text"] for x in log.range(T - 1, T + 5)] == ["first pass"]
        assert [x["text"] for x in log.since(0)[0]] == ["first pass"]
    finally:
        log.close()


def test_the_newest_fresh_gate_state_per_device(store: Path) -> None:
    write(
        store,
        "alpha",
        (speech_log.ACTIVITY, {"ts_start": T + 1, "device": "alpha", "in_speech": False}),
        (speech_log.ACTIVITY, {"ts_start": T + 2, "device": "alpha", "in_speech": True, "level_dbfs": -26.0}),
    )
    write(store, "beta", (speech_log.ACTIVITY, {"ts_start": T - 10, "device": "beta", "in_speech": True}))
    assert speech_log.activity(T + 2.5, 3.0, base=store) == {"alpha": True}  # beta's state is stale: unknown
    assert speech_log.activity(T + 2.5, 3.0, base=store / "nothing-here") == {}


def test_a_gate_state_written_after_the_first_ask_is_followed_without_reading_the_window_again(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(store, "alpha", (speech_log.ACTIVITY, {"ts_start": T + 1, "device": "alpha", "in_speech": False}))
    assert speech_log.activity(T + 2, 3.0, base=store) == {"alpha": False}
    reads = []
    real = speech_log.lines
    monkeypatch.setattr(speech_log, "lines", lambda *a, **k: reads.append(a) or real(*a, **k))
    write(
        store,
        "alpha-later",
        (speech_log.ACTIVITY, {"ts_start": T + 3, "device": "alpha", "in_speech": True}),
        (speech_log.ACTIVITY, {"ts_start": T + 3, "device": "beta", "in_speech": False}),
    )
    assert speech_log.activity(T + 3.5, 3.0, base=store) == {"alpha": True, "beta": False}
    assert speech_log.activity(T + 7, 3.0, base=store) == {}  # every state has gone stale
    assert reads == []


# ---- SpeechLog: the history behind since and mark, following, and writing


def test_a_reader_takes_only_new_lines_and_the_mark_moves_with_them(broker: Path, store: Path) -> None:
    write(store, "Mic A", (speech_log.UTTERANCES, {"ts_start": T + 400, "text": "one"}))
    log = speech_log.SpeechLog(broker, segments=store)  # the mark's purge half is the broker's
    try:
        first, seq = log.since(0)
        mark = log.mark()
        assert [line["text"] for line in first] == ["one"]
        assert log.since(seq) == ([], seq)  # nothing new
        assert log.mark() == mark  # and the mark holds still

        write(store, "voice", (speech_log.REPLIES, {"ts_start": T + 400.5, "text": "hello"}))
        assert log.mark() == mark, "a reply doesn't move the utterances' mark"
        write(store, "Mic A", (speech_log.UTTERANCES, {"ts_start": T + 401, "text": "two"}))
        new, after = log.since(seq)
        assert [line["text"] for line in new] == ["two"]
        assert new[0]["seq"] == after
        assert log.mark()[0] == after  # a new line moves the mark
        both, _ = log.since(seq, (speech_log.UTTERANCES, speech_log.REPLIES))
        assert [line["text"] for line in both] == ["hello", "two"]
    finally:
        log.close()


def test_an_empty_store_reads_as_empty(store: Path) -> None:
    log = speech_log.SpeechLog(segments=store)
    try:
        streams = (speech_log.UTTERANCES, speech_log.REPLIES)
        assert log.since(0, streams) == ([], 0)
        assert log.range(0.0, 2.0**32, streams) == []
    finally:
        log.close()


def test_retained_is_every_utterance_in_time_order_as_read_only_rows(
    broker: Path, store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(
        store,
        "mic",
        (speech_log.UTTERANCES, {"ts_start": T + 5, "text": "later"}),
        (speech_log.UTTERANCES, {"ts_start": T + 1, "text": "sooner"}),
    )
    write(store, "voice", (speech_log.REPLIES, {"ts_start": T + 2, "text": "a reply"}))
    log = speech_log.SpeechLog(broker, segments=store)  # its cache key's purge half is the broker's
    monkeypatch.setattr(speech_log, "shared", lambda: log)
    try:
        got = speech_log.retained()
        assert "seq" not in got[0]
        assert isinstance(got, tuple), "a caller cannot append to the cache"
        assert speech_log.retained() is got, "the same rows, no copy per call"
        transcript_view.View(store).lines(got)  # a projection reads them without changing them
        assert [line["text"] for line in speech_log.retained()] == ["sooner", "later"], "the projection leaves them"
        write(store, "mic", (speech_log.UTTERANCES, {"ts_start": T + 3, "text": "new"}))
        assert [line["text"] for line in speech_log.retained()] == ["sooner", "new", "later"], "a new line is seen"
    finally:
        log.close()


def test_following_from_now_passes_over_what_was_there_then_yields_each_line(store: Path) -> None:
    write(store, "mic", (speech_log.UTTERANCES, {"ts_start": T, "text": "before"}))
    log = speech_log.SpeechLog(client_name="voice", segments=store)
    try:
        follower = log.follow(idle_s=0.05)
        assert next(follower) is None, "nothing new yet: an idle tick"
        write(store, "mic", (speech_log.UTTERANCES, {"ts_start": T + 1, "text": "after"}))
        line = next(x for x in follower if x is not None)
        follower.close()
    finally:
        log.close()
    assert line["text"] == "after"


def test_following_from_a_time_catches_up_from_it(store: Path) -> None:
    write(
        store,
        "mic",
        (speech_log.UTTERANCES, {"ts_start": T + 5, "ts_end": T + 6, "text": "early"}),
        (speech_log.UTTERANCES, {"ts_start": T + 50, "ts_end": T + 51, "text": "hello"}),
    )
    log = speech_log.SpeechLog(segments=store)
    try:
        follower = log.follow(since=T + 40)
        first = next(follower)
        follower.close()
    finally:
        log.close()
    assert first is not None and first["text"] == "hello"


def test_an_appended_reply_is_written_under_the_logs_name_and_followed(store: Path) -> None:
    log = speech_log.SpeechLog(client_name="voice", segments=store)
    try:
        follower = log.follow([speech_log.REPLIES], idle_s=0.05)
        assert next(follower) is None
        line = {"ts_start": T + 10, "ts_end": T + 11, "speaker": "Bob", "text": "hi"}
        reply_id = log.append(speech_log.REPLIES, line)
        got = next(x for x in follower if x is not None)
        follower.close()
    finally:
        log.close()
    assert (got["id"], got["text"], got["assistant"]) == (reply_id, "hi", True)
    assert (store / speech_log.REPLIES / "voice").is_dir()


def test_a_durable_follower_resumes_after_the_last_line_it_saved(store: Path) -> None:
    write(store, "mic", *[(speech_log.UTTERANCES, {"ts_start": T + n, "text": t}) for n, t in enumerate(("a", "b"))])
    fresh = speech_log.Follower(base=store)
    assert fresh.poll() == [], "with no saved cursor it starts from now"
    write(store, "mic", (speech_log.UTTERANCES, {"ts_start": T + 3, "text": "three"}))
    write(store, "other mic", (speech_log.UTTERANCES, {"ts_start": T + 4, "text": "four"}))
    got = fresh.poll()
    assert [line["text"] for line, _ in got] == ["three", "four"]
    saved = got[0][1]  # "three" handled, then a restart
    again = speech_log.Follower(cursor=saved, base=store)
    assert [line["text"] for line, _ in again.poll()] == ["four"]


def test_two_writers_of_one_slot_cannot_both_hold_it(store: Path) -> None:
    first = speech_log.Writer("mic", store)
    first.write(speech_log.UTTERANCES, {"ts_start": T, "text": "hi"})
    try:
        with pytest.raises(Exception, match="another writer holds"):
            speech_log.Writer("mic", store).write(speech_log.UTTERANCES, {"ts_start": T, "text": "again"})
    finally:
        first.close()


@pytest.mark.parametrize(
    ("name", "slot"),
    [("Wave 3 (USB)", "Wave-3--USB-"), ("..hidden", "hidden"), ("", "unnamed"), ("mic_a.1-2", "mic_a.1-2")],
)
def test_a_slot_is_a_valid_store_name_for_any_device_name(name: str, slot: str) -> None:
    assert segment_log.slot(name) == slot


def test_history_reads_are_safe_from_several_threads(store: Path) -> None:
    write(store, "mic", *[(speech_log.UTTERANCES, {"ts_start": T + n, "text": str(n)}) for n in range(50)])
    log = speech_log.SpeechLog(segments=store)
    counts: list[int] = []
    try:
        threads = [threading.Thread(target=lambda: counts.append(len(log.since(0)[0]))) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
    finally:
        log.close()
    assert counts == [50] * 8


# ---- The broker side: purge marks and the voice turn


async def _commit_other(sock: Path) -> str:
    """A payload on some other broker stream (speech isn't on the broker any more), for a purge to take."""
    async with await continuum_py.Client.connect(str(sock), client_name="writer") as c:
        reg = await c.register_stream(
            "test.thing", kind="derived", format="Text", shape_name="test.thing", slab_class=0
        )
        pid = str(uuid.uuid7())
        header: continuum_py.PayloadHeader = {
            "id": pid,
            "stream_id": reg["stream_id"],
            "start_ns": 400 * 10**9,
            "end_ns": 400 * 10**9,
            "layer": "Layer1Raw",
            "format": "Text",
            "shape_name": "test.thing",
            "parents": [],
            "subjects": [],
            "seq": 0,
        }
        await c.ingest_acked(continuum_py.encode_header(header), inline=b"{}")
        return pid


async def _purge(sock: Path, ids: list[str]) -> None:
    async with await continuum_py.Client.connect(str(sock), client_name="forgetter") as c:
        await c.purge(ids=ids, reason="test")


async def _purge_checkpoints(sock: Path) -> None:
    async with await continuum_py.Client.connect(str(sock), client_name="checkpoints") as c:
        name = "worker.checkpoint.abc.parts"
        await c.register_stream(name, kind="derived", format="Text", shape_name="worker.checkpoint", slab_class=0)
        await c.purge(start_ns=0, end_ns=10**18, streams=[name], reason="expired worker checkpoints")


async def _age_out(sock: Path, ids: list[str]) -> None:
    async with await continuum_py.Client.connect(str(sock), client_name="ager") as c:
        await c.age_out(ids, 2**63 - 1, reason="test age-out")


def test_a_purge_only_of_worker_checkpoints_is_not_news_to_a_reader(broker: Path, store: Path) -> None:
    payload = asyncio.run(_commit_other(broker))
    log = speech_log.SpeechLog(broker, segments=store)
    try:
        before = log.newest_purge()
        asyncio.run(_purge_checkpoints(broker))
        assert log.newest_purge() == before
        assert log.newest_purge(before) == before
        asyncio.run(_purge(broker, [payload]))
        taken = log.newest_purge(before)
        assert taken not in (None, before)
        assert log.mark()[1] == taken
        asyncio.run(_purge_checkpoints(broker))
        assert log.newest_purge() == taken  # a later checkpoint trim doesn't hide the purge that took data
    finally:
        log.close()


async def _age_out_video(sock: Path) -> None:
    async with await continuum_py.Client.connect(str(sock), client_name="video-window") as c:
        await c.register_stream(
            "cam0.video", kind="sensor", format="EncodedVideo", shape_name="H264AnnexB", slab_class=0
        )
        await c.purge(start_ns=0, end_ns=10**18, streams=["cam0.video"], reason="the 30-day video window")


def test_a_purge_only_of_camera_video_is_news_to_readers_but_not_to_voice_ones(broker: Path, store: Path) -> None:
    """Voice groups keep their cached answer across a camera-video age-out."""
    payload = asyncio.run(_commit_other(broker))
    log = speech_log.SpeechLog(broker, segments=store)
    try:
        before = log.newest_voice_purge()
        asyncio.run(_age_out_video(broker))
        assert log.newest_purge() not in (None, before), "a video purge is still news to a reader of video"
        assert log.newest_voice_purge() == before
        asyncio.run(_purge(broker, [payload]))
        taken = log.newest_voice_purge()
        assert taken not in (None, before)
        assert log.newest_purge() == taken
    finally:
        log.close()


def test_an_age_out_is_a_purge_but_not_a_removal(broker: Path, store: Path) -> None:
    payload = asyncio.run(_commit_other(broker))
    log = speech_log.SpeechLog(broker, segments=store)
    try:
        assert log.newest_removal() is None
        asyncio.run(_age_out(broker, [payload]))
        aged = log.newest_purge()
        assert aged is not None, "an age-out is news to a reader that keeps lines"
        assert log.newest_removal() is None, "but not a deliberate removal"
        asyncio.run(_purge(broker, [payload]))
        removal = log.newest_removal()
        assert removal not in (None, aged)
        assert log.newest_purge() == removal
    finally:
        log.close()


def test_the_log_sets_the_voice_turn(broker: Path, store: Path) -> None:
    log = speech_log.SpeechLog(broker, segments=store)

    async def hears_the_turn() -> str:
        async with await continuum_py.Client.connect(str(broker)) as other:
            log.set_voice_turn("Speaking")
            event = await asyncio.wait_for(other.next_event(), 5)
            assert event is not None
            return continuum_py.decode_event(event[1]).get("voice_turn", "")

    try:
        assert asyncio.run(hears_the_turn()) == "Speaking"
    finally:
        log.close()
