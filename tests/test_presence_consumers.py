"""One synthetic detector day reaches every count consumer, including the voice handler."""

import re
import time
from unittest.mock import ANY, create_autospec
from zoneinfo import ZoneInfo

import pytest

from liberational import config as cfg
from liberational import memory_log, segment_log
from liberational.assistant import voice
from liberational.assistant.voice_auth import Conversation
from liberational.dashboard import presence as chips
from liberational.memory import ask, ask_history, history_qa, presence_counts
from presence_fake import add_frames
from voices import write_voices

DAY = "2026-10-20"


@pytest.fixture
def detector_day(tmp_path, monkeypatch, request, vision):
    # Review's day_bounds uses the process timezone; history uses its configured
    # zone. Pin both to the same real calendar day without replacing either parser.
    request.addfinalizer(time.tzset)  # Runs after the context restores TZ, even if setup fails.
    with monkeypatch.context() as zone:
        zone.setenv("TZ", "UTC")
        time.tzset()
        zone.setattr(history_qa, "TZ", ZoneInfo("UTC"))
        start, end = chips.day_bounds(DAY)
        rows = [
            ("cam0", 9, False),
            ("cam0", 10, True),
            ("cam0", 11, False),
            ("cam0", 19, False),
            ("cam0", 20, True),
            ("cam0", 21, False),
            ("cam1", 29, False),
            ("cam1", 30, True),
            ("cam1", 31, False),
        ]
        write_voices(tmp_path, {1: "Alice"})
        zone.setattr(voice, "STATE_DIR", tmp_path)
        zone.setattr(voice, "CONVERSATION", Conversation(tmp_path / "unused-password-verifier"))
        zone.setattr(voice, "optional_module", create_autospec(voice.optional_module, return_value=None))
        zone.setattr(
            voice, "owner_answer", create_autospec(voice.owner_answer, side_effect=AssertionError("no agent count"))
        )
        zone.setattr(
            ask, "local_answer", create_autospec(ask.local_answer, side_effect=AssertionError("no model count"))
        )
        with cfg.override({"owner": {"name": "Alice"}, "presence": {"enabled": True, "gap_s": 1.0, "absence_s": 1.0}}):
            add_frames(vision, [(start + offset, camera, present) for camera, offset, present in rows])
            yield tmp_path, start, end


def test_same_day_counts_reach_chips_and_ask_and_voice_agree(detector_day, monkeypatch):
    state, start, end = detector_day
    question = f"How many times did someone walk by on {DAY}?"
    now = end + 12 * 3600
    historical = history_qa.collect(question, state, now=now)
    asked = ask.execute(question, state_dir=state, now=now)
    summary = presence_counts.collect(state, start=start, end=end)
    reviewed = chips.presence(DAY, state)
    assert (historical["query"]["start"], historical["query"]["end"]) == (start, end)
    assert (reviewed["start"], reviewed["end"]) == (start, end)
    # Ask answers walk-bys from see-actions' walk-by spans and movement, never the sampled visits.
    assert "presence" not in historical and "presence" not in asked
    assert {camera: row["count"] for camera, row in summary["cameras"].items()} == {"cam0": 2, "cam1": 1}
    assert {camera: row["unknown_seconds"] for camera, row in summary["cameras"].items()} == {
        "cam0": 86395,  # watched only around its frames: 9 s to 11.5 s and 19 s to 21.5 s
        "cam1": 86397.5,
    }
    assert asked["answer"] == historical["answer"]
    assert historical["answer"].startswith("No camera was watching then"), "no walk-by evidence is never a zero"
    projected_chips = {chip["id"]: chip for chip in reviewed["chips"] if chip["id"].startswith("presence:")}
    assert {key: chip["count"] for key, chip in projected_chips.items()} == {"presence:cam0": 2, "presence:cam1": 1}
    assert [span[0] for span in projected_chips["presence:cam0"]["spans"]] == [start + 10, start + 20]
    assert [span[0] for span in projected_chips["presence:cam1"]["spans"]] == [start + 30]
    assert all("Unknown coverage: 86395.0 s." in note for note in projected_chips["presence:cam0"]["notes"])
    assert all("Unknown coverage: 86397.5 s." in note for note in projected_chips["presence:cam1"]["notes"])

    voice_results = []

    def execute_at_fixed_time(request, mode="local", **kwargs):
        result = ask.execute(request, mode=mode, state_dir=state, now=now, **kwargs)
        voice_results.append(result)
        return result

    backend = create_autospec(ask.ask, side_effect=execute_at_fixed_time)
    monkeypatch.setattr(ask, "ask", backend)
    speaker = create_autospec(voice.Speaker, instance=True)
    said = voice.handle(question, speaker, {"speaker_id": 1, "speaker_sim": 0.9})
    backend.assert_called_once_with(question, mode="local", on_short=ANY)
    assert callable(backend.call_args.kwargs["on_short"])
    assert said == ask.spoken(asked)
    speaker.speak.assert_called_once_with(said)


def test_switch_off_removes_the_review_outputs(detector_day):
    state, _, _ = detector_day
    assert any(chip["id"].startswith("presence:") for chip in chips.presence(DAY, state)["chips"])
    with cfg.override({"presence": {"enabled": False}}):
        assert not any(chip["id"].startswith("presence:") for chip in chips.presence(DAY, state)["chips"])


def walk_by(vision, at, *, cam="cam0", direction="left_to_right", followed=True, line=None):
    """A "Walked past" span as the walk-by worker writes one to the segment store."""
    attributes = {"direction": direction, "line": line or {"kind": "vertical", "position": 0.5}, "crossings": 1}
    return vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=at,
        ts_end=at,
        cam=cam,
        kind="crossing",
        track_id="walker",
        attributes={**attributes, "tracks": 1, "followed": followed},
    )


def test_this_evening_counts_only_walk_bys_inside_the_parsed_window(detector_day, vision):
    state, day_start, day_end = detector_day
    first_hour, last_hour = history_qa.PARTS_OF_DAY["evening"]
    start, end = day_start + first_hour * 3600, day_start + last_hour * 3600
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=start - 60, ts_end=end + 60, cam="cam0", frames=100)
    for at in (start - 1, start + 10, end - 2, end):  # the first is before the evening, the last at its open end
        walk_by(vision, at)
    result = ask.execute("How many people walked by this evening?", state_dir=state, now=day_end - 1)
    counted = result["coverage"]["walked_past"]
    assert counted == {"partial": False}
    assert result["answer"].startswith("People walked past the camera twice.")
    assert result["paths"] == ["presence"]


@pytest.mark.parametrize(
    "question",
    [
        "How many times did someone walk by before 3pm?",
        "How many times did someone walk by two days ago at 3pm?",
        "Who was here two days ago at 3pm?",
    ],
)
def test_unsupported_clock_count_or_relative_day_refuses_without_guessing(detector_day, question):
    state, _, end = detector_day
    result = ask.execute(question, state_dir=state, now=end - 1)
    assert "Unsupported time phrase" in result["answer"]
    assert "presence" not in result
    assert not result["evidence"]


def test_supported_identity_clock_query_retains_its_instant_scope(detector_day):
    state, start, end = detector_day
    result = history_qa.collect("Who was here at 3pm?", state, now=end - 1)
    assert result["query"]["at"] == start + 15 * 3600
    assert result["query"]["kind"] == "who"
    assert "presence" not in result


def _unreachable(stream, lo, hi):
    raise ConnectionRefusedError("private broker path")


class Unreachable:
    """A segment store whose presence streams can't be read; faces and everything else read as before."""

    def __init__(self, vision):
        self.vision = vision

    def range(self, stream, lo, hi, cams=None):
        if stream in (segment_log.PRESENCE_SPANS, segment_log.ACTION_COVERAGE):
            return _unreachable(stream, lo, hi)
        return self.vision.range(stream, lo, hi, cams)


def test_an_unreadable_vision_log_is_explicit_in_history_and_review(detector_day, monkeypatch, vision):
    state, _start, end = detector_day
    monkeypatch.setattr(segment_log, "at", lambda state=None: Unreachable(vision))
    reviewed = chips.presence(DAY, state)
    monkeypatch.setattr(vision, "range", _unreachable)
    historical = history_qa.collect(f"How many times did someone walk by on {DAY}?", state, now=end + 3600)
    assert reviewed["error"] == "Unavailable"
    assert reviewed["partial"]
    assert reviewed["pending"] is True
    assert historical["coverage"]["read_error"]
    assert historical["answer"] == "History source could not be read safely."
    assert "0 times" not in historical["answer"]
    assert "private broker" not in historical["answer"]
    assert "private broker" not in str(reviewed)


def test_review_unavailable_response_is_not_cached_when_the_log_recovers(detector_day, monkeypatch, vision):
    state, _start, _end = detector_day
    monkeypatch.setattr(segment_log, "at", lambda state=None: Unreachable(vision))
    assert chips.presence(DAY, state)["error"] == "Unavailable"
    monkeypatch.setattr(segment_log, "at", lambda state=None: vision)
    result = chips.presence(DAY, state)
    assert "error" not in result
    assert {chip["id"] for chip in result["chips"]} >= {"presence:cam0", "presence:cam1"}


def add_movement(vision, start, *, track="track-a", at=10, exit_at=12):
    coverage = vision.add(
        segment_log.MOVEMENT_COVERAGE,
        ts_start=start + 9,
        ts_end=start + 14,
        cam="cam0",
        version="movement-1",
        frames=11,
        supported=[[start + 9, start + 14]],
        gaps=[],
        tracking="complete",
        camera_motion="stable",
        finished=True,
        closure="fragment",
    )
    source = vision.add(
        segment_log.MOVEMENT_TRACKS,
        ts_start=start + 9,
        ts_end=start + 14,
        cam="cam0",
        track_id=track,
        version="movement-1",
        points=[[start + at, 0.05, 0.8], [start + exit_at, 0.95, 0.8]],
        supported=[[start + at, start + exit_at]],
        gaps=[],
        direction="right",
        camera_motion="stable",
        finished=True,
        closure="absence",
        entry={"status": "supported", "edge": "left", "time": [start + at, start + at]},
        exit={"status": "supported", "edge": "right", "time": [start + exit_at, start + exit_at]},
    )
    return source, coverage


def test_movement_is_separate_from_shared_sampled_counts_and_keeps_ask_evidence(detector_day, vision, monkeypatch):
    state, start, end = detector_day
    source, _ = add_movement(vision, start)
    question = f"How many times did someone walk by on {DAY}?"
    result = history_qa.collect(question, state, now=end + 3600)
    asked = ask.execute(question, state_dir=state, now=end + 3600)
    assert asked["answer"] == result["answer"]
    monkeypatch.setattr(ask, "ask", create_autospec(ask.ask, return_value=asked))
    speaker = create_autospec(voice.Speaker, instance=True)
    spoken = voice.handle(question, speaker, {"speaker_id": 1, "speaker_sim": 0.9})
    assert spoken is not None
    assert "came into the picture at least once" in spoken
    speaker.speak.assert_called_once_with(spoken)
    summary = presence_counts.collect(state, start=start, end=end)
    reviewed = chips.presence(DAY, state)
    assert summary["cameras"]["cam0"]["count"] == 2
    assert result["movement"] == summary["movement"] == reviewed["movement"]
    assert result["movement"]["cameras"]["cam0"]["entry_count"] == 1
    assert result["movement"]["cameras"]["cam0"]["exit_count"] == 1
    assert "in view" not in result["answer"], "the sampled visits are not Ask's count (#1591)"
    assert "came into the picture at least once" in result["answer"]
    assert "image" not in result["answer"] and "supported" not in result["answer"], "no tracker detail"
    tracked = [row for row in result["evidence"] if row.get("movement_evidence")]
    assert {row["movement_evidence"].get("kind") for row in tracked} >= {"entry", "exit"}
    assert any(source["id"] in row["movement_evidence"]["source_ids"] for row in tracked)
    by_id = {chip["id"]: chip for chip in reviewed["chips"]}
    assert by_id["presence:cam0"]["count"] == 2
    assert by_id["presence:cam0"]["camera"] == "cam0"
    visible = by_id["movement:visible:cam0"]
    assert visible["label"] == "Someone in view on cam0"
    assert visible["camera"] == "cam0"
    assert (visible["count"], visible["total_s"]) == (1, 2)
    assert visible["source_ids"] == [source["id"]]
    assert {row["kind"] for row in visible["evidence"]} == {"entry", "exit"}
    assert [chip["id"] for chip in reviewed["chips"] if chip["id"].startswith("movement:")] == ["movement:visible:cam0"]


def test_ended_day_movement_refreshes_on_late_commit_and_purge_without_changing_episodes(detector_day, vision):
    state, start, end = detector_day
    before = presence_counts.collect(state, start=start, end=end)
    first_review = chips.presence(DAY, state)
    assert not before["movement"]["available"]
    source, _ = add_movement(vision, start)
    after = presence_counts.collect(state, start=start, end=end)
    second_review = chips.presence(DAY, state)
    assert after["cameras"] == before["cameras"]
    assert after["movement"]["cameras"]["cam0"]["entry_count"] == 1
    assert any(chip["id"] == "movement:visible:cam0" for chip in second_review["chips"])
    assert not any(chip["id"].startswith("movement:") for chip in first_review["chips"])
    vision.records[segment_log.MOVEMENT_TRACKS].remove(source)
    vision.visibility_revisions = {segment_log.MOVEMENT_TRACKS: 1}
    cleared = chips.presence(DAY, state)
    assert not any(chip["id"].startswith("movement:") for chip in cleared["chips"])
    assert cleared["movement"]["cameras"]["cam0"]["entry_count"] == 0


def test_entries_close_together_are_never_told_as_a_group_or_as_people(detector_day, vision):
    state, start, end = detector_day
    add_movement(vision, start)
    add_movement(vision, start, track="track-b", at=10.5, exit_at=12.5)
    shared = presence_counts.collect(state, start=start, end=end)
    text = presence_counts.answer_text(shared)
    assert "came into the picture at least twice" in text
    assert "group" not in text and "3 people" not in text


def test_review_joins_every_track_on_a_camera_into_one_plain_chip(detector_day, vision):
    state, start, _ = detector_day
    first, _ = add_movement(vision, start)
    second, _ = add_movement(vision, start, track="track-b", at=10.5, exit_at=12.5)
    reviewed = chips.presence(DAY, state)
    tracked = reviewed["movement"]["cameras"]["cam0"]
    assert len(tracked["tracks"]) == 2 and tracked["coarrivals"], "two tracks and a nearby arrival underneath"
    movement = [chip for chip in reviewed["chips"] if chip["id"].startswith("movement:")]
    assert [chip["id"] for chip in movement] == ["movement:visible:cam0"]
    (visible,) = movement
    # 10-12 and 10.5-12.5 overlap: one stretch of 2.5 s, not two tracks or 4 s.
    assert visible["spans"] == [[start + 10, start + 12.5]]
    assert (visible["count"], visible["total_s"]) == (1, 2.5)
    assert visible["source_ids"] == sorted([first["id"], second["id"]])
    assert len(visible["evidence"]) == 4, "both tracks' entries and exits"
    assert "track" not in visible["notes"][0] and "supported" not in visible["notes"][0]


def test_review_movement_chip_counts_separate_stretches_and_skips_unavailable_cameras():
    def track(source, *intervals):
        return {
            "track_id": source,
            "source_ids": [source],
            "supported_intervals": [list(span) for span in intervals],
            "supported_s": sum(hi - lo for lo, hi in intervals),
        }

    result = {
        "available": True,
        "cameras": {
            "cam1": {
                "available": True,
                "partial": True,
                "entries": [],
                "exits": [],
                "tracks": [track("t3", (50, 52)), track("t1", (10, 14)), track("t2", (12, 20)), track("t0")],
            },
            "cam0": {"available": False, "partial": True, "tracks": [track("t9", (1, 2))]},
            "cam2": {"available": True, "partial": False, "entries": [], "exits": [], "tracks": [track("t4")]},
        },
    }
    (visible,) = chips.tracked_movement(result, end=51)
    assert visible["id"] == "movement:visible:cam1"
    assert visible["spans"] == [[10, 20], [50, 51]], "overlaps joined, clipped to the day's end"
    assert (visible["count"], visible["total_s"]) == (2, 11)
    assert visible["source_ids"] == ["t1", "t2", "t3"], "a track never in view adds no source"
    assert "may be more" in visible["notes"][0]


def test_unavailable_movement_never_changes_sampled_fallback_into_physical_zero(detector_day, vision, monkeypatch):
    state, start, end = detector_day
    add_movement(vision, start)
    original = vision.range

    def unreadable_movement(stream, *args, **kwargs):
        if stream.startswith("movement."):
            raise OSError("the segment store is unreadable")
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(vision, "range", unreadable_movement)
    shared = presence_counts.collect(state, start=start, end=end)
    assert shared["cameras"]["cam0"]["count"] == 2
    assert not shared["movement"]["available"]
    text = presence_counts.answer_text(shared)
    assert "Someone was in view of cam0 twice" in text
    assert "came into the picture" not in text, "unavailable movement says nothing, never a zero"
    reviewed = chips.presence(DAY, state)
    assert not any(chip["id"].startswith("movement:") for chip in reviewed["chips"])
    assert any(chip["id"] == "presence:cam0" for chip in reviewed["chips"])


def test_movement_citation_keeps_canonical_source_identity_through_history_reload_and_purge(
    detector_day, vision, memory
):
    state, start, end = detector_day
    source, _ = add_movement(vision, start)
    # The fake catalog is shared across streams; this stands in for the derived payload's canonical ID.
    memory.append(
        memory_log.JOURNAL,
        {"ts_start": start + 10, "text": "Synthetic movement source", "level": 30},
        payload_id=source["id"],
    )
    collected = history_qa.collect(f"How many times did someone walk by on {DAY}?", state, now=end + 3600)
    citation = next(row for row in collected["evidence"] if row.get("movement_evidence", {}).get("kind") == "entry")
    assert citation.get("payload_id") == source["id"]
    assert citation["id"].startswith("C") and citation["id"] != source["id"]
    history = ask_history.History(state)
    exchange = history.start("What movement was supported?", scope="history", created_at=end + 3600)
    result = {"answer": "Supported entry [" + citation["id"] + "].", "evidence": [citation]}
    assert history.finish(exchange, result, started_at=end + 3601, finished_at=end + 3602, status="completed")
    saved = ask_history.History(state).get(exchange)
    assert saved is not None and saved["result"] is not None
    assert saved["citations"][0]["verified"]
    memory.purge([source["id"]], "synthetic source removal")
    removed = ask_history.History(state).get(exchange)
    assert removed is not None and removed["result"] is None and removed["answer_removed"]
    assert removed["citations"][0]["removed"]


def test_grouped_movement_citations_and_zero_coverage_keep_all_canonical_ids(detector_day, vision):
    state, start, end = detector_day
    first, coverage_a = add_movement(vision, start)
    second, coverage_b = add_movement(vision, start, track="track-b", at=10.5, exit_at=12.5)
    question = f"How many times did someone walk by on {DAY}?"
    result = history_qa.collect(question, state, now=end + 3600)
    pair = next(row for row in result["evidence"] if row.get("method") == "camera-local coincident entry times")
    assert pair.get("payload_ids") == sorted([first["id"], second["id"]])
    visible = [row for row in result["evidence"] if row.get("method") == "source-backed tracked visibility intervals"]
    assert {row.get("payload_id") for row in visible} == {first["id"], second["id"]}
    vision.records[segment_log.MOVEMENT_TRACKS].clear()
    vision.visibility_revisions = {segment_log.MOVEMENT_TRACKS: 1}
    cleared = history_qa.collect(question, state, now=end + 3600)
    coverage = next(row for row in cleared["evidence"] if row.get("method") == "movement coverage")
    assert coverage.get("payload_ids") == sorted([coverage_a["id"], coverage_b["id"]])
    assert coverage["id"].startswith("C")


def test_a_walk_by_question_says_how_many_times_people_walked_past_in_plain_words(detector_day, vision):
    state, start, end = detector_day
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=start, ts_end=start + 3600, cam="cam0", frames=7200)
    walk_by(vision, start + 22)
    walk_by(vision, start + 600, direction="right_to_left", followed=False)
    result = history_qa.collect(f"How many times did someone walk by on {DAY}?", state, now=end + 3600)
    assert result["answer"].startswith("People walked past the camera twice.")
    assert "only watched for 60 minutes of the day" in result["answer"]
    walked = result["walked_past"]["cameras"]["cam0"]
    assert walked["crossings"] == {"left_to_right": 1, "right_to_left": 1, "total": 2}
    assert (walked["unverified"], walked["scanned_seconds"]) == (1, 3600)
    cited = [row for row in result["evidence"] if row.get("stream") == "walkby.span"]
    assert [row["ts"] for row in cited] == [start + 22, start + 600]


@pytest.mark.parametrize("mode", ["local", "friends", "everyone"])
def test_direction_requests_answer_and_cite_directions_in_ask_and_voice(detector_day, vision, monkeypatch, mode):
    state, start, end = detector_day
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=start, ts_end=end, cam="cam0", frames=100)
    first = walk_by(vision, start + 10, direction="left_to_right")
    second = walk_by(vision, start + 20, direction="right_to_left")
    question = f"Which way did someone walk past cam0 on {DAY}?"
    result = ask.execute(question, mode=mode, state_dir=state, now=end + 60)
    assert result["mode"] == mode
    assert result["model"] == "local-history-evidence"
    assert "left to right" in result["answer"]
    assert "right to left" in result["answer"]
    assert "image" in result["answer"]
    assert "[C1]" in result["answer"]
    assert {e.get("payload_id") for e in result["evidence"]} == {first["id"], second["id"]}
    assert (result["query"]["start"], result["query"]["end"]) == (start, end)
    backend = create_autospec(ask.ask, return_value=result)
    monkeypatch.setattr(ask, "ask", backend)
    speaker = create_autospec(voice.Speaker, instance=True)
    said = voice.handle(question, speaker, {"speaker_id": 1, "speaker_sim": 0.9})
    backend.assert_called_once_with(question, mode="local", on_short=ANY)
    assert callable(backend.call_args.kwargs["on_short"])
    assert said == ask.spoken(result)
    speaker.speak.assert_called_once_with(said)


def test_a_directional_passage_count_still_reports_the_requested_quantity(detector_day, vision):
    state, start, end = detector_day
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=start, ts_end=end, cam="cam0", frames=100)
    walk_by(vision, start + 10, direction="left_to_right")
    walk_by(vision, start + 20, direction="left_to_right")
    walk_by(vision, start + 30, direction="left_to_right")
    walk_by(vision, start + 40, direction="right_to_left")
    result = ask.execute(f"How many people walked left to right on cam0 on {DAY}?", state_dir=state, now=end + 60)
    without_citations = re.sub(r"\[[A-Z]\d+\]", "", result["answer"])
    assert re.search(r"\b3\b", without_citations)
    assert "left to right" in result["answer"]
    assert result["model"] == "local-history-evidence"


def test_configured_segment_directions_retain_native_witnesses(detector_day, vision):
    state, start, end = detector_day
    segment = {"kind": "segment", "p1": [0.1, 0.2], "p2": [0.8, 0.7]}
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=start, ts_end=end, cam="cam0", frames=100)
    first = walk_by(vision, start + 10, direction="side_a_to_side_b", line=segment)
    second = walk_by(vision, start + 20, direction="side_b_to_side_a", line=segment)
    with cfg.override({"crossings": {"lines": {"cam0": segment}}}):
        result = ask.execute(f"Which way did someone cross cam0 on {DAY}?", state_dir=state, now=end + 60)
    assert "side a" in result["answer"].lower()
    assert "side b" in result["answer"].lower()
    assert {e.get("payload_id") for e in result["evidence"]} == {first["id"], second["id"]}
    assert "[C1]" in result["answer"]


def test_segment_qualified_count_does_not_substitute_the_camera_total(detector_day, vision):
    state, start, end = detector_day
    segment = {"kind": "segment", "p1": [0.1, 0.2], "p2": [0.8, 0.7]}
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=start, ts_end=end, cam="cam0", frames=100)
    walk_by(vision, start + 10, direction="side_a_to_side_b", line=segment)
    walk_by(vision, start + 20, direction="side_a_to_side_b", line=segment)
    walk_by(vision, start + 30, direction="side_a_to_side_b", line=segment)
    walk_by(vision, start + 40, direction="side_b_to_side_a", line=segment)
    with cfg.override({"crossings": {"lines": {"cam0": segment}}}):
        result = ask.execute(
            f"How many people crossed side A to side B on cam0 on {DAY}?", state_dir=state, now=end + 60
        )
    without_citations = re.sub(r"\[[A-Z]\d+\]", "", result["answer"])
    assert re.search(r"\b3\b", without_citations)
    assert "side A to line side B" in result["answer"]
