"""Recorded-only tests: generated video and stub VLM, no real devices/services."""

import datetime as dt
import urllib.error
import urllib.request
from typing import Self
from unittest.mock import patch

import pytest

import speech_fake
from liberational import config as cfg
from liberational.memory import history_qa as h
from liberational.memory import video_evidence as v
from liberational.memory.video_evidence import measured_answer, measurements

NOW = dt.datetime(2026, 10, 26, 12, tzinfo=h.TZ).timestamp()


@pytest.fixture(autouse=True)
def explicit_recorded_analysis(monkeypatch):
    """Recorded-frame analysis is off by default; these tests exercise it."""
    monkeypatch.setenv("PACK_REVIEW_RECORDED_FRAME_ANALYSIS", "true")


def test_review_day_is_the_window_without_one_in_the_question(tmp_path):
    assert h.collect("what did we want done", tmp_path, NOW, day="2026-10-20")["query"]["window"] == "2026-10-20"


@pytest.mark.parametrize("phrase", ["today", "yesterday", "this week", "last week", "2026-10-21"])
def test_explicit_window_beats_the_review_day(phrase):
    assert h.parse("video " + phrase, NOW, "2026-10-20")["window"] == phrase


def test_last_week_is_monday_to_monday():
    q = h.parse("video last week", NOW)
    assert dt.datetime.fromtimestamp(q["start"], h.TZ).weekday() == 0
    assert q["end"] - q["start"] == 7 * 86400


def test_planning_participants_context_and_exclusions(tmp_path):
    rows = [
        dict(ts_start=NOW - 100, speaker="Dana", text="I want to fix the garden."),
        dict(ts_start=NOW - 90, speaker="Alice", text="The fence too."),
        dict(ts_start=NOW - 80, speaker="Dana", text="password rosebud"),
        dict(ts_start=NOW - 70, speaker="Alice", text="We should sing.", synthetic=True),
    ]
    speech_fake.fake().write(rows)
    with cfg.override({"owner": {"name": "Alice"}}):
        result = h.collect("what did Dana and I want to get done this week", tmp_path, NOW)
    assert result["query"]["participants"] == ["Dana", "Alice"]
    assert [e["speaker"] for e in result["evidence"]] == ["Dana", "Alice", "Dana"]  # the synthetic line is left out
    assert result["query"]["person"] is None


def test_duration_not_transition_route(tmp_path):
    with patch.object(v, "collect", autospec=True, side_effect=AssertionError("no tiny frame-window duration")):
        result = h.collect("how long were the lights on today", tmp_path, NOW)
    assert result["query"]["kind"] == "light_duration"
    assert result["coverage"]["lights"] is None  # no colour-cast measure for this window: unknown, never a total
    assert result["answer"] == h.LIGHTS_NO_RECORDINGS  # today is a whole-day window with nothing recorded


def test_recorded_frames_are_inspected_cached_and_unknowns_kept(tmp_path, vision):
    import numpy as np

    vision.video.append(("cam7", 100.0, 106.0))
    for k in range(13):  # a frame every half second
        vision.frames[("cam7", 100.0 + k / 2)] = np.full((64, 64, 3), 10 * k, np.uint8)
    query = dict(start=99, end=108, cameras=["cam7"])

    def caption(image, timeout, question):
        assert image[:2] == b"\xff\xd8"
        return dict(people=1, lighting="unknown", description="A person near a desk.")

    with (
        cfg.override({"review": {"video_max_frames": 6}}),
        patch.object(v, "_caption", autospec=True, side_effect=caption) as model,
    ):
        evidence, coverage, warnings = v.collect(tmp_path, query)
        assert len(evidence) == 3
        assert model.call_count == 3
        assert evidence[0]["link"].startswith("/review#day=")
        assert coverage["cameras"]["cam7"]["exact_crossings"] is None
        assert coverage["cameras"]["cam7"]["unknown_intervals"]
        v.collect(tmp_path, query)
        assert model.call_count == 3, "cached"
        assert "electrical" in warnings[0]


def test_a_camera_without_video_is_never_decoded(tmp_path, vision):
    with patch.object(v, "_frame", autospec=True) as decode:
        evidence, coverage, warnings = v.collect(tmp_path, dict(start=0, end=5, cameras=["cam3"]))
    decode.assert_not_called()
    assert not evidence
    assert coverage["cameras"]["cam3"]["unknown_intervals"] == [[0, 5]]
    assert any("No recorded video" in w for w in warnings)


def test_adaptive_sampling_bounded_and_cache_invalidated(tmp_path, vision):
    vision.video.append(("cam2", 100.0, 120.0))
    query = dict(start=100, end=120, cameras=[])
    with (
        cfg.override({"review": {"video_max_frames": 8}, "vision": {"backend": "ollama"}}),
        patch.object(v, "_frame", autospec=True, return_value=b"jpeg"),
        patch.object(v, "_caption", autospec=True, side_effect=["A person", "Empty room"] * 20) as model,
    ):
        evidence, coverage, _ = v.collect(tmp_path, query)
        assert 4 < len(evidence) <= 8
        assert coverage["attempted_frames"] <= 8
        count = model.call_count
        with cfg.override({"models": {"ollama": {"vlm": "another-model"}}}):  # another model: asked again
            v.collect(tmp_path, query)
        assert model.call_count > count


def test_model_failure_reports_unknown(tmp_path, vision):
    vision.video.append(("cam2", 100.0, 120.0))
    with (
        patch.object(v, "_frame", autospec=True, return_value=b"jpeg"),
        patch.object(v, "_caption", autospec=True, side_effect=OSError("offline")),
    ):
        evidence, coverage, warnings = v.collect(tmp_path, dict(start=100, end=120, cameras=["cam2"]))
    assert not evidence
    assert coverage["cameras"]["cam2"]["unknown_intervals"] == [[100, 120]]
    assert any("unavailable" in w for w in warnings)


def test_ollama_backend_inspects_recorded_frames_when_local_mlx_server_is_unavailable(tmp_path, vision):
    import contextlib
    import json
    from urllib.parse import urlparse

    vision.video.append(("cam2", 100.0, 120.0))

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def read(self, size: int = -1) -> bytes:
            return json.dumps(self._payload).encode()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeOpener:
        def __init__(self) -> None:
            self.ollama_calls = 0

        def open(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
            assert timeout > 0
            path = urlparse(request.full_url).path
            if path == "/inspect":
                raise OSError("refused")
            assert path == "/api/chat"
            self.ollama_calls += 1
            raw = request.data
            assert isinstance(raw, bytes)
            body = json.loads(raw.decode())
            assert body["messages"][0]["images"]
            assert body["format"] == "json"
            return FakeResponse({"message": {"content": '{"people": 1, "lighting": "on", "description": "A person."}'}})

    opener = FakeOpener()
    with (
        cfg.override({"vision": {"backend": "ollama"}, "review": {"video_max_frames": 1}}),
        patch.object(v, "_frame", autospec=True, return_value=b"jpeg"),
        patch("urllib.request.build_opener", autospec=True, return_value=opener),
        patch("liberational.ollama_health.call", autospec=True, side_effect=lambda _model: contextlib.nullcontext()),
    ):
        evidence, coverage, _warnings = v.collect(tmp_path, dict(start=100, end=120, cameras=["cam2"]))
    assert evidence and evidence[0]["quote"] == "A person."
    assert coverage["attempted_frames"] == 1
    assert opener.ollama_calls == 1


def test_an_ollama_that_isnt_running_leaves_the_frames_unknown(tmp_path, vision, monkeypatch):
    # With Ollama as the Linux backend, its refused connection comes back from ollama_health.call as Down (a
    # RuntimeError), which collect must report as unknown frames rather than raise into the question's answer.
    from liberational import ollama_health

    vision.video.append(("cam2", 100.0, 120.0))

    class RefusingOpener:
        def open(self, request: urllib.request.Request, timeout: float) -> None:
            raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr(ollama_health, "HEALTH", ollama_health.Health())
    monkeypatch.setattr(ollama_health, "gate", None)
    with (
        cfg.override({"vision": {"backend": "ollama"}}),
        patch.object(v, "_frame", autospec=True, return_value=b"jpeg"),
        patch("urllib.request.build_opener", autospec=True, return_value=RefusingOpener()),
    ):
        evidence, coverage, warnings = v.collect(tmp_path, dict(start=100, end=120, cameras=["cam2"]))
    assert not evidence
    assert coverage["cameras"]["cam2"]["unknown_intervals"] == [[100, 120]]
    assert any("unavailable" in w for w in warnings)


@pytest.mark.parametrize("question", ["what was the person doing?", "what did the scene look like?"])
def test_review_physical_synonyms(question, tmp_path):
    with patch.object(v, "collect", autospec=True, return_value=([], {}, [])) as inspect:
        result = h.collect(question, tmp_path, day="2026-10-20")
    assert result["query"]["kind"] == "video"
    inspect.assert_called_once()
    assert not h.collect(question, tmp_path)["is_history"]


def test_how_many_people_came_by_is_a_crossing_question_not_vlm_windows(tmp_path):
    # A count question is answered by caption episodes (path A) and the line crossings (path B), never by a few
    # VLM windows; tests/test_crossing_counts.py covers path B.
    question = "how many people came by?"
    with (
        cfg.override({"presence": {"enabled": False}}),
        patch.object(v, "collect", autospec=True, return_value=([], {}, [])) as inspect,
    ):
        result = h.collect(question, tmp_path, day="2026-10-20")
    assert result["query"]["kind"] == "video"
    inspect.assert_not_called()
    assert "crossings" in result
    assert not h.collect(question, tmp_path)["is_history"]


def test_what_happened_combines_raw_sources(tmp_path):
    speech_fake.fake().write([dict(ts_start=NOW - 10, speaker="Dana", text="The package arrived.")])
    frame = dict(ts=NOW - 10, camera="cam2", quote="A parcel on the table", source="camera", link="/review#t=1")
    with patch.object(v, "collect", autospec=True, return_value=([frame], {}, [])):
        result = h.collect("what happened?", tmp_path, NOW, day="2026-10-26")
    assert result["query"]["kind"] == "mixed"
    assert {r["source"] for r in result["evidence"]} == {"camera", "transcript"}


def test_joint_planning_prioritized_over_owner_solo(tmp_path):
    rows = [
        dict(ts_start=NOW - 1000, speaker="Alice", text="I want a solo project."),
        dict(ts_start=NOW - 100, speaker="Dana", text="We should repair the fence."),
        dict(ts_start=NOW - 90, speaker="Alice", text="Yes, the gate too."),
    ]
    speech_fake.fake().write(rows)
    with cfg.override({"owner": {"name": "Alice"}}):
        result = h.collect("what did Dana and I want done this week", tmp_path, NOW)
    assert len(result["evidence"]) == 2
    assert all("solo" not in r["quote"] for r in result["evidence"])


# measurements: episodes and light durations from per-frame observations


def row(ts, people, lighting="on"):
    return dict(ts=ts, camera="cam4", observation=dict(people=people, lighting=lighting))


def test_episode_not_frame_count_and_light_duration():
    evidence = [row(0, 0), row(2, 1), row(4, 1), row(6, 0, "off"), row(8, 0, "off")]
    result = measurements(evidence, dict(start=0, end=20, cameras=[]))["cam4"]
    assert result["observed_passage_candidate_count"] == 1
    assert result["visible_lighting_on_seconds_estimate"] == 4
    assert result["lighting_known_seconds"] == 6
    assert result["lighting_unknown_intervals"] == [[4, 6], [8, 20]]
    assert result["unknown_intervals"] == [[8, 20]]
    assert "1 complete" in measured_answer({"measurements": {"cam4": result}}, "video")


def test_static_person_and_gaps():
    query = dict(start=0, end=100, cameras=[])
    assert measurements([row(i, 1) for i in range(10)], query)["cam4"]["observed_passage_candidate_count"] == 0
    evidence = [row(0, 0), row(2, 1), row(50, 0)]  # a long gap after 2 s: nothing known across it
    result = measurements(evidence, query)["cam4"]
    assert result["observed_passage_candidate_count"] == 0
    assert result["visible_lighting_on_seconds_estimate"] == 2
    assert result["unknown_intervals"] == [[2, 100]]


def test_unknown_light_not_brightness_duration():
    result = measurements([row(0, None, "unknown"), row(2, None, "unknown")], dict(start=0, end=3, cameras=[]))["cam4"]
    assert result["lighting_known_seconds"] == 0
    assert result["lighting_unknown_intervals"] == [[0, 3]]
