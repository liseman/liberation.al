import cv2
import numpy as np
import pytest

from liberational import vision_log
from liberational.see.captions import CaptionPipeline
from liberational.vision import objects

RNG = np.random.default_rng(7)
# A textured room: coarse detail that survives the gate's 64x36 signature and motion's 320-px copy.
SCENE = cv2.resize(RNG.integers(0, 256, (90, 160, 3), dtype=np.uint8), (1280, 720), interpolation=cv2.INTER_CUBIC)
# Someone stands in a twentieth of the view: a local change, too small for the whole-view measures alone.
PERSON = SCENE.copy()
PERSON[300:520, 600:680] = 20


class VLM:
    name = "fake"

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls: list[list[str]] = []

    def caption(self, img: np.ndarray, memory) -> str:
        self.calls.append(list(memory))
        return self.answers.pop(0) if self.answers else "a room"


def feed(p: CaptionPipeline, frame: np.ndarray, ts: float, seq: int = 0):
    out = p.feed(frame, ts, seq)
    for stream, record in ((vision_log.CAPTIONS, out.caption), (vision_log.OBJECTS, out.objects)):
        if record is not None:
            vision_log.encode(stream, 1, record)  # every record fits its stream
    return out


def test_the_first_frame_is_captioned_fresh() -> None:
    vlm = VLM("an empty workshop")
    rec = feed(CaptionPipeline("cam0", vlm), SCENE, 100.0, 4).caption
    assert rec is not None
    assert (rec["caption"], rec["gated"], rec["frame_seq"]) == ("an empty workshop", False, 4)
    assert rec["observation"] | {} == {
        "version": 1,
        "status": "fresh",
        "observed_at": 100.0,
        "camera": "cam0",
        "backend": "fake",
        "prompt_version": "independent-v1",
    }
    assert rec["change"] is None, "nothing to compare the first tile with"
    assert rec["objects"] is None and rec["faces"] is None
    assert 0 < rec["brightness"]["mean"] < 255


def test_an_unchanged_view_repeats_its_last_caption_at_most_every_quiet_interval() -> None:
    vlm = VLM("an empty workshop")
    p = CaptionPipeline("cam0", vlm, quiet_every_s=10.0)
    feed(p, SCENE, 100.0)
    first = feed(p, SCENE, 101.0).caption
    assert first is not None
    assert (first["unchanged"], first["since"], first["caption"]) == (True, 100.0, "an empty workshop")
    assert first["observation"]["status"] == "not_observed" and first["gated"]
    assert first["change"]["mad"] == pytest.approx(0.0)
    assert feed(p, SCENE, 105.0).caption is None, "held back: within 10 s of the last quiet record"
    assert feed(p, SCENE, 111.5).caption is not None
    assert len(vlm.calls) == 1


def test_a_local_change_is_captioned_with_the_views_recent_captions() -> None:
    vlm = VLM("an empty workshop", "a person at the bench")
    p = CaptionPipeline("cam0", vlm)
    feed(p, SCENE, 100.0)
    rec = feed(p, PERSON, 101.0).caption
    assert rec is not None and rec["caption"] == "a person at the bench" and not rec["gated"]
    assert vlm.calls == [[], ["an empty workshop"]]


def test_a_steady_view_is_captioned_again_after_the_idle_limit() -> None:
    vlm = VLM()
    p = CaptionPipeline("cam0", vlm)
    feed(p, SCENE, 100.0)
    feed(p, SCENE, 129.0)
    feed(p, SCENE, 130.0)
    assert len(vlm.calls) == 2


def test_a_failed_vlm_call_is_an_error_and_the_next_frame_tries_again() -> None:
    vlm = VLM("", "a room")
    p = CaptionPipeline("cam0", vlm)
    rec = feed(p, SCENE, 100.0).caption
    assert rec is not None and rec["observation"]["status"] == "error" and "unchanged" not in rec
    assert feed(p, SCENE, 101.0).caption is not None
    assert len(vlm.calls) == 2, "an error leaves the gate open"


def test_grounding_follows_v1s_schedule_and_its_boxes_ride_on_the_captions() -> None:
    answers = ['[{"label": "lamp", "bbox_2d": [100, 200, 300, 600]}]', "boom", "[]"]
    asked: list[float] = []

    def ground(frame: np.ndarray) -> str:
        asked.append(frame.shape[1])
        answer = answers.pop(0)
        if answer == "boom":
            raise RuntimeError(answer)
        return answer

    p = CaptionPipeline("cam0", VLM(), ground=ground, quiet_every_s=0.0)
    out = feed(p, SCENE, 100.0, 1)
    assert out.objects is not None and asked == [1280], "the whole frame, not the tile"
    assert [(o["label"], o["bbox_norm"]) for o in out.objects["objects"]] == [("lamp", [0.1, 0.2, 0.3, 0.6])]
    caption = out.caption
    assert caption is not None
    assert [(o["label"], o["bbox"]) for o in caption["objects"]] == [("lamp", [45, 50, 134, 151])], "tile pixels"
    # A change sooner than objects.MIN_S after the last call isn't grounded, but the standing boxes go out again
    # with its caption, still marked with the time they were grounded.
    between = feed(p, PERSON, 100.0 + objects.MIN_S - 1)
    assert len(asked) == 1
    assert between.objects is not None and [(o["label"], o["ts"]) for o in between.objects["objects"]] == [
        ("lamp", 100.0)
    ]
    failed = feed(p, SCENE, 100.0 + objects.MIN_S + 1)
    assert len(asked) == 2, "asked after MIN_S, but the call failed"
    assert failed.objects is not None and [o["label"] for o in failed.objects["objects"]] == ["lamp"], "kept"
    # A steady view is grounded again after objects.MAX_IDLE_S.
    feed(p, SCENE, 100.0 + objects.MIN_S + 2)
    assert len(asked) == 2
    later = feed(p, SCENE, 100.0 + objects.MIN_S + 1 + objects.MAX_IDLE_S)
    assert len(asked) == 3 and later.objects is not None and later.objects["objects"] == []


def test_grounded_boxes_are_dropped_after_their_ttl() -> None:
    answers = ['[{"label": "lamp", "bbox_2d": [100, 200, 300, 600]}]']

    def ground(frame: np.ndarray) -> str:
        if not answers:
            raise RuntimeError("the grounder is down")
        return answers.pop(0)

    p = CaptionPipeline("cam0", VLM(), ground=ground)
    feed(p, SCENE, 100.0)
    # Every call since has failed, so the boxes from 100 s are the newest, and too old to show.
    rec = feed(p, SCENE, 101.0 + objects.TTL_S).caption
    assert rec is not None and rec["objects"] == []


def test_a_cut_off_grounding_is_a_failure_with_its_reason_and_is_logged_once(caplog) -> None:
    def ground(frame: np.ndarray) -> str:
        raise objects.GroundingTruncated("hit the limit")

    p = CaptionPipeline("cam0", VLM(), ground=ground, quiet_every_s=0.0)
    first = feed(p, SCENE, 100.0, 1)
    again = feed(p, SCENE, 100.0 + objects.MAX_IDLE_S + 1)
    for out in (first, again):
        assert out.objects is not None
        assert (out.objects["observation"]["status"], out.objects["observation"]["reason"]) == (
            "failed",
            "grounding_truncated",
        )
    assert sum("object grounding failed" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("answer", ["[]", '[{"label":"lamp","bbox_2d":[100,200,300,600]}]'])
def test_grounding_decision_records_successful_due_analysis_even_when_empty(answer: str) -> None:
    pipeline = CaptionPipeline("cam0", VLM(), ground=lambda _: answer)
    out = feed(pipeline, SCENE, 100.0)
    assert out.grounding_decision == "fresh"
    assert out.objects is not None and out.objects["observation"]["status"] == "fresh"
    assert "grounding_decision" not in out.objects
    assert out.caption is not None and "grounding_decision" not in out.caption


def test_grounding_decision_distinguishes_cached_and_suppressed_quiet_frames_from_due_analysis() -> None:
    asked: list[int] = []

    def ground(frame: np.ndarray) -> str:
        asked.append(frame.shape[1])
        return '[{"label":"lamp","bbox_2d":[100,200,300,600]}]'

    pipeline = CaptionPipeline("cam0", VLM(), ground=ground)
    feed(pipeline, SCENE, 100.0)
    cached = feed(pipeline, SCENE, 101.0)
    suppressed = feed(pipeline, SCENE, 102.0)
    assert cached.grounding_decision == suppressed.grounding_decision == "not_scheduled"
    assert cached.objects is not None and cached.objects["observation"]["status"] == "cached"
    assert suppressed.caption is None and suppressed.objects is None
    assert asked == [1280]


def test_grounding_decision_records_disabled_even_when_quiet_outputs_are_suppressed() -> None:
    pipeline = CaptionPipeline("cam0", VLM())
    initial = feed(pipeline, SCENE, 100.0)
    feed(pipeline, SCENE, 101.0)
    suppressed = feed(pipeline, SCENE, 102.0)
    assert initial.grounding_decision == suppressed.grounding_decision == "disabled"
    assert initial.objects is not None and initial.objects["observation"]["reason"] == "grounding_disabled"
    assert suppressed.caption is None and suppressed.objects is None


def test_due_grounding_failure_is_failed_even_when_old_boxes_are_retained() -> None:
    answers = iter(['[{"label":"lamp","bbox_2d":[100,200,300,600]}]', "not a grounding array"])
    pipeline = CaptionPipeline("cam0", VLM(), ground=lambda _: next(answers), quiet_every_s=0.0)
    feed(pipeline, SCENE, 100.0)
    failed = feed(pipeline, PERSON, 100.0 + objects.MIN_S + 1)
    assert failed.grounding_decision == "failed"
    assert failed.objects is not None and failed.objects["observation"]["status"] == "failed"
    assert [item["label"] for item in failed.objects["objects"]] == ["lamp"]


def test_a_change_within_the_caption_interval_waits_and_repeats_the_last_caption() -> None:
    vlm = VLM("an empty workshop", "a person at the bench")
    p = CaptionPipeline("patio", vlm, quiet_every_s=0.0, caption_every_s=60.0)
    feed(p, SCENE, 100.0)
    waiting = feed(p, PERSON, 130.0).caption
    assert waiting is not None
    assert (waiting["deferred"], waiting["unchanged"], waiting["since"]) == (True, True, 100.0)
    assert (waiting["caption"], waiting["gated"], waiting["observation"]["status"]) == (
        "an empty workshop",
        True,
        "not_observed",
    )
    assert not vision_log.is_real(waiting)
    fresh = feed(p, PERSON, 160.0).caption
    assert fresh is not None and fresh["caption"] == "a person at the bench" and "deferred" not in fresh
    assert len(vlm.calls) == 2


@pytest.mark.parametrize(
    "kind,own,expected",
    [
        pytest.param("rtsp", {}, 60.0, id="network-default"),
        pytest.param("camera", {}, 0.0, id="usb-no-wait"),
        pytest.param("camera", {"cam": 30}, 30.0, id="usb-override"),
        pytest.param("rtsp", {"cam": 0}, 0.0, id="network-override-to-no-wait"),
    ],
)
def test_a_cameras_caption_interval_comes_from_its_own_entry_then_its_kind(kind, own, expected) -> None:
    from liberational.see.caption_worker import caption_every_s
    from liberational.see.frames import Camera

    assert caption_every_s(Camera("cam", kind), 60.0, own) == expected


@pytest.mark.parametrize("bad", [-1.0, "60", True])
def test_a_caption_interval_that_isnt_a_nonnegative_number_is_refused(bad) -> None:
    from liberational.see.caption_worker import caption_every_s
    from liberational.see.frames import Camera

    with pytest.raises(ValueError, match="caption interval"):
        caption_every_s(Camera("cam", "rtsp"), 60.0, {"cam": bad})


def test_a_still_view_is_grounded_every_max_idle_s_and_a_moved_view_after_min_s() -> None:
    # The change gate re-captions a still view every 30 s (its idle refresh); that alone used to ground it every
    # ~35 s, most of the VLM's time (#1935). Only movement, or MAX_IDLE_S, asks for another grounding now.
    asked: list[float] = []

    def ground(frame: np.ndarray) -> str:
        asked.append(len(asked))
        return "[]"

    vlm = VLM()
    p = CaptionPipeline("cam0", vlm, ground=ground, quiet_every_s=0.0)
    feed(p, SCENE, 100.0)  # first look: captioned and grounded
    for ts in range(130, 400, 30):  # idle refreshes of the same view, each a fresh caption
        feed(p, SCENE, float(ts))
    assert len(vlm.calls) > 5, "the idle refresh keeps captioning the still view"
    assert len(asked) == 1, "but a still view isn't grounded again before MAX_IDLE_S"
    feed(p, SCENE, 100.0 + objects.MAX_IDLE_S)
    assert len(asked) == 2, "MAX_IDLE_S after the last grounding, it is"
    feed(p, PERSON, 100.0 + objects.MAX_IDLE_S + objects.MIN_S - 1)  # someone walks in, too soon after the last
    assert len(asked) == 2, "movement grounds at most every MIN_S"
    feed(p, SCENE, 100.0 + objects.MAX_IDLE_S + objects.MIN_S)  # and walks out: the view moved again
    assert len(asked) == 3, "a moved view is grounded once MIN_S has passed"
