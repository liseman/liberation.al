import numpy as np
import pytest

from liberational.see.captions import CaptionPipeline
from liberational.vision import objects
from test_see_captions import PERSON, SCENE, VLM, feed

LAMP = '[{"label":"lamp","bbox_2d":[100,200,300,600]}]'


class Grounder:
    def __init__(self, *answers: str | Exception) -> None:
        self.answers = list(answers)

    def __call__(self, frame: np.ndarray) -> str:
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_successful_empty_grounding_is_a_fresh_evaluated_absence():
    pipeline = CaptionPipeline("cam0", VLM(), ground=Grounder("[]"))

    out = feed(pipeline, SCENE, 100.0, 4)

    assert out.objects is not None
    assert out.objects["objects"] == []
    assert out.objects["ts_start"] == 100.0
    assert out.objects["frame_seq"] == 4
    observation = out.objects["observation"]
    assert observation["version"] == 1
    assert observation["status"] == "fresh"
    assert observation["observed_at"] == 100.0
    assert observation["attempted_at"] == 100.0


def test_cached_republication_keeps_grounding_time_separate_from_current_caption():
    pipeline = CaptionPipeline("cam0", VLM("a room", "a changed room"), ground=Grounder(LAMP))
    feed(pipeline, SCENE, 100.0)

    out = feed(pipeline, PERSON, 101.0, 5)

    assert out.objects is not None
    assert out.objects["ts_start"] == 101.0
    assert [(box["label"], box["ts"]) for box in out.objects["objects"]] == [("lamp", 100.0)]
    observation = out.objects["observation"]
    assert observation["status"] == "cached"
    assert observation["observed_at"] == 100.0
    assert observation["attempted_at"] is None
    assert out.caption is not None
    assert out.caption["ts_start"] == 101.0
    assert out.caption["observation"]["observed_at"] == 101.0
    assert [box["ts"] for box in out.caption["objects"]] == [100.0]


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(RuntimeError("grounder unavailable"), id="exception"),
        pytest.param("not JSON", id="malformed"),
        pytest.param(LAMP[:-1], id="truncated-array-with-complete-object"),
        pytest.param('{"objects":[]}', id="non-array-root"),
        pytest.param('[{"label":"chair","bbox_2d":[NaN,200,300,600]}]', id="nan"),
        pytest.param('[{"label":"chair","bbox_2d":[100,200,Infinity,600]}]', id="infinity"),
        pytest.param('[{"label":"chair","bbox_2d":[100,200,1e999,600]}]', id="overflow"),
    ],
)
def test_failed_grounding_preserves_last_successful_boxes_and_observation_time(answer):
    pipeline = CaptionPipeline("cam0", VLM(), ground=Grounder(LAMP, answer), quiet_every_s=0.0)
    feed(pipeline, SCENE, 100.0)

    out = feed(pipeline, PERSON, 131.0)

    assert out.objects is not None
    assert [(box["label"], box["ts"]) for box in out.objects["objects"]] == [("lamp", 100.0)]
    observation = out.objects["observation"]
    assert observation["status"] == "failed"
    assert observation["observed_at"] == 100.0
    assert observation["attempted_at"] == 131.0
    assert out.caption is not None
    assert [box["ts"] for box in out.caption["objects"]] == [100.0]


def test_failure_before_any_success_is_not_an_empty_observation():
    pipeline = CaptionPipeline("cam0", VLM(), ground=Grounder(RuntimeError("offline")))

    out = feed(pipeline, SCENE, 100.0)

    assert out.objects is not None and out.objects["objects"] == []
    assert out.objects["observation"]["status"] == "failed"
    assert out.objects["observation"]["observed_at"] is None
    assert out.objects["observation"]["attempted_at"] == 100.0


def test_expired_boxes_after_failure_do_not_claim_fresh_absence():
    pipeline = CaptionPipeline("cam0", VLM(), ground=Grounder(LAMP, RuntimeError("offline")), quiet_every_s=0.0)
    feed(pipeline, SCENE, 100.0)

    failed = feed(pipeline, SCENE, 100.0 + objects.TTL_S + 1)
    quiet = feed(pipeline, SCENE, 100.0 + objects.TTL_S + 2)

    assert failed.objects is not None and failed.objects["objects"] == []
    assert failed.objects["observation"]["status"] == "failed"
    assert failed.objects["observation"]["observed_at"] == 100.0
    assert failed.objects["observation"]["attempted_at"] == 100.0 + objects.TTL_S + 1
    assert quiet.objects is not None and quiet.objects["objects"] == []
    assert quiet.objects["observation"]["status"] in {"failed", "unevaluated"}
    assert quiet.objects["observation"]["observed_at"] == 100.0
    assert quiet.objects["observation"]["attempted_at"] is None


def test_disabled_grounding_emits_unevaluated_objects_at_caption_cadence():
    pipeline = CaptionPipeline("cam0", VLM(), quiet_every_s=10.0)

    first = feed(pipeline, SCENE, 100.0)
    quiet = feed(pipeline, SCENE, 101.0)
    held = feed(pipeline, SCENE, 105.0)
    later = feed(pipeline, SCENE, 111.0)

    assert first.objects is not None and first.objects["objects"] == []
    assert first.objects["observation"]["status"] == "unevaluated"
    assert first.objects["observation"]["observed_at"] is None
    assert first.objects["observation"]["attempted_at"] is None
    assert quiet.objects is not None and quiet.objects["ts_start"] == 101.0
    assert quiet.objects["observation"]["status"] == "unevaluated"
    assert held.caption is None and held.objects is None
    assert later.objects is not None and later.objects["ts_start"] == 111.0
    assert later.objects["observation"]["observed_at"] is None
    assert first.caption is not None and first.caption["objects"] is None


def test_successful_empty_retry_replaces_cached_boxes_and_advances_observation_time():
    pipeline = CaptionPipeline("cam0", VLM(), ground=Grounder(LAMP, "[]"))
    feed(pipeline, SCENE, 100.0)

    out = feed(pipeline, PERSON, 131.0)

    assert out.objects is not None and out.objects["objects"] == []
    assert out.objects["observation"]["status"] == "fresh"
    assert out.objects["observation"]["observed_at"] == 131.0
    assert out.objects["observation"]["attempted_at"] == 131.0
    assert out.caption is not None and out.caption["objects"] == []
