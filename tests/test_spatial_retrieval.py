"""Walkthrough revisit proposals (#2931): MegaLoc similarity through onnxruntime, and the shortlist MASt3R measures. A
tiny ONNX graph (each image's mean normalized colour) stands in for MegaLoc."""

import numpy as np
import pytest

from liberational.spatial import retrieval as R

onnx = pytest.importorskip("onnx")
cv2 = pytest.importorskip("cv2")

RED, GREEN = (0, 0, 255), (0, 255, 0)  # BGR, as cv2 writes them


@pytest.fixture
def model(tmp_path):
    """An ONNX model with MegaLoc's input and output names whose descriptor is each image's per-channel mean."""
    from onnx import TensorProto, helper

    mean = helper.make_node("ReduceMean", ["images"], ["descriptor"], axes=[2, 3], keepdims=0)
    graph = helper.make_graph(
        [mean],
        "mean-colour",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, R.SIDE, R.SIDE])],
        [helper.make_tensor_value_info("descriptor", TensorProto.FLOAT, ["batch", 3])],
    )
    path = tmp_path / "megaloc.onnx"
    built = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    built.ir_version = 8  # one every onnxruntime build here loads
    onnx.save(built, path)
    return path


def frames(tmp_path, colours):
    paths = []
    for k, bgr in enumerate(colours):
        path = tmp_path / f"f{k:03d}.jpg"
        cv2.imwrite(str(path), np.full((48, 64, 3), bgr, np.uint8))
        paths.append(path)
    return paths


def normalized(rgb):
    return np.array([(c / 255 - m) / s for c, m, s in zip(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def test_similarity_is_the_cosine_of_the_images_rgb_descriptors(tmp_path, model):
    sim = R.similarity(frames(tmp_path, [RED, GREEN, RED]), model)
    red, green = normalized((255, 0, 0)), normalized((0, 255, 0))  # read as BGR: -0.36, not -0.49
    c = float(red @ green / (np.linalg.norm(red) * np.linalg.norm(green)))
    assert sim == pytest.approx(np.array([[1, c, 1], [c, 1, c], [1, c, 1]]), abs=0.02)


def test_shortlist_goes_best_first_far_enough_apart_without_repeating_a_revisit_or_overusing_a_frame():
    s = np.zeros((10, 10))
    s[0, 2] = 90  # closer than the gap
    s[0, 6] = 50
    s[1, 7] = 40  # the same revisit as (0, 6), a frame on
    s[0, 9] = 30
    s[0, 3] = 25  # frame 0 is already in two
    s[4, 9] = 20
    assert R.shortlist(s, gap=3, partners=2, cap=3, spread=2) == [(0, 6), (0, 9), (4, 9)]
    assert R.shortlist(s, gap=3, partners=2, cap=2, spread=2) == [(0, 6), (0, 9)]


def test_a_walk_that_comes_back_proposes_the_frames_that_look_alike(tmp_path, model, monkeypatch):
    monkeypatch.setattr(R, "LOOP_GAP", 4)
    greys = [(g, g, g) for g in (40, 70, 100, 130, 160, 190)]
    paths = frames(tmp_path, [RED, *greys[1:5], RED])  # frame 5 is back where frame 0 was
    assert R.proposals(paths, model=model) == [(0, 5)]


def test_a_walk_too_short_for_a_revisit_proposes_none_and_needs_no_model(tmp_path):
    paths = frames(tmp_path, [RED] * R.LOOP_GAP)
    assert R.proposals(paths, model=tmp_path / "missing.onnx") == []


def test_a_missing_model_fails_a_walk_that_needs_it_and_skips_one_that_doesnt(tmp_path):
    paths = frames(tmp_path, [RED] * (R.LOOP_GAP + 1))
    with pytest.raises(RuntimeError, match="mise run models"):
        R.proposals(paths, model=tmp_path / "missing.onnx")
    assert R.proposals(paths, required=False, model=tmp_path / "missing.onnx") == []


@pytest.mark.parametrize(("frames", "pairs"), [(31, 25), (125, 25), (126, 25), (300, 60), (600, 120)])
def test_a_walk_past_two_minutes_may_propose_a_revisit_for_every_five_frames(frames, pairs):
    assert R.cap_for(frames) == pairs
