"""Place recognition between frames: MegaLoc similarity through onnxruntime. A tiny ONNX graph (each image's mean
normalized colour) stands in for MegaLoc."""

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


def test_a_missing_model_says_how_to_fetch_it(tmp_path):
    with pytest.raises(RuntimeError, match="mise run models"):
        R.similarity(frames(tmp_path, [RED]), tmp_path / "missing.onnx")
