"""The animal check's decision rule and crop (vision/animal_verifier.py); the model itself is measured, not
unit tested (see the module's docs)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from liberational.vision import animal_verifier as V


def probs(**given: float) -> list[float]:
    names = [*V.SPECIES, *(f"p{k}" for k in range(len(V.PERSON))), *(f"b{k}" for k in range(len(V.BACKGROUND)))]
    return [given.get(n, 0.0) for n in names]


@pytest.mark.parametrize(
    ("given", "species", "ok"),
    [
        pytest.param({"cat": 0.6, "p0": 0.2, "p1": 0.2}, "cat", True, id="cat-beats-person-prompts"),
        pytest.param({"cat": 0.4, "p1": 0.3, "p2": 0.3}, "cat", True, id="independent-person-scores-do-not-add"),
        pytest.param({"cat": 0.3, "p1": 0.4}, "cat", False, id="a-person-prompt-wins"),
        pytest.param({"cat": 0.3, "b0": 0.4}, "cat", False, id="household-background-wins"),
        pytest.param({"cat": 0.1}, "cat", False, id="best-label-still-needs-minimum-evidence"),
        pytest.param({"cat": 0.3, "dog": 0.4}, "cat", False, id="another-species-wins"),
        pytest.param({"dog": 0.9, "p0": 0.1}, "cat", False, id="another-species-is-not-this-one"),
        pytest.param({"cat": 0.5, "p0": 0.5}, "cat", False, id="a-tie-is-not-a-pass"),
    ],
)
def test_a_box_needs_its_species_above_the_minimum_and_every_competitor(given, species, ok):
    assert V.passes(probs(**given), species, min_score=0.2) is ok


def test_default_gate_accepts_small_sigmoid_matches_without_softmax_sized_scores():
    assert V.passes(probs(cat=0.016, p0=0.0003), "cat") is True
    assert V.passes(probs(cat=0.0005), "cat") is False


@pytest.mark.parametrize(
    ("box", "shape"),
    [
        pytest.param((0.4, 0.2, 0.6, 0.3), (10, 40), id="a-4-to-1-box-keeps-its-shape"),
        pytest.param((-0.1, 0.95, 0.05, 1.2), (5, 10), id="clamped-to-the-frame"),
        pytest.param((0.5, 0.5, 0.5, 0.5), (1, 1), id="never-empty"),
    ],
)
def test_the_tight_crop_is_the_box_itself_clamped_to_the_frame(box, shape):
    frame = np.zeros((100, 200, 3), np.uint8)
    assert V.tight(frame, box).shape[:2] == shape


def test_the_crop_is_a_square_around_the_box_with_context_clamped_to_the_frame():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert V.square(frame, (0.4, 0.4, 0.5, 0.6)).shape[:2] == (22, 22)  # 20 px tall box, 10% context
    # a 10x5 px box at the corner: an 11 px square around its centre (5, 2.5), cut at the frame's edges
    assert V.square(frame, (0.0, 0.0, 0.05, 0.05)).shape[:2] == (8, 10)


class _Placed:
    """A model whose placement on the device fails, as a GPU out of memory does."""

    def half(self):
        return self

    def to(self, device):
        raise RuntimeError("CUDA out of memory")


def test_a_check_that_cant_be_placed_on_its_device_is_unavailable_not_a_crash(monkeypatch):
    model = SimpleNamespace(from_pretrained=lambda *a, **k: _Placed())
    processor = SimpleNamespace(from_pretrained=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=model, AutoProcessor=processor))
    with pytest.raises(V.Unavailable, match="out of memory"):
        V.SiglipVerifier("any/repo", "cuda:1")


def test_independent_image_text_scores_include_the_checkpoint_bias():
    torch = pytest.importorskip("torch")

    class Processor:
        def __call__(self, *, images, return_tensors):
            self.image = images[0]
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros(1, 3, 224, 224)}

    class Model:
        def get_image_features(self, *, pixel_values):
            return SimpleNamespace(pooler_output=torch.tensor([[3.0, 4.0]]))

    verifier = V.SiglipVerifier.__new__(V.SiglipVerifier)
    verifier.device, verifier.half = "cpu", False
    verifier.torch, verifier.processor, verifier.model = torch, Processor(), Model()
    verifier.scale, verifier.bias = 2.0, -1.0
    verifier.text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    crop = np.array([[[10, 20, 30]]], dtype=np.uint8)
    # Unit image [0.6, 0.8] yields logits [0.2, 0.6] after scale 2 and bias -1.
    assert verifier.probabilities(crop) == pytest.approx([0.5498339973, 0.6456563062])
    np.testing.assert_array_equal(verifier.processor.image, [[[30, 20, 10]]])
    verifier.text = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert verifier.probabilities(crop)[:2] == pytest.approx([0.5498339973, 0.6456563062])


def test_independent_person_scores_are_not_summed_like_a_softmax_distribution():
    assert V.passes(probs(cat=0.7, p0=0.4, p1=0.4), "cat") is True


def test_text_features_are_cached_with_the_checkpoint_tokenization_contract(monkeypatch):
    torch = pytest.importorskip("torch")
    loads = []

    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def __call__(self, *, text, return_tensors, padding, max_length, truncation):
            self.text, self.padding, self.max_length, self.truncation = text, padding, max_length, truncation
            assert return_tensors == "pt"
            return Batch(input_ids=torch.zeros(len(text), max_length, dtype=torch.long))

    class Model:
        logit_scale = torch.tensor(0.69314718056)
        logit_bias = torch.tensor(-1.0)

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def get_text_features(self, *, input_ids):
            return SimpleNamespace(pooler_output=torch.tensor([[3.0, 4.0]]).repeat(input_ids.shape[0], 1))

    model, processor = Model(), Processor()

    def model_loader(repo, *, local_files_only):
        loads.append(("model", repo, local_files_only))
        return model

    def processor_loader(repo, *, local_files_only):
        loads.append(("processor", repo, local_files_only))
        return processor

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=model_loader),
            AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
        ),
    )
    verifier = V.SiglipVerifier("example/local-model", "cpu")
    assert loads == [("model", "example/local-model", True), ("processor", "example/local-model", True)]
    assert (processor.padding, processor.max_length, processor.truncation) == ("max_length", 64, True)
    assert "this is a photo of a cat." in processor.text
    assert "this is a photo of a person's hair." in processor.text
    assert "this is a photo of household furniture." in processor.text
    np.testing.assert_allclose(verifier.text[0].numpy(), [0.6, 0.8])
    assert verifier.bias == -1.0


@pytest.mark.parametrize("given", [{"cat": np.nan}, {"cat": np.inf}, {"cat": 1.1}, {"cat": 0.6, "p0": np.nan}])
def test_nonfinite_or_out_of_range_scores_do_not_confirm_an_animal(given):
    assert V.passes(probs(**given), "cat") is False


def test_missing_scores_unknown_species_and_an_empty_crop_do_not_confirm_an_animal():
    assert V.passes([], "cat") is False
    assert V.passes(probs(cat=0.9), "rabbit") is False
    verifier = V.SiglipVerifier.__new__(V.SiglipVerifier)
    assert verifier.verify_crop(np.zeros((0, 0, 3), dtype=np.uint8), "cat") is False
    assert verifier.verify_crop(np.zeros((1, 1, 3), dtype=np.uint8), "rabbit") is False


def test_the_minimum_score_can_be_selected_independently_of_competing_prompts():
    scores = probs(cat=0.6, p0=0.2)
    assert V.passes(scores, "cat", min_score=0.55) is True
    assert V.passes(scores, "cat", min_score=0.65) is False


@pytest.mark.parametrize(
    ("choice", "gpus", "mps", "expected"),
    [("auto", 2, True, "cuda:1"), ("auto", 0, True, "mps"), ("auto", 0, False, "cpu"), ("cpu", 2, False, "cpu")],
)
def test_device_selection_keeps_cpu_and_apple_fallbacks_and_honors_an_override(
    monkeypatch, choice, gpus, mps, expected
):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: gpus > 0, device_count=lambda: gpus),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        ),
    )
    assert V.torch_device(choice) == expected


@pytest.mark.parametrize("minimum", [-0.1, 1.1, float("nan")])
def test_an_invalid_minimum_is_rejected_before_any_model_can_load(minimum):
    with pytest.raises(ValueError, match="between zero and one"):
        V.SiglipVerifier("example/local-model", "cpu", min_score=minimum)


def test_the_probe_initializes_cuda_before_resetting_its_memory_counters(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "probe_animal_verifier", Path(__file__).parents[1] / "tools/probe_animal_verifier.py"
    )
    assert spec and spec.loader
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    events = []

    def initialize():
        events.append("init")

    def reset(device):
        assert events == ["init"]
        assert device == "cuda:0"
        events.append("reset")

    class FakeVerifier:
        def __init__(self, repo, device, *, min_score):
            assert events == ["init", "reset"]
            self.repo, self.half = repo, True

        def probabilities(self, crop):
            return np.asarray(probs(cat=0.9), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda _: None,
            cuda=SimpleNamespace(
                init=initialize,
                reset_peak_memory_stats=reset,
                memory_allocated=lambda _: 1024**2,
                max_memory_allocated=lambda _: 2 * 1024**2,
            ),
        ),
    )
    monkeypatch.setattr(probe.verifier, "SiglipVerifier", FakeVerifier)
    path = tmp_path / "crop.jpg"
    assert cv2.imwrite(str(path), np.zeros((4, 4, 3), dtype=np.uint8))
    report = probe.run([{"path": str(path), "sha256": "sample", "species": "cat"}], "example/model", "cuda:0", 0.2)
    assert report["torch_peak_allocated_mib"] == 2
    assert report["rows"][0]["passes"] is True
