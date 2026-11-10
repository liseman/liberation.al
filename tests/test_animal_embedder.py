"""The pet embedding adapter contract, with synthetic CPU tensors and no model weights."""

import sys
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

from liberational import config
from liberational.vision import animal_embedder as E

torch = pytest.importorskip("torch")
REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"


class Model:
    def __init__(self):
        self.output = torch.zeros((2, 384), dtype=torch.float16)
        self.output[0, :2] = torch.tensor([3, 4])
        self.output[1, :2] = torch.tensor([0, -2])
        self.device = None
        self.is_half = False
        self.is_eval = False
        self.inputs = []

    def half(self):
        self.is_half = True
        return self

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.is_eval = True
        return self

    def __call__(self, *, pixel_values):
        self.inputs.append(pixel_values)
        assert not torch.is_grad_enabled()
        # Patch/register features deliberately disagree with CLS: averaging them is the wrong embedding.
        return SimpleNamespace(pooler_output=self.output, last_hidden_state=torch.ones((2, 201, 384)))


class Processor:
    def __init__(self):
        self.images = []

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        self.images.extend(images)
        return {"pixel_values": torch.zeros((len(images), 3, 224, 224), dtype=torch.float32)}


def load(repo, *, local_files_only, revision=None):
    raise AssertionError("every pretrained loader is replaced")


@pytest.fixture
def adapter(monkeypatch):
    model, processor = Model(), Processor()
    model_loader = create_autospec(load, return_value=model)
    processor_loader = create_autospec(load, return_value=processor)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=model_loader),
            AutoImageProcessor=SimpleNamespace(from_pretrained=processor_loader),
        ),
    )
    monkeypatch.setattr(E.model_runtime, "loaded", create_autospec(E.model_runtime.loaded))
    return SimpleNamespace(
        model=model, processor=processor, model_loader=model_loader, processor_loader=processor_loader
    )


def test_configured_model_and_processor_are_loaded_offline(adapter):
    with config.override({"models": {"hf": {"animal_embedding": {"repo": REPO, "revision": ""}}}}):
        embedder = E.DinoEmbedder(device="cpu")
    adapter.model_loader.assert_called_once_with(REPO, local_files_only=True)
    adapter.processor_loader.assert_called_once_with(REPO, local_files_only=True)
    assert embedder.name == REPO


def test_configured_revision_pins_both_weights_and_preprocessing(adapter):
    with config.override({"models": {"hf": {"animal_embedding": {"repo": REPO, "revision": "fixed-commit"}}}}):
        E.DinoEmbedder(device="cpu")
    adapter.model_loader.assert_called_once_with(REPO, local_files_only=True, revision="fixed-commit")
    adapter.processor_loader.assert_called_once_with(REPO, local_files_only=True, revision="fixed-commit")


def test_explicit_repo_does_not_inherit_another_models_configured_revision(adapter):
    with config.override({"models": {"hf": {"animal_embedding": {"revision": "configured-commit"}}}}):
        E.DinoEmbedder("local/installed-snapshot", "cpu")
    adapter.model_loader.assert_called_once_with("local/installed-snapshot", local_files_only=True)
    adapter.processor_loader.assert_called_once_with("local/installed-snapshot", local_files_only=True)


@pytest.mark.parametrize(("device", "half"), [("cpu", False), ("mps", False), ("cuda:1", True)])
def test_placement_uses_half_precision_only_on_cuda(adapter, device, half):
    E.DinoEmbedder(REPO, device)
    assert adapter.model.device == device
    assert adapter.model.is_half is half
    assert adapter.model.is_eval


def test_bgr_batch_uses_processor_and_unit_cls_vectors_in_input_order(adapter):
    embedder = E.DinoEmbedder(REPO, "cpu")
    crops = [np.full((3, 5, 3), [10, 20, 30], np.uint8), np.full((5, 2, 3), [40, 50, 60], np.uint8)]
    vectors = embedder.embed_crops(crops)
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors[:, :2], [[0.6, 0.8], [0, -1]])
    assert np.count_nonzero(vectors[:, 2:]) == 0
    np.testing.assert_array_equal(adapter.processor.images[0][0, 0], [30, 20, 10])
    np.testing.assert_array_equal(adapter.processor.images[1][0, 0], [60, 50, 40])
    assert adapter.processor.images[0].flags.c_contiguous
    assert adapter.model.inputs[0].dtype == torch.float32
    np.testing.assert_array_equal(crops[0][0, 0], [10, 20, 30])


def test_empty_batch_never_calls_processor_or_model(adapter):
    result = E.DinoEmbedder(REPO, "cpu").embed_crops([])
    assert result.shape == (0, 384)
    assert result.dtype == np.float32
    assert adapter.processor.images == []
    assert adapter.model.inputs == []


def test_embed_crops_a_normalized_frame_box_tight(adapter):
    adapter.model.output = adapter.model.output[:1]
    result = E.DinoEmbedder(REPO, "cpu").embed(np.zeros((100, 200, 3), np.uint8), [0.4, 0.4, 0.5, 0.6])
    assert adapter.processor.images[0].shape == (20, 20, 3)  # the box itself, no square context
    np.testing.assert_allclose(result[:2], [0.6, 0.8])


@pytest.mark.parametrize("failed_loader", ["model_loader", "processor_loader"])
def test_missing_cached_files_are_unavailable_without_network_retry(adapter, failed_loader):
    loader = getattr(adapter, failed_loader)
    loader.side_effect = OSError("not cached")
    with pytest.raises(E.Unavailable, match="weights are missing"):
        E.DinoEmbedder(REPO, "cpu")
    loader.assert_called_once_with(REPO, local_files_only=True)


def test_device_placement_failure_is_unavailable(adapter, monkeypatch):
    def fail(device):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(adapter.model, "to", fail)
    with pytest.raises(E.Unavailable, match="device unavailable"):
        E.DinoEmbedder(REPO, "cpu")


@pytest.mark.parametrize("shape", [(2, 768), (1, 384), (2, 1, 384)])
def test_wrong_width_batch_or_token_output_cannot_enter_pet_history(adapter, shape):
    adapter.model.output = torch.ones(shape)
    with pytest.raises(E.Unavailable, match="384"):
        E.DinoEmbedder(REPO, "cpu").embed_crops([np.zeros((2, 2, 3), np.uint8)] * 2)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_invalid_vectors_cannot_enter_pet_history(adapter, value):
    adapter.model.output = torch.full((2, 384), value)
    with pytest.raises(E.Unavailable, match="finite.*nonzero"):
        E.DinoEmbedder(REPO, "cpu").embed_crops([np.zeros((2, 2, 3), np.uint8)] * 2)
