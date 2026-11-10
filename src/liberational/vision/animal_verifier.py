"""A second opinion on detected animals: SigLIP 2 compares a square crop with species and confounder prompts.

The detector can call hair, part of a person, or household objects an animal. A crop passes only when the detected
species has a sufficient independent sigmoid score and beats every other species, person, and household-background
prompt. These scores are image/text matches, not calibrated probabilities that a pet is present. They must not be
softmaxed or summed as mutually exclusive person probabilities.

The complete SigLIP 2 Base B/16 image/text model comes from [models.hf.animal_verifier], installed by `mise run models`
and never downloaded here. Its own processor handles 224px RGB resize/normalization; text uses lowercased
"this is a photo of ..." prompts padded/truncated to 64 tokens. Cached normalized text embeddings, the checkpoint's
learned logit scale AND bias, and the image embedding reproduce its sigmoid scoring without re-encoding text per crop.

`vision.animal_models_device` "auto" takes the last CUDA GPU, else Apple's GPU, else CPU. CUDA uses fp16 weights;
CPU and MPS use fp32. Missing weights or a placement failure leave the animal check explicitly unavailable.

Model and license: https://huggingface.co/google/siglip2-base-patch16-224 (Apache-2.0).
Processor/scoring: https://huggingface.co/docs/transformers/model_doc/siglip2
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import model_runtime

SPECIES = ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")
PERSON = ("a person", "the top of a person's head", "a person's hair", "a person's face", "a person's shoulder and arm")
BACKGROUND = ("household furniture", "a blanket", "clothing", "an empty room", "a toy animal")
LABELS = tuple(f"{'an' if s[0] in 'aeiou' else 'a'} {s}" for s in SPECIES) + PERSON + BACKGROUND
PROMPTS = tuple(f"this is a photo of {label}." for label in LABELS)
# Small paired local crop check: animal-candidate scores started at 0.00138; person/furniture candidates stayed
# below 0.00031. This is an initial operating point, not a calibrated probability or an accuracy claim.
MIN_SCORE = 0.001
CONTEXT = 1.1  # the square crop's side, as a multiple of the box's longer side


class Unavailable(RuntimeError):
    """The verifier can't run (weights missing, or the runtime failed to load)."""


def square_bounds(height: int, width: int, box: Sequence[float]) -> tuple[int, int, int, int]:
    """The pixel bounds (left, top, right, bottom) of `square`'s crop in a frame this size."""
    h, w = height, width
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    side = max(x2 - x1, y2 - y1, 1.0) * CONTEXT
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, max(a + 1, cx + side / 2))), int(min(h, max(b + 1, cy + side / 2)))
    return a, b, c, d


def square(frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """The box's square crop with a little context, clamped to the frame."""
    a, b, c, d = square_bounds(frame.shape[0], frame.shape[1], box)
    return frame[b:d, a:c]


def tight_bounds(height: int, width: int, box: Sequence[float]) -> tuple[int, int, int, int]:
    """The pixel bounds (left, top, right, bottom) of the box itself in a frame this size, clamped to it and at least a
    pixel each way: the crop the pet embedder reads, where `square` would pad a 2:1 animal with background."""
    h, w = height, width
    # Rounded, not floored: a float32 box (0.6 as 0.60000002) must cut the same pixels as its float64 twin.
    a, b = min(max(0, round(box[0] * w)), w - 1), min(max(0, round(box[1] * h)), h - 1)
    c, d = min(w, max(a + 1, round(box[2] * w))), min(h, max(b + 1, round(box[3] * h)))
    return a, b, c, d


def tight(frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """The box's own crop of a frame (`tight_bounds`)."""
    a, b, c, d = tight_bounds(frame.shape[0], frame.shape[1], box)
    return frame[b:d, a:c]


def passes(scores: Sequence[float] | np.ndarray, species: str, min_score: float = MIN_SCORE) -> bool:
    """The candidate clears its minimum sigmoid score and beats every competing image/text prompt."""
    values = np.asarray(scores, dtype=np.float32)
    if species not in SPECIES or values.shape != (len(PROMPTS),) or not np.isfinite(values).all():
        return False
    if not ((0 <= values) & (values <= 1)).all():
        return False
    index = SPECIES.index(species)
    candidate = float(values[index])
    return candidate >= min_score and candidate > float(np.max(np.delete(values, index)))


def torch_device(choice: str) -> str:
    """The torch device for the animal models (`vision.animal_models_device`)."""
    import torch

    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.device_count() - 1}"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class SiglipVerifier:
    """The SigLIP 2 check, its text prompts encoded once at load."""

    def __init__(self, repo: str | None = None, device: str | None = None, *, min_score: float = MIN_SCORE) -> None:
        if not 0 <= min_score <= 1:
            raise ValueError("the animal check minimum score must be between zero and one")
        self.min_score = min_score
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise Unavailable("the animal check needs PyTorch and Transformers (the segmentation extra)") from error
        repo = repo or cfg.hf_repo("animal_verifier")
        self.repo = repo
        self.device = torch_device(device or str(cfg.get("vision.animal_models_device")))
        try:
            model: Any = AutoModel.from_pretrained(repo, local_files_only=True)
            self.processor: Any = AutoProcessor.from_pretrained(repo, local_files_only=True)
        except OSError as error:
            raise Unavailable(f"animal check weights are missing ({repo}); run mise run models") from error
        self.half = self.device.startswith("cuda")
        self.torch = torch
        try:
            # Placing the model can fail for want of GPU memory (a RuntimeError). That disables the check, never
            # the body worker it runs in.
            self.model = (model.half() if self.half else model).to(self.device).eval()
            with torch.inference_mode():
                tokens = self.processor(
                    text=list(PROMPTS), return_tensors="pt", padding="max_length", max_length=64, truncation=True
                ).to(self.device)
                self.text = self._unit(self.model.get_text_features(**tokens))
            self.scale = float(self.model.logit_scale.detach().float().exp())
            self.bias = float(self.model.logit_bias.detach().float())
        except Exception as error:
            raise Unavailable(f"the animal check couldn't load on {self.device}: {error}") from error
        model_runtime.loaded("models.hf.animal_verifier", repo, "animal check", self.device, owner=self)

    def _unit(self, features: Any) -> Any:
        tensor = getattr(features, "pooler_output", features)
        return self.torch.nn.functional.normalize(tensor.float(), dim=-1)

    def probabilities(self, crop: np.ndarray) -> np.ndarray:
        """Independent sigmoid scores over LABELS for a BGR crop; unrelated prompts never change a score."""
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        with self.torch.inference_mode():
            pixels = self.processor(images=[rgb], return_tensors="pt")["pixel_values"].to(self.device)
            image = self._unit(self.model.get_image_features(pixel_values=pixels.half() if self.half else pixels))
            return (self.scale * image @ self.text.T + self.bias).sigmoid()[0].cpu().numpy()

    def verify(self, frame: np.ndarray, box: Sequence[float], species: str) -> bool:
        return self.verify_crop(square(frame, box), species)

    def verify_crop(self, crop: np.ndarray, species: str) -> bool:
        """Whether a BGR crop (as `square` cuts it) shows `species`."""
        return species in SPECIES and crop.size > 0 and passes(self.probabilities(crop), species, self.min_score)
