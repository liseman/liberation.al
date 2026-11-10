"""Appearance embeddings of animal crops, for telling pets of one species apart.

Each confirmed animal track's box is cropped tight (`animal_verifier.tight`) and embedded by Meta's DINOv2 ViT-S/14
([models.hf.animal_embedding]: Apache-2.0, ungated, a pinned revision): its 384-d CLS output, unit length.
Self-supervised features keep instance detail (coat pattern, markings) that a text-aligned model tends to fold into
"a cat": SigLIP 2's image embeddings of two cats of a household were 0.91 alike where one cat's own were 0.94.

The crop is the box itself, not `animal_verifier.square`'s square with 10% context: a 2:1 cat or dog's square
is about 60% floor or couch, which pulls one pet's sightings on different backgrounds apart. On the owner's labeled
photos, tight crops made about half the mistakes telling two cats apart at the same recall.

`space()` names the embedding space, the model and the crop: every row and `pet.reference` records it as its
`model`, and pet recognition ignores a row or reference of another space, since vectors of two models or two crops
don't compare (DINOv2 and DINOv3 both make 384 numbers, and they mean different things). After the model changes,
tools/restore_pet_references.py embeds the taught references again from their kept photos (`pet.photo`), and
recognition's thresholds (people/pet_recognition.py) need setting again.

The checkpoint's Transformers image processor owns preprocessing of RGB uint8 crops. DINOv2's resizes the shorter
side to 256 (bicubic), takes the centre 224 x 224, and applies ImageNet normalization, so a long box loses its
ends. The model's `pooler_output` is the final layer-normalized CLS token, not a mean of patch tokens
(https://huggingface.co/docs/transformers/model_doc/dinov2).

The weights are installed by `mise run models`, which checks them against the entry's `sha256`
(tools/fetch_models.py), and are never downloaded here. It runs where the animal check does
(`vision.animal_models_device`). Configured weights and preprocessing use the same revision; an explicitly
supplied repository or local snapshot does not inherit the configured model's revision.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import model_runtime
from liberational.vision.animal_verifier import Unavailable, tight, torch_device

DIM = 384
CROP = "tight"  # how a box is cut before embedding; part of the embedding space


def space() -> str:
    """The embedding space animal embeddings are in: the model and the crop, as rows record it. The model is named
    by `[models.hf.animal_embedding]`'s `space` (the weights' own name, whichever repository they're fetched from),
    else by its `repo`."""
    declared = str(cfg.get("models.hf.animal_embedding.space", "") or "").strip()
    return f"{declared or cfg.hf_repo('animal_embedding')}:{CROP}"


class DinoEmbedder:
    """`embed(frame, box)`: the unit embedding of a normalized box's crop in a BGR frame."""

    def __init__(self, repo: str | None = None, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise Unavailable("pet embeddings need PyTorch and Transformers (the segmentation extra)") from error
        # An explicit repository/local snapshot belongs to the caller; never combine it with another repo's pin.
        revision = cfg.get("models.hf.animal_embedding.revision", None) if repo is None else None
        repo = repo or cfg.hf_repo("animal_embedding")
        pinned = {"revision": revision} if revision else {}
        self.device = torch_device(device or str(cfg.get("vision.animal_models_device")))
        try:
            model: Any = AutoModel.from_pretrained(repo, local_files_only=True, **pinned)
            self.processor: Any = AutoImageProcessor.from_pretrained(repo, local_files_only=True, **pinned)
        except OSError as error:
            raise Unavailable(f"pet embedding weights are missing ({repo}); run mise run models") from error
        self.half = self.device.startswith("cuda")
        try:
            self.model = (model.half() if self.half else model).to(self.device).eval()
        except RuntimeError as error:
            raise Unavailable(f"pet embeddings couldn't load on {self.device}: {error}") from error
        self.torch = torch
        self.name = repo
        model_runtime.loaded("models.hf.animal_embedding", repo, "pet embeddings", self.device, owner=self)

    def embed(self, frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
        return self.embed_crops([tight(frame, box)])[0]

    def embed_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """The unit embeddings of BGR crops (as `animal_verifier.tight` cuts them), N x DIM, in one batch."""
        if not crops:
            return np.zeros((0, DIM), np.float32)
        rgb = [np.ascontiguousarray(crop[:, :, ::-1]) for crop in crops]
        with self.torch.inference_mode():
            pixels = self.processor(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
            out = self.model(pixel_values=pixels.half() if self.half else pixels).pooler_output.float()
            if tuple(out.shape) != (len(crops), DIM):
                raise Unavailable(f"pet embeddings need one {DIM}-value CLS vector per crop; got {tuple(out.shape)}")
            vectors = self.torch.nn.functional.normalize(out, dim=-1).cpu().numpy()
        if not np.isfinite(vectors).all() or not np.all(np.linalg.norm(vectors, axis=-1) > 0):
            raise Unavailable("pet embeddings must be finite and nonzero")
        return vectors.astype(np.float32)
