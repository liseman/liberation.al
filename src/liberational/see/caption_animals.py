"""Animals the captions name: the caption VLM is the animal detector at home.

see-actions' detector (RF-DETR Small, see/gpu_pose.py) boxes animals too. This path was added when that detector was
YOLOX-nano, which on the Linux machine gave the household dog 0.04 to 0.11 (its threshold was 0.35) while about one
caption in six named the dog. So the caption pipeline finds them:

- Any grounding of the frame (the scheduled one, `see/captions.py`) that boxes an animal species gives its boxes.
- Otherwise, a fresh caption that names an animal (a species word such as "dog" or "kittens", "pet" or "animal", or a
  taught pet's name) asks the VLM for the animals in that frame, with a prompt for animals only, at most every
  `every_s` per camera. A caption that says "dog bed" just gets an answer with no animal in it.

Each box becomes an `animal.grounding` row (verified: the VLM's look is the image check) and, when its shorter side
has `MIN_EMBED_PX` pixels, an `animal.grounding_embedding` row from the configured DINO crop embedding: see-actions' streams'
shapes in streams of their own, since a stream has one writer per camera. Pet recognition
(people/pet_recognition.py) and the Pets page see it as a sighting. A box continues the camera's last track of its
species when it is the only one of that species now and was then, within `SAME_VISIT_S`; two cats at once each start
their own, so no track mixes two animals. The embedder is made on the first box (`embed` is called lazily by the
worker), so a camera that never sees an animal never loads it; a failing embedder stops embeddings only.
"""

import re
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from liberational import logs, segment_log
from liberational.vision import objects

LOG = logs.get("see-captions")

# Words a caption or a grounded label may use, each to its household species (people/pets.HOUSEHOLD). Other
# animals aren't pets here: a grounded "horse" or "zebra" is no sighting, and a caption naming one asks no look.
SPECIES_WORDS: dict[str, str] = {
    **dict.fromkeys(("dog", "dogs", "puppy", "puppies", "pup", "pups"), "dog"),
    **dict.fromkeys(("cat", "cats", "kitten", "kittens", "kitty"), "cat"),
    **dict.fromkeys(("bird", "birds", "parrot", "parrots"), "bird"),
}
# Words that name an animal without its species: they trigger a look, whose boxes then say which.
ANY_ANIMAL = ("pet", "pets", "animal", "animals")
EVERY_S = 30.0  # a camera's caption-triggered animal grounding at most this often (an Ollama call, about 1 to 2 s)
SAME_VISIT_S = 120.0  # a lone box continues its species' last lone box on the camera within this long
MIN_EMBED_PX = 48  # a box is embedded only when its shorter side has this many pixels (as see-actions does)
PROMPT = (
    "Find every dog, cat, and bird in this image. Not people, toys, pictures, or animal-shaped objects. Reply with "
    "JSON only, one animal per line, its species as the label (dog, cat, or bird): "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. Reply [] when there is none.'
)
_WORD = re.compile(r"[a-z]+")

Embed = Callable[[np.ndarray, Sequence[float]], np.ndarray]


def species_of(label: str) -> str | None:
    """The COCO species a grounded label names by its last word ("dog", "small white dog", "cats"), or None: a
    "dog bed" or a "cat tree" is a bed or a tree."""
    words = _WORD.findall(label.lower())
    return SPECIES_WORDS.get(words[-1]) if words else None


def names_animal(text: str, pet_names: Iterable[str] = ()) -> bool:
    """Whether a caption names an animal: a species word, "pet" or "animal", or a taught pet's whole name."""
    lowered = text.lower()
    words = set(_WORD.findall(lowered))
    if words & SPECIES_WORDS.keys() or words.intersection(ANY_ANIMAL):
        return True
    for name in pet_names:
        name_words = _WORD.findall(name.lower())
        if name_words and re.search(r"\b" + r"\W+".join(re.escape(word) for word in name_words) + r"\b", lowered):
            return True
    return False


@dataclass
class Found:
    """One animal box of a frame: its species, its box normalized to the frame, its track, and its embedding (None
    when it wasn't embedded)."""

    species: str
    box: list[float]
    track_id: str
    vector: np.ndarray | None = None


@dataclass
class _Track:
    id: str
    last: float
    alone: bool  # the only box of its species on the camera when last seen


class CaptionAnimals:
    """One camera's animals from its captions. `find(frame) -> model text` asks the VLM for the frame's animals (None
    uses scheduled groundings only); `embed(frame, box)` embeds a box's crop (None leaves embeddings out);
    `pet_names()` gives the taught pets' names."""

    def __init__(
        self,
        cam: str,
        find: Callable[[np.ndarray], str] | None,
        embed: Embed | None,
        pet_names: Callable[[], Iterable[str]] = list,
        *,
        every_s: float = EVERY_S,
        shots: Any = None,
    ) -> None:
        """`shots` (a `pet_shots.Writer`) is offered each embedded box, for Pets' photo of its track."""
        self.cam, self.find, self.embed, self.pet_names, self.every_s = cam, find, embed, pet_names, every_s
        self.shots = shots
        self.asked_at = -float("inf")
        self.run = uuid.uuid7().hex[:20]  # track ids no other run used, starting with the run's time (face_shots.hour)
        self.count = 0
        self.tracks: dict[str, _Track] = {}  # each species' last track on this camera
        self.warned: set[str] = set()

    def look(self, frame: np.ndarray, ts: float, caption: str, grounded: Sequence[objects.Item] | None) -> list[Found]:
        """A frame's animals: from `grounded` (this frame's scheduled grounding, None when there was none), else, when
        the fresh `caption` names an animal and the camera wasn't asked within `every_s`, from asking the VLM."""
        boxes = _animal_boxes(grounded or [])
        due = not boxes and self.find is not None and caption and ts - self.asked_at >= self.every_s
        if due and names_animal(caption, self._names()):
            self.asked_at = ts
            boxes = self._ask(frame)
        return self._tracked(frame, ts, boxes)

    def _names(self) -> list[str]:
        try:
            return list(self.pet_names())
        except Exception as error:  # the names only widen the trigger; species words still work without them
            self._warn("names", "couldn't read the taught pets' names: %s", error)
            return []

    def _ask(self, frame: np.ndarray) -> list[tuple[str, list[float]]]:
        from liberational.see.captions import grounded_items

        assert self.find is not None
        try:
            return _animal_boxes(grounded_items(self.find(frame)))
        except Exception as error:  # an unknown answer: no boxes this time, and the next named animal asks again
            self._warn(type(error).__name__, "animal grounding failed: %s", error)
            return []

    def _tracked(self, frame: np.ndarray, ts: float, boxes: list[tuple[str, list[float]]]) -> list[Found]:
        per_species: dict[str, int] = {}
        for species, _ in boxes:
            per_species[species] = per_species.get(species, 0) + 1
        found = []
        for species, box in boxes:
            alone = per_species[species] == 1
            last = self.tracks.get(species)
            if not (alone and last is not None and last.alone and 0 <= ts - last.last <= SAME_VISIT_S):
                self.count += 1
                last = _Track(f"{self.run}:{species}:{self.count}", ts, alone)
            last.last, last.alone = ts, alone
            self.tracks[species] = last
            found.append(Found(species, box, last.id, self._embedded(frame, box)))
            if self.shots is not None and found[-1].vector is not None:
                h, w = frame.shape[:2]
                self.shots.offer(last.id, box, w, h, lambda b: frame[b[1] : b[3], b[0] : b[2]])
        return found

    def _embedded(self, frame: np.ndarray, box: list[float]) -> np.ndarray | None:
        h, w = frame.shape[:2]
        if self.embed is None or min((box[2] - box[0]) * w, (box[3] - box[1]) * h) < MIN_EMBED_PX:
            return None
        try:
            return np.asarray(self.embed(frame, box), np.float32)
        except Exception as error:  # optional: a failing embedder stops embeddings, never sightings
            self._warn("embed", "pet embeddings failed, so pets aren't told apart from captions: %s", error)
            self.embed = None
            return None

    def _warn(self, kind: str, message: str, *args: Any) -> None:
        if kind not in self.warned:  # once per kind, so a broken path is seen without flooding the log
            self.warned.add(kind)
            LOG.warning("%s: " + message, self.cam, *args)


def _animal_boxes(items: Iterable[objects.Item]) -> list[tuple[str, list[float]]]:
    """Grounded items that are animals, as (species, box normalized to the frame)."""
    out = []
    for item in items:
        species = species_of(item["label"])
        if species is not None:
            out.append((species, [min(1.0, max(0.0, float(v) / 1000)) for v in item["box"][:4]]))
    return out


class LazyEmbedder:
    """The configured DINO pet embedder (vision/animal_embedder.py), made on its first call, so cameras that never see an
    animal never load it. A failed load raises each call; `CaptionAnimals` stops calling after the first."""

    def __init__(self) -> None:
        self.model: Any = None

    def __call__(self, frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
        if self.model is None:
            from liberational.vision.animal_embedder import DinoEmbedder

            self.model = DinoEmbedder()
        return self.model.embed(frame, box)


def rows(found: Sequence[Found], frame_id: str, frame_ns: int, model: str, embedding_model: str) -> dict[str, list]:
    """A frame's animals as `animal.grounding` and `animal.grounding_embedding` rows, by stream."""
    detections = [
        {
            "ts_ns": frame_ns,
            "frame_id": frame_id,
            "track_id": f.track_id,
            "species": f.species,
            "box": [*f.box, 1.0],  # the VLM gives no score
            "verified": True,  # the VLM's look is the image check
            "model": model,
        }
        for f in found
    ]
    embeddings = [
        {
            "ts_ns": frame_ns,
            "track_id": f.track_id,
            "species": f.species,
            "box": [*f.box, 1.0],
            "vector": f.vector.tolist(),
            "model": embedding_model,
        }
        for f in found
        if f.vector is not None
    ]
    return {segment_log.ANIMAL_GROUNDINGS: detections, segment_log.ANIMAL_GROUNDING_EMBEDDINGS: embeddings}
