"""The gate and the kept frames for learning surroundings (#257): which sampled frames are likely to improve a
place's map, and the bounded store of the ones kept. docs/SPATIAL.md has the design.

**What "likely to improve" is measured by, before a frame has a pose.** Each frame gets MegaLoc's place-recognition
descriptor (`Describer`: the pinned ONNX export on the CPU, [models.files.megaloc]), and its likeness to a kept
frame is their cosine similarity. Measured read only on six live cameras on the Linux machine (0.26 s a frame on 4
threads):

- a still camera's frames 5 s apart: 0.95 to 0.99 alike (54 pairs);
- the same view one to three hours apart, as the daylight moved: 0.61 to 0.83 (9 pairs);
- two different cameras: at most 0.18 (15 pairs).

So `decide` reads a frame's highest likeness to the kept frames:

- under `SAME_FRAME` (0.85): **keep**. Under `SAME_VIEW` (0.5) too it's a view the store hasn't got (a new viewpoint
  or an uncovered area); between them it's a kept view that now looks different (the light, or what's in the room).
- at least `SAME_FRAME`, and `SHARPER` (1.5) times as sharp as the kept frame it matches: **replace** that frame
  (better detail).
- otherwise: **drop**. It's a copy of what's kept.

The map step measures a kept frame a second time, against the map itself, once it has a pose (#258).

**The store** is `state/surroundings/`: `index.json` and, per frame, `<id>.jpg`, `<id>.npy` (its descriptor), and
`<id>.covered.png` when someone was in view (the pixels the map step leaves out).
It never holds more than `MAX_KEPT` frames, or more than `PER_VIEW` frames of one view (the frames at least
`SAME_VIEW` alike): another look past that pushes out its oldest, so a view's kept frames follow how it looks now.
At `MAX_KEPT` the frame most like another kept frame goes first, the older of a tie. All of it is machine-made:
`pack-reset-catalog` deletes it (unless `--keep-places`), and `pack-reset-places` leaves it, so the next map is
made from the same frames.

One process writes the store (the sampler); anything may read it. The index is replaced whole, after the files it
names are written, so a reader never sees a frame without its files.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from liberational import config as cfg
from liberational import json_state

DIR = "surroundings"  # under paths.state
SAME_FRAME = 0.85  # at least this alike: a copy of a kept frame
SAME_VIEW = 0.5  # at least this alike: the same view, though it may look different now
SHARPER = 1.5  # times a kept frame's sharpness a copy needs to replace it
MAX_KEPT = 300  # frames, about 45 MB of JPEGs at 960 x 540 and 10 MB of descriptors
PER_VIEW = 16
JPEG_QUALITY = 92

SIDE = 322  # MegaLoc's input, square
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet's, RGB
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THREADS = 4

Reason = Literal["view", "changed", "sharper", "copy"]


def root(state: Path | None = None) -> Path:
    return (cfg.path("paths.state") if state is None else state) / DIR


class _Session(Protocol):
    def run(self, output_names: Any, input_feed: Any, /) -> Any: ...
    def get_inputs(self) -> Any: ...


class Describer:
    """MegaLoc descriptors of BGR frames, unit length. The model is loaded on the first frame, so a sampler that's
    switched off, or looking at a still scene, never loads it. Raises RuntimeError when the model isn't installed."""

    def __init__(self, model: Path | None = None, session: _Session | None = None) -> None:
        self._model = model
        self._session = session

    def _load(self) -> _Session:
        import onnxruntime as ort

        model = cfg.model_path("megaloc") if self._model is None else self._model
        if not model.exists():
            raise RuntimeError(f"MegaLoc isn't at {model}; run `mise run models`")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1
        return ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])

    def __call__(self, pixels: np.ndarray) -> np.ndarray:
        import cv2

        if self._session is None:
            self._session = self._load()
        rgb = cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        name = self._session.get_inputs()[0].name
        d = np.asarray(self._session.run(None, {name: x})[0], dtype=np.float32).ravel()
        return d / max(float(np.linalg.norm(d)), 1e-12)

    def unload(self) -> None:
        """Free the model (about 460 MB) while nothing is being sampled."""
        self._session = None


@dataclass(frozen=True)
class Decision:
    """What the gate made of a frame. `keep` says whether it's stored; `reason` why (or "copy" for a drop);
    `alike` is its highest likeness to a kept frame (0 with none kept) and `nearest` that frame's id; `replaces` is
    the kept frame a sharper copy takes the place of."""

    keep: bool
    reason: Reason
    alike: float
    nearest: str | None = None
    replaces: str | None = None


@dataclass(frozen=True)
class Kept:
    """A kept frame: which camera took it, when (Unix seconds), how sharp it was, why it was kept, and its size.
    What the map step makes of it (its place and pose) is that step's to keep (spatial/surroundings_map.py)."""

    id: str
    camera: str
    t: float
    sharpness: float
    reason: str
    width: int
    height: int


FIELDS = tuple(f.name for f in fields(Kept))


def decide(descriptor: np.ndarray, sharp: float, kept: Sequence[Kept], descriptors: np.ndarray) -> Decision:
    """The gate: whether a frame with this unit `descriptor` and sharpness adds to `kept`, whose descriptors are
    the rows of `descriptors` in the same order."""
    if not kept:
        return Decision(True, "view", 0.0)
    alike = np.asarray(descriptors, dtype=np.float32) @ np.asarray(descriptor, dtype=np.float32)
    best = int(np.argmax(alike))
    top, near = float(alike[best]), kept[best]
    if top < SAME_VIEW:
        return Decision(True, "view", top, near.id)
    if top < SAME_FRAME:
        return Decision(True, "changed", top, near.id)
    if sharp >= SHARPER * near.sharpness:
        return Decision(True, "sharper", top, near.id, replaces=near.id)
    return Decision(False, "copy", top, near.id)


def most_redundant(kept: Sequence[Kept], descriptors: np.ndarray) -> int:
    """The index of the kept frame most like another kept frame, the older of the two that tie (and of any other
    tie): the one whose loss costs the store least. Needs at least two frames."""
    alike = np.asarray(descriptors, dtype=np.float32) @ np.asarray(descriptors, dtype=np.float32).T
    np.fill_diagonal(alike, -np.inf)
    top = alike.max(axis=1)
    return min(range(len(kept)), key=lambda i: (-float(top[i]), kept[i].t))


class Store:
    """The kept frames in `folder`, loaded once and then changed only through this object."""

    def __init__(self, folder: Path, most: int = MAX_KEPT, per_view: int = PER_VIEW) -> None:
        self.folder = folder
        self.most = most
        self.per_view = per_view
        self.kept: list[Kept] = []
        self._rows: list[np.ndarray] = []
        for row in json_state.read(folder / "index.json", []) or []:
            try:
                frame = Kept(**{k: row[k] for k in FIELDS})  # a field of an earlier version is passed by
                descriptor = np.load(folder / f"{frame.id}.npy")
            except TypeError, OSError, ValueError:
                continue  # a row from another version, or a frame whose files are gone: not kept
            if (folder / f"{frame.id}.jpg").is_file():
                self.kept.append(frame)
                self._rows.append(np.asarray(descriptor, dtype=np.float32).ravel())

    @property
    def descriptors(self) -> np.ndarray:
        return np.stack(self._rows) if self._rows else np.zeros((0, 0), dtype=np.float32)

    def path(self, frame: Kept | str) -> Path:
        return self.folder / f"{frame if isinstance(frame, str) else frame.id}.jpg"

    def covered_path(self, frame: Kept | str) -> Path:
        """Where a kept frame's covered pixels are (a PNG its size, white where covered), when it has any."""
        return self.folder / f"{frame if isinstance(frame, str) else frame.id}.covered.png"

    def decide(self, descriptor: np.ndarray, sharp: float) -> Decision:
        return decide(descriptor, sharp, self.kept, self.descriptors)

    def offer(
        self,
        camera: str,
        pixels: np.ndarray,
        descriptor: np.ndarray,
        sharp: float,
        now: float | None = None,
        covered: np.ndarray | None = None,
    ) -> tuple[Decision, Kept | None]:
        """Put a sampled frame through the gate, and store it when it's kept; the decision and the stored frame.
        `covered` (H x W bool) marks what in it isn't the place (someone in view): it's kept beside the frame
        (`covered_path`), for the map step to leave out."""
        decision = self.decide(descriptor, sharp)
        if not decision.keep:
            return decision, None
        gone: set[str] = set()
        if decision.replaces is not None:
            gone.add(decision.replaces)
        elif decision.reason == "changed":
            # The same view as some kept frames: at most `per_view` of them stay, the oldest going first.
            alike = self.descriptors @ np.asarray(descriptor, dtype=np.float32)
            view = sorted((f for f, a in zip(self.kept, alike, strict=True) if a >= SAME_VIEW), key=lambda f: f.t)
            gone.update(f.id for f in view[: max(0, len(view) + 1 - self.per_view)])
        frame = self._write(camera, pixels, descriptor, sharp, decision.reason, time.time() if now is None else now)
        if covered is not None and covered.any():
            import cv2

            cv2.imwrite(str(self.covered_path(frame)), covered.astype(np.uint8) * 255)
        self._drop(gone)
        while len(self.kept) > self.most:
            evicted = self.kept[most_redundant(self.kept, self.descriptors)].id
            self._drop({evicted})
            gone.add(evicted)
        self._save()
        self._unlink(gone)
        return decision, frame

    def remove(self, frame_id: str) -> None:
        """Drop a kept frame (one the map step couldn't place, or found the map already draws)."""
        self._drop({frame_id})
        self._save()
        self._unlink({frame_id})

    def _write(
        self, camera: str, pixels: np.ndarray, descriptor: np.ndarray, sharp: float, reason: str, now: float
    ) -> Kept:
        import cv2

        self.folder.mkdir(parents=True, exist_ok=True)
        frame = Kept(uuid.uuid4().hex, camera, now, sharp, reason, int(pixels.shape[1]), int(pixels.shape[0]))
        ok, jpeg = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise ValueError("the frame doesn't encode as a JPEG")
        row = np.asarray(descriptor, dtype=np.float32).ravel()
        self.path(frame).write_bytes(jpeg.tobytes())
        np.save(self.folder / f"{frame.id}.npy", row)
        self.kept.append(frame)
        self._rows.append(row)
        return frame

    def _drop(self, ids: set[str]) -> None:
        keep = [i for i, f in enumerate(self.kept) if f.id not in ids]
        self.kept = [self.kept[i] for i in keep]
        self._rows = [self._rows[i] for i in keep]

    def _unlink(self, ids: set[str]) -> None:
        for frame_id in ids:
            for suffix in ("jpg", "npy", "covered.png", "depth.png"):  # the last is the map step's, of this frame
                (self.folder / f"{frame_id}.{suffix}").unlink(missing_ok=True)

    def _save(self) -> None:
        json_state.write(self.folder / "index.json", [asdict(f) for f in self.kept])
