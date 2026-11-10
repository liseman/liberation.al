"""Captions on Continuum (#549): each camera's view described by the VLM when it has changed, as `camera.caption`
records, and its notable objects grounded now and then, as `camera.objects` records.

`CaptionPipeline` is one camera's path, synchronous so it tests without a broker, a camera, or a model: `feed` a frame
(its pixels, time, and sequence) and get back what it produced. Around v1's models and rules, which it uses as they
are (`vision/loop.py`, `vision/objects.py`):

- The change gate: a 64x36 blurred grey signature of the 448-px tile, compared with the tile at the last VLM call (not
  the last frame, so slow drift adds up). The VLM is called when the mean difference, the share of pixels that
  moved, or one block of the view changed enough, or when the view has gone `max_idle_s` without a caption.
- The VLM: any `Captioner` (v1's `OllamaVLM`, `MlxVLM`, and `MlxServerVLM` all fit), given the tile and this camera's
  last three real captions, with v1's independent-observation prompt.
- Grounding: an optional `ground` call (v1's Ollama grounder) on the whole frame, on a view's first look, when the view
  moved (at most every `objects.MIN_S`; the gate's idle refresh isn't movement), and otherwise every
  `objects.MAX_IDLE_S`, its boxes kept for `objects.TTL_S` and
  carried on the caption records in between.

Animals (#1988): with `animals` set, each frame's fresh grounding gives its animal boxes, and a fresh caption that
names an animal asks the VLM for them (see/caption_animals.py), for pet sightings.

A caption the gate skipped is committed as a reference to the last real one (`unchanged`, `since`), and a failed VLM
call as an `error` observation, each at most every `quiet_every_s`: one row per tick, as v1 kept, would be most of
the stream. Faces aren't checked here (`faces` is None): the face worker's records hold them.
"""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np

from liberational import logs
from liberational.see.caption_animals import CaptionAnimals, Found
from liberational.vision import objects

TILE_PX = 448
PROMPT_VERSION = "independent-v1"
# v1's gate thresholds (vision/loop.py, measured on steady workshop frames: mad about 1.0, frac about 0.01).
GATE_MAD = 2.5
GATE_FRAC = 0.03
GATE_LOCAL = 18.0
GATE_MAX_IDLE_S = 30.0
QUIET_EVERY_S = 10.0
MEMORY = 3
MAX_GROUNDING_BYTES = 65_536
MAX_GROUNDING_OBJECTS = 64

LOG = logs.get("see-captions")


def grounded_items(text: str) -> list[objects.Item]:
    """A complete finite grounding array; malformed/truncated responses are unknown, never an empty scene.

    Keep the legacy box normalization/filtering after validating the whole model response. Code fences and
    surrounding prose remain accepted, but every array row must have a finite box and a real label.
    """
    if not isinstance(text, str) or len(text.encode()) > MAX_GROUNDING_BYTES:
        raise ValueError("grounding response exceeds its text budget")
    start = text.find("[")
    if start < 0 or "{" in text[:start]:
        raise ValueError("grounding response has no array root")

    def constant(value: str) -> None:
        raise ValueError(f"grounding contains nonfinite value {value}")

    rows, _ = json.JSONDecoder(parse_constant=constant).raw_decode(text[start:])
    if not isinstance(rows, list) or len(rows) > MAX_GROUNDING_OBJECTS:
        raise ValueError("grounding response exceeds its object budget")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("label"), str) or not row["label"].strip():
            raise ValueError("grounding object has no label")
        box = row.get("bbox_2d", row.get("bbox"))
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("grounding object has no complete box")
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in box):
            raise ValueError("grounding object has a nonfinite box")
    return objects.parse(json.dumps(rows, allow_nan=False))


class Captioner(Protocol):
    """A VLM backend: `caption(tile, memory)` gives the description, or "" when the call failed."""

    name: str

    def caption(self, img: np.ndarray, memory: Sequence[str], /) -> str: ...


def tile_of(frame: np.ndarray, px: int = TILE_PX) -> np.ndarray:
    """The frame scaled so its longer side is `px` (v1's `downscale`), never up."""
    h, w = frame.shape[:2]
    s = px / max(h, w)
    if s >= 1:
        return frame
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def signature(tile: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
    g = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)


@dataclass
class Change:
    """The gate's look at a tile: whether to call the VLM, and how much it moved (None for a view's first tile)."""

    call: bool
    mad: float | None = None
    frac: float | None = None
    moved: bool = True  # the view itself changed (or it's the first look); False for the gate's idle refresh


class ChangeGate:
    """v1's change gate for one view."""

    def __init__(
        self,
        mad: float = GATE_MAD,
        frac: float = GATE_FRAC,
        local: float = GATE_LOCAL,
        max_idle_s: float = GATE_MAX_IDLE_S,
    ) -> None:
        self.mad, self.frac, self.local, self.max_idle_s = mad, frac, local, max_idle_s
        self.sig: np.ndarray | None = None
        self.ts = 0.0

    def look(self, tile: np.ndarray, ts: float) -> Change:
        if self.sig is None:
            return Change(True)
        d = np.abs(signature(tile) - self.sig)
        mad, frac = float(d.mean()), float((d > 25).mean())
        # A small person, hand, or fixture can change locally without moving 3% of the whole view.
        local = max(float(b.mean()) for row in np.array_split(d, 4) for b in np.array_split(row, 8, axis=1))
        idle = self.max_idle_s > 0 and ts - self.ts >= self.max_idle_s
        moved = mad >= self.mad or frac >= self.frac or local >= self.local
        return Change(moved or idle, mad, frac, moved)

    def captioned(self, tile: np.ndarray, ts: float) -> None:
        self.sig, self.ts = signature(tile), ts


@dataclass
class CaptionFrame:
    """What one frame produced: a `camera.caption` record (None when a quiet one was held back) and a `camera.objects`
    record when the frame was grounded."""

    caption: dict[str, Any] | None = None
    objects: dict[str, Any] | None = None
    animals: list[Found] = field(default_factory=list)  # its animal boxes (see/caption_animals.py)
    grounding_decision: str | None = None  # Internal scheduling witness; never serialized as observation evidence.


class CaptionPipeline:
    """One camera's captions. `ground(frame) -> model text` grounds objects (None leaves them out)."""

    def __init__(
        self,
        cam: str,
        vlm: Captioner,
        *,
        ground: Callable[[np.ndarray], str] | None = None,
        gate: ChangeGate | None = None,
        quiet_every_s: float = QUIET_EVERY_S,
        caption_every_s: float = 0.0,
        animals: CaptionAnimals | None = None,
    ) -> None:
        self.cam, self.vlm, self.ground, self.animals = cam, vlm, ground, animals
        # The camera's caption interval (#1641): a change within this long of the last VLM call is deferred until it
        # passes. Grounding keeps its own cadence.
        self.caption_every_s = caption_every_s
        self.called_at = -float("inf")
        self.gate = gate or ChangeGate()
        self.quiet_every_s = quiet_every_s
        self.memory: list[str] = []
        self.last: dict[str, Any] | None = None  # the last real caption: its text and observed time
        self.last_quiet = -float("inf")
        self.grounded_at: float | None = None
        self.items: list[objects.Item] = []
        self.items_ts = 0.0
        self.has_objects_observation = False
        self.grounding_warned: set[str] = set()  # failure reasons already logged

    def feed(self, frame: np.ndarray, ts: float, frame_seq: int) -> CaptionFrame:
        tile = tile_of(frame)
        change = self.gate.look(tile, ts)
        out = CaptionFrame()
        if self.ground is None:
            out.grounding_decision = "disabled"
        elif self._ground_due(change.moved, ts):
            out.objects = self._ground(frame, ts, frame_seq)
            grounding_status = out.objects.get("observation", {}).get("status")
            if grounding_status in {"fresh", "failed"}:
                out.grounding_decision = grounding_status
        else:
            out.grounding_decision = "not_scheduled"
        text, vlm_ms = "", 0.0
        deferred = change.call and ts - self.called_at < self.caption_every_s
        if deferred:
            change = Change(False, change.mad, change.frac, change.moved)
        if change.call:
            self.called_at = ts
            t0 = cv2.getTickCount()
            text = self.vlm.caption(tile, self.memory[-MEMORY:])
            vlm_ms = 1000 * (cv2.getTickCount() - t0) / cv2.getTickFrequency()
            if text:
                self.gate.captioned(tile, ts)
                self.memory = [*self.memory[-(MEMORY - 1) :], text]
                self.last = {"caption": text, "at": ts}
        if self.animals is not None:
            grounded = out.objects is not None and out.objects["observation"]["status"] == "fresh"
            out.animals = self.animals.look(frame, ts, text, self.items if grounded else None)
        status = "fresh" if change.call and text else "error" if change.call else "not_observed"
        if status != "fresh":
            if ts - self.last_quiet < self.quiet_every_s:
                return out
            self.last_quiet = ts
        out.caption = self._record(tile, ts, frame_seq, change, status, text, vlm_ms)
        if deferred:
            out.caption["deferred"] = True
        if out.objects is None:
            current = self.has_objects_observation and 0 <= ts - self.items_ts <= objects.TTL_S
            status = "cached" if self.ground is not None and current else "unevaluated"
            reason = "cached_grounding" if current else "expired_or_missing_grounding"
            if self.ground is None:
                reason = "grounding_disabled"
            out.objects = self._object_record(frame, ts, frame_seq, status, reason=reason)
        return out

    def _record(
        self, tile: np.ndarray, ts: float, frame_seq: int, change: Change, status: str, text: str, vlm_ms: float
    ) -> dict[str, Any]:
        g = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
        record: dict[str, Any] = {
            "ts_start": ts,
            "cam": self.cam,
            "frame_seq": frame_seq,
            "yaw_deg": 0.0,
            "caption": text,
            "faces": None,  # not checked here: the face worker's records hold them
            "objects": self._objects(tile, ts),
            "source": "camera",
            "synthetic": False,
            "gated": not change.call,
            "observation": {
                "version": 1,
                "status": status,
                "observed_at": ts if status == "fresh" else None,
                "camera": self.cam,
                "backend": self.vlm.name,
                "prompt_version": PROMPT_VERSION,
            },
            "change": None
            if change.mad is None or change.frac is None
            else {"mad": round(change.mad, 2), "frac": round(change.frac, 3)},
            "brightness": {"mean": round(float(g.mean()), 1), "dark_fraction": round(float((g < 40).mean()), 3)},
            "ms": {"vlm": round(vlm_ms)},
        }
        if status == "not_observed" and self.last is not None:
            # The view hasn't changed: its last real caption, marked unchanged since then.
            record.update(caption=self.last["caption"], unchanged=True, since=self.last["at"])
        return record

    def _ground_due(self, moved: bool, ts: float) -> bool:
        """Grounding is the VLM's costliest call (about 2 s), so a view is grounded when it first appears, when it
        moved (at most every `objects.MIN_S`), and otherwise every `objects.MAX_IDLE_S`. The change gate's idle
        refresh isn't movement: counting it grounded still views every ~35 s, most of the VLM's time (#1935)."""
        last = self.grounded_at
        return last is None or (moved and ts - last >= objects.MIN_S) or ts - last >= objects.MAX_IDLE_S

    def _object_record(
        self,
        frame: np.ndarray,
        ts: float,
        frame_seq: int,
        status: str,
        *,
        attempted_at: float | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        current = self.has_objects_observation and 0 <= ts - self.items_ts <= objects.TTL_S
        observation: dict[str, Any] = {
            "version": 1,
            "status": status,
            "observed_at": self.items_ts if self.has_objects_observation else None,
            "attempted_at": attempted_at,
        }
        if reason is not None:
            observation["reason"] = reason
        return {
            "ts_start": ts,
            "cam": self.cam,
            "frame_seq": frame_seq,
            "objects": objects.to_pixels(self.items, frame.shape, self.items_ts, norm=True) if current else [],
            "source": self.vlm.name,
            "observation": observation,
        }

    def _ground(self, frame: np.ndarray, ts: float, frame_seq: int) -> dict[str, Any]:
        assert self.ground is not None
        self.grounded_at = ts
        try:
            items = grounded_items(self.ground(frame))
        except Exception as error:  # Unknown observation: keep cached geometry and its original observation time.
            reason = getattr(error, "reason", "grounding_failed")
            if reason not in self.grounding_warned:  # once per kind of failure, so a broken grounding is seen (#1454)
                self.grounding_warned.add(reason)
                LOG.warning("%s: object grounding failed (%s): %s", self.cam, reason, error)
            return self._object_record(frame, ts, frame_seq, "failed", attempted_at=ts, reason=reason)
        self.items, self.items_ts = items, ts
        self.has_objects_observation = True
        return self._object_record(frame, ts, frame_seq, "fresh", attempted_at=ts)

    def _objects(self, tile: np.ndarray, ts: float) -> list[dict[str, Any]] | None:
        """The last grounded boxes in the tile's pixels (and the frame's share), dropped after `objects.TTL_S`; None
        when this pipeline doesn't ground."""
        if self.ground is None:
            return None
        if ts - self.items_ts > objects.TTL_S:
            return []
        return objects.to_pixels(self.items, tile.shape, self.items_ts, norm=True)
