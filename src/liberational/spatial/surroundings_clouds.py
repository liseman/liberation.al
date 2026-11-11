"""When a camera's own cloud is made again, and from which picture (#417). spatial/camera_clouds.py makes and keeps
the clouds; spatial/surroundings_worker.py calls `Clouds.look` with every picture it samples.

**When.** A camera gets a cloud when it has none, and a new one when its view has changed: the picture's 32 x 18
thumbnail (the sampler's own, which an exposure step doesn't move) is under `CHANGED` alike to the thumbnail of the
picture its cloud was made from. That one test covers a room that was rearranged and a camera that was moved or
turned, and it survives a restart, since the thumbnail is kept with the cloud. A camera with a cloud is checked every
`CHECK_EVERY_S`, and remade at most every `REFRESH_MIN_S`; a try that made nothing waits `RETRY_S`.

**From which picture.** One with nobody in it. While a camera is due, each of its sampled pictures goes through
see-actions' person and animal detector (the sampler's, on the CPU), and a picture with anyone in it starts the
count again. `CLEAN` pictures in a row with nobody in them, all alike (`STILL`), give one clean picture: their
per-pixel median. With nobody in view the scene is still, so the median only removes sensor noise and whatever
brief thing the detector missed. At a sample every 5 s that is 20 s of an empty view. Without the detector's model
a picture is taken to be empty, as the sampler takes it, and the log says so once.

The new cloud replaces the camera's cloud; nothing of the old one is carried forward. Making it takes ten seconds
or so (most of it loading the model), so it runs on a thread of its own, one cloud at a time, and the sampler's
looks carry on meanwhile.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Sequence

import numpy as np

from liberational import logs
from liberational.spatial import camera_clouds

LOG = logs.get("surroundings")

CHANGED = 0.9  # a picture under this alike to the cloud's own is another view; a still camera reads 0.99 and up
STILL = 0.98  # the clean pictures of one median are at least this alike to its first
CLEAN = 5  # pictures in a row with nobody in them
CHECK_EVERY_S = 30.0
REFRESH_MIN_S = 300.0
RETRY_S = 120.0

#: The people and animals in a BGR frame, as boxes.
Detector = Callable[[np.ndarray], Sequence[tuple[float, float, float, float]]]
#: `camera_clouds.refresh` with the store and the runner bound: (camera, picture, now, frames, thumbnail).
Refresh = Callable[[str, np.ndarray, float, int, Sequence[float]], object]


class Clouds:
    """Decides, a sampled picture at a time, whether a camera's cloud is due and which picture it's made from.
    `look` is called from the sampler's one thread; the cloud itself is made on `submit`'s."""

    def __init__(
        self,
        store: camera_clouds.Store,
        refresh: Refresh,
        detect: Detector | None,
        lit: Callable[[np.ndarray], bool],
        thumbnail: Callable[[np.ndarray], np.ndarray],
        configured: Callable[[], bool] = lambda: True,
        submit: Callable[[Callable[[], None]], object] | None = None,
    ) -> None:
        self.store = store
        self.refresh = refresh
        self.detect = detect
        self.lit = lit
        self.thumbnail = thumbnail
        self.configured = configured
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        if submit is None:
            pool = self._pool = concurrent.futures.ThreadPoolExecutor(1, "camera-cloud")
            self._submit: Callable[[Callable[[], None]], object] = lambda job: pool.submit(job)
        else:
            self._submit = submit
        self._clean: dict[str, list[np.ndarray]] = {}  # each due camera's clean pictures so far
        self._first: dict[str, np.ndarray] = {}  # and the thumbnail of the first of them
        self._checked: dict[str, float] = {}  # when a camera's view was last found unchanged
        self._wait: dict[str, float] = {}  # no try for a camera before this
        self._making: set[str] = set()

    def look(self, camera: str, pixels: np.ndarray, now: float) -> str:
        """Take one sampled picture of `camera`. What came of it: "off" (no runtime configured), "making" (its cloud
        is being made), "wait" (too soon after the last try), "fine" (its view is as its cloud shows it), "dark",
        "people", "collecting" (a clean picture, more wanted), or "made" (the clean picture is on its way to the
        model)."""
        if not self.configured():
            self.store.note(camera, "no_runtime", now)
            return "off"
        if camera in self._making:
            return "making"
        if now < self._wait.get(camera, 0.0):
            return "wait"
        have = self.store.current(camera)
        collecting = camera in self._clean
        if have is not None and not collecting and now - self._checked.get(camera, -CHECK_EVERY_S) < CHECK_EVERY_S:
            return "fine"
        if not self.lit(pixels):
            self._drop(camera)
            if have is None:
                self.store.note(camera, "dark", now)
            return "dark"
        thumb = self.thumbnail(pixels)
        if have is not None and len(have.thumb) == len(thumb) and float(np.asarray(have.thumb) @ thumb) >= CHANGED:
            self._drop(camera)
            self._checked[camera] = now
            return "fine"
        if self.detect is not None and self.detect(pixels):
            self._drop(camera)
            if have is None:
                self.store.note(camera, "people", now)
            return "people"
        first = self._first.get(camera)
        if first is None or float(first @ thumb) < STILL:
            self._clean[camera], self._first[camera] = [], thumb  # the view moved on: this picture starts the count
        self._clean[camera].append(np.asarray(pixels))
        if len(self._clean[camera]) < CLEAN:
            return "collecting"
        stack = self._clean[camera]
        self._drop(camera)
        picture = np.median(np.stack(stack), axis=0).astype(np.uint8)
        self._making.add(camera)
        self._submit(
            lambda: self._make(camera, picture, now, len(stack), tuple(float(v) for v in self.thumbnail(picture)))
        )
        return "made"

    def _make(self, camera: str, picture: np.ndarray, now: float, frames: int, thumb: Sequence[float]) -> None:
        try:
            made = self.refresh(camera, picture, now, frames, thumb)
        except Exception:  # a cloud that can't be made must not stop the sampling
            LOG.exception("%s: making its cloud failed", camera)
            made = None
        finally:
            self._making.discard(camera)
        self._wait[camera] = now + (REFRESH_MIN_S if made is not None else RETRY_S)
        self._checked[camera] = now

    def _drop(self, camera: str) -> None:
        self._clean.pop(camera, None)
        self._first.pop(camera, None)

    def forget(self, camera: str) -> None:
        """Start a camera's count afresh (it's being followed again after a break)."""
        self._drop(camera)
        self._checked.pop(camera, None)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)


def for_service(
    state: object,
    detect: Detector | None,
    lit: Callable[[np.ndarray], bool],
    thumbnail: Callable[[np.ndarray], np.ndarray],
) -> Clouds:
    """The service's `Clouds`: the pack's store and the configured runtime."""
    from pathlib import Path

    store = camera_clouds.Store(camera_clouds.root(Path(str(state))))

    def refresh(camera: str, picture: np.ndarray, now: float, frames: int, thumb: Sequence[float]) -> object:
        return camera_clouds.refresh(
            store, camera, picture, now, run=camera_clouds.runner(), frames=frames, thumb=thumb
        )

    return Clouds(store, refresh, detect, lit, thumbnail, configured=lambda: camera_clouds.runtime() is not None)
