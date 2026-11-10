"""The object worker on Continuum: finds the objects remembered on the Objects page in the camera video, and
writes where it saw them as `object.match` rows in the segment store (`memory.objects.match_record`).

Per camera, the newest frame every `interval_s` is matched against every object's
template (`memory.objects.match`: SIFT features and a RANSAC homography, on the CPU). A match counts once a second,
distinct frame within CONFIRM_S puts the object in the same place; a confirmed sighting is written with the frame as
its evidence and synced at once (matches are rare), at most once per REPEAT_S for an object that hasn't moved. The
objects are read again every REFRESH_S, so one remembered or forgotten on the page is picked up without a restart.

With no object taught, the worker waits and decodes nothing: it reads the store every REFRESH_S, starts
following the cameras once one is taught, and goes back to waiting when every one is forgotten.
"""

import asyncio
import base64
import concurrent.futures
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from liberational import continuum_link, logs, segment_log
from liberational.memory import objects
from liberational.see import frames
from liberational.see.frames import NS, Camera, cameras_from_manifest, check_h264_decoder

if TYPE_CHECKING:
    from continuum_py import Client

log = logs.get("see-objects")

CONFIRM_S = 20.0  # a match is confirmed by another, in the same place, within this long
REPEAT_S = 60.0  # an object seen again in the same place is committed again only after this long
REFRESH_S = 30.0  # how often the remembered objects are read again


@dataclass
class Template:
    id: str
    name: str
    features: tuple[Any, Any, tuple[int, int]]


@dataclass
class Seen:
    """An object's last match on a camera, waiting for its confirmation, and its last committed sighting."""

    ts: float = 0.0
    bbox: list[float] | None = None
    frame_seq: int | None = None
    committed_at: float = -1e18
    committed_bbox: list[float] | None = None


def templates(rows: Sequence[dict[str, Any]]) -> list[Template]:
    """The visual objects' templates, decoded with their features computed once."""
    out = []
    for o in rows:
        if o.get("mode") != "visual":
            continue
        image = objects.decode(base64.b64decode(o["template"]))
        if image is None:
            continue
        kp, desc = objects.features(image, objects.template_mask(o, image))
        out.append(Template(o["id"], o["name"], (kp, desc, image.shape[:2])))
    return out


class ObjectWorker:
    """Follows every camera in `cameras` until `stop` is set, matching the store's objects in their frames."""

    def __init__(
        self,
        client: Client,
        cameras: Sequence[Camera],
        store: objects.Store,
        *,
        interval_s: float = 5.0,
        matches: segment_log.Writers | None = None,
    ) -> None:
        self.client, self.cameras, self.store, self.interval_s = client, list(cameras), store, interval_s
        # One writer per camera, used only from the matching thread.
        self.matches = matches if matches is not None else segment_log.Writers(segment_log.root(store.state))
        self.templates: list[Template] = []
        self.read_at = -1e18
        self.seen: dict[tuple[str, str], Seen] = {}
        self.committed = 0
        # SIFT on the CPU, one frame at a time: the other workers keep the GPU and the rest of the CPU.
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="see-objects")

    def refresh(self, now: float) -> None:
        if now - self.read_at < REFRESH_S:
            return
        self.read_at = now
        try:
            self.templates = templates(self.store.objects())
        except Exception as e:  # the event log out of reach: keep the objects we had
            log.warning("objects not read again: %s", e)

    def observe(self, cam: str, image: Any, ts: float, frame_id: str, frame_seq: int | None) -> int:
        """Matches every object in one frame; commits the confirmed sightings and returns how many."""
        self.refresh(time.time())
        if not self.templates:
            return 0
        frame = objects.features(image)
        frame_f = (frame[0], frame[1], image.shape[:2])
        n = 0
        for t in self.templates:
            s = self.seen.setdefault((t.id, cam), Seen())
            found = objects.match(t.features, frame_f)
            if found is None:
                s.bbox = None
                continue
            bbox, score = found
            confirmed = (
                s.bbox is not None
                and 0 < ts - s.ts <= CONFIRM_S
                and s.frame_seq != frame_seq
                and objects.consistent(s.bbox, bbox)
            )
            s.ts, s.bbox, s.frame_seq = ts, bbox, frame_seq
            if not confirmed:
                continue
            moved = s.committed_bbox is None or not objects.consistent(s.committed_bbox, bbox)
            if not moved and ts - s.committed_at < REPEAT_S:
                continue
            record = objects.match_record(
                object_id=t.id,
                name=t.name,
                cam=cam,
                at=ts,
                bbox=bbox,
                score=score,
                frame=image,
                frame_id=frame_id,
                frame_seq=frame_seq,
            )
            self.matches.append(segment_log.OBJECT_MATCHES, cam, [record])
            self.matches.sync()
            s.committed_at, s.committed_bbox = ts, bbox
            n += 1
        self.committed += n
        return n

    async def _camera(self, cam: Camera) -> None:
        loop = asyncio.get_running_loop()
        async for image, header in frames.newest(self.client, cam, self.interval_s):
            ts = header["start_ns"] / NS
            seq = (header.get("capture") or {}).get("device_seq")
            n = await loop.run_in_executor(self._pool, self.observe, cam.name, image, ts, header["id"], seq)
            if n:
                log.info("%s: %d sighting(s) at %.0f", cam.name, n, ts)

    async def _taught(self, stop: asyncio.Event) -> bool:
        """Waits until an object is taught (the store is read every REFRESH_S): True then, False when `stop` is set
        first. Until then nothing is decoded or matched."""
        said = False
        while not stop.is_set():
            self.read_at = -1e18  # read now, whatever refresh last did
            await asyncio.to_thread(self.refresh, time.time())
            if self.templates:
                return True
            if not said:
                log.info("no object taught yet: waiting until one is taught on the Objects page")
                said = True
            try:
                await asyncio.wait_for(stop.wait(), REFRESH_S)
            except TimeoutError:
                pass
        return False

    async def _forgotten(self) -> None:
        """Returns once every object is forgotten (the store read every REFRESH_S)."""
        while True:
            await asyncio.sleep(REFRESH_S)
            await asyncio.to_thread(self.refresh, time.time())
            if not self.templates:
                return

    async def run(self, stop: asyncio.Event) -> None:
        """Follows the cameras while at least one object is taught, and waits, decoding nothing, while none is."""
        try:
            while await self._taught(stop):
                log.info(
                    "%d object(s) taught: watching %s", len(self.templates), ", ".join(c.stream for c in self.cameras)
                )
                cameras = [asyncio.create_task(self._camera(cam)) for cam in self.cameras]
                forgotten = asyncio.create_task(self._forgotten())
                stopping = asyncio.create_task(stop.wait())
                done, _ = await asyncio.wait([stopping, forgotten, *cameras], return_when=asyncio.FIRST_COMPLETED)
                for task in (stopping, forgotten):
                    task.cancel()
                if forgotten in done:
                    for task in cameras:
                        task.cancel()
                    await asyncio.gather(*cameras, return_exceptions=True)
                    log.info("every object is forgotten: waiting again")
                    continue
                await frames.until_stopped(stop, cameras)  # stopped, or a camera's end raised as before
                return
        finally:
            self._pool.shutdown(wait=True, cancel_futures=True)  # no match is mid-write when the writers close
            self.matches.close()


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-see-objects`: the object worker as a service. It reads the cameras in capture's manifest from the broker
    (run continuum-broker and continuum-capture with it), and the objects remembered on the Objects page."""
    import argparse

    from liberational import config as cfg

    parser = argparse.ArgumentParser(prog="pack-see-objects", description=main.__doc__)
    parser.add_argument("--manifest", type=Path, default=cfg.devices_file(), help="capture's device manifest")
    parser.add_argument("--socket", type=Path, default=None, help="the broker's socket (default: under paths.state)")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between a camera's frames looked at")
    args = parser.parse_args(argv)

    cameras = cameras_from_manifest(args.manifest)
    if not cameras:
        raise SystemExit(f"no cameras in {args.manifest}")
    check_h264_decoder(cameras)
    cv2.setNumThreads(1)

    async def run() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        async with continuum_link.connected(args.socket, client_name="see-objects") as client:
            worker = ObjectWorker(
                client, cameras, objects.Store(cfg.path("paths.state")), interval_s=max(1.0, args.interval)
            )
            await worker.run(stop)

    continuum_link.run_service(run())


if __name__ == "__main__":
    main()
