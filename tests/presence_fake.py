"""Presence test data as see-actions and see-zones would write it (#1825): frames run through the real episode rules
(memory/presence_episodes.py), their `presence.span` rows and the frames' `action.coverage` added to the suite's fake
segment log, which presence_counts reads."""

from collections.abc import Iterable
from typing import Any

from liberational import config as cfg
from liberational import segment_log
from liberational.memory.presence_episodes import Episodes

FRAME_S = 0.5  # the coverage one analyzed frame stands for


def add_frames(vision: Any, frames: Iterable[tuple[float, str, bool]]) -> None:
    """Frames as (time, camera, someone tracked): each is analyzed time, and each camera's episodes are written as
    see-zones writes them, the last one closed as if nobody came after it."""
    by_cam: dict[str, list[tuple[float, bool]]] = {}
    for at, cam, positive in frames:
        by_cam.setdefault(cam, []).append((at, positive))
    for cam, mine in by_cam.items():
        mine.sort()
        episodes = Episodes(cam, absence_s=cfg.get("presence.absence_s"), gap_s=cfg.get("presence.gap_s"))
        ended: list[dict[str, Any]] = []
        for at, positive in mine:
            vision.add(segment_log.ACTION_COVERAGE, ts_start=at, ts_end=at + FRAME_S, cam=cam, frames=1, model="test")
            ended += episodes.frame(at, ["1"] if positive else [])
        if episodes.open is not None:
            ended.append(episodes._close(right_censored=False))
        for e in ended:
            vision.add(segment_log.PRESENCE_SPANS, **e)
