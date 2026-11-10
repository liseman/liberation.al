"""Detection log events as v1's backfill wrote them (its _face_events and _empty_run_event, kept here for the tests
of the detection log's readers after the backfill was removed)."""

import json
import math
import time
import uuid
from typing import Any

Event = dict[str, Any]


def make_event(
    event_type: str,
    *,
    occurred_at: float,
    stream: str,
    payload: Any,
    source: dict[str, Any],
    evidence: Any = None,
    embedding: Any = None,
    event_id: str | None = None,
) -> Event:
    """One event in v1's detector-journal shape, which the presence projection (memory/presence.py) reads: the
    journal (memory/detection_log.py) is deleted, so tests build its events here."""
    return {
        "event_id": event_id if event_id is not None else str(uuid.uuid4()),
        "type": event_type,
        "schema_version": 1,
        "occurred_at": float(occurred_at),
        "recorded_at": time.time(),
        "stream": stream,
        "source": json.loads(json.dumps(source)),
        "evidence": {} if evidence is None else json.loads(json.dumps(evidence)),
        "payload": json.loads(json.dumps(payload)),
        "embedding": embedding,
    }


def face_events(
    seg: dict[str, Any],
    version: str,
    idx: int,
    fps: float,
    source: dict[str, Any],
    width: int | None,
    height: int | None,
    hits: list[Any],
    error: Exception | None = None,
) -> list[Event]:
    """Live frame shape with recording evidence, no offline tracks or embeddings.

    FaceDetected adds raw det_score. Embeddings stay in sightings.sqlite3.
    """
    common = dict(cam=seg["cam"], frame_seq=idx, width=width, height=height)
    evidence = dict(segment=seg["file"], frame=idx, offset_s=idx / fps, scan_version=version)
    status = "error" if error else "evaluated"

    def event(kind: str, suffix: str, payload: dict[str, Any]) -> Event:
        return make_event(
            kind,
            occurred_at=seg["start"] + idx / fps,
            stream=f"camera:{seg['cam']}",
            source=source,
            payload=payload,
            evidence=evidence,
            event_id=f"face:{version}:{idx}:{suffix}",
        )

    events = [
        event(
            "FaceFrameObserved",
            status,
            dict(
                **common,
                status=status,
                face_count=None if error else len(hits),
                error=type(error).__name__ if error else None,
            ),
        )
    ]
    if not error:
        for number, hit in enumerate(hits):
            assert width and height
            events.append(
                event(
                    "FaceDetected",
                    f"detected:{number}",
                    dict(
                        **common,
                        bbox=[float(v) / size for v, size in zip(hit.bbox, (width, height, width, height))],
                        det_score=float(hit.det_score),
                    ),
                )
            )
    return events


def empty_run_event(
    family: str,
    seg: dict[str, Any],
    version: str,
    first_frame: int,
    last_frame: int,
    fps: float,
    frame_step: int,
    source: dict[str, Any],
    width: int,
    height: int,
) -> Event:
    """A checked sample grid, never continuous absence; readers union overlapping retry fragments.

    Runs flush before checkpoints, so a crash cannot acknowledge unjournalled
    coverage. Different retry budgets may produce overlapping immutable spans.
    Their integer sample grids (and legacy empty points) must be unioned by
    family/camera/segment/scan, never summed. No model or filesystem I/O here.
    """
    if (
        family not in ("face", "body")
        or first_frame < 0
        or last_frame < first_frame
        or frame_step < 1
        or not math.isfinite(fps)
        or fps <= 0
        or (last_frame - first_frame) % frame_step
    ):
        raise ValueError("invalid empty-run sample grid")
    payload = dict(
        cam=seg["cam"],
        width=width,
        height=height,
        status="evaluated",
        first_frame=first_frame,
        last_frame=last_frame,
        sample_count=(last_frame - first_frame) // frame_step + 1,
        frame_step=frame_step,
        fps=fps,
        sample_interval_s=frame_step / fps,
    )
    payload.update(dict(face_count=0, error=None) if family == "face" else dict(boxes=[]))
    return make_event(
        "FaceEmptyRunObserved" if family == "face" else "BodyEmptyRunObserved",
        occurred_at=seg["start"] + first_frame / fps,
        stream=f"camera:{seg['cam']}",
        source=source,
        payload=payload,
        evidence=dict(
            segment=seg["file"],
            frame=first_frame,
            offset_s=first_frame / fps,
            last_offset_s=last_frame / fps,
            scan_version=version,
        ),
        event_id=f"{family}-empty:{version}:{first_frame}:{last_frame}:{frame_step}",
    )
