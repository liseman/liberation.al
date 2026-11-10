"""Opt-in counter that scores the camera-local movement tracks (#176, #1262) against hand-labeled entries and exits.

Run from the repository with an explicit matching tolerance:
  uv run python -m tools.measure --part entries --entry-counter tools.measure_movement_entries:counter --entry-tolerance 2

Predictions are the supported image-edge entries and exits that `memory/movement_counts.py` answers from, at the
middle of each edge's time estimate. They are image-edge candidates, not doorway claims. Track birth, disappearance,
camera motion, tracking gaps, and edges censored by the query boundary are never predictions: the reader reports
them as unknown, and they go to the scorer as unknown spans, which never erase a labeled miss. A read that fails is
an error, not an empty answer.
"""

from pathlib import Path
from typing import Any

from liberational.memory import movement_counts


def _unknown(items: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    """The reader's unknown intervals as the scorer's spans: clipped to [start, end), with width, not overlapping."""
    spans = sorted(
        (max(start, float(u["start"])), min(end, float(u["end"])), "; ".join(u.get("reasons") or [])) for u in items
    )
    out: list[dict[str, Any]] = []
    for lo, hi, reason in spans:
        if hi <= lo:
            continue
        if out and lo <= out[-1]["end"]:
            last = out[-1]
            last["end"] = max(last["end"], hi)
            reasons = [*last["reason"].split("; "), *reason.split("; ")]
            last["reason"] = "; ".join(sorted({r for r in reasons if r}))
        else:
            out.append({"start": lo, "end": hi, "reason": reason})
    return out


def counter(*, camera: str, start: float, end: float, state: Path) -> dict[str, Any]:
    """Return the #81 scorer contract without receiving any ground-truth labels."""
    result = movement_counts.collect(state, start=start, end=end, cameras=[camera])
    data = result["cameras"].get(camera)
    if result["errors"] or data is None:
        raise RuntimeError("movement tracks unavailable: " + ", ".join(result["errors"] or ["no camera record"]))
    events = [
        {"kind": event["kind"], "at": (event["time"][0] + event["time"][1]) / 2}
        for event in [*data["entries"], *data["exits"]]
    ]
    return {
        "events": sorted(events, key=lambda e: (e["at"], e["kind"])),
        "supported_kinds": ["entry", "exit"],
        "unknown": _unknown(data["unknown"], start, end),
        "diagnostics": {
            "method": result["method"],
            "event_kind_semantics": "supported image-edge entries and exits of camera-local tracks",
            "supported_s": data["supported_s"],
            "uncertain_events": len(data["uncertain_events"]),
            "coarrivals": len(data["coarrivals"]),
            "tracks": len(data["tracks"]),
            "source_ids": data["source_ids"],
        },
    }
