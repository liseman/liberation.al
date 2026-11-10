"""Plain sentences about people walking by a camera, from crossing and movement counts (#1398, #1403): the day
summary's "People walking by" section, and Ask's and voice's answer to "how many times did someone walk by"."""

import collections
import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

WATCHED_ENOUGH = 0.95  # a camera watched this share of the day so far gets no "only watched for" note
PEAK_MIN = 5  # walk-bys in a day before the busiest hour is worth naming


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n} times")


def _watched(seconds: float) -> str:
    if seconds < 60:
        return "less than a minute"
    if seconds < 90 * 60:
        minutes = round(seconds / 60)
        return "1 minute" if minutes == 1 else f"{minutes} minutes"
    hours = round(seconds / 3600)
    return "about an hour" if hours == 1 else f"about {hours} hours"


def _peak(times: Sequence[float], tz: dt.tzinfo | None) -> str | None:
    """The local clock hour with the most walk-bys, named only when it clearly leads. Hours are the clock's, not
    24 equal slices of the day, so a 23 or 25 hour day (a daylight saving change) names the right one; the repeated
    hour of a fall-back day is two hours here, each counted on its own."""
    hours = collections.Counter()
    for ts in times:
        local = dt.datetime.fromtimestamp(ts, tz)
        hours[(local.date(), local.hour, local.fold)] += 1
    ranked = hours.most_common(2)
    if len(times) < PEAK_MIN or len(ranked) < 2 or ranked[0][1] <= ranked[1][1]:
        return None
    (_, hour, _), _ = ranked[0]
    return {0: "midnight", 12: "noon"}.get(hour, f"{hour % 12} {'am' if hour < 12 else 'pm'}")


def walked_past_text(
    crossed: Mapping[str, Any] | None,
    moved: Mapping[str, Any] | None,
    elapsed_s: float,
    tz: dt.tzinfo | None,
    so_far: bool = False,
) -> str:
    """Plain sentences for a summary: per camera, how many times people walked past its counting line (a
    `crossing_counts.from_spans`, its walk-by `times` read by local hour) and came into the picture (a
    `movement_counts.collect`). Cameras are never summed. A camera the body worker never watched says nothing rather
    than zero, one watched for less than the day so far (`elapsed_s`) says how long, and movement counted on partial
    coverage is "at least". "" when there is nothing to say."""
    crossings = (crossed or {}).get("cameras") or {}
    movement = (moved or {}).get("cameras") or {} if (moved or {}).get("available") else {}
    sentences = []
    for cam in sorted(set(crossings) | set(movement)):
        c, m = crossings.get(cam) or {}, movement.get(cam) or {}
        watched = float(c.get("scanned_seconds") or 0)
        walked = int((c.get("crossings") or {}).get("total") or 0) if watched > 0 else 0
        entries = int(m.get("entry_count") or 0) if m.get("available") else 0
        clauses = []
        if walked:
            peak = _peak(c.get("times") or [], tz)
            clauses.append(f"walked past {_times(walked)}" + (f" (busiest around {peak})" if peak else ""))
        if entries:
            clauses.append(f"came into the picture {'at least ' if m.get('partial') else ''}{_times(entries)}")
        if not clauses:
            continue
        said = " and ".join(clauses)
        if len(set(crossings) | set(movement)) == 1:
            sentence = f"People {said.replace('walked past', 'walked past the camera', 1)}."
            subject = "The camera"
        else:
            sentence = f"On {cam}, people {said}."
            subject = "It"
        if walked and watched < WATCHED_ENOUGH * elapsed_s:
            day = "the day so far" if so_far else "the day"
            sentence += f" {subject} only watched for {_watched(watched)} of {day}, so there may have been more."
        sentences.append(sentence)
    return " ".join(sentences)


def answer_text(
    crossed: Mapping[str, Any] | None,
    moved: Mapping[str, Any] | None,
    elapsed_s: float,
    tz: dt.tzinfo | None,
    so_far: bool = False,
    *,
    entries: bool = False,
) -> str:
    """Ask's and voice's walk-by answer (#1591): walked_past_text, or what can be said when it has nothing, and that
    these are times, not people. With `entries` (the question asked about coming into view, not walking by), a camera
    that wasn't watching is said in those words."""
    said = walked_past_text(crossed, moved, elapsed_s, tz, so_far=so_far)
    if not said:
        crossings = (crossed or {}).get("cameras") or {}
        watched = any(float(c.get("scanned_seconds") or 0) > 0 for c in crossings.values()) or bool(
            (moved or {}).get("available")
        )
        if not watched:
            happened = "came into view" if entries else "walked by"
            return f"No camera was watching then, so I can't say how many times someone {happened}."
        partial = (crossed or {}).get("partial") or (moved or {}).get("partial")
        said = "Nobody was seen walking past or coming into the picture." + (
            " The camera wasn't watching the whole time, so someone may have." if partial else ""
        )
    return said + " These are times, not different people, and they can't say who."


def direction_text(crossed: Mapping[str, Any] | None, *, include_counts: bool = False) -> str:
    """Describe observed image directions per camera, never deriving them from counts.

    Native activity results keep one citeable `direction_examples` witness per direction. Older results
    may supply only `examples`; an explicitly empty witness list remains unavailable. Coverage gaps cannot
    establish that no other direction occurred, and image movement cannot establish identity or destination.
    With `include_counts`, label each witnessed direction with its observed passage count when available;
    never substitute a camera total or the raw line-crossing count for a direction's walk-by quantity.
    """
    labels = {
        "left_to_right": "left to right",
        "right_to_left": "right to left",
        "top_to_bottom": "top to bottom",
        "bottom_to_top": "bottom to top",
        "side_a_to_side_b": "line side A to line side B",
        "side_b_to_side_a": "line side B to line side A",
    }
    cameras = (crossed or {}).get("cameras") or {}
    sentences: list[str] = []
    for name, camera in sorted(cameras.items()):
        examples = camera.get("direction_examples", camera.get("examples")) or []
        observed = {example["direction"] for example in examples if isinstance(example.get("direction"), str)}
        directions: list[str] = []
        for direction, label in labels.items():
            if direction not in observed:
                continue
            if include_counts:
                quantity = (camera.get("crossings") or {}).get(direction)
                if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity >= 0:
                    unit = "passage" if quantity == 1 else "passages"
                    label += f" ({quantity} observed {unit})"
                else:
                    label += " (passage quantity unavailable)"
            directions.append(label)
        if directions:
            mixed = "mixed directions: " if len(directions) > 1 else ""
            sentences.append(f"On {name}, observed movement was {mixed}{' and '.join(directions)} in the camera image.")
        else:
            sentences.append(f"On {name}, direction evidence is unavailable.")
    if not sentences:
        sentences.append("Direction evidence is unavailable for this window.")
    if (crossed or {}).get("partial") or any(camera.get("partial") for camera in cameras.values()):
        sentences.append("Coverage is partial, so movement in other directions may not have been observed.")
    if include_counts:
        sentences.append("These quantities count observed passages, not distinct people.")
    sentences.append("These image directions do not establish compass direction, destination, or who moved.")
    return " ".join(sentences)
