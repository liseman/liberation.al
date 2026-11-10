"""Owner activity names and feedback, separate from measurement gold labels (Review's EventForm)."""

import datetime as dt
from pathlib import Path
from typing import Any

from liberational import config as cfg
from liberational.memory import activity_kinds, events


def name(body: object, control: bool) -> tuple[int, dict[str, Any]]:
    """Validate POST /activity/name with the dashboard's existing local-control decision."""
    if not control:
        return 403, {"ok": False, "error": "activities can only be named on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    activity, start, label = body.get("activity"), body.get("start"), body.get("name")
    if (
        not isinstance(activity, str)
        or not isinstance(label, str)
        or isinstance(start, bool)
        or not isinstance(start, int | float)
    ):
        return 400, {"ok": False, "error": "activity id, source start, and name required"}
    try:
        command = events.NameActivity(activity, float(start), label)
        record = events.name_activity(cfg.path("paths.state"), command)
    except ValueError as exc:
        return 400, {"ok": False, "error": str(exc)}
    except LookupError as exc:
        return 404, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, "activity": record["activity"], "name": record["name"]}


def feedback(body: object, control: bool) -> tuple[int, dict[str, Any]]:
    """Validate POST /activity/feedback with the same local-control rule as owner naming."""
    if not control:
        return 403, {"ok": False, "error": "activity feedback is only available on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    activity, start, verdict = body.get("activity"), body.get("start"), body.get("verdict")
    if (
        not isinstance(activity, str)
        or not isinstance(verdict, str)
        or isinstance(start, bool)
        or not isinstance(start, int | float)
    ):
        return 400, {"ok": False, "error": "activity id, source start, and verdict required"}
    try:
        command = events.FeedbackActivity(activity, float(start), verdict)
        record = events.feedback_activity(cfg.path("paths.state"), command)
    except (ValueError, OverflowError) as exc:
        return 400, {"ok": False, "error": str(exc)}
    except LookupError as exc:
        return 404, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, "activity": record["activity"], "verdict": record["verdict"]}


def _kind_row(
    k: activity_kinds.Kind,
    taught: dict[str, dict[str, tuple[str, ...]]],
    answered: dict[str, int],
    promotion: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    learned = taught.get(k.kind, {})
    return {
        "kind": k.kind,
        "name": k.name,
        "description": k.description,
        "rule": k.rule,
        "classes": {m: list(getattr(k, m)) for m in activity_kinds.MODELS},
        "learned": {m: [c for c in learned.get(m, ()) if c not in getattr(k, m)] for m in activity_kinds.MODELS},
        "answers": answered.get(k.kind, 0),
        "promotion": promotion.get(k.kind),
    }


def kinds(state: Path) -> tuple[int, dict[str, Any]]:
    """GET /activity/kinds: the owner's activities, each with the model classes it was defined with and
    learned, how many clips were answered, and whether Ask counts it; and the deleted ones, which can be restored."""
    every = activity_kinds.every(state)
    live = [k for k in every if not k.deleted]
    taught = activity_kinds.learned(state, live)
    answered, promotion = events.kind_answers(state)
    return 200, {
        "kinds": [_kind_row(k, taught, answered, promotion) for k in live],
        "deleted": [{"kind": k.kind, "name": k.name, "answers": answered.get(k.kind, 0)} for k in every if k.deleted],
    }


def edit_kind(path: str, body: object, control: bool, state: Path) -> tuple[int, dict[str, Any]]:
    """POST /activity/kinds ({name, description}: add one), /activity/kinds/delete and /activity/kinds/restore
    ({kind}). A delete keeps the activity's answers, so a restore (the panel's Undo) brings it all back."""
    if not control:
        return 403, {"ok": False, "error": "activities can only be changed on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    try:
        if path == "/activity/kinds":
            name, description = body.get("name"), body.get("description") or ""
            if not isinstance(name, str) or not isinstance(description, str):
                return 400, {"ok": False, "error": "an activity needs a name"}
            made = activity_kinds.add(state, name, description)
        else:
            kind = body.get("kind")
            if not isinstance(kind, str):
                return 400, {"ok": False, "error": "which activity?"}
            change = activity_kinds.delete if path == "/activity/kinds/delete" else activity_kinds.restore
            made = change(state, kind)
    except ValueError as exc:
        return 400, {"ok": False, "error": str(exc)}
    except LookupError as exc:
        return 404, {"ok": False, "error": str(exc)}
    answered, promotion = events.kind_answers(state)
    return 200, {"ok": True, "kind": _kind_row(made, activity_kinds.learned(state), answered, promotion)}


FACE_REACH_S = 30.0  # how far either side of an activity the faces on its track are looked for
FACE_READ_GAP_S = 300.0  # activities whose windows are this close share one read of the faces
SETTLED_S = 90.0  # an activity that ended this long ago has all its faces: its name can be kept
_NAMED: dict[tuple[str, str], tuple[object, str | None]] = {}  # (state, activity id): (identity version, name)
_NAMED_MAX = 5000


def _track_names(items: list[dict[str, Any]], state: Path) -> None:
    """Who did an activity that was written without a person (a walk-by is closed before its face is known): the
    most common name among the known faces on its camera and pose track around it, as Review's body boxes are named
    (review_boxes._bodies), read now so a rename since applies. Sets each such item's person_name, or leaves None.
    The faces are read once per stretch of nearby activities, not once per activity (a read costs about its range),
    and a settled activity's answer is kept until the people registry changes (a correction or a rename), so a day
    read again costs nothing more."""
    import time
    from collections import Counter

    from liberational.people import registry, transcript_view
    from liberational.vision import sightings

    log = registry.at(state).log
    probe = getattr(log, "identity_version", None) or getattr(log, "version", None)  # as the registry reads it
    version = probe() if probe is not None else None  # None: nothing to key on, so nothing is kept
    settled = time.time() - SETTLED_S
    todo = []
    for item in items:
        if (
            item.get("person_name")
            or not isinstance(item.get("track_id"), str)
            or not isinstance(item.get("camera"), str)
        ):
            continue
        kept = _NAMED.get((str(state), str(item["id"])))
        if kept is not None and version is not None and kept[0] == version:
            item["person_name"] = kept[1]
        else:
            todo.append(item)
    todo.sort(key=lambda item: float(item["start"]))
    stretches: list[list[dict[str, Any]]] = []
    for item in todo:
        last = stretches[-1] if stretches else None
        if last and float(item["start"]) - max(float(i["end"]) for i in last) < FACE_READ_GAP_S:
            last.append(item)
        else:
            stretches.append([item])
    view = transcript_view.View(state)
    for stretch in stretches:
        start = min(float(i["start"]) for i in stretch) - FACE_REACH_S
        end = max(float(i["end"]) for i in stretch) + FACE_REACH_S
        by_track: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for f in sightings.faces(start, end, None, state):
            if isinstance(f.get("track_id"), str) and isinstance(f["name"], str) and f["name"].strip():
                by_track.setdefault((f["cam"], f["track_id"]), []).append(f)
        for item in stretch:
            a, b = float(item["start"]) - FACE_REACH_S, float(item["end"]) + FACE_REACH_S
            votes = Counter(
                view.current(f["name"], float(f["ts"]))
                for f in by_track.get((item["camera"], item["track_id"]), [])
                if a <= float(f["ts"]) < b
            )
            if votes:
                item["person_name"] = votes.most_common(1)[0][0]
            if version is not None and float(item["end"]) < settled:
                if len(_NAMED) >= _NAMED_MAX:
                    _NAMED.clear()
                _NAMED[(str(state), str(item["id"]))] = (version, item.get("person_name"))


def listing(day: str, state: Path, kind: str | None = None, unrated: bool = False) -> tuple[int, dict[str, Any]]:
    """GET /activities?day=YYYY-MM-DD, at most 200 most recent candidates of that local day. `kind` (the empty
    string for unnamed movement) and `unrated=1` narrow it to one kind's candidates that have no rating yet."""
    from liberational.dashboard.presence import day_bounds

    try:
        if dt.date.fromisoformat(day).isoformat() != day:
            raise ValueError("noncanonical day")
        start, end = day_bounds(day)
    except ValueError, OverflowError:
        return 400, {"error": "day must be YYYY-MM-DD"}
    from liberational.people import registry

    wanted: object = events.ANY_KIND if kind is None else kind or None
    reply = events.detected(state, start, end, kind=wanted, unrated=unrated)
    reg = registry.at(state)
    with reg.lock:
        # Refresh once, then resolve historical IDs against the same merge/name snapshot for every row.
        names = {person.id: person.display_name for person in reg.people()}
        for item in reply["items"]:
            current = reg.resolve(item["person"])
            item["person_name"] = names.get(current) if current is not None else None
    _track_names(reply["items"], state)
    return 200, reply
