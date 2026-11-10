"""Detected activity for count questions (#228): see-actions' walk-by spans (`walkby.span`) and model actions
(`action.span`) in the segment store, as memory/event_counts.py reads them. Retired rules are excluded (#1742).

- events(state, start, end): the promoted spans as {id, camera, start, end, count, name, person, kind}. `count` is
  supplied only when the model measures repetitions. `name` is the owner's name for it (an `activity.named`
  record in the knowledge store), else the detector's kind, else None. `person` is a PersonId, or None while nobody's face was matched to
  the track.
- coverage(state, start, end): (camera, start, end) of each stretch walk-bys were looked for (`walkby.coverage`),
  so a window nothing looked at isn't counted as a zero.

A recount's replaced source is omitted. Owner feedback calibrates each kind from distinct source events, using the
latest retained verdict per event. A camera angle with enough ratings overrides the kind-wide policy. Known Wrong
sources are rejected, not unknown candidates. Non-crossing kinds require sufficient Right feedback; crossings keep
their existing eligible default until sufficient feedback demotes them. Camera-motion exclusions remain per span.

Owner names and ratings are what the owner taught, so they live in the knowledge store (knowledge.sqlite3,
#1832), outside the catalog reset: `activity.named` and `activity.feedback` records keyed by the span's id, over its
time, with its kind, camera, and source stream. A rating without its source stream is from the rule-based era, whose
kinds the model reuses, so it is left out of calibration. promotion_measurements exposes the current
owner-agreement aggregates; these are an owner-selected cohort, not held-out detector accuracy or recall.

"""

import dataclasses
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from liberational import config as cfg
from liberational import knowledge, segment_log
from liberational.memory import activity_kinds
from liberational.people import registry

JOIN_S = 1.0  # coverage stretches this close together are one
# Kinds only the retired rules produced (#1742): never projected, rated, or annotated. see-actions' model reuses
# some of the old names (sit_down, stand_up, phone_use), so those count again.
RETIRED_KINDS = frozenset({None, "still", "repeated_movement", "pickup", "putdown", "eating_or_drinking"})
NAMED = "activity.named"  # knowledge-store collections (#1832)
FEEDBACK = "activity.feedback"
MAX_ANNOTATIONS = 50_000  # owner names or ratings read at once; more is an error, never a truncated cohort
_STORES: dict[Path, knowledge.Store] = {}
_STORES_LOCK = threading.Lock()


def _store(state: Any) -> knowledge.Store:
    path = registry.knowledge_path(state)
    with _STORES_LOCK:
        if path not in _STORES:
            _STORES[path] = knowledge.Store(path)
        return _STORES[path]


def _ns(seconds: float) -> int:
    """Epoch seconds as the knowledge store's nanoseconds, clamped to its range: an open-ended read (0 to 1e12, as
    event_counts' name lookup makes) would otherwise overflow it."""
    return min(max(round(seconds * 1e9), 0), 2**63 - 1)


def _names(state: Any, start: float, end: float) -> dict[str, str]:
    """The owner's latest name for each span overlapping [start, end), by span id."""
    if not registry.knowledge_path(state).exists():
        return {}  # nothing taught yet; reading must not create the store (registry.log_exists reads its presence)
    rows = _store(state).query(NAMED, start_ns=_ns(start), end_ns=_ns(end), limit=MAX_ANNOTATIONS + 1)
    if len(rows) > MAX_ANNOTATIONS:
        raise RuntimeError("activity names exceed the safe read bound")
    out: dict[str, str] = {}
    for row in rows:
        value = row.record.value
        if (
            value.get("source") == "owner"
            and isinstance(value.get("activity"), str)
            and isinstance(value.get("name"), str)
        ):
            out[value["activity"]] = value["name"]
    return out


def _votes(state: Any, *, deleted: bool = False) -> list[dict[str, Any]]:
    """The owner's latest Right or Wrong per span, with the span's kind and camera. A rating without its source
    stream (the rule-based era's, imported by #1796) is left out, and so is a deleted activity's (#3244) unless
    `deleted`."""
    if not registry.knowledge_path(state).exists():
        return []
    gone = frozenset() if deleted else activity_kinds.deleted(state)
    rows = _store(state).query(FEEDBACK, limit=MAX_ANNOTATIONS + 1)
    if len(rows) > MAX_ANNOTATIONS:
        raise RuntimeError("activity ratings exceed the safe read bound")
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.record.value
        if (
            value.get("source") != "owner"
            or value.get("verdict") not in {"right", "wrong"}
            or not isinstance(value.get("activity"), str)
            or not isinstance(value.get("stream"), str)
            or value.get("kind") in gone
        ):
            continue
        latest[value["activity"]] = {
            "activity": value["activity"],
            "feedback": row.record.id,
            "verdict": value["verdict"],
            "kind": value.get("kind"),
            "cam": value.get("cam"),
        }
    return list(latest.values())


@dataclasses.dataclass(frozen=True)
class RatingStats:
    rated: int = 0
    right: int = 0


class _Promotion:
    def __init__(self, votes: list[dict[str, Any]]) -> None:
        minimum = cfg.get("activity.promotion_min_ratings")
        precision = cfg.get("activity.promotion_min_precision")
        if type(minimum) is not int or minimum < 1:
            raise ValueError("activity promotion_min_ratings must be a positive integer")
        if type(precision) not in (int, float) or not math.isfinite(precision) or not 0 <= precision <= 1:
            raise ValueError("activity promotion_min_precision must be finite and in [0, 1]")
        self.minimum, self.precision = minimum, float(precision)
        self.answers_enabled = bool(cfg.get("events.answer_counts"))
        votes = [vote for vote in votes if vote.get("kind") not in RETIRED_KINDS]
        self.verdicts = {row["activity"]: row["verdict"] for row in votes}
        self.stats: dict[tuple[str | None, str | None], RatingStats] = {}
        for vote in votes:
            kind, cam = vote.get("kind"), vote.get("cam")
            if kind is not None and not isinstance(kind, str):
                continue
            for key in [(kind, None), *(([(kind, cam)]) if isinstance(cam, str) else [])]:
                old = self.stats.get(key, RatingStats())
                self.stats[key] = RatingStats(old.rated + 1, old.right + (vote["verdict"] == "right"))

    def summary(self, kind: str | None, camera: str | None) -> dict[str, Any]:
        own = self.stats.get((kind, camera), RatingStats())
        use_camera = camera is not None and own.rated >= self.minimum
        stats = own if use_camera else self.stats.get((kind, None), RatingStats())
        precision = stats.right / stats.rated if stats.rated else None
        enough = stats.rated >= self.minimum
        counted = (precision is not None and precision >= self.precision) if enough else kind == "crossing"
        status = "counted" if counted else "needs_correct_ratings" if enough else "needs_ratings"
        if not self.answers_enabled:
            status = "disabled"
        return {
            "kind": kind,
            "camera": camera,
            "status": status,
            "remaining_ratings": max(0, self.minimum - stats.rated) if not enough and not counted else None,
            "rated": stats.rated,
            "right": stats.right,
            "precision": precision,
            "min_ratings": self.minimum,
            "min_precision": self.precision,
        }

    def rows(self, spans: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        keys = {key for key, stats in self.stats.items() if key[1] is None or stats.rated >= self.minimum}
        for span in spans or []:
            if span.get("kind") is None or isinstance(span.get("kind"), str):
                keys.add((span.get("kind"), None))
        return [
            self.summary(kind, camera) for kind, camera in sorted(keys, key=lambda key: (key[0] or "", key[1] or ""))
        ]

    def promoted(self, span: dict[str, Any]) -> bool:
        if self.verdicts.get(span["id"]) == "wrong" or span.get("camera_moved"):
            return False
        kind = span.get("kind")
        if kind is not None and not isinstance(kind, str):
            return False
        stats = self.stats.get((kind, span.get("cam")), RatingStats())
        if stats.rated < self.minimum:
            stats = self.stats.get((kind, None), RatingStats())
        if stats.rated >= self.minimum:
            return stats.right / stats.rated >= self.precision
        return kind == "crossing" and span.get("candidate") is not True


def kind_answers(state: Any) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Per activity kind, how many clips the owner answered (deleted activities' too, which a restore brings back),
    and the live kinds' counting status in Ask, as promotion_measurements' kind-wide rows (#3244)."""
    answered: dict[str, int] = {}
    for vote in _votes(state, deleted=True):
        if isinstance(vote.get("kind"), str):
            answered[vote["kind"]] = answered.get(vote["kind"], 0) + 1
    policy = _Promotion(_votes(state))
    return answered, {k.kind: policy.summary(k.kind, None) for k in activity_kinds.kinds(state)}


def promotion_measurements(state: Any) -> list[dict[str, Any]]:
    """Authoritative current owner-agreement aggregates; these are not held-out precision or recall."""
    return _Promotion(_votes(state)).rows()


def _spans(state: Any, start: float, end: float) -> list[dict[str, Any]]:
    """Every activity span overlapping [start, end), from the segment store (#1779): see-actions' walk-bys and model
    actions, oldest first, with retired kinds and the owner's deleted activities (#3244) left out. Each carries its
    `stream`."""
    log = segment_log.at(state)
    gone = RETIRED_KINDS | activity_kinds.deleted(state)
    found = [
        {**span, "stream": stream}
        for stream in (segment_log.WALKBY_SPANS, segment_log.ACTION_SPANS)
        for span in log.range(stream, start, end)
    ]
    found.sort(key=lambda s: (float(s["ts_start"]), str(s["id"])))
    return [s for s in found if s.get("kind") not in gone]


def events(state: Any, start: float, end: float) -> list[dict[str, Any]]:
    """The promoted activity spans overlapping [start, end), oldest first."""
    return count_evidence(state, start, end)[0]


def count_evidence(state: Any, start: float, end: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One projection of promoted and unverified spans, so count answers cannot mistake withheld evidence for absence."""
    spans = _spans(state, start, end)
    replaced = {str(s["replaces"]) for s in spans if s.get("replaces")}
    names = _names(state, start, end)
    policy = _Promotion(_votes(state))
    promoted: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for s in sorted(spans, key=lambda s: (s["ts_start"], s["id"])):
        if s["id"] in replaced or policy.verdicts.get(s["id"]) == "wrong":
            continue
        target = promoted if policy.promoted(s) else unverified
        target.append(
            {
                "id": s["id"],
                "camera": s.get("cam"),
                "start": s["ts_start"],
                "end": s["ts_end"],
                "count": s.get("count"),
                "name": names.get(s["id"]) or s.get("kind"),
                "person": s.get("person"),
                "person_name": s.get("person_name"),
                "kind": s.get("kind"),
                "stream": s.get("stream"),
            }
        )
    return promoted, unverified


def _review_evidence(value: Any) -> Any:
    """Keep review evidence without copying source-record arrays into the dashboard DTO."""
    if isinstance(value, dict):
        return {
            key: _review_evidence(item)
            for key, item in value.items()
            if key not in {"records", "parents", "source_ids", "pose_ids"}
        }
    if isinstance(value, list):
        return [_review_evidence(item) for item in value]
    return value


ANY_KIND = object()  # detected(): no kind filter; source candidates must have an explicit current kind


def detected(
    state: Path, start: float, end: float, *, limit: int = 200, kind: object = ANY_KIND, unrated: bool = False
) -> dict[str, Any]:
    """A bounded list of detected candidates, including unpromoted spans, for explicit owner naming. `kind` and
    `unrated` narrow the items to one kind and to those with no owner verdict yet, before the bound, so a rating
    queue on a busy day still reaches a rare kind; the promotion rows always cover every kind."""
    limit = max(1, min(limit, 200))
    spans = _spans(state, start, end)
    replaced = {str(s["replaces"]) for s in spans if s.get("replaces")}
    names = _names(state, start, end)
    policy = _Promotion(_votes(state))
    rows = [
        s
        for s in spans
        if s["id"] not in replaced
        and (kind is ANY_KIND or s.get("kind") == kind)
        and not (unrated and s["id"] in policy.verdicts)
    ]
    rows.sort(key=lambda s: (s["ts_start"], s["id"]))
    return {
        "items": [
            {
                "id": s["id"],
                "camera": s.get("cam"),
                "start": s["ts_start"],
                "end": s["ts_end"],
                "count": s.get("count"),
                "name": names.get(s["id"]),
                "kind": s.get("kind"),
                "stream": s.get("stream"),
                "person": s.get("person"),
                "track_id": s.get("track_id"),
                "attributes": _review_evidence(s.get("attributes") or {}),
                "attribute_evidence": _review_evidence(s.get("attribute_evidence") or {}),
                "uncertainty": _review_evidence(s.get("uncertainty") or {}),
                "feedback": policy.verdicts.get(s["id"]),
                "promoted": policy.promoted(s),
            }
            for s in rows[-limit:]
        ],
        "truncated": len(rows) > limit,
        "promotion": policy.rows(spans),
    }


def coverage(state: Any, start: float, end: float, *, kind: str | None = None) -> list[tuple[str, float, float]]:
    """(camera, start, end) of each stretch `kind` was looked for in [start, end), joined, oldest first: see-actions'
    `action.coverage` for a model activity, else (no kind, or the walk-by line's) `walkby.coverage`."""
    model = kind is not None and kind != activity_kinds.CROSSING
    stream = segment_log.ACTION_COVERAGE if model else segment_log.WALKBY_COVERAGE
    rows = segment_log.at(state).range(stream, start, end)
    return _joined(rows, start, end)


def _joined(
    records: list[dict[str, Any]], start: float, end: float, *, join_s: float = JOIN_S
) -> list[tuple[str, float, float]]:
    rows = sorted(
        ((str(c.get("cam")), max(c["ts_start"], start), min(c["ts_end"], end)) for c in records),
        key=lambda r: (r[0], r[1]),
    )
    out: list[tuple[str, float, float]] = []
    for cam, a, b in rows:
        if b <= a:
            continue
        if out and out[-1][0] == cam and a - out[-1][2] <= join_s:
            out[-1] = (cam, out[-1][1], max(out[-1][2], b))
        else:
            out.append((cam, a, b))
    return out


@dataclasses.dataclass(frozen=True)
class NameActivity:
    """An explicit owner name for one existing detected span, located by its source time."""

    activity: str
    start: float
    name: str

    def __post_init__(self) -> None:
        uuid.UUID(self.activity)
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError("activity start must be a finite timestamp")
        if not self.name.strip() or len(self.name.strip()) > 120 or any(ord(c) < 32 for c in self.name):
            raise ValueError("activity name must be 1 to 120 characters without control characters")


@dataclasses.dataclass(frozen=True)
class FeedbackActivity:
    """An owner's right/wrong judgment, separate from naming, measurements, and promotion."""

    activity: str
    start: float
    verdict: str

    def __post_init__(self) -> None:
        uuid.UUID(self.activity)
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError("activity start must be a finite timestamp")
        if self.verdict not in {"right", "wrong"}:
            raise ValueError("verdict must be right or wrong")


def name_activity(state: Path, command: NameActivity) -> dict[str, Any]:
    """Name an existing candidate without promoting it or changing its measurements.

    The name is kept over the span's own time, so it shows with its activity even when the owner names yesterday's
    recording today.
    """
    return _annotate_activity(state, command.activity, command.start, {"name": command.name.strip()}, NAMED)


def feedback_activity(state: Path, command: FeedbackActivity) -> dict[str, Any]:
    """Append owner feedback on the exact source span; never promote or rename it."""
    return _annotate_activity(state, command.activity, command.start, {"verdict": command.verdict}, FEEDBACK)


def _annotate_activity(
    state: Path, activity: str, source_start: float, fields: dict[str, Any], collection: str
) -> dict[str, Any]:
    """Write one owner annotation of a span to the knowledge store (outside the catalog reset) and return it."""
    source = next((s for s in _spans(state, source_start, source_start + 0.001) if s["id"] == activity), None)
    if source is None:
        raise LookupError("detected activity no longer exists at that time")
    start, end = float(source["ts_start"]), float(source["ts_end"])
    if not math.isfinite(start) or not math.isfinite(end) or end < start or not source.get("cam"):
        raise ValueError("detected activity has invalid span provenance")
    value = {
        "activity": source["id"],
        **fields,
        "source": "owner",
        "kind": source.get("kind"),
        "cam": source.get("cam"),
        "stream": source["stream"],
    }
    classes = _profile(state, source)
    if classes:
        value["classes"] = classes  # what activity_kinds.learned learns from (#3244)
    record = knowledge.Record(str(uuid.uuid7()), collection, source["id"], _ns(start), _ns(end), time.time_ns(), value)
    _store(state).put([record])
    return {**value, "ts_start": start, "ts_end": end}


PROFILE_TOP = 5  # classes an answer keeps per model


def _profile(state: Any, source: dict[str, Any]) -> dict[str, list[list[Any]]]:
    """What the models saw in a span, kept with each answer for activity_kinds.learned (#3244): per model ("ntu",
    "kinetics"), its PROFILE_TOP classes by mean score over the span's track's windows, [name, score] pairs, most
    first. A span with no track (a walk-by) or no windows left (a reset) keeps none."""
    from liberational.see import action_clips

    track = source.get("track_id")
    if track is None:
        return {}
    sums: dict[str, dict[int, float]] = {m: {} for m in activity_kinds.MODELS}
    seen = dict.fromkeys(activity_kinds.MODELS, 0)
    rows = segment_log.at(state).range(
        segment_log.ACTION_WINDOWS, float(source["ts_start"]), float(source["ts_end"]), [str(source["cam"])]
    )
    for w in rows:
        if str(w.get("track_id")) != str(track):
            continue
        model = "kinetics" if w.get("model") == action_clips.MODEL else "ntu"
        seen[model] += 1
        for c, p in zip(w.get("classes") or [], w.get("scores") or [], strict=False):
            sums[model][int(c)] = sums[model].get(int(c), 0.0) + float(p)
    out: dict[str, list[list[Any]]] = {}
    for model, totals in sums.items():
        names = activity_kinds.CLASSES[model]
        top = sorted(((p / seen[model], c) for c, p in totals.items() if 0 <= c < len(names)), reverse=True)
        if top:
            out[model] = [[names[c], round(p, 4)] for p, c in top[:PROFILE_TOP]]
    return out


def profiled(state: Any) -> list[tuple[dict[str, Any], str]]:
    """The owner's latest answer per span in each of FEEDBACK and NAMED that kept the models' classes (`_profile`),
    with its collection, for activity_kinds.learned."""
    if not registry.knowledge_path(state).exists():
        return []
    out: list[tuple[dict[str, Any], str]] = []
    for collection in (FEEDBACK, NAMED):
        rows = _store(state).query(collection, limit=MAX_ANNOTATIONS + 1)
        if len(rows) > MAX_ANNOTATIONS:
            raise RuntimeError("activity answers exceed the safe read bound")
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            value = row.record.value
            if value.get("source") == "owner" and isinstance(value.get("activity"), str):
                latest[value["activity"]] = value
        out.extend((v, collection) for v in latest.values() if v.get("classes"))
    return out
