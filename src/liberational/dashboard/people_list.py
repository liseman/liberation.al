"""The people the dashboard knows, from every source: voices, faces, contacts, people added by hand, and callers
(`people_roster`), with the name rules they share (moved out of the server, #61). No routes: the People reads and
writes are built on this.
"""

import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from types import ModuleType
from typing import Any

from liberational import config as cfg
from liberational.dashboard import call_log, context, known_voices, photo_files, transcript_lines, voice_clones
from liberational.dashboard.context import LOG
from liberational.people import card_prefs, contacts, face_shots, profiles, registry, transcript_view
from liberational.vision import views


def _ready_faces(
    rows: list[dict[str, Any]], saved: AbstractSet[str] | None = None, *, grid: bool = False
) -> tuple[list[dict[str, Any]], int, int]:
    """Saved face crops only; schedule missing IDs without reading video on this request."""
    revision = photo_files._shot_revision()
    kind = "shots_grid" if grid else "shots"
    saved = face_shots.saved(context.STATE, kind, revision) if saved is None else saved
    ready, pending, unavailable = [], 0, 0
    for row in rows:
        sid = str(row["id"])
        if (
            sid in saved
            and face_shots.path(context.STATE, sid, revision, kind).is_file()
            and (not grid or face_shots.path(context.STATE, sid, revision).is_file())
        ):
            ready.append({**row, "url": f"/faces/shot/{sid}.jpg?size=80"} if grid else row)
        else:
            queued = photo_files._crop_jobs.offer(context.STATE, kind, sid, revision)
            pending += queued
            unavailable += not queued
    return ready, pending, unavailable


ROOM_TAIL = 20000  # room lines read for People: the newest this many


def _room_lines(
    raw: list[dict[str, Any]] | None = None,
    view: transcript_view.View | None = None,
    *,
    named_lines: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    # Each line's person by their current name, as of the line's time: a renamed person keeps the lines heard under
    # their old name, and someone new who takes that name does not get them.
    if named_lines is None:
        view = view if view is not None else transcript_lines._view()
        named_lines = view.named(raw[-ROOM_TAIL:] if raw is not None else transcript_lines.heard_tail(ROOM_TAIL))
    for e in named_lines:
        if e.get("media"):  # a TV's line is nobody's
            continue
        try:
            who = e.get("speaker") or "unknown"
            if (e.get("text") or "").strip():
                ts = float(e.get("ts_start", 0) or 0)
                rows.append(
                    {"who": who, "text": e["text"].strip(), "ts": ts, "src": "room", "sid": e.get("speaker_id")}
                )
        except Exception:
            pass
    return rows


def _iso_ts(v: object) -> float:
    try:
        import datetime as _dt

        return _dt.datetime.fromisoformat(str(v)).timestamp()
    except Exception:
        return 0.0


def _call_lines(view: transcript_view.View | None = None) -> list[dict[str, Any]]:
    """The far end's lines from recent calls, each under the name of who said it now, read as of the call the way
    profile rebuilds read them (`profiles.call_name`): a renamed person keeps their calls under their new name, and a
    forgotten person's lines are left out, so neither leaves an entry in People under a name nobody has."""
    rows, view, memo = [], view if view is not None else transcript_lines._view(), {}
    for c in call_log.calls_list(500):
        t = call_log.call_transcript(c["day"], c["dir"]) or {"transcript": []}
        start = profiles.call_start(call_log.CALLS_DIR / c["day"] / c["dir"])
        for e in t["transcript"]:
            if e.get("speaker") == "bot" or not (e.get("text") or "").strip():
                continue
            if not e.get("speaker_label"):
                continue  # an unnamed far end is not a person; it made a "Caller" entry in People
            if (who := profiles.call_name(view, e["speaker_label"], profiles.call_at(start, e.get("t")), memo)) is None:
                continue
            rows.append(
                {
                    "who": who,
                    "text": e["text"].strip(),
                    "ts": _iso_ts(c.get("started")),
                    "src": "call:" + c["dir"],
                }
            )
    return rows


def _norm(name: str | None) -> str:
    """The key a name is matched by, as the registry matches it (registry.norm), with a "(phone)" suffix dropped."""
    return registry.norm(re.sub(r"\s*\(phone\)$", "", (name or "")))


FACE_GROUP = re.compile(r"face \d+")  # unnamed face groups, normalized ("Face 3")


VOICE_GROUP = re.compile(r"Speaker \d+")  # unnamed voice clusters


_seen: dict[str, Any] = {"t": 0.0, "by": {}}


def _faces_mod() -> ModuleType:
    """vision/faces.py without loading InsightFace (only its database helpers are used here)."""
    from liberational.vision import faces

    return faces


def _face_last_seen_at() -> dict[str, tuple[float, str | None]]:
    """{normalized name: (last seen, the camera)}: each person's newest face visit on the broker, read through the
    registry (`people/face_identity.py`), so a rename or a deletion shows on the next read."""
    from liberational.people import face_identity

    projection = face_identity.last_seen(context.STATE)
    _seen["identity"] = {"pending": False, "source": "face.visit"}
    by: dict[str, tuple[float, str | None]] = {}
    for row in projection["matches"]:
        key = _norm(row["name"])
        if key not in by or row["ts"] > by[key][0]:
            by[key] = (row["ts"], row.get("cam"))
    return by


def _owner_keys() -> set[str]:
    """The owner's names (owner.name and owner.speaker_label), as _norm keys. Every owner check compares with these,
    so a name that differs only in capitals or spacing ("GROSS" for "Groß") is the owner's too."""
    return {_norm(cfg.owner_name()), _norm(cfg.owner_label())} - {""}


def people_roster(
    raw: list[dict[str, Any]] | None = None,
    view: transcript_view.View | None = None,
    speakers: list[dict[str, Any]] | None = None,
    *,
    named_lines: list[dict[str, Any]] | None = None,
    saved_shots: AbstractSet[str] | None = None,
) -> list[dict[str, Any]]:
    """One entry per person, joining their voices (by person_of, through _speakers), tagged faces (the face
    database), the lines they said, and their record in the person registry: added by hand, phone, role,
    organization, and old names (people/contacts.py reads the first two, with contacts.json and people_added.json as
    the fallback before the backfill). Keyed by normalized name; the display name keeps capitals."""
    by = {}
    # A voice made from one uploaded recording is a clone, listed under Voice clones only: not a person here, and not a
    # name offered when tagging. Its voice and the room lines once filed under it are left out, so a name
    # only a clone carries is too. A person who shares the name keeps everything else.
    clone_ids = voice_clones._clone_only_ids()

    def person(name: str) -> dict[str, Any]:
        return by.setdefault(
            _norm(name),
            {"name": name.strip(), "lines": 0, "room": 0, "calls": 0, "last": 0, "voices": [], "faces": 0},
        )

    room = (
        _room_lines(raw, view, named_lines=named_lines) if raw is not None or named_lines is not None else _room_lines()
    )
    calls = _call_lines(view) if view is not None else _call_lines()
    for r in room + calls:
        if r["who"] in ("unknown", "", cfg.assistant_name()) or re.fullmatch(r"Speaker \d+|Caller \S*", r["who"] or ""):
            continue
        if r.get("sid") in clone_ids:
            continue
        b = person(r["who"])
        b["lines"] += 1
        b["room" if r["src"] == "room" else "calls"] += 1
        b["last"] = max(b["last"], r["ts"])
    # Named voices are people too, even before they've said a transcribed line (an owner enrolled from a recording).
    skip = {"unknown", "multiple speakers", cfg.assistant_name().lower()}
    for sp in speakers if speakers is not None else known_voices._speakers()["speakers"]:
        lab = (sp.get("label") or "").strip()
        if lab and lab.lower() not in skip and not VOICE_GROUP.fullmatch(lab) and sp["id"] not in clone_ids:
            b = person(lab)
            b["voices"].append(sp["id"])
            b.update(voice=True, enrolled=b.get("enrolled") or bool(sp.get("enrolled")))
            b["last"] = max(b["last"], sp.get("last_seen") or 0)
    try:
        for nm, n in _faces_mod().counts().items():
            if nm.strip() and not FACE_GROUP.fullmatch(_norm(nm)):
                person(nm)["faces"] += n
    except Exception:  # no face database yet, or the faces extra isn't installed
        pass
    book = _book()
    for nm in book.added_names():
        person(nm)
    for nm, number in book.phones().items():  # so the dashboard can place calls to them
        person(nm)["number"] = number
    seen, owner, described, here = _face_last_seen_at(), _owner_keys(), _attributes(), _here_now()
    named = views.names()  # each camera's name; a camera with no name goes by its id
    saved_shots = (
        face_shots.saved(context.STATE, revision=photo_files._shot_revision()) if saved_shots is None else saved_shots
    )
    for key, b in by.items():
        cam = seen[key][1] if key in seen else None
        b.update(
            last_heard=b["last"] or None,
            last_seen=seen[key][0] if key in seen else None,
            last_seen_cam=cam,
            last_seen_camera_name=named.get(cam, cam) if cam else None,
            here_now=(
                {"cam": here[key][0], "camera_name": named.get(here[key][0], here[key][0]), "since": here[key][1]}
                if key in here
                else None
            ),
            owner=key in owner,
            last_seen_pending=_seen.get("identity", {}).get("pending", False),
            last_seen_error=(
                _seen.get("identity", {}).get("error")
                if _seen.get("identity", {}).get("error") != "refresh_busy"
                else None
            ),
            **described.get(key, {}),
        )
        b["photo"] = (main_photo(b["name"], saved_shots=saved_shots) or {}).get("url")
    return sorted(by.values(), key=lambda x: (not x["owner"], -x["lines"], x["name"].lower()))


def _attributes() -> dict[str, dict[str, Any]]:
    """{normalized name: {"role", "organization", "aliases"}} for each person who has one. They live in the event log
    (people/registry.py, step 6 of docs/DOMAIN-MODEL.md); with no log yet, nobody has one."""
    if registry.log_exists(context.STATE):
        with registry.safely("roster attributes"):
            return {
                _norm(p.display_name): {k: getattr(p, k) for k in registry.ATTRIBUTES if getattr(p, k)}
                | ({"aliases": list(p.aliases)} if p.aliases else {})
                for p in registry.at(context.STATE).people()
            }
    return {}


class _People:
    """People's current names for Who's this? suggestions: a line's person, a recorded face's name as the face gallery
    spelled it (resolved through the event log, so a rename or a merge gives the person's name now), and the owner.
    Without an event log, names are as written and nobody is renamed."""

    def __init__(self) -> None:
        self.reg: registry.Registry | None = None
        if _has_log():
            with registry.safely("who's this suggestions"):
                self.reg = registry.at(context.STATE)
        self._names: dict[str, str | None] = {}

    def name(self, written: str) -> str | None:
        """The person a name written on a recorded face is now called, or None for someone forgotten."""
        if written not in self._names:
            if self.reg is None:
                self._names[written] = written
            else:
                pid = self.reg.known_as(written, refresh=False)
                person = self.reg.person(pid) if pid is not None else None
                self._names[written] = person.display_name if person is not None else None
        return self._names[written]

    def owner(self) -> list[str]:
        """The owner's names: as configured, and as the owner is called now (`name`)."""
        given = [n for n in (cfg.owner_name(), cfg.owner_label()) if n]
        return given + [m for n in given if (m := self.name(n))]


def _live_faces(window: float, now: float | None = None) -> list[dict[str, Any]]:
    """The faces the face worker found in the last `window` seconds (vision/sightings.py faces()), oldest first, each
    with the name it shows under: who it is, else its face group ("Face 3"), else "unknown". Empty when the broker
    can't be asked."""
    from liberational.vision import sightings

    now = time.time() if now is None else now
    try:
        found = sightings.faces(now - window, now + 1, state=context.STATE)
    except Exception as e:
        LOG.warning("faces: the vision log couldn't be read: %s", e)
        return []
    return [{**f, "label": f["name"] or f["cluster"] or "unknown"} for f in found]


HERE_NOW_S = 30  # "here now" means the face worker found and named them in the last 30 s


def _shot(row: Mapping[str, Any]) -> dict[str, Any]:
    """A ranked sighting (vision/sightings.py best() and best_spread()) as the card's Shot."""
    return {
        "id": row["id"],
        "url": f"/faces/shot/{row['id']}.jpg",
        "ts": row["ts"],
        "cam": row["cam"],
        "score": row["score"],
    }


def main_photo(
    name: str, *, ready: list[dict[str, Any]] | None = None, saved_shots: AbstractSet[str] | None = None
) -> dict[str, Any] | None:
    """A person's main photo as a Shot: their best-scoring recorded shot (sightings.best), else, for someone tagged
    live with no shot found yet, their newest saved crop (id, cam, and score None). A shot
    chosen on their card ("Use as main photo") comes first, and shots hidden from it are left out (card_prefs)."""
    from liberational.vision import sightings

    prefs = card_prefs.get(context.STATE, name)
    saved_shots = (
        face_shots.saved(context.STATE, revision=photo_files._shot_revision()) if saved_shots is None else saved_shots
    )
    if ready is None:
        ready, _pending, _unavailable = _ready_faces(
            [r for r in sightings.ranked(name, context.STATE) if r["id"] not in prefs["hidden"]["shot"]], saved_shots
        )
    if prefs["main"] is not None and (pinned := sightings.ranked_shot(name, prefs["main"], context.STATE)):
        selected, _pending, _unavailable = _ready_faces([pinned], saved_shots)
        if selected:
            return _shot(pinned)
    if ready:
        return _shot(ready[0])
    newest = photo_files._photos(name)[:1]
    if not newest:
        return None
    url = f"/people/{urllib.parse.quote(name)}/photos/{newest[0].name}"
    return {"id": None, "url": url, "ts": int(newest[0].stem) / 1000, "cam": None, "score": None}


# (normalized name, camera) -> (in view since, last seen): when each person's current run on a camera began. Each
# call reads only the last HERE_NOW_S of faces, so a run that outlasts it is remembered here, for as long as it doesn't
# break. A run first seen here is looked back for in the faces found earlier (_run_began), so a restart, or a night
# with nobody watching, still gives the time they arrived. _here_cam is the camera each person is shown on, kept while
# it still sees them so two cameras seeing someone don't take turns. Requests run on threads: _here_lock guards both.
_here_since: dict[tuple[str, str], tuple[float, float]] = {}


_here_cam: dict[str, str] = {}


_here_lock = threading.Lock()


LOOK_BACK_S = 12 * 3600  # how far back the faces found are searched for the start of a run


RUN_STEP_S = 600  # how much of that is read at a time (a camera can find a face every frame)


def _run_began(key: str, cam: str, start: float) -> float:
    """When the run of `key` on `cam` that was going at `start` began, from the faces the face worker found there:
    walking back RUN_STEP_S at a time, as far as LOOK_BACK_S, until no face is found there for more than HERE_NOW_S
    or the faces found aren't theirs for that long. `start` when there's no history."""
    from liberational.vision import sightings

    hi = first = start
    try:
        while first - hi < LOOK_BACK_S:
            rows = sightings.faces(hi - RUN_STEP_S, hi, cam, state=context.STATE)
            for f in reversed(rows):
                if start - f["ts"] > HERE_NOW_S:
                    return start
                if _norm(f["name"]) == key:
                    start = f["ts"]
            if not rows:  # nobody was seen there in that step
                return start
            hi -= RUN_STEP_S
    except Exception as e:
        LOG.warning("here: the vision log couldn't be read: %s", e)
    return start


def _here_now() -> dict[str, tuple[str, float]]:
    """{normalized name: (camera, in view there since)} for each named face the face worker found in the last
    HERE_NOW_S seconds. "Since" is when their run on that camera began: seen with no gap longer than HERE_NOW_S,
    across the calls that saw it, or before that in the faces found earlier (_run_began). Someone two cameras see stays
    on the one they were shown on."""
    now = time.time()
    runs: dict[tuple[str, str], tuple[float, float]] = {}
    for f in _live_faces(HERE_NOW_S, now):  # oldest first
        ts, cam, key = f["ts"], f.get("cam"), _norm(f["name"])
        if not cam or not key or key == "unknown" or FACE_GROUP.fullmatch(key):
            continue
        start, last = runs.get((key, str(cam)), (ts, ts))
        runs[(key, str(cam))] = (start if ts - last <= HERE_NOW_S else ts, ts)
    current = {k: v for k, v in runs.items() if now - v[1] <= HERE_NOW_S}
    with _here_lock:
        fresh = [k for k in current if k not in _here_since or current[k][0] - _here_since[k][1] > HERE_NOW_S]
    began = {k: _run_began(k[0], k[1], current[k][0]) for k in fresh}  # outside the lock: it reads a database
    out: dict[str, tuple[str, float]] = {}
    with _here_lock:
        for (key, cam), (start, last) in current.items():
            before = _here_since.get((key, cam))
            if (key, cam) in began:
                start = min(start, began[(key, cam)])
            elif before is not None:  # the same run as last time: it began then
                start = min(start, before[0])
            _here_since[(key, cam)] = (start, last)
        for k in [k for k, (_, last) in _here_since.items() if now - last > HERE_NOW_S]:
            _here_since.pop(k, None)
        for key in {k for k, _ in current}:
            cams = {cam: _here_since[(k, cam)] for k, cam in current if k == key}
            shown = _here_cam.get(key)
            if shown is None or shown not in cams:
                shown = max(cams, key=lambda c: cams[c][1])  # the camera that saw them last
            _here_cam[key] = shown
            out[key] = (shown, cams[shown][0])
        for key in [k for k in _here_cam if k not in out]:
            _here_cam.pop(key, None)
    return out


def _has_log() -> bool:
    """Whether the event log exists. Opening the registry makes it, so a dual write checks first (as the backfill)."""
    return registry.log_exists(context.STATE)


PEOPLE_ADDED = context.STATE / contacts.ADDED  # read models of the person registry (people/contacts.py)


CONTACTS_FILE = context.STATE / contacts.CONTACTS


def _book() -> contacts.Book:
    return contacts.Book(context.STATE, CONTACTS_FILE, PEOPLE_ADDED)


def _person_name(raw: object) -> str | None:
    """A name typed for a person: letters in any script, then letters, digits, spaces, . ' or - (at most 40). Never a
    label that names nobody, in any capitals: an automatic one ("Speaker 2", "Face 5"), "unknown", "multiple
    speakers", or the assistant's name. A rename to "Speaker 2" merged the person into that anonymous voice."""
    n = re.sub(r"\s+", " ", str(raw or "")).strip()[:40].rstrip()  # cut, then strip: no trailing space at the cut
    if not re.fullmatch(r"[^\W\d_][\w .'-]*", n):
        return None
    automatic = FACE_GROUP.fullmatch(n.casefold()) or VOICE_GROUP.fullmatch(n.capitalize())
    return n if registry.is_name(n) and not automatic else None
