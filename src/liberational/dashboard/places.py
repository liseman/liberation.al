"""Known › Places' reads: the places the house knows, and where people and pets were in one.

    GET /places                        {places: [{id, name, named, plan, plan_source, machine_newer, revision,
                                       last_visit, last_activity}]}: every place (memory/places.py), its plan in the
                                       floorplan DSL (the owner's correction if there is one, else the newest machine
                                       plan; null before the first), when the owner was last there (a `place.visit`,
                                       the rig's), and when anyone was last seen there by its fixed cameras (a
                                       position record; within ACTIVITY_LOOKBACKS_S), each null when unknown (a page
                                       load with text/html gets the app instead)
    POST /places/plan {place, plan, base}
                                       the owner's corrected plan (DSL), saved as taught state over the machine's
                                       (memory/places.py `correct_plan`): it holds until withdrawn, and the
                                       machine's plan is never edited. `base` is the revision whose frame the edited
                                       text is in (the plan's plan_revision when the editor opened it). An empty plan
                                       withdraws the correction. An edit of a plan drawn on its own page (an
                                       imported one) is on that page, and is moved into the plan frame.
    GET /places/plan/history?place=    {place, versions: [{id, at, rooms, source, withdrawn}], previous}: the owner's
                                       plans, oldest first, and the one Revert plan puts back (`plan_versions`)
    POST /places/plan/restore {place, record}
                                       make that earlier owner plan hold again, written as the newest record, so the
                                       one it replaces stays in the history (memory/places.py `restore_plan`)
    GET /places/scene?place=           the 3D view's scene: the newest revision with a map, its map's URL and
                                       size, its splat's when it has one (its own, else the newest made in the same
                                       map frame; one from another frame is only named, `splat_stale`) and what
                                       it was trained from (`splat_from`, as its trainer named it), which state the
                                       splat is in (`splat_state`: "current", "improving", "earlier", "stale", or
                                       "none") and, while the map has no splat of its own, why (`splat_wait`), its
                                       floor plane, plan transform, registered cameras, and the plan in its frame;
                                       {revision: null} while the place has no map. Either way `learning` says how
                                       the map is learned from the cameras (`_learning`): whether learning is on, how
                                       many cameras it samples, whether a view is being added now, how many views
                                       the place's map was built from, and when it last improved. The page asks
                                       again every few seconds while it's open, and draws a newer map or splat
                                       when the reply names one
    GET /places/map?place=&revision=   that revision's map file (a point cloud, PLY), with byte ranges, an ETag, and
                                       caching for good: a revision's files never change once written
    GET /places/splat?place=&revision=&format=sog
                                       that revision's 3D Gaussian splat (SOG), served the same way; the scene
                                       names it (`splat`, null while the revision has none). The URL and the ETag
                                       name the format, so bytes a browser kept in an earlier encoding can't
                                       answer for it
    GET /places/positions?place=&start=&end=[&entity=]
                                       {place, start, end, entities, tracks, rooms, cameras, camera_notes}: where
                                       each entity was on the place's plan over [start, end), from the position
                                       layer's stored records (spatial/positions.py); how many cameras can place
                                       people there (`cameras`: with none, nobody can be, and the page says so); and
                                       each camera of the place with how it's posed and what it lacks
                                       (`camera_notes`: [{camera, by: "spot" | "map", missing}], `missing` any of
                                       "height", "facing", "lens", "tilt", "check", spatial/spots.py); the read
                                       stops if the client hangs up

The GETs only read state. The POSTs are teaching, so they follow dashboard.remote_teaching like the rest.

Positions are grouped by entity: a known one by its id ("person:<id>", "pet:<id>", "rig", named from faces and pet
labels when read, spatial/positions.py's `identify`), someone not known yet by their camera track. Each path
(`tracks[].track`) is a camera track, or tracks on several cameras that the reader fused into one (positions.py's
`joined`: one entity's, or one unnamed person's seen by two cameras at once; their "+"-joined ids, one point a
second where both cameras saw them), keyed by its entity, else its path; tracks the reader didn't join stay separate
paths, since joining them would zig-zag between where each camera put them. A point the reader marks uncertain (a foot
on an occluding edge no other camera settled) is left off its path: it can be metres off. A point with neither
an entity nor a track can't be joined to anyone's path (joining them zig-zags between people), so it's left out; a
room-only span with neither still says someone was in the room. A sample with plan coordinates is a point on its
track's path; a zone visit (room only, no x and y) is a span in `rooms`, with the plan's room id when it has one and
the zone's name either way. A long window is thinned so a week stays drawable, by distance and time (`moved`): a path
keeps each point that moved MOVED_M from the last one kept, so a walk keeps its shape, and otherwise one point
every span / MAX_POINTS seconds, with its last; the reader first thins to FINE_SHARE times finer than that (its
`every`) as it scans the stored records, for it to choose from. What's left is capped at MAX_POINTS per path, evenly, keeping its first and last point.

Points come back in the frame of the plan the tab draws (the place's `plan_revision`: records are placed in it when
written, and the reader moves one placed in an earlier frame onto it), turned onto the owner's page when the plan is drawn on one (an imported plan, upright and with
the page's handedness: `GET /places` gives that page's DSL as `plan`; the 3D scene keeps the plan frame). A point that
can't be put in that frame (`in_frame` false: a camera that revision doesn't register) would land off the drawn plan, so
it's left out; a room-only span is kept either way.
"""

import importlib
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from liberational import logs
from liberational.dashboard import routes
from liberational.memory import places

if TYPE_CHECKING:
    from liberational.dashboard.server import H

Reply = tuple[int, dict[str, Any]]

MAX_POINTS = 2000  # per path, in one reply
# A point that moved this far (metres) from the last one kept on its path is kept whatever the window: a walk across the
# house in 20 s stays a walk on a Day's plan, not a straight line between its two ends 43 s apart. Standing still, a
# path keeps one point every span / MAX_POINTS seconds as before.
MOVED_M = 0.3
FINE_SHARE = 20  # the reader's own thinning is this much finer than span / MAX_POINTS, for `moved` to choose from
MAX_SPAN_S = 31 * 86400  # the longest window a query may ask for
_LATEST = 4102444800.0  # 2100-01-01: well past any recording, far inside what a float time can hold
LAST_VISIT_LOOKBACK_S = 365 * 86400  # how far back "last visit" looks
#: How far back "last activity" looks, in widening steps: a place seen in the last hour costs an hour's read, and one
#: not seen all week costs a week's, never more.
ACTIVITY_LOOKBACKS_S = (3600, 86400, 7 * 86400)


def last_activity(state: Path, now: float, ids: Iterable[str]) -> dict[str, float]:
    """When anyone was last seen in each place, by the position layer's newest record there
    (spatial/positions.py's `last_seen`). A place visit is the owner's (the rig was there), so a place only its fixed
    cameras watch has none; this is the activity, apart from visits. Read newest first, over ACTIVITY_LOOKBACKS_S,
    each place only until it's found and only its own records; a place not seen in the longest is left out."""
    from liberational.spatial import positions

    return positions.last_seen(state, sorted(set(ids)), now, ACTIVITY_LOOKBACKS_S)


def listing(state: Path, now: float | None = None) -> Reply:
    """GET /places: every place, with its plan and when the house was last there."""
    now = time.time() if now is None else now
    last: dict[str, float] = {}
    for v in places.visits(state, now - LAST_VISIT_LOOKBACK_S, now):
        last[v.place] = max(last.get(v.place, 0.0), float(v.end))
    known = places.places(state)
    seen = last_activity(state, now, [p.id for p in known])
    out = []
    for p in known:
        rev = p.revision
        out.append(
            {
                "id": p.id,
                "name": p.name,
                "named": p.named,
                # an owner plan drawn on its own page is shown as drawn; positions come turned onto it
                "plan": p.view.plan if p.view is not None else p.plan,
                "plan_source": p.plan_source,
                # An owner's plan is the place's only plan: no machine plan is offered beside it, newer or not.
                "machine_newer": False,
                "revision": {"id": rev.id, "at": rev.at, "producer": rev.producer} if rev else None,
                "plan_revision": p.plan_revision,
                "machine_plan": None,
                "last_visit": last.get(p.id),
                "last_activity": seen.get(p.id),
            }
        )
    return 200, {"places": out}


def _map_revision(state: Path, place: str, revision: str | None = None) -> places.Revision | None:
    """The revision whose map the 3D view shows: `revision` if it's one of the place's and its map file is still kept
    (maps are pruned to the newest few), else (with none asked for) the newest such revision. A revision's map is
    always a bare `<uuid>.map` in the place's folder: memory/places.py refuses any other name when it reads one."""
    chain = places.revisions(state, place)
    found = [r for r in chain if r.map and (places.root(state) / place / r.map).is_file()]
    if revision is not None:
        return next((r for r in found if r.id == revision), None)
    return found[-1] if found else None


def map_file(state: Path, place: str, revision: str) -> Path | None:
    """GET /places/map: the file of `revision`'s map, or None (no such place or revision, or its map is gone)."""
    if places.get(state, place) is None:
        return None
    rev = _map_revision(state, place, revision)
    return places.root(state) / place / rev.map if rev and rev.map else None


def splat_file(state: Path, place: str, revision: str) -> Path | None:
    """GET /places/splat: `revision`'s splat (SOG), or None (no such place or revision, or it has no splat)."""
    if places.get(state, place) is None:
        return None
    rev = next((r for r in places.revisions(state, place) if r.id == revision), None)
    return places.splat_path(state, rev, SPLAT_FORMAT) if rev and places.has_splat(state, rev) else None


SPLAT_FORMAT = "sog"  # what GET /places/splat serves, and what its URL and ETag name


LEARNED = "surroundings"  # what a learned map's producer starts with, and what its splat says it was trained from


def _summary(state: Path) -> dict[str, Any]:
    """What the map step knows (spatial/surroundings_map.py's `summary`, #258): the switch, the cameras sampled, the
    kept frames waiting for a place, whether a step is running, and each place's views and when it last improved.
    On an install without the map step, only the switch."""
    try:
        step = importlib.import_module("liberational.spatial.surroundings_map")
    except ImportError:
        from liberational.spatial import surroundings

        return {"on": surroundings.read().on}
    return step.summary(state)


def _learning(state: Path, place: str) -> dict[str, Any]:
    """How `place`'s map is learned from the cameras, for the 3D view to say in plain words: `on` (the Live tab's
    switch), `cameras` (how many it samples now; none while off), `improving` (on, and a kept frame is waiting to be
    placed or a map step is running), `views` (the kept frames the place's map was built from; None for a place the
    map step hasn't built), and `improved` (when a step last added a view to it, epoch seconds, or None)."""
    told = _summary(state)
    on = told.get("on") is True
    by_place = told.get("places")
    mine = by_place.get(place) if isinstance(by_place, dict) else None
    mine = mine if isinstance(mine, dict) else {}
    sampling = told.get("sampling")
    return {
        "on": on,
        "cameras": len(sampling) if isinstance(sampling, list) else 0,
        "improving": on and bool(told.get("waiting") or told.get("running")),
        "views": mine.get("views"),
        "improved": mine.get("improved"),
    }


def _splat_wait(state: Path, rev: places.Revision) -> dict[str, Any]:
    """Why `rev` has no splat of its own: `reason` is the splat pass's own note when it has tried this revision
    ("few_frames", "no_cameras", "no_runtime", or "failed", with `tried`, when), else "learned" for a map learned
    from the cameras (its splat is written right after it, so it's on its way or its training failed), else
    "no_cameras" for a map with no cameras registered (there is nothing to make one from), else "pass": no pass has
    tried this map. `pass_at` is the time of day a scheduled pass runs; None, since nothing makes splats on a
    schedule at present (`pack-spatial-splat` does by hand)."""
    note = places.splat_note(state, rev)
    if note is not None:
        reason, tried = note["reason"], note["at"]
    elif rev.producer.startswith(LEARNED):
        reason, tried = "learned", None
    elif not rev.cameras:
        reason, tried = "no_cameras", None
    else:
        reason, tried = "pass", None
    return {"reason": reason, "tried": tried, "pass_at": None}


def scene(state: Path, place: str) -> Reply:
    """GET /places/scene: what the 3D view draws for a place, all in its map's frame (metres). While the place has an
    owner's plan, that is the plan, whatever the map: `plan_source` is "owner", and no machine plan is given in its
    place. It's laid on the map only when it's drawn in this revision's frame (`plan_transform`, with `plan_aligned`
    true); until a learned map is lined up with it the transform is None, so the view draws the map alone, and
    `plan_lineup` says why (spatial/plan_lineup.py `status`). A place with no owner's plan gives the revision's own
    machine plan, in its frame."""
    held = places.get(state, place)
    if held is None:
        return 404, {"error": "no such place"}
    learning = _learning(state, place)
    rev = _map_revision(state, place)
    if rev is None or rev.map is None:
        return 200, {"place": place, "revision": None, "learning": learning}
    by_id = {r.id: r for r in places.revisions(state, place)}
    drawn = by_id.get(held.plan_revision or "")
    from liberational.spatial import plan_lineup  # numpy and scipy: only a scene needs them

    owned = held.plan_source == "owner"
    aligned = owned and drawn is not None and places.same_plan_frame(drawn, rev)
    try:
        size = (places.root(state) / place / rev.map).stat().st_size
    except FileNotFoundError:  # pruned since it was found
        return 200, {"place": place, "revision": None, "learning": learning}
    # The revision's own splat, else the newest one made in the same map frame (a later revision that kept the
    # frame); a splat of another frame (or an unknown one) would sit out of place, so it's only named as stale.
    chain = places.revisions(state, place)
    with_splat = [r for r in chain if places.has_splat(state, r)]
    fits = [r for r in with_splat if r.id == rev.id or (rev.frame is not None and r.frame == rev.frame)]
    shown = next((r for r in reversed(fits) if r.id == rev.id), fits[-1] if fits else None)
    try:
        splat_size = places.splat_path(state, shown, SPLAT_FORMAT).stat().st_size if shown else None
    except FileNotFoundError:  # pruned since it was found
        shown, splat_size = None, None
    stale = with_splat[-1] if shown is None and with_splat else None
    # Which state the view is in, so the page doesn't guess: this map's own splat (still improving while learning
    # is adding views to it), an earlier map's that still lines up (same frame), one that doesn't (named, not
    # offered), or none. A revision that kept its map (one that only set the floor or a camera) has the splat of the
    # revision that wrote that map: it's the same map's.
    own = shown is not None and (shown.id == rev.id or shown.map == rev.map)
    if shown is not None and own:
        learned = places.splat_source(state, shown) == LEARNED
        splat_state = "improving" if learned and learning["improving"] else "current"
    elif shown is not None:
        splat_state = "earlier"
    else:
        splat_state = "stale" if stale is not None else "none"
    splat_query = {"place": place, "revision": shown.id if shown else "", "format": SPLAT_FORMAT}
    if owned:
        # an owner plan drawn in a revision with no plan file is in its floor's frame; one not drawn in this map's
        # frame has no place on it yet
        transform = places.plan_frame(rev) if aligned else None
    else:
        transform = rev.plan_transform
    return 200, {
        "place": place,
        "revision": rev.id,
        "at": rev.at,
        "map": "/places/map?" + urlencode({"place": place, "revision": rev.id}),
        "map_format": rev.map_format,
        "map_bytes": size,
        "splat": None if shown is None else "/places/splat?" + urlencode(splat_query),
        "splat_format": None if shown is None else SPLAT_FORMAT,
        "splat_bytes": splat_size,
        "splat_revision": None if shown is None else shown.id,
        "splat_stale": None if stale is None else {"revision": stale.id, "at": stale.at},
        "splat_from": None if shown is None else places.splat_source(state, shown),
        "splat_state": splat_state,
        "splat_at": None if shown is None else shown.at,
        "splat_wait": None if own else _splat_wait(state, rev),
        "learning": learning,
        "floor": list(rev.floor) if rev.floor else None,
        "plan_transform": list(transform) if transform else None,
        "cameras": {cam: pose.as_dict() for cam, pose in rev.cameras.items()},
        "plan": held.plan if owned else places.machine_plan(state, rev),
        "plan_source": "owner" if owned else "machine",
        "plan_aligned": aligned if owned else transform is not None,
        "plan_lineup": plan_lineup.status(state, place) if owned else None,
    }


MAX_PLAN_BYTES = 64 * 1024  # a hand-corrected plan is a few KB; anything this big isn't one


def correct(body: Mapping[str, Any], control: bool, state: Path, now: float | None = None) -> Reply:
    """POST /places/plan: save the owner's corrected plan for a place, or withdraw it with an empty plan. `control` is
    the handler's _can_teach. The tab is the real validator: it reads the plan with the floorplan library and shows its
    findings before it offers to save. This route only checks that the plan has a room, that it isn't huge, and that
    `base` (when sent) is one of the place's revisions."""
    from liberational.spatial import floorplan

    if not control:
        return 403, {"ok": False, "error": "plans can only be corrected from this machine (Remote teaching is off)"}
    place, plan, base = body.get("place"), body.get("plan"), body.get("base")
    if not isinstance(place, str) or not isinstance(plan, str) or not (base is None or isinstance(base, str)):
        return 400, {"ok": False, "error": "send the place's id, its plan as text, and the revision it was drawn in"}
    if len(plan.encode()) > MAX_PLAN_BYTES:
        return 400, {"ok": False, "error": "that plan is too long to be a floorplan"}
    if places.get(state, place) is None:
        return 404, {"ok": False, "error": "no such place"}
    if plan.strip() and not floorplan.rooms_from_dsl(plan):
        return 400, {"ok": False, "error": "the plan has no room in it"}
    held = places.get(state, place)
    try:
        if plan.strip() and held is not None and held.view is not None:
            # the tab shows an owner plan drawn on its own page as drawn, so an edit of it is on that page too
            plan, base, view = _from_page(plan, held.view), held.plan_revision, {**held.view.as_dict(), "plan": plan}
            places.correct_plan(state, place, plan, now=now, base=base, view=view)
        else:
            places.correct_plan(state, place, plan if plan.strip() else "", now=now, base=base)
    except ValueError as e:
        return 400, {"ok": False, "error": str(e)}
    held = places.get(state, place)
    shown = (held.view.plan if held.view is not None else held.plan) if held else None
    return 200, {"ok": True, "plan": shown, "plan_source": held.plan_source if held else None}


def plan_versions(state: Path, place: str) -> Reply:
    """GET /places/plan/history?place=: the place's owner plans (memory/places.py `plan_history`), oldest first, each
    {id, at, rooms, source, withdrawn}, and `previous`: the version Revert plan puts back, or null with none. That's
    the newest owner plan before the one that holds that isn't the same plan (`same_plan`: a re-registration of one
    plan into a newer frame moves its corners by millimetres, and putting that back would change nothing the owner
    can see), so reverting a revert goes back again; while the machine's plan holds, the newest owner plan."""
    from liberational.spatial import floorplan

    held = places.get(state, place)
    if held is None:
        return 404, {"error": "no such place"}
    history = places.plan_history(state, place)
    texts = [_plan_text(r.value) for r in history]
    versions = [
        {
            "id": r.id,
            "at": r.start_ns / 1e9,
            "rooms": len(floorplan.rooms_from_dsl(t)),
            "source": r.value.get("source", "owner"),
            "withdrawn": not t.strip(),
        }
        for r, t in zip(history, texts, strict=True)
    ]
    # the newest record is the one that holds while the owner's plan does; with the machine's holding, none of them
    current = texts[-1] if held.plan_source == "owner" and texts else None
    upto = len(history) - 1 if current is not None else len(history)
    previous = next(
        (
            versions[i]
            for i in range(upto - 1, -1, -1)
            if not versions[i]["withdrawn"] and (current is None or not same_plan(texts[i], current))
        ),
        None,
    )
    return 200, {"place": held.id, "versions": versions, "previous": previous}


def restore(body: Mapping[str, Any], control: bool, state: Path, now: float | None = None) -> Reply:
    """POST /places/plan/restore {place, record}: make an earlier owner plan hold again (Revert plan), written as the
    newest record (memory/places.py `restore_plan`), so the one it replaces stays in the history and can be put back
    the same way. A teaching write, like POST /places/plan."""
    if not control:
        return 403, {"ok": False, "error": "plans can only be changed from this machine (Remote teaching is off)"}
    place, record = body.get("place"), body.get("record")
    if not isinstance(place, str) or not isinstance(record, str):
        return 400, {"ok": False, "error": "send the place's id and the plan record to put back"}
    if places.get(state, place) is None:
        return 404, {"ok": False, "error": "no such place"}
    try:
        places.restore_plan(state, place, record, now=now)
    except KeyError:
        return 404, {"ok": False, "error": "no such plan in this place's history"}
    held = places.get(state, place)
    shown = (held.view.plan if held.view is not None else held.plan) if held else None
    return 200, {"ok": True, "plan": shown, "plan_source": held.plan_source if held else None}


SAME_PLAN_M = 0.01  # two plans' corners this close are one plan: editor drags snap to 5 cm
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def same_plan(a: str, b: str, within: float = SAME_PLAN_M) -> bool:
    """Whether two plans' DSL is one plan as the owner draws it: the same rooms, walls, and openings, each number within
    `within` (metres) of the other's, as one plan registered into two nearly equal frames is (Place 1's imported plan,
    registered again, moved by 3 mm). Spatial memory's `x-` lines (cameras placed on the plan, anchors) aren't the
    owner's drawing, and the editor sets them aside too, so they don't count."""
    a, b = _drawn(a), _drawn(b)
    na, nb = _NUMBER.findall(a), _NUMBER.findall(b)
    return (
        _NUMBER.split(a) == _NUMBER.split(b)
        and len(na) == len(nb)
        and all(abs(float(x) - float(y)) <= within for x, y in zip(na, nb, strict=True))
    )


def _drawn(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("x-"))


def _plan_text(value: Mapping[str, Any]) -> str:
    plan = value.get("plan")
    return plan if isinstance(plan, str) else ""


def _from_page(text: str, view: places.PageView) -> str:
    """An owner plan's DSL edited on its page, moved into the plan frame (spatial/plan_import.py). Raises ValueError
    for a plan it can't read."""
    from liberational.spatial import plan_import

    parsed = plan_import.parse(text)
    return plan_import.to_dsl(parsed, plan_import.Move(view.turn, view.shift, view.mirror), parsed.title)


def _kind(entity: str) -> str:
    head = entity.split(":", 1)[0]
    if head == "face":  # an unnamed face group: one person, drawn and listed as one
        return "person"
    return head if head in ("person", "pet", "rig") else "unknown"


def thin[T](points: Sequence[T], most: int = MAX_POINTS) -> list[T]:
    """At most `most` of `points`, evenly spaced, the first and last always among them."""
    if len(points) <= most:
        return list(points)
    if most < 2:
        return [points[-1]]
    step = (len(points) - 1) / (most - 1)
    return [points[round(i * step)] for i in range(most)]


def moved(points: Sequence[Mapping[str, Any]], every: float, step: float = MOVED_M) -> list[Mapping[str, Any]]:
    """A path's points (in time order, each with t, x, and y) thinned by distance and time: each one `step` metres or
    more from the last kept, or `every` seconds or more after it, and the last (where the path ends)."""
    out: list[Mapping[str, Any]] = []
    for p in points:
        last = out[-1] if out else None
        if last is None or p["t"] - last["t"] >= every or math.hypot(p["x"] - last["x"], p["y"] - last["y"]) >= step:
            out.append(p)
    if points and out[-1] is not points[-1]:
        out.append(points[-1])
    return out


def _epoch(value: str) -> float:
    t = float(value)
    if not math.isfinite(t) or not 0 <= t <= _LATEST:
        raise ValueError(value)
    return t


def positions_view(query: Mapping[str, str], state: Path, cancelled: Callable[[], bool] | None = None) -> Reply:
    """GET /places/positions: one place's positions over a window, grouped by who they are. `cancelled` says whether
    the client has gone: the read asks it as it goes and raises `positions.Cancelled` when so."""
    from liberational.spatial import positions

    place = query.get("place") or ""
    try:
        start, end = _epoch(query.get("start", "")), _epoch(query.get("end", ""))
    except ValueError:
        return 400, {"error": "start and end are epoch times"}
    if not place or end <= start or end - start > MAX_SPAN_S:
        return 400, {"error": "a place, and a window of at most 31 days that ends after it starts"}
    held = places.get(state, place)
    if held is None:
        return 404, {"error": "no such place"}
    every = (end - start) / MAX_POINTS
    fine = every / FINE_SHARE
    try:
        rows = positions.positions(
            state,
            start,
            end,
            place=place,
            entity=query.get("entity") or None,
            every=fine if fine >= 1 else None,
            frame_revision=held.plan_revision,
            cancelled=cancelled,
        )
    except ValueError as e:  # a frame the reader can't use: an unknown revision, or one of another place
        return 400, {"error": str(e)}
    notes = positions.placing(state, place)
    entities: dict[str, dict[str, Any]] = {}
    tracks: dict[tuple[str, str], list[dict[str, Any]]] = {}  # (entity, its path's track) to the path's points
    rooms: list[dict[str, Any]] = []
    for r in rows:
        key = r.entity or r.path or r.track
        if key is None and (r.x is not None or not r.room):
            continue  # a point no path can claim
        key = key or "someone"
        if key not in entities:
            entities[key] = {"id": key, "name": r.name, "kind": _kind(key)}
        elif r.name and not entities[key]["name"]:
            entities[key]["name"] = r.name
        if r.uncertain:
            continue  # a foot on an occluding edge no other camera settled: drawn, it jumps metres off the path
        if r.x is not None and r.y is not None and r.in_frame:
            x, y = held.view.to_page(r.x, r.y) if held.view is not None else (r.x, r.y)
            tracks.setdefault((key, r.path or r.track or key), []).append(
                {
                    "t": r.t,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "room": r.room_id,
                    "confidence": r.confidence,
                    "source": r.source,
                }
            )
        elif r.room:
            rooms.append(
                {
                    "entity": key,
                    "room": r.room_id,
                    "name": r.room,
                    "start": r.t,
                    "end": max(r.end, r.t),
                    "confidence": r.confidence,
                    "source": r.source,
                }
            )
    return 200, {
        "place": place,
        "start": start,
        "end": end,
        "entities": list(entities.values()),
        "tracks": [{"entity": k, "track": path, "points": thin(moved(v, every))} for (k, path), v in tracks.items()],
        "rooms": rooms,
        # with no camera that has a pose (from its spot on the owner's plan, or in the place's map), the placer has
        # nothing to place a footprint by; a camera whose spot lacks something says what
        "cameras": sum(1 for c in notes if not c["missing"]),
        "camera_notes": notes,
    }


# ---- routes (dashboard/routes.py): what the server answers for this area ----


@routes.get("/places")
def _get_places(h: H) -> None:
    code, reply = listing(h.state)
    return h._json(reply, code)


@routes.get("/places/scene")
def _get_places_scene(h: H) -> None:
    code, reply = scene(h.state, h._query().get("place", ""))
    return h._json(reply, code)


@routes.get("/places/map", "/places/splat")
def _get_places_map(h: H) -> None:
    p = h.path.split("?")[0]
    q = h._query()
    files = map_file if p == "/places/map" else splat_file
    found = files(h.state, q.get("place", ""), q.get("revision", ""))
    # A splat is asked for by format: a URL without one is an earlier page's, whose bytes were another encoding.
    splat = p == "/places/splat"
    if found is None or (splat and q.get("format", SPLAT_FORMAT) != SPLAT_FORMAT):
        return h._json({"error": f"no such {p.rsplit('/', 1)[1]}"}, 404)
    # A revision's map and splat never change once written: its id (and the splat's format) is their ETag, and they
    # can be kept for good.
    tag = f'"{q.get("revision", "")}.{SPLAT_FORMAT}"' if splat else f'"{q.get("revision", "")}"'
    if h.headers.get("If-None-Match") == tag:
        h.send_response(304)
        h.send_header("ETag", tag)
        h.end_headers()
        return None
    return h._media(found, "application/octet-stream", {"ETag": tag}, cache="private, max-age=31536000, immutable")


@routes.get("/places/positions")
def _get_places_positions(h: H) -> None:
    from liberational.spatial import positions

    try:
        code, reply = positions_view(h._query(), h.state, cancelled=h._client_gone)
    except positions.Cancelled:
        h.close_connection = True  # nobody is reading: nothing to send
        return logs.get("dashboard").debug("places positions read stopped: the client hung up")
    return h._json(reply, code)


@routes.get("/places/plan/history")
def _get_places_plan_history(h: H) -> None:
    code, reply = plan_versions(h.state, h._query().get("place", ""))
    return h._json(reply, code)


@routes.post("/places/plan")
def _post_places_plan(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = correct(body, h._can_teach(), h.state)
    return h._json(o, code)


@routes.post("/places/plan/restore")
def _post_places_plan_restore(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = restore(body, h._can_teach(), h.state)
    return h._json(o, code)
