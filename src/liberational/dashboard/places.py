"""Known › Places' reads: the places the house knows, and where people and pets were in one.

    GET /places                        {places: [{id, name, named, plan, plan_source, machine_newer, revision,
                                       last_visit, last_activity}]}: every place (memory/places.py), its plan as
                                       stored, a Pascal scene's JSON (the owner's if there is one, else the newest
                                       machine plan; null before the first), when the owner was last there (a `place.visit`,
                                       the rig's), and when anyone was last seen there by its fixed cameras (a
                                       position record; within ACTIVITY_LOOKBACKS_S), each null when unknown (a page
                                       load with text/html gets the app instead)
    POST /places/plan {place, plan, base}
                                       the owner's plan (a Pascal scene), saved as taught state over the machine's
                                       (memory/places.py `correct_plan`): it holds until withdrawn, and the
                                       machine's plan is never edited. `base` is the revision whose frame the edited
                                       text is in (the plan's plan_revision when the editor opened it). An empty plan
                                       withdraws the correction. An edit of a plan drawn on its own page (an
                                       imported one) is on that page, and is moved into the plan frame.
    GET /places/plan/history?place=    {place, versions: [{id, at, rooms, source, withdrawn, current, summary}]}:
                                       the owner's plans, oldest first (`plan_versions`)
    GET /places/plan/scene?place=[&record=]
                                       {place, scene, base}: the plan as the plan editor opens it, or one of its
                                       earlier versions (`plan_as_scene`)
    POST /places/plan/restore {place, record}
                                       make that earlier owner plan hold again, written as the newest record, so the
                                       one it replaces stays in the history (memory/places.py `restore_plan`)
    GET /places/scene?place=[&camera=][&all=1]
                                       the 3D view's scene: the newest revision with a map, its map's URL and
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
                                       when the reply names one. `views` lists the cameras whose own clouds
                                       the view offers (`camera_views`); with `camera`, or with no map of the place,
                                       the scene is one camera's own cloud, named in `view` (`_camera_scene`).
                                       With `all=1` it is every camera's cloud placed in the place
                                       (`_all_cameras_scene`, spatial/cloud_scene.py): the clouds' points as one
                                       file, the owner's plan under them, and `combined`: how each camera is placed
                                       or why it isn't, and what each pair of cameras' pictures said
    POST /places/scene/spot {place, camera}
                                       take the spot the scene suggests for a camera that moved: that camera's spot
                                       on the owner's plan is moved and turned to it, saved as a new version of the
                                       plan (`accept_spot`); nothing is saved until this is asked
    POST /places/scene/spot/undo {place, camera, record}
                                       take that back: the plan version it replaced holds again, and the camera
                                       is a moved camera once more (`undo_spot`)
    GET /places/map?place=&revision=[&camera=]
                                       that revision's map file (a point cloud, PLY), with byte ranges, an ETag, and
                                       caching for good: a revision's files never change once written. With
                                       `camera`, that camera's own cloud, whose id is the revision; with
                                       `scene=1`, the points of the scene of all cameras, whose id it is
    GET /places/splat?place=&revision=&format=sog
                                       that revision's 3D Gaussian splat (SOG), served the same way; the scene
                                       names it (`splat`, null while the revision has none). The URL and the ETag
                                       name the format, so bytes a browser kept in an earlier encoding can't
                                       answer for it
    GET /places/positions?place=&start=&end=[&entity=]
                                       {place, start, end, entities, tracks, rooms, cameras, camera_notes, map}: where
                                       each entity was on the place's plan over [start, end), from the position
                                       layer's stored records (spatial/positions.py); how many cameras can place
                                       people there (`cameras`: with none, nobody can be, and the page says so); and
                                       each camera of the place with how it's posed and what it lacks
                                       (`camera_notes`: [{camera, by: "spot" | "map", missing}], `missing` any of
                                       "height", "facing", "lens", "tilt", "check" for a spot, spatial/spots.py),
                                       and whether the place has a map
                                       (`map`); the read stops if the client hangs up

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
the page's handedness: `GET /places` gives that page's plan as `plan`; the 3D scene keeps the plan frame). A point that
can't be put in that frame (`in_frame` false: a camera that revision doesn't register) would land off the drawn plan, so
it's left out; a room-only span is kept either way.
"""

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from liberational import logs
from liberational.dashboard import routes
from liberational.memory import places
from liberational.spatial import plan_scene

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
    """The revision whose map the 3D view shows: `revision` if it's one of the place's, a producer of today wrote it
    (`places.SHOWN_PRODUCERS`: a map an earlier version made stays on disk, unread), and its map file is still kept
    (maps are pruned to the newest few), else (with none asked for) the newest such revision. A revision's map is
    always a bare `<uuid>.map` in the place's folder: memory/places.py refuses any other name when it reads one."""
    chain = [r for r in places.revisions(state, place) if places.shown(r)]
    found = [r for r in chain if r.map and (places.root(state) / place / r.map).is_file()]
    if revision is not None:
        return next((r for r in found if r.id == revision), None)
    return found[-1] if found else None


def map_file(state: Path, place: str, revision: str, camera: str | None = None) -> Path | None:
    """GET /places/map: the file of `revision`'s map, or None (no such place or revision, or its map is gone). With
    `camera`, that camera's own cloud (spatial/camera_clouds.py), when `revision` is still the one it has."""
    if places.get(state, place) is None:
        return None
    if camera is not None:
        from liberational.spatial import camera_clouds  # numpy: only the 3D view needs it

        store = camera_clouds.Store(camera_clouds.root(state))
        cloud = store.current(camera)
        return store.file(cloud) if cloud is not None and cloud.id == revision else None
    rev = _map_revision(state, place, revision)
    return places.root(state) / place / rev.map if rev and rev.map else None


def splat_file(state: Path, place: str, revision: str) -> Path | None:
    """GET /places/splat: `revision`'s splat (SOG), or None (no such place or revision, or it has no splat)."""
    if places.get(state, place) is None:
        return None
    rev = next((r for r in places.revisions(state, place) if r.id == revision and places.shown(r)), None)
    return places.splat_path(state, rev, SPLAT_FORMAT) if rev and places.has_splat(state, rev) else None


SPLAT_FORMAT = "sog"  # what GET /places/splat serves, and what its URL and ETag name


def _learning() -> dict[str, Any]:
    """Whether pictures are collected from the cameras, for the 3D view to say in plain words: `on` (the Live tab's
    switch) and `cameras` (how many it samples now; none while off)."""
    from liberational.spatial import surroundings

    switch = surroundings.read()
    return {"on": switch.on, "cameras": len(switch.cameras) if switch.on else 0}


#: Why a camera has no cloud that the service doesn't note itself: learning is off, or the camera isn't chosen.
VIEW_OFF, VIEW_NOT_CHOSEN, VIEW_WAITING = "off", "not_chosen", "waiting"


def camera_views(state: Path, place: str) -> list[dict[str, Any]]:
    """The cameras whose own clouds the 3D view of `place` offers, in name order: the ones learned from now and the
    ones that have a cloud, among the cameras with a spot on this place's plan and the ones with no spot on any
    plan (a camera is an angle, and nothing else says which place it looks at). Each says whether it has a cloud
    (`cloud`, with `at`, `points`, and `size`: "approximate", the model's own metres) and what keeps it from a cloud or a newer
    one (`why`: "off" while learning is off, "not_chosen", one of camera_clouds.REASONS from its last try, or
    "waiting" for a clean picture; None for a camera whose cloud is kept up to date)."""
    from liberational.spatial import camera_clouds, surroundings

    store = camera_clouds.Store(camera_clouds.root(state))
    switch = surroundings.read()
    spots = places.spot_places(state)
    notes = store.notes()
    names = (set(switch.cameras) if switch.on else set()) | set(store.cameras())
    out = []
    for camera in sorted(names):
        if spots.get(camera, place) != place:
            continue
        cloud = store.current(camera)
        note = notes.get(camera)
        # Learning off, or the camera not chosen, comes first: nothing is made or refreshed then, whatever the last
        # try said, and the page says so even for a camera that has a cloud.
        if not switch.on:
            why = VIEW_OFF
        elif camera not in switch.cameras:
            why = VIEW_NOT_CHOSEN
        elif note is not None:
            why = str(note.get("reason"))
        else:
            why = None if cloud is not None else VIEW_WAITING
        out.append(
            {
                "camera": camera,
                "cloud": cloud is not None,
                "at": None if cloud is None else cloud.at,
                "points": None if cloud is None else cloud.points,
                "size": None if cloud is None else cloud.size,
                "why": why,
            }
        )
    return out


def _camera_scene(state: Path, place: str, camera: str, learning: dict[str, Any], views: list[dict[str, Any]]) -> Reply:
    """The scene of one camera's own cloud, in the cloud's level frame (spatial/camera_clouds.py): its points, the
    floor at z = 0, and the camera where it stands in it. No plan is laid on it and no splat goes with it: the cloud
    is that camera's view alone, not the place's map."""
    from liberational.spatial import camera_clouds

    store = camera_clouds.Store(camera_clouds.root(state))
    cloud = store.current(camera)
    base = {"place": place, "learning": learning, "views": views}
    if cloud is None:
        return 200, {**base, "revision": None, "view": {"camera": camera}}
    try:
        size = store.file(cloud).stat().st_size
    except FileNotFoundError:  # replaced since it was read
        return 200, {**base, "revision": None, "view": {"camera": camera}}
    pose = {
        "pose": list(cloud.pose),
        "focal": cloud.focal,
        "width": cloud.width,
        "height": cloud.height,
        "at": cloud.at,
    }
    return 200, {
        **base,
        "revision": cloud.id,
        "at": cloud.at,
        "map": "/places/map?" + urlencode({"place": place, "revision": cloud.id, "camera": camera}),
        "map_format": "ply",
        "map_bytes": size,
        "splat": None,
        "floor": [0.0, 0.0, 1.0, 0.0],
        "plan_transform": None,
        "cameras": {camera: pose},
        "plan": None,
        "view": {
            "camera": camera,
            "size": cloud.size,
            "floor_found": cloud.floor_found,
            "points": cloud.points,
            "at": cloud.at,
        },
    }


def _scene_of(state: Path, held: places.Place) -> Any:
    """`held`'s scene of all its cameras as it stands (spatial/cloud_scene.py `scene`), with the owner's plan's rooms
    for the view to leave out what lies outside them."""
    from liberational.spatial import camera_clouds, cloud_scene

    spots = cloud_scene.spots_of(state)
    mirrored = next((s.mirrored for s in spots.values() if s.place == held.id), False)
    rooms = cloud_scene.rooms_of(held.plan if held.plan_source == "owner" else None, mirrored)
    return cloud_scene.scene(camera_clouds.Store(camera_clouds.root(state)), held.id, spots, rooms)


def _all_cameras_scene(state: Path, held: places.Place, learning: dict[str, Any], views: list[dict[str, Any]]) -> Reply:
    """The scene of every camera's cloud placed in the place (spatial/cloud_scene.py), in the plan's metres: the
    clouds' points as one file, each point saying whose it is, the plan under them, each placed camera where it
    stands, and `combined`: how every camera is placed, or why it isn't, and what each pair of cameras' pictures said
    of each other. `revision` is None while no camera is placed."""
    from liberational.spatial import camera_clouds, cloud_scene

    store = camera_clouds.Store(camera_clouds.root(state))
    found = _scene_of(state, held)
    base = {"place": held.id, "learning": learning, "views": views, "combined": cloud_scene.as_dict(found)}
    placed = [p for p in found.cameras if p.placed]
    if not placed:
        return 200, {**base, "revision": None}
    clouds = {p.camera: store.current(p.camera) for p in placed}
    cameras = {
        p.camera: {"pose": list(p.pose or ()), "focal": c.focal, "width": c.width, "height": c.height, "at": c.at}
        for p in placed
        if (c := clouds[p.camera]) is not None
    }
    turned = -1.0 if not found.mirrored else 1.0  # the scene is right-handed; a page's y runs the other way
    return 200, {
        **base,
        "revision": found.id,
        "at": max(c.at for c in clouds.values() if c is not None),
        "map": "/places/map?" + urlencode({"place": held.id, "revision": found.id, "scene": "1"}),
        "map_format": "ply",
        "map_bytes": cloud_scene.drawn(store, found).stat().st_size,
        "splat": None,
        "floor": [0.0, 0.0, 1.0, 0.0],
        "plan_transform": [1.0, 0.0, 0.0, 0.0, 0.0, turned, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "cameras": cameras,
        "plan": held.plan if held.plan_source == "owner" else None,
        "plan_source": "owner",
        "plan_aligned": True,
    }


def scene_file(state: Path, place: str, revision: str) -> Path | None:
    """GET /places/map with `scene`: the points of the place's scene of all its cameras, when `revision` is still the
    scene there is."""
    from liberational.spatial import camera_clouds, cloud_scene

    held = places.get(state, place)
    if held is None:
        return None
    store = camera_clouds.Store(camera_clouds.root(state))
    found = _scene_of(state, held)
    return cloud_scene.drawn(store, found) if found.id == revision and any(p.placed for p in found.cameras) else None


def accept_spot(body: Mapping[str, Any], control: bool, state: Path, now: float | None = None) -> Reply:
    """POST /places/scene/spot: the owner takes the spot the scene suggests for a camera that moved. The camera's
    spot on their plan is moved and turned to it and the plan saved as a new version, the way the plan editor saves
    one (`places.correct_plan`); nothing else of the plan changes. `control` is the handler's _can_teach."""
    from liberational.spatial import camera_clouds, cloud_scene

    if not control:
        return 403, {"ok": False, "error": "plans can only be corrected from this machine (Remote teaching is off)"}
    place, camera = body.get("place"), body.get("camera")
    if not isinstance(place, str) or not isinstance(camera, str):
        return 400, {"ok": False, "error": "send the place's id and the camera"}
    held = places.get(state, place)
    if held is None:
        return 404, {"ok": False, "error": "no such place"}
    if held.plan_source != "owner" or not held.plan:
        return 400, {"ok": False, "error": "the place has no plan of yours to set a spot on"}
    if held.view is not None:
        return 400, {"ok": False, "error": "this plan was drawn on its own page: move the spot in the plan editor"}
    store = camera_clouds.Store(camera_clouds.root(state))
    found = _scene_of(state, held)
    offered = next((p.suggest for p in found.cameras if p.camera == camera), None)
    if offered is None:
        return 400, {"ok": False, "error": "no spot is suggested for that camera now"}
    try:
        plan = plan_scene.with_spot(held.plan, camera, (offered[0], offered[1]), offered[2])
    except KeyError, plan_scene.SceneError:
        return 400, {"ok": False, "error": "that camera has no spot on the plan to move"}
    history = places.plan_history(state, place)
    places.correct_plan(state, place, plan, now=now, base=held.plan_revision)
    spot = cloud_scene.spots_of(state).get(camera)
    if spot is not None:
        cloud_scene.accept(store, camera, spot.basis)  # its picture now is the one it's placed with
    return 200, {
        "ok": True,
        "spot": {"at": [offered[0], offered[1]], "yaw": offered[2]},
        # the version this one replaced, for Undo (POST /places/scene/spot/undo)
        "previous": history[-1].id if history else None,
    }


def undo_spot(body: Mapping[str, Any], control: bool, state: Path, now: float | None = None) -> Reply:
    """POST /places/scene/spot/undo: take back a spot accepted from the scene. The plan version it replaced
    (`record`) holds again, written on top as a new version (`places.restore_plan`), and the camera is once more one
    that moved from the picture it was placed with: it isn't left at its old spot."""
    from liberational.spatial import camera_clouds, cloud_scene

    if not control:
        return 403, {"ok": False, "error": "plans can only be corrected from this machine (Remote teaching is off)"}
    place, camera, record = body.get("place"), body.get("camera"), body.get("record")
    if not isinstance(place, str) or not isinstance(camera, str) or not isinstance(record, str):
        return 400, {"ok": False, "error": "send the place's id, the camera, and the plan version to put back"}
    if places.get(state, place) is None:
        return 404, {"ok": False, "error": "no such place"}
    try:
        places.restore_plan(state, place, record, now=now)
    except (KeyError, ValueError) as e:
        return 400, {"ok": False, "error": str(e)}
    cloud_scene.take_back(camera_clouds.Store(camera_clouds.root(state)), camera)
    return 200, {"ok": True}


def scene(state: Path, place: str, camera: str | None = None, combined: bool = False) -> Reply:
    """GET /places/scene: what the 3D view draws for a place, all in its map's frame (metres). While the place has an
    owner's plan, that is the plan, whatever the map: `plan_source` is "owner", and no machine plan is given in its
    place. It's laid on the map only when it's drawn in this revision's frame (`plan_transform`, with `plan_aligned`
    true); otherwise the transform is None, and the view draws the map alone. A place with no owner's plan gives the
    revision's own machine plan, in its frame. No map is made at present (the 3D map is being rebuilt with a new
    method), so `revision` is None until a producer of today writes one (`places.SHOWN_PRODUCERS`).

    With `combined`, the scene is every camera's cloud placed in the place (`_all_cameras_scene`).

    What there is meanwhile is each camera's own cloud (`camera_views`, always given as `views`). With `camera`, the
    scene is that camera's cloud (`_camera_scene`); with none, and no map of the place, it's the first camera's that
    has one, so the view opens on something. `view` names the camera whose cloud the scene is."""
    held = places.get(state, place)
    if held is None:
        return 404, {"error": "no such place"}
    learning = _learning()
    views = camera_views(state, place)
    if combined:
        return _all_cameras_scene(state, held, learning, views)
    if camera is not None:
        return _camera_scene(state, place, camera, learning, views)
    rev = _map_revision(state, place)
    if rev is None or rev.map is None:
        first = next((v["camera"] for v in views if v["cloud"]), None)
        if first is not None:
            return _camera_scene(state, place, first, learning, views)
        return 200, {"place": place, "revision": None, "learning": learning, "views": views}
    by_id = {r.id: r for r in places.revisions(state, place)}
    drawn = by_id.get(held.plan_revision or "")
    owned = held.plan_source == "owner"
    aligned = owned and drawn is not None and places.same_plan_frame(drawn, rev)
    try:
        size = (places.root(state) / place / rev.map).stat().st_size
    except FileNotFoundError:  # pruned since it was found
        return 200, {"place": place, "revision": None, "learning": learning, "views": views}
    # The revision's own splat, else the newest one made in the same map frame (a later revision that kept the
    # frame); a splat of another frame (or an unknown one) would sit out of place, so it's only named as stale.
    chain = [r for r in places.revisions(state, place) if places.shown(r)]
    with_splat = [r for r in chain if places.has_splat(state, r)]
    fits = [r for r in with_splat if r.id == rev.id or (rev.frame is not None and r.frame == rev.frame)]
    shown = next((r for r in reversed(fits) if r.id == rev.id), fits[-1] if fits else None)
    try:
        splat_size = places.splat_path(state, shown, SPLAT_FORMAT).stat().st_size if shown else None
    except FileNotFoundError:  # pruned since it was found
        shown, splat_size = None, None
    stale = with_splat[-1] if shown is None and with_splat else None
    # Which state the view is in, so the page doesn't guess: this map's own splat, an earlier map's that still lines
    # up (same frame), one that doesn't (named, not offered), or none. A revision that kept its map (one that only
    # set the floor or a camera) has the splat of the revision that wrote that map: it's the same map's.
    own = shown is not None and (shown.id == rev.id or shown.map == rev.map)
    if shown is not None and own:
        splat_state = "current"
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
        "learning": learning,
        "views": views,
        "floor": list(rev.floor) if rev.floor else None,
        "plan_transform": list(transform) if transform else None,
        "cameras": {cam: pose.as_dict() for cam, pose in rev.cameras.items()},
        "plan": held.plan if owned else places.machine_plan(state, rev),
        "plan_source": "owner" if owned else "machine",
        "plan_aligned": aligned if owned else transform is not None,
    }


# A plan is a Pascal scene (spatial/plan_scene.py), which holds every default of every node: about 1.5 KB a door, so a
# large home's is a few hundred KB. Anything past this isn't a plan.
MAX_SCENE_BYTES = 2 * 1024 * 1024


def correct(body: Mapping[str, Any], control: bool, state: Path, now: float | None = None) -> Reply:
    """POST /places/plan: save the owner's plan for a place (a Pascal scene, as the plan editor saves it), or withdraw
    it with an empty plan. `control` is the handler's _can_teach. The plan must be a scene that reads and has a room,
    not be huge, and `base` (when sent) be one of the place's revisions."""
    if not control:
        return 403, {"ok": False, "error": "plans can only be corrected from this machine (Remote teaching is off)"}
    place, plan, base = body.get("place"), body.get("plan"), body.get("base")
    if not isinstance(place, str) or not isinstance(plan, str) or not (base is None or isinstance(base, str)):
        return 400, {"ok": False, "error": "send the place's id, its plan as text, and the revision it was drawn in"}
    if len(plan.encode()) > MAX_SCENE_BYTES:
        return 400, {"ok": False, "error": "that plan is too long to be a floorplan"}
    if places.get(state, place) is None:
        return 404, {"ok": False, "error": "no such place"}
    if plan.strip():
        try:
            scene = plan_scene.load(plan)
        except plan_scene.SceneError as e:
            return 400, {"ok": False, "error": str(e)}
        if not plan_scene.rooms(scene):
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
    """GET /places/plan/history?place=: every saved version of the place's owner plan (memory/places.py
    `plan_history`), oldest first, each {id, at, rooms, source, withdrawn, current, summary}. `current` marks the one
    that holds (none while the machine's plan does), and `summary` is what changed from the version before it, in a
    few words (`plan_change`). A version saved in the line DSL from before scenes can no longer be read, so it is
    not listed (the conversion gave each one a scene copy, which is); how many were left out is logged."""
    held = places.get(state, place)
    if held is None:
        return 404, {"error": "no such place"}
    everything = places.plan_history(state, place)
    history = [r for r in everything if not _plan_text(r.value).strip() or plan_scene.is_scene(_plan_text(r.value))]
    if len(history) < len(everything):
        logs.get("dashboard").info(
            "%s: %d plan version(s) in the old format are not listed", held.id, len(everything) - len(history)
        )
    texts = [_plan_text(r.value) for r in history]
    owned = held.plan_source == "owner"
    versions = [
        {
            "id": r.id,
            "at": r.start_ns / 1e9,
            "rooms": len(plan_scene.rooms_of(t)),
            "source": r.value.get("source", "owner"),
            "withdrawn": not t.strip(),
            # the newest record of all is the one that holds, so an old-format one on top means none listed does
            "current": owned and r is everything[-1],
            "summary": plan_change(texts[i - 1] if i else None, t),
        }
        for i, (r, t) in enumerate(zip(history, texts, strict=True))
    ]
    return 200, {"place": held.id, "versions": versions}


MOVED_SPOT_M = 0.01  # a camera spot this far from where it was has moved
TURNED_SPOT_DEG = 0.5
MAX_CHANGES = 3  # how many changes a version's summary names before it counts the rest


def plan_change(before: str | None, after: str) -> str:
    """What changed between two versions of a plan, in a few words, for the versions timeline: rooms added, removed,
    renamed, or reshaped, doors and windows added or removed, and camera spots added, removed, moved, or turned.
    `before` is None for a place's first version. A version that can't be read says so, and the one after it is
    summed up as if it were the first."""
    from liberational.spatial import plan_import

    if not after.strip():
        return "Plan withdrawn"

    def read(text: str | None) -> plan_import.Parsed | None:
        if text is None or not text.strip():
            return None
        try:
            return plan_import.parse(text)
        except ValueError:
            return None

    now = read(after)
    if now is None:
        return "A plan that can't be read"
    was = read(before)
    if was is None:
        cameras = f", {_count(len(now.cameras), 'camera spot')}" if now.cameras else ""
        return f"{_count(len(now.rooms), 'room')}{cameras}"
    changes: list[str] = []
    old_rooms, new_rooms = {r.id: r for r in was.rooms}, {r.id: r for r in now.rooms}
    changes += [f"added {_room(r)}" for i, r in new_rooms.items() if i not in old_rooms]
    changes += [f"removed {_room(r)}" for i, r in old_rooms.items() if i not in new_rooms]
    for i, r in new_rooms.items():
        if i not in old_rooms:
            continue
        if r.name != old_rooms[i].name:
            changes.append(f"renamed {_room(old_rooms[i])} to {_room(r)}")
        if not _same_outline(r.polygon, old_rooms[i].polygon):
            changes.append(f"reshaped {_room(r)}")
    for kind, label in (("door", "door"), ("cased", "doorway"), ("window", "window")):
        n = sum(o.kind == kind for o in now.openings) - sum(o.kind == kind for o in was.openings)
        if n:
            changes.append(f"{'added' if n > 0 else 'removed'} {_count(abs(n), label)}")
    old_spots, new_spots = {c.id: c for c in was.cameras}, {c.id: c for c in now.cameras}
    changes += [f"added camera {i}" for i in new_spots if i not in old_spots]
    changes += [f"removed camera {i}" for i in old_spots if i not in new_spots]
    for i, c in new_spots.items():
        if i not in old_spots:
            continue
        o = old_spots[i]
        if math.dist(c.at, o.at) > MOVED_SPOT_M:
            changes.append(f"moved camera {i}")
        elif (c.yaw is None) != (o.yaw is None) or (
            c.yaw is not None and o.yaw is not None and abs((c.yaw - o.yaw + 180) % 360 - 180) > TURNED_SPOT_DEG
        ):
            changes.append(f"turned camera {i}")
        elif (c.height, c.fov) != (o.height, o.fov):
            changes.append(f"changed camera {i}")
    if not changes:
        return "No change to rooms, openings, or cameras"
    shown = changes[:MAX_CHANGES]
    more = len(changes) - len(shown)
    text = ", ".join(shown) + (f", and {more} more" if more else "")
    return text[0].upper() + text[1:]


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _room(room: Any) -> str:
    return room.name or room.id


def _same_outline(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> bool:
    """Whether two room outlines are one shape: the same corners within a centimetre, whatever corner each starts
    at and whichever way round it goes."""
    if len(a) != len(b):
        return False

    def near(p: tuple[float, float], q: tuple[float, float]) -> bool:
        return math.dist(p, q) <= SAME_PLAN_M

    n = len(a)
    return any(
        all(near(a[i], ring[(i + k) % n]) for i in range(n)) for ring in (list(b), list(b)[::-1]) for k in range(n)
    )


def plan_as_scene(state: Path, place: str, record: str | None = None) -> Reply:
    """GET /places/plan/scene?place=[&record=]: a place's plan as the plan editor opens it, a Pascal scene
    (spatial/plan_scene.py): the plan that holds, as the owner draws it (on its own page when it has one), or with
    `record` one of the place's earlier owner plans (`plan_versions`), to look at before putting it back. `base` is
    the revision the plan is drawn in, which a save sends back. A place with no plan is a 404 (the editor starts an
    empty one), and a plan that isn't a scene that reads a 422."""
    held = places.get(state, place)
    if held is None:
        return 404, {"error": "no such place"}
    if record is None:
        text = (held.view.plan if held.view is not None else held.plan) or ""
    else:
        found = next((r for r in places.plan_history(state, place) if r.id == record), None)
        if found is None:
            return 404, {"error": "no such version of this place's plan"}
        view = places.page_view(found.value.get("view"))
        text = view.plan if view is not None else _plan_text(found.value)
    if not text.strip():
        return 404, {"error": "this place has no plan" if record is None else "that version withdrew the plan"}
    if not plan_scene.is_scene(text):
        return 422, {"error": "it is in an old format that is no longer read"}
    try:
        scene = plan_scene.load(text)
    except plan_scene.SceneError as e:
        return 422, {"error": str(e)}
    return 200, {"place": held.id, "scene": scene, "base": held.plan_revision}


def restore(body: Mapping[str, Any], control: bool, state: Path, now: float | None = None) -> Reply:
    """POST /places/plan/restore {place, record}: make an earlier version of the owner's plan hold again, written as
    the newest record (memory/places.py `restore_plan`), so the one it replaces stays in the history and can be put
    back the same way. A version in the old format can't be restored (it isn't listed either). A teaching write, like
    POST /places/plan."""
    if not control:
        return 403, {"ok": False, "error": "plans can only be changed from this machine (Remote teaching is off)"}
    place, record = body.get("place"), body.get("record")
    if not isinstance(place, str) or not isinstance(record, str):
        return 400, {"ok": False, "error": "send the place's id and the plan record to put back"}
    if places.get(state, place) is None:
        return 404, {"ok": False, "error": "no such place"}
    was = next((r for r in places.plan_history(state, place) if r.id == record), None)
    if was is None:
        return 404, {"ok": False, "error": "no such plan in this place's history"}
    if _plan_text(was.value).strip() and not plan_scene.is_scene(_plan_text(was.value)):
        return 422, {"ok": False, "error": "that version is in an old format that is no longer read"}
    places.restore_plan(state, place, record, now=now)
    held = places.get(state, place)
    shown = (held.view.plan if held.view is not None else held.plan) if held else None
    return 200, {"ok": True, "plan": shown, "plan_source": held.plan_source if held else None}


SAME_PLAN_M = 0.01  # two outlines whose corners are this close are one shape


def _plan_text(value: Mapping[str, Any]) -> str:
    plan = value.get("plan")
    return plan if isinstance(plan, str) else ""


def _from_page(text: str, view: places.PageView) -> str:
    """An owner plan edited on its page, moved into the plan frame (spatial/plan_import.py). Raises ValueError for a
    plan it can't read."""
    from liberational.spatial import plan_import

    return plan_import.moved(text, plan_import.Move(view.turn, view.shift, view.mirror))


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
        "map": held.revision.map is not None,  # with no map and no camera spot, the page says "no map yet"
    }


# ---- routes (dashboard/routes.py): what the server answers for this area ----


@routes.get("/places")
def _get_places(h: H) -> None:
    code, reply = listing(h.state)
    return h._json(reply, code)


@routes.get("/places/scene")
def _get_places_scene(h: H) -> None:
    q = h._query()
    code, reply = scene(h.state, q.get("place", ""), q.get("camera") or None, combined=q.get("all") == "1")
    return h._json(reply, code)


@routes.get("/places/map", "/places/splat")
def _get_places_map(h: H) -> None:
    p = h.path.split("?")[0]
    q = h._query()
    if p == "/places/map" and q.get("scene") == "1":
        found = scene_file(h.state, q.get("place", ""), q.get("revision", ""))
    elif p == "/places/map":
        found = map_file(h.state, q.get("place", ""), q.get("revision", ""), q.get("camera") or None)
    else:
        found = splat_file(h.state, q.get("place", ""), q.get("revision", ""))
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


@routes.get("/places/plan/scene")
def _get_places_plan_scene(h: H) -> None:
    query = h._query()
    code, reply = plan_as_scene(h.state, query.get("place", ""), query.get("record") or None)
    return h._json(reply, code)


@routes.post("/places/plan")
def _post_places_plan(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = correct(body, h._can_teach(), h.state)
    return h._json(o, code)


@routes.post("/places/scene/spot")
def _post_places_scene_spot(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = accept_spot(body, h._can_teach(), h.state)
    return h._json(o, code)


@routes.post("/places/scene/spot/undo")
def _post_places_scene_spot_undo(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = undo_spot(body, h._can_teach(), h.state)
    return h._json(o, code)


@routes.post("/places/plan/restore")
def _post_places_plan_restore(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = restore(body, h._can_teach(), h.state)
    return h._json(o, code)
