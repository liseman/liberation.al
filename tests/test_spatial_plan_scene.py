"""A plan as a Pascal scene (spatial/plan_scene.py), and the readers of a stored plan."""

import json
import math

import numpy as np
import pytest

import plans
from liberational import camera_rename
from liberational.dashboard import places as view
from liberational.memory import places
from liberational.spatial import plan_import as pi
from liberational.spatial import plan_scene

T = 1_762_500_000.0

# A made-up two-room flat: a hall and a den side by side, a front door, a doorway between them, and a window.
FLAT = plans.scene(
    """plan "Flat" walls 0.3/0.1
room hall "Hall" hall rect 0,0 3x4
room den "The \\"den\\"" living rect 3,0 5x4
door exterior>hall at 0,2 w0.9
cased hall>den at 3,1 w1.2
window exterior>den at 5.5,0 w1.6
x-camera porch at 0.4,0.5 height 2.2 yaw 135 fov 100
x-camera den-cam at 7.5,3.5
"""
)
HALL = ("hall", "Hall", ((0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0)))
DEN = ("den", 'The "den"', ((3.0, 0.0), (8.0, 0.0), (8.0, 4.0), (3.0, 4.0)))


def nodes(text: str, kind: str) -> list[dict]:
    return [n for n in json.loads(text)["nodes"].values() if n["type"] == kind]


def test_rooms_become_named_zones_with_a_wall_for_each_stretch_of_room_edge():
    assert [(z["id"], z["name"], z["metadata"]["kind"]) for z in nodes(FLAT, "zone")] == [
        ("zone_hall", "Hall", "hall"),
        ("zone_den", 'The "den"', "living"),
    ]
    # Seven stretches: the hall's three outside walls, the den's three, and the one they share.
    walls = {(tuple(w["start"]), tuple(w["end"])): w["thickness"] for w in nodes(FLAT, "wall")}
    assert len(walls) == 7
    assert walls[((3.0, 0.0), (3.0, 4.0))] == pytest.approx(0.1)  # the shared wall is a partition
    assert walls[((0.0, 0.0), (0.0, 4.0))] == pytest.approx(0.3)
    assert sum(1 for t in walls.values() if t == pytest.approx(0.3)) == 6


def test_walls_are_cut_where_another_rooms_corner_meets_them():
    # a long room beside two short ones: its side is two walls, one for each neighbour
    long_room = ((0.0, 0.0), (2.0, 0.0), (2.0, 6.0), (0.0, 6.0))
    upper = ((2.0, 0.0), (5.0, 0.0), (5.0, 3.0), (2.0, 3.0))
    lower = ((2.0, 3.0), (5.0, 3.0), (5.0, 6.0), (2.0, 6.0))
    cut = plan_scene.cut_walls([long_room, upper, lower], 0.3, 0.1)
    shared = [(a, b) for a, b, thickness in cut if thickness == 0.1]
    assert shared == [((2.0, 0.0), (2.0, 3.0)), ((2.0, 3.0), (2.0, 6.0)), ((2.0, 3.0), (5.0, 3.0))]
    assert len(cut) == 3 + 7  # the three shared stretches, and seven outside ones


def test_an_opening_hangs_on_the_wall_its_point_is_on_at_its_distance_from_the_walls_start():
    door, cased = nodes(FLAT, "door")
    (window,) = nodes(FLAT, "window")
    by_id = json.loads(FLAT)["nodes"]
    for opening, start, end, along, width in (
        (door, [0.0, 0.0], [0.0, 4.0], 2.0, 0.9),
        (cased, [3.0, 0.0], [3.0, 4.0], 1.0, 1.2),
        (window, [3.0, 0.0], [8.0, 0.0], 2.5, 1.6),
    ):
        wall = by_id[opening["wallId"]]
        assert (wall["start"], wall["end"]) == (start, end)
        assert opening["id"] in wall["children"]
        assert (opening["position"][0], opening["width"]) == (pytest.approx(along), width)
    assert (door["openingKind"], cased["openingKind"]) == ("door", "opening")


def test_a_scene_reads_as_its_rooms_openings_wall_thicknesses_and_camera_spots():
    plan = pi.parse(FLAT)
    assert [(r.id, r.name, r.kind, r.polygon) for r in plan.rooms] == [
        (*HALL[:2], "hall", HALL[2]),
        (*DEN[:2], "living", DEN[2]),
    ]
    assert plan.title == "Flat"
    assert plan.walls == pytest.approx((0.3, 0.1))
    # an opening is between the rooms either side of its wall, the outside first; two rooms come in either order
    assert [(o.kind, frozenset(o.between.split(">")), o.width) for o in plan.openings] == [
        ("door", frozenset({"exterior", "hall"}), 0.9),
        ("cased", frozenset({"hall", "den"}), 1.2),
        ("window", frozenset({"exterior", "den"}), 1.6),
    ]
    assert [o.between.split(">")[0] for o in plan.openings if o.kind != "cased"] == ["exterior", "exterior"]
    assert [o.at for o in plan.openings] == [pytest.approx(p) for p in ((0, 2), (3, 1), (5.5, 0))]
    assert plan.cameras == (
        pi.Camera("porch", (0.4, 0.5), 2.2, 135.0, 100.0),
        pi.Camera("den-cam", (7.5, 3.5), None, None, None),
    )
    cut = pi.walls(plan)
    assert len(cut.a) == 7 and sorted(set(cut.half.tolist())) == pytest.approx([0.05, 0.15])


def test_the_room_and_spot_readers_take_a_stored_plan():
    assert plan_scene.rooms_of(FLAT) == [HALL, DEN]
    assert pi.spots(FLAT) == pi.parse(FLAT).cameras


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("{not json", id="not-json"),
        pytest.param('{"nodes": {}, "rootNodeIds": []}', id="no-level"),
        pytest.param('{"rootNodeIds": []}', id="no-nodes"),
        pytest.param('{"nodes": {"a": {"id": "a"}}, "rootNodeIds": ["a"]}', id="untyped-node"),
        pytest.param(
            'plan "Home" walls 0.2/0.1\nroom den "Den" living rect 0,0 4x6\n', id="the-line-dsl-from-before-scenes"
        ),
        pytest.param("", id="nothing"),
    ],
)
def test_a_plan_that_isnt_a_scene_has_no_rooms_or_spots_and_is_refused_by_the_full_reader(text):
    assert plan_scene.rooms_of(text) == []
    assert pi.spots(text) == ()
    assert plan_scene.spot_names(text) == []
    assert camera_rename.rename_plan(text, {"porch": "x"}) == text
    with pytest.raises(pi.PlanError):
        pi.parse(text)


def test_a_scene_with_walls_but_no_rooms_is_a_plan_with_no_rooms():
    bare = plan_scene.dump(plan_scene.build(walls=[((0.0, 0.0), (4.0, 0.0), 0.2)]))
    with pytest.raises(pi.PlanError, match="no rooms"):
        pi.parse(bare)


def test_an_opening_on_no_wall_is_refused_with_where_it_is():
    wall = [((0.0, 0.0), (4.0, 0.0), 0.2)]
    assert plan_scene.on_a_wall(wall, (1.0, 0.01)) and not plan_scene.on_a_wall(wall, (1.0, 1.0))
    with pytest.raises(plan_scene.SceneError, match="door at 1,1 is on no wall"):
        plan_scene.build(walls=wall, openings=[("door", (1.0, 1.0), 0.9)])


def test_what_the_editor_adds_to_a_scene_doesnt_change_what_is_read():
    # The editor saves a floor slab and a ceiling per room beside the zones, fills every node with defaults, and
    # may leave a zone unlisted by its level for a moment: none of it is a room, and the zone still is.
    scene = json.loads(FLAT)
    level = scene["nodes"]["level_ground"]
    square = [[0, 0], [8, 0], [8, 4], [0, 4]]
    scene["nodes"]["slab_1"] = {"id": "slab_1", "type": "slab", "parentId": level["id"], "polygon": square}
    scene["nodes"]["ceiling_1"] = {"id": "ceiling_1", "type": "ceiling", "parentId": level["id"], "polygon": square}
    level["children"] += ["slab_1", "ceiling_1"]
    level["children"].remove("zone_den")
    scene["nodes"]["zone_den"].update(autoFromWalls=True, boundaryWallIds=["wall_0"], color="#84cc16")
    scene["nodes"]["zone_stub"] = {"id": "zone_stub", "type": "zone", "parentId": level["id"], "polygon": [[0, 0]]}
    assert plan_scene.rooms_of(json.dumps(scene)) == [HALL, DEN]


@pytest.mark.parametrize(
    ("spot", "read"),
    [
        pytest.param({"at": [1, 2]}, ("cam", (1.0, 2.0), None, None, None), id="position-only"),
        pytest.param({"at": [1, 2], "height": 2, "yaw": 0, "fov": 90}, ("cam", (1.0, 2.0), 2.0, 0.0, 90.0), id="all"),
        pytest.param({"at": [1, 2], "height": "tall", "yaw": True}, ("cam", (1.0, 2.0), None, None, None), id="junk"),
        pytest.param({"at": [1]}, None, id="half-a-position"),
        pytest.param({"height": 2.0}, None, id="no-position"),
        pytest.param("on the shelf", None, id="not-a-record"),
    ],
)
def test_a_spot_needs_a_position_and_takes_only_numbers(spot, read):
    scene = plan_scene.build(walls=[((0.0, 0.0), (4.0, 0.0), 0.2)], records={"cameras": {"cam": spot}})
    assert plan_scene.spots(scene) == ([read] if read else [])


def test_renaming_a_camera_moves_its_spot_and_leaves_a_scene_without_it_untouched():
    renamed = camera_rename.rename_plan(FLAT, {"porch": "front-door"})
    assert [c.id for c in pi.spots(renamed)] == ["front-door", "den-cam"]
    assert pi.spots(renamed)[0] == pi.Camera("front-door", (0.4, 0.5), 2.2, 135.0, 100.0)
    assert plan_scene.rooms_of(renamed) == [HALL, DEN]
    assert camera_rename.rename_plan(FLAT, {"garage": "shed"}) is FLAT


def test_a_scene_moved_into_another_frame_moves_every_outline_and_turns_each_camera():
    # A quarter turn and a shift: (x, y) goes to (10 - y, x), and a camera looking up the page (0) looks east (90).
    scene = json.loads(FLAT)
    side = plan_scene.records(scene)
    side["objects"] = {"lamp": {"label": "Lamp", "at": [4.0, 1.0]}}
    side["unexplored"] = {"back": {"name": "Back room", "rect": [6.0, 3.0, 2.0, 1.0]}}
    text = pi.moved(json.dumps(scene), pi.Move(math.pi / 2, (10.0, 0.0), False))
    hall = plan_scene.rooms_of(text)[0]
    assert hall[:2] == ("hall", "Hall")
    np.testing.assert_allclose(hall[2], [(10, 0), (10, 3), (6, 3), (6, 0)], atol=1e-9)
    porch, den = pi.spots(text)
    assert (porch.at, porch.yaw, porch.height) == (pytest.approx((9.5, 0.4)), pytest.approx(225.0), 2.2)
    assert (den.at, den.yaw) == (pytest.approx((6.5, 7.5)), None)
    front = next(w for w in nodes(text, "wall") if w["children"] == ["door_0"])
    assert (front["start"], front["end"]) == (pytest.approx([10, 0]), pytest.approx([6, 0]))
    assert nodes(text, "door")[0]["position"][0] == pytest.approx(2.0)  # still 2 m along its own wall
    moved = plan_scene.records(json.loads(text))
    assert moved["objects"]["lamp"]["at"] == pytest.approx([9, 4])
    assert "rect" not in moved["unexplored"]["back"]  # a turned rectangle is a polygon
    assert moved["unexplored"]["back"]["poly"] == [pytest.approx(p) for p in ([7, 6], [7, 8], [6, 8], [6, 6])]
    site = nodes(text, "site")[0]["polygon"]["points"]
    assert site == [pytest.approx(p) for p in ([12, -2], [12, 10], [4, 10], [4, -2])]


def test_wall_thicknesses_are_the_median_of_each_kind_with_the_usual_value_for_a_kind_there_is_none_of():
    square = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
    other = ((4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0))
    outer = [((0.0, 0.0), (4.0, 0.0), 0.25), ((0.0, 0.0), (0.0, 4.0), 0.35), ((0.0, 4.0), (4.0, 4.0), 0.45)]
    one_room = plan_scene.build(walls=outer, rooms=[("a", "A", "other", square)])
    assert plan_scene.thicknesses(one_room) == (0.35, 0.1)
    shared = [*outer, ((4.0, 0.0), (4.0, 4.0), 0.15)]
    two_rooms = plan_scene.build(walls=shared, rooms=[("a", "A", "other", square), ("b", "B", "other", other)])
    assert plan_scene.thicknesses(two_rooms) == (0.35, 0.15)
    # a wall out in the yard, with a room on neither side, is neither kind
    yard = plan_scene.build(walls=[((20.0, 0.0), (24.0, 0.0), 0.9)], rooms=[("a", "A", "other", square)])
    assert plan_scene.thicknesses(yard) == (0.2, 0.1)


@pytest.fixture
def home(tmp_path):
    rev = places.revise(tmp_path, None, producer="footprints-v1", plan=FLAT, now=T)
    return rev.place, rev.id


def test_a_cameras_spot_on_an_owners_plan_puts_it_in_that_place(tmp_path, home):
    assert places.spot_places(tmp_path) == {}, "a machine plan's spots are not the owner's"
    places.correct_plan(tmp_path, home[0], FLAT, now=T + 5)
    assert places.spot_places(tmp_path) == {"porch": home[0], "den-cam": home[0]}


def test_a_scene_saved_from_the_editor_is_the_places_plan_whatever_its_size(tmp_path, home):
    scene = json.loads(FLAT)
    scene["nodes"]["zone_den"]["floorFinish"] = "x" * 200_000  # an editor's save holds every default of every node
    big = json.dumps(scene)
    code, reply = view.correct({"place": home[0], "plan": big}, True, tmp_path, now=T + 10)
    assert (code, reply["plan"], reply["plan_source"]) == (200, big, "owner")


@pytest.mark.parametrize(
    ("plan", "why"),
    [
        pytest.param(plan_scene.dump(plan_scene.build(walls=[((0.0, 0.0), (4.0, 0.0), 0.2)])), "no room", id="no-room"),
        pytest.param("{not a scene", "isn't a scene", id="not-json"),
        pytest.param('plan "Home"\nroom den "Den" living rect 0,0 4x6\n', "isn't a scene", id="the-old-line-dsl"),
        pytest.param(
            '{"nodes": {}, "rootNodeIds": [], "pad": "' + "x" * view.MAX_SCENE_BYTES + '"}', "too long", id="too-long"
        ),
    ],
)
def test_a_plan_the_readers_cant_use_is_refused_with_why(tmp_path, home, plan, why):
    code, reply = view.correct({"place": home[0], "plan": plan}, True, tmp_path, now=T + 10)
    assert code == 400 and why in reply["error"]
    assert view.listing(tmp_path, now=T + 60)[1]["places"][0]["plan_source"] == "machine"


def test_a_scene_edited_on_its_page_is_moved_into_the_plan_frame(tmp_path, home):
    place, frame = home
    quarter = {"turn": math.pi / 2, "shift": [10.0, 0.0], "mirror": False}
    in_frame = pi.moved(FLAT, pi.Move(math.pi / 2, (10.0, 0.0), False))
    places.correct_plan(tmp_path, place, in_frame, now=T + 1, base=frame, view={"plan": FLAT, **quarter})
    code, reply = view.correct({"place": place, "plan": FLAT}, True, tmp_path, now=T + 10)
    assert (code, reply["plan"]) == (200, FLAT)
    held = places.get(tmp_path, place)
    assert held is not None and held.plan is not None
    np.testing.assert_allclose(plan_scene.rooms_of(held.plan)[0][2], [(10, 0), (10, 3), (6, 3), (6, 0)], atol=1e-9)


def test_the_editor_gets_the_plan_that_holds_and_opening_it_stores_nothing(tmp_path, home):
    place, frame = home
    places.correct_plan(tmp_path, place, FLAT.replace('"Hall"', '"Lobby"'), now=T + 5)
    code, reply = view.plan_as_scene(tmp_path, place)
    assert (code, reply["base"], set(reply)) == (200, frame, {"place", "scene", "base"})
    assert plan_scene.rooms(reply["scene"])[0][1] == "Lobby"
    assert len(places.plan_history(tmp_path, place)) == 1


def test_an_earlier_version_of_the_plan_opens_by_its_record(tmp_path, home):
    place, _ = home
    places.correct_plan(tmp_path, place, FLAT.replace('"Hall"', '"Lobby"'), now=T + 5)
    places.correct_plan(tmp_path, place, FLAT, now=T + 10)
    first = places.plan_history(tmp_path, place)[0]
    code, reply = view.plan_as_scene(tmp_path, place, first.id)
    assert code == 200 and plan_scene.rooms(reply["scene"])[0][1] == "Lobby"


def test_a_plan_drawn_on_its_own_page_opens_as_drawn(tmp_path, home):
    place, frame = home
    in_frame = pi.moved(FLAT, pi.Move(math.pi / 2, (10.0, 0.0), False))
    quarter = {"turn": math.pi / 2, "shift": [10.0, 0.0], "mirror": False}
    places.correct_plan(tmp_path, place, in_frame, now=T + 1, base=frame, view={"plan": FLAT, **quarter})
    assert view.plan_as_scene(tmp_path, place)[1]["scene"] == json.loads(FLAT)


def test_what_cant_be_opened_in_the_editor_says_why(tmp_path, home):
    place, _ = home
    assert view.plan_as_scene(tmp_path, "no-such-place")[0] == 404
    assert view.plan_as_scene(tmp_path, place, "no-such-record") == (
        404,
        {"error": "no such version of this place's plan"},
    )
    bare = places.revise(tmp_path, None, producer="cameras-v1", now=T + 1).place
    assert view.plan_as_scene(tmp_path, bare) == (404, {"error": "this place has no plan"})
    # a plan saved before scenes, still in the store: it is no longer read
    places.correct_plan(tmp_path, place, 'plan "x"\nroom a "A" other rect 0,0 4x4\n', now=T + 5)
    assert view.plan_as_scene(tmp_path, place) == (422, {"error": "it is in an old format that is no longer read"})
    places.correct_plan(tmp_path, place, "", now=T + 9)
    withdrawn = places.plan_history(tmp_path, place)[-1]
    assert view.plan_as_scene(tmp_path, place, withdrawn.id) == (404, {"error": "that version withdrew the plan"})
