"""A plan as a Pascal scene (spatial/plan_scene.py), the converter from DSL, and the readers that take either."""

import json
import math

import numpy as np
import pytest

from liberational import camera_rename
from liberational.dashboard import places as view
from liberational.memory import places
from liberational.spatial import floorplan, plan_scene
from liberational.spatial import plan_import as pi

T = 1_762_500_000.0

# A made-up two-room flat: a hall and a den side by side, a front door, a doorway between them, and a window.
DSL = """plan "Flat" walls 0.3/0.1

room hall "Hall" hall rect 0,0 3x4
room den "The \\"den\\"" living rect 3,0 5x4

door exterior>hall at 0,2 w0.9 entrance
cased hall>den at 3,1 w1.2
window exterior>den at 5.5,0 w1.6
fixture sofa in:den at 5,3 size 2x0.8 "Sofa"

x-anchor place:flat yaw 0 origin 0,0,0
x-source wall:w3 machine 0.4
x-camera porch at 0.4,0.5 height 2.2 yaw 135 fov 100
x-camera den-cam at 7.5,3.5
x-object lamp "Lamp" at 4,1 size 0.4x0.4 owner
x-unexplored back "Back room" rect 6,3 2x1
"""


@pytest.fixture
def scene_text() -> str:
    return pi.to_scene(DSL).scene


def nodes(text: str, kind: str) -> list[dict]:
    return [n for n in json.loads(text)["nodes"].values() if n["type"] == kind]


def test_a_converted_plan_has_its_rooms_as_named_zones_and_a_wall_for_each_stretch_of_room_edge(scene_text):
    assert [(z["id"], z["name"], z["metadata"]["kind"]) for z in nodes(scene_text, "zone")] == [
        ("zone_hall", "Hall", "hall"),
        ("zone_den", 'The "den"', "living"),
    ]
    # Seven stretches: the hall's three outside walls, the den's three, and the one they share.
    walls = {(tuple(w["start"]), tuple(w["end"])): w["thickness"] for w in nodes(scene_text, "wall")}
    assert len(walls) == 7
    assert walls[((3.0, 0.0), (3.0, 4.0))] == pytest.approx(0.1)  # the shared wall is a partition
    assert walls[((0.0, 0.0), (0.0, 4.0))] == pytest.approx(0.3)
    assert sum(1 for t in walls.values() if t == pytest.approx(0.3)) == 6


def test_a_converted_opening_hangs_on_the_wall_its_point_is_on_at_its_distance_from_the_walls_start(scene_text):
    door, cased = nodes(scene_text, "door")
    (window,) = nodes(scene_text, "window")
    by_id = json.loads(scene_text)["nodes"]
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


def test_a_converted_plan_keeps_its_spots_objects_areas_and_fixtures_and_says_what_it_left(scene_text):
    side = plan_scene.records(json.loads(scene_text))
    assert side["title"] == "Flat"
    assert side["cameras"] == {
        "porch": {"at": [0.4, 0.5], "height": 2.2, "yaw": 135.0, "fov": 100.0},
        "den-cam": {"at": [7.5, 3.5]},
    }
    assert side["objects"] == {"lamp": {"label": "Lamp", "at": [4.0, 1.0], "size": [0.4, 0.4], "source": "owner"}}
    assert side["unexplored"] == {"back": {"name": "Back room", "rect": [6.0, 3.0, 2.0, 1.0]}}
    assert side["fixtures"] == [
        {"kind": "sofa", "room": "den", "poly": [[5.0, 3.0], [7.0, 3.0], [7.0, 3.8], [5.0, 3.8]], "words": ['"Sofa"']}
    ]
    assert pi.to_scene(DSL).left == ("x-anchor place:flat yaw 0 origin 0,0,0", "x-source wall:w3 machine 0.4")


def test_a_scene_reads_as_the_same_plan_its_dsl_does(scene_text):
    was, now = pi.parse(DSL), pi.parse(scene_text)
    assert [(r.id, r.kind, r.polygon) for r in now.rooms] == [(r.id, r.kind, r.polygon) for r in was.rooms]
    assert [r.name for r in now.rooms] == ["Hall", 'The "den"']
    assert now.cameras == was.cameras
    assert now.title == "Flat"
    assert now.walls == pytest.approx((0.3, 0.1))
    # an opening is between the rooms either side of its wall, the outside first; two rooms come in either order
    assert [(o.kind, frozenset(o.between.split(">")), o.width) for o in now.openings] == [
        ("door", frozenset({"exterior", "hall"}), 0.9),
        ("cased", frozenset({"hall", "den"}), 1.2),
        ("window", frozenset({"exterior", "den"}), 1.6),
    ]
    assert [o.between.split(">")[0] for o in now.openings if o.kind != "cased"] == ["exterior", "exterior"]
    assert [o.at for o in now.openings] == [pytest.approx(o.at) for o in was.openings]
    before, after = pi.walls(was), pi.walls(now)
    np.testing.assert_allclose(after.a, before.a)
    np.testing.assert_allclose(after.b, before.b)
    np.testing.assert_allclose(after.half, before.half)


def test_the_room_and_spot_readers_take_a_scene(scene_text):
    assert floorplan.rooms_from_dsl(scene_text) == [
        ("hall", "Hall", ((0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0))),
        ("den", 'The "den"', ((3.0, 0.0), (8.0, 0.0), (8.0, 4.0), (3.0, 4.0))),
    ]
    assert pi.spots(scene_text) == pi.spots(DSL)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("{not json", id="not-json"),
        pytest.param('{"nodes": {}, "rootNodeIds": []}', id="no-level"),
        pytest.param('{"rootNodeIds": []}', id="no-nodes"),
        pytest.param('{"nodes": {"a": {"id": "a"}}, "rootNodeIds": ["a"]}', id="untyped-node"),
    ],
)
def test_a_scene_that_doesnt_read_has_no_rooms_or_spots_and_is_refused_by_the_full_reader(text):
    assert floorplan.rooms_from_dsl(text) == []
    assert pi.spots(text) == ()
    assert plan_scene.spot_names(text) == []
    with pytest.raises(pi.PlanError):
        pi.parse(text)


def test_a_scene_with_walls_but_no_rooms_is_a_plan_with_no_rooms():
    bare = plan_scene.dump(plan_scene.build(walls=[((0.0, 0.0), (4.0, 0.0), 0.2)]))
    with pytest.raises(pi.PlanError, match="no rooms"):
        pi.parse(bare)


def test_an_opening_on_no_wall_is_refused_with_where_it_is():
    with pytest.raises(pi.PlanError, match="door at 1,1 is on no wall"):
        pi.to_scene('plan "x"\nroom a "A" other rect 0,0 4x4\ndoor exterior>a at 1,1 w0.9\n')


def test_what_the_editor_adds_to_a_scene_doesnt_change_what_is_read(scene_text):
    # The editor saves a floor slab and a ceiling per room beside the zones, fills every node with defaults, and
    # may leave a zone unlisted by its level for a moment: none of it is a room, and the zone still is.
    scene = json.loads(scene_text)
    level = scene["nodes"]["level_ground"]
    square = [[0, 0], [8, 0], [8, 4], [0, 4]]
    scene["nodes"]["slab_1"] = {"id": "slab_1", "type": "slab", "parentId": level["id"], "polygon": square}
    scene["nodes"]["ceiling_1"] = {"id": "ceiling_1", "type": "ceiling", "parentId": level["id"], "polygon": square}
    level["children"] += ["slab_1", "ceiling_1"]
    level["children"].remove("zone_den")
    scene["nodes"]["zone_den"].update(autoFromWalls=True, boundaryWallIds=["wall_0"], color="#84cc16")
    scene["nodes"]["zone_stub"] = {"id": "zone_stub", "type": "zone", "parentId": level["id"], "polygon": [[0, 0]]}
    assert floorplan.rooms_from_dsl(json.dumps(scene)) == floorplan.rooms_from_dsl(scene_text)


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


def test_renaming_a_camera_moves_its_spot_and_leaves_a_scene_without_it_untouched(scene_text):
    renamed = camera_rename.rename_plan(scene_text, {"porch": "front-door"})
    assert [c.id for c in pi.spots(renamed)] == ["front-door", "den-cam"]
    assert pi.spots(renamed)[0] == pi.Camera("front-door", (0.4, 0.5), 2.2, 135.0, 100.0)
    assert floorplan.rooms_from_dsl(renamed) == floorplan.rooms_from_dsl(scene_text)
    assert camera_rename.rename_plan(scene_text, {"garage": "shed"}) is scene_text
    assert camera_rename.rename_plan("{broken", {"porch": "x"}) == "{broken"


def test_a_scene_moved_into_another_frame_moves_every_outline_and_turns_each_camera(scene_text):
    # A quarter turn and a shift: (x, y) goes to (10 - y, x), and a camera looking up the page (0) looks east (90).
    text = pi.moved(scene_text, pi.Move(math.pi / 2, (10.0, 0.0), False))
    hall = floorplan.rooms_from_dsl(text)[0]
    assert hall[:2] == ("hall", "Hall")
    np.testing.assert_allclose(hall[2], [(10, 0), (10, 3), (6, 3), (6, 0)], atol=1e-9)
    porch, den = pi.spots(text)
    assert (porch.at, porch.yaw, porch.height) == (pytest.approx((9.5, 0.4)), pytest.approx(225.0), 2.2)
    assert (den.at, den.yaw) == (pytest.approx((6.5, 7.5)), None)
    front = next(w for w in nodes(text, "wall") if w["children"] == ["door_0"])
    assert (front["start"], front["end"]) == (pytest.approx([10, 0]), pytest.approx([6, 0]))
    assert nodes(text, "door")[0]["position"][0] == pytest.approx(2.0)  # still 2 m along its own wall
    side = plan_scene.records(json.loads(text))
    assert side["objects"]["lamp"]["at"] == pytest.approx([9, 4])
    assert "rect" not in side["unexplored"]["back"]  # a turned rectangle is a polygon
    assert side["unexplored"]["back"]["poly"] == [pytest.approx(p) for p in ([7, 6], [7, 8], [6, 8], [6, 6])]
    site = nodes(text, "site")[0]["polygon"]["points"]
    assert site == [pytest.approx(p) for p in ([12, -2], [12, 10], [4, 10], [4, -2])]


def test_moving_a_dsl_plan_still_writes_dsl():
    text = pi.moved(DSL, pi.Move(0.0, (1.0, 0.0), False))
    assert not plan_scene.is_scene(text)
    assert floorplan.rooms_from_dsl(text)[0][2][0] == (1.0, 0.0)


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


def test_a_cameras_spot_on_an_owners_scene_puts_it_in_that_place(tmp_path, scene_text):
    home = places.revise(tmp_path, None, producer="footprints-v1", plan=DSL, now=T)
    places.correct_plan(tmp_path, home.place, scene_text, now=T + 5)
    assert places.spot_places(tmp_path) == {"porch": home.place, "den-cam": home.place}


@pytest.fixture
def home(tmp_path):
    rev = places.revise(tmp_path, None, producer="footprints-v1", plan=DSL, now=T)
    return rev.place, rev.id


def test_a_scene_saved_from_the_editor_is_the_places_plan_whatever_its_size(tmp_path, home, scene_text):
    scene = json.loads(scene_text)
    scene["nodes"]["zone_den"]["floorFinish"] = "x" * (
        2 * view.MAX_PLAN_BYTES
    )  # an editor's save is far past a DSL plan
    big = json.dumps(scene)
    code, reply = view.correct({"place": home[0], "plan": big}, True, tmp_path, now=T + 10)
    assert (code, reply["plan"], reply["plan_source"]) == (200, big, "owner")


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(plan_scene.dump(plan_scene.build(walls=[((0.0, 0.0), (4.0, 0.0), 0.2)])), id="no-room"),
        pytest.param("{not a scene", id="not-json"),
        pytest.param('{"nodes": {}, "rootNodeIds": [], "pad": "' + "x" * view.MAX_SCENE_BYTES + '"}', id="too-long"),
    ],
)
def test_a_scene_the_readers_cant_use_is_refused(tmp_path, home, plan):
    assert view.correct({"place": home[0], "plan": plan}, True, tmp_path, now=T + 10)[0] == 400
    assert view.listing(tmp_path, now=T + 60)[1]["places"][0]["plan_source"] == "machine"


def test_a_scene_edited_on_its_page_is_moved_into_the_plan_frame(tmp_path, home, scene_text):
    place, frame = home
    quarter = {"turn": math.pi / 2, "shift": [10.0, 0.0], "mirror": False}
    in_frame = pi.moved(scene_text, pi.Move(math.pi / 2, (10.0, 0.0), False))
    places.correct_plan(tmp_path, place, in_frame, now=T + 1, base=frame, view={"plan": scene_text, **quarter})
    code, reply = view.correct({"place": place, "plan": scene_text}, True, tmp_path, now=T + 10)
    assert (code, reply["plan"]) == (200, scene_text)
    held = places.get(tmp_path, place)
    assert held is not None and held.plan is not None
    np.testing.assert_allclose(floorplan.rooms_from_dsl(held.plan)[0][2], [(10, 0), (10, 3), (6, 3), (6, 0)], atol=1e-9)
