"""Places and visits: machine revisions on disk, the owner's names and plan corrections over them."""

import json
from unittest.mock import create_autospec

import pytest

from liberational import knowledge, knowledge_cli
from liberational import segment_log as S
from liberational.memory import places
from liberational.people import registry
from liberational.see import zones

T = 1_800_000_000.0
PLAN_A = 'plan "Home" walls 0.2/0.1\nroom living "Living" living rect 0,0 5x4\n'
PLAN_B = PLAN_A + 'room wc "WC" wc rect 5,0 1.2x2\n'
OWNER_PLAN = 'plan "Home" walls 0.2/0.1\nroom living "Living room" living rect 0,0 5.2x4\n'
UNKNOWN = "0190a000-0000-7000-8000-000000000000"


def first(tmp_path, *, now=T, signals=None, **kw):
    sig = {"wifi": ["aa:bb"]} if signals is None else signals
    return places.revise(tmp_path, None, producer="tier1-v1", signals=sig, now=now, **kw)


def held(tmp_path, place):
    got = places.get(tmp_path, place)
    assert got is not None
    return got


def test_a_revision_carries_over_what_it_doesnt_replace(tmp_path):
    one = first(tmp_path, map_bytes=b"cloud-1", map_format="ply-zstd", plan=PLAN_A, sources=["v1"])
    two = places.revise(tmp_path, one.place, producer="tier1-v1", plan=PLAN_B, sources=["v2"], now=T + 60)
    assert two.place == one.place
    assert places.read_map(tmp_path, two) == b"cloud-1", "the map wasn't given, so the earlier one holds"
    assert two.map_format == "ply-zstd"
    assert two.signals == {"wifi": ["aa:bb"]}
    assert two.sources == ("v2",)
    assert places.machine_plan(tmp_path, two) == PLAN_B
    assert places.machine_plan(tmp_path, one) == PLAN_A, "the earlier revision is never edited"
    assert [r.id for r in places.revisions(tmp_path, one.place)] == [one.id, two.id]


def test_only_the_newest_maps_are_kept_and_every_plan_stays(tmp_path):
    revs = [first(tmp_path, map_bytes=b"m0", map_format="ply", plan="p0")]
    for i in (1, 2, 3):
        rev = places.revise(
            tmp_path, revs[0].place, producer="t", map_bytes=f"m{i}".encode(), map_format="ply", plan=f"p{i}", now=T + i
        )
        revs.append(rev)
    # MAPS_KEPT is 2; m0 stays too, as the only (so the newest) map of its producer
    assert [places.read_map(tmp_path, r) for r in revs] == [b"m0", None, b"m2", b"m3"]
    assert [places.machine_plan(tmp_path, r) for r in revs] == ["p0", "p1", "p2", "p3"]


def test_a_clock_step_back_still_makes_the_new_revision_the_newest(tmp_path):
    one = first(tmp_path, plan=PLAN_A)
    two = places.revise(tmp_path, one.place, producer="t", plan=PLAN_B, now=T - 3600)
    assert places.latest(tmp_path, one.place) == two
    assert two.at > one.at


def test_a_half_written_or_foreign_file_is_no_revision(tmp_path):
    one = first(tmp_path)
    folder = places.root(tmp_path) / one.place
    (folder / f".{one.id}.json.123.tmp").write_text("{")
    (folder / f"{UNKNOWN}.json").write_text("{not json")
    stray = "0190a000-0000-7000-8000-000000000001"
    elsewhere = {**one.as_dict(), "id": stray, "place": "0190a000-0000-7000-8000-00000000000f"}
    (folder / f"{stray}.json").write_text(json.dumps(elsewhere))  # filed under the wrong place
    assert places.revisions(tmp_path, one.place) == [one]


def test_a_map_must_name_its_format(tmp_path):
    with pytest.raises(ValueError, match="format"):
        first(tmp_path, map_bytes=b"cloud")
    assert places.place_ids(tmp_path) == []


def test_places_read_as_numbered_placeholders_in_the_order_they_were_made(tmp_path):
    later = first(tmp_path, now=T + 10)
    earlier = first(tmp_path, now=T)
    got = [(p.id, p.name, p.named) for p in places.places(tmp_path)]
    assert got == [(earlier.place, "Place 1", False), (later.place, "Place 2", False)]


def test_the_owners_name_replaces_the_placeholder_and_an_empty_name_brings_it_back(tmp_path):
    one = first(tmp_path)
    places.name(tmp_path, one.place, "  The   cabin ", now=T + 1)
    named = held(tmp_path, one.place)
    assert (named.name, named.named) == ("The cabin", True)
    places.name(tmp_path, one.place, "", now=T + 2)
    back = held(tmp_path, one.place)
    assert (back.name, back.named) == ("Place 1", False)


def test_a_name_is_taught_state_that_keeps_the_places_signals(tmp_path):
    one = first(tmp_path, signals={"wifi": ["aa:bb", "cc:dd"]})
    places.name(tmp_path, one.place, "Office", now=T + 1)
    (row,) = knowledge.Store(registry.knowledge_path(tmp_path)).query(places.NAMED, key=one.place)
    assert row.record.value == {
        "place": one.place,
        "name": "Office",
        "signals": {"wifi": ["aa:bb", "cc:dd"]},
        "source": "owner",
    }


@pytest.mark.parametrize(
    ("call", "place", "arg", "error"),
    [
        pytest.param(places.name, None, "x" * 81, ValueError, id="name-too-long"),
        pytest.param(places.name, UNKNOWN, "Home", KeyError, id="name-unknown-place"),
        pytest.param(places.correct_plan, UNKNOWN, PLAN_A, KeyError, id="plan-unknown-place"),
    ],
)
def test_owner_input_for_a_bad_name_or_an_unknown_place_is_refused(tmp_path, call, place, arg, error):
    known = first(tmp_path).place
    with pytest.raises(error):
        call(tmp_path, place or known, arg)
    assert not registry.knowledge_path(tmp_path).exists(), "nothing was written"


def test_the_owners_names_and_plans_cross_an_export_and_import(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    one = first(source, plan=PLAN_A)
    places.name(source, one.place, "Office", now=T + 1)
    places.correct_plan(source, one.place, OWNER_PLAN, now=T + 2)
    artifact = tmp_path / "taught.zip"
    monkeypatch.setattr(knowledge_cli, "_files", create_autospec(knowledge_cli._files, return_value={}))
    knowledge_cli.main(["export", "--source", "sqlite", "--state", str(source), "--output", str(artifact)])
    registry.close_all()
    knowledge_cli.main(["import", "--state", str(destination), "--input", str(artifact)])
    store = knowledge.Store(registry.knowledge_path(destination))
    assert [r.record.value["name"] for r in store.query(places.NAMED, key=one.place)] == ["Office"]
    assert [r.record.value["plan"] for r in store.query(places.PLAN, key=one.place)] == [OWNER_PLAN]


def test_a_place_with_no_plan_says_so(tmp_path):
    one = first(tmp_path)
    p = held(tmp_path, one.place)
    assert (p.plan, p.plan_source, p.machine_newer) == (None, None, False)


def test_the_owners_plan_holds_over_a_newer_machine_plan_until_withdrawn(tmp_path):
    one = first(tmp_path, plan=PLAN_A)
    places.correct_plan(tmp_path, one.place, OWNER_PLAN, now=T + 10)
    p = held(tmp_path, one.place)
    assert (p.plan, p.plan_source, p.machine_newer) == (OWNER_PLAN, "owner", False)

    places.revise(tmp_path, one.place, producer="tier2-v1", sources=["v9"], now=T + 20)  # no new plan
    assert held(tmp_path, one.place).machine_newer is False, "a revision that kept the old plan isn't newer"

    places.revise(tmp_path, one.place, producer="tier2-v1", plan=PLAN_B, now=T + 30)
    p = held(tmp_path, one.place)
    assert (p.plan, p.plan_source, p.machine_newer) == (OWNER_PLAN, "owner", True)
    assert places.machine_plan(tmp_path, p.revision) == PLAN_B, "the machine plan is kept beside the owner's"

    places.correct_plan(tmp_path, one.place, "", now=T + 40)
    p = held(tmp_path, one.place)
    assert (p.plan, p.plan_source, p.machine_newer) == (PLAN_B, "machine", False)


def test_a_newest_revision_with_no_machine_plan_has_none_newer_than_the_owners(tmp_path):
    one = first(tmp_path, plan=PLAN_A)
    places.correct_plan(tmp_path, one.place, OWNER_PLAN, now=T + 10)
    places.revise(tmp_path, one.place, producer="tier2-v1", plan=PLAN_B, now=T + 20)
    # a joined map that derives no plan: a new map, so the old plan isn't carried over
    places.revise(tmp_path, one.place, producer="join", map_bytes=b"m", map_format="ply", now=T + 30)
    p = held(tmp_path, one.place)
    assert (p.plan, p.plan_source, p.machine_newer) == (OWNER_PLAN, "owner", False)


def test_reading_a_state_with_nothing_finds_no_places_and_creates_nothing(tmp_path):
    assert places.places(tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_visits_are_read_in_time_order_and_filtered_by_place(tmp_path, vision):
    home, office = S.new_id(), S.new_id()
    for start, place in ((T + 600, home), (T, office), (T + 100, home), (T + 9000, home)):
        vision.add(
            S.PLACE_VISITS, ts_start=start, ts_end=start + 50, cam="rig", place=place, producer="p", basis="wifi"
        )
    got = places.visits(tmp_path, T, T + 3600, place=home)
    assert [(v.start, v.place, v.slot) for v in got] == [(T + 100, home, "rig"), (T + 600, home, "rig")]
    assert [v.place for v in places.visits(tmp_path, T, T + 3600)] == [office, home, home]


def test_a_visit_record_round_trips_through_the_place_visit_columns():
    v = places.Visit(S.new_id(), S.new_id(), T, T + 90.5, "place-detect-v1", "wifi", "rig")
    spec = S.STREAMS[S.PLACE_VISITS]
    record = {
        "id": v.id,
        "place": v.place,
        "ts_start": v.start,
        "ts_end": v.end,
        "producer": v.producer,
        "basis": v.basis,
    }
    assert places.Visit.from_record(spec.to_record(spec.to_row(record), "rig")) == v


# The journal's view

BOX = [[0.1, 0.6], [0.3, 0.6], [0.3, 0.9], [0.1, 0.9]]


def zone_set(tmp_path):
    """A camera with two rooms and a TV screen drawn; returns their ids by name."""
    drawn = zones.parse_zones(
        [
            {"name": "Kitchen", "polygon": BOX},
            {"name": "Hall", "polygon": [[0.5, 0.1], [0.9, 0.1], [0.9, 0.4], [0.5, 0.4]]},
            {"name": "TV", "polygon": [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.5, 0.9]], "kind": "screen"},
        ]
    )
    zones.define(tmp_path, "cam0", drawn, None, now=T)
    return {z.name: z.id for z in drawn}


def zone_visit(vision, ids, name, start, end, kind="person"):
    vision.add(S.ZONE_VISITS, ts_start=start, ts_end=end, cam="cam0", zone=ids[name], zone_name=name, subject_kind=kind)


def place_visit(vision, place, start, end):
    vision.add(S.PLACE_VISITS, ts_start=start, ts_end=end, cam="rig", place=place, producer="p", basis="wifi")


def test_a_span_lists_its_places_in_visit_order_clipped_to_it_with_their_names(tmp_path, vision):
    home, cafe = first(tmp_path, now=T - 9000).place, first(tmp_path, now=T - 8000).place
    places.name(tmp_path, cafe, "Riverside Cafe", now=T)
    place_visit(vision, home, T - 600, T + 1000)  # began before the span
    place_visit(vision, cafe, T + 1200, T + 2000)
    place_visit(vision, home, T + 2500, T + 5000)  # back home, past the span's end
    got = places.places_for_span(tmp_path, T, T + 3600)
    assert got == [
        places.PlaceRef(home, "Place 1", False, T, T + 3600, ()),
        places.PlaceRef(cafe, "Riverside Cafe", True, T + 1200, T + 2000, ()),
    ]


def test_rooms_are_the_area_zones_a_person_was_in_filed_under_the_place_visited_then(tmp_path, vision):
    ids = zone_set(tmp_path)
    home, cafe = first(tmp_path, now=T - 9000).place, first(tmp_path, now=T - 8000).place
    place_visit(vision, home, T, T + 1000)
    place_visit(vision, cafe, T + 1200, T + 2000)
    zone_visit(vision, ids, "Hall", T + 100, T + 110)
    zone_visit(vision, ids, "TV", T + 120, T + 900)  # a screen, not a room
    zone_visit(vision, ids, "Kitchen", T + 200, T + 300, kind="animal")  # only people's visits name rooms
    zone_visit(vision, ids, "Kitchen", T + 400, T + 500)
    zone_visit(vision, ids, "Hall", T + 600, T + 610)  # seen again: listed once
    zone_visit(vision, ids, "Kitchen", T + 1500, T + 1600)  # while the rig was at the cafe
    got = {r.place: r.rooms for r in places.places_for_span(tmp_path, T, T + 3600)}
    assert got == {home: ("Hall", "Kitchen"), cafe: ("Kitchen",)}


def test_rooms_seen_with_no_place_visit_come_last_under_no_place(tmp_path, vision):
    ids = zone_set(tmp_path)
    home = first(tmp_path).place
    place_visit(vision, home, T + 1000, T + 2000)
    zone_visit(vision, ids, "Kitchen", T + 100, T + 200)
    zone_visit(vision, ids, "Hall", T + 1100, T + 1200)
    got = places.places_for_span(tmp_path, T, T + 3600)
    assert [(r.place, r.name, r.start, r.end, r.rooms) for r in got] == [
        (home, "Place 1", T + 1000, T + 2000, ("Hall",)),
        (None, "", T + 100, T + 200, ("Kitchen",)),
    ]


@pytest.mark.parametrize(("start", "end"), [(T, T + 3600), (T + 10, T)], ids=["nothing-known", "empty-span"])
def test_a_span_with_nothing_known_has_no_places(tmp_path, vision, start, end):
    del vision
    assert places.places_for_span(tmp_path, start, end) == []


POSE = places.CameraPose(tuple(float(x) for x in range(16)), 380.5, 512, 288)


def test_registered_camera_poses_are_read_back_and_carry_over_until_replaced(tmp_path):
    rev = first(tmp_path, cameras={"cam0": POSE})
    again = places.revise(tmp_path, rev.place, producer="tier1-v2", now=T + 1)
    assert places.revisions(tmp_path, rev.place)[0].cameras == {"cam0": POSE}
    assert again.cameras == {"cam0": POSE}
    moved = places.CameraPose(POSE.pose, 400.0, 512, 288)
    assert places.revise(tmp_path, rev.place, producer="p", cameras={"cam1": moved}, now=T + 2).cameras == {
        "cam1": moved
    }


def test_a_camera_pose_that_isnt_a_4x4_matrix_is_no_revision(tmp_path):
    rev = first(tmp_path, cameras={"cam0": POSE})
    path = places.root(tmp_path) / rev.place / f"{rev.id}.json"
    data = json.loads(path.read_text())
    data["cameras"]["cam0"]["pose"] = [1.0, 2.0]
    path.write_text(json.dumps(data))
    assert places.revisions(tmp_path, rev.place) == []


def registered(at):
    return places.CameraPose(POSE.pose, POSE.focal, POSE.width, POSE.height, at)


def test_a_camera_belongs_to_the_place_that_registered_it_most_recently(tmp_path):
    old = first(tmp_path, now=T, cameras={"cam0": registered(T), "cam1": registered(T)})
    new = first(tmp_path, now=T + 100, cameras={"cam1": registered(T + 100)})
    assert places.camera_places(tmp_path) == {"cam0": old.place, "cam1": new.place}


def test_a_later_revision_that_only_carries_a_moved_camera_over_doesnt_take_it_back(tmp_path):
    a = first(tmp_path, now=T, cameras={"cam0": registered(T)}).place
    b = first(tmp_path, now=T + 100, cameras={"cam0": registered(T + 100)}).place
    carried = places.revise(tmp_path, a, producer="plan job", plan=PLAN_A, now=T + 200)  # A is the newest revision now
    assert carried.cameras == {"cam0": registered(T)}
    assert places.camera_places(tmp_path) == {"cam0": b}


def test_a_floor_or_plan_transform_given_alone_replaces_the_old_one(tmp_path):
    rev = first(tmp_path, map_bytes=b"x", map_format="ply", floor=(0.0, 1.0, 0.0, 0.0), plan=PLAN_A)
    refit = (0.0, 1.0, 0.0, -0.1)
    moved = TO_PLAN[:3] + (2.0,) + TO_PLAN[4:]
    again = places.revise(tmp_path, rev.place, producer="p", floor=refit, plan_transform=moved, now=T + 1)
    assert (again.floor, again.plan_transform) == (refit, moved)
    assert (again.map, again.plan) == (rev.map, rev.plan)


def test_a_registered_fixed_cameras_rooms_are_filed_under_its_place_wherever_the_rig_was(tmp_path, vision):
    ids = zone_set(tmp_path)
    home = first(tmp_path, now=T - 9000, cameras={"cam0": POSE}).place
    cafe = first(tmp_path, now=T - 8000).place
    place_visit(vision, cafe, T + 1000, T + 2000)
    zone_visit(vision, ids, "Kitchen", T + 1500, T + 1600)  # someone at home while the rig was at the cafe
    zone_visit(vision, ids, "Hall", T + 2500, T + 2600)  # and after the rig left the cafe
    got = places.places_for_span(tmp_path, T, T + 3600)
    assert [(r.place, r.start, r.end, r.rooms, r.visited) for r in got] == [
        (cafe, T + 1000, T + 2000, (), True),
        (home, T + 1500, T + 2600, ("Kitchen", "Hall"), False),  # observed only: the owner was out
    ]


def test_a_registered_cameras_place_the_rig_also_visited_counts_as_visited(tmp_path, vision):
    ids = zone_set(tmp_path)
    home = first(tmp_path, now=T - 9000, cameras={"cam0": POSE}).place
    place_visit(vision, home, T + 1000, T + 2000)
    zone_visit(vision, ids, "Kitchen", T + 2500, T + 2600)  # seen after the owner left: still the same place
    (ref,) = places.places_for_span(tmp_path, T, T + 3600)
    assert (ref.place, ref.rooms, ref.visited) == (home, ("Kitchen",), True)


# MASt3R's frame is y-down: a floor at y = 0 has its up normal at -y, so a point's height above it is -y.
FLOOR = (0.0, -1.0, 0.0, 0.0)
# Map (x, y, z) to plan (x, z, height -y).
TO_PLAN = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0)
# A level camera 2 m above the floor at x = 1, looking along +z, with a 288 px tall frame and a 144 px focal length:
# the bottom row's ray drops 1 m for every metre forward.
LEVEL = places.CameraPose((1.0, 0, 0, 1.0, 0, 1.0, 0, -2.0, 0, 0, 1.0, 0.0, 0, 0, 0, 1.0), 144.0, 512, 288)


def floor_place(tmp_path, **kw):
    args = {"cameras": {"cam0": LEVEL}, "floor": FLOOR, "plan_transform": TO_PLAN, "plan": PLAN_A, "map_bytes": b"x"}
    return first(tmp_path, map_format="ply", **(args | kw))


@pytest.mark.parametrize(
    ("u", "v", "want"),
    [
        pytest.param(0.5, 1.0, (1.0, 2.0), id="bottom-middle-lands-2m-ahead"),
        pytest.param(0.5, 0.75, (1.0, 4.0), id="halfway-down-lands-4m-ahead"),
        # 256 px left of centre at a 144 px focal: 1.778 m left per metre forward; 2 m forward, x = 1 - 3.556
        pytest.param(0.0, 1.0, (-2.556, 2.0), id="bottom-left-is-off-to-the-side"),
    ],
)
def test_a_pixel_of_a_registered_camera_lands_on_the_floor_in_plan_metres(tmp_path, u, v, want):
    place = floor_place(tmp_path).place
    assert places.camera_to_plan(tmp_path, "cam0", u, v) == (place, *want)


def test_a_plan_shifted_from_the_map_moves_the_point_with_it(tmp_path):
    shifted = (1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0)  # TO_PLAN, 10 m along plan x
    place = floor_place(tmp_path, plan_transform=shifted).place
    assert places.camera_to_plan(tmp_path, "cam0", 0.5, 1.0) == (place, 11.0, 2.0)


@pytest.mark.parametrize(
    ("u", "want"),
    [
        # The y-down floor's frame: the map's x as x, up (-y) cross x = the map's z as y
        pytest.param(0.5, (1.0, 2.0), id="bottom-middle-lands-2m-ahead"),
        pytest.param(0.0, (-2.556, 2.0), id="bottom-left-is-off-to-the-side"),
    ],
)
def test_a_lone_cameras_pixel_lands_in_its_floors_frame_without_a_plan(tmp_path, u, want):
    place = floor_place(tmp_path, plan_transform=None, plan=None).place
    assert places.camera_to_plan(tmp_path, "cam0", u, 1.0) == (place, *want)


def test_a_revision_with_no_plan_works_out_its_floor_frame_once_for_many_pixels(tmp_path, monkeypatch):
    # A rebuilt revision has no plan file of its own, so every point a positions read projects through it goes through
    # its floor's frame: that's the floor's, worked out once.
    rev = floor_place(tmp_path, plan_transform=None, plan=None, floor=(0.0, -1.0, 0.0, 0.25))
    frame = places.floor_frame
    calls = []
    monkeypatch.setattr(places, "floor_frame", lambda floor: calls.append(floor) or frame(floor))
    hits = [places.project(rev, "cam0", u, 1.0) for u in (0.0, 0.5, 1.0)]
    assert len(calls) == 1
    assert hits[1] == (1.0, pytest.approx(2.25))  # 2.25 m up, the ray dropping a metre a metre


def test_a_floor_whose_up_is_the_maps_x_takes_the_maps_y_as_its_x():
    # Height above the floor is (2x - 4) / 2 = x - 2; x along the map's y, y = up (x) cross that = the map's z
    tr = places.floor_frame((2.0, 0.0, 0.0, -4.0))
    assert tr == pytest.approx((0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, -2))


def test_a_floor_with_no_normal_has_no_frame():
    assert places.floor_frame((0.0, 0.0, 0.0, 1.0)) is None


@pytest.mark.parametrize(
    ("camera", "v", "kw"),
    [
        pytest.param("cam0", 0.5, {}, id="looking-at-the-horizon"),
        pytest.param("cam0", 0.2, {}, id="above-the-horizon"),
        pytest.param("cam9", 1.0, {}, id="not-registered"),
        pytest.param("cam0", 1.0, {"floor": None}, id="no-floor"),
    ],
)
def test_a_pixel_with_no_floor_point_in_a_plan_frame_has_none(tmp_path, camera, v, kw):
    floor_place(tmp_path, **kw)
    assert places.camera_to_plan(tmp_path, camera, 0.5, v) is None


def test_the_floor_goes_with_its_map_and_the_transform_with_its_plan(tmp_path):
    rev = floor_place(tmp_path)
    same = places.revise(tmp_path, rev.place, producer="p", signals={"wifi": []}, now=T + 1)
    assert (same.floor, same.plan_transform) == (FLOOR, TO_PLAN)
    new_plan = places.revise(tmp_path, rev.place, producer="p", plan=PLAN_B, now=T + 2)
    assert (new_plan.floor, new_plan.plan_transform) == (FLOOR, None)
    new_map = places.revise(tmp_path, rev.place, producer="p", map_bytes=b"y", map_format="ply", now=T + 3)
    # a plan is drawn in its map's frame: a new map without a plan of its own keeps neither the old plan nor its frame
    assert (new_map.floor, new_map.plan, new_map.plan_transform) == (None, None, None)
    assert places.revisions(tmp_path, rev.place)[0].floor == FLOOR


def test_registrations_made_at_the_same_time_go_to_the_more_recently_revised_place(tmp_path):
    unstamped = places.CameraPose(POSE.pose, POSE.focal, POSE.width, POSE.height)  # at = 0, as in unstamped poses
    first(tmp_path, now=T, cameras={"cam0": unstamped})
    newer = first(tmp_path, now=T + 100, cameras={"cam0": unstamped}).place
    assert places.camera_places(tmp_path) == {"cam0": newer}


def test_a_producers_newest_map_survives_newer_maps_from_others(tmp_path):
    fused = places.revise(tmp_path, None, producer="fuse", map_bytes=b"fused", map_format="ply", now=T)
    place = fused.place
    for i in (1, 2, 3):  # snapshots that didn't fuse, day after day
        places.revise(tmp_path, place, producer="snapshot", map_bytes=f"s{i}".encode(), map_format="ply", now=T + i)
    revs = places.revisions(tmp_path, place)
    assert [places.read_map(tmp_path, r) for r in revs] == [b"fused", None, b"s2", b"s3"]


def test_a_retired_producers_last_map_goes_once_it_falls_out_of_the_recent_revisions(tmp_path):
    old = places.revise(tmp_path, None, producer="fuse v1", map_bytes=b"v1", map_format="ply", now=T)
    for i in range(1, places.PRODUCER_WINDOW):  # v1 is still among the last PRODUCER_WINDOW revisions
        places.revise(tmp_path, old.place, producer="fuse v2", map_bytes=f"m{i}".encode(), map_format="ply", now=T + i)
    assert places.read_map(tmp_path, old) == b"v1"
    places.revise(tmp_path, old.place, producer="fuse v2", map_bytes=b"last", map_format="ply", now=T + 100)
    assert places.read_map(tmp_path, old) is None  # v1 fell out of the window


def test_a_revisions_splat_is_written_once_and_only_the_newest_two_are_kept(tmp_path):
    revs = [places.revise(tmp_path, None, producer="p", now=T)]
    place = revs[0].place
    revs += [places.revise(tmp_path, place, producer="p", now=T + i) for i in range(1, 3)]
    for i, rev in enumerate(revs):
        places.write_splat(tmp_path, rev, master=b"ply%d" % i, web=b"sog%d" % i)
    assert [places.has_splat(tmp_path, r) for r in revs] == [False, True, True]
    assert not places.splat_path(tmp_path, revs[0], "gs.ply").exists()
    assert places.splat_path(tmp_path, revs[2], "sog").read_bytes() == b"sog2"
    with pytest.raises(ValueError, match="already has a splat"):
        places.write_splat(tmp_path, revs[2], master=b"again", web=b"again")
    assert places.splat_path(tmp_path, revs[2], "gs.ply").read_bytes() == b"ply2"


def test_a_splat_says_what_it_was_trained_from_when_told(tmp_path):
    told = places.revise(tmp_path, None, producer="p", now=T)
    untold = places.revise(tmp_path, told.place, producer="p", now=T + 1)
    bare = places.revise(tmp_path, told.place, producer="p", now=T + 2)
    places.write_splat(tmp_path, told, master=b"ply", web=b"sog", source="walk")
    places.write_splat(tmp_path, untold, master=b"ply", web=b"sog")
    assert places.splat_source(tmp_path, told) == "walk"
    assert places.splat_source(tmp_path, untold) is None and places.splat_source(tmp_path, bare) is None
    assert len(places.revisions(tmp_path, told.place)) == 3, "the note isn't read as a revision"
    places.write_splat(tmp_path, bare, master=b"ply", web=b"sog", source="cameras")
    assert not places.splat_path(tmp_path, told, "gs.from").exists(), "pruned with its splat"


def test_a_retired_place_is_listed_nowhere_and_reads_as_the_place_it_was_merged_into(tmp_path):
    cam = places.CameraPose(tuple(float(x) for x in range(16)), 300.0, 640, 360, T)
    home = first(tmp_path, map_bytes=b"home", map_format="ply", cameras={"hall": cam})
    den = first(tmp_path, now=T + 1, map_bytes=b"den", map_format="ply", cameras={"den": cam})
    places.retire(tmp_path, den.place, home.place, now=T + 2)
    assert places.place_ids(tmp_path) == [home.place]
    assert [p.id for p in places.places(tmp_path)] == [home.place]
    assert places.camera_places(tmp_path) == {"hall": home.place}, "the den's camera is no longer registered there"
    assert places.merged_into(tmp_path, den.place) == home.place
    assert places.merged_into(tmp_path, home.place) == home.place
    assert places.merged_into(tmp_path, UNKNOWN) is None
    assert len(places.revisions(tmp_path, den.place)) == 2, "its earlier revision stays as the record"


def test_a_chain_of_merges_reads_as_the_last_place(tmp_path):
    a, b, c = (first(tmp_path, now=T + k) for k in range(3))
    places.retire(tmp_path, a.place, b.place, now=T + 10)
    places.retire(tmp_path, b.place, c.place, now=T + 11)
    assert places.merged_into(tmp_path, a.place) == c.place


def test_a_place_isnt_merged_into_itself_or_into_no_place(tmp_path):
    one = first(tmp_path)
    with pytest.raises(ValueError, match="itself"):
        places.retire(tmp_path, one.place, one.place)
    with pytest.raises(KeyError):
        places.retire(tmp_path, one.place, UNKNOWN)
    assert places.place_ids(tmp_path) == [one.place]


def test_a_name_carried_over_by_a_merge_says_so(tmp_path):
    one = first(tmp_path)
    places.name(tmp_path, one.place, "Cottage", now=T + 1, source="merge")
    record = places._owned(tmp_path, places.NAMED)[one.place]
    assert record.value["source"] == "merge" and held(tmp_path, one.place).name == "Cottage"


def test_an_owner_plan_replaced_stays_in_the_history_and_can_be_restored_after_an_export(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    one = first(source, plan=PLAN_A)
    later = places.revise(source, one.place, producer="tier1-v1", map_bytes=b"m", map_format="ply", now=T + 1)
    places.correct_plan(source, one.place, OWNER_PLAN, now=T + 2, base=one.id)
    places.correct_plan(source, one.place, PLAN_B, now=T + 3, base=later.id)  # an imported plan, in another frame
    artifact = tmp_path / "taught.zip"
    monkeypatch.setattr(knowledge_cli, "_files", create_autospec(knowledge_cli._files, return_value={}))
    knowledge_cli.main(["export", "--source", "sqlite", "--state", str(source), "--output", str(artifact)])
    registry.close_all()
    knowledge_cli.main(["import", "--state", str(destination), "--input", str(artifact)])
    history = places.plan_history(destination, one.place)
    assert [(r.value["plan"], r.value["base"]) for r in history] == [(OWNER_PLAN, one.id), (PLAN_B, later.id)]

    places.restore_plan(destination, one.place, history[0].id, now=T + 4)
    after = places.plan_history(destination, one.place)
    assert [r.value["plan"] for r in after] == [OWNER_PLAN, PLAN_B, OWNER_PLAN], "restoring appends, removing nothing"
    assert after[-1].value["base"] == one.id, "the restored plan keeps the frame it was drawn in"


def test_restoring_a_plan_record_the_place_never_had_is_refused(tmp_path):
    one = first(tmp_path, plan=PLAN_A)
    places.correct_plan(tmp_path, one.place, OWNER_PLAN, now=T + 1)
    with pytest.raises(KeyError):
        places.restore_plan(tmp_path, one.place, "not-a-record")
    assert len(places.plan_history(tmp_path, one.place)) == 1


def test_revisions_share_a_plan_frame_by_plan_file_or_without_one_by_map_frame_and_floor(tmp_path):
    floor = (0.0, 0.0, 1.0, 0.0)
    drawn = first(tmp_path, map_bytes=b"m", map_format="ply", floor=floor, plan=PLAN_A)
    unplanned = places.revise(tmp_path, drawn.place, producer="t", map_bytes=b"n", map_format="ply", floor=floor)
    carried = places.revise(tmp_path, drawn.place, producer="t", sources=["v2"], now=T + 5)  # same map, same frame
    new_map = places.revise(tmp_path, drawn.place, producer="t", map_bytes=b"o", map_format="ply", floor=floor)
    assert places.same_plan_frame(unplanned, carried)
    assert not places.same_plan_frame(unplanned, new_map), "a new map is a new frame"
    assert not places.same_plan_frame(drawn, unplanned), "one has a plan file, the other none"
    assert places.plan_frame(unplanned) == places.floor_frame(floor)
