"""A place's map (spatial/cloud_map.py) and the agreement it asks of a shot (spatial/cloud_merge.py), on made-up
rooms: a shot is what a camera standing somewhere in them would see, offered where it is said to stand."""

import dataclasses
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from liberational.spatial import cloud_align, cloud_map, cloud_merge
from liberational.spatial import cloud_scene as scenes
from test_spatial_cloud_scene import ROOMS, UPRIGHT, at, view_from

EDGES = cloud_align.edges_of(ROOMS)
#: A table in the hall, 1 m by 0.7 m, its top 0.7 m up.
TABLE = ((1.5, 3.5), (2.5, 4.2))


def table_top(where=TABLE, n: int = 2500, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    (x1, y1), (x2, y2) = where
    return np.c_[x1 + rng.random(n) * (x2 - x1), y1 + rng.random(n) * (y2 - y1), np.full(n, 0.7)]


def cupboard(y: float, n: int = 9000, seed: int = 8) -> np.ndarray:
    """The front of a cupboard against the hall's far wall, standing `y` up the plan: 2 m wide, 1.8 m tall."""
    rng = np.random.default_rng(seed)
    return np.c_[1.0 + rng.random(n) * 2.0, np.full(n, y), rng.random(n) * 1.8]


def shot(
    name: str,
    truth: scenes.At,
    said: scenes.At | None = None,
    *,
    held: bool = True,
    score: float = 0.9,
    things: np.ndarray | None = None,
    flat: bool = False,
    camera: str = "hall",
    when: float = 0.0,
) -> cloud_map.Offer:
    """A shot of what a camera really standing at `truth` sees (the rooms' floor and walls, and `things`, points
    of the scene), offered as standing at `said`."""
    said = said or truth
    own = view_from(truth)
    pose = np.array(UPRIGHT).reshape(4, 4)
    if things is not None:
        move = scenes._transform(truth)
        seen = (things - move[:3, 3]) @ move[:3, :3]
        # what stands behind a thing isn't seen: the room's points the thing hides are not in the shot
        hides = cloud_merge.view_of(seen, np.linalg.inv(pose), 60.0, 128, 96)
        behind = cloud_merge.through(hides, own)[1]
        own = np.concatenate(
            [own[behind | (hides.far[np.maximum(cloud_merge._cells(hides.eye, 60.0, 128, 96, own)[0], 0)] == 0)], seen]
        )
    planes = cloud_align.planes(own)
    fit = cloud_align.fit(cloud_align.lines(planes), said.x, said.y, said.bearing, EDGES) if flat else None
    stand = cloud_merge.Stand(said.x, said.y, said.bearing)
    if fit is not None and fit.state == "fitted":
        stand = cloud_merge.Stand(said.x + fit.dx, said.y + fit.dy, said.bearing, fit.scale)
    return cloud_map.Offer(
        name, name, camera, when, held, score if held else 0.0, own, np.full((len(own), 3), 120, np.uint8),
        cloud_merge.surfaces(planes), stand, stand.scale, pose, 60.0, 128, 96, fit,
    )  # fmt: skip


@pytest.fixture(autouse=True)
def coarse_views(monkeypatch):
    """A made-up shot is a few thousand points where a picture gives a point a pixel, so what a shot saw is kept
    on a coarser grid here: one fine enough for a picture would be mostly holes."""
    monkeypatch.setattr(cloud_merge, "GRID", 16)


def the_map(where) -> cloud_map.Held:
    held = cloud_map.read(where)
    assert held is not None
    return held


@pytest.fixture
def where(tmp_path):
    return cloud_map.folder(tmp_path / "camera_clouds", "place-1")


def fullest(points: np.ndarray) -> int:
    """The most points any one cell of a map holds."""
    _, count = np.unique(np.floor(points / cloud_map.CELL_M).astype(np.int64), axis=0, return_counts=True)
    return int(count.max())


def far_wall(held: cloud_map.Held) -> np.ndarray:
    """A map's points on or near the hall's far wall (y = 5), above the floor, as they are drawn flat."""
    flat = held.flat
    return flat[(flat[:, 2] > 0.3) & (flat[:, 1] > 4.5) & (flat[:, 0] > 0.2) & (flat[:, 0] < 3.8)]


def test_a_camera_panned_across_a_room_in_several_shots_ends_as_one_wall_and_a_map_that_fills_in(where):
    stood = (2.0, 1.5)
    counts, added = [], []
    for n, facing in enumerate((-40.0, -15.0, 10.0, 35.0)):
        # each shot is said to stand a few centimetres from where it does, as a real placement is
        said = at(stood[0] + 0.04 * (n - 1.5), stood[1] + 0.03 * (n % 2), facing)
        told = cloud_map.offer(where, ROOMS, shot(f"pan-{n}", at(*stood, facing), said, flat=True))
        assert told.state == "added"
        added.append(told.points)
        counts.append(len(the_map(where).xyz))
    held = the_map(where)
    assert held is not None
    # one wall: every point of the far wall lies on the plan's wall, however many shots saw it
    assert far_wall(held)[:, 1] == pytest.approx(5.0, abs=1e-3)
    # each shot added what the map lacked and no more: the map grew every time, by less than a whole shot after
    # the first, and no cell holds more than the cap
    assert counts == sorted(counts) and all(a > 0 for a in added)
    assert max(added[1:]) < 0.8 * added[0]
    assert fullest(held.flat) <= cloud_map.CAP


def test_shots_of_the_same_view_dont_grow_the_map_past_the_cap(where):
    first = cloud_map.offer(where, ROOMS, shot("one", at(2.0, 1.5, 0.0), flat=True))
    size = len(the_map(where).xyz)
    again = [cloud_map.offer(where, ROOMS, shot(f"again-{n}", at(2.0, 1.5, 0.0), flat=True)) for n in range(3)]
    assert first.points == size and [told.points for told in again] == [0, 0, 0]
    held = the_map(where)
    assert len(held.xyz) == size and fullest(held.flat) <= cloud_map.CAP


def test_a_shot_offered_twice_gets_the_answer_it_got_and_changes_nothing(where):
    first = cloud_map.offer(where, ROOMS, shot("one", at(2.0, 1.5, 0.0)))
    revision = the_map(where).meta["rev"]
    assert cloud_map.offer(where, ROOMS, shot("one", at(2.0, 1.5, 0.0))) == first
    assert the_map(where).meta["rev"] == revision


def test_a_held_shot_placed_a_metre_off_is_refused_and_the_map_is_as_it_was(where):
    cloud_map.offer(where, ROOMS, shot("good", at(2.0, 1.5, 0.0)))
    before = the_map(where)
    told = cloud_map.offer(where, ROOMS, shot("off", at(2.0, 1.5, 0.0), at(2.0, 0.5, 0.0)))  # said a metre back
    # its far wall stands a metre inside the room, where the map's shot saw straight through to the wall
    assert (told.state, told.why, told.points) == ("refused", "apart", 0) and told.conflict > cloud_map.WHOLE
    after = the_map(where)
    assert np.array_equal(after.xyz, before.xyz)
    assert (cloud_map.answered(after.meta, "off") or {})["why"] == "apart"  # remembered, for its row to say


def test_a_loose_shot_a_metre_off_and_turned_is_brought_onto_the_wall_and_table_the_map_holds(where):
    cloud_map.offer(where, ROOMS, shot("good", at(2.0, 1.5, 0.0), things=table_top()))
    truth = at(1.2, 2.0, 25.0)  # another camera, looking at the same wall and table from the side
    said = truth.step(0.8, -0.6, math.radians(14.0))  # its spot is a metre out, and faces 14 degrees wrong
    told = cloud_map.offer(where, ROOMS, shot("loose", truth, said, held=False, things=table_top(), camera="porch"))
    assert told.state == "added" and told.points > 0
    assert (told.moved_m, told.turned_deg) == (pytest.approx(1.0, abs=0.1), pytest.approx(14.0, abs=1.5))
    assert told.stand is not None
    assert scenes.moved_by(scenes.At(*told.stand[:3]), truth) == (
        pytest.approx(0.0, abs=cloud_merge.CLOSE_M),
        pytest.approx(0.0, abs=1.5),
    )
    held = the_map(where)
    # one table: every point at table height lies on the table, and one wall: none stands inside the room
    inside = (held.xyz[:, 0] > 0.3) & (held.xyz[:, 0] < 3.7) & (held.xyz[:, 1] < 4.7)
    tops = held.xyz[(np.abs(held.xyz[:, 2] - 0.7) < 0.05) & inside]
    assert tops[:, 0].min() > 1.4 and tops[:, 0].max() < 2.6 and tops[:, 1].min() > 3.4 and tops[:, 1].max() < 4.3
    standing = held.xyz[(held.xyz[:, 2] > 1.0) & (held.xyz[:, 0] > 0.3) & (held.xyz[:, 0] < 3.7)]
    assert standing[:, 1].min() > 4.85
    assert cloud_map.by_camera(held)["porch"] == told.points


def test_a_tracked_shot_is_added_exactly_where_its_track_puts_it_or_refused_and_never_moved_to_agree(where):
    cloud_map.offer(where, ROOMS, shot("good", at(2.0, 1.5, 0.0), things=table_top()))
    truth = at(1.2, 2.0, 25.0)
    wrong = truth.step(0.8, -0.6, math.radians(14.0))

    def tracked(name: str, lies: scenes.At) -> cloud_map.Offer:
        offer = shot(name, truth, wrong, held=False, things=table_top(), camera="porch")
        return dataclasses.replace(offer, lie=cloud_map._lie(cloud_merge.Stand(lies.x, lies.y, lies.bearing), 1.0))

    # the track has it right: it goes in where the track says, whatever its stand says, and is not moved
    told = cloud_map.offer(where, ROOMS, tracked("right", truth))
    assert (told.state, told.moved_m, told.turned_deg) == ("added", 0.0, 0.0) and told.points > 0
    standing = the_map(where).xyz
    standing = standing[(standing[:, 2] > 1.0) & (standing[:, 0] > 0.3) & (standing[:, 0] < 3.7)]
    assert standing[:, 1].min() > 4.85  # one wall
    # the track has it a metre off: a loose shot would be brought onto the map's wall and table, this one is refused
    before = len(the_map(where).xyz)
    told = cloud_map.offer(where, ROOMS, tracked("drifted", wrong))
    assert (told.state, told.why, told.moved_m) == ("refused", "apart", 0.0)
    assert len(the_map(where).xyz) == before


def test_a_loose_shot_that_cant_be_made_to_agree_is_added_without_the_part_that_contradicts_the_map(where):
    cloud_map.offer(where, ROOMS, shot("good", at(2.0, 1.5, 0.0), things=table_top()))
    before = the_map(where)
    # its table stands 0.8 m along the wall from the map's: no turn, slide, and resize fits both wall and table
    elsewhere = table_top(((0.7, 3.5), (1.7, 4.2)))
    told = cloud_map.offer(where, ROOMS, shot("odd", at(2.0, 1.5, 0.0), held=False, things=elsewhere, camera="porch"))
    assert told.state == "added" and told.off_m > cloud_merge.CLOSE_M and (told.moved_m, told.replaced) == (0.0, 0)
    assert told.left_out > 500  # the table where the map's shot saw the floor
    after = the_map(where)
    assert (after.shot == 0).sum() == len(before.xyz)  # nothing of the map gave way to it
    inside = (after.xyz[:, 0] > 0.3) & (after.xyz[:, 0] < 3.7) & (after.xyz[:, 1] < 4.7)
    tops = after.xyz[(np.abs(after.xyz[:, 2] - 0.7) < 0.05) & inside]
    # one table, the map's: none of the second one beside it, but for a fringe as wide as one cell of the
    # coarse grid these made-up shots are looked at through
    assert tops[:, 0].min() > 1.2


def test_a_loose_shot_that_as_a_whole_stands_somewhere_else_is_refused(where):
    cloud_map.offer(where, ROOMS, shot("good", at(2.0, 1.5, 0.0)))
    before = len(the_map(where).xyz)
    # a spot two metres out and facing a third of the way round: farther than a shot is moved to agree
    told = cloud_map.offer(
        where, ROOMS, shot("lost", at(2.0, 1.5, 0.0), at(2.2, 3.4, 110.0), held=False, camera="porch")
    )
    assert (told.state, told.why, told.points) == ("refused", "apart", 0) and told.conflict > cloud_map.WHOLE
    assert len(the_map(where).xyz) == before


def test_a_shot_that_shares_nothing_with_the_map_is_added_where_it_stands(where):
    cloud_map.offer(where, ROOMS, shot("hall", at(2.0, 1.5, 0.0)))
    told = cloud_map.offer(where, ROOMS, shot("den", at(8.0, 1.0, 60.0), held=False, camera="porch"))  # another room
    assert (told.state, told.moved_m, told.turned_deg) == ("added", 0.0, 0.0) and told.points > 0


def test_a_better_placed_shot_takes_the_place_of_a_worse_one_where_it_looks_and_not_the_other_way_round(where):
    # The first shot has the cupboard 90 cm out from the wall (its depth was off); the plan explains less of it.
    worse = shot("worse", at(2.0, 1.0, 0.0), score=0.6, things=cupboard(4.1))
    better = shot("better", at(2.0, 1.0, 0.0), score=0.9, things=cupboard(4.6))
    cloud_map.offer(where, ROOMS, worse)
    told = cloud_map.offer(where, ROOMS, better)
    assert told.state == "added" and told.replaced > 0
    held = the_map(where)
    middle = (held.xyz[:, 0] > 1.1) & (held.xyz[:, 0] < 2.9)
    fronts = held.xyz[(held.xyz[:, 2] > 0.3) & (held.xyz[:, 1] > 3.8) & (held.xyz[:, 1] < 4.9) & middle]
    assert fronts[:, 1] == pytest.approx(4.6, abs=0.02)  # one cupboard, where the better shot has it
    size = len(held.xyz)
    # and a worse shot of the same view, offered after, changes nothing
    late = cloud_map.offer(where, ROOMS, shot("late", at(2.0, 1.0, 0.0), score=0.6, things=cupboard(4.1)))
    assert (late.state, late.why) == ("refused", "apart") and len(the_map(where).xyz) == size


def test_the_same_picture_placed_again_takes_the_place_of_its_earlier_self(where):
    first = shot("cloud-1@spot", at(2.0, 1.5, 0.0))
    cloud_map.offer(where, ROOMS, first)
    moved = shot("cloud-1@followed", at(2.0, 1.5, 0.0), at(2.0, 1.2, 0.0))  # placed 30 cm back this time
    again = replace(moved, cloud=first.cloud)
    told = cloud_map.offer(where, ROOMS, again)
    held = the_map(where)
    assert told.state == "added" and [row["id"] for row in held.meta["shots"]] == ["cloud-1@followed"]
    assert far_wall(held)[:, 1] == pytest.approx(4.7, abs=0.03)  # one wall, where it is now said to be


def test_a_map_is_kept_on_disk_and_a_plan_whose_walls_changed_drops_it(where):
    cloud_map.offer(where, ROOMS, shot("one", at(2.0, 1.5, 0.0)))
    cloud_map.offer(where, ROOMS, shot("two", at(2.0, 1.5, 60.0)))
    kept = the_map(where)  # as a service that has just started reads it
    assert kept is not None and [row["id"] for row in kept.meta["shots"]] == ["one", "two"]
    assert len(kept.xyz) == sum(row["points"] for row in kept.meta["shots"]) > 0
    assert sorted(path.name for path in where.iterdir()) == ["lock", "map.json", "points.npz"]
    cloud_map.offer(where, (ROOMS[0],), shot("three", at(2.0, 1.5, 0.0)))  # the den was taken off the plan
    after = the_map(where)
    assert after.meta["plan"] == cloud_align.plan_key((ROOMS[0],))
    assert [row["id"] for row in after.meta["shots"]] == ["three"]


def test_a_map_says_how_much_of_the_plans_floor_it_covers_and_that_grows_as_cameras_look_around(where):
    whole = 4.0 * 5.0 + (6.0 * 3.5 + 3.0 * 2.5)  # the hall, and the den's two parts
    cloud_map.offer(where, ROOMS, shot("hall", at(2.0, 0.5, 0.0)))
    first = the_map(where).meta
    assert first["plan_m2"] == pytest.approx(whole)
    assert 5.0 < first["floor_m2"] < 20.0  # some of the hall's 20 square metres, and none of the den
    cloud_map.offer(where, ROOMS, shot("again", at(2.0, 0.5, 0.0)))
    assert the_map(where).meta["floor_m2"] == first["floor_m2"]  # the same view again covers nothing more
    cloud_map.offer(where, ROOMS, shot("den", at(5.0, 1.0, 60.0), camera="porch"))
    both = the_map(where).meta
    assert first["floor_m2"] + 3.0 < both["floor_m2"] <= whole


def chair(x: float, y: float, seed: int = 9) -> np.ndarray:
    """The back of a chair standing in the hall, facing the camera: 60 cm wide, 90 cm tall."""
    rng = np.random.default_rng(seed)
    return np.c_[x + rng.random(1500) * 0.6, np.full(1500, y), rng.random(1500) * 0.9]


def test_a_chair_that_was_moved_is_in_the_map_where_it_is_now_and_not_where_it_was(where):
    cloud_map.offer(where, ROOMS, shot("before", at(2.0, 1.0, 0.0), things=chair(1.2, 3.0)))
    told = cloud_map.offer(where, ROOMS, shot("after", at(2.0, 1.0, 0.0), things=chair(2.4, 3.6), when=600.0))
    assert told.state == "added" and told.replaced > 300  # the chair where it was: the new shot looked through it
    held = the_map(where)
    upright = held.xyz[(held.xyz[:, 2] > 0.2) & (held.xyz[:, 1] > 2.5) & (held.xyz[:, 1] < 4.0)]
    middle = upright[(upright[:, 0] > 0.5) & (upright[:, 0] < 3.5)]
    assert len(middle) > 500 and middle[:, 1] == pytest.approx(3.6, abs=0.01)
    assert middle[:, 0].min() >= 2.4 and middle[:, 0].max() <= 3.0


def test_a_shot_the_plan_explains_clearly_worse_moves_nothing_of_the_map_and_leaves_out_what_contradicts_it(where):
    cloud_map.offer(where, ROOMS, shot("good", at(2.0, 1.0, 0.0), score=0.9))
    before = the_map(where)
    # a poorer shot from the side, with a chair the good shot saw straight through where it stands
    told = cloud_map.offer(where, ROOMS, shot("poor", at(3.0, 1.5, -30.0), score=0.5, things=chair(1.7, 3.0)))
    assert (told.state, told.replaced) == ("added", 0) and told.left_out > 1000
    after = the_map(where)
    assert (after.shot == 0).sum() == len(before.xyz)  # the good shot's points are all still there
    ghost = after.xyz[(after.xyz[:, 2] > 0.2) & (np.abs(after.xyz[:, 1] - 3.0) < 0.02) & (after.xyz[:, 0] > 1.7)]
    assert len(ghost[ghost[:, 0] < 2.3]) == 0  # and the chair it alone saw is not drawn


def test_a_map_kept_before_its_floor_was_counted_is_counted_once_and_drawn_as_it_was(where):
    cloud_map.offer(where, ROOMS, shot("hall", at(2.0, 0.5, 0.0)))
    was = the_map(where).meta
    older = {k: v for k, v in was.items() if k not in ("floor_m2", "plan_m2")}
    (where / "map.json").write_text(json.dumps(older))
    assert cloud_map.count_floor(where, ROOMS)
    now = the_map(where).meta
    assert (now["floor_m2"], now["plan_m2"], now["rev"]) == (was["floor_m2"], was["plan_m2"], was["rev"])
    assert not cloud_map.count_floor(where, ROOMS)  # once
    assert not cloud_map.count_floor(where, (ROOMS[0],))  # and not for a plan the map isn't of
