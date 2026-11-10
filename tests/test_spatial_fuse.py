"""Tier 1 refinement: levelling, registration against a known offset, and fused revisions of a place."""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from liberational import segment_log
from liberational.memory import places
from liberational.spatial import cameras, floorplan, footprints, fuse, ply, poses

T = 1_800_000_000.0


def room(seed=0, n=30_000):
    """An L-shaped room (6 x 4 m less a 2 x 2 m corner) 2.6 m high, with a table in one corner, levelled: floor z = 0.
    The notch and the table make it asymmetric, so only one yaw fits."""
    rng = np.random.default_rng(seed)
    outline = [(0, 0), (6, 0), (6, 2), (4, 2), (4, 4), (0, 4), (0, 0)]
    segs = list(itertools.pairwise(outline))
    lengths = np.array([math.dist(a, b) for a, b in segs])
    pick = rng.choice(len(segs), n, p=lengths / lengths.sum())
    u = rng.uniform(0, 1, n)
    wall = np.array([np.add(segs[k][0], u[i] * np.subtract(segs[k][1], segs[k][0])) for i, k in enumerate(pick)])
    walls = np.c_[wall, rng.uniform(0, 2.6, n)]
    fl = rng.uniform([0, 0], [6, 4], (n, 2))
    fl = fl[~((fl[:, 0] > 4) & (fl[:, 1] > 2))]
    floor = np.c_[fl, np.zeros(len(fl))]
    table = np.c_[rng.uniform(0.3, 1.5, n // 6), rng.uniform(3.0, 3.7, n // 6), np.full(n // 6, 0.75)]
    pts = np.r_[walls, floor, table]
    return pts + rng.normal(0, 0.005, pts.shape)


def similarity(yaw_deg, shift, scale=1.0):
    a = math.radians(yaw_deg)
    out = np.eye(4)
    out[:3, :3] = scale * np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    out[:3, 3] = shift
    return out


def test_voxelize_keeps_one_point_per_cell_at_its_mean_with_its_colour():
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.5, 0.5, 0.5]])
    cols = np.array([[0, 0, 0], [100, 200, 50], [9, 9, 9]], np.uint8)
    out, c = fuse.voxelize(pts, cols, 0.1)
    assert len(out) == 2
    i = int(np.argmin(out[:, 0]))
    assert out[i] == pytest.approx([0.015, 0.015, 0.015])
    assert c is not None and c[i].tolist() == [50, 100, 25]


def test_levelling_puts_a_tilted_floor_at_zero_with_up_as_z():
    n = np.array([0.3, -0.9, 0.2])
    n /= np.linalg.norm(n)
    d = 1.7  # height = n . p + d
    rng = np.random.default_rng(0)
    a = np.cross(n, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    on_floor = (rng.uniform(-2, 2, (200, 1)) * a + rng.uniform(-2, 2, (200, 1)) * b) - d * n
    above = on_floor + 1.0 * n
    level = fuse.leveller((*n, d))
    assert fuse.apply(level, on_floor)[:, 2] == pytest.approx(np.zeros(200), abs=1e-9)
    assert fuse.apply(level, above)[:, 2] == pytest.approx(np.ones(200), abs=1e-9)


def test_a_cloud_with_a_known_offset_registers_back_onto_the_map():
    target = room(0)
    truth = similarity(37, [1.2, -0.7, 0.0], 1.03)  # source to map
    part = room(1)
    part = part[part[:, 0] < 4.5]  # the new visit saw most of the room, not all
    source = fuse.apply(np.linalg.inv(truth), part)
    reg = fuse.register(source, target)
    assert reg.ok and reg.fitness > 0.8
    err = np.linalg.norm(fuse.apply(reg.transform, source) - part, axis=1)
    assert float(np.median(err)) < 0.02


def test_a_whole_room_registers_onto_a_map_of_part_of_it():
    target = room(0)
    target = target[target[:, 0] < 2.5]  # the first map saw only the room's west end
    truth = similarity(120, [-0.4, 2.2, 0.0], 1.05)
    whole = room(1)
    reg = fuse.register(fuse.apply(np.linalg.inv(truth), whole), target)
    assert reg.ok, (reg.fitness, reg.inliers)
    err = np.linalg.norm(fuse.apply(reg.transform, fuse.apply(np.linalg.inv(truth), whole)) - whole, axis=1)
    assert float(np.median(err)) < 0.02


def as_fused(points):
    return fuse.Fused(points, None, np.ones(len(points), np.int64), np.zeros(len(points), np.int64))


def person(seed=4):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(2.0, 2.4, 3000), rng.uniform(1.5, 1.8, 3000), rng.uniform(0.0, 1.7, 3000)]


def above_floor_near(points, lo, hi):
    inside = (points[:, :2] >= lo).all(axis=1) & (points[:, :2] <= hi).all(axis=1)
    return int((inside & (points[:, 2] > 0.3)).sum())


def test_something_that_comes_and_goes_fades_and_rooms_out_of_view_stay():
    base = fuse.voxelize(room(0), None, fuse.VOXEL)[0]
    fused = fuse.merge(as_fused(base), np.r_[room(1), person()], None)
    assert above_floor_near(fused.points, (2.05, 1.55), (2.35, 1.75)) > 0, "seen once, it's in the map"
    for seed in (3, 5):  # the daily snapshot sees the west half, now without the person
        part = room(seed)
        fused = fuse.merge(fused, part[part[:, 0] < 3.0], None)
    assert above_floor_near(fused.points, (2.05, 1.55), (2.35, 1.75)) == 0, "missed twice in view, it faded"
    assert above_floor_near(fused.points, (4.5, 0.2), (6.0, 1.8)) > 100, "the east end, out of view, stays"
    assert fused.dropped > 0


JITTER = 0.02  # metres per axis: under one VOXEL, so a re-observed point lands in its voxel or a neighbour


def jitter(n):
    return hnp.arrays(np.float64, (n, 3), elements=st.floats(-JITTER, JITTER))


def own(fused, pts):
    """Which of `fused`'s voxels are `pts`'s own (a re-observation's jittered points can add voxels next to them)."""
    return np.isin(fuse._cell(fuse._keys(fused.points, fuse.VOXEL)), fuse._cell(fuse._keys(pts, fuse.VOXEL)))


@st.composite
def surface_and_jitter(draw):
    """A map of up to 150 voxels anywhere in a 4 m box, and two per-voxel jitters of up to JITTER on each axis."""
    raw = draw(hnp.arrays(np.float64, st.tuples(st.integers(1, 150), st.just(3)), elements=st.floats(-2.0, 2.0)))
    pts = fuse.voxelize(raw, None, fuse.VOXEL)[0]
    return pts, draw(jitter(len(pts))), draw(jitter(len(pts)))


@settings(max_examples=60, deadline=None)
@given(surface_and_jitter())
# A point on a cell's edge, averaged back a hair below it, once moved to the cell next door.
@example((np.array([[1.65, 1.65, 1.65]]), np.zeros((1, 3)), np.zeros((1, 3))))
def test_a_surface_seen_again_with_a_jitter_under_a_voxel_is_never_missed(case):
    """MASt3R's noise puts a surface that's still there in a neighbouring voxel; that isn't a miss."""
    pts, first, second = case
    fused = fuse.merge(as_fused(pts), pts + first, None)
    fused = fuse.merge(fused, pts + second, None)
    assert fused.dropped == 0
    # Only the map's own voxels: a jittered point that landed in a new voxel may be two cells from the next visit's.
    mine = own(fused, pts)
    assert mine.sum() == len(pts)
    assert not fused.missed[mine].any()


_GRID = np.mgrid[0:0.5:0.01, 0:0.5:0.01].reshape(2, -1).T
FLOOR = fuse.voxelize(np.c_[_GRID, np.zeros(len(_GRID))], None, fuse.VOXEL)[0]  # a 0.5 m floor patch at z = 0


@settings(max_examples=40, deadline=None)
@given(st.floats(0.05, 0.36), st.floats(0.05, 0.36), st.floats(0.1, 1.5), jitter(len(FLOOR)), jitter(len(FLOOR)))
def test_an_object_gone_from_a_jittered_floor_fades_on_its_second_miss(x, y, z, first, second):
    """An object at least two voxels above the floor, gone from two visits that see the floor under it with a jitter,
    is missed once (and kept), then missed twice and dropped; the floor stays."""
    cube = np.mgrid[0:0.09:0.03, 0:0.09:0.03, 0:0.09:0.03].reshape(3, -1).T + (x, y, z) + fuse.VOXEL / 2
    thing = fuse.voxelize(cube, None, fuse.VOXEL)[0]
    fused = fuse.merge(as_fused(np.r_[FLOOR, thing]), FLOOR + first, None)
    assert fused.dropped == 0
    assert (fused.points[:, 2] > 0.06).sum() == len(thing), "missed once, it's still there"
    fused = fuse.merge(fused, FLOOR + second, None)
    assert fused.dropped == len(thing)
    assert not (fused.points[:, 2] > 0.06).any(), "missed twice, it faded"
    floor = own(fused, FLOOR)
    assert floor.sum() == len(FLOOR) and not fused.missed[floor].any(), "the jittered floor was never missed"


def test_an_unrelated_cloud_doesnt_register():
    rng = np.random.default_rng(3)
    blob = np.c_[rng.uniform(0, 1.5, (20_000, 2)), rng.uniform(0, 2.6, 20_000)]
    assert not fuse.register(blob, room(0)).ok


def mast3r_frame(points):
    """Levelled points as a MASt3R cloud has them (y down) with its floor plane, height = n . p + d."""
    return np.c_[points[:, 0], -points[:, 2], points[:, 1]], (0.0, -1.0, 0.0, 0.0)


def raw_revision(state, place, points, now, cameras=None):
    pts, floor = mast3r_frame(points)
    return places.revise(
        state,
        place,
        producer="walk",
        map_bytes=ply.dumps(pts, np.full((len(pts), 3), 120, np.uint8)),
        map_format="ply",
        floor=floor,
        cameras=cameras,
        now=now,
    )


def pose_at(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return places.CameraPose(tuple(float(v) for v in m.ravel()), 300.0, 640, 360, T)


def test_the_first_revision_becomes_the_levelled_map_and_a_second_is_fused_into_it(tmp_path):
    first = raw_revision(tmp_path, None, room(0), T, cameras={"s2": pose_at(1.0, -2.0, 1.0)})
    fused = fuse.refine(tmp_path, first.place, now=T + 1)
    assert fused is not None and fused.producer == fuse.PRODUCER
    one, _, _ = ply.read(places.root(tmp_path) / fused.place / str(fused.map))
    assert float(np.percentile(one[:, 2], 1)) == pytest.approx(0.0, abs=0.03), "the floor is at z = 0"
    assert fused.floor is not None and fused.floor[2] == pytest.approx(1.0, abs=0.02)
    assert fused.cameras["s2"].pose[11] == pytest.approx(2.0), "the camera, 2 m above the floor, is moved with it"
    assert fused.frame == fused.id, "a first map stands in a frame of its own"
    assert fuse.refine(tmp_path, first.place) is None, "tier 1's own revision isn't fused again"

    part = room(1)
    part = part[part[:, 0] < 4.5]
    moved = fuse.apply(np.linalg.inv(similarity(-50, [0.8, 1.5, 0.0], 0.97)), part)
    raw_revision(tmp_path, first.place, moved, T + 2, cameras={"s9": pose_at(0.5, -1.0, 0.5)})
    again = fuse.refine(tmp_path, first.place, now=T + 3)
    assert again is not None and again.sources[-1] == fused.id
    two, _, counts = ply.read(places.root(tmp_path) / again.place / str(again.map))
    assert counts["seen"].max() == 2, "the overlap was seen by both visits"
    rot = np.asarray(again.cameras["s9"].pose).reshape(4, 4)[:3, :3]
    assert rot @ rot.T == pytest.approx(np.eye(3), abs=1e-6), "a moved camera's axes stay a rotation"
    lo, hi = np.percentile(two, 0.5, axis=0), np.percentile(two, 99.5, axis=0)
    assert lo[:2] == pytest.approx([0, 0], abs=0.06) and hi[:2] == pytest.approx([6, 4], abs=0.06), (
        "the second visit lands on the room, not beside it"
    )
    assert again.cameras["s2"] == fused.cameras["s2"], "the map's frame and its registrations are kept"
    assert again.frame == fused.frame, "fused into the same map, so a splat of the first still fits"


def test_a_visit_that_doesnt_register_changes_nothing(tmp_path):
    first = raw_revision(tmp_path, None, room(0), T)
    fuse.refine(tmp_path, first.place, now=T + 1)
    rng = np.random.default_rng(5)
    raw_revision(tmp_path, first.place, np.c_[rng.uniform(0, 1.5, (20_000, 2)), rng.uniform(0, 2.6, 20_000)], T + 2)
    before = len(places.revisions(tmp_path, first.place))
    assert fuse.refine(tmp_path, first.place, now=T + 3) is None
    assert len(places.revisions(tmp_path, first.place)) == before


def test_a_revision_without_a_map_has_nothing_to_fuse(tmp_path):
    rev = places.revise(tmp_path, None, producer="named", now=T)
    assert fuse.refine(tmp_path, rev.place) is None


def test_ply_read_gives_points_and_colours_back(tmp_path):
    pts = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    cols = np.array([[1, 2, 3], [250, 251, 252]], np.uint8)
    path = tmp_path / "c.ply"
    path.write_bytes(ply.dumps(pts, cols, {"seen": np.array([3, 7]), "missed": np.array([0, 2])}))
    got, c, counts = ply.read(path)
    assert got.tolist() == pts.tolist() and c is not None and c.tolist() == cols.tolist()
    assert {k: v.tolist() for k, v in counts.items()} == {"seen": [3, 7], "missed": [0, 2]}


def test_the_fused_plan_gets_where_people_walked_through_the_fused_cameras(tmp_path, monkeypatch):
    from liberational.spatial import cameras, footprints

    first = raw_revision(tmp_path, None, room(0), T, cameras={"s2": pose_at(1.0, -2.0, 1.0)})
    seen = [footprints.Cells("s2", np.array([[0.5, 0.9]]), np.array([3]))]  # three footprints in one cell
    monkeypatch.setattr(fuse.footprints, "walked", lambda state, cams, now: seen if cams == ["s2"] else [])
    got = []
    real = cameras.layout

    def layout(points, up, title, walked=None, floor=None):
        got.append(walked((0.0, 0.0, 1.0, 0.0)) if walked is not None else None)
        return real(points, up, title, walked, floor)

    monkeypatch.setattr(fuse.cameras, "layout", layout)
    fuse.refine(tmp_path, first.place, now=T + 1)
    (walked,) = got
    assert walked is not None
    feet, weights = walked
    assert len(feet) == 1 and weights.tolist() == [3]
    # The camera sits 2 m up, level, looking along the room; the bottom-middle ray drops 0.4 * 360 / 300 = 0.48 m a
    # metre forward, so it meets the heels' plane (FOOT_M = 0.07 m up) 1.93 / 0.48 m ahead of the camera.
    assert feet[0][2] == pytest.approx(0.0, abs=1e-6)
    assert math.hypot(feet[0][0] - 1.0, feet[0][1] - 1.0) == pytest.approx(1.93 / 0.48, rel=1e-3)


def tilt_deg(transform):
    """How far a 4x4 similarity tips +z (degrees)."""
    up = transform[:3, :3] @ [0.0, 0.0, 1.0]  # atan2, not acos: acos of 1 - 1e-16 reads as a 1e-6 degree tilt
    return math.degrees(math.atan2(math.hypot(up[0], up[1]), up[2]))


def tipped(degrees):
    """A rotation tipping the map `degrees` about x, as a fused map that drifted would be."""
    a = math.radians(degrees)
    out = np.eye(4)
    out[1:3, 1:3] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return out


def test_the_least_squares_step_recovers_a_yaw_shift_and_scale_exactly():
    src = room(0, n=5000)
    truth = similarity(-63, [0.4, -1.1, 0.05], 0.96)
    assert fuse.umeyama(src, fuse.apply(truth, src)) == pytest.approx(truth, abs=1e-9)


def test_a_level_cloud_isnt_tilted_to_match_a_tilted_map():
    # A correctly levelled snapshot must not be tilted by ICP to match a map that had drifted.
    target = fuse.apply(tipped(1.5), room(0))
    source = fuse.apply(np.linalg.inv(similarity(25, [0.6, 0.3, 0.0])), room(1))
    reg = fuse.register(source, target)
    assert reg.ok
    assert tilt_deg(reg.transform) < 1e-6


def test_repeated_fusions_of_partial_views_keep_the_map_level():
    rng = np.random.default_rng(7)
    fused = as_fused(fuse.voxelize(room(0, n=6000), None, fuse.VOXEL)[0])
    for seed in range(1, 5):  # four days of snapshots, each of part of the room, each in its own frame
        part = room(seed, n=6000)
        part = part[part[:, 0] < rng.uniform(3.5, 5.5)]
        offset = similarity(float(rng.uniform(-40, 40)), [*rng.uniform(-1, 1, 2), 0.0], float(rng.uniform(0.95, 1.05)))
        reg = fuse.register(fuse.apply(np.linalg.inv(offset), part), fused.points)
        assert reg.ok
        assert tilt_deg(reg.transform) < 1e-6
        fused = fuse.merge(fused, fuse.apply(reg.transform, fuse.apply(np.linalg.inv(offset), part)), None)
    normal, _ = floorplan.floor_plane(fused.points, (0.0, 0.0, 1.0), 0.03, np.random.default_rng(0))
    assert math.degrees(math.acos(min(1.0, float(normal[2])))) < 0.2


def test_a_fused_map_that_drifted_is_turned_level_with_its_cameras():
    # Place 1's fused map had drifted 1.59 degrees before registration was made yaw-only.
    tip = tipped(1.59)
    camera = places.CameraPose(
        tuple(float(x) for x in (tip @ similarity(0, [2.0, 1.0, 1.2])).ravel()), 400.0, 640, 360, T
    )
    drifted = as_fused(fuse.apply(tip, room(0, n=8000)))
    level, cams = fuse.relevel(drifted, {"cam0": camera})
    normal, d = floorplan.floor_plane(level.points, (0.0, 0.0, 1.0), 0.03, np.random.default_rng(0))
    assert math.degrees(math.acos(min(1.0, float(normal[2])))) < 0.05
    assert d == pytest.approx(0.0, abs=0.01)  # the floor at z = 0
    pose = np.asarray(cams["cam0"].pose).reshape(4, 4)
    assert pose[2, 3] == pytest.approx(1.2, abs=0.02)  # the camera's height above the floor is kept
    assert tilt_deg(pose) < 0.05  # and it hangs level again, as it did before the drift


def test_a_level_fused_map_is_left_as_it_is():
    fused = as_fused(room(0, n=8000))
    camera = places.CameraPose(tuple(float(x) for x in similarity(30, [1.0, 1.0, 1.5]).ravel()), 400.0, 640, 360, T)
    level, cams = fuse.relevel(fused, {"cam0": camera})
    assert level is fused
    assert cams == {"cam0": camera}


def test_a_plane_with_the_map_under_it_is_not_taken_for_a_drifted_floor():
    # One camera's view of a bedroom holds more dresser top than floor. Here the floor is a small patch at
    # z = 0 and a table top three times its size is tipped 10 degrees, 0.8 m up: a floor fit takes the table.
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 1, (2000, 2)), np.zeros(2000)]
    xy = rng.uniform(0, 2, (6000, 2))
    table = np.c_[xy, 0.8 + np.tan(np.radians(10)) * (xy[:, 0] - 1)]
    fused = as_fused(np.r_[floor, table])
    normal, _ = floorplan.floor_plane(fused.points, (0.0, 0.0, 1.0), 0.03, np.random.default_rng(0))
    assert math.degrees(math.acos(float(normal[2]))) > 5  # the fit did take the table
    camera = places.CameraPose(tuple(float(x) for x in similarity(0, [0.5, -1.0, 1.8]).ravel()), 400.0, 640, 360, T)
    level, cams = fuse.relevel(fused, {"cam0": camera})
    assert level is fused
    assert cams == {"cam0": camera}


def test_a_fused_map_keeps_its_levelled_floor_when_furniture_tops_outnumber_it(tmp_path):
    # Place 2's bedroom camera sees more dresser top than floor. The map is levelled on its floor (z = 0), so
    # its stored floor is that one, not the larger flat top 0.8 m up that a fit to the cloud would take.
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 1, (2000, 2)), np.zeros(2000)]
    top = np.c_[rng.uniform(0, 2, (6000, 2)), np.full(6000, 0.8)]
    first = raw_revision(tmp_path, None, np.r_[floor, top], T, cameras={"bed": pose_at(0.5, -1.5, 1.0)})
    fused = fuse.refine(tmp_path, first.place, now=T + 1)
    assert fused is not None and fused.floor is not None
    camera = np.asarray(fused.cameras["bed"].pose).reshape(4, 4)[:3, 3]
    height = float(np.dot(fused.floor[:3], camera) + fused.floor[3])
    assert height == pytest.approx(1.5, abs=0.02), "the camera stands 1.5 m above the floor, not 0.7 m"


# Calibration from the owner's camera heights and from people seen by two cameras at once.


def test_measured_camera_heights_put_the_map_at_metric_scale(tmp_path, mounts):
    # MASt3R made the room at 0.6 of its size, and the camera 2 m up at 1.2; the owner measured 2 m
    small = room(0) * 0.6
    first = raw_revision(tmp_path, None, small, T, cameras={"s2": pose_at(0.6, -1.2, 0.6), "s3": pose_at(1, -1, 1)})
    mounts({"s2": {"height_m": 2.0}})
    fused = fuse.refine(tmp_path, first.place, now=T + 1)
    assert fused is not None
    pts, _, _ = ply.read(places.root(tmp_path) / fused.place / str(fused.map))
    hi = np.percentile(pts, 99.5, axis=0)
    assert hi[:2] == pytest.approx([6, 4], abs=0.06), "the room is back at 6 x 4 m"
    assert fused.cameras["s2"].pose[11] == pytest.approx(2.0)
    assert fused.cameras["s3"].pose[11] == pytest.approx(1 / 0.6), "a camera with no measured height is scaled too"


def test_without_measured_heights_the_map_keeps_its_scale(tmp_path):
    first = raw_revision(tmp_path, None, room(0) * 0.6, T, cameras={"s2": pose_at(0.6, -1.2, 0.6)})
    fused = fuse.refine(tmp_path, first.place, now=T + 1)
    assert fused is not None and fused.cameras["s2"].pose[11] == pytest.approx(1.2)


def test_a_map_fused_before_heights_were_known_is_rescaled_and_still_registers(tmp_path, mounts):
    cams = {"s2": pose_at(0.6, -1.2, 0.6)}
    first = raw_revision(tmp_path, None, room(0) * 0.6, T, cameras=cams)
    fused = fuse.refine(tmp_path, first.place, now=T + 1)  # fused at MASt3R's scale
    assert fused is not None
    part = room(1)
    part = part[part[:, 0] < 4.5] * 0.6
    raw_revision(tmp_path, first.place, part, T + 2, cameras={"s2": pose_at(0.6, -1.2, 0.6)})
    mounts({"s2": {"height_m": 2.0}})
    again = fuse.refine(tmp_path, first.place, now=T + 3)
    assert again is not None
    assert again.sources[-1] == fused.id, "the new map registered onto the rescaled one"
    assert again.frame == again.id != fused.frame, "a rescaled map is a new frame: an old splat wouldn't fit it"
    pts, _, _ = ply.read(places.root(tmp_path) / again.place / str(again.map))
    assert np.percentile(pts, 99.5, axis=0)[:2] == pytest.approx([6, 4], abs=0.06)
    assert again.cameras["s2"].pose[11] == pytest.approx(2.0, abs=0.02)


def looking(x, y, height, heading_deg, pitch_deg=40.0, focal=300.0, size=(640, 360)):
    """A levelled camera at (x, y, height) facing `heading_deg` and `pitch_deg` down: x right, y down, z forward."""
    h, p = math.radians(heading_deg), math.radians(pitch_deg)
    forward = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), -math.sin(p)])
    right = np.array([math.sin(h), -math.cos(h), 0.0])
    down = np.cross(forward, right)
    m = np.eye(4)
    m[:3, :3] = np.c_[right, down, forward]
    m[:3, 3] = (x, y, height)
    return places.CameraPose(tuple(float(v) for v in m.ravel()), focal, size[0], size[1], T)


def pixel(cam, point):
    m = np.asarray(cam.pose).reshape(4, 4)
    c = m[:3, :3].T @ (np.asarray(point, dtype=float) - m[:3, 3])
    return c[0] / c[2] * cam.focal / cam.width + 0.5, c[1] / c[2] * cam.focal / cam.height + 0.5


def walk(cams, spots, people=1):
    """Sightings of `people` at each spot (one second apart) by every camera; extra people stand 2 m east."""
    out = []
    for t, (x, y) in enumerate(spots):
        for k in range(people):
            feet, head = (x + 2 * k, y, footprints.FOOT_M), (x + 2 * k, y, fuse.STATURE_M)  # heels are above the floor
            for name, cam in cams.items():
                u, v = pixel(cam, feet)
                out.append(footprints.Sighting(name, T + t + 0.1 * k, k, u, v, pixel(cam, head), upright=True))
    return out


TRUE = {"a": looking(0, 0, 1.5, 0), "b": looking(6, 1, 2.3, 180)}


def test_people_each_camera_sees_alone_dont_move_the_cameras(tmp_path, monkeypatch):
    # In most seconds where each of two cameras saw one person, they were two people, each hidden from the
    # other camera. Here camera a sees one person and camera b another a metre east, every second for five minutes.
    spots = np.random.default_rng(1).uniform([2, -1], [4, 2], (300, 2))
    seen = walk({"a": TRUE["a"]}, spots) + walk({"b": TRUE["b"]}, spots + [1.0, 0.0])
    monkeypatch.setattr(fuse, "_sightings", lambda state, cams, now: seen)
    # the cameras as a MASt3R revision has them: in its y-down frame, the inverse of tier 1's levelling
    unlevel = np.linalg.inv(fuse.leveller(mast3r_frame(np.zeros((1, 3)))[1]))
    raw = fuse._moved(TRUE, unlevel)
    first = raw_revision(tmp_path, None, room(0), T, cameras=raw)
    fused = fuse.refine(tmp_path, first.place, now=T + 1)
    assert fused is not None
    assert fused.cameras["a"].pose == pytest.approx(TRUE["a"].pose, abs=1e-9)
    assert fused.cameras["b"].pose == pytest.approx(TRUE["b"].pose, abs=1e-9)


def test_people_measure_their_height_through_a_right_registration():
    rng = np.random.default_rng(4)
    spots = rng.uniform([2, -1], [4, 2], (300, 2))
    # Each second's spot differs, so every sighting after the first is a step
    got = fuse.people(walk({"a": TRUE["a"]}, spots), {"a": TRUE["a"]})
    assert got is not None
    height, measured = got
    assert height == pytest.approx(fuse.STATURE_M, abs=1e-6) and measured > fuse.MIN_WALKERS


def test_people_not_upright_are_not_measured():
    spots = np.random.default_rng(6).uniform([2, -1], [4, 2], (300, 2))
    seen = [replace(s, upright=False) for s in walk({"a": TRUE["a"]}, spots)]
    assert fuse.people(seen, {"a": TRUE["a"]}) is None


def test_people_too_few_to_measure_give_no_height():
    spots = np.random.default_rng(5).uniform([2, -1], [4, 2], (fuse.MIN_WALKERS // 2, 2))
    assert fuse.people(walk({"a": TRUE["a"]}, spots), {"a": TRUE["a"]}) is None


def test_a_new_map_its_cameras_would_scale_far_from_one_isnt_fused(tmp_path, mounts):
    # A wrong floor puts a camera at a tenth of its height: scaling the map eight times over is refused
    first = raw_revision(tmp_path, None, room(0), T, cameras={"s2": pose_at(1.0, -2.0, 1.0)})
    mounts({"s2": {"height_m": 16.0}})
    assert fuse.refine(tmp_path, first.place, now=T + 1) is None
    assert [r.id for r in places.revisions(tmp_path, first.place)] == [first.id]


def test_a_cameras_background_rides_through_fusion_and_still_reads_in_metres(tmp_path, mounts):
    # Positions read the fused revision. MASt3R made the room at 0.6 of its size; the background is a fraction of
    # the camera's height, so the couch cell (0.45 of the camera's 2 m) still reads 0.9 m up at metric scale.
    bg = places.Background(2, 1, (0.0, 0.45))
    cam = replace(pose_at(0.6, -1.2, 0.6), background=bg)
    first = raw_revision(tmp_path, None, room(0) * 0.6, T, cameras={"s2": cam})
    mounts({"s2": {"height_m": 2.0}})
    fused = fuse.refine(tmp_path, first.place, now=T + 1)
    assert fused is not None and fused.cameras["s2"].background == bg
    # The camera looks level along the room; at v = 0.9 its ray drops 0.48 m a metre: the couch hides 1.73 m of floor
    assert footprints.at_edge(fused.cameras["s2"], fused.floor, 0.75, 0.9) is True
    assert footprints.at_edge(fused.cameras["s2"], fused.floor, 0.25, 0.9) is False


def fused_place(tmp_path, points, now, cams):
    """A place whose fused map is `points` (levelled), with `cams` registered in it."""
    first = raw_revision(tmp_path, None, points, now, cameras=cams)
    fused = fuse.refine(tmp_path, first.place, now=now + 1)
    assert fused is not None
    return fused


def test_a_joined_place_brings_its_map_cameras_and_signals_into_the_other_and_is_retired(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    away = fused_place(tmp_path, room(1), T + 10, {"den": looking(2.0, 1.0, 1.5, 90.0)})
    move = similarity(90, [10.0, 2.0, 0.0])  # the den's room stands 10 m along x from the hall's, turned 90 degrees
    joined = fuse.join(tmp_path, home.place, {away.place: move}, now=T + 20)
    assert joined.producer == fuse.PRODUCER and joined.frame == home.frame, (
        "the home's frame holds: nothing in it moved"
    )
    assert set(joined.cameras) == {"hall", "den"}
    assert joined.cameras["hall"] == home.cameras["hall"]
    x, y, z = np.asarray(away.cameras["den"].pose).reshape(4, 4)[:3, 3]
    den = np.asarray(joined.cameras["den"].pose).reshape(4, 4)
    assert den[:3, 3] == pytest.approx([10.0 - y, 2.0 + x, z]), "the den's camera moved with its room: turned, shifted"
    pts, _, counts = ply.read(places.root(tmp_path) / joined.place / str(joined.map))
    assert pts[:, 0].max() > 9.0 and "seen" in counts, "the den's map is in the home's, with the fused counts"
    assert places.place_ids(tmp_path) == [home.place], "the den's place is retired"
    assert places.merged_into(tmp_path, away.place) == home.place
    assert places.camera_places(tmp_path) == {"hall": home.place, "den": home.place}


def test_a_joined_place_takes_the_name_the_owner_gave_the_merged_one_only_when_it_has_none(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    away = fused_place(tmp_path, room(1), T + 10, {"den": looking(2.0, 1.0, 1.5, 90.0)})
    places.name(tmp_path, away.place, "Cottage", now=T + 11)
    fuse.join(tmp_path, home.place, {away.place: similarity(0, [10.0, 0.0, 0.0])}, now=T + 20)
    got = places.get(tmp_path, home.place)
    assert got is not None and got.name == "Cottage" and got.named


def test_a_place_with_the_owners_name_keeps_it_when_another_is_joined(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    away = fused_place(tmp_path, room(1), T + 10, {"den": looking(2.0, 1.0, 1.5, 90.0)})
    places.name(tmp_path, home.place, "Home", now=T + 11)
    places.name(tmp_path, away.place, "Cottage", now=T + 12)
    fuse.join(tmp_path, home.place, {away.place: similarity(0, [10.0, 0.0, 0.0])}, now=T + 20)
    got = places.get(tmp_path, home.place)
    assert got is not None and got.name == "Home"


def test_joining_a_place_without_a_fused_map_is_refused(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    raw = raw_revision(tmp_path, None, room(1), T + 10)  # never fused
    with pytest.raises(KeyError, match="no fused map"):
        fuse.join(tmp_path, home.place, {raw.place: np.eye(4)}, now=T + 20)
    assert set(places.place_ids(tmp_path)) == {home.place, raw.place}


def test_cameras_registered_in_a_map_give_the_similarity_that_puts_a_new_run_on_it():
    # Two cameras as the map has them, and as a new run has them: turned -35 degrees, shifted, at 0.8 of the scale.
    old = {"a": looking(1.0, 1.0, 1.6, 0.0), "b": looking(4.0, 2.5, 2.2, 200.0)}
    truth = similarity(35, [2.0, -1.0, 0.0], 1.25)  # the new run's frame to the map's
    new = fuse._moved(old, np.linalg.inv(truth))
    found = fuse.by_cameras({**new, "c": looking(0.0, 0.0, 1.0, 0.0)}, old)  # c isn't in the map: it doesn't count
    assert found == pytest.approx(truth, abs=1e-9)
    assert fuse.by_cameras(new, {"c": old["a"]}) is None, "no camera in both"
    assert fuse.by_cameras(fuse._moved(old, np.diag([0.5, 0.5, 0.5, 1.0])), old) is None, "a scale of 2 is refused"


# A merged place's owner plan is carried into the joined plan


def plan_xy(rev, x, y):
    """Where map point (x, y) on the floor of `rev`'s map lands on its plan."""
    tr = np.asarray(rev.plan_transform).reshape(3, 4)
    return tuple(float(v) for v in tr[:2] @ np.array([x, y, 0.0, 1.0]))


def drawn(rev, name, lo, hi, ident="r1"):
    """A room line the owner drew on `rev`'s plan over the map's floor from `lo` to `hi`."""
    corners = [plan_xy(rev, x, y) for x, y in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]
    return f'room {ident} "{name}" other poly ' + " ".join(f"{x:.4f},{y:.4f}" for x, y in corners)


def room_on_plan(state, place, x, y):
    """The room the holding plan of `place` has at map point (x, y) of its newest map."""
    held = places.get(state, place)
    assert held is not None and held.plan is not None
    frame = next(r for r in places.revisions(state, place) if r.id == held.plan_revision)
    found = floorplan.room_at(floorplan.rooms_from_dsl(held.plan), *plan_xy(frame, x, y))
    return None if found is None else found[1]


def test_a_merged_places_owner_plan_is_carried_into_the_joined_plan_where_its_rooms_stand(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    away = fused_place(tmp_path, room(1), T + 10, {"den": looking(2.0, 1.0, 1.5, 90.0)})
    assert away.plan_transform is not None
    places.correct_plan(tmp_path, away.place, drawn(away, "Den", (0.5, 0.5), (3.5, 3.5)) + "\n", now=T + 11)
    fuse.join(tmp_path, home.place, {away.place: similarity(90, [10.0, 2.0, 0.0])}, now=T + 20)
    # the den's map point (2, 2) is at (10 - 2, 2 + 2) in the home's map
    assert room_on_plan(tmp_path, home.place, 8.0, 4.0) == "Den", "the den's room lies where its floor now is"
    assert room_on_plan(tmp_path, home.place, 2.0, 1.0) not in (None, "Den"), "the home's own rooms stay"
    held = places.get(tmp_path, home.place)
    assert held is not None and held.plan_source == "owner" and held.plan is not None
    (den,) = (poly for _, name, poly in floorplan.rooms_from_dsl(held.plan) if name == "Den")
    assert abs(floorplan._shoelace(den)) == pytest.approx(9.0, abs=0.01), "its size holds"
    assert places._owned(tmp_path, places.PLAN)[home.place].value["source"] == "merge"


def test_the_joined_places_own_owner_rooms_hold_where_both_places_drew_one(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    away = fused_place(tmp_path, room(1), T + 10, {"den": looking(2.0, 1.0, 1.5, 90.0)})
    places.correct_plan(tmp_path, home.place, drawn(home, "Living", (0.5, 0.5), (3.5, 3.5)) + "\n", now=T + 11)
    theirs = [drawn(away, "Den", (0.5, 0.5), (3.5, 3.5)), drawn(away, "Closet", (4.2, 0.2), (5.8, 1.8), "r2")]
    places.correct_plan(tmp_path, away.place, "\n".join(theirs) + "\n", now=T + 12)
    fuse.join(tmp_path, home.place, {away.place: np.eye(4)}, now=T + 20)  # the same room, seen apart
    held = places.get(tmp_path, home.place)
    assert held is not None and held.plan_revision == home.id, "the home's correction keeps its frame"
    assert [name for _, name, _ in floorplan.rooms_from_dsl(held.plan or "")] == ["Living", "Closet"]
    assert room_on_plan(tmp_path, home.place, 2.0, 2.0) == "Living"
    assert room_on_plan(tmp_path, home.place, 5.0, 1.0) == "Closet"
    (kept,) = [line for line in (held.plan or "").splitlines() if '"Den"' in line]
    assert kept.startswith("# from "), "the den the owner drew is kept as a comment, not dropped"


def test_a_second_join_gives_carried_rooms_ids_the_first_join_didnt_take(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    away = fused_place(tmp_path, room(1), T + 10, {"den": looking(2.0, 1.0, 1.5, 90.0)})
    # the home was joined before: a carried room of that join has the id m1_r1
    places.correct_plan(tmp_path, home.place, drawn(home, "Den", (0.5, 0.5), (3.5, 3.5), "m1_r1") + "\n", now=T + 11)
    places.correct_plan(tmp_path, away.place, drawn(away, "Closet", (0.5, 0.5), (3.5, 3.5)) + "\n", now=T + 12)
    fuse.join(tmp_path, home.place, {away.place: similarity(0, [10.0, 0.0, 0.0])}, now=T + 20)
    held = places.get(tmp_path, home.place)
    assert held is not None
    rooms = floorplan.rooms_from_dsl(held.plan or "")
    assert sorted(name for _, name, _ in rooms) == ["Closet", "Den"]
    assert len({rid for rid, _, _ in rooms}) == 2, "two rooms with one id read as one in the Places tab"
    assert room_on_plan(tmp_path, home.place, 12.0, 2.0) == "Closet"
    assert room_on_plan(tmp_path, home.place, 2.0, 2.0) == "Den"


def test_two_revisions_of_a_place_in_different_frames_are_related_by_their_cameras(tmp_path):
    to_raw = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # as `mast3r_frame` moves points
    hall = fuse._moved({"hall": looking(1.0, 1.0, 1.6, 0.0)}, to_raw)
    raw = raw_revision(tmp_path, None, room(0), T, cameras=hall)
    fused = fuse.refine(tmp_path, raw.place, now=T + 1)
    assert fused is not None and fused.frame != raw.frame
    found = fuse.frame_between(raw, fused)
    assert found is not None
    assert fuse.apply(found, np.array([[1.0, -2.0, 3.0]])) == pytest.approx(np.array([[1.0, 3.0, 2.0]]), abs=1e-6), (
        "MASt3R's (x, -height, y) is the levelled map's (x, y, height)"
    )
    bigger = places.revise(
        tmp_path,
        raw.place,
        producer="rescaled",
        cameras=fuse._moved(fused.cameras, np.diag([1.5, 1.5, 1.5, 1.0])),
        frame=str(raw.id),
        now=T + 2,
    )
    assert fuse.frame_between(fused, bigger) == pytest.approx(np.diag([1.5, 1.5, 1.5, 1.0]), abs=1e-9)
    alone = replace(bigger, cameras={"porch": fused.cameras["hall"]}, frame=None)
    assert fuse.frame_between(fused, alone) is None, "no camera in both"


def test_a_joined_map_with_no_plan_of_its_own_takes_the_carried_plan_square_to_the_page(tmp_path, monkeypatch):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    away = fused_place(tmp_path, room(1), T + 10, {"den": looking(2.0, 1.0, 1.5, 90.0)})
    places.correct_plan(tmp_path, away.place, drawn(away, "Den", (0.5, 0.5), (3.5, 3.5)) + "\n", now=T + 11)
    # the joined map's walls enclose no room, as a CUT3R walk's doubled walls don't
    monkeypatch.setattr(cameras, "layout", lambda *a, **k: cameras.Layout(None, None, (0.0, 0.0, 1.0, 0.0)))
    fuse.join(tmp_path, home.place, {away.place: similarity(30, [10.0, 2.0, 0.0])}, now=T + 20)
    held = places.get(tmp_path, home.place)
    assert held is not None and held.plan is not None
    ((_, _, den),) = floorplan.rooms_from_dsl(held.plan)
    edges = zip(den, [*den[1:], den[0]], strict=True)
    assert all(abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6 for a, b in edges), "its walls stay on the axes"
    assert min(x for x, _ in den) == pytest.approx(0) and min(y for _, y in den) == pytest.approx(0)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    assert room_on_plan(tmp_path, home.place, 10 + 2 * c - 2 * s, 2 + 2 * s + 2 * c) == "Den"


def test_a_small_room_seats_in_a_whole_homes_map_its_thinning_would_starve(tmp_path, monkeypatch):
    # The rest of the home: a 40 x 40 m floor, many times what registration keeps of a cloud (REG_POINTS, made small
    # here to keep the maps small), so thinned whole, the den's own points would be a sliver of what ICP sees
    monkeypatch.setattr(fuse, "REG_POINTS", 15_000)
    rng = np.random.default_rng(7)
    rest = np.c_[rng.uniform(-50.0, -10.0, (400_000, 2)), np.zeros(400_000)]
    home = fused_place(tmp_path, np.vstack([room(0), rest]), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    away = fused_place(tmp_path, room(1), T + 10, {"den": looking(2.0, 1.0, 1.5, 90.0)})
    joined = fuse.join(tmp_path, home.place, {away.place: similarity(90, [10.0, 2.0, 0.0])}, now=T + 20)
    # tonight's snapshot of the den: its room as its own map holds it, the den camera where that map registered it
    snapshot = fuse._cloud(tmp_path, away)
    assert snapshot is not None
    rev = fuse.seat(tmp_path, home.place, snapshot[0], None, (0.0, 0.0, 1.0, 0.0), dict(away.cameras), now=T + 30)
    assert rev is not None, "a room that stayed put seats"
    assert np.asarray(rev.cameras["den"].pose).reshape(4, 4)[:3, 3] == pytest.approx(
        np.asarray(joined.cameras["den"].pose).reshape(4, 4)[:3, 3], abs=0.05
    ), "where the join put it"


def test_a_seat_keeps_the_fuller_background_its_camera_had_and_fills_tonights_gaps(tmp_path):
    # Tonight's registration knew 98 of cam0's 2,304 cells, and the seat wrote that over a full background
    had = places.Background(20, 20, (0.0,) * 200 + (0.45,) * 200)  # the floor, then the couch's back
    home = fused_place(tmp_path, room(0), T, {"hall": replace(looking(1.0, 1.0, 1.6, 0.0), background=had)})
    snapshot = fuse._cloud(tmp_path, home)
    assert snapshot is not None

    def seated(background, now):
        poses = {"hall": replace(home.cameras["hall"], background=background)}
        rev = fuse.seat(tmp_path, home.place, snapshot[0], None, (0.0, 0.0, 1.0, 0.0), poses, now=now)
        assert rev is not None
        return rev.cameras["hall"].background

    sparse = places.Background(20, 20, (0.0,) * 30 + (None,) * 370)
    assert seated(sparse, T + 30) == had
    # The same view with someone on the couch tonight (cells 300 to 309 unknown) and the floor's cell 0 seen higher
    gaps = (0.05,) + (0.0,) * 199 + (0.45,) * 100 + (None,) * 10 + (0.45,) * 90
    assert seated(places.Background(20, 20, gaps), T + 40).heights == (0.05,) + had.heights[1:]


def test_a_seated_cloud_is_polished_onto_the_map_around_it_and_the_maps_cameras_hold(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    truth = similarity(-30, [0.8, 1.5, 0.0])  # the walk's levelled frame to the map's
    part = room(1)
    part = part[part[:, 0] < 4.5]
    walk = fuse.apply(np.linalg.inv(truth), part)
    seat = similarity(3, [0.3, -0.2, 0.0]) @ truth  # where the cameras put it: 0.3 m and 3 degrees off
    rev = fuse.fuse_in(
        tmp_path, home.place, walk, None, fuse.LEVEL_FLOOR, {"walk-0m00s": pose_at(0.0, 0.0, 1.4)}, [], seated=seat
    )
    assert rev is not None and rev.producer == fuse.PRODUCER
    look = np.asarray(rev.cameras["walk-0m00s"].pose).reshape(4, 4)
    assert look[:3, 3] == pytest.approx(truth[:3, 3] + [0.0, 0.0, 1.4], abs=0.05), "ICP took the nudge out"
    assert rev.cameras["hall"] == home.cameras["hall"]
    assert rev.frame == home.frame


def test_a_seat_the_map_around_it_cant_polish_stands_as_found(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    rng = np.random.default_rng(5)
    shed = np.c_[rng.uniform(0, 1.5, (20_000, 2)), rng.uniform(0, 2.6, 20_000)]  # nothing like the map
    seat = similarity(90, [20.0, 0.0, 0.0])
    rev = fuse.fuse_in(
        tmp_path, home.place, shed, None, fuse.LEVEL_FLOOR, {"walk-0m00s": pose_at(1.0, 0.0, 1.4)}, [], seated=seat
    )
    assert rev is not None
    look = np.asarray(rev.cameras["walk-0m00s"].pose).reshape(4, 4)
    assert look[:3, 3] == pytest.approx([20.0, 1.0, 1.4], abs=1e-6)
    pts, _, _ = ply.read(places.root(tmp_path) / rev.place / str(rev.map))
    assert int(((pts[:, 0] > 18.4) & (pts[:, 0] < 20.1)).sum()) > 1000, "the shed is in the map where it was seated"
    assert rev.cameras["hall"] == home.cameras["hall"]


def test_a_place_is_restored_to_a_fused_revision_after_a_map_without_its_cameras(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    before, _, _ = ply.read(places.root(tmp_path) / home.place / str(home.map))
    raw_revision(tmp_path, home.place, room(2), T + 10, cameras={})  # a map that took the place without its cameras
    assert places.camera_places(tmp_path) == {}
    rev = fuse.restore(tmp_path, home.place, home.id, now=T + 20)
    assert rev is not None
    assert places.latest(tmp_path, home.place) == rev
    assert rev.cameras == home.cameras and rev.signals == home.signals and rev.frame == home.frame
    assert places.camera_places(tmp_path) == {"hall": home.place}
    after, _, _ = ply.read(places.root(tmp_path) / rev.place / str(rev.map))
    assert after == pytest.approx(before)


def test_restoring_a_revision_that_isnt_a_fused_one_writes_nothing(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    raw = raw_revision(tmp_path, home.place, room(2), T + 10, cameras={})
    assert fuse.restore(tmp_path, home.place, raw.id) is None
    assert fuse.restore(tmp_path, home.place, "no-such-revision") is None
    assert places.latest(tmp_path, home.place) == raw


def test_a_room_joined_in_the_wrong_spot_moves_with_its_camera_and_the_rest_stays(tmp_path):
    hall = room(0)
    den = room(1) + [20.0, 0.0, 0.0]  # joined 20 m along x, where it isn't
    pts, _ = fuse.voxelize(np.r_[hall, den], None, fuse.VOXEL)  # a fused map: one point a voxel
    # the den's camera in a corner, wide, seeing most of its room (a fixed camera's room is what it sees)
    cams = {"hall": looking(0.5, 2.0, 1.6, 0.0), "den": looking(20.1, 0.1, 2.4, 30.0, 30.0, focal=150.0)}
    ones = np.ones(len(pts), np.int64)
    home = places.revise(
        tmp_path,
        None,
        producer=fuse.PRODUCER,
        map_bytes=ply.dumps(pts, None, {"seen": ones, "missed": 0 * ones}),
        map_format="ply",
        floor=fuse.LEVEL_FLOOR,
        cameras=cams,
        now=T,
    )
    move = similarity(90, [-8.0, 10.0, 0.0])  # where the den is: (x, y) to (-y - 8, x + 10)
    got = fuse.move_room(tmp_path, home.place, "den", move, now=T + 1)
    assert got is not None
    rev, voxels = got
    after, _, _ = ply.read(places.root(tmp_path) / rev.place / str(rev.map))
    den_now = np.asarray(rev.cameras["den"].pose).reshape(4, 4)
    assert den_now[:3, 3] == pytest.approx([-8.1, 30.1, 2.4])
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    assert den_now[:3, 2] == pytest.approx([-c * s, c * c, -s], abs=1e-9), "facing 120 degrees: turned 90 with it"
    assert rev.cameras["hall"] == cams["hall"] and rev.frame == home.frame
    in_new_spot = (after[:, 0] > -12.1) & (after[:, 0] < -7.9) & (after[:, 1] > 29.9) & (after[:, 1] < 36.1)
    assert in_new_spot.sum() == voxels > 0.6 * len(fuse.voxelize(den, None, fuse.VOXEL)[0]), "most of the den moved"
    hall_voxels = fuse.voxelize(hall, None, fuse.VOXEL)[0]
    in_hall = (after[:, 0] > -0.1) & (after[:, 0] < 6.1) & (after[:, 1] > -0.1) & (after[:, 1] < 4.1)
    assert int(in_hall.sum()) == len(hall_voxels), "the hall's voxels stay where they were"


def test_moving_a_camera_the_place_doesnt_register_writes_nothing(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    assert fuse.move_room(tmp_path, home.place, "den", np.eye(4)) is None
    assert places.latest(tmp_path, home.place) == home


# An owner plan drawn on a fused map with no plan file (an imported one), and the rebuild from walks

VIEW = places.PageView("", 0.4, (3.0, -1.0), True)  # how the owner's page lies in the plan frame


def rect(name, lo, hi, ident, on_page=False):
    """A room over map (= plan, on a plan-less levelled map) `lo` to `hi`, in the plan frame or on VIEW's page."""
    corners = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    if on_page:
        corners = [VIEW.to_page(x, y) for x, y in corners]
    return f'room {ident} "{name}" other poly ' + " ".join(f"{x:.4f},{y:.4f}" for x, y in corners)


def no_machine_plan(monkeypatch):
    # walls that enclose no room, as Place 1's fused map's didn't: its revisions have no plan file
    monkeypatch.setattr(cameras, "layout", lambda *a, **k: cameras.Layout(None, None, (0.0, 0.0, 1.0, 0.0)))


def owned(tmp_path, place, base, text, page=None, now=T + 11):
    view = None if page is None else {**VIEW.as_dict(), "plan": page}
    places.correct_plan(tmp_path, place, text, now=now, base=base, view=view)


def on_plan(state, place, x, y):
    """The room the holding plan has at map (x, y), for a plan drawn on a plan-less levelled map (plan = map)."""
    held = places.get(state, place)
    assert held is not None
    found = floorplan.room_at(floorplan.rooms_from_dsl(held.plan or ""), x, y)
    return None if found is None else found[1]


def centre(poly):
    return (sum(x for x, _ in poly) / len(poly), sum(y for _, y in poly) / len(poly))


def test_a_join_carries_owner_plans_drawn_on_plan_less_maps_and_keeps_the_page(tmp_path, monkeypatch):
    # The carry read the drawn revision's plan_transform, None on a plan-less map, so the den was kept as comments
    no_machine_plan(monkeypatch)
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    away = fused_place(tmp_path, room(1), T + 10, {"den": looking(2.0, 1.0, 1.5, 90.0)})
    assert home.plan is None and home.plan_transform is None
    living = rect("Living", (0.5, 0.5), (3.5, 3.5), "r1")
    owned(tmp_path, home.place, home.id, living + "\n", page=rect("Living", (0.5, 0.5), (3.5, 3.5), "r1", True) + "\n")
    owned(tmp_path, away.place, away.id, rect("Den", (0.5, 0.5), (3.5, 3.5), "r1") + "\n", now=T + 12)
    fuse.join(tmp_path, home.place, {away.place: similarity(90, [10.0, 2.0, 0.0])}, now=T + 20)
    # the den's map point (2, 2) is at (10 - 2, 2 + 2) in the home's map
    assert on_plan(tmp_path, home.place, 8.0, 4.0) == "Den"
    assert on_plan(tmp_path, home.place, 2.0, 2.0) == "Living"
    held = places.get(tmp_path, home.place)
    assert held is not None and held.view is not None, "the page the owner drew on is kept"
    page = {name: poly for _, name, poly in floorplan.rooms_from_dsl(held.view.plan)}
    assert centre(page["Den"]) == pytest.approx(VIEW.to_page(8.0, 4.0), abs=1e-3), "the den is drawn on the page"
    assert centre(page["Living"]) == pytest.approx(VIEW.to_page(2.0, 2.0), abs=1e-3)


def slab(x0, x1, z, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(x0, x1, n), rng.uniform(0.0, 1.0, n), np.full(n, z)]


def first_map(points):
    return as_fused(fuse.voxelize(points, None, fuse.VOXEL)[0])


@pytest.mark.parametrize("walked", [True, False], ids=["walked", "camera-made"])
def test_a_camera_cloud_never_fades_what_a_walk_saw(walked):
    old = first_map(slab(0.0, 1.0, 0.0))
    old = replace(old, walked=np.full(len(old.points), int(walked), np.int64))  # a walk's floor, or a camera's
    floor = len(old.points)
    # three times a camera's cloud sees that floor 9 cm higher, as a soft MASt3R snapshot can: in view, nowhere near
    for k in range(3):
        old = fuse.merge(old, slab(0.0, 1.0, 0.09, seed=2 + k), None)
    kept = int((np.abs(old.points[:, 2]) < 0.02).sum())
    assert (kept == floor) is walked
    assert kept < floor / 2 or walked


def test_a_walks_misses_fade_what_it_no_longer_sees():
    fused = fuse.merge(first_map(np.r_[slab(0.0, 1.0, 0.0), slab(0.0, 1.0, 0.5, seed=1)]), slab(0, 1, 0, seed=2), None)
    for k in range(2):  # the box at 0.5 m is gone: two walks see the floor under it
        fused = fuse.merge(fused, slab(0.0, 1.0, 0.0, seed=3 + k), None, walked=True)
    assert not (np.abs(fused.points[:, 2] - 0.5) < 0.02).any()
    assert fused.walks.max() == 2 and fused.counts()["walked"].max() == 2


def test_a_camera_cloud_goes_in_beside_a_walk_and_wherever_no_walk_saw():
    walked = fuse.Fused(slab(0.0, 1.0, 0.0), None, np.ones(2000, np.int64), np.zeros(2000, np.int64))
    walked = replace(walked, walked=np.ones(2000, np.int64))
    near = np.array([[0.5, 0.5, 0.03]])  # 3 cm over the walk's floor
    off = np.array([[0.5, 0.5, 0.2]])  # 20 cm over it, where the walk saw only the floor
    away = np.array([[3.0, 0.5, 0.2]])  # where no walk saw
    assert fuse.gate(walked, np.r_[near, off, away]).tolist() == [True, False, True]
    assert fuse.gate(replace(walked, walked=None), off).tolist() == [True], "nothing walked: all of it goes in"


def owner_place(tmp_path, monkeypatch, spot=(1.2, 1.0, 1.6)):
    """A place with a plan-less fused map of room(0) and an owner plan drawn on it: the room, and camera hall's
    measured spot at `spot` (map metres = plan metres)."""
    no_machine_plan(monkeypatch)
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    den = 'plan "Home" walls 0.02/0.02\nroom r1 "Den" other poly 0,0 6,0 6,2 4,2 4,4 0,4\n'
    owned(tmp_path, home.place, home.id, den + f"x-camera hall at {spot[0]},{spot[1]} height {spot[2]}\n", page=den)
    return home


def test_a_rebuild_places_a_snapshot_where_it_fits_the_walk_with_its_camera_on_the_owners_spot():
    walk = room(0)
    walk = walk[walk[:, 0] < 4.5]  # the walk saw the room up to x = 4.5
    truth = similarity(-30, [0.8, 1.5, 0.0])  # the snapshot's levelled frame to the map's
    rng = np.random.default_rng(9)
    box = np.c_[rng.uniform(5.0, 5.3, 3000), rng.uniform(0.5, 0.8, 3000), rng.uniform(0.0, 0.5, 3000)]  # unwalked
    ghost = np.c_[rng.uniform(0.5, 3.5, 3000), np.full(3000, 1.0), rng.uniform(0.3, 2.0, 3000)]  # a wall the walk lacks
    seen = np.r_[room(1), box, ghost]
    snapshot = fuse.Piece(
        fuse.apply(np.linalg.inv(truth), seen),
        None,
        fuse.LEVEL_FLOOR,
        {"hall": fuse._moved({"hall": looking(1.0, 1.0, 1.6, 0.0)}, np.linalg.inv(truth))["hall"]},
    )
    spot = looking(1.25, 0.95, 1.6, 0.0)  # the owner's tape puts it 7 cm from where its cloud says
    outside = np.array([[20.0, 20.0, 1.0]])
    rebuilt = fuse.rebuild(
        [(fuse.Piece(np.r_[walk, outside], None, fuse.LEVEL_FLOOR, {"walk-0m00s": pose_at(1, 1, 1.4)}), np.eye(4))],
        [snapshot],
        {"hall": spot},
        inside=lambda p: p[:, 0] < 10,
    )
    (seat,) = rebuilt.seats
    assert seat.polished and seat.shift == pytest.approx(math.dist((1.0, 1.0), (1.25, 0.95)), abs=0.03)
    hall = np.asarray(rebuilt.cameras["hall"].pose).reshape(4, 4)
    assert hall[:3, 3] == pytest.approx([1.25, 0.95, 1.6], abs=1e-9), "the camera stands on its spot"
    pts = rebuilt.fused.points
    in_box = (pts[:, 0] > 5.0) & (pts[:, 0] < 5.3) & (pts[:, 1] > 0.5) & (pts[:, 1] < 0.8) & (pts[:, 2] > 0.05)
    assert in_box.sum() > 200, "the snapshot fills in where the walk didn't go, where it fits the walk"
    near_ghost = (np.abs(pts[:, 1] - 1.0) < 0.02) & (pts[:, 2] > 0.4) & (pts[:, 0] > 0.6) & (pts[:, 0] < 3.4)
    assert not near_ghost.any(), "where the walk saw, the snapshot's own wall stays out"
    assert not (pts[:, 0] > 10).any() and rebuilt.outside == 1
    assert set(rebuilt.cameras) == {"hall", "walk-0m00s"}
    assert rebuilt.fused.walks.sum() > 0 and (rebuilt.fused.walks == 0).sum() > 0


def test_a_rebuild_is_written_in_the_owner_plans_frame_and_the_plan_moves_onto_it(tmp_path, monkeypatch):
    home = owner_place(tmp_path, monkeypatch)
    rebuilt = fuse.rebuild(
        [(fuse.Piece(room(0), None, fuse.LEVEL_FLOOR, {}), np.eye(4))], [], {"hall": looking(1.2, 1.0, 1.6, 0.0)}
    )
    # a map whose walls now enclose a room: no machine plan while the owner's is drawn on the frame
    monkeypatch.setattr(cameras, "layout", lambda *a, **k: cameras.Layout('plan "x"\n', (1.0,) * 12, fuse.LEVEL_FLOOR))
    before = places.get(tmp_path, home.place)
    rev = fuse.write_rebuild(tmp_path, home.place, rebuilt, ["walk"], now=T + 30)
    assert rev.frame == home.frame and rev.plan is None and rev.plan_transform is None
    held = places.get(tmp_path, home.place)
    assert held is not None and before is not None
    assert held.plan_revision == rev.id and held.plan == before.plan and held.view == before.view
    assert len(places.plan_history(tmp_path, home.place)) == 2, "the earlier record stays"
    assert held.revision.cameras["hall"].pose[3] == pytest.approx(1.2)


def test_a_nightly_seat_writes_tonights_pose_and_records_it_as_a_seat_observation(tmp_path, monkeypatch):
    home = owner_place(tmp_path, monkeypatch, spot=(1.2, 1.0, 1.6))  # off where the map has it
    snapshot = fuse._cloud(tmp_path, home)
    assert snapshot is not None
    own = similarity(20, [0.5, -0.3, 0.0])  # tonight's run's frame to the map's
    tonight = fuse._moved(home.cameras, np.linalg.inv(own))
    rev = fuse.seat(
        tmp_path, home.place, fuse.apply(np.linalg.inv(own), snapshot[0]), None, fuse.LEVEL_FLOOR, tonight, now=T + 30
    )
    assert rev is not None
    hall = np.asarray(rev.cameras["hall"].pose).reshape(4, 4)
    was = np.asarray(home.cameras["hall"].pose).reshape(4, 4)
    assert hall[:3, 3] == pytest.approx(was[:3, 3], abs=0.03), "where the seat placed it: own undone"
    assert hall[:3, :3] == pytest.approx(was[:3, :3], abs=0.01)
    assert hall[:3, 3] != pytest.approx([1.2, 1.0, 1.6], abs=0.1), "not pinned to the owner's spot"
    monkeypatch.setattr(segment_log, "at", lambda state=None: segment_log.Log(segment_log.root(state)))  # on disk
    (seen,) = poses.read(tmp_path, T, T + 60)
    assert (seen.t, seen.camera, seen.source, seen.place) == (T + 30, "hall", "seat", home.place)
    assert seen.frame == rev.frame
    assert seen.pose == pytest.approx(hall, abs=1e-5)
    assert 0.35 < seen.quality <= 1.0


def test_a_lidar_walk_is_placed_by_the_owner_plan_only_when_it_fits_it(tmp_path, monkeypatch):
    home = owner_place(tmp_path, monkeypatch)
    truth = similarity(-30, [0.8, 1.5, 0.0])  # the walk's levelled frame to the map's
    found = fuse.by_plan(tmp_path, home.place, fuse.apply(np.linalg.inv(truth), room(1)))
    assert found is not None
    assert found[0] == pytest.approx(truth, abs=0.03)
    rng = np.random.default_rng(5)
    shed = np.c_[rng.uniform(0, 1.5, (20_000, 2)), rng.uniform(0, 2.6, 20_000)]
    assert fuse.by_plan(tmp_path, home.place, shed) is None


def test_no_owner_plan_on_the_frame_places_nothing_by_plan(tmp_path):
    home = fused_place(tmp_path, room(0), T, {"hall": looking(1.0, 1.0, 1.6, 0.0)})
    assert fuse.by_plan(tmp_path, home.place, room(1)) is None
    assert fuse.measured_spots(tmp_path, home.place) == {}
