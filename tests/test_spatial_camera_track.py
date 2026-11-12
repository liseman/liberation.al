"""Tracking a camera while it moves (spatial/camera_track.py), on made-up rooms and made-up moves. The model never
runs: a tracker's answers are made up from where the camera really is at each frame, with the drift a real tracker
has added on, and a keyframe's points by casting the rooms' rays from there."""

import math

import numpy as np
import pytest

from liberational.spatial import camera_clouds as clouds
from liberational.spatial import camera_track as tracks
from liberational.spatial import cloud_map
from liberational.spatial import cloud_scene as scenes

T = 1_762_500_000.0
PLACE = "11111111-1111-7111-8111-111111111111"
#: Two made-up rooms that share a wall: a hall 4 m by 5 m, and beside it a den shaped like an L.
HALL = ((0.0, 0.0), (4.0, 0.0), (4.0, 5.0), (0.0, 5.0))
DEN = ((4.0, 0.0), (10.0, 0.0), (10.0, 3.5), (7.0, 3.5), (7.0, 6.0), (4.0, 6.0))
ROOMS = (HALL, DEN)
W, H, FOCAL = 128, 96, 70.0
HEIGHT, CEILING = 1.4, 2.4
PITCH = math.radians(20.0)  # the made-up camera looks this far below level


def at(x: float, y: float, degrees: float) -> scenes.At:
    return scenes.At(x, y, math.radians(degrees))


def camera_in_scene(where: scenes.At) -> np.ndarray:
    """A camera's pose in the scene (camera to scene, 4 x 4), standing at `where`, HEIGHT up, pitched down."""
    s, c = math.sin(where.bearing), math.cos(where.bearing)
    right, level_ahead, up = np.array([c, -s, 0.0]), np.array([s, c, 0.0]), np.array([0.0, 0.0, 1.0])
    ahead = math.cos(PITCH) * level_ahead - math.sin(PITCH) * up
    down = -math.sin(PITCH) * level_ahead - math.cos(PITCH) * up
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, ahead], axis=1)
    pose[:3, 3] = (where.x, where.y, HEIGHT)
    return pose


def seen_from(where: scenes.At, walls: bool = True) -> clouds.Pointmap:
    """What a camera standing at `where` in ROOMS sees, as a point for every pixel in its own frame: each pixel's
    ray as far as the floor, the ceiling, or the first wall. Without `walls` the model is sure only of the floor
    (a view with no wall to go by: the walls and ceiling give no point)."""
    pose = camera_in_scene(where)
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - W / 2) / FOCAL, (v - H / 2) / FOCAL, np.ones((H, W))], axis=2)
    out = rays @ pose[:3, :3].T  # each ray in the scene
    with np.errstate(divide="ignore", invalid="ignore"):
        to_floor = np.where(out[..., 2] < 0, -HEIGHT / out[..., 2], np.inf)
        reach = np.minimum(to_floor, np.where(out[..., 2] > 0, (CEILING - HEIGHT) / out[..., 2], np.inf))
        for ring in ROOMS:
            for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
                run = np.array(b) - np.array(a)
                gap = np.array(a) - pose[:2, 3]
                cross = out[..., 0] * run[1] - out[..., 1] * run[0]
                along_ray = (gap[0] * run[1] - gap[1] * run[0]) / cross
                along_wall = (gap[0] * out[..., 1] - gap[1] * out[..., 0]) / cross
                met = (np.abs(cross) > 1e-12) & (along_ray > 1e-6) & (along_wall >= 0) & (along_wall <= 1)
                reach = np.minimum(reach, np.where(met, along_ray, np.inf))
    on_floor = np.isfinite(reach) & (reach >= to_floor - 1e-9)
    reach = np.where(np.isfinite(reach), reach, 30.0)
    return clouds.Pointmap(
        points=(rays * reach[..., None]).astype(np.float32),
        confidence=np.where(walls | on_floor, 3.0, 0.0).astype(np.float32),
        colour=np.full((H, W, 3), 150, dtype=np.uint8),
        covered=np.zeros((H, W), dtype=bool),
    )


def turned(degrees: float) -> np.ndarray:
    """A turn about the camera's own down axis: what a tracker's drift in facing looks like."""
    a = math.radians(degrees)
    out = np.eye(4)
    out[:3, :3] = [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    return out


def tipped(degrees: float) -> np.ndarray:
    """A turn about the camera's own sideways axis: what a tracker's drift in pitch looks like."""
    a = math.radians(degrees)
    out = np.eye(4)
    out[:3, :3] = [[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]
    return out


class Moves:
    """A made-up tracker (`camera_track.Session`): the camera really goes through `truth`, a pose a frame. Its
    answers are the camera in the frame of the track's first keyframe (the first frame, or with `stood` the resting
    picture taken there, when the track begins on it), off by `drift` degrees of facing and `tip` degrees of pitch more with every frame;
    every `every`th frame is a keyframe. Frames numbered in `blind` match no keyframe; one numbered in `found`
    lost the current keyframe and was found again among the earlier ones; a keyframe numbered in `floor_only` sees
    no wall."""

    def __init__(self, truth, drift=0.0, every=3, blind=(), floor_only=(), stood=None, found=(), tip=0.0):
        self.truth, self.drift, self.tip, self.every = list(truth), drift, tip, every
        self.blind, self.floor_only, self.found = set(blind), set(floor_only), set(found)
        self.stood = stood
        self.sent = 0
        self.resets: list[int] = []
        self.anchors = 0
        self.closed = False
        self._first = 0
        self._origin = camera_in_scene(self.truth[0])

    def step(self, picture, overlay, reset, anchor=False):
        if anchor:
            self.anchors += 1
            if self.stood is None:
                return tracks.Step(False, anchor=True)  # no resting picture to begin on
            self._first, self._origin = 0, camera_in_scene(self.stood)
            return tracks.Step(True, np.eye(4), True, None, anchor=True)
        i = min(self.sent, len(self.truth) - 1)
        self.sent += 1
        if reset:
            self.resets.append(i)
            self._first, self._origin = i, camera_in_scene(self.truth[i])
        elif i in self.blind:
            return tracks.Step(False)
        own = np.linalg.inv(self._origin) @ camera_in_scene(self.truth[i])
        pose = own @ turned(self.drift * (i - self._first)) @ tipped(self.tip * (i - self._first))
        key = reset or (i - self._first) % self.every == 0
        points = seen_from(self.truth[i], walls=i not in self.floor_only) if key else None
        return tracks.Step(True, pose, reset, points, recovered=i in self.found)

    def close(self):
        self.closed = True


@pytest.fixture
def store(tmp_path):
    return clouds.Store(clouds.root(tmp_path))


def walk(start: scenes.At, steps: int, ahead: float = 0.0, turn: float = 0.0) -> list[scenes.At]:
    """`steps` poses from `start`, each `ahead` metres on and `turn` degrees round from the one before."""
    out = [start]
    for _ in range(steps):
        out.append(out[-1].step(0.0, ahead, math.radians(turn)))
    return out


def follow(store, session: Moves, stood: scenes.At, frames: int, each=None) -> tracks.Track:
    """Track `frames` frames of `session`, half a second apart, for a camera that stood at `stood`; `each` is
    called with the track after every frame."""
    rest = None if session.stood is None else camera_in_scene(session.stood)
    track = tracks.Track(store, "desk", PLACE, "desk-spot", ROOMS, stood, T, rest)
    if track.anchored:
        track.take(session.step(None, None, True, True), T)
    for n in range(frames):
        track.take(session.step(None, None, track.begin), T + 0.5 * (n + 1))
        if each is not None:
            each(track)
    return track


def off(track: tracks.Track, truth: scenes.At) -> tuple[float, float]:
    return scenes.moved_by(track.at, truth)


def test_a_camera_carried_across_a_room_is_placed_frame_by_frame_where_it_really_is(store):
    truth = walk(at(1.0, 1.0, 20.0), 12, ahead=0.15, turn=4.0)  # 1.8 m on, turning 48 degrees
    track = follow(store, Moves(truth), stood=at(1.0, 1.0, 20.0), frames=13)
    assert off(track, truth[-1]) == (pytest.approx(0.0, abs=0.05), pytest.approx(0.0, abs=1.0))
    assert track.height == pytest.approx(HEIGHT, abs=0.05)
    assert len(track.path) == 13 and track.held == T + 0.5 * 13  # the last keyframe was held to the walls


def test_the_plans_walls_take_out_what_the_tracker_drifts(store):
    truth = walk(at(1.0, 1.0, 20.0), 24, ahead=0.05, turn=2.0)
    # the tracker's facing runs off by 1.5 degrees a frame: 36 degrees by the end, were nothing to hold it
    track = follow(store, Moves(truth, drift=1.5), stood=at(1.0, 1.0, 20.0), frames=25)
    assert off(track, truth[-1]) == (pytest.approx(0.0, abs=0.1), pytest.approx(0.0, abs=4.5))  # two frames' worth


def test_a_track_begins_facing_the_way_its_first_frame_looks_not_the_way_the_camera_stood(store):
    # by the time a move is noticed the camera has turned: it stood facing 20 degrees, and its first frame faces 110
    track = follow(store, Moves([at(1.0, 1.0, 110.0)]), stood=at(1.0, 1.0, 20.0), frames=1)
    assert off(track, at(1.0, 1.0, 110.0)) == (pytest.approx(0.0, abs=0.05), pytest.approx(0.0, abs=1.0))


def test_a_keyframe_with_no_wall_in_view_stays_where_the_track_puts_it_and_the_track_says_it_is_unheld(store):
    truth = walk(at(1.0, 1.0, 20.0), 6, ahead=0.1)
    track = follow(store, Moves(truth, floor_only={3, 6}), stood=at(1.0, 1.0, 20.0), frames=7)
    assert off(track, truth[-1]) == (pytest.approx(0.0, abs=0.05), pytest.approx(0.0, abs=1.0))  # no drift here
    assert track.held == T + 0.5  # only the first keyframe was held
    assert [row["held"] for row in track.keyframes] == [True, False, False]


def test_a_stretch_the_tracker_cant_match_begins_the_track_again_from_what_it_then_sees(store):
    truth = walk(at(1.0, 1.0, 20.0), 14, ahead=0.05, turn=3.0)
    session = Moves(truth, blind=range(3, 9))  # six frames in a row: three seconds
    track = follow(store, session, stood=at(1.0, 1.0, 20.0), frames=15)
    # two seconds without a match and it begins again, on the frame after: the fifth blind one
    assert session.resets == [0, 8] and track.restarts == 1
    # Its facing is the walls' again. Where it stands is not: it went 0.25 m unseen, and the two walls in view
    # can't tell a camera that far along from a cloud that much too big.
    shift, turn = off(track, truth[-1])
    assert turn == pytest.approx(0.0, abs=1.0) and shift <= 0.3
    assert len(track.path) == 10  # the frames that matched nothing have no pose: none was made up


def test_a_jerk_that_loses_the_keyframe_and_finds_an_earlier_one_goes_on_in_the_same_track(store):
    stood = at(1.0, 1.0, 20.0)
    there = walk(stood, 6, turn=6.0)
    truth = there + [there[-1].step(0.0, 0.0, math.radians(40.0))] * 2 + there[::-1]  # swung away, and back
    session = Moves(truth, stood=stood, blind={7, 8}, found={9})
    track = follow(store, session, stood, frames=len(truth))
    assert (session.resets, track.restarts, track.recovered) == ([], 0, 1)  # one second lost, found, never begun again
    assert off(track, truth[-1]) == (pytest.approx(0.0, abs=0.05), pytest.approx(0.0, abs=1.0))
    assert len(track.path) == len(truth) - 2


def test_a_track_begun_on_the_cameras_resting_picture_stands_on_the_plan_from_its_first_frame_wall_or_no_wall(store):
    stood = at(1.0, 1.0, 20.0)
    # by the time the move is noticed it has turned 70 degrees and looks at the floor: nothing to place it by
    truth = [at(1.0, 1.0, 90.0)] + walk(at(1.0, 1.0, 90.0), 5, turn=4.0)
    session = Moves(truth, stood=stood, floor_only=range(len(truth)))
    track = follow(store, session, stood, frames=len(truth))
    assert session.anchors == 1 and session.resets == []
    assert off(track, truth[-1]) == (pytest.approx(0.0, abs=0.02), pytest.approx(0.0, abs=0.5))
    assert track.held is None and len(track.path) == len(truth)


def test_a_slow_pan_adds_to_the_map_as_it_goes_frame_by_frame_and_not_only_at_the_end(store):
    stood = at(2.0, 1.5, -40.0)
    truth = walk(stood, 30, turn=3.0)  # panned through 90 degrees in 15 seconds
    where = cloud_map.folder(store.folder, PLACE)
    seen: list[tuple[int, int, float]] = []  # after each frame: the track's count, the map's points, its floor

    def look(track):
        meta = cloud_map.record(where) or {"shots": [], "floor_m2": 0.0}
        seen.append((track.added, sum(row["points"] for row in meta["shots"]), meta.get("floor_m2", 0.0)))

    track = follow(store, Moves(truth, stood=stood, every=5), stood, frames=31, each=look)
    added = [count for count, _, _ in seen]
    assert added == sorted(added) and len(set(added)) >= 5  # it grew at keyframe after keyframe
    assert added[10] > 0 and added[20] > added[10] and added[30] > added[20]  # a third, two thirds, all the way
    held = [points for _, points, _ in seen]
    assert held[10] > 0 and held[20] > held[10] and held[30] > held[20]  # and the map itself held more each time
    floor = [covered for _, _, covered in seen]
    assert floor[30] > floor[10] > 0  # the floor it covers moved with it
    assert track.added_at == T + 0.5 * 31 and (scenes.tracked(store, "desk") or {})["added"] == track.added


def test_a_keyframe_no_wall_holds_is_added_where_the_track_puts_it_and_one_that_stands_elsewhere_is_refused(
    store, tmp_path
):
    stood = at(2.0, 1.5, 0.0)
    where = cloud_map.folder(store.folder, PLACE)
    truth = walk(stood, 6, ahead=0.1)
    track = follow(store, Moves(truth, stood=stood, floor_only={3, 6}), stood, frames=7)
    rows = {row["id"].split(".")[-1]: row for row in (cloud_map.record(where) or {})["shots"]}
    assert [row["held"] for row in track.keyframes] == [True, False, False]
    assert not rows["2"]["held"] and rows["2"]["points"] > 0  # the floor it saw past what the first keyframe did
    # the same move in a house of its own, with the tracker's pitch running off 3 degrees a frame and no wall to
    # hold it: its floor lies tilted through the floor the map's first shot saw, and the map refuses it
    other = clouds.Store(clouds.root(tmp_path / "other"))
    wild = follow(other, Moves(truth, stood=stood, floor_only={3, 6}, tip=3.0), stood, frames=7)
    refused = [row for row in (cloud_map.record(cloud_map.folder(other.folder, PLACE)) or {})["refused"]]
    assert [row["added"] for row in wild.keyframes[1:]] == [0, 0]
    assert [row["why"] for row in refused if row["state"] == "refused"] == ["apart", "apart"]


def test_the_keyframes_the_walls_hold_fill_in_the_places_map_and_a_wall_seen_in_several_is_one_wall(store):
    truth = walk(at(1.0, 1.0, 20.0), 30, turn=3.0)  # turned through 90 degrees where it stands
    track = follow(store, Moves(truth, every=5), stood=at(1.0, 1.0, 20.0), frames=31)
    held = cloud_map.read(cloud_map.folder(store.folder, PLACE))
    assert held is not None and [row["camera"] for row in held.meta["shots"]] == ["desk"] * len(held.meta["shots"])
    added = [row["added"] for row in track.keyframes]
    assert len(added) == 7 and all(row["held"] for row in track.keyframes)
    assert added[0] > 0 and sum(added[1:]) > 0  # what later keyframes saw that the first didn't went in too
    assert max(added[1:]) < added[0]  # and what they saw again did not
    # the wall between the two rooms, seen in most of the keyframes, is drawn once: on the plan's wall
    wall = held.flat[
        (held.flat[:, 2] > 0.3) & (held.flat[:, 0] > 3.5) & (held.flat[:, 1] > 0.3) & (held.flat[:, 1] < 4.7)
    ]
    assert len(wall) > 100 and wall[:, 0] == pytest.approx(4.0, abs=0.02)
    _, count = np.unique(np.floor(held.flat / cloud_map.CELL_M).astype(np.int64), axis=0, return_counts=True)
    assert count.max() <= cloud_map.CAP


def tracker_of(store, tmp_path, monkeypatch, session, stood=None, people=False):
    stood = stood or at(1.0, 1.0, 20.0)
    spot = scenes.SpotAt("desk", PLACE, stood, (stood.x, -stood.y), False, "desk-spot")
    monkeypatch.setattr(scenes, "spots_of", lambda state: {"desk": spot, "shelf": spot})
    monkeypatch.setattr(scenes, "plan_rooms", lambda state, place, mirrored: ROOMS)
    rest = (stood, camera_in_scene(stood), lambda: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(scenes, "resting", lambda store, camera, spots, rooms: rest)
    return tracks.Tracker(
        tmp_path,
        store,
        lit=lambda pixels: True,
        detect=lambda pixels: [(0.1, 0.1, 0.5, 0.9)] if people else [],
        prepare=lambda camera, pixels: (pixels, None),
        open_session=lambda: session,
        structure=lambda pixels: np.asarray(pixels, dtype=np.float64),
    )


STILL_VIEW, OTHER_VIEW = np.array([1.0, 0.0]), np.array([0.0, 1.0])  # two pictures whose edges are nothing alike


def test_a_camera_is_tracked_from_the_sample_that_shows_its_view_moving_and_one_camera_at_a_time(
    store, tmp_path, monkeypatch
):
    tracker = tracker_of(store, tmp_path, monkeypatch, Moves(walk(at(1.0, 1.0, 20.0), 3)))
    assert not tracker.look("desk", STILL_VIEW, T)  # its first sample: nothing to compare with
    assert not tracker.look("desk", STILL_VIEW, T + 5)  # the same view
    assert tracker.look("desk", OTHER_VIEW, T + 10)
    assert tracker.active == "desk"
    tracker.look("shelf", STILL_VIEW, T + 10)
    assert not tracker.look("shelf", OTHER_VIEW, T + 15) and tracker.active == "desk"  # the tracker is taken


def test_a_view_that_changes_with_someone_in_it_or_with_no_tracker_to_be_had_is_not_tracked(
    store, tmp_path, monkeypatch
):
    crowded = tracker_of(store, tmp_path, monkeypatch, Moves([at(1.0, 1.0, 20.0)]), people=True)
    crowded.look("desk", STILL_VIEW, T)
    assert not crowded.look("desk", OTHER_VIEW, T + 5)
    none = tracker_of(store, tmp_path, monkeypatch, None)  # no runtime set up, or no GPU memory to spare
    none.look("desk", STILL_VIEW, T)
    assert not none.look("desk", OTHER_VIEW, T + 5) and none.active is None


def room_picture(shift: int = 0, light=None, seed: int = 11) -> np.ndarray:
    """A made-up picture of a room, 320 x 180: blocks of furniture against a wall. `shift` moves the camera (the
    picture slides that many pixels); `light` multiplies its brightness, a number or a map of them."""
    rng = np.random.default_rng(seed)
    picture = np.full((180, 320 + 80), 150.0)
    for _ in range(14):
        x, y, w, h = rng.integers(0, 360), rng.integers(20, 150), rng.integers(15, 60), rng.integers(15, 50)
        picture[y : y + h, x : x + w] = rng.integers(40, 230)
    picture = picture[:, 40 + shift : 360 + shift]
    lit = picture * (1.0 if light is None else light)
    return np.repeat(np.clip(lit, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


def real_tracker(store, tmp_path, monkeypatch, people=()):
    """A tracker that looks at pictures as the service's does (`structure`), with a tracker always to be had."""
    made = tracker_of(store, tmp_path, monkeypatch, Moves([at(1.0, 1.0, 20.0)] * 3))
    made.structure = tracks.structure
    made.detect = lambda pixels: [(0.3, 0.2, 0.6, 1.0)] if pixels[0, 0, 0] in people else []
    return made


@pytest.mark.parametrize(
    "light",
    [
        pytest.param(0.45, id="the-room-dimmed"),
        pytest.param(np.where(np.arange(320) < 160, 1.7, 0.8)[None, :], id="a-lamp-on-one-side"),
        pytest.param(0.6 + 0.9 * np.exp(-(((np.arange(320) - 230) / 70.0) ** 2))[None, :], id="a-pool-of-light"),
    ],
)
def test_a_change_of_light_with_the_camera_still_starts_no_track(store, tmp_path, monkeypatch, light):
    tracker = real_tracker(store, tmp_path, monkeypatch)
    assert not tracker.look("desk", room_picture(), T)
    assert not tracker.look("desk", room_picture(light=light), T + 5)
    assert not tracker.look("desk", room_picture(), T + 10) and tracker.active is None  # nor does the light going back


@pytest.mark.parametrize("pixels", [6, 20, 60])
def test_a_camera_that_turned_starts_a_track_however_the_light_is(store, tmp_path, monkeypatch, pixels):
    tracker = real_tracker(store, tmp_path, monkeypatch)
    tracker.look("desk", room_picture(), T)
    assert tracker.look("desk", room_picture(shift=pixels, light=0.8), T + 5) and tracker.active == "desk"


def test_someone_coming_into_or_leaving_the_picture_starts_no_track(store, tmp_path, monkeypatch):
    empty, sat = room_picture(), room_picture(seed=12)  # another picture altogether: someone fills it
    sat[0, 0] = 7  # the made-up detector finds someone in a picture whose first pixel is 7
    tracker = real_tracker(store, tmp_path, monkeypatch, people={7})
    tracker.look("desk", empty, T)
    assert not tracker.look("desk", sat, T + 5)  # they sat down
    assert not tracker.look("desk", sat, T + 10)
    assert not tracker.look("desk", empty, T + 15) and tracker.active is None  # and left again


def test_a_track_ends_once_the_camera_has_rested_and_the_camera_is_left_alone_for_a_while(store, tmp_path, monkeypatch):
    moved = walk(at(1.0, 1.0, 20.0), 8, ahead=0.1, turn=5.0)
    session = Moves(moved + [moved[-1]] * 40)  # it goes 0.8 m in four seconds, then stays put
    tracker = tracker_of(store, tmp_path, monkeypatch, session)
    tracker.look("desk", STILL_VIEW, T)
    assert tracker.look("desk", OTHER_VIEW, T + 5)
    over = [tracker.step("desk", STILL_VIEW, T + 5 + 0.5 * (n + 1)) for n in range(40)]
    ended = over.index(True)
    # eight frames of moving, then `REST_S` of frames within `REST` of each other
    assert ended == 8 + round(tracks.REST_S / 0.5) and session.closed and tracker.active is None
    kept = scenes.tracked(store, "desk")
    assert kept is not None and kept["state"] == "ended"
    assert scenes.moved_by(scenes.At(*kept["xyb"]), moved[-1]) == (
        pytest.approx(0.0, abs=0.05),
        pytest.approx(0.0, abs=1.0),
    )
    assert tracker.step("desk", STILL_VIEW, T + 60)  # no track: nothing to give a frame to
    tracker.look("desk", STILL_VIEW, T + 30)
    assert not tracker.look("desk", OTHER_VIEW, T + 35)  # too soon after its last track
    assert tracker.look("desk", STILL_VIEW, T + 5 + 0.5 * (ended + 1) + tracks.COOLDOWN_S)


def test_a_camera_that_came_to_rest_somewhere_else_is_asked_for_a_cloud_and_one_that_went_nowhere_is_not(
    store, tmp_path, monkeypatch
):
    asked: list[str] = []
    moved = walk(at(1.0, 1.0, 20.0), 8, ahead=0.1, turn=5.0)
    went = tracker_of(store, tmp_path, monkeypatch, Moves(moved + [moved[-1]] * 40))
    went.rested = asked.append
    went.look("desk", STILL_VIEW, T)
    went.look("desk", OTHER_VIEW, T + 5)
    assert any(went.step("desk", STILL_VIEW, T + 5 + 0.5 * (n + 1)) for n in range(40))
    assert asked == ["desk"] and (scenes.tracked(store, "desk") or {})["state"] == "ended"
    # what changed the picture wasn't the camera moving (a lamp, someone the detector missed): it never left
    stayed = tracker_of(store, tmp_path, monkeypatch, Moves([at(1.0, 1.0, 20.0)] * 40))
    stayed.rested = asked.append
    stayed.look("shelf", STILL_VIEW, T + 200)
    stayed.look("shelf", OTHER_VIEW, T + 205)
    assert any(stayed.step("shelf", STILL_VIEW, T + 205 + 0.5 * (n + 1)) for n in range(40))
    assert asked == ["desk"] and scenes.tracked(store, "shelf") is None


def test_a_tracker_that_stops_answering_ends_the_track_and_says_so(store, tmp_path, monkeypatch):
    class Broken(Moves):
        def step(self, picture, overlay, reset, anchor=False):
            raise RuntimeError("the tracker stopped")

    session = Broken([at(1.0, 1.0, 20.0)])
    tracker = tracker_of(store, tmp_path, monkeypatch, session)
    tracker.look("desk", STILL_VIEW, T)
    tracker.look("desk", OTHER_VIEW, T + 5)
    assert tracker.step("desk", STILL_VIEW, T + 6) and session.closed and tracker.active is None
    assert scenes.tracked(store, "desk") is None  # it went nowhere: there is nothing to say of it


def test_a_track_left_open_by_a_service_that_stopped_is_over_when_the_next_one_starts(store, tmp_path, monkeypatch):
    stood = at(1.0, 1.0, 20.0)
    made = clouds.make("desk", seen_from(stood), None, T)
    store.replace(made)
    follow(store, Moves(walk(stood, 3, ahead=0.1)), stood, frames=4)
    assert (scenes.tracked(store, "desk") or {})["state"] == "tracking"
    tracker_of(store, tmp_path, monkeypatch, None)
    assert (scenes.tracked(store, "desk") or {})["state"] == "ended"


# ---- in the scene


def placed_camera(store, stood: scenes.At) -> dict[str, scenes.SpotAt]:
    """A camera with a cloud of what it sees from its spot, placed there with that picture as its anchor."""
    spots = {"desk": scenes.SpotAt("desk", PLACE, stood, (stood.x, -stood.y), False, "desk-spot")}
    made = clouds.make("desk", seen_from(stood), None, T)
    store.replace(clouds.replace(made, picture=np.zeros((36, 64, 3), np.uint8)))
    scenes.record(store, made.cloud, {}, spots)
    scenes.align(store, "desk", spots, lambda place, mirrored: ROOMS)
    return spots


def test_a_moving_camera_is_shown_where_its_track_has_it_and_what_it_sees_on_the_way_joins_the_map(store, tmp_path):
    stood = at(1.0, 1.0, 20.0)
    spots = placed_camera(store, stood)
    before = scenes.scene(store, PLACE, spots, ROOMS)
    own = scenes.ply.vertices(scenes.points(store, before))
    assert before.cameras[0].track is None and own is not None
    truth = walk(stood, 9, ahead=0.1, turn=6.0)
    follow(store, Moves(truth), stood, frames=10)
    found = scenes.scene(store, PLACE, spots, ROOMS)
    desk = found.cameras[0]
    assert desk.placed and desk.track is not None
    assert (desk.track["state"], desk.track["since"], desk.track["at"], desk.track["held"]) == (
        "tracking", T, T + 5.0, T + 5.0,
    )  # fmt: skip
    assert desk.track["went_m"] == pytest.approx(0.9, abs=0.05)
    assert (desk.track["from_m"], desk.track["from_deg"]) == (
        pytest.approx(math.hypot(truth[-1].x - 1.0, truth[-1].y - 1.0), abs=0.05),
        pytest.approx(54.0, abs=1.0),
    )
    now = np.array(desk.track["pose"]).reshape(4, 4)  # the camera, drawn where it is now
    assert now[:3, 3] == pytest.approx([truth[-1].x, truth[-1].y, HEIGHT], abs=0.05)
    # what is drawn is the place's map: it grew by what the keyframes saw that the camera's own cloud hadn't, it
    # is all this camera's, and all of it lies in the plan's rooms
    assert found.id != before.id
    rows = scenes.ply.vertices(scenes.points(store, found))
    assert rows is not None and len(rows) > len(own)
    charted = cloud_map.read(cloud_map.folder(store.folder, PLACE))
    assert charted is not None and len(rows) == len(charted.xyz)
    assert set(rows["view"].tolist()) == {0} and not rows["outside"].any()
    assert [path.name for path in tmp_path.iterdir()] == ["camera_clouds"]  # no spot and no plan was written
    assert spots["desk"].at == stood


def test_when_it_rests_its_new_cloud_is_placed_from_where_the_track_left_it_and_the_track_is_gone(store):
    stood = at(1.0, 1.0, 20.0)
    spots = placed_camera(store, stood)
    # carried into the den to a bare corner: its cloud alone couldn't say which corner of the house this is
    truth = walk(stood, 20, ahead=0.35, turn=3.5)
    rest = truth[-1]
    track = follow(store, Moves(truth), stood, frames=21)
    track.end(T + 11.0)
    made = clouds.make("desk", seen_from(rest), None, T + 40.0)
    store.replace(clouds.replace(made, picture=np.zeros((36, 64, 3), np.uint8)))
    scenes.record(store, made.cloud, {"anchor": {"match": [0.0, 0.05], "sure": [0.0, 0.0]}}, spots)
    scenes.align(store, "desk", spots, lambda place, mirrored: ROOMS)
    found = scenes.scene(store, PLACE, spots, ROOMS)
    desk = found.cameras[0]
    assert (desk.how, (desk.follow or {})["by"], desk.track) == ("following", "tracking", None)
    pose = np.array(desk.pose).reshape(4, 4)
    assert pose[:2, 3] == pytest.approx([rest.x, rest.y], abs=0.1)
    assert scenes.tracked(store, "desk") is None


def test_a_track_that_ended_before_the_cameras_cloud_was_made_says_nothing_of_where_it_is(store):
    stood = at(1.0, 1.0, 20.0)
    spots = placed_camera(store, stood)
    follow(store, Moves(walk(stood, 3, ahead=0.1)), stood, frames=4).end(T - 60.0)
    kept = scenes.tracked(store, "desk")
    assert kept is not None
    scenes.json_state.write(store.folder / "desk" / scenes.TRACK, {**kept, "at": T - 60.0})
    assert scenes.scene(store, PLACE, spots, ROOMS).cameras[0].track is None


def test_a_camera_put_back_where_it_was_is_an_ordinary_camera_again_with_no_track_left(store):
    stood = at(1.0, 1.0, 20.0)
    spots = placed_camera(store, stood)
    there_and_back = walk(stood, 4, turn=10.0) + walk(at(1.0, 1.0, 60.0), 4, turn=-10.0)[1:]
    follow(store, Moves(there_and_back), stood, frames=9).end(T + 20.0)
    assert scenes.scene(store, PLACE, spots, ROOMS).cameras[0].track is not None  # it moved, and waits for a cloud
    same = {"match": [0.3, 0.3], "sure": [0.9, 0.9], "rotation": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
            "shift": [0.0, 0.0, 0.0], "scale": 1.0, "disagree": [0.5, 0.02, 0.01]}  # fmt: skip
    made = clouds.make("desk", seen_from(stood), None, T + 60.0)
    store.replace(clouds.replace(made, picture=np.zeros((36, 64, 3), np.uint8)))
    scenes.record(store, made.cloud, {"anchor": same}, spots)  # its new picture is its old view
    desk = scenes.scene(store, PLACE, spots, ROOMS).cameras[0]
    assert (desk.how, desk.track) == ("spot", None) and scenes.tracked(store, "desk") is None
