"""Tracking a camera while it moves (spatial/camera_track.py), on made-up rooms and made-up moves. The model never
runs: a tracker's answers are made up from where the camera really is at each frame, with the drift a real tracker
has added on, and a keyframe's points by casting the rooms' rays from there."""

import math

import numpy as np
import pytest

from liberational.spatial import camera_clouds as clouds
from liberational.spatial import camera_track as tracks
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
    ray as far as the floor, the ceiling, or the first wall (without `walls`, a floor that never ends)."""
    pose = camera_in_scene(where)
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - W / 2) / FOCAL, (v - H / 2) / FOCAL, np.ones((H, W))], axis=2)
    out = rays @ pose[:3, :3].T  # each ray in the scene
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = np.where(out[..., 2] < 0, -HEIGHT / out[..., 2], np.inf)
        if walls:
            reach = np.minimum(reach, np.where(out[..., 2] > 0, (CEILING - HEIGHT) / out[..., 2], np.inf))
            for ring in ROOMS:
                for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
                    run = np.array(b) - np.array(a)
                    gap = np.array(a) - pose[:2, 3]
                    cross = out[..., 0] * run[1] - out[..., 1] * run[0]
                    along_ray = (gap[0] * run[1] - gap[1] * run[0]) / cross
                    along_wall = (gap[0] * out[..., 1] - gap[1] * out[..., 0]) / cross
                    met = (np.abs(cross) > 1e-12) & (along_ray > 1e-6) & (along_wall >= 0) & (along_wall <= 1)
                    reach = np.minimum(reach, np.where(met, along_ray, np.inf))
    reach = np.where(np.isfinite(reach), reach, 30.0)
    return clouds.Pointmap(
        points=(rays * reach[..., None]).astype(np.float32),
        confidence=np.full((H, W), 3.0, dtype=np.float32),
        colour=np.full((H, W, 3), 150, dtype=np.uint8),
        covered=np.zeros((H, W), dtype=bool),
    )


def turned(degrees: float) -> np.ndarray:
    """A turn about the camera's own down axis: what a tracker's drift in facing looks like."""
    a = math.radians(degrees)
    out = np.eye(4)
    out[:3, :3] = [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    return out


class Moves:
    """A made-up tracker (`camera_track.Session`): the camera really goes through `truth`, a pose a frame. Its
    answers are the camera in the first frame's camera frame, off by `drift` degrees of facing more with every
    frame; every `every`th frame is a keyframe. Frames numbered in `blind` can't be matched, and a keyframe numbered
    in `floor_only` sees no wall."""

    def __init__(self, truth, drift=0.0, every=3, blind=(), floor_only=()):
        self.truth, self.drift, self.every = list(truth), drift, every
        self.blind, self.floor_only = set(blind), set(floor_only)
        self.sent = 0
        self.resets: list[int] = []
        self.closed = False
        self._first = 0

    def step(self, picture, overlay, reset):
        i = min(self.sent, len(self.truth) - 1)
        self.sent += 1
        if reset:
            self.resets.append(i)
            self._first = i
        elif i in self.blind:
            return tracks.Step(False)
        own = np.linalg.inv(camera_in_scene(self.truth[self._first])) @ camera_in_scene(self.truth[i])
        pose = own @ turned(self.drift * (i - self._first))
        key = reset or (i - self._first) % self.every == 0
        points = seen_from(self.truth[i], walls=i not in self.floor_only) if key else None
        return tracks.Step(True, pose, reset, points)

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


def follow(store, session: Moves, stood: scenes.At, frames: int) -> tracks.Track:
    track = tracks.Track(store, "desk", "desk-spot", ROOMS, stood, T)
    for n in range(frames):
        track.take(session.step(None, None, track.begin), T + 0.5 * (n + 1))
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


def test_a_stretch_the_tracker_cant_match_begins_the_track_again_where_it_was_last_placed(store):
    truth = walk(at(1.0, 1.0, 20.0), 14, ahead=0.05, turn=3.0)
    session = Moves(truth, blind=range(3, 9))  # six frames in a row
    track = follow(store, session, stood=at(1.0, 1.0, 20.0), frames=15)
    assert session.resets == [0, 9] and track.restarts == 1
    # Its facing is the walls' again. Where it stands is not: it went 0.3 m unseen, and the two walls in view
    # can't tell a camera that far along from a cloud that much too big.
    shift, turn = off(track, truth[-1])
    assert turn == pytest.approx(0.0, abs=1.0) and shift <= 0.35
    assert len(track.path) == 9  # the frames that couldn't be matched have no pose: none was made up


def test_only_the_last_keyframes_clouds_are_kept_each_thinned(store):
    truth = walk(at(1.0, 1.0, 20.0), 30, turn=3.0)
    track = follow(store, Moves(truth, every=2), stood=at(1.0, 1.0, 20.0), frames=31)
    kept = sorted(int(path.stem) for path in track.folder.glob("*.ply"))
    assert kept == [row["n"] for row in track.keyframes] == [11, 12, 13, 14, 15, 16]
    assert all(row["points"] <= tracks.THIN for row in track.keyframes)


# ---- when a camera is tracked, and when it stops


def tracker_of(store, tmp_path, monkeypatch, session, stood=None, people=False):
    stood = stood or at(1.0, 1.0, 20.0)
    spot = scenes.SpotAt("desk", PLACE, stood, (stood.x, -stood.y), False, "desk-spot")
    monkeypatch.setattr(scenes, "spots_of", lambda state: {"desk": spot, "shelf": spot})
    monkeypatch.setattr(scenes, "plan_rooms", lambda state, place, mirrored: ROOMS)
    monkeypatch.setattr(scenes, "standing", lambda store, camera, spots, rooms: stood)
    return tracks.Tracker(
        tmp_path,
        store,
        thumbnail=lambda pixels: np.asarray(pixels, dtype=np.float64),
        lit=lambda pixels: True,
        detect=lambda pixels: [(0.1, 0.1, 0.5, 0.9)] if people else [],
        prepare=lambda camera, pixels: (pixels, None),
        open_session=lambda: session,
    )


STILL_VIEW, OTHER_VIEW = np.array([1.0, 0.0]), np.array([0.0, 1.0])  # two thumbnails nothing alike


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


def test_a_tracker_that_stops_answering_ends_the_track_and_says_so(store, tmp_path, monkeypatch):
    class Broken(Moves):
        def step(self, picture, overlay, reset):
            raise RuntimeError("the tracker stopped")

    session = Broken([at(1.0, 1.0, 20.0)])
    tracker = tracker_of(store, tmp_path, monkeypatch, session)
    tracker.look("desk", STILL_VIEW, T)
    tracker.look("desk", OTHER_VIEW, T + 5)
    assert tracker.step("desk", STILL_VIEW, T + 6) and session.closed and tracker.active is None
    kept = scenes.tracked(store, "desk")
    assert kept is not None and kept["state"] == "ended"


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


def test_a_moving_camera_is_shown_where_its_track_has_it_with_its_keyframes_points_and_nothing_else_is_written(
    store, tmp_path
):
    stood = at(1.0, 1.0, 20.0)
    spots = placed_camera(store, stood)
    before = scenes.scene(store, PLACE, spots, ROOMS)
    assert before.cameras[0].track is None
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
    # its cloud still stands where it was made, and the keyframes' points are drawn with it, in the plan's rooms
    assert found.id != before.id
    rows = scenes.ply.vertices(scenes.points(store, found))
    own = scenes.ply.vertices(scenes.points(store, before))
    assert rows is not None and own is not None
    assert len(rows) == len(own) + sum(row["points"] for row in (scenes.tracked(store, "desk") or {})["keyframes"])
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
    assert scenes.tracked(store, "desk") is None and not track.folder.exists()


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
