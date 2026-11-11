"""When a camera's own cloud is made again, and from which picture (spatial/surroundings_clouds.py)."""

import numpy as np
import pytest

from liberational.spatial import camera_clouds
from liberational.spatial import surroundings_clouds as due
from liberational.spatial import surroundings_worker as worker

T = 1_762_500_000.0
STEP = 5.0  # the sampler's pace


def picture(seed: int, noise: int = 0) -> np.ndarray:
    """A made-up 64 x 36 frame: a different seed is a different view, and `noise` the same view on another sample."""
    base = np.random.default_rng(seed).integers(40, 160, (36, 64, 3)).astype(np.int64)
    return np.clip(base + np.random.default_rng(1000 + noise).integers(-2, 3, base.shape), 0, 255).astype(np.uint8)


class Made:
    """What the cloud maker was asked for, and what it answers: it keeps a cloud unless told to fail."""

    def __init__(self, store):
        self.store = store
        self.asked: list[tuple[str, np.ndarray, int]] = []
        self.fail = False
        self.pictures = True  # whether a cloud's picture is kept with it

    def __call__(self, camera, pixels, now, frames, thumb):
        self.asked.append((camera, pixels, frames))
        if self.fail:
            return None
        cloud = camera_clouds.Cloud(
            id=f"00000000-0000-7000-8000-{len(self.asked):012d}", camera=camera, at=now, points=10, share=0.8,
            size="approximate", floor_found=True, pose=tuple(np.eye(4).ravel()), focal=70.0, width=64,
            height=36, thumb=tuple(thumb),
        )  # fmt: skip
        self.store.replace(camera_clouds.Made(cloud, b"ply", pixels if self.pictures else None))
        return cloud


@pytest.fixture
def store(tmp_path):
    return camera_clouds.Store(camera_clouds.root(tmp_path))


@pytest.fixture
def made(store):
    return Made(store)


@pytest.fixture
def people():
    """The frames someone is in, by their first pixel's bytes."""
    return set()


@pytest.fixture
def clouds(store, made, people):
    def detect(pixels):
        return [(0.4, 0.5, 0.6, 1.0)] if pixels[0, 0].tobytes() in people else []

    return due.Clouds(
        store, made, detect, lambda pixels: pixels.mean() > 10, worker.thumbnail, submit=lambda job: job()
    )


def looks(clouds, frames, start=T, camera="hall"):
    return [clouds.look(camera, frame, start + i * STEP) for i, frame in enumerate(frames)]


def test_a_camera_with_no_cloud_gets_one_from_the_median_of_five_empty_pictures(clouds, made):
    frames = [picture(1, noise=n) for n in range(5)]
    assert looks(clouds, frames) == ["collecting"] * 4 + ["made"]
    (camera, pixels, count) = made.asked[0]
    assert (camera, count) == ("hall", 5)
    assert np.array_equal(pixels, np.median(np.stack(frames), axis=0).astype(np.uint8))


def test_someone_in_view_starts_the_count_of_empty_pictures_again(clouds, made, people, store):
    frames = [picture(1, noise=n) for n in range(8)]
    people.add(frames[2][0, 0].tobytes())
    assert looks(clouds, frames) == ["collecting", "collecting", "people"] + ["collecting"] * 4 + ["made"]
    assert made.asked[0][2] == 5
    assert np.array_equal(made.asked[0][1], np.median(np.stack(frames[3:]), axis=0).astype(np.uint8))


def test_a_camera_that_only_ever_has_someone_in_view_says_so(clouds, made, people, store):
    frame = picture(1)
    people.add(frame[0, 0].tobytes())
    assert looks(clouds, [frame] * 3) == ["people"] * 3
    assert made.asked == [] and store.notes()["hall"]["reason"] == "people"


def test_a_dark_picture_is_never_made_into_a_cloud(clouds, made, store):
    assert looks(clouds, [np.zeros((36, 64, 3), np.uint8)] * 6) == ["dark"] * 6
    assert made.asked == [] and store.notes()["hall"]["reason"] == "dark"


def test_a_view_that_moves_on_while_collecting_starts_the_count_again(clouds, made):
    frames = [picture(1, noise=n) for n in range(3)] + [picture(2, noise=n) for n in range(5)]
    assert looks(clouds, frames)[-1] == "made"
    assert np.array_equal(made.asked[0][1], np.median(np.stack(frames[3:]), axis=0).astype(np.uint8))


def test_a_cloud_stays_while_the_view_is_as_it_shows_it(clouds, made):
    looks(clouds, [picture(1, noise=n) for n in range(5)])
    later = T + due.REFRESH_MIN_S + 60
    assert looks(clouds, [picture(1, noise=n) for n in range(5, 15)], start=later) == ["fine"] * 10
    assert len(made.asked) == 1


def test_a_cloud_made_before_pictures_were_kept_is_made_again_once(clouds, made):
    made.pictures = False
    looks(clouds, [picture(1, noise=n) for n in range(5)])
    made.pictures = True
    later = T + due.REFRESH_MIN_S + 60
    assert (
        looks(clouds, [picture(1, noise=n) for n in range(5, 15)], start=later)
        == ["collecting"] * 4 + ["made"] + ["wait"] * 5
    )
    assert len(made.asked) == 2


def test_a_changed_view_gets_a_new_cloud_in_place_of_the_old_one(clouds, made, store):
    looks(clouds, [picture(1, noise=n) for n in range(5)])
    first = store.current("hall")
    later = T + due.REFRESH_MIN_S + 60
    assert looks(clouds, [picture(2, noise=n) for n in range(5)], start=later) == ["collecting"] * 4 + ["made"]
    assert store.current("hall") != first and len(made.asked) == 2


def test_a_cloud_isnt_made_again_right_after_one_was(clouds, made):
    looks(clouds, [picture(1, noise=n) for n in range(5)])
    soon = T + 60
    assert looks(clouds, [picture(2, noise=n) for n in range(5)], start=soon) == ["wait"] * 5
    assert len(made.asked) == 1


def test_a_try_that_made_nothing_is_tried_again_sooner_than_a_refresh(clouds, made):
    made.fail = True
    looks(clouds, [picture(1, noise=n) for n in range(5)])
    assert clouds.look("hall", picture(1, noise=9), T + due.RETRY_S - 1) == "wait"
    assert clouds.look("hall", picture(1, noise=9), T + due.RETRY_S + 30) == "collecting"


def test_each_camera_is_counted_and_made_by_itself(clouds, made):
    for n in range(5):
        clouds.look("hall", picture(1, noise=n), T + n * STEP)
        clouds.look("porch", picture(2, noise=n), T + n * STEP)
    assert [(camera, frames) for camera, _, frames in made.asked] == [("hall", 5), ("porch", 5)]
    assert np.array_equal(
        made.asked[1][1], np.median(np.stack([picture(2, noise=n) for n in range(5)]), axis=0).astype(np.uint8)
    )


def test_with_no_runtime_configured_nothing_is_collected_and_the_camera_says_why(store, made):
    clouds = due.Clouds(
        store, made, None, lambda p: True, worker.thumbnail, configured=lambda: False, submit=lambda j: j()
    )
    assert looks(clouds, [picture(1, noise=n) for n in range(6)]) == ["off"] * 6
    assert made.asked == [] and store.notes()["hall"]["reason"] == "no_runtime"
