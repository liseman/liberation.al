"""Learning surroundings (#257): the gate that keeps a frame likely to improve the map, and the bounded store."""

import json
import math

import numpy as np
import pytest

from liberational.spatial import surroundings_frames as sf


def view(angle_deg: float, tilt_deg: float = 0.0) -> np.ndarray:
    """A unit descriptor. Two with the same tilt are cos(angle difference) alike, so likeness is set by hand:
    cos 20 = 0.94 (a copy), cos 45 = 0.71 (the same view, changed), and cos 90 = 0 (another view)."""
    a, t = math.radians(angle_deg), math.radians(tilt_deg)
    return np.array([math.cos(a) * math.cos(t), math.sin(a) * math.cos(t), math.sin(t)], dtype=np.float32)


def picture(shade: int = 100) -> np.ndarray:
    return np.full((18, 32, 3), shade, dtype=np.uint8)


@pytest.fixture
def store(tmp_path):
    return sf.Store(tmp_path / "surroundings", most=5, per_view=2)


def offer(store, angle, sharp=100.0, now=0.0, tilt=0.0):
    return store.offer("front", picture(), view(angle, tilt), sharp, now=now)


def test_the_first_frame_is_kept_and_a_copy_of_it_is_dropped(store):
    first, kept = offer(store, 0)
    assert (first.keep, first.reason, kept.camera) == (True, "view", "front")
    again, none = offer(store, 20, now=5.0)
    assert (again.keep, again.reason, again.nearest, none) == (False, "copy", kept.id, None)
    assert again.alike == pytest.approx(math.cos(math.radians(20)), abs=1e-6)
    assert [f.id for f in store.kept] == [kept.id]


@pytest.mark.parametrize(
    ("angle", "reason"),
    [(90, "view"), (61, "view"), (59, "changed"), (33, "changed")],
    ids=["another view", "just under the same view", "just the same view", "just short of a copy"],
)
def test_a_frame_unlike_every_kept_frame_is_kept_with_why(store, angle, reason):
    offer(store, 0)
    decision, kept = offer(store, angle, now=5.0)
    assert (decision.keep, decision.reason) == (True, reason)
    assert len(store.kept) == 2
    assert kept.reason == reason


def test_a_much_sharper_copy_replaces_the_kept_frame_and_a_slightly_sharper_one_does_not(store):
    _, blurred = offer(store, 0, sharp=100.0)
    slightly, _ = offer(store, 10, sharp=149.0, now=5.0)
    assert slightly.keep is False
    much, sharp = offer(store, 10, sharp=150.0, now=10.0)
    assert (much.reason, much.replaces) == ("sharper", blurred.id)
    assert [f.id for f in store.kept] == [sharp.id]
    assert not store.path(blurred).exists()
    assert store.path(sharp).is_file()


def test_a_view_keeps_only_its_newest_looks(store):
    # One view, looking different each time: b is 0.77 alike to a, and c is 0.71 alike to a and 0.54 to b.
    _, a = offer(store, 0, now=1.0)
    _, b = offer(store, 40, now=2.0)
    _, c = offer(store, 0, now=3.0, tilt=45)
    assert c is not None
    assert [f.id for f in store.kept] == [b.id, c.id]  # two per view: the oldest went
    assert not store.path(a).exists()


def test_at_its_bound_the_store_drops_the_frame_most_like_another_and_the_older_of_the_pair(store):
    # Five views 72 degrees apart (0.31 alike), then a sixth that is 0.48 alike to the second: those two are the
    # most alike pair in the store, and the second is the older.
    frames = [offer(store, a, now=float(i))[1] for i, a in enumerate((0, 72, 144, 216, 288))]
    _, sixth = offer(store, 72, now=9.0, tilt=61.3)
    assert sixth is not None
    assert len(store.kept) == 5
    assert [f.id for f in store.kept] == [frames[0].id, frames[2].id, frames[3].id, frames[4].id, sixth.id]
    assert not store.path(frames[1]).exists()


def test_what_was_kept_is_read_back_by_the_next_process(store):
    _, a = offer(store, 0)
    _, b = offer(store, 90, now=5.0)
    again = sf.Store(store.folder)
    assert [f.id for f in again.kept] == [a.id, b.id]
    assert again.decide(view(90), 100.0).keep is False  # its descriptors came back with it


def test_a_row_with_a_field_of_an_earlier_version_is_still_kept(store):
    _, a = offer(store, 0)
    index = store.folder / "index.json"
    rows = json.loads(index.read_text())
    rows[0]["notes"] = {"mapped": True}
    index.write_text(json.dumps(rows))
    assert [f.id for f in sf.Store(store.folder).kept] == [a.id]


def test_a_frame_whose_picture_is_gone_is_not_kept(store):
    _, a = offer(store, 0)
    _, b = offer(store, 90, now=5.0)
    store.path(a).unlink()
    assert [f.id for f in sf.Store(store.folder).kept] == [b.id]


def test_a_removed_frame_leaves_no_files_and_its_view_can_be_kept_again(store):
    _, a = offer(store, 0)
    store.remove(a.id)
    assert list(store.folder.glob(f"{a.id}.*")) == []
    assert offer(store, 0, now=5.0)[0].keep is True


class FakeSession:
    """An ONNX session that answers each picture with its mean colour, to see what the describer feeds it."""

    def __init__(self):
        self.seen = []

    def get_inputs(self):
        return [type("Input", (), {"name": "image"})()]

    def run(self, _outputs, feed):
        x = feed["image"]
        self.seen.append(x)
        return [x.mean(axis=(2, 3)) * np.array([[3.0, 0.0, 4.0]], dtype=np.float32)]


def test_the_describer_feeds_the_model_a_normalized_rgb_square_and_returns_a_unit_vector():
    session = FakeSession()
    blue = np.zeros((90, 160, 3), dtype=np.uint8)
    blue[..., 0] = 255  # BGR: all blue
    d = sf.Describer(session=session)(blue)
    [x] = session.seen
    assert x.shape == (1, 3, 322, 322)
    # Red and green are 0 and blue is 1 before normalizing, in RGB order.
    assert x[0, :, 0, 0] == pytest.approx([-0.485 / 0.229, -0.456 / 0.224, (1 - 0.406) / 0.225], abs=1e-4)
    assert float(np.linalg.norm(d)) == pytest.approx(1.0, abs=1e-6)


def test_the_describer_says_how_to_get_a_missing_model(tmp_path):
    with pytest.raises(RuntimeError, match="mise run models"):
        sf.Describer(model=tmp_path / "megaloc.onnx")(picture())


def test_the_default_per_view_cap_keeps_distinct_viewpoints_across_one_room(tmp_path):
    # Ten distinct viewpoints around one room: each is 0.55 to 0.78 alike to the room's centre (SAME_VIEW..COPY_ALIKE)
    # and under COPY_ALIKE to every other viewpoint, so none is a copy and all ten stay kept.
    room_store = sf.Store(tmp_path / "room")
    centre = np.zeros(12, dtype=np.float32)
    centre[0] = 1.0
    kept_ids = []
    for i in range(10):
        vec = np.zeros(12, dtype=np.float32)
        vec[0] = 0.65
        vec[i + 1] = math.sqrt(1.0 - 0.65**2)  # 0.65 alike to centre, 0.4225 alike to every other viewpoint
        decision, kept = room_store.offer("walk", picture(80 + i), vec, 100.0, now=float(i))
        assert decision.keep and kept is not None
        kept_ids.append(kept.id)
    assert [f.id for f in room_store.kept] == kept_ids
