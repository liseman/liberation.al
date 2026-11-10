"""Learning surroundings (#254): the live switch and the cameras it samples, and the Live tab's route for them."""

import io
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from liberational import config as cfg
from liberational.dashboard import surroundings as route
from liberational.spatial import surroundings
from liberational.spatial import surroundings_frames as frames

MANIFEST = """
[[device]]
name = "mic"
kind = "microphone"

[[device]]
name = "front"
kind = "camera"

[[device]]
name = "hall"
kind = "rtsp"

[[device]]
name = "spare"
kind = "rtsp"
enabled = false
"""


@pytest.fixture
def pack_toml(tmp_path, monkeypatch):
    path = tmp_path / "pack.toml"
    monkeypatch.setattr(cfg, "local_config", lambda: path)
    manifest = tmp_path / "devices.toml"
    manifest.write_text(MANIFEST)
    monkeypatch.setattr(cfg, "devices_file", lambda: manifest)
    return path


def test_learning_is_off_and_samples_the_first_camera_until_told_otherwise(pack_toml):
    switch = surroundings.read()
    assert (switch.on, switch.cameras, switch.chosen, switch.default) == (False, ("front",), (), "front")


def test_a_saved_change_is_read_at_once_without_a_restart(pack_toml):
    assert surroundings.read().on is False  # the process has now read its settings once
    surroundings.save(on=True, cameras=["hall"])
    switch = surroundings.read()
    assert (switch.on, switch.cameras) == (True, ("hall",))
    pack_toml.write_text("[spatial]\nlearn_surroundings = false\n")  # an edit by hand is followed too
    assert surroundings.read() == surroundings.Switch(False, ("front",), (), "front")


def test_saving_one_of_the_two_leaves_the_other(pack_toml):
    surroundings.save(cameras=["hall", "front"])
    surroundings.save(on=True)
    assert tomllib.loads(pack_toml.read_text())["spatial"] == {
        "learn_surroundings": True,
        "learn_cameras": ["front", "hall"],  # the manifest's order
    }
    surroundings.save(on=False)
    assert surroundings.read().cameras == ("front", "hall")


def test_an_empty_choice_goes_back_to_the_first_camera(pack_toml):
    surroundings.save(cameras=["hall"])
    assert surroundings.save(cameras=[]).cameras == ("front",)
    assert "learn_cameras" not in tomllib.loads(pack_toml.read_text()).get("spatial", {})


@pytest.mark.parametrize("name", ["attic", "spare", "mic"], ids=["unknown", "disabled", "a microphone"])
def test_a_camera_the_manifest_does_not_offer_is_refused_and_nothing_is_saved(pack_toml, name):
    with pytest.raises(surroundings.UnknownCamera, match=name):
        surroundings.save(on=True, cameras=[name])
    assert not pack_toml.exists()


def test_a_chosen_camera_that_was_removed_is_not_replaced_by_the_default(pack_toml):
    pack_toml.write_text('[spatial]\nlearn_surroundings = true\nlearn_cameras = ["attic"]\n')
    switch = surroundings.read()
    assert (switch.on, switch.cameras, switch.chosen) == (True, (), ("attic",))


def test_a_pack_toml_that_does_not_parse_never_turns_learning_on(pack_toml):
    pack_toml.write_text("[spatial]\nlearn_surroundings = tru")
    assert surroundings.read().on is False


def test_no_manifest_means_no_camera(pack_toml, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "devices_file", lambda: tmp_path / "missing.toml")
    switch = surroundings.read()
    assert (switch.cameras, switch.default) == ((), None)


def test_the_view_lists_every_camera_with_the_sampled_and_default_ones_marked(pack_toml):
    surroundings.save(on=True, cameras=["hall"])
    assert route.view(control=False) == {
        "ok": True,
        "control": False,
        "on": True,
        "cameras": [
            {"name": "front", "sampled": False, "default": True},
            {"name": "hall", "sampled": True, "default": False},
        ],
        "chosen": ["hall"],
    }


@pytest.mark.parametrize(
    "body",
    [{}, [], {"on": "yes"}, {"cameras": "front"}, {"cameras": [1]}, {"on": True, "extra": 1}],
    ids=["empty", "not an object", "on not a bool", "cameras not a list", "a name not text", "an unknown key"],
)
def test_a_malformed_change_is_refused(pack_toml, body):
    code, reply = route.change(body)
    assert (code, reply["ok"]) == (400, False)
    assert not pack_toml.exists()


def test_a_change_naming_an_unknown_camera_says_which(pack_toml):
    code, reply = route.change({"on": True, "cameras": ["attic"]})
    assert code == 400
    assert "attic" in reply["error"]
    assert surroundings.read().on is False


@dataclass
class Handler:
    """The parts of the dashboard's request handler the route uses."""

    control: bool = True
    same_origin: bool = True
    sent: list[tuple[int, dict[str, Any]]] = field(default_factory=list)

    def _can_control(self) -> bool:
        return self.control

    def _same_origin_json(self) -> bool:
        return self.same_origin

    def _json(self, reply: dict[str, Any], code: int = 200) -> None:
        self.sent.append((code, reply))


def test_this_machine_turns_it_on_and_gets_the_new_view(pack_toml):
    h = Handler()
    route._post(h, {"on": True})
    [(code, reply)] = h.sent
    assert (code, reply["on"], reply["control"]) == (200, True, True)
    assert surroundings.read().on is True


@pytest.mark.parametrize(
    "handler",
    [Handler(control=False), Handler(same_origin=False)],
    ids=["another device", "a cross-site page"],
)
def test_a_viewer_without_control_sees_the_switch_but_cannot_change_it(pack_toml, handler):
    route._post(handler, {"on": True})
    assert handler.sent[0][0] == 403
    assert surroundings.read().on is False
    route._get(handler)
    assert handler.sent[1][1]["on"] is False
    assert handler.sent[1][1]["control"] is handler.control


def test_the_status_route_answers_with_the_map_steps_summary_for_a_fresh_install(pack_toml, tmp_path):
    h = Handler()
    h.state = tmp_path / "state"  # ty: ignore[unresolved-attribute]  # the handler's state directory
    route._get_status(h)
    [(code, reply)] = h.sent
    assert code == 200
    assert reply == {
        "on": False, "sampling": [], "kept": 0, "waiting": 0, "parked": 0, "parked_cameras": {},
        "spot_suggestions": {}, "running": False,
        "step": None,
        "last_step": None,
        "cameras": {}, "places": {}, "frames": {},
    }  # fmt: skip


# ---- a camera's newest kept frame, as a small picture (#294)


@dataclass
class Sender(Handler):
    """A handler that also takes a picture: what `_send` or the header calls and the body came to."""

    command: str = "GET"
    status: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    wfile: io.BytesIO = field(default_factory=io.BytesIO)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.status = code

    def send_response(self, code: int) -> None:
        self.status = code

    def send_header(self, name: str, value: str) -> None:
        self.headers[name] = value

    def end_headers(self) -> None:
        pass


@pytest.fixture
def detector(tmp_path):
    """The sampler's person detector, as far as the route checks for it."""
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"")
    route.thumbnail.cache_clear()
    with cfg.override({"models": {"files": {"person_detector": {"path": str(model)}}}}):
        assert cfg.model_path("person_detector") == model
        yield model


def kept_frame(state: Path, camera: str, t: float, angle: float, covered: np.ndarray | None = None) -> frames.Kept:
    """Put a made-up 320 x 180 white frame in the kept store, as the sampler would have."""
    store = frames.Store(frames.root(state))
    descriptor = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    _, kept = store.offer(
        camera, np.full((180, 320, 3), 255, dtype=np.uint8), descriptor, 100.0, now=t, covered=covered
    )
    assert kept is not None
    return kept


def thumb(state: Path, frame_id: str) -> Sender:
    h = Sender()
    h.state = state  # ty: ignore[unresolved-attribute]  # the handler's state directory
    match = re.fullmatch(r"/surroundings/thumb/([0-9a-f]{32})\.jpg", f"/surroundings/thumb/{frame_id}.jpg")
    assert match is not None
    route._get_thumb(h, match)
    return h


def test_the_status_names_each_cameras_newest_kept_frame(pack_toml, tmp_path, detector):
    state = tmp_path / "state"
    kept_frame(state, "front", 100.0, 0.0)
    newer = kept_frame(state, "front", 200.0, 1.5)
    hall = kept_frame(state, "hall", 150.0, 3.0)
    h = Handler()
    h.state = state  # ty: ignore[unresolved-attribute]  # the handler's state directory
    route._get_status(h)
    assert h.sent[0][1]["frames"] == {"front": newer.id, "hall": hall.id}


def test_a_kept_frame_is_served_small_with_whoever_was_in_view_painted_over(tmp_path, detector):
    import cv2

    state = tmp_path / "state"
    covered = np.zeros((180, 320), dtype=bool)
    covered[:, :160] = True  # someone in the left half
    h = thumb(state, kept_frame(state, "front", 100.0, 0.0, covered).id)
    assert h.status == 200
    assert h.headers["Content-Type"] == "image/jpeg"
    assert "immutable" in h.headers["Cache-Control"]
    picture = cv2.imdecode(np.frombuffer(h.wfile.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert picture is not None
    assert picture.shape == (108, route.THUMB_WIDTH, 3)
    # the covered half is the flat grey, well inside its edge, and the rest is still the white frame
    assert np.abs(picture[:, :80].astype(int) - route.COVER).max() <= 6
    assert picture[:, 112:].min() >= 245


def test_no_picture_is_served_without_the_person_detector_or_for_a_frame_not_kept(pack_toml, tmp_path, detector):
    state = tmp_path / "state"
    kept = kept_frame(state, "front", 100.0, 0.0)
    assert thumb(state, "0" * 32).status == 404
    assert thumb(state, kept.id).status == 200
    detector.unlink()  # nothing marked who was in view, so no frame is shown
    assert thumb(state, kept.id).status == 404
    h = Handler()
    h.state = state  # ty: ignore[unresolved-attribute]  # the handler's state directory
    route._get_status(h)
    assert h.sent[0][1]["frames"] == {}


def test_a_frame_whose_mask_does_not_read_is_not_shown(tmp_path, detector):
    state = tmp_path / "state"
    kept = kept_frame(state, "front", 100.0, 0.0, np.ones((180, 320), dtype=bool))
    frames.Store(frames.root(state)).covered_path(kept).write_bytes(b"not a picture")
    assert thumb(state, kept.id).status == 404
