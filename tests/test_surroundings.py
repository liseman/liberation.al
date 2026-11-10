"""Learning surroundings (#254): the live switch and the cameras it samples, and the Live tab's route for them."""

import tomllib
from dataclasses import dataclass, field
from typing import Any

import pytest

from liberational import config as cfg
from liberational.dashboard import surroundings as route
from liberational.spatial import surroundings

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
        "on": False, "sampling": [], "kept": 0, "waiting": 0, "running": False, "step": None, "last_step": None,
        "cameras": {}, "places": {},
    }  # fmt: skip
