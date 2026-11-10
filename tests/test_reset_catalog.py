"""pack-reset-catalog against a throwaway state directory, with a real temporary broker as the core service."""

import asyncio
import dataclasses
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import tomlkit

from continuum_broker import broker_binary
from liberational import reset_catalog

DEVICES = """# the house's devices
[[device]]
name = "cam0"
kind = "camera"

[device.video]
fps = 15  # the rig's rate

[[device]]
name = "mic0"
kind = "microphone"
"""


class House:
    """A service manager over one real broker (core) and a pretend dashboard, recording what was asked of it."""

    def __init__(self, state: Path, socket: Path, *, stuck: Sequence[str] = ()) -> None:
        self.state, self.socket, self.stuck = state, socket, set(stuck)
        self.broker: subprocess.Popen[bytes] | None = None
        self.dashboard = self.bodies = False
        self.retired = False  # an old worker still installed here after its code was deleted, and running
        self.broker_unit = True  # the broker's service is installed
        # A crash-looping broker: its service manager starts it again after this many stops (launchd's KeepAlive
        # rebinding the socket before bootout finishes), and reports this state while it runs.
        self.respawns = 0
        self.broker_state = "active"
        self.calls: list[str] = []

    def installed(self) -> list[str]:
        broker = ["continuum-broker"] if self.broker_unit else []
        return [*broker, "dashboard", *(["see-bodies"] if self.retired else [])]

    def core(self) -> set[str]:
        return {"continuum-broker"}

    def declared(self) -> set[str]:
        return {"continuum-broker", "dashboard"}

    def states(self) -> dict[str, str]:
        broker = self.broker is not None and self.broker.poll() is None
        out = {
            "continuum-broker": self.broker_state if broker else "inactive",
            "dashboard": "active" if self.dashboard else "inactive",
        }
        if self.retired:
            out["see-bodies"] = "active" if self.bodies else "inactive"
        return out

    def stop(self, names: Sequence[str]) -> None:
        for name in names:
            self.calls.append(f"stop {name}")
            if name in self.stuck:
                continue
            if name == "see-bodies":
                self.bodies = False
            elif name == "dashboard":
                self.dashboard = False
            elif self.broker is not None:
                self.broker.terminate()
                self.broker.wait(timeout=5)
                self.broker = None
                if self.respawns:
                    self.respawns -= 1
                    self.start(["continuum-broker"])
                    self.calls.pop()  # the service manager's own restart, not one the reset asked for

    def start(self, names: Sequence[str]) -> None:
        for name in names:
            self.calls.append(f"start {name}")
            if name == "see-bodies":
                self.bodies = True
            elif name == "dashboard":
                self.dashboard = True
            elif self.broker is None:
                catalog = self.state / "continuum" / "catalog.redb"
                catalog.parent.mkdir(parents=True, exist_ok=True)
                self.broker = subprocess.Popen(
                    [broker_binary(), "--socket", self.socket, "--catalog", catalog, "--tier0-classes", "16384x16"],
                    stderr=subprocess.DEVNULL,
                )
                deadline = 100
                while not reset_catalog.listening(self.socket) and deadline:
                    assert self.broker.poll() is None, "the broker exited at startup"
                    deadline -= 1
                    asyncio.run(asyncio.sleep(0.05))

    def close(self) -> None:
        if self.broker is not None:
            self.broker.kill()
            self.broker.wait()


@pytest.fixture
def house(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[House, reset_catalog.Plan]]:
    pytest.importorskip("continuum_py")
    state = tmp_path / "state"
    (state / "continuum").mkdir(parents=True)
    devices = tmp_path / "config" / "devices.toml"  # a throwaway checkout
    devices.parent.mkdir()
    devices.write_text(DEVICES)
    monkeypatch.setattr(reset_catalog.cfg, "OWNER_ROOT", tmp_path)
    with tempfile.TemporaryDirectory(prefix="rc-") as short:
        socket = Path(short) / "b.sock"
        host = House(state, socket)
        host.start(host.installed())
        plan = reset_catalog.Plan(state, tmp_path / "faces", devices, socket=socket)
        try:
            yield host, plan
        finally:
            host.close()


def derived_state(state: Path) -> None:
    """What the broker era left: media, a vector index, segments, and worker positions; and what must survive."""
    for path in (
        "continuum/tier/a.chunk",
        "continuum/tier2/b.chunk",
        "continuum/vector/index",
        "segments/x.arrows",
        "places/p/r.map",
        "surroundings/index.json",
        "cache/device_tallies.json",
    ):
        (state / path).parent.mkdir(parents=True, exist_ok=True)
        (state / path).write_bytes(b"old")
    (state / "see-zones.json").write_text("{}")
    (state / "identity.speech.cursor").write_text("7")
    (state / "inbox_skips.json").write_text("{}")
    (state / "voice_tags.offset").write_text("3")


def test_a_reset_deletes_the_derived_state_and_restarts_what_ran(house):
    host, plan = house
    derived_state(plan.state)
    said: list[str] = []
    reset_catalog.run(plan, host, say=said.append, socket_wait_s=10)
    state = plan.state
    for gone in (
        "continuum/tier",
        "continuum/tier2",
        "continuum/vector",
        "segments",
        "places",
        "surroundings",
        "see-zones.json",
        "identity.speech.cursor",
        "cache",
    ):
        assert not (state / gone).exists(), gone
    assert (state / "inbox_skips.json").exists() and (state / "voice_tags.offset").exists(), "local state is kept"
    devices = tomlkit.parse(plan.devices.read_text())
    assert devices["device"][0]["video"].unwrap() == {"fps": 15, "encoding": {"enabled": True}}
    assert "# the rig's rate" in plan.devices.read_text(), "the file's comments are kept"
    assert host.states() == {"continuum-broker": "active", "dashboard": "active"}
    assert host.calls[2:] == [
        "stop dashboard",
        "stop continuum-broker",
        "start continuum-broker",
        "start dashboard",
    ], "the dashboard stops before the broker, and starts again only once the broker is back"


def test_a_broker_that_wont_stop_means_nothing_is_deleted(tmp_path, house, monkeypatch):
    host, plan = house
    derived_state(plan.state)
    host.stuck = {"continuum-broker"}
    monkeypatch.setattr(reset_catalog, "STOP_RETRY_S", 0.05)
    with pytest.raises(reset_catalog.Refused, match="still running"):
        reset_catalog.run(plan, host, say=lambda line: None, stop_wait_s=0.3)
    assert (plan.state / "continuum/catalog.redb").exists() and (plan.state / "segments").exists()
    assert host.states()["dashboard"] == "active"


def test_an_unknown_camera_refuses_before_stopping_anything(house):
    host, plan = house
    plan = dataclasses.replace(plan, cameras=("cam9",))
    with pytest.raises(reset_catalog.Refused, match="cam9 is not"):
        reset_catalog.run(plan, host, say=lambda line: None)
    assert host.calls == ["start continuum-broker", "start dashboard"], "only the fixture's own start"


TWO_CAMERAS = """[[device]]
name = "hall"
kind = "camera"

[[device]]
name = "desk"
kind = "camera"
video = { format = "UYVY" }

[[device]]
name = "mic0"
kind = "microphone"
"""


def test_with_no_camera_named_the_reset_encodes_every_camera_capture_can(house):
    host, plan = house
    plan.devices.write_text(TWO_CAMERAS)  # no cam0: the old default named a camera this file doesn't have
    said: list[str] = []
    reset_catalog.run(plan, host, say=said.append, socket_wait_s=10)
    devices = tomlkit.parse(plan.devices.read_text())["device"]
    assert devices[0]["video"].unwrap() == {"encoding": {"enabled": True}}
    assert devices[1]["video"].unwrap() == {"format": "UYVY"}, "a raw camera is left alone"
    assert "Turned on H.264 for hall" in said
    assert "Left desk's encoding alone: it captures UYVY, and capture encodes only MJPEG" in said


def test_a_named_camera_is_the_only_one_encoded(house):
    host, plan = house
    plan.devices.write_text(
        '[[device]]\nname = "hall"\nkind = "camera"\n\n[[device]]\nname = "desk"\nkind = "camera"\n'
    )
    plan = dataclasses.replace(plan, cameras=("desk",))
    reset_catalog.run(plan, host, say=lambda line: None, socket_wait_s=10)
    devices = tomlkit.parse(plan.devices.read_text())["device"]
    assert "video" not in devices[0] and devices[1]["video"].unwrap() == {"encoding": {"enabled": True}}


def test_a_file_with_no_camera_turns_nothing_on(tmp_path):
    devices = tmp_path / "devices.toml"
    devices.write_text('[[device]]\nname = "mic0"\nkind = "microphone"\n')
    plan = reset_catalog.Plan(tmp_path / "state", tmp_path / "faces", devices)
    assert reset_catalog.cameras(plan) == []
    assert any(line.endswith("has none") for line in reset_catalog.describe(plan, NoServices()))


class NoServices:
    """A machine with nothing installed, for a dry run's plan."""

    def installed(self) -> list[str]:
        return []

    def core(self) -> set[str]:
        return set()

    def declared(self) -> set[str]:
        return set()

    def states(self) -> dict[str, str]:
        return {}

    def stop(self, names: Sequence[str]) -> None:
        raise AssertionError("a dry run stops nothing")

    def start(self, names: Sequence[str]) -> None:
        raise AssertionError("a dry run starts nothing")


def test_keep_places_keeps_the_places_and_resets_the_rest(house):
    host, plan = house
    derived_state(plan.state)
    plan = dataclasses.replace(plan, keep_places=True)
    assert plan.state / "places" not in reset_catalog.deleted(plan)
    reset_catalog.run(plan, host, say=lambda line: None, socket_wait_s=10)
    assert (plan.state / "places/p/r.map").read_bytes() == b"old"
    assert (plan.state / "surroundings/index.json").exists()  # the frames those maps were learned from stay with them
    assert not (plan.state / "segments").exists() and not (plan.state / "continuum/tier").exists()


def test_turning_on_h264_twice_changes_the_file_once(tmp_path):
    devices = tmp_path / "devices.toml"
    devices.write_text(DEVICES)
    assert reset_catalog.encode_camera(devices, "cam0") is True
    once = devices.read_text()
    assert reset_catalog.encode_camera(devices, "cam0") is False
    assert devices.read_text() == once


def test_a_camera_without_video_settings_gets_them(tmp_path):
    devices = tmp_path / "devices.toml"
    devices.write_text('[[device]]\nname = "cam1"\nkind = "camera"\n')
    reset_catalog.encode_camera(devices, "cam1")
    assert tomlkit.parse(devices.read_text())["device"][0]["video"].unwrap() == {"encoding": {"enabled": True}}


@pytest.mark.parametrize(
    ("text", "name"),
    [
        pytest.param(
            '[[device]]\nname = "cam0"\nkind = "camera"\nvideo = { format = "UYVY" }\n', "cam0", id="own-uyvy"
        ),
        pytest.param(
            '[video]\nformat = "UYVY"\n\n[[device]]\nname = "cam0"\nkind = "camera"\nvideo = { fps = 15 }\n',
            "cam0",
            id="inherits-uyvy",
        ),
        pytest.param(DEVICES, "mic0", id="microphone"),
    ],
)
def test_a_device_capture_cant_encode_is_left_alone(tmp_path, text, name):
    devices = tmp_path / "devices.toml"
    devices.write_text(text)
    assert reset_catalog.encode_camera(devices, name) is False
    assert devices.read_text() == text


def test_a_cameras_own_mjpeg_wins_over_the_files_raw_default(tmp_path):
    devices = tmp_path / "devices.toml"
    devices.write_text(
        '[video]\nformat = "UYVY"\n\n[[device]]\nname = "cam0"\nkind = "camera"\nvideo = { format = "MJPEG" }\n'
    )
    assert reset_catalog.encode_camera(devices, "cam0") is True
    assert tomlkit.parse(devices.read_text())["device"][0]["video"]["encoding"].unwrap() == {"enabled": True}


def test_a_reset_of_a_raw_camera_leaves_its_encoding_off_and_says_why(house):
    host, plan = house
    text = DEVICES.replace("fps = 15  # the rig's rate", 'fps = 15  # the rig\'s rate\nformat = "UYVY"')
    plan.devices.write_text(text)
    said: list[str] = []
    reset_catalog.run(plan, host, say=said.append, socket_wait_s=10)
    assert plan.devices.read_text() == text
    assert "Left cam0's encoding alone: it captures UYVY, and capture encodes only MJPEG" in said
    assert host.states() == {"continuum-broker": "active", "dashboard": "active"}


def test_the_dry_run_changes_nothing(house, capsys, monkeypatch):
    host, plan = house
    derived_state(plan.state)
    monkeypatch.setattr(reset_catalog, "Services", lambda: host)
    monkeypatch.setattr(reset_catalog.cfg, "path", lambda key: plan.state if key == "paths.state" else plan.faces)
    reset_catalog.main(["--keep-taught"])
    assert "Nothing was changed" in capsys.readouterr().out
    assert (plan.state / "continuum/tier/a.chunk").exists()
    assert host.calls == ["start continuum-broker", "start dashboard"]


@pytest.mark.parametrize(
    "looping",
    [
        pytest.param("activating", id="systemd-auto-restart"),
        pytest.param("spawn scheduled", id="launchd-keepalive"),
    ],
)
def test_a_crash_looping_broker_is_stopped_until_it_stays_down_then_started_on_the_fresh_catalog(
    house, monkeypatch, looping
):
    host, plan = house
    derived_state(plan.state)
    host.broker_state, host.respawns = looping, 2
    monkeypatch.setattr(reset_catalog, "STOP_RETRY_S", 0.05)
    reset_catalog.run(plan, host, say=lambda line: None, socket_wait_s=10)
    assert not (plan.state / "continuum/tier").exists() and not (plan.state / "segments").exists()
    assert host.calls[2:] == [
        "stop dashboard",
        "stop continuum-broker",
        "stop continuum-broker",
        "stop continuum-broker",
        "start continuum-broker",
        "start dashboard",
    ], "stopped until it stays down, and started again even though its manager never called it running"
    assert reset_catalog.listening(plan.broker_socket)


def test_a_broker_stopped_by_hand_is_started_again_after_the_reset(house):
    host, plan = house
    derived_state(plan.state)
    host.stop(["continuum-broker"])
    reset_catalog.run(plan, host, say=lambda line: None, socket_wait_s=10)
    assert host.states() == {"continuum-broker": "active", "dashboard": "active"}


def test_a_reset_refuses_before_stopping_anything_when_the_broker_isnt_installed(house):
    host, plan = house
    derived_state(plan.state)
    host.broker_unit = False
    with pytest.raises(reset_catalog.Refused, match="pack-services install"):
        reset_catalog.run(plan, host, say=lambda line: None)
    assert host.calls == ["start continuum-broker", "start dashboard"], "only the fixture's own start"
    assert (plan.state / "continuum/tier/a.chunk").exists()


def test_a_service_gone_from_the_code_is_stopped_but_never_started_again(house):
    host, plan = house
    host.retired = host.bodies = True
    said: list[str] = []
    reset_catalog.run(plan, host, say=said.append, socket_wait_s=10)
    assert "stop see-bodies" in host.calls and "start see-bodies" not in host.calls
    assert any("pack-services uninstall see-bodies" in line for line in said)


def taught_state(root: Path, state: Path) -> list[Path]:
    """Every kind of taught store, as files with something in them."""
    paths = [
        state / "knowledge.sqlite3",
        state / "knowledge_blobs/a.bin",
        state / "voiceprints.json",
        state / "voice_tags.jsonl",
        state / "clones/2.wav",
        state / "shots/x.jpg",
        state / "transcript_revisions.sqlite3",
        state / "transcript_revisions_qwen.sqlite3",
        state / "ingest.sqlite3",
        root / "faces/db.npz",
        root / "faces/sam/1.jpg",
        root / "voices/owner_ref.wav",
        root / "premove/people/brandon/1.jpg",
        root / "taught-old.zip",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"taught")
    return paths


def test_a_fresh_reset_deletes_every_taught_store_but_not_the_settings(house):
    host, plan = house
    root = plan.devices.parent.parent
    derived_state(plan.state)
    taught = taught_state(root, plan.state)
    settings = root / "pack.toml"
    settings.write_text('[owner]\nname = "Sam"\n')
    log = root / "logs/listen.log"
    log.parent.mkdir()
    log.write_text("Sam said hello\n")
    fresh = dataclasses.replace(plan, mode="fresh", voices=root / "voices", logs=root / "logs")
    reset_catalog.run(fresh, host, say=lambda line: None, socket_wait_s=10)
    assert [p for p in taught if p.exists()] == []
    assert log.exists() and log.read_text() == "", "logs are emptied, not removed, so open handles keep working"
    assert (root / "faces").is_dir() and not any((root / "faces").iterdir()), "the gallery is emptied, not removed"
    assert not (plan.state / "continuum/tier").exists() and not (plan.state / "segments").exists()
    assert settings.read_text() == '[owner]\nname = "Sam"\n'
    # People's inbox put-asides name face and voice ids, which start over with the gallery and voices.
    assert not (plan.state / "inbox_skips.json").exists()
    assert host.states() == {"continuum-broker": "active", "dashboard": "active"}


def test_a_keep_taught_reset_resets_the_catalog_and_leaves_the_taught_stores(house):
    host, plan = house
    root = plan.devices.parent.parent
    derived_state(plan.state)
    taught = taught_state(root, plan.state)
    keep = dataclasses.replace(plan, voices=root / "voices")
    reset_catalog.run(keep, host, say=lambda line: None, socket_wait_s=10)
    assert [p for p in taught if not p.exists()] == []
    assert not (plan.state / "continuum/tier").exists() and not (plan.state / "segments").exists()


def test_a_fresh_dry_run_names_each_taught_path(house, capsys, monkeypatch):
    host, plan = house
    root = plan.devices.parent.parent
    taught_state(root, plan.state)
    monkeypatch.setattr(reset_catalog, "Services", lambda: host)
    paths = {
        "paths.state": plan.state,
        "paths.faces": root / "faces",
        "paths.voices": root / "voices",
        "paths.logs": root / "logs",
    }
    monkeypatch.setattr(reset_catalog.cfg, "path", lambda key: paths[key])
    reset_catalog.main(["--fresh"])
    out = capsys.readouterr().out
    for shown in ("knowledge.sqlite3", "voiceprints.json", "faces/db.npz", "faces/sam", "premove", "taught-old.zip"):
        assert shown in out, shown
    assert (plan.state / "knowledge.sqlite3").exists(), "the dry run deletes nothing"
