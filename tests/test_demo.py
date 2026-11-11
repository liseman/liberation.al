"""Demo mode (demo/, #17): the seeded household as the dashboard reads it, and a launcher that keeps off real data."""

import datetime as dt
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from liberational import config as cfg
from liberational.dashboard import journal as journal_view
from liberational.demo import cli, dataset
from liberational.memory import journal, places, research
from liberational.people import pets, registry, transcript_view
from liberational.spatial import plan_scene, spots

TODAY = dt.date(2026, 11, 9)
DAY = dt.date(2026, 11, 8)  # the demo's day is the day before
NOW = journal.day_bounds(TODAY)[0] + 9 * 3600


def test_demo_configuration_cannot_resolve_install_files(tmp_path: Path) -> None:
    cli.prepare(tmp_path)
    inherited = dict(os.environ, PACK_ROOT="/real", PACK_OWNER_NAME="Private", PACK_AGENT_OPENCLAW_CONFIG="/real/key")
    env = cli.environment(tmp_path, cli.PORT, inherited)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
from liberational import config as cfg
root = Path(cfg.local_config()).parent
paths = [cfg.devices_file(), cfg.config_dir(), cfg.data_dir(), cfg.workdir(),
         cfg.owner_file('.env.phone'), cfg.path('agent.openclaw_config'), cfg.path('owner.voice_ref'),
         *[cfg.path('paths.' + key) for key in cfg.load()['paths']],
         *[cfg.model_path(key) for key in cfg.load()['models']['files']]]
assert all(path.resolve().is_relative_to(root) for path in paths), paths
assert cfg.owner_name() == 'Alice'
assert cfg.get('agent.openclaw_config') == ''
from liberational.vision import views
assert views.camera_ids() == ['front-room', 'kitchen', 'porch', 'study']
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dashboard_module_entrypoint_can_import_its_objects_page(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import runpy
from liberational import model_runtime
model_runtime.reporting = lambda *a, **kw: lambda fn: lambda: None
runpy.run_module('liberational.dashboard.server', run_name='__main__', alter_sys=True)
from liberational.dashboard.server import app_page
from liberational.dashboard import routes
assert routes.find('GET', '/audio/levels') is not None
from liberational.dashboard import assets, objects
assets.built_page = lambda name: ('<html><body>Demo</body></html>', True)
assets.stale = lambda: False
class Request:
    path = '/objects'
    def _send(self, code, content_type, body):
        assert code == 200
        assert b'Demo' in body
    def _json(self, *args):
        raise AssertionError(args)
assert objects.handle(Request(), 'GET')
""",
        ],
        env=cli.environment(tmp_path, cli.PORT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def seeded(tmp_path, memory):
    assert dataset.seed(tmp_path, memory, today=TODAY) == DAY
    return tmp_path


def test_journal_lists_the_demo_day_with_its_page(seeded):
    code, reply = journal_view.days(DAY - dt.timedelta(days=DAY.weekday()), seeded, NOW)

    assert code == 200
    assert [r["label"] for r in reply["items"] if r["key"] == DAY.isoformat()]

    code, reply = journal_view.hours(DAY, seeded, NOW)

    assert code == 200
    assert len(reply["items"]) == 12
    assert reply["items"][0]["label"] == "07:00"
    assert reply["items"][-1]["label"] == "22:00"
    assert reply["page"]["md"].startswith("# Sunday 2026-11-08\n")
    assert "## In short" in reply["page"]["md"]


def test_an_hour_lists_its_events(seeded):
    six_pm = journal.day_bounds(DAY)[0] + 18 * 3600

    code, reply = journal_view.events(six_pm, seeded)

    assert code == 200
    assert [r["text"][:5] for r in reply["items"]] == ["Alice", "Sam s", "Dana ", "Alice"]


def test_people_pet_and_place_are_seeded(seeded):
    assert sorted(p.display_name for p in registry.at(seeded).people()) == ["Alice", "Dana", "Sam"]
    assert [(p["name"], p["species"]) for p in pets.at(seeded).listing()] == [("Rex", "dog")]
    (place,) = places.places(seeded)
    assert place.name == "Demo house"
    assert (place.plan_source, len(plan_scene.rooms_of(place.plan or ""))) == ("owner", 6)


@pytest.mark.parametrize("camera", ["front-room", "kitchen", "porch", "study"])
def test_demo_camera_can_place_people_on_its_plan(seeded, camera):
    spot = spots.read(seeded)[camera]

    assert spot.missing == ()
    assert spot.pose is not None
    assert spot.aim is not None
    assert spot.aim.pitch == pytest.approx(28.63, abs=0.1)


def test_demo_inbox_has_readable_reports_and_a_waiting_question(seeded):
    store = research.Store(seeded)
    jobs = store.jobs()

    assert [j["status"] for j in jobs] == ["queued", "done", "done", "done"]
    report = store.get(jobs[1]["id"])
    assert report is not None
    assert "pasta" in report["result"]["answer"]
    assert report["result"]["evidence"]
    assert store.mark(report["id"], True)
    saved = store.get(report["id"])
    assert saved is not None and saved["read"] is True


def test_the_pet_has_a_training_set_with_one_example_unlike_the_rest(seeded):
    from liberational.people import pet_recognition

    listing = pets.at(seeded).listing()
    rows = pet_recognition.examples(seeded, listing[0]["id"], listing)
    assert len(rows) == dataset.EXAMPLES and {r["picture"] for r in rows} == {"kept"}
    assert rows[0]["track_id"] == f"demo:dog:{dataset.EXAMPLES}", "the odd one out comes first"
    assert rows[0]["similarity"] < 0.3 < rows[1]["similarity"]


def test_removing_the_odd_example_leaves_the_pet_recognized(seeded):
    # The tutorial's correction: taking out the one unlike the rest must not drop the pet below what recognition needs
    from liberational.people import pet_recognition

    listing = pets.at(seeded).listing()
    rows = pet_recognition.examples(seeded, listing[0]["id"], listing)
    assert pet_recognition.remove_examples(seeded, [rows[0]["track_id"]], listing[0]["id"], listing, NOW) == 1

    (dog,) = pet_recognition.view(seeded, listing, NOW)["recognition"]
    assert dog["on"], dog["reason"]


def test_face_rows_are_seeded_only_into_the_gallery_this_process_writes(tmp_path, monkeypatch):
    from liberational.vision import faces

    monkeypatch.setattr(faces, "DB", tmp_path / "faces" / "db.npz")
    monkeypatch.setattr(faces, "LEARNED_DB", tmp_path / "faces" / "learned.npz")
    assert dataset.seed_faces(tmp_path / "elsewhere", NOW) == 0
    assert not (tmp_path / "faces").exists() and not (tmp_path / "elsewhere").exists()
    assert dataset.seed_faces(tmp_path / "faces", NOW) == 3 * dataset.EXAMPLES
    rows = faces.examples("Dana")
    assert len(rows) == dataset.EXAMPLES and all(r["picture"] for r in rows)
    assert rows[0]["similarity"] < 0.3 < rows[1]["similarity"]
    assert dataset.seed_faces(tmp_path / "faces", NOW) == 0, "a gallery that is there is left as it is"


def test_the_transcript_is_said_by_the_demo_people(tmp_path, memory, speech):
    dataset.seed(tmp_path, memory, today=TODAY, say=speech.say)
    lines = speech.range(0, 2e9)
    said = transcript_view.View(tmp_path).named_speakers(lines)

    assert len(lines) == len(dataset.TRANSCRIPT)
    assert {e["speaker"] for e, _ in said} == {"Alice", "Dana", "Sam"}
    assert [e["speaker"] for e, _ in said] == [line.speaker for line in dataset.TRANSCRIPT]
    assert all(journal.day_bounds(DAY)[0] <= e["ts_start"] < journal.day_bounds(DAY)[1] for e in lines)


def test_environment_points_every_path_into_the_demo_dir(tmp_path):
    root = tmp_path / "d"
    real = {"PACK_PATHS_STATE": "/real/state", "PACK_STATE": "/real/state", "OLLAMA_HOST": "127.0.0.1:11434"}

    env = cli.environment(root, cli.PORT, base=real)

    path_vars = [k for k in env if k.startswith("PACK_PATHS_")]
    assert sorted(path_vars) == sorted(f"PACK_PATHS_{k.upper()}" for k in cfg.load()["paths"])
    assert all(Path(env[k]).parent == root for k in path_vars)
    assert env["PACK_STATE"] == str(root / "state")
    assert env["PACK_CONFIG"] == str(root / "pack.toml")
    assert env["PACK_OLLAMA_URL"] == cli.NO_OLLAMA
    assert env["PACK_PORTS_GATEWAY"] == str(cli.NO_GATEWAY)
    assert "OLLAMA_HOST" not in env
    assert env["CONTINUUM_SOCKET"] == str(root / "state" / "continuum" / "broker.sock")


def test_the_gateway_is_sent_the_socket_the_broker_listens_on(tmp_path):
    root = tmp_path / "d"

    command = cli.gateway_command(cli.environment(root, cli.PORT, base={}), 9000)

    assert Path(command[command.index("--broker-socket") + 1]) == root / "state" / "continuum" / "broker.sock"


def test_a_long_demo_dir_sends_the_gateway_the_short_socket_too(tmp_path):
    root = tmp_path / ("x" * 120)

    env = cli.environment(root, cli.PORT, base={})
    command = cli.gateway_command(env, 9000)
    sock = Path(command[command.index("--broker-socket") + 1])

    assert str(sock) == env["CONTINUUM_SOCKET"]
    assert not sock.is_relative_to(root)
    sock.parent.rmdir()


def test_a_socket_path_too_long_for_a_socket_goes_to_a_short_dir(tmp_path):
    root = tmp_path / ("x" * 120)

    sock = Path(cli.environment(root, cli.PORT, base={})["CONTINUUM_SOCKET"])

    assert not sock.is_relative_to(root)
    assert len(str(sock).encode()) <= cli.SOCKET_MAX
    sock.parent.rmdir()


@pytest.mark.parametrize("inside", ["data", "data/demo"])
def test_a_demo_dir_in_the_real_data_dir_is_refused(inside):
    with pytest.raises(SystemExit, match="real data directory"):
        cli.check_dir(cfg.workdir() / inside)


def test_a_dir_holding_something_else_is_refused(tmp_path):
    (tmp_path / "notes.txt").write_text("mine")

    with pytest.raises(SystemExit, match="isn't a demo directory"):
        cli.check_dir(tmp_path)


@pytest.mark.parametrize("marker", [False, True], ids=["empty", "seeded"])
def test_an_empty_or_seeded_dir_is_accepted(tmp_path, marker):
    if marker:
        (tmp_path / cli.SEEDED).write_text("pack-demo\n")

    cli.check_dir(tmp_path)


@pytest.fixture
def listening():
    """A socket listening on a free port, as another program would: its port."""
    import socket

    with socket.socket() as other:
        other.bind(("0.0.0.0", 0))
        other.listen()
        yield other.getsockname()[1]


def test_an_asked_port_that_is_taken_is_refused_before_anything_starts(listening):
    with pytest.raises(SystemExit, match=f"port {listening} is in use"):
        cli.choose_port(listening)


def test_an_asked_port_that_is_free_is_used():
    import socket

    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 0))
        free = probe.getsockname()[1]

    assert cli.choose_port(free) == free


def test_the_default_port_is_none_of_the_packs(monkeypatch):
    monkeypatch.setattr(cli, "port_free", lambda port: True)

    port = cli.choose_port(None)

    assert port == cli.PORT
    assert port not in {8765, 8766, 8767, 8776, 8790, 8081, 8082, 11435, 5173}


def test_a_taken_default_moves_to_a_free_port_the_pack_does_not_use(monkeypatch, listening, capsys):
    monkeypatch.setattr(cli, "PORT", listening)

    port = cli.choose_port(None)

    assert port != listening
    assert port not in cli.reserved_ports()
    assert cli.port_free(port)
    assert f"takes port {port}" in capsys.readouterr().out


def test_the_default_port_skips_one_this_install_gave_a_service(monkeypatch):
    monkeypatch.setattr(cli, "port_free", lambda port: True)
    monkeypatch.setattr(cli, "any_free_port", lambda reserved: 40000)
    with cfg.override({"ports": {"vlm": cli.PORT}}):
        assert cli.choose_port(None) == 40000


def test_demo_restart_discards_edits_without_following_external_links(tmp_path: Path) -> None:
    root, outside = tmp_path / "demo", tmp_path / "outside"
    cli.prepare(root)
    (root / cli.SEEDED).write_text("pack-demo")
    outside.mkdir()
    (outside / "keep").write_text("untouched")
    (root / "faces").symlink_to(outside, target_is_directory=True)
    (root / "state").mkdir()
    (root / "state" / "edit").write_text("changed")
    (root / "uploaded.wav").write_bytes(b"demo recording")
    cli.reset(root)
    cli.prepare(root)
    assert (outside / "keep").read_text() == "untouched"
    assert not (root / "faces").exists()
    assert not (root / "state").exists()
    assert not (root / "uploaded.wav").exists()
    assert not (root / cli.SEEDED).exists()
    assert (root / "pack.toml").read_text() == cli.CONFIG


def test_demo_plan_places_each_camera_in_its_room() -> None:
    spots = {
        name: at
        for name, at, height, yaw, fov in plan_scene.spots(plan_scene.load(dataset.PLAN))
        if height and yaw is not None and fov
    }
    assert set(spots) == {"front-room", "kitchen", "porch", "study"}
    assert 4 < spots["front-room"][0] < 9 and 0 < spots["front-room"][1] < 5
    assert 0 < spots["kitchen"][0] < 4 and 0 < spots["kitchen"][1] < 3.5
    assert -4 < spots["porch"][0] < 0 and 3.5 < spots["porch"][1] < 6.5
    assert 4 < spots["study"][0] < 9 and 5 < spots["study"][1] < 8.5


def test_demo_clouds_combine_at_the_editor_spots_with_a_refined_doorway_pair(seeded: Path) -> None:
    import numpy as np

    from liberational.dashboard import places as place_view
    from liberational.demo import household
    from liberational.spatial import camera_clouds, cloud_scene, ply

    store = camera_clouds.Store(camera_clouds.root(seeded))
    pose = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 2.65, 0.0, 0.0, 0.0, 1.0)
    points = np.array([[0.0, 0.5, 0.0], [0.2, 0.5, 0.0], [0.0, 0.5, 1.0]])
    for camera in ("front-room", "kitchen", "porch", "study"):
        cloud = camera_clouds.Cloud(str(uuid.uuid7()), camera, NOW, 3, 1.0, "approximate", True, pose, 300, 64, 36)
        store.replace(
            camera_clouds.Made(
                cloud, ply.dumps(points, np.full((3, 3), 100, np.uint8)), np.zeros((36, 64, 3), np.uint8)
            )
        )
    household.camera_pairs(seeded)
    place = places.places(seeded)[0]
    spots = cloud_scene.spots_of(seeded)
    result = cloud_scene.scene(store, place.id, spots)
    assert {camera.camera: camera.how for camera in result.cameras} == {
        "front-room": "adjusted",
        "kitchen": "adjusted",
        "porch": "spot",
        "study": "spot",
    }
    assert all(camera.placed for camera in result.cameras)
    assert result.pairs[0].state == "tightened"
    code, combined = place_view.scene(seeded, place.id, combined=True)
    assert code == 200 and combined["revision"]
    assert len(combined["cameras"]) == 4
    assert set(plan_scene.spot_names(combined["plan"])) == set(spots)
    code, editor = place_view.plan_as_scene(seeded, place.id)
    assert code == 200
    assert {name for name, *_ in plan_scene.spots(editor["scene"])} == set(spots)
