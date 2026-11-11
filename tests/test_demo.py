"""Demo mode (demo/, #17): the seeded household as the dashboard reads it, and a launcher that keeps off real data."""

import datetime as dt
from pathlib import Path

import pytest

from liberational import config as cfg
from liberational.dashboard import journal as journal_view
from liberational.demo import cli, dataset
from liberational.memory import journal, places
from liberational.people import pets, registry, transcript_view
from liberational.spatial import plan_scene

TODAY = dt.date(2026, 11, 9)
DAY = dt.date(2026, 11, 8)  # the demo's day is the day before
NOW = journal.day_bounds(TODAY)[0] + 9 * 3600


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
    assert (place.plan_source, len(plan_scene.rooms_of(place.plan or ""))) == ("machine", 5)


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


def test_the_demo_place_has_a_3d_map_from_two_cameras_that_say_which_points_are_theirs(seeded):
    (place,) = places.places(seeded)
    rev = place.revision
    assert sorted(rev.cameras) == ["kitchen", "study"]
    found = places.sources(seeded, rev)
    assert found is not None
    assert [v["camera"] for v in found["views"]] == ["kitchen", "study", "study"]
    assert min(found["points"]) > 0 and found["gaussians"] == found["points"]
    assert places.has_splat(seeded, rev) and places.splat_views(seeded, rev) is not None
    # the plan stays the place's, drawn in the map's own frame
    assert (place.plan_source, len(plan_scene.rooms_of(place.plan or ""))) == ("machine", 5)
