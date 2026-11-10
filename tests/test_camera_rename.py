import fcntl
import json
import sqlite3
from pathlib import Path

import pytest

from liberational import camera_rename, knowledge
from liberational.memory import reports

NAMES = {"attic": "loft", "porch-cam": "sunroom"}
PLACE = "place-1"


def _record(id: str, collection: str, key: str | None, value: dict, t: int = 10) -> knowledge.Record:
    return knowledge.Record(id, collection, key, t, t, t, value)


@pytest.fixture
def state(tmp_path: Path) -> Path:
    """A state directory with one of each kind of record that names a camera, for cameras attic and porch-cam (to
    rename) and garage (kept)."""
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    store.put(
        [
            _record("z1", "zone.defined", "attic", {"cam": "attic", "source": "owner", "zones": [{"name": "Desk"}]}),
            _record("p1", "pet.sighting", "track-7", {"cam": "porch-cam", "pet": "Rex", "track_id": "7"}),
            _record("r1", "pet.reference", "ref-1", {"cam": "attic", "owner_label": {"cam": "attic"}}),
            _record("f1", "activity.feedback", "act-1", {"cam": "garage", "verdict": "right"}),
            _record("i1", "identity.record", "dana", {"faces": [{"cam": "attic", "cams": ["attic", "garage"]}]}),
            _record(
                "plan1",
                "place.plan",
                PLACE,
                {"place": PLACE, "plan": 'plan "Home"\nx-camera attic at 1,2 height 2.2\nx-camera garage at 3,4\n'},
            ),
            _record("e1", "event:people", None, {"name": "Dana"}),
        ]
    )
    store.close()

    rs = reports.Store(tmp_path / "reports.sqlite3")
    rs.set_meta(
        "layout",
        {
            "reports": [
                {"id": "walkbys", "kind": "walkbys", "arg": None, "title": "Walk-bys by camera", "cameras": ["attic"]},
                {"id": "walkbys:attic", "kind": "walkbys", "arg": "attic", "title": "Walk-bys on attic"},
                {"id": "conversations", "kind": "conversations", "arg": None, "title": "Conversations"},
            ]
        },
    )
    rs.put("walkbys", 1, "2026-01-01", True, 1.0, {"counts": {"attic": [1], "garage": [2]}})
    rs.put("walkbys", 1, "2026-01-02", True, 1.0, {"counts": {"garage": [2]}})

    rev = tmp_path / "places" / PLACE
    rev.mkdir(parents=True)
    (rev / "rev1.json").write_text(
        json.dumps({"cameras": {"attic": {"focal": 300}, "garage": {"focal": 200}}, "signals": {"camera": ["attic"]}})
    )
    (rev / "rev1.plan").write_text("plan\nx-camera porch-cam at 0,0\n")
    (tmp_path / "places" / "footprints.json").write_text(json.dumps({"attic": 5.0, "garage": 6.0}))

    (tmp_path / "see-zones.json").write_text(
        json.dumps({"cursor": {"pose.track": {"attic": 9}}, "pipelines": {"attic": {"definition": {"cam": "attic"}}}})
    )
    vector = tmp_path / "continuum" / "vector"
    vector.mkdir(parents=True)
    (vector / "segments.face.embedding.json").write_text(json.dumps({"revision": 3, "cursor": {"porch-cam": 4}}))

    for cam in ("attic", "garage"):
        slot = tmp_path / "segments" / "pose.track" / cam
        slot.mkdir(parents=True)
        (slot / "2026010100-0000-s0-5.arrows").write_bytes(b"rows")
    return tmp_path


def _value(state: Path, id: str) -> dict:
    record = knowledge.Store(state / "knowledge.sqlite3").get(id)
    assert record is not None
    return record.record.value


def _tree(state: Path) -> dict[str, object]:
    """Every file's content: a database's as its SQL dump, leaving out SQLite's side files and lock files, which a
    read touches."""
    out: dict[str, object] = {}
    for p in sorted(state.rglob("*")):
        if not p.is_file() or p.name.endswith(("-shm", "-wal", ".lock")):
            continue
        if p.suffix == ".sqlite3":
            db = sqlite3.connect(p)
            out[str(p.relative_to(state))] = list(db.iterdump())
            db.close()
        else:
            out[str(p.relative_to(state))] = p.read_bytes()
    return out


def test_rename_moves_owner_records_and_keeps_their_ids(state: Path):
    result = camera_rename.run(state, NAMES, dry=False)

    assert result.conflicts == []
    store = knowledge.Store(state / "knowledge.sqlite3")
    zone = store.query("zone.defined")
    assert [(r.record.id, r.record.key, r.record.value["cam"]) for r in zone] == [("z1", "loft", "loft")]
    assert _value(state, "p1")["cam"] == "sunroom"
    assert _value(state, "r1") == {"cam": "loft", "owner_label": {"cam": "loft"}}
    assert _value(state, "f1")["cam"] == "garage"
    assert _value(state, "i1") == {"faces": [{"cam": "loft", "cams": ["loft", "garage"]}]}
    assert _value(state, "plan1")["plan"] == 'plan "Home"\nx-camera loft at 1,2 height 2.2\nx-camera garage at 3,4\n'
    assert camera_rename.leftovers(state, NAMES) == []


def test_rename_moves_reports_layout_and_drops_stale_days(state: Path):
    camera_rename.run(state, NAMES, dry=False)

    rs = reports.Store(state / "reports.sqlite3")
    assert rs.meta("layout")["reports"][:2] == [
        {"id": "walkbys", "kind": "walkbys", "arg": None, "title": "Walk-bys by camera", "cameras": ["loft"]},
        {"id": "walkbys:loft", "kind": "walkbys", "arg": "loft", "title": "Walk-bys on loft"},
    ]
    assert list(rs.days("walkbys", 1)) == ["2026-01-02"]


def test_rename_moves_machine_state_that_follows_the_slots(state: Path):
    camera_rename.run(state, NAMES, dry=False)

    rev = json.loads((state / "places" / PLACE / "rev1.json").read_text())
    assert rev == {"cameras": {"loft": {"focal": 300}, "garage": {"focal": 200}}, "signals": {"camera": ["loft"]}}
    assert (state / "places" / PLACE / "rev1.plan").read_text() == "plan\nx-camera sunroom at 0,0\n"
    assert json.loads((state / "places" / "footprints.json").read_text()) == {"loft": 5.0, "garage": 6.0}
    assert json.loads((state / "see-zones.json").read_text()) == {
        "cursor": {"pose.track": {"loft": 9}},
        "pipelines": {"loft": {"definition": {"cam": "loft"}}},
    }
    cursor = json.loads((state / "continuum" / "vector" / "segments.face.embedding.json").read_text())
    assert cursor == {"revision": 3, "cursor": {"sunroom": 4}}
    slots = sorted(p.name for p in (state / "segments" / "pose.track").iterdir())
    assert slots == ["garage", "loft"]
    assert (state / "segments" / "pose.track" / "loft" / "2026010100-0000-s0-5.arrows").read_bytes() == b"rows"


def test_second_run_changes_nothing(state: Path):
    camera_rename.run(state, NAMES, dry=False)
    before = _tree(state)
    head = knowledge.Store(state / "knowledge.sqlite3").head()

    again = camera_rename.run(state, NAMES, dry=False)

    assert again.changes == []
    assert knowledge.Store(state / "knowledge.sqlite3").head() == head
    assert _tree(state) == before


def test_dry_run_reports_and_writes_nothing(state: Path):
    before = _tree(state)

    result = camera_rename.run(state, NAMES, dry=True)

    assert "knowledge zone.defined: 1 records" in result.changes
    assert "segments: 1 camera slots renamed" in result.changes
    assert _tree(state) == before
    assert camera_rename.leftovers(state, NAMES) == [
        "knowledge identity.record: 1 records",
        "knowledge pet.reference: 1 records",
        "knowledge pet.sighting: 1 records",
        "knowledge place.plan: 1 records",
        "knowledge zone.defined: 1 records",
        "reports layout",
    ]


def test_refuses_while_a_writer_holds_an_old_slot(state: Path):
    before = _tree(state)
    lock = state / "segments" / "pose.track" / "attic" / ".writer.lock"
    with lock.open("a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        result = camera_rename.run(state, NAMES, dry=False)

    assert result.conflicts == ["segments/pose.track/attic/.writer.lock is held: stop the services first"]
    assert result.changes == []
    assert _tree(state) == before


def test_refuses_when_the_new_slot_is_taken(state: Path):
    (state / "segments" / "pose.track" / "loft").mkdir()

    result = camera_rename.run(state, NAMES, dry=False)

    assert len(result.conflicts) == 1
    assert result.changes == []
    assert _value(state, "z1")["cam"] == "attic"


@pytest.mark.parametrize(
    "pairs",
    [
        pytest.param(["attic"], id="no-equals"),
        pytest.param(["attic=attic"], id="same-name"),
        pytest.param(["attic=loft", "attic=den"], id="renamed-twice"),
        pytest.param(["attic=loft", "den=loft"], id="one-new-name-twice"),
        pytest.param(["attic=loft", "loft=den"], id="chain"),
        pytest.param(["attic=.hidden"], id="bad-name"),
        pytest.param([], id="none"),
    ],
)
def test_parse_names_refuses(pairs: list[str]):
    with pytest.raises(ValueError):
        camera_rename.parse_names(pairs)


def test_config_edits_name_devices_and_every_camera_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    devices = tmp_path / "devices.toml"
    devices.write_text('[[device]]\nname = "attic"\nkind = "camera"\n[[device]]\nname = "garage"\nkind = "rtsp"\n')
    pack = tmp_path / "pack.toml"
    pack.write_text('[spatial]\ncamera_heights_m = { attic = 1.5, garage = 2.0 }\n[vision]\nplaces = ["attic: Desk"]\n')
    monkeypatch.setenv("PACK_CONFIG", str(pack))
    settings = {
        "spatial": {"camera_heights_m": {"attic": 1.5, "garage": 2.0}, "outdoor_cameras": ["garage"]},
        "vision": {"places": ["attic: Desk"], "actions_camera_fps": {"attic": 30.0}, "fps": 30},
    }

    edits = camera_rename.config_edits(NAMES, devices, settings)

    assert edits == [
        f'{devices}: [[device]] name = "attic"  ->  name = "loft"',
        f"{devices}: no device named porch-cam",
        "pack.toml: [spatial] camera_heights_m = {loft = 1.5, garage = 2.0}   (now from pack.toml)",
        'pack.toml: [vision] places = ["loft: Desk"]   (now from pack.toml)',
        "pack.toml: [vision] actions_camera_fps = {loft = 30.0}   (now from defaults.toml; add it to pack.toml)",
    ]


def test_amend_keeps_the_id_and_moves_the_record_to_a_new_change(tmp_path: Path):
    store = knowledge.Store(tmp_path / "k.sqlite3")
    store.put([_record("a", "zone.defined", "attic", {"cam": "attic"}), _record("b", "pet.not", None, {})])

    store.amend([_record("a", "zone.defined", "loft", {"cam": "loft"})])

    rows = store.query("zone.defined")
    assert [(r.record.id, r.record.key, r.record.value) for r in rows] == [("a", "loft", {"cam": "loft"})]
    assert [(c.operation, c.id) for c in store.changes(after=2)] == [("put", "a")]


def test_amend_refuses_a_missing_record_or_a_changed_time(tmp_path: Path):
    store = knowledge.Store(tmp_path / "k.sqlite3")
    store.put([_record("a", "zone.defined", "attic", {"cam": "attic"})])

    with pytest.raises(KeyError):
        store.amend([_record("nope", "zone.defined", "attic", {})])
    with pytest.raises(ValueError, match="key and value"):
        store.amend([_record("a", "zone.defined", "attic", {"cam": "loft"}, t=11)])
    assert store.query("zone.defined")[0].record.value == {"cam": "attic"}
