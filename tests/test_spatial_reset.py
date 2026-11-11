"""`pack-reset-places`: every place's machine-made state goes, and a place the owner's plan holds stays listed."""

import json

import pytest

import plans
from liberational import segment_log as S
from liberational.memory import places
from liberational.spatial import camera_clouds, positions, reset

T = 1_800_000_000.0
PLAN = plans.scene('plan "Home" walls 0.2/0.1\nroom den "Den" other rect 0,0 4x3\n')
OWNER_PLAN = plans.scene(
    'plan "Home" walls 0.2/0.1\nroom den "Den" other rect 0,0 4x3\nroom hall "Hall" hall rect 4,0 2x3\n'
)
LEVEL = (1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, -1.0, 0, 2.0, 0, 0, 0, 1.0)
CAM = places.CameraPose(LEVEL, 300.0, 640, 360, T)


@pytest.fixture
def house(tmp_path):
    """Two mapped places with a camera each and a splat: the owner corrected the first one's plan and only named the
    second. Beside them, the placer's marks, a camera's recorded-to mark, a walk import's files, and the position
    layer's streams with one other stream."""
    kept = places.revise(
        tmp_path, None, producer="mapper v1", plan=PLAN, map_bytes=b"cloud", map_format="ply", cameras={"cam0": CAM},
        signals={"wifi": ["aa:bb"], "camera": ["cam0"], "walkthrough": ["w1"]}, now=T,
    )  # fmt: skip
    places.write_splat(tmp_path, kept, master=b"ply", web=b"sog")
    places.correct_plan(tmp_path, kept.place, OWNER_PLAN, now=T + 10, base=kept.id)
    places.name(tmp_path, kept.place, "Home", now=T + 11)
    gone = places.revise(
        tmp_path, None, producer="mapper v1", plan=PLAN, map_bytes=b"cloud", map_format="ply", cameras={"cam1": CAM},
        now=T + 20,
    )  # fmt: skip
    places.name(tmp_path, gone.place, "Office", now=T + 21)
    root = places.root(tmp_path)
    marks = {"placed_to": T, "generations": {kept.place: 2, gone.place: 1}, "basis": {}}
    (root / positions.LAYER_MARKS).write_text(json.dumps(marks))
    (root / "footprints.json").write_text('{"cam0": 1}')
    (camera_clouds.root(tmp_path) / "hall").mkdir(parents=True)
    (camera_clouds.root(tmp_path) / "hall" / "cloud.json").write_bytes(b"{}")
    for stream in (S.POSITIONS, S.condensed(S.POSITIONS), S.PLACED, S.PLACED_PATHS, S.CAMERA_POSES, "speech.utterance"):
        (S.root(tmp_path) / stream / "slot").mkdir(parents=True)
        (S.root(tmp_path) / stream / "slot" / "part").write_bytes(b"rows")
    return tmp_path, kept, gone


def test_a_dry_run_lists_what_goes_and_what_stays_and_changes_nothing(house, capsys, monkeypatch):
    state, kept, gone = house
    monkeypatch.setattr(reset.cfg, "path", lambda key: state)
    before = sorted(p for p in state.rglob("*") if p.is_file() and "knowledge" not in p.name)
    reset.main([])
    said = capsys.readouterr().out
    assert sorted(p for p in state.rglob("*") if p.is_file() and "knowledge" not in p.name) == before
    assert "Nothing was changed" in said
    delete, keep = said.split("Keep:")
    assert f"Home ({kept.place}): 1 revisions, 1 maps, 1 splats, 1 camera registrations" in delete
    assert f"Office ({gone.place})" in delete and str(camera_clouds.root(state)) in delete
    assert str(S.root(state) / S.PLACED) in delete and "speech.utterance" not in delete
    assert f"Home ({kept.place}): your corrected plan, 2 rooms" in keep
    assert "not listed until mapped again (no corrected plan of yours): Office" in keep


def test_a_reset_deletes_the_machines_state_and_keeps_the_place_the_owners_plan_holds(house):
    state, kept, gone = house
    reset.run(reset.plan(state))
    (home,) = places.places(state)  # the place with only a name is no longer listed
    assert (home.id, home.name, home.plan, home.plan_source) == (kept.place, "Home", OWNER_PLAN, "owner")
    (stub,) = places.revisions(state, kept.place)
    assert stub.id != kept.id and home.plan_revision == stub.id
    assert (stub.map, stub.plan, stub.cameras, stub.floor) == (None, None, {}, None)
    assert stub.signals == {"wifi": ["aa:bb"]}  # where the place is, not what its old map was made from
    assert not places.has_splat(state, kept) and places.camera_places(state) == {}
    assert not (places.root(state) / gone.place).exists()
    assert not camera_clouds.root(state).exists() and not (places.root(state) / "footprints.json").exists()
    assert sorted(p.name for p in S.root(state).iterdir()) == ["speech.utterance"]
    # the kept place's placer slot stays the generation it was; nothing else of the marks survives
    assert json.loads((places.root(state) / positions.LAYER_MARKS).read_text()) == {"generations": {kept.place: 2}}
    assert [r.value["plan"] for r in places.plan_history(state, kept.place)] == [OWNER_PLAN]


def test_a_kept_places_plan_can_still_be_corrected_and_a_second_reset_keeps_it(house):
    state, kept, _ = house
    reset.run(reset.plan(state))
    places.correct_plan(state, kept.place, PLAN, now=T + 100)
    again = reset.plan(state)
    assert [(k.place.id, k.rooms) for k in again.kept] == [(kept.place, 1)]
    reset.run(again)
    assert [(p.id, p.plan) for p in places.places(state)] == [(kept.place, PLAN)]


def test_a_withdrawn_owner_plan_keeps_nothing_and_no_place_at_all_is_fine(house, tmp_path_factory):
    state, kept, _ = house
    places.correct_plan(state, kept.place, "", now=T + 50)  # withdrawn: the machine's plan held again
    reset.run(reset.plan(state))
    assert places.places(state) == []
    empty = tmp_path_factory.mktemp("empty")
    found = reset.plan(empty)
    assert (found.found, found.kept, found.targets) == ((), (), ())
    reset.run(found)
    assert places.places(empty) == []
