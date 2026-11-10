"""`pack-plan-convert` (spatial/plan_convert.py): each place's owner plan stored as a scene, with a dry run."""

import math

import numpy as np
import pytest

from liberational import config
from liberational.dashboard import places as view
from liberational.memory import places
from liberational.spatial import plan_convert, plan_scene
from liberational.spatial import plan_import as pi

T = 1_762_500_000.0

# A made-up flat: two rooms, a front door, a doorway between them, a window, and two camera spots.
DSL = """plan "Flat" walls 0.3/0.1
room hall "Hall" hall rect 0,0 3x4
room den "Den" living rect 3,0 5x4
window exterior>den at 5.5,0 w1.6
door exterior>hall at 0,2 w0.9 entrance
cased hall>den at 3,1 w1.2
x-source wall:w3 machine 0.4
x-camera porch at 0.4,0.5 height 2.2 yaw 135 fov 100
x-camera den-cam at 7.5,3.5
"""


@pytest.fixture
def state(tmp_path):
    with config.override({"paths": {"state": str(tmp_path)}}):
        yield tmp_path


@pytest.fixture
def home(state):
    rev = places.revise(state, None, producer="footprints-v1", plan=DSL, now=T)
    places.correct_plan(state, rev.place, DSL, now=T + 1)
    return rev.place


def held(state, place):
    found = places.get(state, place)
    assert found is not None and found.plan is not None
    return found


def test_a_dry_run_says_what_each_plan_holds_before_and_after_and_changes_nothing(state, home, capsys):
    plan_convert.main([])
    out = capsys.readouterr().out
    counts = "2 rooms, 7 walls, 1 door, 1 doorway, 1 window, 2 camera spots"
    assert f"before: {counts}" in out and f"after:  {counts}" in out
    assert "not carried: x-source wall:w3 machine 0.4" in out
    assert "Nothing was changed. Add --yes to convert." in out
    assert held(state, home).plan == DSL
    assert len(places.plan_history(state, home)) == 1


def test_converting_saves_the_scene_as_a_new_version_in_the_same_frame_and_keeps_the_old_one(state, home, capsys):
    frame = held(state, home).plan_revision
    plan_convert.main(["--yes"])
    now = held(state, home)
    assert plan_scene.is_scene(now.plan) and now.plan_source == "owner" and now.plan_revision == frame
    was, read = pi.parse(DSL), pi.parse(now.plan)
    assert [(r.id, r.name, r.kind, r.polygon) for r in read.rooms] == [
        (r.id, r.name, r.kind, r.polygon) for r in was.rooms
    ]
    assert read.cameras == was.cameras
    history = places.plan_history(state, home)
    assert [(r.value["source"], plan_scene.is_scene(r.value["plan"])) for r in history] == [
        ("owner", False),
        ("convert", True),
    ]
    assert places.spot_places(state) == {"porch": home, "den-cam": home}
    assert "saved as a scene" in capsys.readouterr().out


def test_running_it_again_changes_nothing(state, home, capsys):
    plan_convert.main(["--yes"])
    capsys.readouterr()
    plan_convert.main(["--yes"])
    out = capsys.readouterr().out
    assert "already a scene" in out and "Nothing to convert." in out
    assert len(places.plan_history(state, home)) == 2


def test_a_plan_that_cant_be_converted_is_reported_and_left_and_the_others_still_convert(state, home, capsys):
    bad = places.revise(state, None, producer="footprints-v1", plan=DSL, now=T + 2).place
    off_wall = 'plan "x"\nroom a "A" other rect 0,0 4x4\ndoor exterior>a at 9,9 w0.9\n'
    places.correct_plan(state, bad, off_wall, now=T + 3)
    plan_convert.main(["--yes"])
    out = capsys.readouterr().out
    assert "NOT converted: the door at 9,9 is on no wall" in out
    assert held(state, bad).plan == off_wall
    assert plan_scene.is_scene(held(state, home).plan)


def test_a_place_with_only_a_machine_plan_or_none_is_left_as_it_is(state, capsys):
    machine = places.revise(state, None, producer="footprints-v1", plan=DSL, now=T).place
    bare = places.revise(state, None, producer="cameras-v1", now=T + 1).place
    plan_convert.main(["--yes"])
    out = capsys.readouterr().out
    assert "only a machine plan, left as it is" in out and "no plan" in out
    assert held(state, machine).plan == DSL
    assert places.plan_history(state, machine) == [] and places.plan_history(state, bare) == []


def test_one_place_can_be_converted_alone(state, home, capsys):
    other = places.revise(state, None, producer="footprints-v1", plan=DSL, now=T + 2).place
    places.correct_plan(state, other, DSL, now=T + 3)
    plan_convert.main(["--yes", "--place", other])
    assert plan_scene.is_scene(held(state, other).plan)
    assert held(state, home).plan == DSL


def test_a_plan_drawn_on_its_own_page_has_the_page_converted_too(state, home):
    frame = held(state, home).plan_revision
    turned = pi.Move(math.pi / 2, (10.0, 0.0), False)
    view = {"plan": DSL, "turn": math.pi / 2, "shift": [10.0, 0.0], "mirror": False}
    places.correct_plan(state, home, pi.moved(DSL, turned), now=T + 5, base=frame, view=view)
    plan_convert.main(["--yes"])
    now = held(state, home)
    assert now.view is not None and plan_scene.is_scene(now.view.plan) and plan_scene.is_scene(now.plan)
    assert [r.polygon for r in pi.parse(now.view.plan).rooms] == [r.polygon for r in pi.parse(DSL).rooms]
    hall = pi.parse(now.plan).rooms[0].polygon  # still in the plan frame: (x, y) at (10 - y, x)
    np.testing.assert_allclose(hall, [(10, 0), (10, 3), (6, 3), (6, 0)], atol=1e-9)


def test_the_page_draws_each_plan_before_and_after_and_goes_where_it_is_told(state, home, tmp_path, capsys):
    page = tmp_path / "look.html"
    plan_convert.main(["--check", str(page)])
    text = page.read_text()
    assert (
        text.count("<svg") == 2
        and "Before (DSL): 2 rooms, 7 walls" in text
        and "After (scene): 2 rooms, 7 walls" in text
    )
    assert text.count(">Hall<") == 2 and text.count(">porch<") == 2
    # each drawing has its three openings: a door and a doorway as squares, a window as a circle with a blue edge
    assert text.count("<rect") == 4 and text.count('stroke="#2f6fed"') == 2
    assert "<code>x-source wall:w3 machine 0.4</code>" in text
    assert str(page) in capsys.readouterr().out
    assert held(state, home).plan == DSL


@pytest.mark.parametrize(
    ("change", "why"),
    [
        pytest.param(lambda p: p.replace("3x4", "3x4.5"), "room hall has a corner that moved", id="corner"),
        pytest.param(
            lambda p: p.replace("cased hall>den at 3,1 w1.2\n", ""), "it has 2 openings where the plan has 3", id="lost"
        ),
        pytest.param(lambda p: p.replace("w0.9", "w0.8"), "the door at 0,2 differs", id="narrower"),
        pytest.param(lambda p: p.replace("yaw 135", "yaw 140"), "its camera spots differ", id="turned"),
        pytest.param(
            lambda p: (
                p.replace('room den "Den" living', 'room lounge "Den" living')
                .replace("hall>den", "hall>lounge")
                .replace("exterior>den", "exterior>lounge")
            ),
            "its rooms differ",
            id="room-id",
        ),
        pytest.param(
            lambda p: p.replace("window exterior>den at 5.5,0 w1.6\ndoor", "door").replace(
                "cased hall>den at 3,1 w1.2\n", "cased hall>den at 3,1 w1.2\nwindow exterior>den at 5.5,0 w1.6\n"
            ),
            None,
            id="reordered",
        ),
    ],
)
def test_a_scene_that_doesnt_read_back_as_its_plan_is_caught(change, why):
    assert plan_convert.differs(pi.parse(DSL), pi.parse(pi.to_scene(change(DSL)).scene)) == why


LOBBY = DSL.replace('"Hall"', '"Lobby"')
WIDER = DSL.replace("3,0 5x4", "3,0 6x4")


@pytest.fixture
def versions(state, home):
    """A place with three DSL versions of its plan: the hall called Lobby, then Hall, then a wider den (current)."""
    places.correct_plan(state, home, LOBBY, now=T + 10)
    places.correct_plan(state, home, DSL, now=T + 20)
    places.correct_plan(state, home, WIDER, now=T + 30)
    return home


def test_earlier_versions_get_converted_copies_and_the_timeline_shows_only_scenes(state, versions, capsys):
    before = [(r.id, r.value) for r in places.plan_history(state, versions)]
    plan_convert.main([])
    assert "3 earlier versions to convert" in capsys.readouterr().out
    assert [(r.id, r.value) for r in places.plan_history(state, versions)] == before, "a dry run writes nothing"
    plan_convert.main(["--yes"])
    assert "3 earlier versions converted" in capsys.readouterr().out
    history = places.plan_history(state, versions)
    # the four DSL records are as they were, each with its copy right behind it; the newest's copy is the conversion
    kept = [(r.id, r.value) for r in history if not plan_scene.is_scene(r.value["plan"])]
    assert kept == before
    assert [plan_scene.is_scene(r.value["plan"]) for r in history] == [False, True] * 4
    assert [r.value.get("converted_from") for r in history[1::2]] == [before[0][0], before[1][0], before[2][0], None]
    listed = view.plan_versions(state, versions)[1]["versions"]
    assert [(v["format"], v["current"]) for v in listed] == [("scene", False)] * 3 + [("scene", True)]
    assert [v["summary"] for v in listed] == [
        "2 rooms, 2 camera spots",
        "Renamed Hall to Lobby",
        "Renamed Lobby to Hall",
        "Reshaped Den",
    ]
    assert held(state, versions).plan == history[-1].value["plan"], "the newest version still holds"


def test_a_converted_copy_of_an_earlier_version_can_be_looked_at_and_restored(state, versions):
    plan_convert.main(["--yes"])
    lobby = view.plan_versions(state, versions)[1]["versions"][1]
    code, reply = view.plan_as_scene(state, versions, lobby["id"])
    assert (code, reply["converted"]) == (200, False)
    assert plan_scene.rooms(reply["scene"])[0][1] == "Lobby"
    assert view.restore({"place": versions, "record": lobby["id"]}, True, state, now=T + 100)[0] == 200
    assert pi.parse(held(state, versions).plan).rooms[0].name == "Lobby"


def test_converting_the_history_twice_adds_nothing(state, versions, capsys):
    plan_convert.main(["--yes"])
    once = len(places.plan_history(state, versions))
    capsys.readouterr()
    plan_convert.main(["--yes"])
    assert "earlier version" not in capsys.readouterr().out
    assert len(places.plan_history(state, versions)) == once


def test_an_earlier_version_that_cant_convert_is_reported_and_the_rest_still_are(state, home, capsys):
    off_wall = 'plan "x"\nroom a "A" other rect 0,0 4x4\ndoor exterior>a at 9,9 w0.9\n'
    places.correct_plan(state, home, off_wall, now=T + 10)
    places.correct_plan(state, home, LOBBY, now=T + 20)
    places.correct_plan(state, home, DSL, now=T + 30)
    plan_convert.main(["--yes"])
    out = capsys.readouterr().out
    assert "2 earlier versions converted" in out
    assert "NOT converted: the version saved" in out and "the door at 9,9 is on no wall" in out
    listed = view.plan_versions(state, home)[1]["versions"]
    assert [v["format"] for v in listed] == ["scene", "dsl", "scene", "scene"]


def test_a_version_with_another_saved_in_the_same_instant_is_reported_not_squeezed_in(state, home, capsys):
    places.correct_plan(state, home, LOBBY, now=T + 10)
    places.correct_plan(state, home, WIDER, now=T + 10)  # the store puts it one nanosecond after
    places.correct_plan(state, home, DSL, now=T + 30)
    plan_convert.main(["--yes"])
    assert "another version was saved in the same instant" in capsys.readouterr().out
    assert held(state, home).plan is not None and pi.parse(held(state, home).plan).rooms[0].name == "Hall"
