"""Tests for dropping kept views that don't register from their camera's spot (`tools/prune_surroundings.py`, #414)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from liberational.memory import places
from liberational.spatial import surroundings_frames as frames
from liberational.spatial import surroundings_map as sm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prune():
    spec = importlib.util.spec_from_file_location("prune_surroundings", ROOT / "tools" / "prune_surroundings.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prune_surroundings"] = module
    spec.loader.exec_module(module)
    return module


def _state(state: Path, owned: str, stray: str) -> sm.State:
    """A map step's state: `owned` in its plan's frame with views a1 and a2 of camera A and b1 of camera B, and
    `stray` learned by pairing alone with c1."""
    st = sm.State()
    st.places = {owned: {"plan_frame": True, "frame": "f", "rounds": 2}, stray: {"frame": "g", "rounds": 1}}
    pose = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
    st.frames = {
        "a1": {"place": owned, "pose": pose, "spot": True},
        "a2": {"place": owned, "pose": pose, "spot": True},
        "b1": {"place": owned, "pose": pose, "spot": True},
        "c1": {"place": stray, "pose": pose},
    }
    st.write(frames.root(state))
    return st


def test_pruning_drops_what_does_not_register_founds_its_place_again_and_does_nothing_twice(prune, tmp_path):
    state = tmp_path / "state"
    owned = places.revise(state, None, producer="test", now=1.0).place
    places.correct_plan(state, owned, 'plan "Home" walls 0.2/0.1\nroom r1 "Living" other rect 0,0 4x3\n', now=2.0)
    stray = places.revise(state, None, producer="test", now=3.0).place
    st = _state(state, owned, stray)
    views = [
        prune.View("a1", "A", 1.0, owned, "spot", True, None, ""),
        prune.View("a2", "A", 2.0, owned, "spot", False, "disagrees", ""),  # taken after camera A was moved
        prune.View("b1", "B", 1.0, owned, "spot", None, None, "its spot lacks tilt: not checked"),
        prune.View("c1", "C", 1.0, stray, "pairing", True, None, "no plan to check against"),
    ]

    without_stray = prune.plan(state, views, st)
    assert ([v.id for v in without_stray.drop], without_stray.refound, without_stray.remove) == (["a2"], (owned,), ())
    pruning = prune.plan(state, views, st, stray=True)
    assert pruning.remove == (stray,)  # nothing of the owner's: no plan, no name
    prune.apply(state, pruning)

    after = sm.State.read(frames.root(state))
    assert after.drop == ["a2"]  # the sampler deletes it from the kept store
    assert after.places == {}  # the owned place is founded again from a1 and b1; the stray one is gone
    assert after.frames == {}  # a1, b1, and c1 wait to be placed again
    assert not (places.root(state) / stray).exists()
    assert (places.root(state) / owned).is_dir()  # the owner's place and its plan stay

    again = prune.plan(state, [v for v in views if v.id != "a2"], after, stray=True)
    assert (again.drop, again.refound, again.remove) == ((), (), ())


def test_pruning_a_named_camera_drops_all_its_views_even_those_that_register(prune, tmp_path):
    state = tmp_path / "state"
    owned = places.revise(state, None, producer="test", now=1.0).place
    places.correct_plan(state, owned, 'plan "Home" walls 0.2/0.1\nroom r1 "Living" other rect 0,0 4x3\n', now=2.0)
    stray = places.revise(state, None, producer="test", now=3.0).place
    st = _state(state, owned, stray)
    views = [
        prune.View("a1", "A", 1.0, owned, "spot", True, None, ""),
        prune.View("a2", "A", 2.0, owned, "spot", True, None, ""),
        prune.View("b1", "B", 1.0, owned, "spot", True, None, ""),
    ]
    pruning = prune.plan(state, views, st, cameras=["A"])
    assert [v.id for v in pruning.drop] == ["a1", "a2"]
