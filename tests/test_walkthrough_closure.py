"""tools/walkthrough_closure.py: a simulated walk of several minutes that returns to its start closes within the
stated tolerance (spatial/walkthrough.py `CLOSE_M`, `CLOSE_DEG`)."""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from liberational.spatial import walkthrough as W

_spec = importlib.util.spec_from_file_location(
    "walkthrough_closure", pathlib.Path(__file__).resolve().parents[1] / "tools" / "walkthrough_closure.py"
)
assert _spec is not None and _spec.loader is not None
wc = importlib.util.module_from_spec(_spec)
sys.modules["walkthrough_closure"] = wc
_spec.loader.exec_module(wc)


@pytest.mark.parametrize("one_lap", [False, True], ids=["laps", "one-lap"])
def test_the_simulated_walk_ends_where_it_began_facing_the_way_it_set_out(one_lap):
    true = wc.route(240, one_lap)
    assert abs(len(true) - 240) <= 20
    assert np.linalg.norm(true[-wc.TAIL, :3, 3] - true[0, :3, 3]) < wc.STEP_M
    assert true[-1, :3, :3] == pytest.approx(true[0, :3, :3])


@pytest.mark.parametrize("one_lap", [False, True], ids=["laps", "one-lap"])
def test_a_four_minute_walk_that_returns_to_its_start_closes_within_the_tolerance(one_lap):
    run = wc.simulate(240, seed=0, one_lap=one_lap)
    assert run["before_m"] > 5.0 and run["before_deg"] > W.GATE_MAX_DEG, "the drift is past what the gate lets in"
    assert run["kept"] == run["revisits"] > 0
    assert run["after_m"] <= W.CLOSE_M and run["after_deg"] <= W.CLOSE_DEG


def test_a_six_minute_single_loop_is_bent_less_than_the_tolerance_only_with_its_walls():
    held = wc.simulate(360, seed=0, one_lap=True)
    assert held["walls"] > 100
    assert held["bend_m"] <= W.BEND_SHARE * held["metres"]
    loose = wc.simulate(360, seed=0, one_lap=True, walls=False)
    assert loose["bend_m"] > 5 * W.BEND_SHARE * loose["metres"], "revisits alone leave its middle metres off"


def test_bend_ignores_where_the_walk_points_and_measures_what_a_rigid_move_cant_fix():
    true = wc.route(120)
    moved = np.einsum("ij,njk->nik", wc.upright(0.7, 3.0, -2.0), true)
    assert wc.bend(moved, true).max() == pytest.approx(0.0, abs=1e-9)
    kinked = true.copy()
    kinked[60, :3, 3] += [0.0, 0.0, 2.0]
    assert wc.bend(kinked, true).max() == pytest.approx(2.0, abs=0.05)
