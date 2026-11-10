"""tools/walkthrough_closure.py: a simulated walk of several minutes that returns to its start closes within the
stated tolerance (spatial/walkthrough.py `CLOSE_M`, `CLOSE_DEG`)."""

import importlib.util
import sys

import numpy as np
import pytest

from liberational import config as cfg
from liberational.spatial import walkthrough as W

_spec = importlib.util.spec_from_file_location("walkthrough_closure", cfg.ROOT / "tools" / "walkthrough_closure.py")
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
