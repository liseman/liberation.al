"""tools/rebuild_place.py (#3286): the cameras put on the owner's measured spots."""

import importlib.util
import math
import sys

import numpy as np
import pytest

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import plan_import as pi

_spec = importlib.util.spec_from_file_location("rebuild_place", cfg.ROOT / "tools" / "rebuild_place.py")
assert _spec is not None and _spec.loader is not None
rp = importlib.util.module_from_spec(_spec)
sys.modules["rebuild_place"] = rp  # its dataclasses look their module up
_spec.loader.exec_module(rp)

LEVEL = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # plan = map on a levelled floor


def facing(heading_deg, x=1.0, y=1.0, z=1.5):
    """A camera at (x, y, z) looking along `heading_deg` (counterclockwise from +x) and 30 degrees down."""
    h, p = math.radians(heading_deg), math.radians(30.0)
    forward = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), -math.sin(p)])
    right = np.array([math.sin(h), -math.cos(h), 0.0])
    m = np.eye(4)
    m[:3, :3] = np.c_[right, np.cross(forward, right), forward]
    m[:3, 3] = (x, y, z)
    return places.CameraPose(tuple(float(v) for v in m.ravel()), 300.0, 640, 360)


def revision(cams):
    return places.Revision(
        "0" * 8 + "-0000-7000-8000-" + "0" * 12, "1" * 8 + "-0000-7000-8000-" + "0" * 12, 0.0, "t", (), cameras=cams
    )


def test_a_camera_goes_to_its_measured_spot_and_height_turned_to_the_facing_given():
    plan = pi.parse(
        'room r "Den" other poly 0,0 5,0 5,5 0,5\n'
        "x-camera a at 2,3 height 2.0 yaw 90\n"  # a page bearing: 90 is +x (the page's y grows down)
        "x-camera b at 4,1\n"
    )
    cams = {"a": facing(30.0), "b": facing(200.0, 3.0, 1.0, 1.2), "walk-0m00s": facing(0.0), "c": facing(10.0)}
    got = rp.spots(revision(cams), plan, LEVEL)
    a = np.asarray(got["a"].pose).reshape(4, 4)
    assert a[:3, 3] == pytest.approx([2.0, 3.0, 2.0])
    assert math.degrees(math.atan2(a[1, 2], a[0, 2])) == pytest.approx(0.0, abs=1e-6), "turned to face +x"
    assert -a[2, 2] == pytest.approx(math.sin(math.radians(30.0))), "its tilt kept"
    b = np.asarray(got["b"].pose).reshape(4, 4)
    assert b[:3, 3] == pytest.approx([4.0, 1.0, 1.2]), "no height given: its own kept"
    assert math.degrees(math.atan2(b[1, 2], b[0, 2])) == pytest.approx(-160.0), "no facing given: its own kept"
    assert got["c"] == cams["c"], "no spot: as registered"
    assert "walk-0m00s" not in got
