"""tools/restore_owner_plan.py: which place it means, and what it says when there is nothing to restore
(tests/test_spatial_fuse.py covers the restore itself, `fuse.onto_owner_plan`)."""

import importlib.util
import pathlib

import pytest

from liberational.memory import places

_spec = importlib.util.spec_from_file_location(
    "restore_owner_plan", pathlib.Path(__file__).resolve().parents[1] / "tools" / "restore_owner_plan.py"
)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

T = 1_800_000_000.0


@pytest.fixture
def two(tmp_path):
    home = places.revise(tmp_path, None, producer="cameras-v1", now=T).place
    office = places.revise(tmp_path, None, producer="cameras-v1", now=T + 10).place
    places.name(tmp_path, home, "Home", now=T + 20)
    return home, office


def test_a_place_is_found_by_its_id_the_start_of_it_or_its_name(tmp_path, two):
    home, office = two
    found = [tool.find(tmp_path, wanted) for wanted in (home, "Home", "Place 2", office[:-2], "nowhere")]
    assert [p.id if p else None for p in found] == [home, home, office, office, None]


def test_the_start_of_an_id_that_two_places_share_names_neither(tmp_path, two):
    home, office = two
    shared = home[:4]
    assert office.startswith(shared), "ids made seconds apart share their first characters"
    assert tool.find(tmp_path, shared) is None


@pytest.mark.parametrize("write", [False, True], ids=["dry-run", "write"])
def test_a_place_with_nothing_to_restore_is_left_alone_and_says_so(tmp_path, two, write):
    home, _ = two
    before = places.revisions(tmp_path, home)
    wrote, said = tool.run(tmp_path, "Home", write, now=T + 30)
    assert not wrote and said == f"Home ({home}): no such place, or it has no fused map"
    assert places.revisions(tmp_path, home) == before


def test_a_place_nobody_has_is_refused(tmp_path, two):
    assert tool.run(tmp_path, "nowhere", True) == (False, "no one place matches 'nowhere'")
