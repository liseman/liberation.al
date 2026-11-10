"""Dry runs of spatial writes (#3286, #3289): a throwaway state that never lets a write reach the live one, and the
measures the tools print."""

import math
import sqlite3
from dataclasses import replace

import pytest

from liberational.spatial import dryrun, positions


@pytest.fixture
def live(tmp_path):
    state = tmp_path / "live"
    place = state / "places" / "p1"
    place.mkdir(parents=True)
    (place / "r1.json").write_text("{}")
    (state / "places" / "seats.json").write_text('{"den": {"nights": 1}}')
    (state / "places" / "p2").mkdir()
    for stream in ("camera.pose", "position.sample"):
        (state / "segments" / stream / "den").mkdir(parents=True)
        (state / "segments" / stream / "den" / "rows").write_text("live")
    (state / "places.lock").write_text("")
    with sqlite3.connect(state / "knowledge.sqlite3") as db:
        db.execute("create table t (x)")
        db.execute("insert into t values (1)")
    return state


def test_writes_to_a_dry_runs_state_never_reach_the_live_one(live, tmp_path):
    shadow = dryrun.overlay(live, tmp_path / "dry", "p1")
    (shadow / "places" / "p1" / "r2.json").write_text("{}")  # a revision written
    (shadow / "places" / "p1" / "r1.json").unlink()  # a map pruned
    (shadow / "places" / "seats.json").write_text("{}")  # a refusal count reset
    (shadow / "segments" / "camera.pose" / "den" / "rows").write_text("dry")  # a pose observation appended
    with sqlite3.connect(shadow / "knowledge.sqlite3") as db:
        db.execute("insert into t values (2)")  # a plan or a held lens saved
    assert sorted(p.name for p in (live / "places" / "p1").iterdir()) == ["r1.json"]
    assert (live / "places" / "seats.json").read_text() == '{"den": {"nights": 1}}'
    assert (live / "segments" / "camera.pose" / "den" / "rows").read_text() == "live"
    with sqlite3.connect(live / "knowledge.sqlite3") as db:
        assert db.execute("select count(*) from t").fetchone() == (1,)
    assert not (shadow / "places.lock").exists(), "no live lock is taken through the dry run"


def test_a_dry_run_reads_the_rest_of_the_live_state(live, tmp_path):
    shadow = dryrun.overlay(live, tmp_path / "dry", "p1")
    assert (shadow / "segments" / "position.sample" / "den" / "rows").read_text() == "live"
    assert (shadow / "places" / "p2").resolve() == (live / "places" / "p2").resolve()
    assert (shadow / "segments" / "camera.pose" / "den" / "rows").read_text() == "live", "its own copy, as it was"


def row(camera, t, x, y, entity="person:a"):
    return positions.Position(
        f"{camera}{t}", t, t, entity, None, None, "fixed-camera", 1.0, "p1", None, None, x, y, camera, 0.5, 0.9, "r1"
    )


def test_two_cameras_agree_by_how_far_apart_they_put_one_entity_at_the_same_moment():
    found = [
        row("a", 10.0, 0.0, 0.0),
        row("b", 10.2, 0.3, 0.4),  # 0.5 m apart, 0.2 s later
        row("a", 20.0, 1.0, 1.0),
        row("b", 21.0, 1.0, 1.0),  # a second apart: not the same moment
        row("a", 30.0, 0.0, 0.0),
        row("b", 30.1, 0.0, 1.0, entity="person:b"),  # someone else
        row("a", 40.0, 2.0, 2.0),
        row("b", 39.9, 2.0, 3.5),  # 1.5 m apart
    ]
    got = dryrun.agreement(found, "a", "b")
    assert got.pairs == 2
    assert got.median == pytest.approx(1.0)
    assert got.p90 == pytest.approx(1.4)


def test_cameras_that_never_place_one_entity_together_have_no_agreement():
    got = dryrun.agreement([row("a", 1.0, 0.0, 0.0), row("b", 1.0, 0.0, 0.0, entity=None)], "a", "b")
    assert got.pairs == 0 and math.isnan(got.median)


def test_rows_are_counted_by_room_with_no_point_and_out_of_frame_apart():
    rows = [row("a", 1.0, 0.0, 0.0), row("a", 2.0, None, None)]
    rows[0] = replace(rows[0], room="Den")
    rows.append(replace(row("a", 3.0, 1.0, 1.0), in_frame=False))
    rows.append(row("a", 4.0, 5.0, 5.0))
    assert dryrun.rooms(rows) == {"Den": 1, dryrun.NO_POINT: 1, dryrun.OUT_OF_FRAME: 1, dryrun.OUTSIDE: 1}
