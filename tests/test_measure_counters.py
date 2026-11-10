"""The movement-track counter scores Continuum streams against made-up labels, without models."""

import importlib.util
import pathlib
import sys

import pytest

from liberational import segment_log
from test_movement_counts import cover, edge, fragment


def load_tool(name):
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


measure = load_tool("measure")
movement = load_tool("measure_movement_entries")


def entries(state, labels, *, start=0.0, end=20.0):
    window = measure.EntryWindow(
        "fixture", "cam0", start, end, tuple(measure.EntryEvent(kind, at) for kind, at in labels)
    )
    by_kind = measure.score_entries([window], state, movement.counter, tolerance=1.0)["per_camera"]["cam0"]["by_kind"]
    return {kind: (m["tp"], m["fp"], m["fn"]) for kind, m in by_kind.items()}


def test_supported_edges_are_scored_as_entries_and_exits(tmp_path, vision):
    cover(vision)
    fragment(vision, entry=edge(2.0), exit=edge(8.0, side="right"))
    assert entries(tmp_path, [("entry", 2.5), ("exit", 8.0), ("entry", 15.0)]) == {
        "entry": (1, 0, 1),
        "exit": (1, 0, 0),
    }


def test_a_track_born_in_view_predicts_nothing_and_its_gap_stays_unknown(tmp_path, vision):
    coverage = cover(vision)
    coverage["gaps"] = [[11.0, 13.0]]
    fragment(vision, start=4.0, end=10.0, entry=edge(4.0, status="unsupported", reason="track-birth"))
    result = movement.counter(camera="cam0", start=0.0, end=20.0, state=tmp_path)
    assert result["events"] == []
    assert result["supported_kinds"] == ["entry", "exit"]
    assert [(u["start"], u["end"]) for u in result["unknown"]] == [(11.0, 13.0)]


def test_unknown_spans_are_merged_and_clipped_to_the_window():
    merged = movement._unknown(
        [
            {"start": -5.0, "end": 2.0, "reasons": ["tracking-gap"]},
            {"start": 1.0, "end": 3.0, "reasons": ["track-occlusion"]},
            {"start": 6.0, "end": 6.0, "reasons": ["query-boundary"]},
            {"start": 18.0, "end": 30.0, "reasons": []},
        ],
        0.0,
        20.0,
    )
    assert merged == [
        {"start": 0.0, "end": 3.0, "reason": "track-occlusion; tracking-gap"},
        {"start": 18.0, "end": 20.0, "reason": ""},
    ]


def test_an_unreadable_movement_log_fails_instead_of_scoring_zero(tmp_path, monkeypatch):
    class Offline:
        def range(self, stream, start, end):
            raise ConnectionRefusedError("no broker")

    monkeypatch.setattr(segment_log, "at", lambda state: Offline())
    with pytest.raises(RuntimeError, match="ConnectionRefusedError"):
        movement.counter(camera="cam0", start=0.0, end=20.0, state=tmp_path)
