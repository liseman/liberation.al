"""Entry/exit measurements: independent matching oracle and isolated counter contracts."""

import importlib.util
import itertools
import json
import math
import pathlib
import sys
import types

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

_spec = importlib.util.spec_from_file_location(
    "measure_entries_under_test", pathlib.Path(__file__).resolve().parents[1] / "tools/measure.py"
)
assert _spec is not None and _spec.loader is not None
measure = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = measure
_spec.loader.exec_module(measure)

WINDOW = {"id": "window", "camera": "cam0", "start": 0.0, "end": 100.0, "events": []}


def load_rows(directory, *rows):
    (directory / "entries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return measure.load(directory, "entries")


def counter_for(events, *, unknown=(), supported=("entry", "exit")):
    def counter(*, camera, start, end, state):
        return {"events": events, "unknown": list(unknown), "supported_kinds": list(supported), "diagnostics": {}}

    return counter


def event(at, kind="entry"):
    return {"kind": kind, "at": at}


def metrics(result, camera="cam0", kind="entry"):
    return result["per_camera"][camera]["by_kind"][kind]


def test_tolerance_boundary_and_duplicate_predictions_are_one_to_one(tmp_path):
    windows = load_rows(tmp_path, dict(WINDOW, events=[event(10), event(30)]))
    result = measure.score_entries(windows, tmp_path, counter_for([event(12), event(12), event(32.001)]), tolerance=2)
    score = metrics(result)
    assert (score["tp"], score["fp"], score["fn"]) == (1, 2, 1)
    assert score["precision"] == pytest.approx(1 / 3)
    assert score["recall"] == 0.5


def test_camera_and_kind_boundaries_prevent_cross_matching(tmp_path):
    windows = load_rows(
        tmp_path,
        dict(WINDOW, events=[event(10), event(20, "exit")]),
        dict(WINDOW, id="second", camera="cam1", events=[event(50)]),
    )

    def counter(*, camera, start, end, state):
        return {
            "supported_kinds": ["entry", "exit"],
            "events": [event(10, "exit"), event(50)] if camera == "cam0" else [event(10)],
        }

    result = measure.score_entries(windows, tmp_path, counter, tolerance=1)
    assert (metrics(result)["tp"], metrics(result)["fp"], metrics(result)["fn"]) == (0, 1, 1)
    assert (metrics(result, kind="exit")["tp"], metrics(result, kind="exit")["fn"]) == (0, 1)
    assert (metrics(result, "cam1")["tp"], metrics(result, "cam1")["fn"]) == (0, 1)


def test_unsure_excludes_events_but_does_not_bridge_certain_regions(tmp_path):
    windows = load_rows(
        tmp_path,
        dict(WINDOW, events=[event(9), event(10), event(20)], unsure=[{"start": 10, "end": 20, "reason": "occluded"}]),
    )
    result = measure.score_entries(windows, tmp_path, counter_for([event(10), event(20), event(21)]), tolerance=20)
    # 10 is excluded, 20 is not. A prediction after the unsure span cannot match truth at9.
    score = metrics(result)
    assert (score["tp"], score["fp"], score["fn"]) == (1, 1, 1)
    assert result["per_window"][0]["excluded_labels"] == 1
    assert result["per_window"][0]["excluded_predictions"] == 1
    assert result["per_window"][0]["excluded_seconds"] == 10


def test_counter_unknown_is_diagnostic_and_does_not_erase_misses(tmp_path):
    windows = load_rows(tmp_path, dict(WINDOW, events=[event(10)]))
    result = measure.score_entries(
        windows, tmp_path, counter_for([], unknown=[{"start": 0, "end": 100, "reason": "missing video"}]), tolerance=2
    )
    score = metrics(result)
    assert (score["tp"], score["fp"], score["fn"], score["recall"]) == (0, 0, 1, 0.0)
    assert score["precision"] is None
    assert result["per_window"][0]["unknown_seconds"] == 100
    assert result["per_window"][0]["unknown"][0]["reason"] == "missing video"


def test_unsupported_exit_is_unscored_not_a_fabricated_miss(tmp_path):
    windows = load_rows(tmp_path, dict(WINDOW, events=[event(10), event(20, "exit")]))
    result = measure.score_entries(windows, tmp_path, counter_for([event(10)], supported=("entry",)), tolerance=1)
    assert metrics(result)["tp"] == 1
    exit_score = metrics(result, kind="exit")
    assert exit_score["precision"] is None and exit_score["recall"] is None
    assert exit_score["supported"] is False
    assert exit_score["unsupported_windows"] == 1


def test_empty_denominators_are_not_perfect_scores(tmp_path):
    windows = load_rows(tmp_path, WINDOW)
    score = metrics(measure.score_entries(windows, tmp_path, counter_for([]), tolerance=0))
    assert (score["tp"], score["fp"], score["fn"]) == (0, 0, 0)
    assert score["precision"] is None and score["recall"] is None


def test_counter_receives_only_window_coordinates_and_state(tmp_path):
    windows = load_rows(
        tmp_path,
        dict(
            WINDOW,
            events=[event(10)],
            conditions=["crowded"],
            unsure=[{"start": 30, "end": 40, "reason": "private label"}],
        ),
    )
    calls = []

    def counter(*, camera, start, end, state):
        calls.append((camera, start, end, state))
        return {"events": [], "supported_kinds": ["entry", "exit"]}

    measure.score_entries(windows, tmp_path, counter, tolerance=1)
    assert calls == [("cam0", 0.0, 100.0, tmp_path)]


@pytest.mark.parametrize(
    "patch",
    [
        pytest.param({"start": math.nan}, id="nan-start"),
        pytest.param({"end": math.inf}, id="infinite-end"),
        pytest.param({"end": 0}, id="empty-window"),
        pytest.param({"events": [event(math.nan)]}, id="nan-event"),
        pytest.param({"events": [event(100)]}, id="event-at-exclusive-end"),
        pytest.param({"events": [event(10, "jump")]}, id="unknown-kind"),
        pytest.param({"events": "ten entries"}, id="not-events"),
        pytest.param({"unsure": [{"start": 20, "end": 10}]}, id="backward-unsure"),
        pytest.param({"unsure": [{"start": 0, "end": 101}]}, id="unsure-outside"),
        pytest.param({"conditions": "crowded"}, id="not-condition-list"),
    ],
)
def test_invalid_label_rows_report_file_and_line(tmp_path, patch):
    with pytest.raises(measure.LabelError, match=r"entries\.jsonl:1:"):
        load_rows(tmp_path, dict(WINDOW, **patch))


def test_overlapping_same_camera_windows_rejected_but_touching_allowed(tmp_path):
    with pytest.raises(measure.LabelError, match="overlap"):
        load_rows(tmp_path, WINDOW, dict(WINDOW, id="overlap", start=99, end=110))
    assert len(load_rows(tmp_path, WINDOW, dict(WINDOW, id="touching", start=100, end=110))) == 2


@pytest.mark.parametrize("tolerance", [-1, math.nan, math.inf])
def test_invalid_tolerance_rejected(tmp_path, tolerance):
    windows = load_rows(tmp_path, WINDOW)
    with pytest.raises(ValueError, match="tolerance"):
        measure.score_entries(windows, tmp_path, counter_for([]), tolerance=tolerance)


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(None, id="not-mapping"),
        pytest.param({"events": []}, id="missing-capabilities"),
        pytest.param({"events": [event(math.nan)], "supported_kinds": ["entry"]}, id="nan-event"),
        pytest.param({"events": [event(100)], "supported_kinds": ["entry"]}, id="outside-window"),
        pytest.param({"events": [event(10, "exit")], "supported_kinds": ["entry"]}, id="undeclared-kind"),
        pytest.param({"events": [], "supported_kinds": ["jump"]}, id="bad-kind"),
        pytest.param(
            {"events": [], "supported_kinds": ["entry"], "unknown": [{"start": 0, "end": math.inf}]},
            id="nonfinite-unknown",
        ),
    ],
)
def test_invalid_counter_result_fails_with_context(tmp_path, response):
    windows = load_rows(tmp_path, WINDOW)
    with pytest.raises(measure.RunError, match="window"):
        measure.score_entries(windows, tmp_path, lambda **kwargs: response, tolerance=1)


def brute_match_score(expected, predicted, tolerance):
    # Exhaustively enumerate partial bijections, independent of a sorted dynamic program.
    candidates = [(0, 0.0)]
    for size in range(1, min(len(expected), len(predicted)) + 1):
        for selected in itertools.combinations(range(len(expected)), size):
            for assigned in itertools.permutations(range(len(predicted)), size):
                errors = [abs(expected[i] - predicted[j]) for i, j in zip(selected, assigned)]
                if all(error <= tolerance for error in errors):
                    candidates.append((size, sum(errors)))
    return min(candidates, key=lambda value: (-value[0], value[1]))


@settings(max_examples=100, deadline=None)
@given(st.lists(st.integers(0, 12), max_size=5), st.lists(st.integers(0, 12), max_size=5), st.integers(0, 5))
@example([1, 3], [0, 2], 1)
@example([0, 4], [3], 4)
@example([2, 2], [2, 2, 2], 0)
@example([], [], 0)
def test_matching_maximizes_cardinality_then_minimizes_time_error(expected, predicted, tolerance):
    pairs = measure.match_events(expected, predicted, tolerance)
    assert len({i for i, _ in pairs}) == len(pairs)
    assert len({j for _, j in pairs}) == len(pairs)
    errors = [abs(expected[i] - predicted[j]) for i, j in pairs]
    assert all(error <= tolerance for error in errors)
    assert (len(pairs), sum(errors)) == brute_match_score(expected, predicted, tolerance)


def test_cli_default_never_imports_counter(tmp_path, monkeypatch, capsys):
    load_rows(tmp_path, dict(WINDOW, events=[event(10)]))
    imported = []
    real_import = measure.importlib.import_module

    def forbidden_counter(name, package=None):
        imported.append(name)
        if name == "missing_counter":
            pytest.fail("default run imported counter")
        return real_import(name, package)

    monkeypatch.setattr(measure.importlib, "import_module", forbidden_counter)
    code = measure.main(["--dir", str(tmp_path), "--state-dir", str(tmp_path), "--json"])
    assert code == 0
    assert "missing_counter" not in imported
    report = json.loads(capsys.readouterr().out)
    assert "entries" not in report


def test_cli_explicit_adapter_runs_without_ground_truth_argument(tmp_path, monkeypatch, capsys):
    load_rows(tmp_path, dict(WINDOW, events=[event(10)]))
    adapter = types.ModuleType("entry_fixture_counter")
    calls = []

    def count(*, camera, start, end, state):
        calls.append((camera, start, end, state))
        return {"events": [event(11)], "supported_kinds": ["entry"]}

    adapter.__dict__["count"] = count
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    code = measure.main(
        [
            "--dir",
            str(tmp_path),
            "--state-dir",
            str(tmp_path),
            "--part",
            "entries",
            "--entry-counter",
            "entry_fixture_counter:count",
            "--entry-tolerance",
            "1",
            "--json",
        ]
    )
    assert code == 0
    assert calls == [("cam0", 0.0, 100.0, tmp_path)]
    report = json.loads(capsys.readouterr().out)
    assert metrics(report["entries"])["tp"] == 1


def test_cli_entries_requires_counter_and_explicit_tolerance(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        measure.main(["--dir", str(tmp_path), "--part", "entries"])
