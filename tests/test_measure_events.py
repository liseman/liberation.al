"""Event labels (what happened on video, and how many times) and the event-counter scorer in tools/measure.py."""

import importlib.util
import json
import pathlib
import sys
import types

import pytest

_spec = importlib.util.spec_from_file_location(
    "measure_events_under_test", pathlib.Path(__file__).resolve().parents[1] / "tools/measure.py"
)
assert _spec is not None and _spec.loader is not None
measure = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = measure
_spec.loader.exec_module(measure)

FIXTURES = pathlib.Path(__file__).parent / "data" / "measurement"
SPAN = dict(id="s", person="Dana", camera="cam1", what="sat down", start=1, end=2, count=5)


def write(directory, *rows, name="events"):
    (directory / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def without(row, key):
    return {k: v for k, v in row.items() if k != key}


def labels(tmp_path):
    """Made up: Dana sat down 8 times, then 5 more, on cam1, and someone walked past cam0 3 times (nobody named)."""
    write(
        tmp_path,
        dict(SPAN, id="sit-1", what="sat down", start=100, end=140, count=8),
        dict(SPAN, id="sit-2", what="Sat down", start=400, end=430, count=5),
        dict(
            without(SPAN, "person"), id="past", camera="cam0", what="someone walked past", start=1000, end=1100, count=3
        ),
    )
    return measure.load(tmp_path, "events")


class StubCounter:
    """Answers by window: a span it counts one short, one it misses, and the walk-pasts it splits in two."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, camera, what, start, end, state):
        self.calls.append((camera, what, start, end, state))
        found = {
            70: [dict(start=102, end=141, count=7), dict(start=150, end=160, count=2)],  # around sit-1, plus an extra
            370: [],  # around sit-2: nothing
            970: [dict(start=1000, end=1050, count=2), dict(start=1050, end=1100, count=1)],  # the walk-pasts, in two
        }[start]
        return {"spans": found, "diagnostics": {"model": "stub"}}


# ---- labels -----------------------------------------------------------------------------------------------------


def test_fixture_loads_any_event_with_its_attributes_and_notes():
    sat, fridge, dog = measure.load(FIXTURES, "events")
    assert (sat.what, sat.count, sat.person, sat.attributes) == ("sat down", 3, "Dana", {})
    assert sat.notes == ({"nth": 2, "note": "on the arm of the couch"}, {"note": "same chair the other times"})
    assert (fridge.what, fridge.count, fridge.attributes) == (
        "opened the fridge",
        2,
        {"door open": {"value": 40, "unit": "s"}},
    )
    assert (dog.what, dog.count, dog.person) == ("the dog got on the couch", 1, None)


def test_an_event_needs_no_person(tmp_path):
    write(tmp_path, without(SPAN, "person"))
    (event,) = measure.load(tmp_path, "events")
    assert (event.person, event.what, event.count) == (None, "sat down", 5)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        pytest.param(without(SPAN, "what"), "what must be", id="no-what"),
        pytest.param(without(SPAN, "count"), "count must be", id="no-count"),
        pytest.param(dict(SPAN, count=2.5), "count must be", id="count-fraction"),
        pytest.param(dict(SPAN, count=-1), "count must be", id="count-negative"),
        pytest.param(dict(SPAN, count=True), "count must be", id="count-bool"),
        pytest.param(dict(SPAN, person=""), "person must be", id="person-empty"),
        pytest.param(dict(SPAN, person=None), "person must be", id="person-null"),
        pytest.param(without(SPAN, "camera"), "camera must be", id="no-camera"),
        pytest.param(dict(SPAN, attributes=[60]), "attributes must be an object", id="attributes-list"),
        pytest.param(dict(SPAN, attributes={"door open": {"value": 40}}), "unit", id="attribute-unit"),
        pytest.param(dict(SPAN, attributes={"door open": {"value": "long", "unit": "s"}}), "needs a value", id="value"),
        pytest.param(dict(SPAN, notes=[{"nth": 1}]), "note", id="note-text"),
        pytest.param(dict(SPAN, notes=["sloppy"]), "each note is an object", id="note-item"),
        pytest.param(dict(SPAN, notes=[{"nth": 1.5, "note": "x"}]), "nth is which time", id="nth-fraction"),
        pytest.param(dict(SPAN, notes=[{"nth": 0, "note": "x"}]), "nth is which time", id="nth-zero"),
        pytest.param(dict(SPAN, notes=[{"nth": 6, "note": "x"}]), r"from 1 to the count \(5\)", id="nth-past-count"),
        pytest.param(dict(SPAN, start=2, end=1), "end is before start", id="backwards"),
    ],
)
def test_bad_event_labels_are_refused_with_their_line(tmp_path, row, message):
    write(tmp_path, row)
    with pytest.raises(measure.LabelError, match=rf"events\.jsonl:1: .*{message}"):
        measure.load(tmp_path, "events")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        pytest.param(dict(without(SPAN, "what"), activity="sat down"), "activity is now what", id="activity"),
        pytest.param(dict(without(SPAN, "count"), reps=5), "reps is now count", id="reps"),
        pytest.param(dict(SPAN, notes=[{"rep": 1, "note": "x"}]), "a note's rep is now nth", id="note-rep"),
    ],
)
def test_the_old_field_names_are_refused_with_the_new_ones(tmp_path, row, message):
    write(tmp_path, row)
    with pytest.raises(measure.LabelError, match=rf"events\.jsonl:1: {message} .*Moving old activity labels"):
        measure.load(tmp_path, "events")


@pytest.mark.parametrize("old", ["activities", "squats"])
def test_an_old_label_file_stops_the_run_and_says_how_to_move_it(tmp_path, capsys, old):
    write(tmp_path, SPAN)
    write(tmp_path, SPAN, name=old)
    code = measure.main(["--dir", str(tmp_path), "--state-dir", str(tmp_path), "--part", "events"])
    assert code == 2
    err = capsys.readouterr().err
    assert f"{old}.jsonl is now events.jsonl" in err
    assert "docs/MEASUREMENT.md says (Moving old activity labels)" in err


# ---- scoring ----------------------------------------------------------------------------------------------------


def test_without_a_counter_the_labels_are_summed_by_what_happened(tmp_path):
    r = measure.score_events(labels(tmp_path))
    assert (r["spans"], r["occurrences"], r["scored"]) == (3, 16, False)
    assert r["kinds"] == {"sat down": 2, "someone walked past": 1}  # compared without case


def test_the_counter_sees_a_window_and_what_happened_only(tmp_path):
    counter = StubCounter()
    measure.score_events(labels(tmp_path), tmp_path, counter, margin=30)
    assert counter.calls == [
        ("cam1", "sat down", 70, 170, tmp_path),
        ("cam1", "Sat down", 370, 460, tmp_path),
        ("cam0", "someone walked past", 970, 1130, tmp_path),
    ]


def test_count_error_and_alignment_per_span(tmp_path):
    r = measure.score_events(labels(tmp_path), tmp_path, StubCounter(), margin=30)
    sit1, sit2, past = r["per_span"]
    # sit-1: found 102-141 with 7 against 100-140 with 8. Shared 38 s of the 41 s either covers.
    assert (sit1["what"], sit1["count"], sit1["person"]) == ("sat down", 8, "Dana")
    assert (sit1["predicted_count"], sit1["count_error"], sit1["found"]) == (7, -1, True)
    assert sit1["overlap"] == pytest.approx(38 / 41)
    assert (sit1["start_offset_s"], sit1["end_offset_s"]) == (2, 1)
    assert sit1["extra_spans"] == 1  # 150-160 overlaps no labeled "sat down" on cam1
    assert sit1["diagnostics"] == {"model": "stub"}
    # sit-2: nothing found, so every labeled time is missed.
    assert (sit2["found"], sit2["predicted_count"], sit2["count_error"], sit2["overlap"]) == (False, 0, -5, 0)
    assert sit2["start_offset_s"] is None
    # The walk-pasts: split into two halves; the earlier half is compared, and the split is reported. Nobody is named.
    assert (past["pieces"], past["predicted_count"], past["count_error"], past["person"]) == (2, 2, -1, None)
    assert past["overlap"] == pytest.approx(0.5)
    assert (past["start_offset_s"], past["end_offset_s"]) == (0, -50)
    assert past["extra_spans"] == 0
    assert (r["found"], r["missed"], r["exact_counts"]) == (2, 1, 0)
    assert r["mean_abs_count_error"] == pytest.approx(7 / 3)  # 1, 5, and 1 off
    assert r["mean_overlap"] == pytest.approx((38 / 41 + 0.5) / 2)  # over the spans it found


def test_an_exact_count_on_the_same_span_scores_perfectly(tmp_path):
    write(tmp_path, dict(SPAN, start=100, end=140, count=8))

    def exact(*, camera, what, start, end, state):
        return {"spans": [dict(start=100, end=140, count=8)]}

    r = measure.score_events(measure.load(tmp_path, "events"), tmp_path, exact, margin=10)
    span = r["per_span"][0]
    assert (span["count_error"], span["overlap"], span["start_offset_s"], span["end_offset_s"]) == (0, 1, 0, 0)
    assert (r["exact_counts"], r["mean_abs_count_error"]) == (1, 0)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        pytest.param({"spans": [dict(start=0, end=5, count=1)]}, "outside the window", id="outside"),
        pytest.param({"spans": [dict(start=120, end=110, count=1)]}, "in order", id="backwards"),
        pytest.param({"spans": [dict(start=110, end=120, count=1.5)]}, "count must be", id="count"),
        pytest.param({"spans": [dict(start=110, end=120)]}, "start, end, and count", id="no-count"),
        pytest.param({"spans": [dict(start=110, end=120, reps=1)]}, "start, end, and count", id="old-reps"),
        pytest.param({"events": []}, "spans and optional diagnostics", id="wrong-shape"),
    ],
)
def test_a_bad_counter_result_names_the_span(tmp_path, result, message):
    write(tmp_path, dict(SPAN, start=100, end=140))
    with pytest.raises(measure.RunError, match=rf"event counter failed for s: .*{message}"):
        measure.score_events(measure.load(tmp_path, "events"), tmp_path, lambda **_: result, margin=10)


# ---- the command ------------------------------------------------------------------------------------------------


def test_cli_scores_an_explicit_counter_and_reports_each_span(tmp_path, monkeypatch, capsys):
    labels(tmp_path)
    module = types.ModuleType("event_fixture_counter")
    module.__dict__["count"] = StubCounter()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = ["--dir", str(tmp_path), "--state-dir", str(tmp_path), "--part", "events"]
    code = measure.main([*args, "--event-counter", "event_fixture_counter:count", "--event-margin", "30"])
    out = capsys.readouterr().out
    assert code == 0
    assert "found 2 of 3; exact counts 0 of 3; mean count error 2.3 times" in out
    assert "sit-1: sat down, counted 7 of 8 (-1); overlap 0.93, start +2.0 s, end +1.0 s, 1 extra" in out
    assert "sit-2: Sat down, counted 0 of 5 (-5); not found" in out
    assert "past: someone walked past, counted 2 of 3 (-1); overlap 0.50, start +0.0 s, end -50.0 s, in 2 pieces" in out


def test_cli_default_run_loads_events_without_a_counter(tmp_path, capsys):
    labels(tmp_path)
    code = measure.main(["--dir", str(tmp_path), "--state-dir", str(tmp_path), "--part", "events"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Events: 3 labeled spans (2 sat down, 1 someone walked past), 16 occurrences" in out
    assert "not scored: no counter given" in out


def test_cli_report_says_one_occurrence_in_the_singular(tmp_path, capsys):
    write(tmp_path, dict(SPAN, count=1))
    assert measure.main(["--dir", str(tmp_path), "--state-dir", str(tmp_path), "--part", "events"]) == 0
    assert "Events: 1 labeled span (1 sat down), 1 occurrence\n" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--event-counter", "x:y"], id="counter-without-part"),
        pytest.param(["--part", "events", "--event-margin", "-1"], id="negative-margin"),
        pytest.param(["--part", "events", "--activity-counter", "x:y"], id="old-option"),
    ],
)
def test_cli_refuses_counter_options_it_cannot_use(tmp_path, args):
    with pytest.raises(SystemExit, match="2"):
        measure.main(["--dir", str(tmp_path), *args])
