"""Spoken measurements use explicit text rules, not inferred activity, speaker, or units."""

import pytest

from liberational.people import registry


@pytest.fixture
def file_log(tmp_path):
    """The log `record` writes to: the test's broker, through the registry."""
    return registry.at(tmp_path).log


from liberational.memory.quantities import parse


@pytest.mark.parametrize(
    "text,value,unit,name",
    [
        ("135 kg", 135.0, "kg", "mass"),
        ("five kilograms", 5.0, "kg", "mass"),
        ("250 grams", 250.0, "g", "mass"),
        ("2 g", 2.0, "g", "mass"),
        ("one hundred and thirty-five pounds", 135.0, "lb", "mass"),
        ("135 lb", 135.0, "lb", "mass"),
        ("eight ounces", 8.0, "oz", "mass"),
        ("8 oz", 8.0, "oz", "mass"),
        ("five kilometers", 5.0, "km", "distance"),
        ("5 km", 5.0, "km", "distance"),
        ("three meters", 3.0, "m", "distance"),
        ("3 m", 3.0, "m", "distance"),
        ("forty centimeters", 40.0, "cm", "distance"),
        ("40 cm", 40.0, "cm", "distance"),
        ("ten millimeters", 10.0, "mm", "distance"),
        ("10 mm", 10.0, "mm", "distance"),
        ("two miles", 2.0, "mi", "distance"),
        ("2 mi", 2.0, "mi", "distance"),
        ("six feet", 6.0, "ft", "distance"),
        ("6 ft", 6.0, "ft", "distance"),
        ("twelve inches", 12.0, "in", "distance"),
        ("12 in", 12.0, "in", "distance"),
        ("twenty-one reps", 21.0, "rep", "count"),
        ("1 rep", 1.0, "rep", "count"),
        ("one thousand steps", 1000.0, "step", "count"),
        ("1 step", 1.0, "step", "count"),
        ("three laps", 3.0, "lap", "count"),
        ("1 lap", 1.0, "lap", "count"),
    ],
)
def test_explicit_measurements_keep_values_and_canonical_units(text, value, unit, name):
    quantities = parse(text)
    assert len(quantities) == 1
    quantity = quantities[0]
    assert (quantity.value, quantity.unit, quantity.name, quantity.rule) == (value, unit, name, "explicit_unit")
    assert (quantity.start, quantity.end) == (0, len(text))
    assert text[quantity.start : quantity.end] == text


@pytest.mark.parametrize(
    "text,value,unit",
    [
        ("1.5 kg", 1.5, "kg"),
        ("0.05 km", 0.05, "km"),
        ("one point five kilometers", 1.5, "km"),
        ("zero point zero five kilograms", 0.05, "kg"),
        ("1,350 pounds", 1350.0, "lb"),
        ("one thousand two hundred and thirty-four steps", 1234.0, "step"),
        ("ninety-nine reps", 99.0, "rep"),
        ("zero laps", 0.0, "lap"),
    ],
)
def test_conventional_cardinals_and_clean_decimals_are_not_split(text, value, unit):
    quantities = parse(text)
    assert len(quantities) == 1
    assert quantities[0].value == pytest.approx(value)
    assert quantities[0].unit == unit
    assert (quantities[0].start, quantities[0].end) == (0, len(text))


@pytest.mark.parametrize(
    "text,value,start,end",
    [
        ("putting on 135", 135.0, 11, 14),
        ("that's ten", 10.0, 7, 10),
        ("that’s ten", 10.0, 7, 10),
        ("PUTTING ON twenty-one.", 21.0, 11, 21),
        ("That's zero.", 0.0, 7, 11),
    ],
)
def test_bare_cues_preserve_unknown_unit_and_do_not_guess_activity(text, value, start, end):
    quantities = parse(text)
    assert len(quantities) == 1
    quantity = quantities[0]
    assert (quantity.value, quantity.unit, quantity.name, quantity.rule) == (value, None, "quantity", "bare_cue")
    assert (quantity.start, quantity.end) == (start, end)


def test_mixed_measurements_have_original_offsets_and_do_not_duplicate_unitful_cues():
    text = "Putting on 135 lb; that's ten reps; putting on 45."
    quantities = parse(text)
    assert [(q.value, q.unit, q.name, q.start, q.end, q.rule) for q in quantities] == [
        (135.0, "lb", "mass", 11, 17, "explicit_unit"),
        (10.0, "rep", "count", 26, 34, "explicit_unit"),
        (45.0, None, "quantity", 47, 49, "bare_cue"),
    ]
    assert [text[q.start : q.end] for q in quantities] == ["135 lb", "ten reps", "45"]


def test_unicode_prefix_and_hyphenated_cardinal_offsets_refer_to_original_text():
    text = "Dana: 🏃 five kilometers, then twenty-one reps."
    quantities = parse(text)
    assert [(q.value, q.unit, q.start, q.end) for q in quantities] == [(5.0, "km", 8, 23), (21.0, "rep", 30, 45)]
    assert [text[q.start : q.end] for q in quantities] == ["five kilometers", "twenty-one reps"]


def test_equal_mentions_at_different_offsets_are_distinct_measurements():
    quantities = parse("five km, five km")
    assert [(q.value, q.unit, q.start, q.end) for q in quantities] == [(5.0, "km", 0, 7), (5.0, "km", 9, 16)]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "There were ten people in room 135.",
        "I am putting on a coat.",
        "That is a useful idea.",
        "Meet at 10:30 on 2026-10-28.",
        "five minutes",
        "two hours",
        "ten percent",
        "twenty dollars",
        "a kilogram",
        "model135kg",
        "X5km",
        "-5 kg",
        "+5 kg",
        "minus five kilograms",
        "5/10 km",
        "5-10 kg",
        "five to ten kilograms",
        "1.2.3 kg",
        "1,35 kg",
        "five six kilograms",
        "twenty ten reps",
        "one point kilograms",
        "NaN kg",
        "Infinity kilometers",
        "−5 kg",
        "that's 1..2",
        "that's 1e309",
        "that's 1e3",
        "that's 5 / 6",
        "that's 5 - 6",
        "that's 5%",
        "that's 1,35",
        "that's five to ten",
        "that's five six",
        "putting on 135-145",
    ],
)
def test_unrelated_or_ambiguous_number_text_does_not_become_a_measurement(text):
    assert parse(text) == ()


@pytest.mark.parametrize(
    "text", ["there were five in the room", "five in total", "five in a row", "five in the morning", "five in each box"]
)
def test_preposition_in_is_not_an_inches_unit(text):
    assert parse(text) == ()


def test_spelled_inches_remain_valid_before_descriptive_words():
    quantities = parse("five inches long")
    assert [(q.value, q.unit) for q in quantities] == [(5.0, "in")]


@pytest.mark.parametrize(
    "timestamps,devices", [((1100.123, 1100.123), ("mic-a", "mic-b")), ((1100.1231, 1100.1232), ("mic-a", "mic-a"))]
)
def test_receipts_preserve_distinct_lines_with_same_rounded_evidence(tmp_path, timestamps, devices, file_log):
    from liberational.memory.quantities import record

    lines = [
        ({"ts_start": timestamps[0], "device": devices[0], "text": "five kg"}, None),
        ({"ts_start": timestamps[1], "device": devices[1], "text": "ten reps"}, None),
    ]
    record(tmp_path, lines)
    record(tmp_path, lines)
    events = [stored.event for stored in file_log.read("measurements")]
    assert [(event.payload["value"], event.payload["unit"]) for event in events] == [(5.0, "kg"), (10.0, "rep")]
    assert [event.payload["device"] for event in events] == list(devices)


def test_receipts_name_the_line_by_its_device(tmp_path, file_log):
    from liberational.memory.quantities import record

    record(tmp_path, [({"ts_start": 1100.123, "device": d, "text": "five kg"}, None) for d in ("Mic A", "Mic B")])
    events = [stored.event for stored in file_log.read("measurements")]
    assert [e.payload["utterance_id"] for e in events] == ["utterance:mic-a@1100.123", "utterance:mic-b@1100.123"]
    assert [e.evidence[0].id for e in events] == ["utterance:mic-a@1100.123", "utterance:mic-b@1100.123"]


def test_a_line_measured_twice_is_one_event(tmp_path, file_log):
    """The event's id comes from the line (its exact time, device, and the quantity's place), so measuring a line
    again, in any process, sends the same event and the log keeps one. A line measured before this (under a receipt
    key, with a random id) is measured once more after the import; v1's old measurements can be dropped by the owner."""
    from liberational.continuum import keyed_id
    from liberational.memory.quantities import record

    line = ({"ts_start": 1100.123, "device": "Mic A", "text": "five kg"}, None)
    record(tmp_path, [line])
    record(tmp_path, [line])
    events = [stored.event for stored in file_log.read("measurements")]
    assert len(events) == 1
    assert events[0].event_id == keyed_id(1100.123, "spoken-quantity", _key(1100.123, "Mic A", 0))


def _key(timestamp: float, device: str, ordinal: int) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(["1", float(timestamp).hex(), device, ordinal]).encode()).hexdigest()
