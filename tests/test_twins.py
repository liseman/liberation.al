"""The same speech heard on two microphones (audio/twins.py): the rule, and the transcriber writing it once."""

import json

import pytest

from liberational import config as cfg
from liberational.audio import twins


def heard(device, start, text, end=None):
    """A line from device, a second long unless end says otherwise."""
    return twins.Heard(f"{device}@{start:.3f}", device, start, start + 1.0 if end is None else end, text)


@pytest.mark.parametrize(
    "a, b, same",
    [
        pytest.param(heard("wave", 10.0, "Turn it off."), heard("usb", 10.4, "turn it off"), True, id="alike"),
        pytest.param(
            heard("wave", 10.0, "Turn the lights off"),
            heard("usb", 10.6, "turn the light off"),
            True,
            id="nearly-alike",
        ),
        pytest.param(heard("wave", 10.0, "Turn it off."), heard("wave", 10.4, "Turn it off."), False, id="same-mic"),
        pytest.param(heard("wave", 10.0, "Turn it off."), heard("usb", 12.0, "Turn it off."), False, id="too-late"),
        pytest.param(heard("wave", 10.0, "Turn it off."), heard("usb", 10.2, "Is dinner ready?"), False, id="other"),
        pytest.param(heard("wave", 10.0, "..."), heard("usb", 10.2, "?"), False, id="no-words"),
    ],
)
def test_two_lines_are_one_thing_said_from_two_mics_close_and_alike(a, b, same):
    assert twins.same_speech(a, b, window_s=1.5, min_similarity=0.9) is same


def test_a_primary_mics_line_is_kept_even_after_another_mics_copy():
    recent = twins.Recent()
    recent.add(heard("Powerbeats Pro", 10.0, "Turn it off."))
    later = heard("reSpeaker 4 Mic Array", 10.3, "Turn it off.")
    assert recent.twin_of(later, 1.5, 0.9, primary=[]) is not None  # first wins
    assert recent.twin_of(later, 1.5, 0.9, primary=["respeaker"]) is None  # a primary mic's line stays
    recent.add(later)
    third = heard("Powerbeats Pro", 10.5, "turn it off")
    twin = recent.twin_of(third, 1.5, 0.9, primary=["respeaker"])
    assert twin is not None and twin.device == "reSpeaker 4 Mic Array"  # the other mic's copy goes


def test_a_window_or_a_similarity_of_zero_turns_it_off():
    recent = twins.Recent()
    recent.add(heard("wave", 10.0, "Turn it off."))
    assert recent.twin_of(heard("usb", 10.1, "Turn it off."), 0, 0.9, []) is None
    assert recent.twin_of(heard("usb", 10.1, "Is dinner ready?"), 1.5, 0, []) is None  # not "everything is a copy"


def test_two_people_saying_yes_a_moment_apart_are_two_lines():
    first, second = heard("Powerbeats", 0.0, "Yes.", end=0.4), heard("reSpeaker", 1.0, "yes", end=1.4)
    assert not twins.same_speech(first, second, 1.5, 0.9)  # alike and close, but they don't overlap
    assert twins.same_speech(first, heard("reSpeaker", 0.1, "yes", end=0.5), 1.5, 0.9)  # an echo does


def test_settings_that_cant_be_read_fall_back_to_their_defaults():
    with cfg.override({"audio": {"dedup_window_s": "1.5s", "dedup_similarity": None, "primary_devices": 1}}):
        assert twins.settings() == twins.DEFAULTS
    with cfg.override({"audio": {"primary_devices": "Wave"}}):
        assert twins.settings()[2] == ["Wave"]  # one name, not its letters


# ---- the transcriber writes the first copy only ----


def written(tx):
    return (
        [(e["device"], e["text"]) for e in map(json.loads, tx.jsonl.read_text().splitlines())]
        if tx.jsonl.exists()
        else []
    )


def test_copies_written_anyway_point_at_the_line_shown_for_them():
    lines = [
        heard("Powerbeats", 10.0, "Lights on."),  # 0: first
        heard("reSpeaker", 10.3, "lights on"),  # 1: its copy, on the primary mic
        heard("USB", 10.5, "Lights on!"),  # 2: a third copy
        heard("USB", 20.0, "Is dinner ready?"),  # 3: nothing like it
        heard("Wave", 20.2, "Is dinner ready"),  # 4: its copy
    ]
    assert twins.pairs(lines, 1.5, 0.9, primary=["respeaker"]) == {0: 1, 2: 1, 4: 3}  # the primary's line is shown
    assert twins.pairs(lines, 1.5, 0.9, primary=[]) == {1: 0, 2: 0, 4: 3}  # else the first
    assert twins.pairs(lines, 0, 0.9, primary=[]) == {}


def test_a_mic_saying_the_same_words_again_is_never_its_own_copy():
    lines = [
        heard("Kitchen", 0.0, "lights on", end=1.0),
        heard("Hall", 0.3, "lights on", end=1.6),  # the Kitchen line's copy
        heard("Kitchen", 1.2, "lights on", end=2.2),  # said again: overlaps the Hall copy, but it's new speech
    ]
    assert twins.pairs(lines, 1.5, 0.9, primary=[]) == {1: 0}
