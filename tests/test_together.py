"""Who was heard while a face was in view, and seen while a voice spoke: people/together.py."""

import pytest

from liberational.people import together as T

H = 3600.0  # an hour in, so times are plainly apart


def test_times_within_a_minute_are_one_moment():
    assert T.moments([H + 50, H, H + 30, H + 200, H + 255, H + 400]) == [H, H + 200, H + 400]
    assert T.moments([]) == []


@pytest.mark.parametrize(
    ("pairs", "name"),
    [
        pytest.param([(H, "Dana"), (H + 300, "Dana")], "Dana", id="two-moments"),
        pytest.param([(H, "Dana"), (H + 20, "Dana")], None, id="one-moment-only"),
        pytest.param([(H, "Dana"), (H + 300, "Dana"), (H + 600, "Eli")], "Dana", id="two-thirds"),
        pytest.param([(H, "Dana"), (H + 300, "Dana"), (H + 600, "Eli"), (H + 900, "Eli")], None, id="a-tie"),
        pytest.param([], None, id="nothing"),
    ],
)
def test_a_suggestion_needs_two_moments_and_two_thirds_of_them(pairs, name):
    got = T.suggest(pairs)
    assert (got or {}).get("name") == name


def test_a_suggestion_carries_its_evidence_newest_first():
    pairs = [(H + i * 300, "Dana") for i in range(7)] + [(H + 5000, "Eli")]
    got = T.suggest(pairs)
    assert got == {
        "name": "Dana",
        "moments": 7,
        "last": H + 1800,
        "times": [H + 1800, H + 1500, H + 1200, H + 900, H + 600],
        "others": {"Eli": 1},
    }


LINES = [(H, H + 4, "Dana"), (H + 2, H + 5, "Joe"), (H + 100, H + 103, "Eli")]


def test_a_voice_sees_the_named_faces_in_view_while_it_spoke():
    seen = [(H - 10, "Dana"), (H - 2, "Dana"), (H + 3, "Eli"), (H + 9, "Eli"), (H + 30, "Fay")]
    assert sorted(T.seen_with([(H, H + 5)], seen)) == [(H, "Dana"), (H, "Eli")]  # each name once per line
    assert T.seen_with([], seen) == []


def test_a_face_hears_the_named_voices_around_it_but_never_the_owners():
    pairs = T.Heard(LINES, [], ["joe"])([H + 6, H + 99, H + 200])
    # H + 6 is 1 s after Dana's line ends (Joe's too, but the owner's voice never counts); H + 99 is 1 s before Eli's
    # starts; H + 200 is near nothing
    assert sorted(pairs) == [(H + 6, "Dana"), (H + 99, "Eli")]


def test_a_voice_whose_own_face_was_in_view_explains_itself():
    pairs = T.Heard(LINES, [(H + 2, "Dana")], ["Joe"])([H + 1])
    assert pairs == []  # Dana's face was there: her speech isn't this face's


def test_naming_a_voice_suggests_the_face_heard_with_it_end_to_end():
    face_times = [H + 1, H + 1801, H + 3601]
    lines = [(H, H + 4, "Dana"), (H + 1800, H + 1804, "Dana"), (H + 3600, H + 3602, "Dana")]
    got = T.suggest(T.Heard(lines, [], ["Joe"])(face_times))
    assert got is not None and (got["name"], got["moments"]) == ("Dana", 3)
