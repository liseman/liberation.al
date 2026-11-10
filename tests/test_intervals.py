"""Shared temporal grouping policies, independent of caller sorting or aggregation."""

import math

import pytest

from liberational.intervals import interval_groups


@pytest.mark.parametrize(
    "items, gap, endpoint, expected",
    [
        pytest.param([], 10, "maximum", [], id="empty"),
        pytest.param([(4, 2)], 10, "maximum", [[(4, 2)]], id="singleton-retains-endpoints"),
        pytest.param([(0, 20), (2, 3), (15, 16)], 1, "maximum", [[(0, 20), (2, 3), (15, 16)]], id="nested-maximum"),
        pytest.param([(0, 20), (2, 3), (15, 16)], 1, "last", [[(0, 20), (2, 3)], [(15, 16)]], id="nested-last"),
        pytest.param([(0, 1), (11, 12)], 10, "maximum", [[(0, 1), (11, 12)]], id="inclusive-equality"),
        pytest.param([(0, 1), (12, 13)], 10, "maximum", [[(0, 1)], [(12, 13)]], id="inclusive-above-gap"),
        pytest.param(
            [(20, 20), (5, 5), (16, 16)],
            10,
            "last",
            [[(20, 20), (5, 5)], [(16, 16)]],
            id="last-keeps-backward-order",
        ),
        pytest.param([(0, 0), (0, 0)], 0, "maximum", [[(0, 0), (0, 0)]], id="duplicate-items-retained"),
        pytest.param(
            [(0, 20), (30, 31), (40, 41)],
            1,
            "maximum",
            [[(0, 20)], [(30, 31)], [(40, 41)]],
            id="boundary-reset-after-split",
        ),
    ],
)
def test_grouping_policy_on_single_pass_input(items, gap, endpoint, expected):
    source = (item for item in items)
    assert (
        list(
            interval_groups(
                source,
                gap,
                start=lambda item: item[0],
                end=lambda item: item[1],
                endpoint=endpoint,
            )
        )
        == expected
    )


@pytest.mark.parametrize(
    "gap", [-1.0, math.nan, math.inf, -math.inf], ids=["negative", "nan", "positive-infinity", "negative-infinity"]
)
@pytest.mark.parametrize("items", [[], [(0.0, 1.0)]], ids=["empty", "nonempty"])
def test_invalid_gap_is_rejected_even_without_observations(gap, items):
    with pytest.raises(ValueError, match="gap"):
        list(interval_groups(items, gap, start=lambda item: item[0], end=lambda item: item[1]))


@pytest.mark.parametrize(
    "invalid", [math.nan, math.inf, -math.inf], ids=["nan", "positive-infinity", "negative-infinity"]
)
@pytest.mark.parametrize("at", [0, 1], ids=["start", "end"])
@pytest.mark.parametrize("prefix", [[], [(0.0, 1.0)]], ids=["first", "later"])
def test_nonfinite_timestamp_is_rejected(invalid, at, prefix):
    span = [2.0, 3.0]
    span[at] = invalid
    with pytest.raises(ValueError, match="finite"):
        list(interval_groups([*prefix, span], 10.0, start=lambda item: item[0], end=lambda item: item[1]))
