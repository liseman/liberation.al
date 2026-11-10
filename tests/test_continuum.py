import math
import uuid

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from liberational import continuum as c

U64 = 2**64 - 1
LAST_SECOND = U64 // 10**9  # the last whole second a u64 of nanoseconds holds
relaxed = settings(deadline=None)  # niced runs on a busy machine; the conversions themselves are quick


def with_version(n: int, version: int, variant: int = 0x2) -> uuid.UUID:
    """A UUID from 128 bits with its version nibble and variant bits set."""
    n = (n & ~(0xF << 76)) | (version << 76)
    n = (n & ~(0x3 << 62)) | (variant << 62)
    return uuid.UUID(int=n)


def uuid7(n: int) -> uuid.UUID:
    return with_version(n, 7)


bits = st.integers(min_value=0, max_value=2**128 - 1)
uuid7s = bits.map(uuid7)
not_uuid7s = st.one_of(
    st.builds(with_version, bits, st.sampled_from([v for v in range(16) if v != 7])),
    st.builds(with_version, bits, st.just(7), st.sampled_from([0x0, 0x1, 0x3])),  # a v7 nibble, another variant
)
instants = st.floats(min_value=0, max_value=LAST_SECOND, allow_nan=False, allow_infinity=False)


def test_an_id_that_isnt_a_string_is_a_type_error():
    with pytest.raises(TypeError):
        c.check_id(uuid7(1 << 90).bytes)  # ty: ignore[invalid-argument-type]  # the wrong type is the point


@relaxed
@given(instants)
@example(0.0)
@example(1793227877.341349)
@example(float(LAST_SECOND))
@example(5e-324)
def test_seconds_round_trip_through_nanoseconds_within_a_nanosecond(t):
    assert abs(c.from_ns(c.to_ns(t)) - t) <= 1e-9


@relaxed
@given(st.floats(min_value=2**23, max_value=LAST_SECOND, allow_nan=False))
def test_seconds_after_april_1970_round_trip_exactly(t):
    assert c.from_ns(c.to_ns(t)) == t


# Up to the last whole second: in the u64's final second (the year 2554) the nearest float rounds past its end.
@relaxed
@given(st.integers(min_value=0, max_value=LAST_SECOND * 10**9))
def test_nanoseconds_through_seconds_stay_within_half_a_float_step(ns):
    t = c.from_ns(ns)
    assert abs(c.to_ns(t) - ns) <= math.ulp(t) * 10**9 / 2 + 0.5


@relaxed
@given(instants, instants)
def test_to_ns_keeps_order(a, b):
    lo, hi = sorted((a, b))
    assert c.to_ns(lo) <= c.to_ns(hi)


def test_to_ns_is_the_nearest_nanosecond_to_the_float_itself():
    # The float 1793227877.341349 is exactly 1793227877.341348886489868...; multiplying by 1e9 in floating point
    # would round to ...348864 first.
    assert c.to_ns(1793227877.341349) == 1_793_227_877_341_348_886


@pytest.mark.parametrize(
    "bad, error",
    [
        pytest.param(-1.0, ValueError, id="negative"),
        pytest.param(math.nan, ValueError, id="nan"),
        pytest.param(1793227877341.0, ValueError, id="milliseconds"),
        pytest.param(float(LAST_SECOND + 1), ValueError, id="past-u64"),
        pytest.param(True, TypeError, id="bool"),
    ],
)
def test_to_ns_refuses_what_isnt_an_instant(bad, error):
    with pytest.raises(error):
        c.to_ns(bad)


@pytest.mark.parametrize(
    "bad, error",
    [
        pytest.param(-1, ValueError, id="negative"),
        pytest.param(U64 + 1, ValueError, id="past-u64"),
        pytest.param(1.5, TypeError, id="float"),
        pytest.param(True, TypeError, id="bool"),
    ],
)
def test_from_ns_refuses_what_isnt_a_u64(bad, error):
    with pytest.raises(error):
        c.from_ns(bad)


@pytest.mark.parametrize("ns", [np.uint64(5_000_000_000), np.int64(5_000_000_000)], ids=["uint64", "int64"])
def test_from_ns_takes_numpy_integers(ns):
    assert c.from_ns(ns) == 5.0


def test_an_instant_spans_one_point_and_a_backwards_span_is_refused():
    assert c.span_ns(10.0) == (10 * 10**9, 10 * 10**9)
    assert c.span_ns(10.0, 12.5) == (10 * 10**9, 12_500_000_000)
    with pytest.raises(ValueError):
        c.span_ns(12.0, 10.0)


def test_a_span_backwards_by_less_than_a_nanosecond_is_still_refused():
    with pytest.raises(ValueError):
        c.span_ns(1.0, 0.9999999999)
