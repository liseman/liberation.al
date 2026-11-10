"""Pure temporal grouping shared by presence projections and caption estimates.

Callers own ordering, key partitioning, hard boundaries, and aggregation. Sharing
this arithmetic does not promote caption interpretations into presence evidence.
"""

import math
from collections.abc import Callable, Iterable, Iterator
from typing import Literal


def interval_groups[T](
    items: Iterable[T],
    gap: float,
    *,
    start: Callable[[T], float],
    end: Callable[[T], float],
    endpoint: Literal["maximum", "last"] = "maximum",
) -> Iterator[list[T]]:
    """Join observations in supplied order, retaining every item and its order.

    A singleton keeps its endpoints; no padding or sorting is done.
    ``maximum`` compares to the furthest end so nested spans cannot shorten a
    group. ``last`` compares to the previous item, including backward timestamps.
    Equality joins. Call separately for each key and each uninterrupted run;
    a missing/negative observation is a caller boundary. Gaps must be finite and
    nonnegative, and endpoints finite; invalid inputs raise ValueError.
    """
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("interval gap must be finite and nonnegative")
    group: list[T] = []
    boundary = 0.0
    for item in items:
        first, last = start(item), end(item)
        if not math.isfinite(first) or not math.isfinite(last):
            raise ValueError("interval endpoints must be finite")
        if group and first - boundary > gap:
            yield group
            group = []
        boundary = last if not group or endpoint == "last" else max(boundary, last)
        group.append(item)
    if group:
        yield group
