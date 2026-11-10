"""Frozen pre-extraction oracles: caller-specific ordering, endpoints, and grouping must survive extraction.

The SQL oracle deliberately retains SQLite's peer-row window semantics. The Python
oracles retain the original loops instead of calling any shared merge helper.
"""

import collections
import math

from hypothesis import example, given
from hypothesis import strategies as st

from liberational.dashboard import presence
from liberational.domain.kernel import PersonId
from liberational.memory import timeline

TIMES = st.integers(-30, 60).map(float)
GAPS = st.sampled_from([0.0, 1.0, 10.0, 20.0])
SPANS = st.lists(st.tuples(TIMES, TIMES), max_size=30)


def _presence_before(spans, gap):
    out = []
    for start, end in sorted(spans):
        if out and start - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


@given(SPANS, GAPS)
@example([], 10.0)
@example([(10.0, 10.0), (0.0, 0.0), (10.0, 10.0), (21.0, 21.0)], 10.0)
@example([(0.0, 30.0), (5.0, 6.0), (35.0, 40.0)], 10.0)
def test_presence_preserves_sorted_max_end_and_inclusive_gap(spans, gap):
    assert presence.merge(iter(spans), gap) == _presence_before(spans, gap)


def _timeline_before(observed, gap):
    by_person = collections.defaultdict(list)
    for item in observed:
        by_person[item.person].append(item)
    groups = []
    for person, items in by_person.items():
        items.sort(key=lambda item: (item.evidence.start, item.evidence.end))
        group = []
        end = -math.inf
        for item in items:
            if group and item.evidence.start - end > gap:
                groups.append((person, group))
                group = []
            group.append(item)
            end = item.evidence.end if len(group) == 1 else max(end, item.evidence.end)
        if group:
            groups.append((person, group))
    entries = [
        timeline.Entry(
            person,
            min(item.evidence.start for item in group),
            max(item.evidence.end for item in group),
            frozenset(item.modality for item in group),
            tuple(item.evidence for item in group if isinstance(item.evidence, timeline.Visit)),
            tuple(item.evidence for item in group if isinstance(item.evidence, timeline.Line)),
        )
        for person, group in groups
    ]
    return sorted(entries, key=lambda entry: (entry.start, entry.end, entry.person))


OBSERVATIONS = st.lists(st.tuples(st.sampled_from(["alice", "dana"]), TIMES, TIMES, st.booleans()), max_size=30)


@given(OBSERVATIONS, GAPS)
@example([], 10.0)
@example(
    [("dana", 0.0, 0.0, True), ("alice", 10.0, 10.0, False), ("alice", 0.0, 0.0, True), ("alice", 10.0, 10.0, True)],
    10.0,
)
def test_timeline_preserves_people_modalities_and_stable_evidence_order(rows, gap):
    observed = [
        timeline.Observed(
            PersonId(person),
            timeline.Visit(person, "cam0", start, end, 0.8) if seen else timeline.Line(start, end, str(index)),
        )
        for index, (person, start, end, seen) in enumerate(rows)
    ]
    assert timeline.project(iter(observed), gap) == _timeline_before(observed, gap)
