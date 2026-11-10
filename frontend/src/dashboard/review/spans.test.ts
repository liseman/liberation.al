import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { facetedSpans, nextSpanStart, nonCooccurring, presentFrom, presentSpans, prevSpanStart, spanAt } from './spans';
import type { Chip, Span } from './types';

const chip = (id: string, spans: [number, number][]): Chip => ({
  id,
  kind: id.split(':')[0] ?? 'object',
  label: id,
  spans,
  total_s: 0,
  count: spans.length,
});
const chips = [
  chip('person:a', [
    [10, 20],
    [100, 110],
  ]),
  chip('object:b', [
    [15, 30],
    [30.5, 40],
    [200, 210],
  ]),
  chip('event:c', [[50, 60]]),
];
const u: Span[] = [
  [10, 40],
  [100, 110],
  [200, 210],
];

describe('presentSpans', () => {
  it('joins alternatives in one facet without bridging a time gap', () => {
    expect(
      presentSpans(
        [
          chip('object:a', [
            [10, 20],
            [100, 110],
          ]),
          ...chips,
        ],
        new Set(['object:a', 'object:b']),
      ),
    ).toEqual([
      [10, 30],
      [30.5, 40],
      [100, 110],
      [200, 210],
    ]);
  });
  it('is empty with nothing selected', () => {
    expect(presentSpans(chips, new Set())).toEqual([]);
  });
});

describe('spanAt', () => {
  it.each([
    [9.9, -1],
    [10, 0],
    [39.9, 0],
    [40, -1], // ends are exclusive
    [205, 2],
  ])('at %s is %s', (ts, i) => {
    expect(spanAt(u, ts)).toBe(i);
  });
});

describe('stepping between stretches', () => {
  it('goes to the next stretch, or stays in the current one when asked', () => {
    expect(nextSpanStart(u, 5)).toBe(10);
    expect(nextSpanStart(u, 20)).toBe(100);
    expect(nextSpanStart(u, 20, true)).toBe(20);
    expect(nextSpanStart(u, 60, true)).toBe(100);
    expect(nextSpanStart(u, 205)).toBeNull();
  });
  it('goes back to the current stretch’s start when well into it, else the one before', () => {
    expect(prevSpanStart(u, 105)).toBe(100);
    expect(prevSpanStart(u, 100.5)).toBe(10);
    expect(prevSpanStart(u, 10)).toBeNull();
  });
});

describe('presentFrom', () => {
  it('keeps a time inside a stretch, snaps one a hair early to its start, and jumps a gap', () => {
    expect(presentFrom(u, 25)).toBe(25);
    expect(presentFrom(u, 9.8)).toBe(10);
    expect(presentFrom(u, 50)).toBe(100);
  });
  it('is null after the last stretch', () => {
    expect(presentFrom(u, 209.97)).toBeNull();
  });
});

it('retains a crossing point at the selected window start without changing its source time', () => {
  const points = presentSpans([chip('event:crossing', [[10, 10]])], new Set(['event:crossing']));
  expect(points).toEqual([[10, 10]]);
  expect(spanAt(points, 10)).toBe(0);
  expect(presentFrom(points, 10)).toBe(10);
  expect(presentFrom(points, 9)).toBe(10);
});

describe('faceted temporal selection', () => {
  it('ORs alternatives within each facet, and ANDs People, Things, and Events', () => {
    const selected = [
      chip('person:alice', [[10, 20]]),
      chip('person:dana', [[30, 40]]),
      chip('object:mug', [[15, 35]]),
      chip('event:lunch', [[18, 32]]),
    ];
    expect(facetedSpans(selected, new Set(selected.map(c => c.id)))).toEqual([
      [18, 20],
      [30, 32],
    ]);
    expect(presentSpans(chips, new Set(['person:a', 'object:b']))).toEqual([[15, 20]]);
  });

  it('keeps actual point events at inclusive starts, excludes exclusive ends, and invents no point at touching spans', () => {
    const selected = [
      chip('person:alice', [[10, 20]]),
      chip('event:crossing', [
        [10, 10],
        [20, 20],
      ]),
    ];
    expect(facetedSpans(selected, new Set(selected.map(c => c.id)))).toEqual([[10, 10]]);
    expect(
      facetedSpans(
        [chip('person:alice', [[10, 20]]), chip('object:mug', [[20, 30]])],
        new Set(['person:alice', 'object:mug']),
      ),
    ).toEqual([]);
    const endpoint = [
      chip('person:alice', [
        [0, 10],
        [10, 10],
      ]),
      chip('event:crossing', [[10, 10]]),
    ];
    expect(facetedSpans(endpoint, new Set(endpoint.map(c => c.id)))).toEqual([[10, 10]]);
  });

  it('preserves unknown OR members while still proving emptiness from disjoint known facets', () => {
    const selected = [chip('person:alice', [[10, 20]]), chip('person:dana', []), chip('object:mug', [[30, 40]])];
    expect(facetedSpans(selected, new Set(selected.map(c => c.id)))).toBeNull();
    const more = [...selected, chip('event:lunch', [[50, 60]])];
    expect(facetedSpans(more, new Set(more.map(c => c.id)))).toEqual([]);
    expect(facetedSpans([chip('person:alice', [[Number.NaN, 20]])], new Set(['person:alice']))).toBeNull();
    expect(facetedSpans([chip('person:alice', [[20, 10]])], new Set(['person:alice']))).toBeNull();
    expect(facetedSpans([], new Set(['person:missing']))).toBeNull();
  });

  it('distinguishes camera-eliminated known timing from unavailable original timing', () => {
    const original = [chip('person:alice', [[10, 20]])];
    expect(facetedSpans([], new Set(['person:alice']), original)).toEqual([]);
    expect(facetedSpans([chip('person:alice', [])], new Set(['person:alice']), original)).toEqual([]);
    expect(facetedSpans([chip('person:alice', [])], new Set(['person:alice']))).toBeNull();
  });

  it('agrees with a timestamp membership oracle for interval unions, intersections, and points', () => {
    const span = fc
      .tuple(fc.integer({ min: 0, max: 15 }), fc.integer({ min: 0, max: 15 }))
      .map(([a, b]): Span => [Math.min(a, b), Math.max(a, b)]);
    const spans = fc.array(span, { minLength: 1, maxLength: 5 });
    fc.assert(
      fc.property(spans, spans, spans, spans, (alice, dana, things, events) => {
        const selected = [
          chip('person:alice', alice),
          chip('person:dana', dana),
          chip('object:mug', things),
          chip('event:lunch', events),
        ];
        const result = facetedSpans(selected, new Set(selected.map(c => c.id)));
        expect(result).not.toBeNull();
        const has = (list: readonly Span[], t: number) => list.some(([a, b]) => (a === b ? t === a : a <= t && t < b));
        for (let t = 0; t <= 15; t += 0.5)
          expect(has(result ?? [], t)).toBe((has(alice, t) || has(dana, t)) && has(things, t) && has(events, t));
        for (let i = 0; i < (result?.length ?? 0); i++) {
          const span = result?.[i];
          if (!span) throw new Error('missing span');
          expect(span[0]).toBeLessThanOrEqual(span[1]);
          const previous = result?.[i - 1];
          if (previous)
            expect(
              previous[1] < span[0] || (previous[1] === span[0] && span[0] === span[1] && previous[0] < previous[1]),
            ).toBe(true);
        }
        expect(facetedSpans([...selected].reverse(), new Set(selected.map(c => c.id).reverse()))).toEqual(result);
      }),
      { numRuns: 100 },
    );
  });
});

describe('co-occurrence knowledge and option availability', () => {
  it('greys a proven disjoint alternative despite an unknown selected facet, but keeps possible overlap neutral', () => {
    const chips = [
      chip('person:alice', []),
      chip('object:mug', [[0, 1]]),
      chip('event:lunch', [[2, 3]]),
      chip('event:dinner', [[0, 1]]),
    ];
    const selected = new Set(['person:alice', 'object:mug']);
    expect(facetedSpans(chips, new Set([...selected, 'event:lunch']))).toEqual([]);
    expect(nonCooccurring(chips, selected)).toEqual(new Set(['event:lunch']));
    expect(facetedSpans(chips, new Set([...selected, 'event:dinner']))).toBeNull();
  });
  const span = fc
    .tuple(fc.integer({ min: 0, max: 10 }), fc.integer({ min: 0, max: 10 }))
    .map(([a, b]): Span => [Math.min(a, b), Math.max(a, b)]);
  const evidence = fc.record({
    spans: fc.array(span, { minLength: 1, maxLength: 4 }),
    mode: fc.constantFrom('known', 'unknown', 'removed'),
  });
  type Evidence = { spans: Span[]; mode: string };
  const ids = ['person:alice', 'person:dana', 'object:mug', 'object:lamp', 'event:lunch', 'event:dinner'];
  function samples(input: Evidence[]) {
    const originals = input.map((item, i) => chip(ids[i] ?? '', item.mode === 'unknown' ? [] : item.spans));
    const filtered = originals.filter((_, i) => input[i]?.mode !== 'removed');
    return { originals, filtered };
  }
  // Independent discrete oracle: integer span boundaries make half-second samples cover every possible interval
  // and point. Unknown OR alternatives make their facet unconstrained, with uncertainty retained unless other
  // known facets exclude every timestamp. It never sorts, unions, or intersects span arrays.
  function oracle(input: Evidence[], selected: ReadonlySet<string>): 'empty' | 'unknown' | 'known' {
    const facets = new Map<string, Evidence[]>();
    input.forEach((item, i) => {
      const id = ids[i] ?? '';
      if (!selected.has(id)) return;
      const kind = id.split(':')[0] ?? '';
      const facet = facets.get(kind) ?? [];
      facet.push(item);
      facets.set(kind, facet);
    });
    if (!facets.size) return 'empty';
    const unknown = [...facets.values()].some(items => items.some(item => item.mode === 'unknown'));
    for (let t = 0; t <= 10; t += 0.5) {
      if (
        [...facets.values()].every(
          items =>
            items.some(item => item.mode === 'unknown') ||
            items.some(
              item => item.mode === 'known' && item.spans.some(([a, b]) => (a === b ? t === a : a <= t && t < b)),
            ),
        )
      )
        return unknown ? 'unknown' : 'known';
    }
    return 'empty';
  }

  it('keeps null distinct from provably empty under unknown OR alternatives and camera removals', () => {
    fc.assert(
      fc.property(fc.array(evidence, { minLength: 6, maxLength: 6 }), input => {
        const { originals, filtered } = samples(input);
        const selected = new Set(ids);
        const result = facetedSpans(filtered, selected, originals);
        const knowledge = oracle(input, selected);
        expect(result === null ? 'unknown' : result.length ? 'known' : 'empty').toBe(knowledge);
      }),
      { numRuns: 100 },
    );
  });

  it("agrees with brute-force greying after ignoring each candidate's own selected facet", () => {
    fc.assert(
      fc.property(fc.array(evidence, { minLength: 6, maxLength: 6 }), fc.subarray(ids), (input, selectedIds) => {
        const { originals, filtered } = samples(input);
        const selected = new Set(selectedIds);
        const result = nonCooccurring(filtered, selected, originals);
        for (const candidate of originals) {
          if (selected.has(candidate.id)) {
            expect(result.has(candidate.id)).toBe(false);
            continue;
          }
          const comparison = new Set(selectedIds.filter(id => id.split(':')[0] !== candidate.kind));
          comparison.add(candidate.id);
          expect(result.has(candidate.id)).toBe(oracle(input, comparison) === 'empty');
        }
      }),
      { numRuns: 100 },
    );
  });
});
