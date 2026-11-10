import fc from 'fast-check';
import { describe, expect, it, vi } from 'vitest';
import {
  boundedDistance,
  createSearchIndex,
  foldSearchText,
  type SearchDocument,
  searchIndex,
  typoBudget,
} from './search';

const document = (id: string, text: string, ts = 0, kind: SearchDocument['kind'] = 'transcript'): SearchDocument => ({
  id,
  text,
  ts,
  kind,
});

// Independent full-matrix OSA oracle: no banding, trie, postings, or early exit.
function referenceDistance(left: string, right: string): number {
  const a = Array.from(left),
    b = Array.from(right);
  const matrix = Array.from({ length: a.length + 1 }, (_, i) =>
    Array.from({ length: b.length + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)),
  );
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      let value = Math.min(
        (matrix[i - 1]?.[j] ?? 0) + 1,
        (matrix[i]?.[j - 1] ?? 0) + 1,
        (matrix[i - 1]?.[j - 1] ?? 0) + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1])
        value = Math.min(value, (matrix[i - 2]?.[j - 2] ?? 0) + 1);
      const row = matrix[i];
      if (row) row[j] = value;
    }
  }
  return matrix[a.length]?.[b.length] ?? 0;
}

const smallWord = fc.array(fc.constantFrom('a', 'b', 'c', '𐐨'), { maxLength: 12 }).map(letters => letters.join(''));
const asciiWord = fc
  .array(fc.constantFrom('a', 'b', 'c'), { minLength: 1, maxLength: 12 })
  .map(letters => letters.join(''));
function prefixDistance(query: string, token: string): number {
  const letters = Array.from(token);
  return Math.min(
    ...Array.from({ length: letters.length + 1 }, (_, i) => referenceDistance(query, letters.slice(0, i).join(''))),
  );
}

/** Best-of-five budget for one search over 5,000 lines (16 ms idle; generous so load never fails it). */
const SEARCH_BUDGET_MS = 100;

describe('bounded optimal-string-alignment distance', () => {
  it.each([
    ['meeting', 'meetign', 1, 1],
    ['abcd', 'abdc', 1, 1],
    ['CA', 'ABC', 2, 3],
    ['abc', 'xyz', 1, 2],
    ['', '', 0, 0],
    ['', 'ab', 2, 2],
    ['𐐨a', 'a𐐨', 1, 1],
  ])('measures %s -> %s with budget %i', (left, right, budget, distance) => {
    expect(boundedDistance(left, right, budget)).toBe(distance);
  });
  it('agrees with the full OSA matrix for edits, swaps, empty strings, and Unicode code points', () => {
    fc.assert(
      fc.property(smallWord, smallWord, fc.integer({ min: 0, max: 2 }), (left, right, budget) => {
        expect(boundedDistance(left, right, budget)).toBe(Math.min(referenceDistance(left, right), budget + 1));
      }),
      { numRuns: 300 },
    );
  });
});

describe('Review fuzzy search', () => {
  it.each([
    ['cat', 'cut', false],
    ['cats', 'cuts', true],
    ['meeting', 'meetign', true],
    ['calendar', 'calxndzr', true],
    ['calendar', 'cxlxndzr', false],
  ])('scales typo allowance for %s versus %s', (query, text, found) => {
    const index = createSearchIndex([document('a', `${text} witness`)]);
    expect(searchIndex(index, `${query} witness`).length > 0).toBe(found);
  });
  it('searches summary and transcript together and requires every query word', () => {
    const index = createSearchIndex([
      document('coffee-only', 'Coffee arrived.', 1),
      document('summary', 'Coffee and the budget were discussed.', 2, 'summary'),
      document('budget-only', 'The budget is ready.', 3),
      document('line', 'Budget review over coffee.', 4),
    ]);
    expect(searchIndex(index, 'coffee budget').map(hit => hit.document.id)).toEqual(['summary', 'line']);
  });
  it('allows prefixes only for the final query word', () => {
    const index = createSearchIndex([document('a', 'Café calendar')]);
    expect(searchIndex(index, 'cafe cal').map(hit => hit.ranges)).toEqual([
      [
        [0, 4],
        [5, 13],
      ],
    ]);
    expect(searchIndex(index, 'caf calendar')).toEqual([]);
  });
  it('ranks full exact matches before prefixes and typos, then sorts by time', () => {
    const index = createSearchIndex([
      document('later-exact', 'calendar', 20),
      document('earlier-exact', 'calendar', 10),
      document('prefix', 'calendars', 1),
      document('typo', 'calandar', 0),
    ]);
    expect(searchIndex(index, 'calendar').map(hit => [hit.document.id, hit.score])).toEqual([
      ['earlier-exact', 0],
      ['later-exact', 0],
      ['prefix', 1],
      ['typo', 2],
    ]);
  });
  it('reuses duplicate query words without dropping their score contribution or prefix distinction', () => {
    const index = createSearchIndex([
      document('exact', 'calendar', 20),
      document('prefix', 'calendars', 1),
      document('typo', 'calandar', 0),
    ]);
    expect(searchIndex(index, 'calendar calendar').map(hit => [hit.document.id, hit.score])).toEqual([
      ['exact', 0],
      ['prefix', 4],
      ['typo', 6],
    ]);
  });

  it('uses deterministic ids to break equal quality/time ties and respects the limit', () => {
    const index = createSearchIndex([document('zeta', 'coffee', 4), document('alpha', 'coffee', 4)]);
    expect(searchIndex(index, 'coffee', 1).map(hit => hit.document.id)).toEqual(['alpha']);
    expect(searchIndex(index, 'coffee', 0)).toEqual([]);
  });
  it('returns original whole-word UTF16 ranges across accents, combining marks, and astral letters', () => {
    const index = createSearchIndex([document('a', '😀 Café e\u0301LAN 𐐀lpha')]);
    const [hit] = searchIndex(index, 'CAFE élan 𐐨lpha');
    expect(hit?.ranges).toEqual([
      [3, 7],
      [8, 13],
      [14, 20],
    ]);
    expect(hit?.ranges.map(([start, end]) => hit.document.text.slice(start, end))).toEqual([
      'Café',
      'e\u0301LAN',
      '𐐀lpha',
    ]);
  });
  it('folds case expansions and ligatures without changing highlight offsets', () => {
    const [hit] = searchIndex(createSearchIndex([document('a', 'Straße ﬃancée Σίσυφος')]), 'STRASSE ffiancee σισυφοσ');
    expect(hit?.ranges.map(([start, end]) => hit.document.text.slice(start, end))).toEqual([
      'Straße',
      'ﬃancée',
      'Σίσυφος',
    ]);
  });
  it('highlights repeated matching words once, without inserting markup', () => {
    const index = createSearchIndex([document('a', '<b>Café</b> café!')]);
    expect(searchIndex(index, 'cafe cafe')[0]?.ranges).toEqual([
      [3, 7],
      [12, 16],
    ]);
  });
  it('keeps an index snapshot independent of later input edits', () => {
    const source = document('a', 'Original recording');
    const index = createSearchIndex([source]);
    source.text = 'Changed recording';
    expect(searchIndex(index, 'original')[0]?.document.text).toBe('Original recording');
  });
  it.each(['', '  ', '!😀?'])('returns no fuzzy hits for empty token query %s', query => {
    expect(searchIndex(createSearchIndex([document('a', 'coffee')]), query)).toEqual([]);
  });
  it('matches full words against an independent OSA oracle', () => {
    fc.assert(
      fc.property(asciiWord, asciiWord, (query, text) => {
        const expected = referenceDistance(query, text) <= typoBudget(Array.from(query).length);
        expect(searchIndex(createSearchIndex([document('a', `${text} witness`)]), `${query} witness`).length > 0).toBe(
          expected,
        );
      }),
      { numRuns: 300 },
    );
  });
  it('matches final-word prefixes against an independent OSA oracle', () => {
    fc.assert(
      fc.property(asciiWord, asciiWord, (query, text) => {
        const expected = prefixDistance(query, text) <= typoBudget(Array.from(query).length);
        expect(searchIndex(createSearchIndex([document('a', text)]), query).length > 0).toBe(expected);
      }),
      { numRuns: 300 },
    );
  });
  it('folding is idempotent for arbitrary Unicode including malformed surrogate input', () => {
    const text = fc
      .array(fc.integer({ min: 0, max: 0x10ffff }), { maxLength: 30 })
      .map(points => points.map(point => String.fromCodePoint(point)).join(''));
    fc.assert(
      fc.property(text, value => {
        expect(foldSearchText(foldSearchText(value))).toBe(foldSearchText(value));
      }),
      { numRuns: 300 },
    );
  });
  it('returns sorted, bounded, nonoverlapping ranges on original word boundaries', () => {
    const word = fc.constantFrom('Café', 'e\u0301LAN', '𐐀lpha', 'Straße', 'résumé', 'coffee');
    fc.assert(
      fc.property(fc.array(word, { minLength: 1, maxLength: 20 }), words => {
        const text = words.join(' 😀 ');
        const [hit] = searchIndex(createSearchIndex([document('a', text)]), foldSearchText(words[0] ?? ''));
        expect(hit).toBeDefined();
        let previous = 0;
        for (const [start, end] of hit?.ranges ?? []) {
          expect(start).toBeGreaterThanOrEqual(previous);
          expect(end).toBeGreaterThan(start);
          expect(end).toBeLessThanOrEqual(text.length);
          expect(words).toContain(text.slice(start, end));
          previous = end;
        }
      }),
      { numRuns: 100 },
    );
  });
  it('keeps long-token DP work in compact bands rather than allocating a full row per trie node', () => {
    const text = 'a'.repeat(10_000);
    const index = createSearchIndex([document('long', text)]);
    const fill = Array.prototype.fill;
    const guard = vi.spyOn(Array.prototype, 'fill').mockImplementation(function (
      this: unknown[],
      value: unknown,
      start?: number,
      end?: number,
    ) {
      if (this.length > 9) throw new Error('full-width DP row allocation');
      return fill.call(this, value, start, end);
    });
    let hits: ReturnType<typeof searchIndex> = [];
    try {
      hits = searchIndex(index, text);
    } finally {
      guard.mockRestore();
    }
    expect(hits?.map(hit => [hit.document.id, hit.score, hit.ranges])).toEqual([['long', 0, [[0, 10_000]]]]);
  });

  // A 16 ms frame on an idle machine; the budget is generous and takes the best of several runs, so a loaded machine
  // or CI doesn't fail it while a search gone quadratic (seconds over 5,000 lines) still does.
  it('searches 5,000 varied lines fast, including a broad prefix and no hit', () => {
    const name = (n: number): string => {
      let word = '';
      do {
        word += String.fromCharCode(97 + (n % 26));
        n = Math.floor(n / 26);
      } while (n);
      return word;
    };
    const index = createSearchIndex(
      Array.from({ length: 5000 }, (_, i) =>
        document(
          String(i),
          `At gathering entry${name(i)} we discussed tomorrow schedule and café reservations for venue${name(4999 - i)}.`,
          i,
        ),
      ),
    );
    const observations = [];
    for (const query of ['schdeule cafe', 'g', 'zzzzzzzzzzzz', 'schedule '.repeat(50).trim()]) {
      let elapsed = Number.POSITIVE_INFINITY;
      let hits = searchIndex(index, query);
      for (let run = 0; run < 5; run++) {
        const start = performance.now();
        hits = searchIndex(index, query);
        elapsed = Math.min(elapsed, performance.now() - start);
      }
      observations.push({ query, elapsedMs: Number(elapsed.toFixed(3)), hits: hits.length });
      expect(elapsed, `${query} took ${elapsed.toFixed(3)} ms at best`).toBeLessThan(SEARCH_BUDGET_MS);
      expect(hits.length).toBe(query.startsWith('z') ? 0 : 100);
    }
    console.info('5,000-line search observations', observations);
  });
});
