// Time spans for Review's chips: the time the selected chips cover, and moving between those stretches.
import type { Chip, Span } from './types';

const valid = (spans: readonly Span[]): boolean =>
  spans.every(([a, b]) => Number.isFinite(a) && Number.isFinite(b) && a <= b);

/** Exact union: do not bridge gaps that could invent temporal co-occurrence. Points at an exclusive end stay. */
function union(spans: readonly Span[]): Span[] {
  const out: Span[] = [];
  for (const [a, b] of [...spans].sort((x, y) => x[0] - y[0] || y[1] - x[1])) {
    const last = out.at(-1);
    if (last && (a < last[1] || (a === last[1] && (a < b || last[0] === last[1])))) last[1] = Math.max(last[1], b);
    else out.push([a, b]);
  }
  return out;
}

const contains = ([a, b]: Span, t: number): boolean => (a === b ? t === a : a <= t && t < b);

/** Intersection of sorted exact unions, retaining actual point events but not creating points at touching ends. */
function intersect(left: readonly Span[], right: readonly Span[]): Span[] {
  const out: Span[] = [];
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    const a = left[i];
    const b = right[j];
    if (!a || !b) break;
    const start = Math.max(a[0], b[0]);
    const end = Math.min(a[1], b[1]);
    if (start < end || (start === end && contains(a, start) && contains(b, start))) out.push([start, end]);
    if (a[1] < b[1] || (a[1] === b[1] && a[0] !== a[1] && b[0] === b[1])) i++;
    else if (b[1] < a[1] || (a[1] === b[1] && b[0] !== b[1] && a[0] === a[1])) j++;
    else {
      i++;
      j++;
    }
  }
  return union(out);
}

type Timing = Span[] | null;
type Constraint = { spans: Span[] | undefined; unknown: boolean };

function timing(filtered: Chip | undefined, original: Chip | undefined): Timing {
  if (!original?.spans.length || !valid(original.spans) || (filtered && !valid(filtered.spans))) return null;
  return union(filtered?.spans ?? []);
}

function selectedFacets(
  filtered: ReadonlyMap<string, Chip>,
  sel: ReadonlySet<string>,
  source: ReadonlyMap<string, Chip>,
): Map<string, Timing> {
  const facets = new Map<string, { spans: Span[]; unknown: boolean }>();
  for (const id of sel) {
    const original = source.get(id);
    const c = filtered.get(id);
    const kind = original?.kind ?? c?.kind ?? 'unknown';
    const facet = facets.get(kind) ?? { spans: [], unknown: false };
    const spans = timing(c, original);
    if (spans === null) facet.unknown = true;
    else for (const span of spans) facet.spans.push(span);
    facets.set(kind, facet);
  }
  return new Map([...facets].map(([kind, facet]) => [kind, facet.unknown ? null : union(facet.spans)]));
}

/** No known constraint is unrestricted; an unknown facet is separate from a proven empty intersection. */
function constrain(facets: Iterable<Timing>): Constraint {
  let spans: Span[] | undefined;
  let unknown = false;
  for (const facet of facets) {
    if (facet === null) unknown = true;
    else {
      spans = spans === undefined ? facet : intersect(spans, facet);
      if (!spans.length) return { spans, unknown: false };
    }
  }
  return { spans, unknown };
}

/** OR within People, Things, or Events; AND across selected facets. null means timing is unavailable.
 *  originals distinguishes unknown source timing from known spans removed by the camera facet. */
export function facetedSpans(
  chips: readonly Chip[],
  sel: ReadonlySet<string>,
  originals: readonly Chip[] = chips,
): Span[] | null {
  const facets = selectedFacets(new Map(chips.map(c => [c.id, c])), sel, new Map(originals.map(c => [c.id, c])));
  const result = constrain(facets.values());
  return result.unknown ? null : (result.spans ?? []);
}

/** Options with no possible loaded overlap under OTHER selected facets. The candidate alone stands for its own
 *  facet, so an existing alternative cannot conceal its lack of overlap. Selected options always stay available.
 *  Build each other-facet constraint once, rather than intersecting selected facets for every rendered option. */
export function nonCooccurring(
  chips: readonly Chip[],
  sel: ReadonlySet<string>,
  originals: readonly Chip[] = chips,
): Set<string> {
  const filtered = new Map(chips.map(c => [c.id, c]));
  const source = new Map(originals.map(c => [c.id, c]));
  const facets = selectedFacets(filtered, sel, source);
  const contexts = new Map<string, Constraint>();
  for (const kind of new Set(originals.map(c => c.kind)))
    contexts.set(kind, constrain([...facets].filter(([facet]) => facet !== kind).map(([, spans]) => spans)));
  const out = new Set<string>();
  for (const original of originals) {
    if (sel.has(original.id)) continue;
    const context = contexts.get(original.kind);
    if (!context) continue;
    const spans = timing(filtered.get(original.id), original);
    if (context.spans?.length === 0 || spans?.length === 0) out.add(original.id);
    // A known disjoint constraint proves impossibility even when another facet's timing is unavailable.
    else if (spans !== null && context.spans && !intersect(spans, context.spans).length) out.add(original.id);
  }
  return out;
}

/** The selected facets' known time. Call facetedSpans directly when unknown and known-empty must differ. */
export function presentSpans(chips: readonly Chip[], sel: ReadonlySet<string>): Span[] {
  return facetedSpans(chips, sel) ?? [];
}

/** Index of the present span containing ts, or -1. */
export function spanAt(spans: readonly Span[], ts: number): number {
  let lo = 0;
  let hi = spans.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((spans[mid]?.[0] ?? 0) <= ts) lo = mid + 1;
    else hi = mid;
  }
  const s = spans[lo - 1];
  return s && (ts < s[1] || (s[0] === s[1] && ts === s[0])) ? lo - 1 : -1;
}

/** The start of the first present span that begins after ts (or ts itself, inside a span, when inside is true). */
export function nextSpanStart(spans: readonly Span[], ts: number, inside = false): number | null {
  for (const [s, e] of spans) {
    if (inside && s <= ts && ts < e) return ts;
    if (s > ts + 0.25) return s;
  }
  return null;
}

/** The start of the present span before the one at ts (or the current one's start when ts is well into it). */
export function prevSpanStart(spans: readonly Span[], ts: number): number | null {
  let best: number | null = null;
  for (const [s] of spans) {
    if (s < ts - 1.5) best = s;
    else break;
  }
  return best;
}

/** Where playback should be when it may only play present time: ts itself when a present span holds it (with a
 *  little slack, since video reports its time a hair early), else the start of the next present span, else null. */
export function presentFrom(spans: readonly Span[], ts: number): number | null {
  for (const [s, e] of spans) {
    if (s === e && ts <= s) return s;
    if (ts < e - 0.05) return ts >= s - 0.3 ? Math.max(ts, s) : s;
  }
  return null;
}
