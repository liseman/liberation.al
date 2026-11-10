// Known › Activities: the detected activities of the last RECENT_DAYS days, read a day at a time through
// Review's own /activities query (so a rating or a name made here or in Review updates both), and the kinds they
// teach. A kind is counted in Ask once enough of its clips are answered right; the server's promotion rows say how
// far each kind has to go.
import { useQueries } from '@tanstack/react-query';
import { activityName } from '../lib/activity';
import { isoDay } from '../lib/format';
import { type ActivityPromotion, activitiesQuery, type DetectedActivity } from '../review/api';
import type { ActivityKind } from './kinds';

export const RECENT_DAYS = 7;

const sentence = (text: string): string => text.charAt(0).toLocaleUpperCase() + text.slice(1);

/** The last `count` local days, today first, as YYYY-MM-DD. */
export function recentDays(now: Date = new Date(), count = RECENT_DAYS): string[] {
  return Array.from({ length: count }, (_, i) => {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    return isoDay(d);
  });
}

export interface Recent {
  items: { day: string; item: DetectedActivity }[]; // every activity, newest first
  promotion: ActivityPromotion[]; // the newest day's promotion rows: counting status is global, not per day
  ranked: boolean; // the newest day has answered (or failed), so which kinds are counted, and sampled, is known
  truncated: string[]; // days whose listing stopped at the server's limit
  pending: boolean;
  error: unknown;
  retry: () => void; // asks again for every day that failed
}

/** The last RECENT_DAYS days of activities, merged newest first. */
export function useRecentActivities(days: readonly string[] = recentDays()): Recent {
  const results = useQueries({ queries: days.map(day => activitiesQuery(day)) });
  const items = results
    .flatMap((r, i) => (r.data?.items ?? []).map(item => ({ day: days[i] ?? '', item })))
    .sort((a, b) => b.item.start - a.item.start);
  return {
    items,
    // The newest day's: an older day that answers first can carry none, and sampling by it, then by the newest's,
    // took most clips back off the page once they were shown (a load shift).
    promotion: results[0]?.data?.promotion ?? results.find(r => r.data?.promotion)?.data?.promotion ?? [],
    ranked: results[0] == null || results[0].data != null || results[0].isError,
    truncated: days.filter((_, i) => results[i]?.data?.truncated),
    pending: results.some(r => r.isPending),
    error: results.find(r => r.error)?.error ?? null,
    retry: () => {
      for (const r of results) if (r.error) void r.refetch();
    },
  };
}

/** A kind as the roster's filter knows it: '' stands for unclassified activities, which have no kind. */
export const kindKey = (kind: string | null): string => kind ?? '';

/** The roster's filter, remembered per browser. Storage can be missing or throw (a private window, blocked site
 * data), so every read and write is guarded and the filter still works for the visit. */
export const FILTER_KEY = 'liberational.activities.kind';

/** The remembered kind's key, or null for every kind. */
export function loadKindFilter(): string | null {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(FILTER_KEY) ?? 'null');
    return typeof value === 'string' ? value : null;
  } catch {
    return null;
  }
}

export function saveKindFilter(key: string | null): void {
  try {
    if (key === null) localStorage.removeItem(FILTER_KEY);
    else localStorage.setItem(FILTER_KEY, JSON.stringify(key));
  } catch {
    // Not remembered this time; the filter still applies until the page is left.
  }
}

/** A clip still waiting for an answer: no rating yet. */
export const unanswered = (item: DetectedActivity): boolean => !item.feedback;

export interface KindRow {
  kind: string | null;
  label: string;
  waiting: number; // clips of this kind with no answer yet
  promotion: ActivityPromotion | undefined;
  counted: boolean;
  defined?: ActivityKind; // the owner's activity, which can be deleted
}

/** One row per activity the owner defined, and per other kind seen lately or rated before: the counted ones
 *  first, then those with the most clips waiting. A defined activity is listed by its own name, and before any clip
 *  of it has been seen. */
export function kindRows(
  items: readonly DetectedActivity[],
  promotion: readonly ActivityPromotion[],
  defined: readonly ActivityKind[] = [],
): KindRow[] {
  const kinds = new Map<string, KindRow>();
  const row = (kind: string | null): KindRow => {
    const key = kind ?? '';
    const known = kinds.get(key);
    if (known) return known;
    const owned = defined.find(d => d.kind === kind);
    const base = promotion.find(p => p.kind === kind && p.camera === null) ?? owned?.promotion ?? undefined;
    const made: KindRow = {
      kind,
      // Sentence case, as every other name in the app: "Watch TV", not "watch tv".
      label: owned ? sentence(owned.name) : kind ? sentence(activityName(kind)) : 'Unclassified activity',
      waiting: 0,
      promotion: base,
      counted: base?.status === 'counted',
      defined: owned,
    };
    kinds.set(key, made);
    return made;
  };
  for (const d of defined) row(d.kind);
  for (const p of promotion) if (p.camera === null) row(p.kind);
  for (const item of items) if (unanswered(item)) row(item.kind).waiting++;
  return [...kinds.values()].sort(
    (a, b) => Number(b.counted) - Number(a.counted) || b.waiting - a.waiting || a.label.localeCompare(b.label),
  );
}

/** Once a kind is counted in Ask, Clips to check shows about 1 in SPOT_EVERY of its new clips as a spot check, so a
 *  kind that starts drifting still gets caught; picking the kind shows them all. */
export const SPOT_EVERY = 10;

/** Whether a counted kind's clip is in the spot-check sample: by a hash of its id (FNV-1a), so the same clips are
 *  picked on every load and in every browser. */
export function spotChecked(id: string): boolean {
  let h = 0x811c9dc5;
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h % SPOT_EVERY === 0;
}

/** The clips Clips to check lists among all activities: every one still unanswered, but only the spot-check sample
 *  of a kind counted in Ask. Known's Activities badge and the list's own count are both this, so they agree. */
export function toCheck<T extends { item: DetectedActivity }>(
  rows: readonly T[],
  promotion: readonly ActivityPromotion[],
): T[] {
  const counted = new Set(promotion.filter(p => p.camera === null && p.status === 'counted').map(p => kindKey(p.kind)));
  return rows.filter(r => unanswered(r.item) && (!counted.has(kindKey(r.item.kind)) || spotChecked(r.item.id)));
}

/** How far a learning kind is from being counted, 0 to 1, from its right answers and those it still needs. */
export function progress(p: ActivityPromotion | undefined): number {
  if (!p) return 0;
  if (p.status === 'counted') return 1;
  const left = p.remaining_ratings;
  if (typeof left !== 'number' || left < 0) return 0;
  return p.right + left > 0 ? p.right / (p.right + left) : 0;
}
