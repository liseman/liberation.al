// Review's visible window as independently cached local-day chunks. The engine owns playback and the viewport;
// this hook fills the days it needs without opening a new player or discarding loaded neighbors.
import { type QueryKey, type UseQueryResult, useQueries, useQueryClient } from '@tanstack/react-query';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useStore } from 'zustand';
import { isoDay } from '../lib/format';
import {
  activitiesQuery,
  chipsQuery,
  dayWorking,
  prioritizeDayReads,
  segmentsQuery,
  spansQuery,
  transcriptQuery,
} from './api';
import { calendarStep } from './days';
import type { ReviewEngine } from './engine';
import { activityChips } from './lanes';
import { dayBounds, type View } from './time';
import type { Chip, DayData } from './types';

/** Visible and playback days with one neighboring day on each side; speculative future days are omitted. */
export function windowDays(day: string | null, range: View | null, today = isoDay()): string[] {
  if (!day) return [];
  const first = range ? isoDay(new Date(range.t0 * 1000)) : day;
  const last = range ? isoDay(new Date((range.t1 - 0.001) * 1000)) : day;
  const end = calendarStep(last, 1);
  const days = new Set<string>([day, calendarStep(day, -1), calendarStep(day, 1)]);
  for (let d = calendarStep(first, -1); d <= end; d = calendarStep(d, 1)) days.add(d);
  const visible = (d: string) => d >= first && d <= last;
  return [
    day,
    ...[...days]
      .filter(d => d !== day && d <= today)
      .sort((a, b) => Number(visible(b)) - Number(visible(a)) || a.localeCompare(b)),
  ];
}

/** Whether a window day's reads poll (#2414): only today, and only while it's the day picked or within the visible
 *  range. As a neighbor kept for panning, today loads once like any other day: before, viewing yesterday still read
 *  today's lines, video, and audio every few seconds, and each read merged the whole window's video list again. An
 *  ended day never polls; its data stays fresh until a correction invalidates it (staleTime Infinity, perOpening). */
export function pollsLive(d: string, day: string | null, range: View | null, today = isoDay()): boolean {
  if (d !== today) return false;
  if (d === day) return true;
  if (!range) return false;
  return d >= isoDay(new Date(range.t0 * 1000)) && d <= isoDay(new Date((range.t1 - 0.001) * 1000));
}

interface Published extends DayData {
  gen: number;
}

export function useReviewWindow(engine: ReviewEngine) {
  const day = useStore(engine.ui, s => s.day);
  const gen = useStore(engine.ui, s => s.gen);
  const range = useStore(engine.ui, s => s.range);
  const today = isoDay();
  const wanted = useMemo(() => windowDays(day, range, today), [day, range, today]);
  const wantedKey = [...wanted].sort().join(',');
  useLayoutEffect(() => prioritizeDayReads(wanted), [wanted]);
  const live = (d: string) => pollsLive(d, day, range, today);
  const transcript = useQueries({ queries: wanted.map((d, i) => transcriptQuery(d, live(d), i)) });
  const segments = useQueries({ queries: wanted.map((d, i) => segmentsQuery(d, live(d), i)) });
  const spans = useQueries({ queries: wanted.map((d, i) => spansQuery(d, live(d), i)) });
  const chips = useQueries({ queries: wanted.map((d, i) => chipsQuery(d, live(d), i)) });
  const activities = useQueries({ queries: wanted.map((d, i) => activitiesQuery(d, live(d), i)) });
  // A day read before it ended (today kept as a neighbor, or shown until midnight) holds that snapshot forever once
  // it's over, since an ended day never polls: it's read once more after its end, then it's settled (Codex on #2416).
  const client = useQueryClient();
  useEffect(() => {
    const all: [UseQueryResult<unknown>[], (d: string) => QueryKey][] = [
      [transcript, d => transcriptQuery(d).queryKey],
      [segments, d => segmentsQuery(d).queryKey],
      [spans, d => spansQuery(d).queryKey],
      [chips, d => chipsQuery(d, false).queryKey],
      [activities, d => activitiesQuery(d).queryKey],
    ];
    wanted.forEach((d, i) => {
      if (d >= today) return;
      const end = dayBounds(d).t1 * 1000;
      for (const [results, key] of all) {
        const q = results[i];
        if (q?.data !== undefined && !q.isFetching && q.dataUpdatedAt > 0 && q.dataUpdatedAt < end)
          void client.invalidateQueries({ queryKey: key(d), exact: true });
      }
    });
  }, [client, wanted, today, transcript, segments, spans, chips, activities]);
  const [initialized, setInitialized] = useState(false);
  const published = useRef(new Map<string, Published>());
  const chipSources = useRef(new Map<string, { chips: Chip[] | undefined; activities: unknown; failed: boolean }>());
  const active = useRef('');
  const statuses = useRef(new Map<string, 'loading' | 'ready' | 'error'>());
  useEffect(() => {
    if (active.current !== wantedKey) {
      active.current = wantedKey;
      engine.setWindowDays(wanted);
      // The query cache keeps evicted snapshots, while the engine drops their rows. Revisiting must publish them
      // again even if the cached array identities have not changed.
      for (const key of published.current.keys()) if (!wanted.includes(key)) published.current.delete(key);
      for (const key of chipSources.current.keys()) if (!wanted.includes(key)) chipSources.current.delete(key);
      for (const key of statuses.current.keys()) if (!wanted.includes(key)) statuses.current.delete(key);
    }
    wanted.forEach((d, i) => {
      const t = transcript[i];
      const v = segments[i];
      const a = spans[i];
      if (!t || !v || !a) return;
      const core = [t, v, a];
      const settled = core.every(q => !q.isPending && (!q.isFetching || q.data !== undefined));
      const status = !settled ? 'loading' : core.some(q => q.isError) ? 'error' : 'ready';
      if (statuses.current.get(d) !== status) {
        statuses.current.set(d, status);
        engine.setDayStatus(d, status);
      }
      if (settled) {
        const before = published.current.get(d);
        // A transient failure with retained data must not delete a previously loaded recording chunk.
        const data: DayData = {
          entries: t.data ?? before?.entries ?? [],
          segments: v.data ?? before?.segments ?? [],
          spans: a.data ?? before?.spans ?? [],
        };
        if (
          before?.gen !== gen ||
          before.entries !== data.entries ||
          before.segments !== data.segments ||
          before.spans !== data.spans
        ) {
          published.current.set(d, { ...data, gen });
          engine.upsertDayData(d, data);
        }
        if (d === day) setInitialized(true);
      }
      const c = chips[i];
      const events = activities[i];
      if (!c || !events) return;
      const failed = c.isError && !dayWorking(c.error);
      const before = chipSources.current.get(d);
      if (before?.chips !== c.data || before?.activities !== events.data || before?.failed !== failed) {
        chipSources.current.set(d, { chips: c.data, activities: events.data, failed });
        if (c.data || events.data || failed)
          engine.setDayChips(d, [
            ...(!failed ? (c.data ?? []) : []),
            ...activityChips(events.data?.items ?? [], events.data?.promotion ?? []),
          ]);
      }
    });
  }, [engine, gen, day, wanted, wantedKey, transcript, segments, spans, chips, activities]);
  const visibleDays = wanted.filter(
    d => !range || (d >= isoDay(new Date(range.t0 * 1000)) && d <= isoDay(new Date((range.t1 - 0.001) * 1000))),
  );
  const truncatedDays = wanted.filter((d, i) => visibleDays.includes(d) && activities[i]?.data?.truncated);
  const activityErrorDays = wanted.filter((d, i) => visibleDays.includes(d) && activities[i]?.isError);
  return { initialized, days: wanted, truncatedDays, activityErrorDays };
}
