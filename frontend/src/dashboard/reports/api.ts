// Requests for the Reports tab (dashboard/reports.py): the chosen reports and their order, and each report's figures,
// which the server aggregates by day and caches (memory/reports.py). A report still reading days says which, and its
// query polls until it has them all.
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { request } from '../../shared/http';

export const REPORTS_KEY = ['reports'] as const;
export const LAYOUT_KEY = [...REPORTS_KEY, 'layout'] as const;
export const POLL_MS = 3000;

export interface ChosenReport {
  id: string;
  kind: string;
  arg: string | null;
  title: string;
  /** Who's around and Activity by person: the people pinned beside Everyone in the filter. */
  people?: string[];
  /** Walk-bys by camera: the cameras pinned beside Every camera in its filter. */
  cameras?: string[];
}

export interface Layout {
  reports: ChosenReport[];
  kinds: Record<string, string>;
  built_in?: string[];
}

interface Progress {
  report: string;
  first: string;
  last: string;
  pending: string[]; // days not read yet: shown as still loading, never as empty
  refreshing: string[]; // days shown from an older copy while they're read again
}

/** One hour of Who's around: who was there and for how many minutes. Null: nothing was recorded that hour. */
export interface AroundCell {
  t: number;
  who: [string, number][];
}

export interface WhosAround extends Progress {
  days: { day: string; hours: (AroundCell | null)[] }[];
  people: { id: string; name: string; hours: number }[];
}

export interface Conversations extends Progress {
  people: {
    id: string;
    name: string;
    lines: number;
    speech_min: number;
    by_hour: { n: number; day: string | null; t: number | null }[];
  }[];
  daily: { day: string; lines: number | null; t: number | null }[];
  words: { word: string; total: number; weeks: { week: string; n: number | null; day: string | null }[] }[];
  pairs: { a: string; b: string; a_name: string; b_name: string; n: number; day: string; t: number }[];
}

/** One series of Activity and events: an event kind, a zone, or an animal, by day and by hour of day. */
export interface Series {
  key: string;
  name: string;
  total: number;
  typical: number | null; // the median of the days it was looked for
  days: { day: string; n: number | null; t: number | null }[];
  by_hour: { n: number; day: string | null; t: number | null }[];
}

export interface Activity extends Progress {
  events: Series[];
  zones: Series[];
  pets: Series[];
  covered_days: { events: number; zones: number; pets: number };
}

export interface CountDay {
  day: string;
  n: number | null;
  t: number | null;
  hours: ({ n: number; t: number | null } | null)[];
}

export interface Walkbys extends Progress {
  cameras: { key: string; name: string; total: number; days: CountDay[] }[];
  covered_days: number;
}

export interface PersonActivity extends Progress {
  people: {
    id: string;
    name: string;
    series: {
      key: string;
      name: string;
      total_min: number;
      verified: boolean;
      days: {
        day: string;
        minutes: number | null;
        t: number | null;
        hours: ({ minutes: number; t: number | null } | null)[];
      }[];
    }[];
  }[];
  covered_days: number;
}

export interface Devices extends Progress {
  sources: { key: string; name: string; total: number; days: CountDay[] }[];
  covered_days: number;
  caveat: string;
}

export interface HealthHour {
  t: number;
  mem_pct: number | null;
  gpu_mem_pct: number | null;
  gpu_util: number | null;
}

export interface Health extends Progress {
  sampled_from: number | null;
  sample_s: number;
  hours: HealthHour[];
  services: { name: string; uptime_pct: number; samples: number; up_now: boolean }[];
  storage: {
    at: number;
    days: { day: string; gb: number; tier2_gb: number; plan_gb: number }[];
    media_gb: number;
    plan_gb: number;
    catalog_gb: number;
    vectors_gb: number;
  } | null;
}

export interface Suggestion extends ChosenReport {
  why: string;
}

export const SUGGESTIONS_KEY = [...REPORTS_KEY, 'suggestions'] as const;

/** Reports the corpus has data for that aren't on the page yet. */
export function useSuggestions() {
  return useQuery({
    queryKey: SUGGESTIONS_KEY,
    queryFn: ({ signal }) =>
      request<{ suggestions: Suggestion[] }>('/reports/api/suggestions', undefined, signal).then(r => r.suggestions),
  });
}

export type ReportData = WhosAround | Conversations | Activity | Health | Walkbys | PersonActivity | Devices;

export const reportKey = (kind: string, arg: string | null) => [...REPORTS_KEY, 'report', kind, arg ?? ''] as const;

export function reportURL(kind: string, arg: string | null): string {
  const q = new URLSearchParams({ kind });
  if (arg) q.set('arg', arg);
  return `/reports/api/report?${q}`;
}

/** A report's figures, polling while days are still being read. With `keep`, a change of argument shows the last
 * figures (isPlaceholderData) until the new ones arrive, instead of going back to loading. */
export function useReport<T extends Progress>(kind: string, arg: string | null, { keep = false } = {}) {
  return useQuery({
    queryKey: reportKey(kind, arg),
    placeholderData: keep ? keepPreviousData : undefined,
    queryFn: ({ signal }) => request<T>(reportURL(kind, arg), undefined, signal),
    refetchInterval: q =>
      q.state.data && (q.state.data.pending.length || q.state.data.refreshing.length) ? POLL_MS : false,
  });
}

export function useLayout() {
  return useQuery({
    queryKey: LAYOUT_KEY,
    queryFn: ({ signal }) => request<Layout>('/reports/api/layout', undefined, signal),
  });
}

/** Save the chosen reports, showing the new order at once and going back to the saved one if the save fails. */
export function useSaveLayout() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (reports: ChosenReport[]) => request<Layout>('/reports/api/layout', { reports }),
    onMutate: async reports => {
      await client.cancelQueries({ queryKey: LAYOUT_KEY });
      const before = client.getQueryData<Layout>(LAYOUT_KEY);
      if (before) client.setQueryData<Layout>(LAYOUT_KEY, { ...before, reports });
      return { before };
    },
    onError: (_e, _reports, context) => {
      if (context?.before) client.setQueryData(LAYOUT_KEY, context.before);
    },
    onSuccess: saved => {
      client.setQueryData(LAYOUT_KEY, saved);
      void client.invalidateQueries({ queryKey: SUGGESTIONS_KEY });
    },
  });
}
