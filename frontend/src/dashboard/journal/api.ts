// The Journal tab's reads: one GET /journal per column (dashboard/journal.py). Each reply is the next column's
// rows and the page of the period they belong to.
import { queryOptions } from '@tanstack/react-query';
import { HttpError, request } from '../../shared/http';

/** The columns, coarsest first: each one lists what the row picked in the column before it holds. */
export const LEVELS = ['month', 'week', 'day', 'hour', 'event'] as const;
export type Level = (typeof LEVELS)[number];

/** One thing an entry cites, as it can be shown now (memory/journal_evidence.py's `resolve`): the recording while
 *  it's kept ("video"), a promoted clip or still, the room's audio for speech, or "expired" with a note saying why. */
export interface Source {
  stage: 'video' | 'clip' | 'still' | 'audio' | 'expired';
  caption: string;
  camera?: string | null;
  t?: number;
  /** A promoted clip's or still's file. */
  url?: string;
  /** A promoted clip's still, for its poster. */
  still?: string;
  /** Why an expired source has nothing to show: "footage expired", "no footage was kept", "audio expired". */
  note?: string;
  kind?: string | null;
  stream?: string | null;
  people: string[];
  start: number;
  end: number;
  review: string;
  audio?: string;
}

export interface Evidence {
  status?: 'verified' | 'unverified' | null;
  sources: Source[];
}

/** A row in a column. Months, weeks, and days say whether their page is written; days say how many hours they hold;
 *  hours and events carry their entry's id, text, and how many sources it cites. */
export interface Row {
  key: string;
  label: string;
  page?: boolean;
  hours?: number;
  id?: string;
  start?: number;
  end?: number;
  text?: string;
  sources?: number;
}

/** A period's journal page: Markdown with names, its citations linking into Review. */
export interface Page {
  level: string;
  period: string;
  md: string;
}

/** Why a period shows no page (dashboard/journal.py's `why_no_page`): not written yet, being written again after a
 *  transcript correction, nothing recorded in it, the model failed on it, or too old to be written now. */
export type NoPage = 'pending' | 'rewriting' | 'quiet' | 'failed' | 'old';

export interface Column {
  items: Row[];
  page: Page | null;
  /** Set when `page` is null for a month, week, or day. */
  why?: NoPage | null;
}

/** The query parameter that names a column's parent row. */
const PARENT: Record<Level, string | null> = { month: null, week: 'month', day: 'week', hour: 'day', event: 'hour' };

/** The URL that lists the `level` column under the parent row `parent` (none for months). */
export function columnUrl(level: Level, parent: string | null): string {
  const param = PARENT[level];
  return param && parent ? `/journal?${new URLSearchParams({ [param]: parent })}` : '/journal';
}

export const columnQuery = (level: Level, parent: string | null) =>
  queryOptions({
    queryKey: ['journal', level, parent],
    queryFn: ({ signal }) => request<Column>(columnUrl(level, parent), undefined, signal),
    staleTime: 60_000,
  });

/** One hour's or event's evidence, resolved when it's shown (resolving asks what video is still kept). */
export const evidenceQuery = (id: string, start: number) =>
  queryOptions({
    queryKey: ['journal', 'evidence', id],
    queryFn: ({ signal }) =>
      request<Evidence>(`/journal?${new URLSearchParams({ entry: id, start: String(start) })}`, undefined, signal),
    staleTime: 60_000,
  });

/** The current day's page (GET /journal/today): the written page once it's settled, else a today-so-far page
 *  built in the background ("pending" while it builds, with the last one as md). Today, or yesterday until its page is
 *  written; any other day is a 404. */
export interface TodayPage {
  day: string;
  status: 'settled' | 'ready' | 'pending' | 'empty' | 'error';
  md: string;
  coverage?: { through: number | null; hours: number; elapsed: number };
}

/** While a today-so-far page builds, or after it failed (the server retries), ask again after 5 s, half as long again
 *  each time, up to a minute. */
export const TODAY_POLL_MS = 5000;
const TODAY_POLL_MAX_MS = 60_000;

export const todayWait = (asked: number): number =>
  Math.min(TODAY_POLL_MAX_MS, TODAY_POLL_MS * 1.5 ** Math.max(0, asked - 1));

export const todayQuery = (day: string) =>
  queryOptions({
    queryKey: ['journal', 'today', day],
    queryFn: ({ signal }) => request<TodayPage>(`/journal/today?${new URLSearchParams({ day })}`, undefined, signal),
    staleTime: 60_000,
    retry: false,
    refetchInterval: q => {
      const { data, error, dataUpdateCount, errorUpdateCount } = q.state;
      // A 404 is a day this route doesn't build (not today or yesterday): nothing to wait for.
      if (error instanceof HttpError && error.status === 404) return false;
      const waiting = error || data?.status === 'pending' || data?.status === 'error';
      return waiting ? todayWait(dataUpdateCount + errorUpdateCount) : false;
    },
  });
