// Review's server data, cached by day. Panning back reuses a retained settled chunk; explicit invalidation
// refreshes corrections, while today's sources keep following recordings as they grow.
import { keepPreviousData, queryOptions } from '@tanstack/react-query';
import { HttpError, RequestTimeout, request } from '../../shared/http';
import { isoDay } from '../lib/format';
import type { TranscriptEntry } from '../live/types';
import type { RecordedDay } from './days';
import { daySegments } from './gatewaySegments';
import { dayBounds } from './time';
import type { AudioSpan, Chip, MomentNotes, RawSegment } from './types';

export const SUMMARY_MS = 5000; // a summary being written shows up within a few seconds
export const CHIPS_MS = 60_000; // today keeps growing: new lines, sightings, and summary bullets
export const BUCKET_S = 5; // camera notes are fetched and cached per 5 s of time

export const reviewKey = {
  day: (day: string) => ['review', 'day', day] as const, // everything of one day, for reloading it
  transcript: (day: string) => ['review', 'day', day, 'transcript'] as const,
  segments: (day: string) => ['review', 'day', day, 'segments'] as const,
  spans: (day: string) => ['review', 'day', day, 'spans'] as const,
  chips: (day: string) => ['review', 'day', day, 'chips'] as const,
  activities: (day: string) => ['review', 'day', day, 'activities'] as const,
  blocks: (day: string, t0: number, t1: number) => ['review', 'day', day, 'blocks', t0, t1] as const,
  notes: (bucket: number) => ['review', 'notes', bucket] as const,
  days: ['review', 'days'] as const,
};

const perOpening = {
  staleTime: Number.POSITIVE_INFINITY,
  gcTime: 20 * 60_000, // inactive chunks expire; panning back within this window reuses their settled data
  refetchOnWindowFocus: false,
  refetchOnReconnect: false,
  refetchOnMount: (query: { state: { isInvalidated: boolean } }) => query.state.isInvalidated,
} as const;
const q = (day: string) => `day=${encodeURIComponent(day)}`;

// Bound source reads across a wide viewport. Collect this turn's requests before admitting them, so the cursor's
// video/audio cannot queue behind all neighboring transcripts. Aborted queued reads never start an HTTP request.
// Actual JSON requests still use shared/http's two-slot gate; this also orders gateway listings and detector work.
const READ_SLOTS = 4;
let reading = 0;
let scheduled = false;
const readOrder = new Map<string, number>();
const waiting: { priority: () => number; start: () => void }[] = [];
/** A new cursor or pan reprioritizes reads that have not started, without canceling retained query data. */
export function prioritizeDayReads(days: string[]): void {
  readOrder.clear();
  days.forEach((day, i) => {
    readOrder.set(day, i);
  });
}
function drainReads(): void {
  scheduled = false;
  waiting.sort((a, b) => a.priority() - b.priority());
  while (reading < READ_SLOTS && waiting.length) waiting.shift()?.start();
}
function dayRead<T>(
  read: () => Promise<T>,
  signal: AbortSignal | undefined,
  day: string,
  rank: number,
  sourcePriority: number,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const abort = () => {
      const at = waiting.indexOf(ticket);
      if (at >= 0) waiting.splice(at, 1);
      reject(new DOMException('aborted', 'AbortError'));
    };
    const ticket = {
      priority: () => (readOrder.get(day) ?? rank) * 10 + sourcePriority,
      start: () => {
        signal?.removeEventListener('abort', abort);
        if (signal?.aborted) {
          abort();
          return;
        }
        reading++;
        Promise.resolve()
          .then(read)
          .then(resolve, reject)
          .finally(() => {
            reading--;
            drainReads();
          });
      },
    };
    if (signal?.aborted) {
      abort();
      return;
    }
    signal?.addEventListener('abort', abort, { once: true });
    waiting.push(ticket);
    if (!scheduled) {
      scheduled = true;
      queueMicrotask(drainReads);
    }
  });
}

/** A day's lines, in time order. */
export async function getTranscript(day: string, signal?: AbortSignal): Promise<TranscriptEntry[]> {
  const r = await request<{ entries?: TranscriptEntry[] }>(`/transcript/day?${q(day)}`, undefined, signal);
  return (r.entries ?? []).slice().sort((a, b) => a.ts_start - b.ts_start);
}

export const transcriptQuery = (day: string, today = false, priority = 0) =>
  queryOptions({
    queryKey: reviewKey.transcript(day),
    queryFn: ({ signal }) => dayRead(() => getTranscript(day, signal), signal, day, priority, 0),
    ...perOpening,
    refetchInterval: today ? LIVE_FILL_MS : false,
  });

/** How often today's video is listed again while Review shows it, so the newest stretch keeps growing. */
export const LIVE_FILL_MS = 5000;

/** The day's video, from Continuum's gateway: every camera's stretches with frames. Today's is listed again every
 *  LIVE_FILL_MS, as recording goes on. */
export const segmentsQuery = (day: string, today = false, priority = 0) =>
  queryOptions({
    queryKey: reviewKey.segments(day),
    queryFn: ({ signal }): Promise<RawSegment[]> => {
      const { t0, t1 } = dayBounds(day);
      return dayRead(() => daySegments(t0, t1, { signal }), signal, day, priority, 1);
    },
    ...perOpening,
    refetchInterval: today ? LIVE_FILL_MS : false,
  });

export const spansQuery = (day: string, today = false, priority = 0) =>
  queryOptions({
    queryKey: reviewKey.spans(day),
    queryFn: ({ signal }) =>
      dayRead(
        async () => {
          const r = await request<{ spans?: AudioSpan[] }>(`/audio/spans?${q(day)}`, undefined, signal);
          return (r.spans ?? []).slice().sort((a, b) => a.start - b.start);
        },
        signal,
        day,
        priority,
        2,
      ),
    ...perOpening,
    refetchInterval: today ? LIVE_FILL_MS : false,
  });

/** Every day with something recorded, oldest first, and the server's today. It refreshes with the chips, since today
 *  can start recording. */
export const daysQuery = () =>
  queryOptions({
    queryKey: reviewKey.days,
    queryFn: async ({ signal }) => {
      const r = await request<{ days?: RecordedDay[]; today?: string }>('/review/days', undefined, signal);
      return { days: r.days ?? [], today: typeof r.today === 'string' ? r.today : undefined };
    },
    refetchInterval: CHIPS_MS,
    refetchOnWindowFocus: false,
  });

class DetectorStorageError extends Error {
  readonly pending: boolean;
  constructor(pending: boolean) {
    super(pending ? 'Detector history is loading from local storage.' : 'Detector history is unavailable.');
    this.pending = pending;
  }
}

export interface DayProgress {
  elapsed_s: number;
  estimate_s: number;
  left_s: number;
}

/** The words for how long an ended day's count has left: "about 3 min left". */
export function timeLeft(seconds: number): string {
  if (seconds < 45) return 'less than a minute left';
  if (seconds < 90) return 'about a minute left';
  return `about ${Math.round(seconds / 60)} min left`;
}

/** "Oct 28", for "Working through Oct 28". */
export const dayName = (day: string): string =>
  new Date(`${day}T12:00:00`).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

/** The server is working out an ended day's count, once, off the request: Review shows how far along, and
 *  asks again every working.pollMs. */
export class DayStillWorking extends Error {
  readonly progress: DayProgress;
  constructor(day: string, progress: DayProgress) {
    super(`Working through ${dayName(day)}, ${timeLeft(progress.left_s)}.`);
    this.progress = progress;
  }
}
export const working = { pollMs: 3000 }; // how often to ask again (tests shorten it)

export const dayWorking = (error: unknown): DayStillWorking | null => (error instanceof DayStillWorking ? error : null);

export const isDetectorStoragePending = (error: unknown): boolean =>
  error instanceof DetectorStorageError && error.pending;

/** Detector storage is loading or unavailable: Review clears the day's chips rather than show stale ones. */
export const isDetectorStorage = (error: unknown): boolean => error instanceof DetectorStorageError;

/** A proxy gave up waiting (502, 503, 504), this page did (RequestTimeout), or the connection dropped: the server is
 *  likely still working on the day, so Review keeps asking and says it is still loading rather than failing. */
export const stillComputing = { retryMs: 5000 }; // how long to wait before asking again (tests shorten it)

export const isStillComputing = (error: unknown): boolean =>
  (error instanceof HttpError && [502, 503, 504].includes(error.status)) ||
  error instanceof RequestTimeout ||
  error instanceof TypeError;

export const chipsQuery = (day: string, today: boolean, priority = 0) =>
  queryOptions({
    queryKey: reviewKey.chips(day),
    queryFn: ({ signal }) =>
      dayRead(
        async () => {
          const result = await request<{ chips?: Chip[]; pending?: boolean; error?: string; working?: DayProgress }>(
            `/review/presence?${q(day)}`,
            undefined,
            signal,
          );
          if (result.working) throw new DayStillWorking(day, result.working);
          if (result.pending || result.error) throw new DetectorStorageError(!!result.pending);
          return result.chips ?? [];
        },
        signal,
        day,
        priority,
        1001,
      ),
    ...perOpening,
    retry: (count, error) =>
      !(error instanceof DetectorStorageError || error instanceof DayStillWorking) &&
      count < (isStillComputing(error) ? 6 : 1),
    retryDelay: (count, error) =>
      isStillComputing(error) ? stillComputing.retryMs : Math.min(1000 * 2 ** count, 30000),
    // Never a second request for the day while one is still running (a slow day used to pile them up).
    // Ended days also recover automatically after errors; a hard failure keeps its error UI between polls.
    refetchInterval: query =>
      query.state.fetchStatus === 'fetching'
        ? false
        : query.state.error instanceof DayStillWorking
          ? working.pollMs
          : query.state.error
            ? SUMMARY_MS
            : today
              ? CHIPS_MS
              : false,
    refetchOnWindowFocus: false,
  });

/** A source explicitly selected for a written summary, with its own camera, people, and evidence kind. */
export interface BlockEvidenceSource {
  id: string;
  stream: string;
  ts_start: number;
  ts_end: number;
  people: { id: string | null; name: string }[];
  camera: string | null;
  kind: 'action' | 'walkby' | 'zone' | null;
  activity_kind?: string;
  text: string;
}

/** One stretch of the day in a sentence (GET /review/summary): a whole journal window, so the first and last
 *  can reach past the range asked for. Tags use the Show only chips' ids (person:, object:, event:) plus topic:. */
export interface Block {
  start: number;
  end: number;
  text: string;
  tags: string[];
  /** Explicit supplied-source selection; older cached blocks may have no attribution. */
  evidence?: { status: 'verified' | 'unverified'; sources: BlockEvidenceSource[] };
}

/** A part of the day summed up in one sentence, from its hour lines; text null while it's being written. */
export interface PartRollup {
  name: string; // "Night", "Morning", "Afternoon", or "Evening": the same hours as partOfDay
  start: number;
  end: number;
  text: string | null;
  status: 'ready' | 'pending';
}

/** A local day's own sentence, from its parts' sentences, and the parts. A part the model couldn't sum up is left
 *  out, so the feed shows its hours instead. */
export interface DayRollup {
  day: string; // YYYY-MM-DD
  start: number;
  end: number;
  text: string | null;
  status: 'ready' | 'pending';
  parts: PartRollup[];
}

/** The blocks of a time range, coarse for a long range and finer for a short one. */
export interface Blocks {
  start: number;
  end: number;
  level: number; // the journal level in seconds; opaque here
  status: 'ready' | 'partial' | 'pending'; // partial and pending: still being written, so ask again
  blocks: Block[];
  days?: DayRollup[]; // at the hour level only (a day's view)
}

export const BLOCKS_POLL_MS = 5000; // while blocks are still being written

/** The summary blocks of the timeline's visible range in one day. A range that is the whole day shares the day
 *  summary's read (dayBlocksQuery), where there is one, instead of asking for the same blocks again (#92): opening
 *  Review made both requests. The server works out the day's bounds in its own timezone, so the shared answer is used only when it
 *  covers exactly the range asked for. */
export const blocksQuery = (day: string, t0: number, t1: number) =>
  queryOptions({
    queryKey: reviewKey.blocks(day, t0, t1),
    queryFn: async ({ signal, client }) => {
      const start = Math.floor(t0);
      const end = Math.ceil(t1);
      const whole = dayBounds(day);
      // Only where the day's summary is already being read (Review's own panel): alone, this would be a second
      // request, not a shared one.
      const dayRead = client.getQueryState(dayBlocksQuery(day).queryKey);
      if (dayRead && start === Math.floor(whole.t0) && end === Math.ceil(whole.t1)) {
        try {
          const shared = await client.fetchQuery({ ...dayBlocksQuery(day), staleTime: BLOCKS_POLL_MS });
          if (shared.start === start && shared.end === end) return shared;
        } catch {
          // the day's own read failed: this range asks for itself, and reports its own failure
        }
      }
      return request<Blocks>(`/review/summary?start=${start}&end=${end}`, undefined, signal);
    },
    refetchInterval: query => (query.state.data && query.state.data.status !== 'ready' ? BLOCKS_POLL_MS : false),
    placeholderData: keepPreviousData, // the last range's blocks stay while a zoom's load
  });

/** What the cameras saw in one 5 s bucket of time. Throws an HttpError with status 403 when the viewer may not see
 *  camera history (it is only shown on the machine Liberational runs on). */
export const notesQuery = (bucket: number) =>
  queryOptions({
    queryKey: reviewKey.notes(bucket),
    queryFn: ({ signal }) =>
      request<MomentNotes>(`/history/at?ts=${(bucket + 0.5) * BUCKET_S}&window=20`, undefined, signal),
    staleTime: Number.POSITIVE_INFINITY, // what the cameras saw doesn't change
    refetchOnWindowFocus: false,
    retry: false,
  });

export interface DetectedActivity {
  id: string;
  camera: string;
  start: number;
  end: number;
  count: number | null;
  name: string | null;
  kind: string | null;
  person: string | null;
  promoted: boolean;
  track_id?: string | null;
  person_name?: string | null;
  attributes?: Record<string, unknown>;
  attribute_evidence?: Record<
    string,
    {
      value?: unknown;
      uncertainty?: { value?: number | null; [key: string]: unknown };
      [key: string]: unknown;
    }
  >;
  uncertainty?: { value?: number | null; [key: string]: unknown };
  feedback?: 'right' | 'wrong' | null;
}
/** Backend calibration uses unique events' latest verdicts, beyond the bounded activity listing. */
export interface ActivityPromotion {
  kind: string | null;
  camera: string | null;
  status: 'counted' | 'needs_ratings' | 'needs_correct_ratings' | 'disabled';
  remaining_ratings: number | null;
  rated: number;
  right: number;
  precision: number | null;
}

export interface ActivitiesResponse {
  items: DetectedActivity[];
  truncated: boolean;
  promotion?: ActivityPromotion[];
}

// Every day's activities asked for in the same turn go out as one request (#52): Known's Activities reads a week,
// which was seven requests on every Known page. A day asked for alone keeps the single-day form.
interface DayAsk {
  resolve: (reply: ActivitiesResponse) => void;
  reject: (error: unknown) => void;
}
let asked = new Map<string, DayAsk[]>();
let sendDue = false; // this turn's request is scheduled

async function askDays(batch: Map<string, DayAsk[]>): Promise<void> {
  const days = [...batch.keys()];
  try {
    const first = days[0] ?? '';
    const replies =
      days.length === 1
        ? { [first]: await request<ActivitiesResponse>(`/activities?${q(first)}`) }
        : (await request<{ days: Record<string, ActivitiesResponse> }>(`/activities?days=${days.join(',')}`)).days;
    for (const [day, asks] of batch)
      for (const ask of asks) {
        const reply = Object.hasOwn(replies, day) ? replies[day] : undefined;
        if (reply) ask.resolve(reply);
        else ask.reject(new Error(`No activities came back for ${day}`));
      }
  } catch (error) {
    for (const asks of batch.values()) for (const ask of asks) ask.reject(error);
  }
}

/** One day's detected activities, sharing a request with the other days asked for this turn. */
export function getActivities(day: string, signal?: AbortSignal): Promise<ActivitiesResponse> {
  return new Promise<ActivitiesResponse>((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('aborted', 'AbortError'));
    const ask: DayAsk = { resolve, reject };
    const batch = asked;
    signal?.addEventListener(
      'abort',
      () => {
        // An ask cancelled before its request went out leaves it; a day nobody wants any more isn't asked for.
        const left = (batch.get(day) ?? []).filter(other => other !== ask);
        if (batch === asked && left.length) batch.set(day, left);
        else if (batch === asked) batch.delete(day);
        reject(new DOMException('aborted', 'AbortError'));
      },
      { once: true },
    );
    asked.set(day, [...(asked.get(day) ?? []), ask]);
    if (sendDue) return;
    sendDue = true;
    setTimeout(() => {
      const sending = asked;
      asked = new Map();
      sendDue = false;
      if (sending.size) void askDays(sending);
    }, 0);
  });
}

/** Ended days stay cached; owner naming/feedback explicitly invalidates reviewKey.activities(day). */
export const activitiesQuery = (day: string, today = false) =>
  queryOptions({
    queryKey: reviewKey.activities(day),
    queryFn: ({ signal }) => getActivities(day, signal),
    ...perOpening,
    refetchInterval: today ? CHIPS_MS : false,
  });

/** One kind's candidates with no rating yet (kind null is unnamed movement), for the rating queue. Each rating session
 *  fetches afresh and nothing is kept after it, so a new session never replays candidates already rated; within a
 *  session the queue steps past what it rated itself and fetches the next batch only when one runs out. */
export const activityQueueQuery = (day: string, kind: string | null) =>
  queryOptions({
    queryKey: ['review', 'day', day, 'activity-queue', kind ?? ''] as const,
    queryFn: ({ signal }) =>
      request<ActivitiesResponse>(
        `/activities?${q(day)}&kind=${encodeURIComponent(kind ?? '')}&unrated=1`,
        undefined,
        signal,
      ),
    staleTime: 0,
    gcTime: 0,
    refetchOnWindowFocus: false,
  });

/** The selected recording day's summary. The server resolves its configured timezone and DST bounds. */
export const dayBlocksQuery = (day: string) =>
  queryOptions({
    queryKey: [...reviewKey.day(day), 'blocks', 'whole-day'] as const,
    queryFn: ({ signal }) => request<Blocks>(`/review/summary?${q(day)}`, undefined, signal),
    refetchInterval: query =>
      day === isoDay() || (query.state.data && query.state.data.status !== 'ready') ? BLOCKS_POLL_MS : false,
    retry: false,
  });

export interface SemanticSearchResult {
  id: string;
  ts: number;
  end: number;
  text: string;
  kind: 'semantic';
}

/** Local semantic retrieval scoped to one recorded day; cancellation follows the search box. */
export async function semanticSearch(
  day: string,
  query: string,
  signal?: AbortSignal,
): Promise<SemanticSearchResult[]> {
  if (query.length > 500) throw new Error('Search queries must be 500 characters or fewer.');
  const result = await request<{ results: SemanticSearchResult[] }>(
    `/review/search?${q(day)}&q=${encodeURIComponent(query)}`,
    undefined,
    signal,
    'page',
  );
  return result.results;
}
