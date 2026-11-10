import { QueryClient, QueryObserver } from '@tanstack/react-query';
import { HttpResponse, http } from 'msw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { server } from '../../test/server';
import { isoDay } from '../lib/format';
import {
  activitiesQuery,
  chipsQuery,
  dayBlocksQuery,
  getActivities,
  prioritizeDayReads,
  SUMMARY_MS,
  segmentsQuery,
  semanticSearch,
  spansQuery,
  timeLeft,
  transcriptQuery,
} from './api';
import { daySegments } from './gatewaySegments';
import type { Chip } from './types';

vi.mock('./gatewaySegments', async original => ({
  ...(await original<typeof import('./gatewaySegments')>()),
  daySegments: vi.fn(),
}));

it('polls an ended day after retries fail, without overlapping a slow recovery request', async () => {
  vi.useFakeTimers();
  const client = new QueryClient();
  let finish: (chips: Chip[]) => void = () => {};
  const recovery = new Promise<Chip[]>(resolve => {
    finish = resolve;
  });
  const queryFn = vi
    .fn<() => Promise<Chip[]>>()
    .mockRejectedValueOnce(new Error('unavailable'))
    .mockRejectedValueOnce(new Error('unavailable'))
    .mockReturnValue(recovery);
  const observer = new QueryObserver(client, {
    ...chipsQuery('2026-10-27', false),
    queryFn,
  });
  const unsubscribe = observer.subscribe(() => {});
  try {
    await vi.advanceTimersByTimeAsync(1000);
    expect(observer.getCurrentResult().isError).toBe(true);
    expect(queryFn).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(SUMMARY_MS);
    expect(queryFn).toHaveBeenCalledTimes(3);
    expect(observer.getCurrentResult().fetchStatus).toBe('fetching');
    await vi.advanceTimersByTimeAsync(SUMMARY_MS * 2);
    expect(queryFn).toHaveBeenCalledTimes(3);
    finish([]);
    await vi.advanceTimersByTimeAsync(0);
    expect(observer.getCurrentResult().isSuccess).toBe(true);
    await vi.advanceTimersByTimeAsync(SUMMARY_MS * 2);
    expect(queryFn).toHaveBeenCalledTimes(3);
  } finally {
    unsubscribe();
    client.clear();
    vi.useRealTimers();
  }
});

describe('timeLeft', () => {
  it.each([
    [10, 'less than a minute left'],
    [60, 'about a minute left'],
    [95, 'about 2 min left'],
    [600, 'about 10 min left'],
  ])('%s s is %s', (seconds, words) => {
    expect(timeLeft(seconds)).toBe(words);
  });
});

afterEach(() => vi.unstubAllEnvs());

it('sends the selected day, encoded query, and cancellation signal to local semantic retrieval', async () => {
  let seen: URL | undefined;
  server.use(
    http.get('/review/search', ({ request }) => {
      seen = new URL(request.url);
      return HttpResponse.json({ results: [{ id: 'one', ts: 10, end: 12, text: 'A moment', kind: 'semantic' }] });
    }),
  );
  const result = await semanticSearch('2026-10-27', 'café & lunch');
  expect(seen?.searchParams.get('day')).toBe('2026-10-27');
  expect(seen?.searchParams.get('q')).toBe('café & lunch');
  expect(result).toEqual([{ id: 'one', ts: 10, end: 12, text: 'A moment', kind: 'semantic' }]);
  const cancelled = new AbortController();
  cancelled.abort();
  await expect(semanticSearch('2026-10-27', 'cancelled', cancelled.signal)).rejects.toMatchObject({
    name: 'AbortError',
  });
});

it('preserves semantic availability errors instead of converting them to an empty result', async () => {
  server.use(http.get('/review/search', () => HttpResponse.json({ error: 'Ollama is unavailable' }, { status: 503 })));
  await expect(semanticSearch('2026-10-27', 'lunch')).rejects.toMatchObject({
    status: 503,
    message: 'Ollama is unavailable',
  });
});

it.each([
  { day: '2026-03-08', start: '2026-03-08T00:00:00-08:00', end: '2026-03-09T00:00:00-07:00', hours: 23 },
  { day: '2026-11-01', start: '2026-11-01T00:00:00-07:00', end: '2026-11-02T00:00:00-08:00', hours: 25 },
])(
  'uses the recording server day with a Tokyo viewer across its $hours-hour DST boundary',
  async ({ day, start, end, hours }) => {
    vi.stubEnv('TZ', 'Asia/Tokyo');
    const canonicalStart = Date.parse(start) / 1000;
    const canonicalEnd = Date.parse(end) / 1000;
    expect(new Date(`${day}T00:00:00`).getTime() / 1000).not.toBe(canonicalStart);
    const seen: URL[] = [];
    server.use(
      http.get('/review/summary', ({ request }) => {
        seen.push(new URL(request.url));
        return HttpResponse.json({
          start: canonicalStart,
          end: canonicalEnd,
          status: 'ready',
          blocks: [
            { start: canonicalStart, end: canonicalStart + 60, text: 'First recording-day moment', tags: [] },
            { start: canonicalEnd - 60, end: canonicalEnd, text: 'Last recording-day moment', tags: [] },
          ],
        });
      }),
    );
    const client = new QueryClient();
    try {
      const result = await client.fetchQuery(dayBlocksQuery(day));
      expect(seen[0]?.searchParams.get('day')).toBe(day);
      expect(seen[0]?.searchParams.has('start')).toBe(false);
      expect(seen[0]?.searchParams.has('end')).toBe(false);
      expect(canonicalEnd - canonicalStart).toBe(hours * 3600);
      expect(result.blocks.map(block => block.text)).toEqual([
        'First recording-day moment',
        'Last recording-day moment',
      ]);
      expect(dayBlocksQuery(day).queryKey).toEqual(['review', 'day', day, 'blocks', 'whole-day']);
    } finally {
      client.clear();
    }
  },
);

it('limits day reads to four and starts higher-priority cursor requests before queued neighbors', async () => {
  const started: string[] = [];
  let release = () => {};
  const held = new Promise<void>(done => {
    release = done;
  });
  vi.mocked(daySegments).mockImplementation(async start => {
    started.push(isoDay(new Date(start * 1000)));
    await held;
    return [];
  });
  const client = new QueryClient();
  const requests = [7, 6, 5, 4, 3, 2, 1, 0].map(priority =>
    client.fetchQuery(segmentsQuery(`2026-10-${20 + priority}`, false, priority)),
  );
  try {
    await vi.waitFor(() => expect(started).toHaveLength(4));
    expect(started).toEqual(['2026-10-20', '2026-10-21', '2026-10-22', '2026-10-23']);
    await client.cancelQueries({ queryKey: segmentsQuery('2026-10-27').queryKey });
    prioritizeDayReads(['2026-10-26']); // a later jump outranks neighboring requests still waiting
    release();
    await Promise.allSettled(requests);
    expect(started).not.toContain('2026-10-27');
    expect(started).toHaveLength(7);
    expect(started[4]).toBe('2026-10-26');
  } finally {
    release();
    await Promise.allSettled(requests);
    client.clear();
    prioritizeDayReads([]);
  }
});

it('expires an inactive day chunk after twenty minutes while preserving cached data during shorter revisits', async () => {
  vi.useFakeTimers();
  const client = new QueryClient();
  const options = transcriptQuery('2026-10-27');
  client.setQueryData(options.queryKey, []);
  const observer = new QueryObserver(client, options);
  const unsubscribe = observer.subscribe(() => {});
  unsubscribe();
  try {
    await vi.advanceTimersByTimeAsync(19 * 60_000);
    expect(client.getQueryData(options.queryKey)).toEqual([]);
    await vi.advanceTimersByTimeAsync(61_000);
    expect(client.getQueryData(options.queryKey)).toBeUndefined();
  } finally {
    client.clear();
    vi.useRealTimers();
  }
});

it('refreshes a ready summary today and refreshes past activities only after explicit feedback invalidation', async () => {
  vi.useFakeTimers();
  const client = new QueryClient();
  const summaryRead = vi.fn(async () => ({ start: 0, end: 1, level: 1, status: 'ready' as const, blocks: [] }));
  const activityRead = vi.fn(async () => ({ items: [], truncated: false }));
  const summary = new QueryObserver(client, { ...dayBlocksQuery(isoDay()), queryFn: summaryRead });
  const pastOptions = activitiesQuery('2020-01-01');
  const past = new QueryObserver(client, { ...pastOptions, queryFn: activityRead });
  const stopSummary = summary.subscribe(() => {}),
    stopPast = past.subscribe(() => {});
  try {
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(summaryRead.mock.calls.length).toBeGreaterThan(1);
    expect(activityRead).toHaveBeenCalledTimes(1);
    await client.invalidateQueries({ queryKey: pastOptions.queryKey });
    expect(activityRead).toHaveBeenCalledTimes(2);
  } finally {
    stopSummary();
    stopPast();
    client.clear();
    vi.useRealTimers();
  }
});

it('rejects an oversized programmatic meaning query before requesting it', async () => {
  await expect(semanticSearch('2026-10-27', 'x'.repeat(501))).rejects.toThrow('500 characters or fewer');
});

it.each([
  { day: '2026-03-08', hours: 23 },
  { day: '2026-11-01', hours: 25 },
])('lists video through next local midnight on a $hours-hour day', async ({ day, hours }) => {
  vi.stubEnv('TZ', 'America/Los_Angeles');
  vi.mocked(daySegments).mockResolvedValue([]);
  const client = new QueryClient();
  try {
    await client.fetchQuery(segmentsQuery(day));
    const start = new Date(`${day}T00:00:00`).getTime() / 1000;
    expect(daySegments).toHaveBeenLastCalledWith(
      start,
      start + hours * 3600,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  } finally {
    client.clear();
  }
});

it.each([
  {
    kind: 'transcript',
    load: (client: QueryClient) => client.fetchQuery(transcriptQuery('2026-10-27')),
    key: ['review', 'day', '2026-10-27', 'transcript'],
    path: '/transcript/day',
    body: { entries: [] },
  },
  {
    kind: 'audio',
    load: (client: QueryClient) => client.fetchQuery(spansQuery('2026-10-27')),
    key: ['review', 'day', '2026-10-27', 'spans'],
    path: '/audio/spans',
    body: { spans: [] },
  },
])(
  'reuses an already loaded $kind day when revisited, but refreshes explicit invalidation',
  async ({ load, key, path, body }) => {
    let reads = 0;
    server.use(
      http.get(path, () => {
        reads++;
        return HttpResponse.json(body);
      }),
    );
    const client = new QueryClient();
    try {
      await load(client);
      await load(client);
      expect(reads).toBe(1);
      await client.invalidateQueries({ queryKey: key });
      await load(client);
      expect(reads).toBe(2);
    } finally {
      client.clear();
    }
  },
);

describe('getActivities (#52)', () => {
  const listing = (id: string) => ({ items: [{ id }], truncated: false });

  it('asks for every day wanted in the same turn in one request, and gives each its own listing', async () => {
    const urls: string[] = [];
    server.use(
      http.get('/activities', ({ request }) => {
        const url = new URL(request.url);
        urls.push(url.pathname + url.search);
        const days = (url.searchParams.get('days') ?? '').split(',');
        return HttpResponse.json({ days: Object.fromEntries(days.map(day => [day, listing(day)])) });
      }),
    );
    const replies = await Promise.all([
      getActivities('2026-10-27'),
      getActivities('2026-10-26'),
      getActivities('2026-10-27'),
    ]);
    expect(urls).toEqual(['/activities?days=2026-10-27,2026-10-26']);
    expect(replies.map(r => r.items[0]?.id)).toEqual(['2026-10-27', '2026-10-26', '2026-10-27']);
  });

  it('asks for a day wanted alone in the single-day form', async () => {
    const urls: string[] = [];
    server.use(
      http.get('/activities', ({ request }) => {
        const url = new URL(request.url);
        urls.push(url.pathname + url.search);
        return HttpResponse.json(listing('alone'));
      }),
    );
    expect((await getActivities('2026-10-27')).items[0]?.id).toBe('alone');
    expect(urls).toEqual(['/activities?day=2026-10-27']);
  });

  it("fails every day of a request that failed, with the server's message", async () => {
    server.use(http.get('/activities', () => HttpResponse.json({ error: 'day must be YYYY-MM-DD' }, { status: 400 })));
    const results = await Promise.allSettled([getActivities('2026-10-27'), getActivities('nope')]);
    expect(results.map(r => r.status)).toEqual(['rejected', 'rejected']);
    expect((results[0] as PromiseRejectedResult).reason).toMatchObject({ message: 'day must be YYYY-MM-DD' });
  });

  it('gives up a cancelled ask without failing the others', async () => {
    server.use(http.get('/activities', () => HttpResponse.json(listing('kept'))));
    const abort = new AbortController();
    const dropped = getActivities('2026-10-26', abort.signal);
    const kept = getActivities('2026-10-27');
    abort.abort();
    await expect(dropped).rejects.toMatchObject({ name: 'AbortError' });
    expect((await kept).items[0]?.id).toBe('kept'); // the day left is asked for alone
  });
});
