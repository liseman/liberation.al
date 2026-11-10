// Live data from GET /events into the query cache, one query per kind (['live', kind]). While the page is visible
// it keeps one EventSource open. After 4 errors, or 40 s without a ping, it polls each kind's URL every 3 s until
// the stream is back. The polled responses have the same shapes as the events.
//
// The stream also carries the shell's status (#52): the privacy pauses, the services badge, and the Inbox's research
// jobs, each written to the query its own pages read (STATUS). A stream-fed query asks its URL only when the stream
// can't answer: nothing has started a feed, the feed is polling, or the stream's opening snapshot came without it.
import { type QueryClient, type QueryKey, queryOptions } from '@tanstack/react-query';
import { type ReadPriority, request } from '../../shared/http';
import { reduce, settle } from './reduce';
import { type Connection, useLiveStatus } from './status';
import type { LiveData, LiveKind, TranscriptEvent } from './types';

export const POLL_URL: Record<LiveKind, string> = {
  transcript: '/transcript?n=80',
  scene: '/scene',
  speakers: '/speakers',
  ambient: '/ambient',
  vision: '/vision',
};
const KINDS = Object.keys(POLL_URL) as LiveKind[];

/** The shell's status kinds: the stream event's name, the query it feeds, and the URL that answers the same. */
export const STATUS = {
  privacy: { key: ['privacy'], url: '/devices/privacy' },
  services: { key: ['services'], url: '/services' },
  research: { key: ['research'], url: '/research/jobs' },
} as const satisfies Record<string, { key: QueryKey; url: string }>;
export type StatusKind = keyof typeof STATUS;
const STATUS_KINDS = Object.keys(STATUS) as StatusKind[];

export const POLL_MS = 3000;
export const STATUS_POLL_TICKS = 5; // while polling, the shell's status is asked every fifth tick (15 s)
export const QUIET_MS = 40_000; // the server pings every 15 s, so this much silence means the stream is dead
export const WATCHDOG_MS = 10_000;
export const NEW_MS = 4000; // how long a new transcript line stays highlighted
export const TICK_MS = 5000;
export const FIRST_EVENT_MS = 5000; // how long a query waits for the stream's opening snapshot before asking its URL

export const liveKey = (kind: LiveKind) => ['live', kind] as const;

// How many stream events each kind has had, per cache. A fetch that saw this change while it was in flight holds
// an older snapshot than the cache, so it keeps what the stream sent instead (a reset snapshot would wipe it).
const eventCounts = new WeakMap<QueryClient, Partial<Record<LiveKind, number>>>();
const eventsSoFar = (client: QueryClient, kind: LiveKind): number => eventCounts.get(client)?.[kind] ?? 0;

// Each cache's stream: whether its opening snapshot is still on the way, and the queries waiting for it.
interface Opening {
  expected: boolean;
  waiters: Set<() => void>;
}
const openings = new WeakMap<QueryClient, Opening>();
function opening(client: QueryClient): Opening {
  let state = openings.get(client);
  if (!state) {
    state = { expected: false, waiters: new Set() };
    openings.set(client, state);
  }
  return state;
}
/** The stream's opening snapshot is expected (a stream was just opened) or isn't any more (it arrived, or the stream
 *  is down): either way every waiting query looks again. */
function expectOpening(client: QueryClient, expected: boolean): void {
  opening(client).expected = expected;
  wake(client);
}
/** Every query waiting on the stream looks at the cache again. */
function wake(client: QueryClient): void {
  for (const look of [...opening(client).waiters]) look();
}

/** Wait for the stream to fill `key`, while its opening snapshot is on the way; true when it did. */
async function filledByStream(client: QueryClient, key: QueryKey, signal?: AbortSignal): Promise<boolean> {
  const state = opening(client);
  const deadline = Date.now() + FIRST_EVENT_MS;
  while (client.getQueryData(key) === undefined && state.expected && !signal?.aborted && Date.now() < deadline)
    await new Promise<void>(resolve => {
      const done = () => {
        clearTimeout(timer);
        state.waiters.delete(done);
        signal?.removeEventListener('abort', done);
        resolve();
      };
      const timer = setTimeout(done, Math.max(0, deadline - Date.now()));
      state.waiters.add(done);
      signal?.addEventListener('abort', done, { once: true });
    });
  return client.getQueryData(key) !== undefined;
}

/** A stream-fed query's read: what the stream put in the cache when this is the first read and the stream is about
 *  to answer it, else `get()` (a refetch after a change, the polling fallback, or no stream). */
export function streamed<T>(
  client: QueryClient,
  key: QueryKey,
  get: () => Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  if (client.getQueryData(key) !== undefined || !opening(client).expected) return get();
  return filledByStream(client, key, signal).then(filled => (filled ? (client.getQueryData<T>(key) as T) : get()));
}

/** The query options of one of the shell's status kinds: fed by the stream, never polled by the page. */
export function statusQuery<T>(kind: StatusKind, priority: ReadPriority = 'background') {
  return queryOptions<T>({
    queryKey: STATUS[kind].key,
    queryFn: ({ client, signal }) =>
      streamed(client, STATUS[kind].key, () => request<T>(STATUS[kind].url, undefined, signal, priority), signal),
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
  });
}

/** The query for one kind: kept fresh by the feed, and fetched from its poll URL when the stream can't fill it. */
export function liveQuery<K extends LiveKind>(
  client: QueryClient,
  kind: K,
  priority: ReadPriority = kind === 'ambient' || kind === 'vision' ? 'background' : 'page',
) {
  return queryOptions<LiveData[K]>({
    queryKey: liveKey(kind),
    queryFn: ({ signal }) =>
      streamed(
        client,
        liveKey(kind),
        async () => {
          const before = eventsSoFar(client, kind);
          const data = await request(POLL_URL[kind], undefined, signal, priority);
          const now = client.getQueryData<LiveData[K]>(liveKey(kind));
          if (eventsSoFar(client, kind) !== before && now !== undefined) return now;
          return reduce(kind, now, data);
        },
        signal,
      ),
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
  });
}

/** Apply one event of a kind to the cache. A new transcript line loses its highlight after NEW_MS. */
export function applyEvent(client: QueryClient, kind: LiveKind, data: unknown): void {
  const counts = eventCounts.get(client) ?? {};
  counts[kind] = (counts[kind] ?? 0) + 1;
  eventCounts.set(client, counts);
  client.setQueryData<LiveData[LiveKind]>(liveKey(kind), prev => reduce(kind, prev as never, data));
  wake(client);
  const t = data as TranscriptEvent | null;
  if (kind === 'transcript' && t?.entries?.length && !t.reset && !t.replace)
    setTimeout(() => client.setQueryData<LiveData['transcript']>(liveKey('transcript'), p => p && settle(p)), NEW_MS);
}

/** Start the feed; returns a function that stops it. */
export function startLive(client: QueryClient): () => void {
  const setConn = (conn: Connection) => useLiveStatus.setState({ conn });
  let es: EventSource | null = null;
  let errors = 0;
  let polling: ReturnType<typeof setInterval> | null = null;
  let reconnect: ReturnType<typeof setTimeout> | null = null;
  let heard = Date.now();
  let stopped = false;

  let ticks = 0;
  const pollAll = () => {
    if (document.hidden) return;
    const again = () => {
      // the next poll tries again
    };
    for (const kind of KINDS) client.fetchQuery({ ...liveQuery(client, kind), staleTime: 0 }).catch(again);
    if (ticks++ % STATUS_POLL_TICKS === 0)
      for (const kind of STATUS_KINDS) client.fetchQuery({ ...statusQuery(kind), staleTime: 0 }).catch(again);
  };
  const startPolling = () => {
    if (polling != null || document.hidden) return;
    expectOpening(client, false);
    ticks = 0;
    setConn({ state: 'bad', text: 'polling' });
    pollAll();
    polling = setInterval(pollAll, POLL_MS);
  };
  const stopPolling = () => {
    if (polling != null) clearInterval(polling);
    polling = null;
  };
  const clearReconnect = () => {
    if (reconnect != null) clearTimeout(reconnect);
    reconnect = null;
  };
  const close = () => {
    es?.close();
    es = null;
    expectOpening(client, false);
  };

  const connect = () => {
    clearReconnect();
    close();
    if (stopped || document.hidden) return;
    if (typeof EventSource === 'undefined') return startPolling();
    const mine = new EventSource('/events');
    es = mine;
    heard = Date.now();
    if (polling == null) expectOpening(client, true);
    mine.onopen = () => {
      if (es !== mine) return;
      clearReconnect();
      errors = 0;
      setConn({ state: 'on', text: 'live' });
      stopPolling();
    };
    mine.onerror = () => {
      if (es !== mine || document.hidden) return;
      errors++;
      setConn({ state: 'bad', text: polling != null ? 'polling' : 'reconnecting' });
      if (errors >= 4 && polling == null) startPolling();
      // One outage gets one retry: old streams' callbacks and repeated errors must not stack reconnects.
      if (mine.readyState === EventSource.CLOSED && reconnect == null) reconnect = setTimeout(connect, 2000);
    };
    const hear = () => {
      heard = Date.now();
    };
    mine.addEventListener('ping', hear);
    mine.addEventListener('hello', hear);
    for (const kind of KINDS)
      mine.addEventListener(kind, e => {
        hear();
        applyEvent(client, kind, JSON.parse((e as MessageEvent<string>).data));
      });
    for (const kind of STATUS_KINDS)
      mine.addEventListener(kind, e => {
        hear();
        client.setQueryData(STATUS[kind].key, JSON.parse((e as MessageEvent<string>).data));
        wake(client);
      });
    // The opening snapshot is all in: a query still waiting (research, for a viewer it isn't sent to) asks its URL.
    mine.addEventListener('ready', () => {
      hear();
      if (es === mine) expectOpening(client, false);
    });
  };

  const onVisibility = () => {
    if (document.visibilityState === 'visible') connect();
    else {
      clearReconnect();
      stopPolling();
      close();
    }
  };
  document.addEventListener('visibilitychange', onVisibility);
  connect();

  // Watchdog: reconnect when the stream closed or went quiet while the page is visible.
  const watchdog = setInterval(() => {
    if (document.visibilityState !== 'visible' || typeof EventSource === 'undefined') return;
    const quiet = Date.now() - heard > QUIET_MS;
    if (!es || es.readyState === EventSource.CLOSED || quiet) {
      errors++;
      setConn({ state: 'bad', text: polling != null ? 'polling' : 'reconnecting' });
      // A silent stream may never fire onerror. Forty seconds without a ping is already an outage.
      if (quiet || errors >= 4) startPolling();
      connect();
    }
  }, WATCHDOG_MS);
  const tick = setInterval(() => useLiveStatus.setState({ now: Date.now() / 1000 }), TICK_MS);

  return () => {
    stopped = true;
    document.removeEventListener('visibilitychange', onVisibility);
    clearInterval(watchdog);
    clearInterval(tick);
    clearReconnect();
    stopPolling();
    close();
  };
}
