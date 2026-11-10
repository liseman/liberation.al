// Live data from GET /events into the query cache, one query per kind (['live', kind]). While the page is visible
// it keeps one EventSource open. After 4 errors, or 40 s without a ping, it polls each kind's URL every 3 s until
// the stream is back. The polled responses have the same shapes as the events.
import { type QueryClient, queryOptions } from '@tanstack/react-query';
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

export const POLL_MS = 3000;
export const QUIET_MS = 40_000; // the server pings every 15 s, so this much silence means the stream is dead
export const WATCHDOG_MS = 10_000;
export const NEW_MS = 4000; // how long a new transcript line stays highlighted
export const TICK_MS = 5000;

export const liveKey = (kind: LiveKind) => ['live', kind] as const;

// How many stream events each kind has had, per cache. A fetch that saw this change while it was in flight holds
// an older snapshot than the cache, so it keeps what the stream sent instead (a reset snapshot would wipe it).
const eventCounts = new WeakMap<QueryClient, Partial<Record<LiveKind, number>>>();
const eventsSoFar = (client: QueryClient, kind: LiveKind): number => eventCounts.get(client)?.[kind] ?? 0;

/** The query for one kind: fetched from its poll URL when nothing has arrived yet, else kept fresh by the feed. */
export function liveQuery<K extends LiveKind>(
  client: QueryClient,
  kind: K,
  priority: ReadPriority = kind === 'ambient' || kind === 'vision' ? 'background' : 'page',
) {
  return queryOptions<LiveData[K]>({
    queryKey: liveKey(kind),
    queryFn: async ({ signal }) => {
      const before = eventsSoFar(client, kind);
      const data = await request(POLL_URL[kind], undefined, signal, priority);
      const now = client.getQueryData<LiveData[K]>(liveKey(kind));
      if (eventsSoFar(client, kind) !== before && now !== undefined) return now;
      return reduce(kind, now, data);
    },
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

  const pollAll = () => {
    if (document.hidden) return;
    for (const kind of KINDS)
      client.fetchQuery({ ...liveQuery(client, kind), staleTime: 0 }).catch(() => {
        // the next poll tries again
      });
  };
  const startPolling = () => {
    if (polling != null || document.hidden) return;
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
  };

  const connect = () => {
    clearReconnect();
    close();
    if (stopped || document.hidden) return;
    if (typeof EventSource === 'undefined') return startPolling();
    const mine = new EventSource('/events');
    es = mine;
    heard = Date.now();
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
