// GET /logs/stream into the query cache (DEBUG_KEY) while the Debug tab is open. A log tail needs none of the live
// feed's polling fallback: EventSource reconnects by itself after a dropped connection, and a backlog sent again after
// a reconnect only adds what is new (logs.ts, receive). An HTTP error (a 5xx, or a proxy's 502 while the dashboard
// restarts) closes it for good, so then the tab opens a new one after RETRY_MS.
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { DEBUG_KEY, EMPTY, type Hello, hello, type LogState, receive, type WireLine } from './logs';

export const STREAM_URL = '/logs/stream';
export const RETRY_MS = 2000;
const CLOSED = 2; // EventSource.CLOSED, without needing EventSource to exist

export type StreamState = 'connecting' | 'open' | 'retrying';

function parse<T>(data: string): T | null {
  try {
    return JSON.parse(data) as T;
  } catch {
    return null;
  }
}

/** The buffer and the connection's state, with the stream open while the calling component is mounted. */
export function useLogStream(): { logs: LogState; conn: StreamState } {
  const client = useQueryClient();
  const [conn, setConn] = useState<StreamState>('connecting');
  useEffect(() => {
    const update = (fn: (s: LogState) => LogState) => client.setQueryData<LogState>(DEBUG_KEY, s => fn(s ?? EMPTY));
    let es: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      retry = undefined;
      const mine = new EventSource(STREAM_URL);
      es = mine;
      mine.onopen = () => setConn('open');
      mine.onerror = () => {
        setConn('retrying');
        if (mine.readyState !== CLOSED || retry !== undefined) return; // the browser is reconnecting by itself
        mine.close();
        retry = setTimeout(connect, RETRY_MS);
      };
      mine.addEventListener('hello', e => {
        const h = parse<Hello>(e.data);
        if (h) update(s => hello(s, h));
        setConn('open');
      });
      mine.addEventListener('lines', e => {
        const batch = parse<WireLine[]>(e.data);
        if (Array.isArray(batch)) update(s => receive(s, batch));
      });
    };
    connect();
    return () => {
      clearTimeout(retry);
      es?.close();
    };
  }, [client]);
  const logs =
    useQuery({
      queryKey: DEBUG_KEY,
      queryFn: () => EMPTY,
      enabled: false,
      initialData: EMPTY,
      staleTime: Number.POSITIVE_INFINITY,
    }).data ?? EMPTY;
  return { logs, conn };
}

/** Change the buffer from a handler (pause, resume, clear). */
export function useLogAction(): (fn: (s: LogState) => LogState) => void {
  const client = useQueryClient();
  return fn => client.setQueryData<LogState>(DEBUG_KEY, s => fn(s ?? EMPTY));
}
