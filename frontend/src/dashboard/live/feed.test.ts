import { QueryClient } from '@tanstack/react-query';
import { delay, HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { server } from '../../test/server';
import { applyEvent, liveKey, liveQuery, NEW_MS, POLL_MS, QUIET_MS, startLive, WATCHDOG_MS } from './feed';
import { useLiveStatus } from './status';
import type { Transcript } from './types';

/** Just enough EventSource: tests open, error, close, and send events on the newest one. */
class FakeEventSource {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;
  static all: FakeEventSource[] = [];
  readyState = FakeEventSource.CONNECTING;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  listeners: Record<string, ((e: MessageEvent<string>) => void)[]> = {};
  readonly url: string;
  constructor(url: string) {
    this.url = url;
    FakeEventSource.all.push(this);
  }
  addEventListener(type: string, fn: (e: MessageEvent<string>) => void) {
    this.listeners[type] = [...(this.listeners[type] ?? []), fn];
  }
  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
  open() {
    this.readyState = FakeEventSource.OPEN;
    this.onopen?.();
  }
  fail(closed = false) {
    if (closed) this.readyState = FakeEventSource.CLOSED;
    this.onerror?.();
  }
  send(type: string, data: unknown) {
    for (const fn of this.listeners[type] ?? []) fn(new MessageEvent(type, { data: JSON.stringify(data) }));
  }
  static get last() {
    const es = FakeEventSource.all.at(-1);
    if (!es) throw new Error('no EventSource was opened');
    return es;
  }
}

let client: QueryClient;
let stop: () => void;
let polls: string[];

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
  vi.stubGlobal('EventSource', FakeEventSource);
  FakeEventSource.all = [];
  polls = [];
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  useLiveStatus.setState({ conn: { state: '', text: 'connecting' } });
  // Every poll URL answers, and says which was asked.
  server.use(
    http.get('*', ({ request }) => {
      const u = new URL(request.url);
      polls.push(u.pathname + u.search);
      if (u.pathname === '/scene') return HttpResponse.json([{ cam: 'cam0', ts: 5 }]);
      if (u.pathname === '/transcript') return HttpResponse.json({ entries: [], size: 0, reset: true });
      return HttpResponse.json({});
    }),
  );
});

afterEach(() => {
  stop?.();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const conn = () => useLiveStatus.getState().conn;

describe('startLive', () => {
  it('writes each event into its query and says live once the stream opens', () => {
    stop = startLive(client);
    const es = FakeEventSource.last;
    expect(es.url).toBe('/events');
    es.open();
    es.send('scene', [{ cam: 'cam1', ts: 9 }]);
    es.send('transcript', { entries: [{ ts_start: 1, text: 'hi' }], reset: true });
    expect(conn()).toEqual({ state: 'on', text: 'live' });
    expect(client.getQueryData(liveKey('scene'))).toEqual([{ cam: 'cam1', ts: 9 }]);
    expect(client.getQueryData<Transcript>(liveKey('transcript'))?.entries.map(e => e.text)).toEqual(['hi']);
  });

  it('reconnects once after the stream closes', () => {
    stop = startLive(client);
    const first = FakeEventSource.last;
    first.open();
    first.fail(true);
    first.fail(true); // a second error on the same outage must not stack another reconnect
    expect(conn()).toEqual({ state: 'bad', text: 'reconnecting' });
    vi.advanceTimersByTime(2000);
    expect(FakeEventSource.all).toHaveLength(2);
    FakeEventSource.last.open();
    expect(conn().state).toBe('on');
  });

  it('polls every kind after four errors, and stops polling when the stream is back', async () => {
    stop = startLive(client);
    const es = FakeEventSource.last;
    for (let i = 0; i < 4; i++) es.fail();
    expect(conn()).toEqual({ state: 'bad', text: 'polling' });
    await vi.waitFor(() => expect(polls).toContain('/scene'));
    expect(new Set(polls)).toEqual(new Set(['/transcript?n=80', '/scene', '/speakers', '/ambient', '/vision']));
    expect(client.getQueryData(liveKey('scene'))).toEqual([{ cam: 'cam0', ts: 5 }]);

    es.open();
    const before = polls.length;
    await vi.advanceTimersByTimeAsync(POLL_MS * 3);
    expect(polls.length).toBe(before);
  });

  it('treats 40 s without a ping as an outage, even with no error', async () => {
    stop = startLive(client);
    FakeEventSource.last.open();
    await vi.advanceTimersByTimeAsync(QUIET_MS); // checks at 10, 20, 30, and 40 s: not yet more than 40 s quiet
    expect(FakeEventSource.all).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(WATCHDOG_MS); // the check at 50 s finds it silent
    expect(conn().text).toBe('polling');
    expect(FakeEventSource.all).toHaveLength(2);
  });

  it('a ping keeps a quiet stream alive', async () => {
    stop = startLive(client);
    const es = FakeEventSource.last;
    es.open();
    for (let i = 0; i < 6; i++) {
      await vi.advanceTimersByTimeAsync(WATCHDOG_MS);
      es.send('ping', {});
    }
    expect(FakeEventSource.all).toHaveLength(1);
    expect(conn().state).toBe('on');
  });

  it('closes the stream and its timers when stopped', async () => {
    stop = startLive(client);
    const es = FakeEventSource.last;
    stop();
    expect(es.readyState).toBe(FakeEventSource.CLOSED);
    await vi.advanceTimersByTimeAsync(QUIET_MS * 2);
    expect(FakeEventSource.all).toHaveLength(1);
    expect(polls).toEqual([]);
  });
});

describe('applyEvent', () => {
  it('highlights new lines for a few seconds, but not a reset', () => {
    applyEvent(client, 'transcript', { entries: [{ ts_start: 1 }], reset: true });
    applyEvent(client, 'transcript', { entries: [{ ts_start: 2 }] });
    const highlighted = () => client.getQueryData<Transcript>(liveKey('transcript'))?.entries.map(e => !!e._new);
    expect(highlighted()).toEqual([false, true]);
    vi.advanceTimersByTime(NEW_MS);
    expect(highlighted()).toEqual([false, false]);
  });
});

describe('a snapshot that started before the stream', () => {
  it('never overwrites lines the stream sent while it was in flight', async () => {
    vi.useRealTimers();
    let answer: () => void = () => {};
    const answered = new Promise<void>(r => {
      answer = r;
    });
    server.use(
      http.get('/transcript', async () => {
        await answered;
        return HttpResponse.json({ entries: [{ ts_start: 1, text: 'old snapshot' }], size: 10, reset: true });
      }),
    );
    const snapshot = client.fetchQuery(liveQuery(client, 'transcript'));
    await delay(10); // the request is out
    applyEvent(client, 'transcript', { entries: [{ ts_start: 1, text: 'from the stream' }], reset: true });
    applyEvent(client, 'transcript', { entries: [{ ts_start: 2, text: 'newer line' }] });
    answer();
    await snapshot;
    const texts = client.getQueryData<Transcript>(liveKey('transcript'))?.entries.map(e => e.text);
    expect(texts).toEqual(['from the stream', 'newer line']);
  });
});

describe('scene read priority', () => {
  it('keeps a Live scene request in the page lane beside a slow ambient poll', async () => {
    const replies = new Map<string, (response: Response) => void>();
    const fetcher = vi.fn<typeof fetch>(
      (url, init) =>
        new Promise<Response>((resolve, reject) => {
          replies.set(String(url), resolve);
          init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), {
            once: true,
          });
        }),
    );
    vi.stubGlobal('fetch', fetcher);
    const ambient = client.fetchQuery(liveQuery(client, 'ambient')).catch((error: unknown) => error);
    const scene = client.fetchQuery(liveQuery(client, 'scene')).catch((error: unknown) => error);
    try {
      await Promise.resolve();
      expect(fetcher.mock.calls.map(([url]) => url)).toEqual(['/ambient', '/scene']);
      replies.get('/scene')?.(Response.json([{ cam: 'cam0', ts: 5 }]));
      expect(await scene).toEqual([{ cam: 'cam0', ts: 5 }]);
      replies.get('/ambient')?.(Response.json({}));
      await ambient;
    } finally {
      await client.cancelQueries();
    }
  });
});
