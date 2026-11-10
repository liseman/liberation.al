import { QueryClient } from '@tanstack/react-query';
import { HttpResponse, http } from 'msw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { liveQuery } from '../dashboard/live/feed';
import { BADGE_POLL_MS, servicesQuery } from '../dashboard/settings/services/api';
import { unknownVoicesQuery } from '../dashboard/whosthis/api';
import { server } from '../test/server';
import { HttpError, RECOVER_MS, request, TIMEOUT_MS, upload } from './http';
import { newQueryClient } from './Providers';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('request', () => {
  it('keeps four browser slots free while slow JSON reads wait, and does not queue writes', async () => {
    const replies: ((r: Response) => void)[] = [];
    const fetcher = vi.fn<typeof fetch>(() => new Promise<Response>(resolve => replies.push(resolve)));
    vi.stubGlobal('fetch', fetcher);
    const reads = [request('/people'), request('/faces/unknown/groups'), request('/voices/unknown/groups')];
    await Promise.resolve();
    expect(fetcher).toHaveBeenCalledTimes(2);
    const write = request('/save', {});
    expect(fetcher).toHaveBeenCalledTimes(3);
    replies[2]?.(Response.json({ ok: true }));
    await write;
    replies[0]?.(Response.json([]));
    await reads[0];
    expect(fetcher).toHaveBeenCalledTimes(4);
    replies[1]?.(Response.json([]));
    replies[3]?.(Response.json([]));
    await Promise.all(reads);
  });

  it('starts page data while two slow status polls are pending, within the two-read budget', async () => {
    const replies: ((r: Response) => void)[] = [];
    const fetcher = vi.fn<typeof fetch>(() => new Promise<Response>(resolve => replies.push(resolve)));
    vi.stubGlobal('fetch', fetcher);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    const ambient = client.fetchQuery(liveQuery(client, 'ambient'));
    const services = servicesQuery(BADGE_POLL_MS).queryFn({});
    const people = request('/people');
    await Promise.resolve();
    expect(fetcher.mock.calls.map(args => args[0])).toEqual(['/ambient', '/people']);
    replies[1]?.(Response.json({}));
    await people;
    // A second poll cannot take the slot reserved for page data while the first poll is slow.
    expect(fetcher).toHaveBeenCalledTimes(2);
    const nextPage = request('/clones');
    await Promise.resolve();
    expect(fetcher).toHaveBeenLastCalledWith('/clones', expect.anything());
    replies[2]?.(Response.json({}));
    await nextPage;
    replies[0]?.(Response.json({}));
    await ambient;
    expect(fetcher).toHaveBeenLastCalledWith('/services', expect.anything());
    replies[3]?.(Response.json({}));
    await services;
    client.clear();
  });

  it('starts queued page data before queued background polls', async () => {
    const replies: ((r: Response) => void)[] = [];
    const fetcher = vi.fn<typeof fetch>(() => new Promise<Response>(resolve => replies.push(resolve)));
    vi.stubGlobal('fetch', fetcher);
    const first = request('/first');
    const second = request('/second');
    const poll = request('/services', undefined, undefined, 'background');
    const page = request('/people');
    await Promise.resolve();
    replies[0]?.(Response.json({}));
    await first;
    expect(fetcher).toHaveBeenLastCalledWith('/people', expect.anything());
    replies[1]?.(Response.json({}));
    await second;
    expect(fetcher).toHaveBeenLastCalledWith('/services', expect.anything());
    replies[2]?.(Response.json({}));
    replies[3]?.(Response.json({}));
    await Promise.all([poll, page]);
  });

  it.each(['page', 'background'] as const)(
    'removes cancelled queued %s reads without consuming a slot',
    async priority => {
      const replies: ((r: Response) => void)[] = [];
      const fetcher = vi.fn<typeof fetch>(() => new Promise<Response>(resolve => replies.push(resolve)));
      vi.stubGlobal('fetch', fetcher);
      const a = request('/a');
      const b = request('/b');
      const ctrl = new AbortController();
      const cancelled = request('/cancelled', undefined, ctrl.signal, priority).catch((e: unknown) => e);
      const next = request('/next');
      ctrl.abort();
      expect(await cancelled).toMatchObject({ name: 'AbortError' });
      replies[0]?.(Response.json({}));
      await a;
      expect(fetcher.mock.calls).toHaveLength(3);
      expect(fetcher).toHaveBeenLastCalledWith('/next', expect.anything());
      replies[1]?.(Response.json({}));
      replies[2]?.(Response.json({}));
      await Promise.all([b, next]);
    },
  );

  it('posts the body as JSON and returns the parsed response', async () => {
    let sent: unknown;
    server.use(
      http.post('/thing', async ({ request }) => {
        sent = { type: request.headers.get('Content-Type'), body: await request.json() };
        return HttpResponse.json({ ok: true });
      }),
    );
    await expect(request('/thing', { a: 1 })).resolves.toEqual({ ok: true });
    expect(sent).toEqual({ type: 'application/json', body: { a: 1 } });
  });

  it('sends a Blob as is, with its own type', async () => {
    let sent: unknown;
    server.use(
      http.post('/clones/record', async ({ request }) => {
        sent = { type: request.headers.get('Content-Type'), bytes: [...new Uint8Array(await request.arrayBuffer())] };
        return HttpResponse.json({ ok: true });
      }),
    );
    await request('/clones/record', new Blob([new Uint8Array([82, 73, 70, 70])], { type: 'audio/wav' }));
    expect(sent).toEqual({ type: 'audio/wav', bytes: [82, 73, 70, 70] });
  });

  it("throws the server's own error message, with the body", async () => {
    const body = { ok: false, error: 'Camera is off.', services: [] };
    server.use(http.get('/thing', () => HttpResponse.json(body, { status: 400 })));
    const error = await request('/thing').catch((e: unknown) => e);
    expect(error).toBeInstanceOf(HttpError);
    expect(error).toMatchObject({ message: 'Camera is off.', reason: 'Camera is off.', status: 400, data: body });
  });

  it('reads a detail message when there is no error', async () => {
    server.use(
      http.post('/calls/place', () => HttpResponse.json({ ok: false, detail: 'Already on a call' }, { status: 409 })),
    );
    await expect(request('/calls/place', {})).rejects.toMatchObject({ reason: 'Already on a call', status: 409 });
  });

  it.each([
    ['not JSON', () => new HttpResponse('Bad gateway', { status: 502 })],
    ['an empty error', () => HttpResponse.json({ error: '' }, { status: 502 })],
  ])('falls back to the status for %s, with no reason', async (_, reply) => {
    server.use(http.get('/thing', reply));
    await expect(request('/thing')).rejects.toMatchObject({ message: 'Request failed (502)', reason: null });
  });

  it('gives up after the timeout', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    server.use(http.get('/slow', () => new Promise<never>(() => {})));
    const pending = request('/slow').catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS);
    expect(await pending).toMatchObject({ message: 'The server took too long to answer' });
  });

  it('times a read from when it gets a slot, not from when it was queued (#2022)', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const replies: ((r: Response) => void)[] = [];
    // Like a real fetch, an aborted one rejects: that's how the deadline ends it.
    const fetcher = (_url: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((resolve, reject) => {
        replies.push(resolve);
        init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
      });
    vi.stubGlobal('fetch', vi.fn<typeof fetch>(fetcher));
    const slow = [request('/people'), request('/faces/unknown/groups')]; // both read slots, for 15 s
    const queued = request('/voices/unknown/groups', undefined, undefined, 'page', false).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(15_000);
    replies[0]?.(Response.json([]));
    replies[1]?.(Response.json([]));
    await Promise.all(slow);
    await vi.advanceTimersByTimeAsync(10_000); // 25 s after it was asked for, 10 s after it got a slot
    replies[2]?.(Response.json({ groups: [] }));
    expect(await queued).toEqual({ groups: [] });
  });

  it('lets a POST run past the timeout: saves wait on service restarts', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    let reply: (r: Response) => void = () => {};
    server.use(http.post('/devices', () => new Promise<Response>(r => (reply = r))));
    const pending = request('/devices', { mics: [], cameras: [] });
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS * 6);
    reply(HttpResponse.json({ ok: true, restarted: ['ambient'] }));
    await expect(pending).resolves.toEqual({ ok: true, restarted: ['ambient'] });
  });

  it("stops when the caller's signal aborts", async () => {
    server.use(http.get('/slow', () => new Promise<never>(() => {})));
    const caller = new AbortController();
    const pending = request('/slow', undefined, caller.signal).catch((e: unknown) => e);
    caller.abort();
    expect(await pending).toMatchObject({ name: 'AbortError' });
  });
});

// A restart of the dashboard drops connections for a few seconds: fetch rejects with a TypeError (Firefox's
// "NetworkError when attempting to fetch resource."). A stubbed fetch, so fake timers drive every step.
describe('request through a dashboard restart', () => {
  /** A fetch where /health refuses `down` times, then answers, and `path` refuses until /health has answered. */
  function restarting(path: string, down: number, data: unknown = { ok: true }) {
    const calls: string[] = [];
    let up = false;
    const fetcher = vi.fn<typeof fetch>(async input => {
      const url = new URL(String(input), 'http://localhost').pathname;
      calls.push(url);
      if (url === '/health' && down-- <= 0) up = true;
      if (url !== '/health' && url !== path) return Response.json({ error: 'not found' }, { status: 404 });
      if (!up) throw new TypeError('NetworkError when attempting to fetch resource.');
      return Response.json(url === '/health' ? { boot: 'b2' } : data);
    });
    vi.stubGlobal('fetch', fetcher);
    return calls;
  }

  it('waits for /health to answer, then asks again', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const calls = restarting('/people', 2, [{ name: 'Dana' }]);
    const pending = request('/people');
    await vi.advanceTimersByTimeAsync(5000);
    await expect(pending).resolves.toEqual([{ name: 'Dana' }]);
    expect(calls).toEqual(['/people', '/health', '/health', '/health', '/people']);
  });

  it('gives up with the network error once the dashboard stays down past RECOVER_MS', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    restarting('/people', Number.POSITIVE_INFINITY);
    const pending = request('/people').catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(RECOVER_MS + 5000);
    expect(await pending).toBeInstanceOf(TypeError);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('asks again while the server answers "starting" (its broker restarting), then has the data', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    let refusals = 2;
    const fetcher = vi.fn<typeof fetch>(async () =>
      refusals-- > 0
        ? Response.json({ ok: false, error: 'The data broker is starting.', starting: true }, { status: 503 })
        : Response.json([{ name: 'Dana' }]),
    );
    vi.stubGlobal('fetch', fetcher);
    const pending = request('/people');
    await vi.advanceTimersByTimeAsync(2000);
    await expect(pending).resolves.toEqual([{ name: 'Dana' }]);
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it('gives up with the "starting" refusal once it lasts past RECOVER_MS', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    const starting = { ok: false, error: 'The data broker is starting.', starting: true };
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>(async () => Response.json(starting, { status: 503 })),
    );
    const pending = request('/people').catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(RECOVER_MS + 5000);
    expect(await pending).toMatchObject({ status: 503, reason: 'The data broker is starting.' });
    expect(vi.getTimerCount()).toBe(0);
  });

  it('shows a 503 without "starting" at once', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => Response.json({ error: 'down' }, { status: 503 }));
    vi.stubGlobal('fetch', fetcher);
    await expect(request('/people')).rejects.toMatchObject({ status: 503 });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('shows an error from the server at once', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => Response.json({ error: 'broken' }, { status: 500 }));
    vi.stubGlobal('fetch', fetcher);
    await expect(request('/people')).rejects.toMatchObject({ status: 500, reason: 'broken' });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('does not send a POST twice: it may have reached the server', async () => {
    let posts = 0;
    server.use(
      http.post('/save', () => {
        posts++;
        return HttpResponse.error();
      }),
    );
    await expect(request('/save', {})).rejects.toBeInstanceOf(TypeError);
    expect(posts).toBe(1);
  });

  it('can be told not to wait, for a caller that polls /health itself', async () => {
    const calls = restarting('/people', 1);
    await expect(request('/people', undefined, undefined, 'page', false)).rejects.toBeInstanceOf(TypeError);
    expect(calls).toEqual(['/people']);
  });

  it("stops waiting when the caller's signal aborts, and leaves no timer behind", async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    restarting('/people', Number.POSITIVE_INFINITY);
    const caller = new AbortController();
    const pending = request('/people', undefined, caller.signal).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1000);
    caller.abort();
    expect(await pending).toMatchObject({ name: 'AbortError' });
    expect(vi.getTimerCount()).toBe(0);
  });

  it('carries a retry: false query through, keeping its last data while it waits', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const client = newQueryClient();
    const voices = { groups: [{ id: 1 }], count: 1, skipped: [], pending: false };
    vi.stubGlobal('fetch', async () => Response.json(voices));
    const before = await client.fetchQuery(unknownVoicesQuery());
    restarting('/voices/unknown/groups', 2, { ...voices, count: 2, groups: [{ id: 1 }, { id: 2 }] });
    const refetch = client.refetchQueries({ queryKey: unknownVoicesQuery().queryKey });
    await vi.advanceTimersByTimeAsync(500);
    const state = client.getQueryState(unknownVoicesQuery().queryKey);
    expect([state?.status, state?.fetchStatus, client.getQueryData(unknownVoicesQuery().queryKey)]).toEqual([
      'success',
      'fetching',
      before,
    ]);
    await vi.advanceTimersByTimeAsync(5000);
    await refetch;
    expect(client.getQueryData(unknownVoicesQuery().queryKey)).toMatchObject({ count: 2 });
    client.clear();
  });
});

describe('newQueryClient', () => {
  const retry = newQueryClient().getDefaultOptions().queries?.retry as (n: number, e: Error) => boolean;

  it.each([
    ['a server error once', new HttpError('broken', 500), 0, true],
    ['a server error only once', new HttpError('broken', 500), 1, false],
    ['a network failure request already waited out', new TypeError('NetworkError'), 0, false],
  ])('retries %s', (_, error, failures, expected) => {
    expect(retry(failures, error)).toBe(expected);
  });
});

describe('upload', () => {
  it('sends the file as the body, with its type, and gives back the JSON', async () => {
    let got: { type: string | null; body: string } | null = null;
    server.use(
      http.post('/up', async ({ request: req }) => {
        got = { type: req.headers.get('content-type'), body: await req.text() };
        return HttpResponse.json({ ok: true });
      }),
    );
    const reply = await upload<{ ok: boolean }>('/up', new Blob(['hello'], { type: 'image/jpeg' }));
    expect(reply).toEqual({ ok: true });
    expect(got).toEqual({ type: 'image/jpeg', body: 'hello' });
  });

  it("refuses with the server's own message, as request() does", async () => {
    server.use(http.post('/up', () => HttpResponse.json({ error: 'too large' }, { status: 413 })));
    const refused = await upload('/up', new Blob(['x'])).catch((e: unknown) => e);
    expect(refused).toBeInstanceOf(HttpError);
    expect((refused as HttpError).status).toBe(413);
    expect((refused as HttpError).reason).toBe('too large');
  });
});
