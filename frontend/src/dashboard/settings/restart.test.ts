import { afterEach, expect, it, vi } from 'vitest';
import { followRestart, RESTART_DEADLINE_MS } from './restart';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
const response = (boot: string) =>
  new Response(JSON.stringify({ boot }), { headers: { 'Content-Type': 'application/json' } });

it('waits through an unavailable server and the old process before accepting a new boot', async () => {
  vi.useFakeTimers();
  const fetch = vi
    .fn()
    .mockRejectedValueOnce(new Error('offline'))
    .mockResolvedValueOnce(response('old'))
    .mockResolvedValueOnce(response('new'));
  vi.stubGlobal('fetch', fetch);
  const done = followRestart('old', 'http://example.test:8888/', new AbortController().signal);
  await vi.advanceTimersByTimeAsync(2100);
  await done;
  expect(fetch).toHaveBeenCalledTimes(3);
  expect(fetch.mock.calls[0]?.[0]).toBe('http://example.test:8888/health');
  expect(vi.getTimerCount()).toBe(0);
});

it('cancels the scheduled retry when the form unmounts', async () => {
  vi.useFakeTimers();
  const fetch = vi.fn();
  vi.stubGlobal('fetch', fetch);
  const controller = new AbortController();
  const done = followRestart('old', 'http://example.test/', controller.signal);
  const rejected = expect(done).rejects.toMatchObject({ name: 'AbortError' });
  controller.abort();
  await rejected;
  await vi.advanceTimersByTimeAsync(2000);
  expect(fetch).not.toHaveBeenCalled();
  expect(vi.getTimerCount()).toBe(0);
});

it('stops waiting at the deadline even when health never answers', async () => {
  vi.useFakeTimers();
  vi.stubGlobal(
    'fetch',
    vi.fn(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () => reject(init.signal?.reason), { once: true });
        }),
    ),
  );
  const done = followRestart('old', 'http://example.test/', new AbortController().signal);
  const rejected = expect(done).rejects.toThrow('120 seconds');
  await vi.advanceTimersByTimeAsync(RESTART_DEADLINE_MS);
  await rejected;
  expect(vi.getTimerCount()).toBe(0);
});
