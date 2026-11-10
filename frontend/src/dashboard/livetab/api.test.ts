import { QueryClient } from '@tanstack/react-query';
import { HttpResponse, http } from 'msw';
import { afterEach, expect, it, vi } from 'vitest';
import { server } from '../../test/server';
import { liveKey } from '../live/feed';
import type { Transcript } from '../live/types';
import { UNKNOWN_GROUPS_KEY, UNKNOWN_VOICES_KEY } from '../whosthis/api';
import { cancelQueuedRefreshes, claim, refreshNames, refreshTranscript } from './api';

const clients: QueryClient[] = [];
function client() {
  const c = new QueryClient();
  clients.push(c);
  return c;
}
afterEach(() => {
  for (const c of clients.splice(0)) c.clear();
  vi.clearAllTimers();
  vi.useRealTimers();
});

it('marks every people query out of date after a naming change, the People list included', async () => {
  server.use(http.get('/transcript', () => HttpResponse.json({ entries: [] })));
  const c = client();
  c.setQueryData(['people'], { people: [] });
  c.setQueryData(['people', 'card', 'Dana'], {});
  await refreshNames(c);
  expect([['people'], ['people', 'card', 'Dana']].map(k => c.getQueryState(k)?.isInvalidated)).toEqual([true, true]);
});

it('coalesces 19 naming refreshes into one tail read and one invalidation per family', async () => {
  let reads = 0;
  server.use(
    http.get('/transcript', () => {
      reads++;
      return HttpResponse.json({ entries: [] });
    }),
  );
  const c = client();
  const invalidates = vi.spyOn(c, 'invalidateQueries');
  await Promise.all(Array.from({ length: 19 }, () => refreshNames(c)));
  expect(invalidates).toHaveBeenCalledTimes(4);
  expect(reads).toBe(1);
});

it('coalesces delayed naming followups instead of creating a refresh per action', async () => {
  vi.useFakeTimers();
  let reads = 0;
  server.use(
    http.get('/transcript', () => {
      reads++;
      return HttpResponse.json({ entries: [] });
    }),
  );
  const c = client();
  const invalidates = vi.spyOn(c, 'invalidateQueries');
  const work = Array.from({ length: 19 }, () => refreshNames(c, 4000));
  await vi.advanceTimersByTimeAsync(3999);
  expect(invalidates).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(1);
  await Promise.all(work);
  expect(invalidates).toHaveBeenCalledTimes(4);
  expect(reads).toBe(1);
});

it('upgrades a mixed correction burst to one complete derived refresh', async () => {
  let reads = 0;
  server.use(
    http.get('/transcript', () => {
      reads++;
      return HttpResponse.json({ entries: [] });
    }),
  );
  const c = client();
  c.setQueryData(['review', 'day', '2026-10-01', 'blocks'], {});
  const invalidates = vi.spyOn(c, 'invalidateQueries');
  await Promise.all([refreshNames(c), ...Array.from({ length: 19 }, () => refreshTranscript(c))]);
  expect(invalidates).toHaveBeenCalledTimes(1);
  expect(invalidates.mock.calls[0]).toEqual([]);
  expect(c.getQueryState(['review', 'day', '2026-10-01', 'blocks'])?.isInvalidated).toBe(true);
  expect(reads).toBe(1);
});

it('does not replace newer transcript labels with a superseded slow refresh', async () => {
  let release = () => {};
  let entered = () => {};
  const hold = new Promise<void>(resolve => {
    release = resolve;
  });
  const started = new Promise<void>(resolve => {
    entered = resolve;
  });
  let reads = 0;
  server.use(
    http.get('/transcript', async () => {
      if (++reads === 1) {
        entered();
        await hold;
        return HttpResponse.json({ entries: [{ ts_start: 1, text: 'older labels' }], reset: true });
      }
      return HttpResponse.json({ entries: [{ ts_start: 1, text: 'newer labels' }], reset: true });
    }),
  );
  const c = client();
  const first = refreshNames(c);
  await started;
  const second = refreshNames(c);
  await second;
  release();
  await first;
  expect(c.getQueryData<Transcript>(liveKey('transcript'))?.entries[0]?.text).toBe('newer labels');
});

// A voice named from a transcript (useNameVoice claims it) leaves the waiting voices at once; only that one,
// only on the server's OK, and a face list is left alone.
it.each([
  ['an accepted name', { ok: true, name: 'Dana' }, 200, [8], 1],
  ['a refused name', { ok: false, error: 'no such voice' }, 404, [7, 8], 2],
])('takes a claimed voice off the waiting list on %s', async (_, reply, status, left, count) => {
  server.use(
    http.post('/people/claim', () => HttpResponse.json(reply, { status })),
    http.get('/transcript', () => HttpResponse.json({ entries: [] })),
  );
  const c = client();
  const faces = { groups: [{ id: 7 }], count: 4, skipped: [], pending: false };
  c.setQueryData(UNKNOWN_VOICES_KEY, { groups: [{ id: 7 }, { id: 8 }], count: 2, skipped: [], pending: false });
  c.setQueryData(UNKNOWN_GROUPS_KEY, faces);
  await claim(c, 'voice', '7', 'Dana');
  const voices = c.getQueryData<{ groups: { id: number }[]; count: number }>(UNKNOWN_VOICES_KEY);
  expect(voices?.groups.map(g => g.id)).toEqual(left);
  expect(voices?.count).toBe(count);
  expect(c.getQueryData(UNKNOWN_GROUPS_KEY)).toEqual(faces);
  await refreshNames(c); // let the naming's own follow-up read finish inside this test
});

it("drops a queued refresh when cancelled, so it can't read after its test (or page) is gone", async () => {
  vi.useFakeTimers();
  try {
    const fetched = vi.spyOn(globalThis, 'fetch');
    const client = new QueryClient();
    const names = refreshNames(client);
    const later = refreshNames(client, 4000); // a tag's follow-up
    cancelQueuedRefreshes();
    vi.advanceTimersByTime(5000);
    await Promise.all([names, later]); // they resolve rather than wait forever
    expect(fetched).not.toHaveBeenCalled();
  } finally {
    vi.useRealTimers();
  }
});
