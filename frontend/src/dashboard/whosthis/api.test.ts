import { QueryObserver } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { newQueryClient } from '../../shared/Providers';
import {
  dropClip,
  UNKNOWN_VOICES_KEY,
  type UnknownVoice,
  unknownGroupsQuery,
  unknownVoicesQuery,
  waitingTotal,
} from './api';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it.each(['faces', 'voices'] as const)(
  'preserves pending and unknown count, then resumes normal polling',
  async kind => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(Response.json({ pending: true, groups: [], count: null, skipped: [] }, { status: 202 }))
      .mockImplementation(() => Promise.resolve(Response.json({ groups: [], count: 0, skipped: [7] })));
    vi.stubGlobal('fetch', fetcher);
    const client = newQueryClient();
    const observer =
      kind === 'faces'
        ? new QueryObserver(client, unknownGroupsQuery())
        : new QueryObserver(client, unknownVoicesQuery());
    const unsubscribe = observer.subscribe(() => {});
    try {
      await vi.advanceTimersByTimeAsync(0);
      expect(observer.getCurrentResult().data).toMatchObject({ pending: true, count: null });
      await vi.advanceTimersByTimeAsync(999);
      expect(fetcher).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(observer.getCurrentResult().data).toMatchObject({ pending: false, count: 0, skipped: [7] });
      expect(fetcher).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(1000);
      expect(fetcher).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(29_000);
      expect(fetcher).toHaveBeenCalledTimes(3);
    } finally {
      unsubscribe();
      client.clear();
    }
  },
);

it.each(['faces', 'voices'] as const)('does not keep fast-polling %s after pending becomes an error', async kind => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  const fetcher = vi
    .fn<typeof fetch>()
    .mockResolvedValueOnce(Response.json({ pending: true, groups: [], count: null, skipped: [] }, { status: 202 }))
    .mockResolvedValueOnce(Response.json({ error: 'bootstrap unavailable' }, { status: 503 }))
    .mockImplementation(() => Promise.resolve(Response.json({ groups: [], count: 0, skipped: [] })));
  vi.stubGlobal('fetch', fetcher);
  const client = newQueryClient();
  const observer =
    kind === 'faces'
      ? new QueryObserver(client, unknownGroupsQuery())
      : new QueryObserver(client, unknownVoicesQuery());
  const unsubscribe = observer.subscribe(() => {});
  try {
    await vi.advanceTimersByTimeAsync(1000);
    expect(observer.getCurrentResult().isError).toBe(true);
    expect(observer.getCurrentResult().error?.message).toBe('bootstrap unavailable');
    expect(fetcher).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(2000);
    expect(fetcher).toHaveBeenCalledTimes(2);
    await observer.refetch();
    expect(observer.getCurrentResult().data).toMatchObject({ pending: false, count: 0 });
  } finally {
    unsubscribe();
    client.clear();
  }
});

it.each(['face', 'voice'] as const)(
  'reads the %s groups again in 2 s while the server is still working out a patched answer',
  async kind => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(Response.json({ groups: [], count: 2, skipped: [], refreshing: true }))
      .mockImplementation(() => Promise.resolve(Response.json({ groups: [], count: 2, skipped: [] })));
    vi.stubGlobal('fetch', fetcher);
    const client = newQueryClient();
    const observer =
      kind === 'face'
        ? new QueryObserver(client, unknownGroupsQuery())
        : new QueryObserver(client, unknownVoicesQuery());
    const unsubscribe = observer.subscribe(() => {});
    try {
      await vi.advanceTimersByTimeAsync(0);
      expect(observer.getCurrentResult().data).toMatchObject({ refreshing: true, count: 2 });
      await vi.advanceTimersByTimeAsync(2000);
      expect(fetcher).toHaveBeenCalledTimes(2);
      expect(observer.getCurrentResult().data).toMatchObject({ refreshing: false });
      await vi.advanceTimersByTimeAsync(2000);
      expect(fetcher).toHaveBeenCalledTimes(2); // back to the usual 30 s once the full answer is in
    } finally {
      unsubscribe();
      client.clear();
    }
  },
);

describe('a clip answered on its own (Multiple speakers, Nobody)', () => {
  const clip = (line: string) => ({ ts: 1, line, text: 'hi', dur: 2, url: `/clip?line=${line}` });
  const voice = (id: number, lines: string[]) =>
    ({ id, label: `Speaker ${id}`, count: 3, last_seen: 1, clips: lines.map(clip) }) as unknown as UnknownVoice;
  const seed = () => {
    const client = newQueryClient();
    client.setQueryData(UNKNOWN_VOICES_KEY, {
      groups: [voice(5, ['mic@1.000', 'mic@2.000']), voice(6, ['mic@3.000'])],
      count: 2,
      skipped: [],
      pending: false,
    });
    return client;
  };
  const listed = (client: ReturnType<typeof newQueryClient>) =>
    client.getQueryData<{ groups: UnknownVoice[]; count: number }>(UNKNOWN_VOICES_KEY);

  it('leaves the list at once, and the voice keeps its other clips', () => {
    const client = seed();
    dropClip(client, 5, { ts: 1, line: 'mic@1.000' });
    expect(listed(client)?.groups.map(g => [g.id, g.clips.map(c => c.line)])).toEqual([
      [5, ['mic@2.000']],
      [6, ['mic@3.000']],
    ]);
    expect(listed(client)?.count).toBe(2);
  });

  it('takes the voice off when it was its last clip, and one more off the count', () => {
    const client = seed();
    dropClip(client, 6, { ts: 1, line: 'mic@3.000' });
    expect(listed(client)?.groups.map(g => g.id)).toEqual([5]);
    expect(listed(client)?.count).toBe(1);
  });
});

describe('waitingTotal', () => {
  const read = (count: number | null) => ({ data: { count } });
  it.each([
    ['voices alone while the faces still load', {}, read(23), null],
    ['faces pending (count unknown yet)', read(null), read(23), null],
    ['both read', read(161), read(23), 184],
    ['the faces failed: the voices still count', { isError: true }, read(23), 23],
    ['a failed refresh keeps the count still shown', { ...read(161), isError: true }, read(23), 184],
    ['both failed', { isError: true }, { isError: true }, 0],
  ])('%s', (_, faces, voices, expected) => {
    expect(waitingTotal(faces, voices)).toBe(expected);
  });
});
