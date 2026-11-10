import { act, renderHook, waitFor } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import type { ReactNode } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { newQueryClient, Providers } from '../../shared/Providers';
import { server } from '../../test/server';
import { isoDay } from '../lib/format';
import { reviewKey } from './api';
import { calendarStep } from './days';
import { ReviewEngine } from './engine';
import { daySegments } from './gatewaySegments';
import { pollsLive, useReviewWindow, windowDays } from './window';

vi.mock('./gatewaySegments', async original => ({
  ...(await original<typeof import('./gatewaySegments')>()),
  daySegments: vi.fn(),
}));

const DAY = '2026-10-27';
const at = (day: string, hour = 12) => new Date(`${day}T${String(hour).padStart(2, '0')}:00:00`).getTime() / 1000;

afterEach(() => vi.unstubAllEnvs());

describe('windowDays', () => {
  it('is the playback day and the days in view, with no day read ahead (#52)', () => {
    expect(windowDays(DAY, { t0: at('2026-10-20'), t1: at('2026-10-21') })).toEqual([DAY, '2026-10-20', '2026-10-21']);
    expect(windowDays(DAY, { t0: at(DAY), t1: at(DAY) + 3600 })).toEqual([DAY]);
    expect(windowDays(DAY, null)).toEqual([DAY]);
    expect(windowDays(null, null)).toEqual([]);
  });

  it('leaves out calendar days after today, even in view', () => {
    expect(windowDays(DAY, { t0: at('2026-10-26', 20), t1: at('2026-10-28', 4) }, DAY)).toEqual([DAY, '2026-10-26']);
  });

  it.each(['2026-03-08', '2026-11-01'])('uses local dates rather than fixed-length days across %s', day => {
    vi.stubEnv('TZ', 'America/Los_Angeles');
    const start = new Date(`${day}T00:00:00`);
    const end = new Date(start);
    end.setDate(end.getDate() + 1);
    // The whole local day, 23 or 25 hours long: its end is the next midnight, which is not in view.
    expect(windowDays(day, { t0: start.getTime() / 1000, t1: end.getTime() / 1000 })).toEqual([day]);
    const next = day === '2026-03-08' ? '2026-03-09' : '2026-11-02';
    expect(windowDays(day, { t0: start.getTime() / 1000, t1: end.getTime() / 1000 + 1 })).toEqual([day, next]);
  });
});

describe('pollsLive (#2414)', () => {
  const TODAY = '2026-10-28';
  it.each([
    ['today, picked', TODAY, TODAY, null, true],
    ['today, a hidden neighbor of the day picked', TODAY, DAY, { t0: at(DAY, 9), t1: at(DAY, 17) }, false],
    ['today, inside the visible range', TODAY, DAY, { t0: at(DAY, 20), t1: at(TODAY, 4) }, true],
    ['an ended day, picked', DAY, DAY, null, false],
    ['an ended day, visible', DAY, TODAY, { t0: at(DAY), t1: at(TODAY) }, false],
  ])('%s: %s', (_, d, day, range, polls) => {
    expect(pollsLive(d, day, range, TODAY)).toBe(polls);
  });
});

it("doesn't read today while yesterday is shown, and polls it once it's the day shown", async () => {
  server.use(
    http.get('/transcript/day', () => HttpResponse.json({ entries: [] })),
    http.get('/audio/spans', () => HttpResponse.json({ spans: [] })),
    http.get('/review/presence', () => HttpResponse.json({ chips: [] })),
    http.get('/activities', () => HttpResponse.json({ items: [], truncated: false })),
  );
  vi.mocked(daySegments).mockResolvedValue([]);
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false } });
  const engine = new ReviewEngine({ cams: [], fetchNotes: async () => ({ cams: {} }) });
  const today = isoDay();
  engine.pickDay(calendarStep(today, -1));
  const polling = () =>
    [reviewKey.transcript(today), reviewKey.segments(today), reviewKey.spans(today)].map(
      queryKey => client.getQueryCache().find({ queryKey })?.observers[0]?.options.refetchInterval,
    );
  const { result, unmount } = renderHook(() => useReviewWindow(engine), {
    wrapper: ({ children }: { children: ReactNode }) => (
      <Providers client={client} test>
        {children}
      </Providers>
    ),
  });
  try {
    await waitFor(() => expect(result.current.initialized).toBe(true));
    expect(client.getQueryCache().find({ queryKey: reviewKey.transcript(today) })).toBeUndefined(); // not in view
    act(() => engine.pickDay(today));
    await waitFor(() => expect(polling()).toEqual([5000, 5000, 5000]));
  } finally {
    unmount();
    client.clear();
    engine.stop();
  }
});

it('reads a day fetched before it ended once more after midnight, then leaves it settled', async () => {
  const reads = new Map<string, number>();
  server.use(
    http.get('/transcript/day', ({ request }) => {
      const day = new URL(request.url).searchParams.get('day') ?? '';
      reads.set(day, (reads.get(day) ?? 0) + 1);
      return HttpResponse.json({ entries: [] });
    }),
    http.get('/audio/spans', () => HttpResponse.json({ spans: [] })),
    http.get('/review/presence', () => HttpResponse.json({ chips: [] })),
    http.get('/activities', () => HttpResponse.json({ items: [], truncated: false })),
  );
  vi.mocked(daySegments).mockResolvedValue([]);
  vi.useFakeTimers({ toFake: ['Date'] });
  vi.setSystemTime(new Date('2026-10-28T23:59:30')); // 10-28 is today, in view beside 10-27
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false } });
  const engine = new ReviewEngine({ cams: [], fetchNotes: async () => ({ cams: {} }) });
  engine.pickDay(DAY);
  const { result, unmount } = renderHook(() => useReviewWindow(engine), {
    wrapper: ({ children }: { children: ReactNode }) => (
      <Providers client={client} test>
        {children}
      </Providers>
    ),
  });
  try {
    await waitFor(() => expect(result.current.initialized).toBe(true));
    act(() => engine.setView(at(DAY, 20), at('2026-10-28', 4), true)); // a pan that brings today into view
    await waitFor(() => expect(reads.get('2026-10-28')).toBe(1));
    vi.setSystemTime(new Date('2026-10-29T00:00:30')); // midnight: 10-28 is over, and its snapshot is from before
    act(() => engine.setView(at(DAY, 21), at('2026-10-28', 3), true)); // anything that draws the window again
    await waitFor(() => expect(reads.get('2026-10-28')).toBe(2));
    act(() => engine.setView(at(DAY, 22), at('2026-10-28', 2), true));
    await new Promise(resolve => setTimeout(resolve, 50));
    expect(reads.get('2026-10-28')).toBe(2); // read after its end: settled, never again
    expect(reads.get(DAY)).toBe(1); // a day read after it ended is left alone
  } finally {
    unmount();
    client.clear();
    engine.stop();
    vi.useRealTimers();
  }
});

it('loads days lazily in either direction and restores cached chunks when panning back', async () => {
  const reads = new Map<string, number>();
  server.use(
    http.get('/transcript/day', ({ request }) => {
      const day = new URL(request.url).searchParams.get('day') ?? '';
      reads.set(day, (reads.get(day) ?? 0) + 1);
      return HttpResponse.json({ entries: [{ ts_start: at(day), ts_end: at(day) + 1, text: day }] });
    }),
    http.get('/audio/spans', () => HttpResponse.json({ spans: [] })),
    http.get('/review/presence', () => HttpResponse.json({ chips: [] })),
    http.get('/activities', () => HttpResponse.json({ items: [], truncated: false })),
  );
  vi.mocked(daySegments).mockResolvedValue([]);
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false } });
  const engine = new ReviewEngine({ cams: [], fetchNotes: async () => ({ cams: {} }) });
  engine.pickDay(DAY);
  const { result, unmount } = renderHook(() => useReviewWindow(engine), {
    wrapper: ({ children }: { children: ReactNode }) => (
      <Providers client={client} test>
        {children}
      </Providers>
    ),
  });
  try {
    await waitFor(() => expect(result.current.initialized).toBe(true));
    await waitFor(() => expect(engine.entries.map(e => e.text)).toContain(DAY));
    expect([...reads.keys()]).toEqual([DAY]); // neither neighbor is read before a pan reaches it
    act(() => engine.setView(at('2026-10-20'), at('2026-10-20') + 3600, true));
    await waitFor(() => expect(engine.entries.map(e => e.text)).toContain('2026-10-20'));
    expect(reads.get('2026-10-20')).toBe(1);
    act(() => engine.setView(at('2026-10-30'), at('2026-10-30') + 3600, true));
    await waitFor(() => expect(engine.entries.map(e => e.text)).toContain('2026-10-30'));
    expect(engine.entries.map(e => e.text)).not.toContain('2026-10-20');
    act(() => engine.setView(at('2026-10-20'), at('2026-10-20') + 3600, true));
    await waitFor(() => expect(engine.entries.map(e => e.text)).toContain('2026-10-20'));
    expect(reads.get('2026-10-20')).toBe(1);
    expect(reads.get(DAY)).toBe(1);
    expect(engine.ui.getState().day).toBe(DAY);
    await act(async () => {
      await client.invalidateQueries({ queryKey: reviewKey.transcript('2026-10-20') });
    });
    expect(reads.get('2026-10-20')).toBe(2);
    server.use(
      http.get('/transcript/day', ({ request }) => {
        const day = new URL(request.url).searchParams.get('day') ?? '';
        return day === '2026-10-20'
          ? HttpResponse.json({ error: 'Recording source unavailable' }, { status: 400 })
          : HttpResponse.json({ entries: [{ ts_start: at(day), text: day }] });
      }),
    );
    await act(async () => {
      await client.invalidateQueries({ queryKey: reviewKey.transcript('2026-10-20') });
    });
    await waitFor(() => expect(engine.ui.getState().chunkStatus['2026-10-20']).toBe('error'));
    expect(engine.entries.map(e => e.text)).toContain('2026-10-20'); // retain known data after a failed refresh
    server.use(http.get('/transcript/day', () => HttpResponse.json({ entries: [] })));
    await act(async () => {
      await client.invalidateQueries({ queryKey: reviewKey.transcript('2026-10-20') });
    });
    await waitFor(() => expect(engine.entries.map(e => e.text)).not.toContain('2026-10-20'));
    expect(engine.entries.map(e => e.text)).toContain(DAY); // authoritative replacement affects only its chunk
  } finally {
    unmount();
    client.clear();
    engine.stop();
  }
});
