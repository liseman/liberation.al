import { createElement } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { renderPage } from '../test/render';
import type { VisionStatus } from './live/types';
import { getPeople, PEOPLE_KEY } from './people/api';
import { useRoute } from './router';
import { StatusPill } from './StatusIcons';
import { camStatus, camText, camTrouble, micStatus, micText, shortAge } from './status';

const NOW = 1_000_000;

describe('micStatus', () => {
  it.each([
    [undefined, 'off', null],
    [{}, 'off', null],
    [{ ts: NOW - 14 }, 'on', 14],
    [{ ts: NOW - 15 }, 'stale', 15],
    [{ ts: NOW - 299 }, 'stale', 299],
    [{ ts: NOW - 300 }, 'off', 300],
  ])('%j is %s', (ambient, state, age) => {
    expect(micStatus(ambient, NOW)).toEqual({ state, age });
  });
});

describe('camStatus', () => {
  it.each(['__proto__', 'constructor', 'toString'])('tracks only actual timestamps for %s', cam => {
    expect(camStatus([], [cam], NOW)).toEqual({ state: 'off', live: [], paused: [], age: null });
    expect(camStatus([{ cam, ts: NOW }], [cam], NOW)).toEqual({ state: 'on', live: [cam], paused: [], age: 0 });
  });
  const cams = ['cam0', 'cam1'];

  it('is on when every camera sent a scene in the last 30 s', () => {
    const s = camStatus(
      [
        { cam: 'cam0', ts: NOW - 30 },
        { cam: 'cam1', ts: NOW - 2 },
      ],
      cams,
      NOW,
    );
    expect(s).toEqual({ state: 'on', live: ['cam0', 'cam1'], paused: [], age: 2 });
  });

  it('is some when only part of them did, naming the quiet one', () => {
    const s = camStatus(
      [
        { cam: 'cam0', ts: NOW - 5 },
        { cam: 'cam1', ts: NOW - 31 },
      ],
      cams,
      NOW,
    );
    expect(s.state).toBe('some');
    expect(camText(s, cams)).toBe('1 of 2 cameras on (cam1 quiet)');
  });

  it("uses each camera's newest entry, whatever the order", () => {
    const s = camStatus(
      [
        { cam: 'cam0', ts: NOW - 5 },
        { cam: 'cam0', ts: NOW - 500 },
        { cam: 'cam1', ts: NOW - 1 },
      ],
      cams,
      NOW,
    );
    expect(s.state).toBe('on');
  });

  it('is stale when none is fresh but one spoke in the last 5 minutes, and off after', () => {
    expect(camStatus([{ cam: 'cam0', ts: NOW - 60 }], cams, NOW)).toEqual({
      state: 'stale',
      live: [],
      paused: [],
      age: 60,
    });
    expect(camStatus([{ cam: 'cam0', ts: NOW - 300 }], cams, NOW).state).toBe('off');
    expect(camStatus([], cams, NOW)).toEqual({ state: 'off', live: [], paused: [], age: null });
  });

  it('ignores scene entries from cameras that are not configured', () => {
    expect(camStatus([{ cam: 'cam9', ts: NOW }], cams, NOW).state).toBe('off');
  });
});

describe('camStatus from frames', () => {
  const cams = ['cam0', 'cam1'];
  const sending = (...on: string[]): VisionStatus => ({
    ts: NOW - 3,
    cameras: Object.fromEntries(on.map(c => [c, { state: 'on' as const }])),
  });
  const captions = [
    { cam: 'cam0', ts: NOW - 40 },
    { cam: 'cam1', ts: NOW - 40 },
  ];

  it('is on while frames arrive, however long since the last caption', () => {
    const s = camStatus(captions, cams, NOW, sending('cam0', 'cam1'));
    expect(s).toEqual({ state: 'on', live: ['cam0', 'cam1'], paused: [], age: 40 });
    expect(camText(s, cams)).toBe('All 2 cameras on');
  });

  it('calls a camera quiet when its frames stop, even with a fresh caption', () => {
    const s = camStatus([{ cam: 'cam1', ts: NOW - 1 }], cams, NOW, sending('cam0'));
    expect(s.state).toBe('some');
    expect(camText(s, cams)).toBe('1 of 2 cameras on (cam1 quiet)');
  });

  it('is off, not late, when vision is fresh and no camera sends frames', () => {
    expect(camStatus([{ cam: 'cam0', ts: NOW - 60 }], cams, NOW, sending()).state).toBe('off');
  });

  it("falls back to captions once vision's status is more than 10 s old", () => {
    const stale = { ...sending('cam0', 'cam1'), ts: NOW - 11 };
    expect(camStatus(captions, cams, NOW, stale).state).toBe('stale');
    expect(camStatus(captions, cams, NOW, { ...stale, ts: NOW - 10 }).state).toBe('on');
  });

  it.each(['__proto__', 'constructor'])('reads only cameras vision named, not %s', cam => {
    expect(camStatus([], [cam], NOW, sending()).state).toBe('off');
  });

  it('says paused, not quiet, for a camera paused for privacy', () => {
    const four = ['cam0', 'cam1', 'cam2', 'cam3'];
    const s = camStatus([], four, NOW, sending('cam0'), ['cam1']);
    expect(s).toEqual({ state: 'some', live: ['cam0'], paused: ['cam1'], age: null });
    expect(camText(s, four)).toBe('1 of 4 cameras on (cam1 paused; cam2, cam3 quiet)');
    const rest = camStatus([], cams, NOW, sending('cam0'), ['cam1']);
    expect(rest.state).toBe('on');
    expect(camText(rest, cams)).toBe('1 of 2 cameras on (cam1 paused)');
  });

  it('says paused for privacy when every camera is paused', () => {
    const s = camStatus([], cams, NOW, sending(), cams);
    expect(camText(s, cams, 'starting')).toBe('Paused for privacy');
  });
});

describe('tooltips', () => {
  it.each([
    [{ state: 'on', age: 3 }, 'Listening'],
    [{ state: 'stale', age: 40 }, 'Microphone is late: last update 40 s ago'],
    [{ state: 'off', age: null }, 'Not listening'],
    [{ state: 'off', age: 7200 }, 'Not listening: last update 2 h ago'],
  ] as const)('mic %j says %s', (status, text) => {
    expect(micText(status)).toBe(text);
  });

  it.each([
    [['cam0'], 'Camera on'],
    [['cam0', 'cam1', 'cam2'], 'All 3 cameras on'],
  ])('cameras on with %j says %s', (cams, text) => {
    expect(camText({ state: 'on', live: cams, paused: [], age: 1 }, cams)).toBe(text);
  });

  it('says how late the cameras are', () => {
    expect(camText({ state: 'stale', live: [], paused: [], age: 150 }, ['cam0'])).toBe(
      'Cameras are late: last scene 3 min ago',
    );
  });

  it.each([
    [59, '59 s'],
    [60, '1 min'],
    [3599, '60 min'],
    [3600, '1 h'],
  ])('shortAge(%i) is %s', (s, text) => {
    expect(shortAge(s)).toBe(text);
  });
});

describe('camTrouble', () => {
  const now = 1000;
  it('is retrying while any camera is being reopened, and starting before a first frame', () => {
    const cameras = { cam0: { state: 'on' as const }, cam1: { state: 'retrying' as const } };
    expect(camTrouble({ ts: now - 1, cameras }, now)).toBe('retrying');
    expect(camTrouble({ ts: now - 1, cameras: { cam0: { state: 'starting' } } }, now)).toBe('starting');
    expect(camTrouble({ ts: now - 1, cameras: { cam0: { state: 'on' } } }, now)).toBeNull();
  });

  it("is nothing when vision's status is old or missing", () => {
    expect(camTrouble({ ts: now - 60, cameras: { cam0: { state: 'retrying' } } }, now)).toBeNull();
    expect(camTrouble(undefined, now)).toBeNull();
  });

  it('names it in the camera tooltip in place of off', () => {
    const off = { state: 'off' as const, live: [], paused: [], age: null };
    expect(camText(off, ['cam0'], 'retrying')).toBe('Camera not responding, retrying');
    expect(camText(off, ['cam0'], 'starting')).toBe('Starting cameras…');
    expect(camText({ state: 'on', live: ['cam0'], paused: [], age: 1 }, ['cam0'], 'retrying')).toBe('Camera on');
  });
});

describe('header status scheduling', () => {
  it('leaves a slot for People while ambient and scene status reads are pending', async () => {
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
    const previous = useRoute.getState();
    useRoute.setState({ tab: 'known', known: 'people', settingsReturnTo: undefined });
    const view = renderPage(createElement(StatusPill));
    const { client } = view;
    const people = client.fetchQuery({ queryKey: PEOPLE_KEY, queryFn: getPeople }).catch((error: unknown) => error);
    try {
      await Promise.resolve();
      expect(fetcher.mock.calls.map(([url]) => url)).toEqual(['/ambient', '/people']);
      replies.get('/people')?.(Response.json({ people: [] }));
      expect(await people).toMatchObject({ people: [] });
      expect(fetcher).toHaveBeenCalledTimes(2); // scene/vision cannot take the page slot while ambient is slow
    } finally {
      view.unmount();
      await client.cancelQueries();
      client.clear();
      useRoute.setState(previous);
      vi.unstubAllGlobals();
    }
  });
});
