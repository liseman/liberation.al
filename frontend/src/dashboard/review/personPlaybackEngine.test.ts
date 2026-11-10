// Real engine and DOM media elements; only media decoding/downloads are controlled at their browser boundaries.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { type EngineDeps, ReviewEngine } from './engine';
import type { RawSegment } from './types';

const DAY = '2026-10-27';
const T = new Date(2026, 9, 27, 10).getTime() / 1000;
const NEXT_DAY = '2026-10-28';
const MIDNIGHT = new Date(2026, 9, 28).getTime() / 1000;
const entries = [
  { ts_start: T, ts_end: T + 2, text: 'First selected line', speaker: 'Alice' },
  { ts_start: T + 3, ts_end: T + 5, text: 'Other speaker', speaker: 'Dana' },
  { ts_start: T + 100, ts_end: T + 103, text: 'Next selected line', speaker: 'Alice' },
];
function medium(kind: 'audio' | 'video') {
  const el = document.createElement(kind);
  let paused = true;
  let time = 0;
  let ended = false;
  let source = '';
  Object.defineProperties(el, {
    src: {
      get: () => source,
      set: (value: string) => {
        source = value;
        paused = true;
        ended = false;
        if (kind === 'audio') queueMicrotask(() => el.dispatchEvent(new Event('loadedmetadata')));
      },
    },
    paused: { get: () => paused },
    readyState: { get: () => 4 },
    duration: { get: () => 500 },
    ended: { get: () => ended },
    seeking: { get: () => false },
    currentTime: {
      get: () => time,
      set: (v: number) => {
        time = v;
        queueMicrotask(() => el.dispatchEvent(new Event('seeked')));
      },
    },
  });
  el.play = vi.fn(async () => {
    const was = paused;
    paused = false;
    if (was) el.dispatchEvent(new Event('playing'));
  });
  el.pause = vi.fn(() => {
    paused = true;
  });
  el.load = vi.fn(() => {
    time = 0;
    ended = false;
    paused = true;
    queueMicrotask(() => el.dispatchEvent(new Event('loadedmetadata')));
  });
  return {
    el,
    reach: (v: number) => {
      time = v;
    },
    finish: () => {
      ended = true;
      paused = true;
    },
  };
}
const flush = async () => {
  for (let i = 0; i < 8; i++) await Promise.resolve();
};
const engines: ReviewEngine[] = [];
function setup({ segments, blob }: { segments?: RawSegment[]; blob?: EngineDeps['fetchBlob'] } = {}) {
  const fetched: string[] = [];
  const engine = new ReviewEngine({
    cams: ['front', 'rear'],
    fetchNotes: async () => ({ cams: {} }),
    fetchBlob: async (url, signal) => {
      fetched.push(url);
      return blob ? blob(url, signal) : new Blob(['audio']);
    },
  });
  engines.push(engine);
  const front = [medium('video'), medium('video')] as const;
  const rear = [medium('video'), medium('video')] as const;
  const audio = medium('audio');
  engine.bindVideo('front', 0, front[0].el);
  engine.bindVideo('front', 1, front[1].el);
  engine.bindVideo('rear', 0, rear[0].el);
  engine.bindVideo('rear', 1, rear[1].el);
  engine.setAudio(audio.el);
  engine.pickDay(DAY);
  engine.setDayData(DAY, engine.ui.getState().gen, {
    entries,
    segments: segments ?? [
      { cam: 'front', start: T - 10, end: T + 500, url: 'gateway:front' },
      { cam: 'rear', start: T - 10, end: T + 500, url: 'gateway:rear' },
    ],
    spans: [],
  });
  return { engine, fetched, front, rear, audio };
}
beforeEach(() => {
  history.replaceState(null, '', '/review');
  Object.assign(URL, { createObjectURL: () => 'blob:audio', revokeObjectURL: () => {} });
});
afterEach(() => {
  for (const engine of engines.splice(0)) engine.stop();
  vi.restoreAllMocks();
});

describe('person playback through the Review engine', () => {
  it('reveals the queued source camera when the camera filter currently excludes it', async () => {
    const { engine } = setup();
    engine.pickCameras(['front']);
    engine.personPlayback.start('Alice', 'video', [{ start: T + 18, end: T + 25, cam: 'rear' }]);
    await flush();
    expect(engine.ui.getState().cameras).toEqual(['front', 'rear']);
    expect(engine.ui.getState().camera).toBe('rear');
    expect(engine.personPlayback.run?.mode).toBe('video');
  });

  it('cancels person playback when a manual camera filter takes over', async () => {
    const { engine } = setup();
    engine.personPlayback.start('Alice', 'video', [{ start: T + 18, end: T + 25, cam: 'rear' }]);
    await flush();
    engine.pickCameras(['front']);
    expect(engine.personPlayback.run).toBeNull();
    expect(engine.playing).toBe(false);
    expect(engine.ui.getState().cameras).toEqual(['front']);
  });
  it('keeps video paused when late seeked events land during an audio-only queue', async () => {
    const { engine, front, rear } = setup();
    engine.personPlayback.start('Alice', 'audio', [{ start: T, end: T + 2, cam: null }]);
    await flush();
    for (const video of [...front, ...rear]) video.el.dispatchEvent(new Event('seeked'));
    await flush();
    expect([...front, ...rear].every(video => video.el.paused)).toBe(true);
  });

  it('waits for an uncached video day instead of declaring its camera recording missing', async () => {
    const { engine } = setup();
    engine.setWindowDays([DAY, NEXT_DAY]);
    engine.setDayStatus(NEXT_DAY, 'loading');
    engine.personPlayback.start('Alice', 'video', [{ start: MIDNIGHT + 10, end: MIDNIGHT + 15, cam: 'rear' }]);
    await flush();
    expect(engine.personPlayback.run?.index).toBe(0);
    expect(engine.ui.getState()).toMatchObject({ waiting: true, day: NEXT_DAY });
    engine.upsertDayData(NEXT_DAY, {
      entries: [],
      spans: [],
      segments: [{ cam: 'rear', start: MIDNIGHT, end: MIDNIGHT + 20, url: 'gateway:rear:tomorrow' }],
    });
    engine.setDayStatus(NEXT_DAY, 'ready');
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.run?.index).toBe(0);
    expect(engine.t).toBe(MIDNIGHT + 10);
    expect(engine.ui.getState()).toMatchObject({ waiting: false, camera: 'rear', playing: true });
  });

  it('preserves its bounded queue when cached video straddles a still-loading day context', async () => {
    const { engine } = setup({
      segments: [{ cam: 'rear', start: MIDNIGHT - 5, end: MIDNIGHT + 20, url: 'gateway:rear:crossing' }],
    });
    engine.setWindowDays([DAY, NEXT_DAY]);
    engine.personPlayback.start('Alice', 'video', [{ start: MIDNIGHT + 10, end: MIDNIGHT + 15, cam: 'rear' }]);
    await flush();
    expect(engine.ui.getState().waiting).toBe(true);
    engine.upsertDayData(NEXT_DAY, {
      entries: [],
      spans: [],
      segments: [{ cam: 'rear', start: MIDNIGHT, end: MIDNIGHT + 20, url: 'gateway:rear:next' }],
    });
    engine.setDayStatus(NEXT_DAY, 'ready');
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.current).toEqual({ start: MIDNIGHT + 10, end: MIDNIGHT + 15, cam: 'rear' });
    expect(engine.t).toBe(MIDNIGHT + 10);
  });

  it('waits at midnight for the same camera and resumes the current moment at the boundary', async () => {
    const { engine, rear } = setup({
      segments: [{ cam: 'rear', start: MIDNIGHT - 5, end: MIDNIGHT, url: 'gateway:rear:today' }],
    });
    engine.setWindowDays([DAY, NEXT_DAY]);
    engine.setDayStatus(NEXT_DAY, 'loading');
    engine.personPlayback.start('Alice', 'video', [{ start: MIDNIGHT - 2, end: MIDNIGHT + 4, cam: 'rear' }]);
    await flush();
    rear[0].reach(5);
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.run?.index).toBe(0);
    expect(engine.ui.getState()).toMatchObject({ waiting: true, day: NEXT_DAY });
    engine.upsertDayData(NEXT_DAY, {
      entries: [],
      spans: [],
      segments: [{ cam: 'rear', start: MIDNIGHT, end: MIDNIGHT + 20, url: 'gateway:rear:next' }],
    });
    engine.setDayStatus(NEXT_DAY, 'ready');
    engine.frame(2000);
    await flush();
    expect(engine.personPlayback.run?.index).toBe(0);
    expect(engine.t).toBe(MIDNIGHT);
    expect(engine.ui.getState()).toMatchObject({ waiting: false, camera: 'rear' });
    rear[0].reach(4);
    rear[1].reach(4);
    engine.frame(3000);
    expect(engine.personPlayback.run).toBeNull();
    expect(engine.playing).toBe(false);
  });

  it('requests a continuation day even when a recording gap ends before midnight and the day is outside the published window', async () => {
    const { engine, rear } = setup({
      segments: [{ cam: 'rear', start: MIDNIGHT - 20, end: MIDNIGHT - 10, url: 'gateway:rear:today' }],
    });
    engine.setWindowDays([DAY]);
    engine.personPlayback.start('Alice', 'video', [{ start: MIDNIGHT - 15, end: MIDNIGHT + 4, cam: 'rear' }]);
    await flush();
    rear[0].reach(10);
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.run?.index).toBe(0);
    expect(engine.t).toBe(MIDNIGHT);
    expect(engine.ui.getState()).toMatchObject({ waiting: true, day: NEXT_DAY, notice: null });
  });
  it('skips the clip lead, excludes other speakers, and stops at the selected line ends', async () => {
    const { engine, fetched, audio } = setup();
    engine.personPlayback.start('Alice', 'audio', [
      { start: T, end: T + 2, cam: null },
      { start: T + 100, end: T + 103, cam: null },
    ]);
    await flush();
    expect(audio.el.currentTime).toBeCloseTo(0.4);
    audio.reach(2.4);
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.current?.start).toBe(T + 100);
    expect(fetched.filter(u => u.startsWith('/clip'))).toEqual([`/clip?ts_start=${T}`, `/clip?ts_start=${T + 100}`]);
    audio.reach(3.4);
    engine.frame(2000);
    expect(engine.personPlayback.run).toBeNull();
    expect(engine.ui.getState().playing).toBe(false);
    expect(audio.el.paused).toBe(true);
  });

  it('advances on an audio ended event without falling through to unrelated room playback', async () => {
    const { engine, audio } = setup();
    engine.personPlayback.start('Alice', 'audio', [
      { start: T, end: T + 2, cam: null },
      { start: T + 100, end: T + 103, cam: null },
    ]);
    await flush();
    audio.finish();
    engine.audioEnded();
    await flush();
    expect(engine.personPlayback.current?.start).toBe(T + 100);
  });

  it.each(['seek', 'pause', 'scrub', 'person', 'day', 'stop'] as const)(
    'cancels the queue on a manual %s',
    async action => {
      const { engine } = setup();
      engine.personPlayback.start('Alice', 'audio', [
        { start: T, end: T + 2, cam: null },
        { start: T + 100, end: T + 103, cam: null },
      ]);
      await flush();
      if (action === 'seek') engine.seek(T + 3, { play: true });
      if (action === 'pause') engine.pause();
      if (action === 'scrub') engine.scrubStart(T + 3);
      if (action === 'person') engine.pickPerson('Dana');
      if (action === 'day') engine.pickDay('2026-10-28');
      if (action === 'stop') engine.stop();
      expect(engine.personPlayback.run).toBeNull();
      engine.audioEnded();
      expect(engine.personPlayback.run).toBeNull();
    },
  );

  it('keeps each video step on its own view and advances before ordinary segment rollover', async () => {
    const { engine, rear, front } = setup();
    engine.personPlayback.start('Alice', 'video', [
      { start: T + 18, end: T + 25, cam: 'rear' },
      { start: T + 98, end: T + 105, cam: 'front' },
    ]);
    await flush();
    expect(engine.ui.getState().camera).toBe('rear');
    rear[0].reach(35);
    rear[1].reach(35);
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.current?.cam).toBe('front');
    expect(engine.ui.getState().camera).toBe('front');
    front[0].reach(115);
    front[1].reach(115);
    engine.frame(2000);
    expect(engine.personPlayback.run).toBeNull();
    expect(engine.ui.getState().playing).toBe(false);
  });

  it('skips a missing own-camera recording instead of playing a different camera at that time', async () => {
    const { engine } = setup({ segments: [{ cam: 'front', start: T - 10, end: T + 500, url: 'gateway:front' }] });
    engine.personPlayback.start('Alice', 'video', [
      { start: T + 18, end: T + 25, cam: 'rear' },
      { start: T + 98, end: T + 105, cam: 'front' },
    ]);
    await flush();
    expect(engine.personPlayback.run?.index).toBe(1);
    expect(engine.t).toBe(T + 98);
    expect(engine.ui.getState()).toMatchObject({ camera: 'front', notice: 'This camera recording is unavailable.' });
  });

  it('crosses consecutive segments of the same camera without advancing the selected moment', async () => {
    const { engine, rear } = setup({
      segments: [
        { cam: 'rear', start: T + 10, end: T + 22, url: 'gateway:rear:first' },
        { cam: 'rear', start: T + 22, end: T + 30, url: 'gateway:rear:second' },
      ],
    });
    engine.personPlayback.start('Alice', 'video', [{ start: T + 18, end: T + 25, cam: 'rear' }]);
    await flush();
    rear[0].reach(12);
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.run?.index).toBe(0);
    expect(engine.t).toBe(T + 22);
    rear[1].reach(3);
    engine.frame(2000);
    expect(engine.personPlayback.run).toBeNull();
    expect(engine.playing).toBe(false);
  });

  it('skips to the next queued camera moment if its own recording ends early', async () => {
    const { engine, rear } = setup({
      segments: [
        { cam: 'rear', start: T + 10, end: T + 22, url: 'gateway:rear' },
        { cam: 'front', start: T - 10, end: T + 500, url: 'gateway:front' },
      ],
    });
    engine.personPlayback.start('Alice', 'video', [
      { start: T + 18, end: T + 25, cam: 'rear' },
      { start: T + 98, end: T + 105, cam: 'front' },
    ]);
    await flush();
    rear[0].reach(12);
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.current?.start).toBe(T + 98);
    expect(engine.t).toBe(T + 98);
    expect(engine.ui.getState().camera).toBe('front');
  });

  it('advances after a missing speech blob and reports the missing recording', async () => {
    const { engine, fetched } = setup({
      blob: async url => {
        if (url === `/clip?ts_start=${T}`) throw new Error('not retained');
        return new Blob(['audio']);
      },
    });
    engine.personPlayback.start('Alice', 'audio', [
      { start: T, end: T + 2, cam: null },
      { start: T + 100, end: T + 103, cam: null },
    ]);
    await flush();
    expect(engine.personPlayback.current?.start).toBe(T + 100);
    expect(fetched.filter(u => u.startsWith('/clip'))).toEqual([`/clip?ts_start=${T}`, `/clip?ts_start=${T + 100}`]);
    expect(engine.ui.getState().notice).toBe('This speech recording is unavailable.');
  });

  it('ignores a late speech blob after a manual seek cancels the queue', async () => {
    let finish!: (blob: Blob) => void;
    const { engine, audio } = setup({
      blob: () =>
        new Promise(resolve => {
          finish = resolve;
        }),
    });
    engine.personPlayback.start('Alice', 'audio', [{ start: T, end: T + 2, cam: null }]);
    engine.seek(T + 3);
    const played = vi.mocked(audio.el.play).mock.calls.length;
    finish(new Blob(['late speech']));
    await flush();
    expect(engine.personPlayback.run).toBeNull();
    expect(audio.el.play).toHaveBeenCalledTimes(played);
    expect(audio.el.paused).toBe(true);
    expect(engine.t).toBe(T + 3);
  });

  it('ignores an old ended event after the next audio source has begun', async () => {
    const { engine, audio } = setup();
    engine.personPlayback.start('Alice', 'audio', [
      { start: T, end: T + 2, cam: null },
      { start: T + 100, end: T + 103, cam: null },
    ]);
    await flush();
    audio.reach(2.4);
    engine.frame(1000);
    await flush();
    engine.audioEnded();
    expect(engine.personPlayback.current?.start).toBe(T + 100);
  });

  it('keeps its selected sequence independent of the global Skip filter', async () => {
    const { engine, audio } = setup();
    engine.setChips(DAY, engine.ui.getState().gen, [
      { id: 'object:mug', kind: 'object', label: 'Mug', spans: [[T + 300, T + 400]], total_s: 100, count: 1 },
    ]);
    engine.toggleChip('object:mug');
    engine.toggleSkip();
    engine.personPlayback.start('Alice', 'audio', [
      { start: T, end: T + 2, cam: null },
      { start: T + 100, end: T + 103, cam: null },
    ]);
    await flush();
    audio.reach(2.4);
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.current?.start).toBe(T + 100);
    expect(engine.t).toBe(T + 100);
  });

  it('preserves its snapshot and generation when an internal queue jump changes the date context', async () => {
    const { engine, audio } = setup();
    const gen = engine.ui.getState().gen;
    engine.personPlayback.start('Alice', 'audio', [
      { start: T, end: T + 2, cam: null },
      { start: T + 86400, end: T + 86402, cam: null },
    ]);
    await flush();
    audio.reach(2.4);
    engine.frame(1000);
    await flush();
    expect(engine.personPlayback.current?.start).toBe(T + 86400);
    expect(engine.ui.getState()).toMatchObject({ day: '2026-10-28', gen, playing: true });
  });
});
