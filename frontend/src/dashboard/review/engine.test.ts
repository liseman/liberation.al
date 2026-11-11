import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HttpError } from '../../shared/http';
import { PREFS_KEY } from '../../shared/prefs';
import { play as playClip, stop as stopClip, usePlayer } from '../lib/player';
import type { TranscriptEntry } from '../live/types';
import {
  type AudioLike,
  endHold,
  glideView,
  hashTiming,
  RANGE_SETTLE_MS,
  ReviewEngine,
  roomWindow,
  type VideoLike,
} from './engine';
import { MIN_SPAN } from './time';
import type { Chip, DayData, MomentNotes, RawSegment } from './types';
import { readHash } from './url';

const DAY = '2026-10-27';
const D0 = new Date(2026, 9, 27).getTime() / 1000;

/** A <video> that records what it was asked to do. Seeks land only when the test says so (land()). */
class FakeVideo implements VideoLike {
  extended: number[] = [];
  extend(duration: number) {
    this.extended.push(duration);
  }
  src = '';
  preload = 'metadata';
  playbackRate = 1;
  defaultPlaybackRate = 1;
  readyState = 1;
  duration = Number.NaN;
  error: MediaError | null = null;
  seeking = false;
  paused = true;
  ended = false;
  plays = 0;
  standby = false;
  seeks: number[] = [];
  loads = 0;
  private listeners: { type: string; fn: () => void; once: boolean }[] = [];
  private time = 0;
  get currentTime() {
    return this.time;
  }
  set currentTime(t: number) {
    this.time = t;
    this.seeks.push(t);
  }
  /** Playback reaching t (not a seek). */
  reach(t: number) {
    this.time = t;
  }
  deferPlaying = false; // hold back the playing event, as a browser does until frames are ready (playing() sends it)
  play() {
    const was = this.paused;
    this.paused = false;
    this.plays++;
    if (was && !this.deferPlaying) this.emit('playing'); // a video already playing fires no playing event
    return Promise.resolve();
  }
  playing() {
    this.emit('playing');
  }
  pause() {
    if (this.paused) return;
    this.paused = true;
    this.emit('pause');
  }
  load() {
    this.loads++;
  }
  getAttribute(name: string) {
    return name === 'src' && this.src ? this.src : null;
  }
  removeAttribute(name: string) {
    if (name === 'src') this.src = '';
  }
  toggleAttribute(_name: string, force?: boolean) {
    this.standby = force ?? !this.standby;
    return this.standby;
  }
  addEventListener(type: string, fn: () => void, options?: AddEventListenerOptions) {
    this.listeners.push({ type, fn, once: !!options?.once });
  }
  removeEventListener(type: string, fn: () => void) {
    this.listeners = this.listeners.filter(l => l.type !== type || l.fn !== fn);
  }
  emit(type: string) {
    for (const l of this.listeners.filter(x => x.type === type)) {
      if (l.once) this.removeEventListener(type, l.fn);
      l.fn();
    }
  }
  callbacks(type: string) {
    return this.listeners.filter(listener => listener.type === type).map(listener => listener.fn);
  }
  land() {
    this.emit('seeked');
  }
}

/** An <audio> that plays what it's given once the test says its metadata loaded (emit('loadedmetadata')). */
class FakeAudio implements AudioLike {
  #src = '';
  get src() {
    return this.#src;
  }
  set src(url: string) {
    this.#src = url;
    this.paused = true; // a new source pauses the element, as the browser's load algorithm does
  }
  currentTime = 0;
  readyState = 0;
  paused = true;
  plays = 0;
  unlocks = 0; // plays of the silent sound that unlocks the element inside a click
  playbackRate = 1;
  refuse: DOMException | null = null; // what the browser says when it refuses play()
  private listeners: { type: string; fn: () => void }[] = [];
  play() {
    if (this.src.startsWith('data:')) this.unlocks++;
    else this.plays++;
    if (this.refuse) return Promise.reject(this.refuse);
    this.paused = false;
    return Promise.resolve();
  }
  pause() {
    this.paused = true;
  }
  load() {}
  addEventListener(type: string, fn: () => void) {
    this.listeners.push({ type, fn });
  }
  emit(type: string) {
    const due = this.listeners.filter(l => l.type === type);
    this.listeners = this.listeners.filter(l => l.type !== type);
    for (const l of due) l.fn();
  }
}

/** A segment's URL, as the gateway's are named (the engine plays whatever URL a segment carries). */
const url = (cam: string, start: number, end: number) => `gateway:${cam}:${start * 1e6}:${end * 1e6}`;

const data = (more: Partial<DayData> = {}): DayData => ({
  entries: [],
  segments: [
    { cam: 'cam0', url: url('cam0', D0 + 36000, D0 + 36300), start: D0 + 36000, end: D0 + 36300 },
    { cam: 'cam0', url: url('cam0', D0 + 36300, D0 + 36600), start: D0 + 36300, end: D0 + 36600 },
    { cam: 'cam0', url: url('cam0', D0 + 39600, D0 + 39900), start: D0 + 39600, end: D0 + 39900 },
  ],
  spans: [],
  ...more,
});

const LINES = [
  { ts_start: D0 + 38000, ts_end: D0 + 38003, text: 'Anyone home?' },
  { ts_start: D0 + 38010, ts_end: D0 + 38012, text: 'In here.' },
];

function setup(
  fetchNotes: (b: number) => Promise<MomentNotes> = async () => ({ cams: {} }),
  {
    more = {},
    blob,
    windowS = 1e9, // a whole segment's room audio at once, unless a test says otherwise
  }: { more?: Partial<DayData>; blob?: (url: string, signal?: AbortSignal) => Promise<Blob>; windowS?: number } = {},
) {
  const fetched: string[] = [];
  const engine = new ReviewEngine({
    cams: ['cam0'],
    fetchNotes,
    now: () => D0 + 2 * 86400,
    fetchBlob: (url, signal) => {
      fetched.push(url);
      return blob ? blob(url, signal) : Promise.resolve(new Blob(['audio']));
    },
    roomWindowS: windowS,
  });
  const v = [new FakeVideo(), new FakeVideo()] as const;
  const audio = new FakeAudio();
  engine.bindVideo('cam0', 0, v[0]);
  engine.bindVideo('cam0', 1, v[1]);
  engine.setAudio(audio);
  engine.pickDay(DAY);
  engine.setDayData(DAY, engine.ui.getState().gen, data({ entries: LINES, ...more }));
  v[0].land(); // the opening seek
  return { engine, v, audio, fetched };
}
const settle = () => new Promise(r => setTimeout(r));

it.each(['__proto__', 'constructor', 'toString'])('binds and reads retained segments for %s', cam => {
  const engine = new ReviewEngine({ cams: [cam], fetchNotes: async () => ({ cams: {} }) });
  engine.bindVideo(cam, 0, new FakeVideo());
  engine.pickDay(DAY);
  engine.setDayData(
    DAY,
    engine.ui.getState().gen,
    data({ segments: [{ cam, url: url(cam, D0 + 10, D0 + 20), start: D0 + 10, end: D0 + 20 }] }),
  );
  expect(engine.segs[cam]?.map(s => [s.start, s.end])).toEqual([[D0 + 10, D0 + 20]]);
  engine.stop();
});

beforeEach(() => {
  hashTiming.everyMs = 0; // each link write lands at once here; the throttle's own test turns it on
  history.replaceState(null, '', '/review');
  Object.assign(URL, { createObjectURL: () => 'blob:clip', revokeObjectURL: () => {} });
});
afterEach(() => {
  vi.useRealTimers();
  usePlayer.setState({ key: null, ts: null, paused: false, status: '' });
});

describe('an embedded recording', () => {
  it('replays a finished embedded clip from its opening point and still resumes a middle pause', () => {
    const engine = new ReviewEngine({ embedded: true, cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    const v = [new FakeVideo(), new FakeVideo()] as const;
    v.forEach((video, i) => {
      video.readyState = 4;
      engine.bindVideo('cam0', i === 0 ? 0 : 1, video);
    });
    const start = D0 + 36000;
    engine.start();
    engine.openClip(start, [{ cam: 'cam0', start, end: start + 10, url: url('cam0', start, start + 10) }]);
    v[0].land();
    v[0].reach(10);
    engine.frame(performance.now() + 1000);
    expect(engine.playing).toBe(false);
    // It rests on its last frame, inside the clip, so the player still has video to show there.
    expect(engine.t).toBeLessThan(start + 10);
    expect(engine.t).toBeGreaterThan(start + 9.9);
    engine.togglePlay();
    expect(engine.t).toBe(start);
    expect(v[0].seeks.at(-1)).toBe(0);
    v[0].land();
    expect(v[0].paused).toBe(false);
    v[0].reach(3);
    engine.frame(performance.now() + 2000);
    engine.togglePlay();
    expect(engine.playing).toBe(false);
    engine.togglePlay();
    expect(engine.t).toBe(start + 3);
    v[0].land();
    expect(v[0].paused).toBe(false);
    engine.stop();
  });

  it('rests just inside the last segment, and never before a segment shorter than the hold', () => {
    expect(endHold(110, { start: 100, end: 110 })).toBeCloseTo(109.95);
    expect(endHold(105, { start: 100, end: 110 })).toBe(105);
    expect(endHold(100.02, { start: 100, end: 100.02 })).toBe(100);
  });

  it('places a video bound after the clip was opened and sought, so a player mounted late plays', () => {
    const engine = new ReviewEngine({ embedded: true, cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.start();
    engine.openClip(D0 + 36010, data().segments);
    engine.seek(D0 + 36012, { play: true }); // before any video: nothing to place yet
    const v = [new FakeVideo(), new FakeVideo()] as const;
    v.forEach((video, i) => {
      engine.bindVideo('cam0', i === 0 ? 0 : 1, video);
    });
    expect(v[0].seeks.at(-1)).toBe(12);
    v[0].land();
    expect(v[0].paused).toBe(false);
    engine.stop();
  });

  it('plays through the existing seek lifecycle without owning the page URL or keyboard', () => {
    history.replaceState(null, '', '/people#Dana');
    const engine = new ReviewEngine({ embedded: true, cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    const v = [new FakeVideo(), new FakeVideo()] as const;
    v.forEach((video, i) => {
      engine.bindVideo('cam0', i === 0 ? 0 : 1, video);
    });
    engine.start();
    engine.openClip(D0 + 36010, data().segments);
    expect(location.pathname + location.hash).toBe('/people#Dana');
    expect(v[0].seeks.at(-1)).toBe(10);
    expect(v[0].plays).toBe(0);
    v[0].land();
    expect(v[0].paused).toBe(false);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }));
    expect(engine.playing).toBe(true);
    engine.stop();
    expect(v[0].paused).toBe(true);
    expect(location.pathname + location.hash).toBe('/people#Dana');
    history.replaceState(null, '', '/');
  });
});

describe('opening a day', () => {
  it('seeks a summary moment from yesterday exactly instead of clamping to the current day', () => {
    const { engine } = setup();
    engine.seek(D0 - 120);
    expect(engine.t).toBe(D0 - 120);
    expect(engine.ui.getState().day).toBe('2026-10-26');
  });
  it('opens a minute before the end of the newest video', () => {
    const { engine, v } = setup();
    expect(engine.t).toBe(D0 + 39900 - 60);
    expect(v[0].src).toBe(url('cam0', D0 + 39600, D0 + 39900));
    expect(v[0].seeks).toEqual([240]);
  });

  it('opens at the time in the link when there is one', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36010}`);
    engine.start();
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    expect(engine.t).toBe(D0 + 36010);
    engine.stop();
  });

  it('opens a clip link at its frame and plays only after the seek lands', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    const v = [new FakeVideo(), new FakeVideo()] as const;
    v.forEach((video, i) => {
      engine.bindVideo('cam0', i === 0 ? 0 : 1, video);
    });
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36010}&play=1`);
    engine.start();
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    expect(v[0].seeks.at(-1)).toBe(10);
    expect(engine.ui.getState().playing).toBe(true);
    expect(v[0].plays).toBe(0);
    v[0].land();
    expect(v[0].plays).toBe(1);
    expect(location.hash).not.toContain('play=');
    engine.stop();
  });

  it('reopens a stopped Review at the same linked moment with fresh play intent', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    const v = [new FakeVideo(), new FakeVideo()] as const;
    v.forEach((video, i) => {
      engine.bindVideo('cam0', i === 0 ? 0 : 1, video);
    });
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36010}`);
    engine.start();
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    v[0].land();
    engine.stop();
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36010}&cam=cam0&play=1`);
    engine.start();
    expect(engine.ui.getState().camera).toBe('cam0');
    expect(engine.ui.getState().playing).toBe(true);
    expect(v[0].seeks.at(-1)).toBe(10);
    v[0].land();
    expect(v[0].paused).toBe(false);
    expect(location.hash).not.toContain('play=');
    engine.stop();
  });

  it('writes the clock only when the time it shows changes, not every frame', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36010}`);
    engine.start();
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    const shown: string[] = [];
    let text: string | null = '';
    engine.setClock({
      get textContent() {
        return text;
      },
      set textContent(v: string | null) {
        text = v;
        shown.push(v ?? '');
      },
    });
    const at = performance.now();
    engine.frame(at + 16);
    engine.frame(at + 32); // paused: the same time, so nothing is written
    engine.seek(D0 + 36070);
    engine.frame(at + 48);
    expect(shown).toEqual(['10:00:10', '10:01:10']);
    engine.stop();
  });

  it('comes back where it was when Review shows again after another tab', () => {
    const engine = new ReviewEngine({
      cams: ['cam0'],
      fetchNotes: async () => ({ cams: {} }),
      now: () => D0 + 2 * 86400,
    });
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36010}`);
    engine.start();
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    engine.seek(D0 + 36100);
    const gen = engine.ui.getState().gen;
    engine.stop(); // hidden
    history.replaceState(null, '', '/review'); // the tab link names no moment
    engine.start();
    // The same day, not opened again (which would load it afresh and move to its latest video)
    expect([engine.ui.getState().day, engine.ui.getState().gen, engine.t]).toEqual([DAY, gen, D0 + 36100]);
    expect(location.hash).toBe(`#day=${DAY}&t=${D0 + 36100}`);
    engine.stop();
  });

  it('follows a link that names a moment when Review shows again', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36010}`);
    engine.start();
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    engine.stop();
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36200}`); // a link from People, say
    engine.start();
    expect(engine.t).toBe(D0 + 36200);
    engine.stop();
  });

  it('leaves the playhead empty on a day with nothing recorded', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.setDayData(DAY, engine.ui.getState().gen, data({ segments: [] }));
    expect(engine.t).toBeNull();
  });

  it('ignores data for a day that is no longer showing', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    const old = engine.ui.getState().gen;
    engine.pickDay('2026-10-26');
    engine.setDayData(DAY, old, data());
    expect(engine.entries).toEqual([]);
    expect(engine.segs).toEqual({});
  });
});

describe('continuous day chunks', () => {
  const NEXT = '2026-10-28';
  const PREV = '2026-10-26';
  const midnight = D0 + 86400;
  const later: DayData = {
    entries: [{ line: 'mic@next', ts_start: midnight + 20, ts_end: midnight + 25, text: 'Next day' }],
    segments: [{ cam: 'cam0', url: url('cam0', midnight, midnight + 300), start: midnight, end: midnight + 300 }],
    spans: [],
  };

  it('initializes a prefetched neighbor selected before the first day finishes loading', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    const video = new FakeVideo();
    engine.bindVideo('cam0', 0, video);
    engine.bindVideo('cam0', 1, new FakeVideo());
    engine.pickDay(DAY);
    engine.setWindowDays([DAY, NEXT]);
    engine.upsertDayData(NEXT, later);
    expect(engine.t).toBeNull();
    expect(engine.entries).toEqual([]);

    engine.pickDay(NEXT);
    const gen = engine.ui.getState().gen;
    // QueryClient republishes a wrapper around the same cached source arrays for the new generation.
    engine.upsertDayData(NEXT, { ...later });
    expect(engine.t).toBe(midnight + 240);
    expect(engine.entries).toEqual(later.entries);
    expect(engine.segs.cam0).toEqual([expect.objectContaining(later.segments[0])]);
    expect(video.src).toBe(later.segments[0]?.url);
    expect(engine.ui.getState().gen).toBe(gen);

    engine.seek(midnight + 60);
    const loads = video.loads;
    engine.upsertDayData(NEXT, { ...later });
    expect(engine.t).toBe(midnight + 60);
    expect(video.loads).toBe(loads);
    engine.stop();
  });

  it('rejoins same-stream midnight media and overlapping audio without mutating cached day snapshots', () => {
    const first: RawSegment = {
      cam: 'cam0',
      start: midnight - 60,
      end: midnight,
      url: `gateway:7:${(midnight - 60) * 1e6}:${midnight * 1e6}`,
    };
    const { engine, v } = setup(undefined, {
      more: { entries: [], segments: [first], spans: [{ device: 'mic-a', start: midnight - 60, end: midnight + 30 }] },
    });
    const source = v[0].src;
    engine.upsertDayData(NEXT, {
      entries: [],
      segments: [
        {
          cam: 'cam0',
          start: midnight,
          end: midnight + 300,
          url: `gateway:7:${midnight * 1e6}:${(midnight + 300) * 1e6}`,
        },
      ],
      spans: [{ device: 'mic-a', start: midnight - 10, end: midnight + 60 }],
    });
    expect(engine.segs.cam0).toHaveLength(1);
    expect(engine.segs.cam0?.[0]).toMatchObject({ start: midnight - 60, end: midnight + 300 });
    expect(v[0].src).toBe(source);
    expect(v[0].extended).toEqual([360]);
    expect(first.end).toBe(midnight);
    expect(engine.spans).toEqual([{ device: 'mic-a', start: midnight - 60, end: midnight + 60 }]);
  });

  it('highlights the exact day snapshot including a line that began before midnight', () => {
    const { engine } = setup();
    engine.upsertDayData(DAY, {
      entries: [
        { ts_start: D0 - 1, ts_end: D0 + 1, text: 'Spans midnight' },
        { ts_start: D0 + 2, ts_end: D0 + 3, text: 'After midnight' },
      ],
      segments: [],
      spans: [],
    });
    const current = vi.fn();
    engine.attachTranscript({ setCur: current });
    engine.seek(D0 + 2);
    expect(current).toHaveBeenLastCalledWith(1, true);
  });

  it('keeps Latest on the playhead day even when the next day is prefetched', () => {
    const { engine } = setup();
    engine.upsertDayData(NEXT, later);
    engine.seek(D0 + 36000);
    engine.latest();
    expect(engine.t).toBe(D0 + 39840);
    expect(engine.ui.getState().day).toBe(DAY);
  });

  it('does not move a failed day landing to its latest recording when Retry later succeeds', () => {
    const { engine } = setup();
    engine.pickDay(NEXT);
    const target = engine.t;
    engine.setDayStatus(NEXT, 'error');
    engine.upsertDayData(NEXT, later);
    engine.setDayStatus(NEXT, 'ready');
    expect(engine.t).toBe(target);
  });

  it('adds either neighbor and replaces one snapshot without resetting cursor, filters, or current video', () => {
    const { engine, v } = setup();
    engine.setWindowDays([PREV, DAY, NEXT]);
    engine.setDayChips(DAY, [chip('person:alice', [[D0 + 36000, D0 + 36600]])]);
    engine.toggleChip('person:alice');
    const cursor = engine.t,
      source = v[0].src,
      loads = v[0].loads;
    engine.upsertDayData(NEXT, later);
    engine.upsertDayData(PREV, {
      entries: [{ line: 'mic@prev', ts_start: D0 - 120, ts_end: D0 - 119, text: 'Yesterday' }],
      segments: [],
      spans: [],
    });
    expect(engine.entries.map(e => e.text)).toEqual(['Yesterday', 'Anyone home?', 'In here.', 'Next day']);
    expect([engine.t, v[0].src, v[0].loads]).toEqual([cursor, source, loads]);
    expect([...engine.ui.getState().sel]).toEqual(['person:alice']);
    engine.upsertDayData(NEXT, { entries: [], segments: [], spans: [] });
    expect(engine.entries.map(e => e.text)).toEqual(['Yesterday', 'Anyone home?', 'In here.']);
    expect(engine.segs.cam0).toHaveLength(3);
  });

  it('ignores a chunk that arrives after its window was left, without moving a newer jump', () => {
    const { engine } = setup();
    engine.seek(midnight + 20);
    engine.setWindowDays([NEXT]);
    engine.upsertDayData(NEXT, later);
    engine.upsertDayData(DAY, data());
    expect(engine.t).toBe(midnight + 20);
    expect(engine.entries.map(e => e.text)).toEqual(['Next day']);
    expect(engine.ui.getState().day).toBe(NEXT);
  });

  it('keeps day-local transcript indices aligned when preceding chunks are inserted', () => {
    const { engine } = setup();
    const current = vi.fn();
    engine.attachTranscript({ setCur: current });
    engine.upsertDayData(PREV, { entries: [{ ts_start: D0 - 60, text: 'Earlier' }], segments: [], spans: [] });
    engine.seek(D0 + 38010);
    expect(current).toHaveBeenLastCalledWith(1, true);
  });

  it('waits for an adjacent pending day and resumes its exact midnight segment without stopping', () => {
    const { engine, v } = setup(undefined, {
      more: {
        segments: [{ cam: 'cam0', url: url('cam0', midnight - 60, midnight), start: midnight - 60, end: midnight }],
        entries: [],
      },
    });
    engine.setWindowDays([DAY, NEXT]);
    engine.setDayStatus(DAY, 'ready');
    engine.setDayStatus(NEXT, 'loading');
    v[0].readyState = 2;
    engine.seek(midnight - 1, { play: true });
    v[0].land();
    v[0].reach(60);
    v[0].ended = true;
    engine.frame(performance.now() + 100);
    expect(engine.playing).toBe(true);
    expect(engine.ui.getState().waiting).toBe(true);
    engine.upsertDayData(NEXT, later);
    engine.setDayStatus(NEXT, 'ready');
    engine.frame(performance.now() + 200);
    expect(engine.t).toBe(midnight);
    expect(engine.playing).toBe(true);
    expect(engine.ui.getState()).toMatchObject({ waiting: false, day: NEXT });
  });

  it('keeps a linked play target pending instead of jumping to a later cached day or stopping', () => {
    const { engine } = setup();
    engine.setWindowDays([DAY, NEXT, '2026-10-29']);
    engine.upsertDayData('2026-10-29', {
      entries: [
        { line: 'much-later', ts_start: midnight + 86400 + 60, ts_end: midnight + 86400 + 90, text: 'Later still' },
      ],
      segments: [],
      spans: [],
    });
    engine.setDayStatus(DAY, 'ready');
    engine.setDayStatus(NEXT, 'loading');
    engine.setDayStatus('2026-10-29', 'ready');
    engine.seekPlay(midnight + 10);
    expect(engine.t).toBe(midnight + 10);
    expect(engine.ui.getState()).toMatchObject({ waiting: true, playing: true, day: NEXT });
    engine.upsertDayData(NEXT, later);
    engine.setDayStatus(NEXT, 'ready');
    engine.frame(performance.now() + 200);
    expect(engine.t).toBe(midnight + 10);
    expect(engine.ui.getState().waiting).toBe(false);
    engine.pause();
    expect(engine.ui.getState()).toMatchObject({ waiting: false, playing: false });
  });

  it('a hash jump while open changes the day and camera at the exact timestamp without replacing the engine generation', () => {
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36010}`);
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.start();
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    const gen = engine.ui.getState().gen;
    history.replaceState(null, '', `/review#day=${PREV}&t=${D0 - 120}&cam=cam0`);
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(engine.t).toBe(D0 - 120);
    expect(engine.ui.getState()).toMatchObject({ day: PREV, gen, camera: 'cam0' });
    engine.stop();
  });
});

describe('offline demo landing', () => {
  afterEach(() => {
    delete (window as Window & { __PACK_DEMO_REVIEW__?: number }).__PACK_DEMO_REVIEW__;
  });
  it.each(['', `#day=${DAY}&t=${D0 + 40}`])('opens the recording and respects an explicit link (%s)', hash => {
    (window as Window & { __PACK_DEMO_REVIEW__?: number }).__PACK_DEMO_REVIEW__ = D0 + 12;
    history.replaceState(null, '', `/review${hash}`);
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.start();
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    expect(engine.t).toBe(hash ? D0 + 40 : D0 + 12);
    expect(engine.ui.getState().day).toBe(DAY);
    engine.stop();
  });
});

describe('landing on a recorded day', () => {
  // The engine's clock says two days after DAY, and nothing was recorded since DAY.
  const opened = (hash = '') => {
    history.replaceState(null, '', `/review${hash}`);
    const engine = new ReviewEngine({
      cams: ['cam0'],
      fetchNotes: async () => ({ cams: {} }),
      now: () => D0 + 2 * 86400 + 3600,
    });
    engine.start();
    return engine;
  };
  const day = (e: ReviewEngine) => e.ui.getState().day;

  it('moves from an empty today to the latest recorded day, once', () => {
    const engine = opened();
    const today = day(engine);
    expect(today).not.toBe(DAY);
    engine.recordedDays(['2026-10-20', DAY]);
    expect(day(engine)).toBe(DAY);
    expect(location.hash).toContain(`day=${DAY}`);
    engine.recordedDays(['2026-10-20', DAY, today as string]); // today starts recording: the page stays put
    expect(day(engine)).toBe(DAY);
    engine.stop();
  });

  it('still moves after a stop and start, as a development remount does', () => {
    const engine = opened(); // the first start writes today into the link
    engine.stop();
    engine.start();
    engine.recordedDays([DAY]);
    expect(day(engine)).toBe(DAY);
    engine.stop();
  });

  it("reads the newest recordings by the server's today when the browser's is behind", () => {
    const engine = opened(); // the browser says DAY + 2
    const later = '2026-10-30';
    engine.recordedDays([DAY, later], later);
    expect(day(engine)).toBe(later);
    engine.stop();
  });

  it('stays on today when today has something', () => {
    const engine = opened();
    const today = day(engine) as string;
    engine.recordedDays([DAY, today]);
    expect(day(engine)).toBe(today);
    engine.stop();
  });

  it('stays on the day a link or a pick asked for', () => {
    const linked = opened(`#day=2026-10-21`);
    linked.recordedDays([DAY]);
    expect(day(linked)).toBe('2026-10-21');
    linked.stop();
    const picked = opened();
    picked.pickDay('2026-10-22');
    picked.recordedDays([DAY]);
    expect(day(picked)).toBe('2026-10-22');
    picked.stop();
  });
});

describe('video', () => {
  it('shows the segment that covers a moment and preloads the next one into the hidden video', () => {
    const { engine, v } = setup();
    engine.seek(D0 + 36100);
    expect(v[0].src).toBe(url('cam0', D0 + 36000, D0 + 36300));
    expect(v[0].seeks.at(-1)).toBe(100);
    expect(v[1].src).toBe(url('cam0', D0 + 36300, D0 + 36600));
    expect(v[1].standby).toBe(true);
  });

  it('swaps to the preloaded video when the playhead moves into the next segment', () => {
    const { engine, v } = setup();
    engine.seek(D0 + 36100);
    v[0].land();
    engine.seek(D0 + 36310);
    expect(v[1].seeks).toEqual([10]);
    expect(v[1].standby).toBe(false);
    expect(v[0].standby).toBe(true);
  });

  it('while scrubbing, drops targets that are overtaken, and seeks at once on release', () => {
    const { engine, v } = setup();
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    engine.scrubStart(D0 + 39610); // within 100 ms of the last seek: waits
    engine.scrubMove(D0 + 39620);
    engine.scrubEnd(D0 + 39640);
    vi.runAllTimers();
    expect(v[0].seeks).toEqual([240, 40]);
  });

  it('while scrubbing, keeps one seek in flight and goes to the latest target when it lands', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    const { engine, v } = setup();
    vi.advanceTimersByTime(200);
    engine.scrubStart(D0 + 39610);
    engine.scrubMove(D0 + 39620);
    engine.scrubMove(D0 + 39630);
    engine.scrubEnd(D0 + 39640);
    expect(v[0].seeks).toEqual([240, 10]); // the first scrub seek is still in flight
    v[0].land();
    expect(v[0].seeks).toEqual([240, 10, 40]); // 20 and 30 were overtaken
  });

  it('lets go of the file when the playhead is where the camera has no video', () => {
    const { engine, v } = setup();
    engine.seek(D0 + 38000);
    expect(v[0].src).toBe('');
    expect(v[0].loads).toBe(1);
  });

  it('grows the stretch still being recorded, tells the player showing it, and adds new ones', () => {
    const { engine, v } = setup();
    engine.seek(D0 + 39800); // in the newest stretch, 39600 to 39900
    const gen = engine.ui.getState().gen;
    const newest = { cam: 'cam0', url: url('cam0', D0 + 39600, D0 + 39900), start: D0 + 39600, end: D0 + 40000 };
    const next = { cam: 'cam0', url: url('cam0', D0 + 40100, D0 + 40200), start: D0 + 40100, end: D0 + 40200 };
    engine.growSegments(DAY, gen, [...data().segments.slice(0, 2), newest, next]);
    expect(engine.segs.cam0?.map(s => [s.start - D0, s.end - D0])).toEqual([
      [36000, 36300],
      [36300, 36600],
      [39600, 40000],
      [40100, 40200],
    ]);
    expect(v[0].extended).toEqual([400]);
    engine.seek(D0 + 39950); // in the part that grew
    engine.frame(performance.now() + 1000);
    expect(engine.ui.getState().avail).toEqual(['cam0']);
    // a listing for another day, or one before the day's data, changes nothing
    engine.growSegments('2026-10-01', gen, [{ ...next, end: D0 + 50000 }]);
    expect(engine.segs.cam0?.at(-1)?.end).toBe(D0 + 40200);
  });

  it('keeps a listed stretch whose start moved under a millisecond, across a millisecond boundary, as the same one', () => {
    const { engine } = setup();
    engine.seek(D0 + 36100);
    const gen = engine.ui.getState().gen;
    const before = engine.segs.cam0?.[0];
    const [first, ...rest] = data().segments as RawSegment[];
    // 0.6 ms later: a different millisecond when rounded, still the same stretch.
    engine.growSegments(DAY, gen, [{ ...(first as RawSegment), start: (first as RawSegment).start + 0.0006 }, ...rest]);
    expect(engine.segs.cam0?.[0]).toBe(before);
  });

  it('marks a segment that fails to load as cut off, and the moment as having no video', () => {
    const { engine, v } = setup();
    engine.seek(D0 + 36100);
    v[0].error = { code: 4 } as MediaError;
    v[0].emit('error');
    engine.frame(performance.now() + 1000);
    expect(engine.segs.cam0?.[0]?.broken).toBe('cut');
    expect(engine.ui.getState()).toMatchObject({ avail: [], gap: 'cut' });
  });

  it('says the video expired where it aged out past the history window, and never plays it', () => {
    const gone = { cam: 'cam0', url: url('cam0', D0 + 30000, D0 + 36000), start: D0 + 30000, end: D0 + 36000 };
    const { engine, v } = setup(undefined, { more: { segments: [{ ...gone, expired: true }, ...data().segments] } });
    engine.seek(D0 + 33000);
    engine.frame(performance.now() + 1000);
    expect(engine.ui.getState()).toMatchObject({ avail: [], gap: 'expired' });
    expect(v.map(x => x.src)).not.toContain(gone.url);
    // A later listing that finds a once-playable stretch aged out replaces it rather than reusing it.
    const gen = engine.ui.getState().gen;
    const [first, ...rest] = data().segments;
    engine.growSegments(DAY, gen, [{ ...(first as RawSegment), expired: true }, ...rest]);
    expect(engine.segs.cam0?.[0]?.broken).toBe('expired');
  });

  it('trims a segment to its file’s length', () => {
    const { engine, v } = setup();
    engine.seek(D0 + 36100);
    Object.assign(v[0], { duration: 120 });
    v[0].emit('loadedmetadata');
    expect(engine.segs.cam0?.[0]?.end).toBe(D0 + 36120);
  });
});

describe('playback', () => {
  it('plays each camera once its seek lands, and pauses on Pause', () => {
    const { engine, v } = setup();
    engine.togglePlay();
    expect(v[0].plays).toBe(0); // not before the seek lands
    v[0].land();
    expect(v[0].plays).toBe(1);
    expect(engine.ui.getState().playing).toBe(true);
    engine.togglePlay();
    expect(v[0].paused).toBe(true);
    expect(engine.ui.getState().playing).toBe(false);
  });

  it('moves the playhead with the video', () => {
    const { engine, v } = setup();
    engine.seekPlay(D0 + 36100);
    v[0].land();
    Object.assign(v[0], { readyState: 4 });
    v[0].reach(130);
    engine.frame(performance.now() + 1000);
    expect(engine.t).toBe(D0 + 36130);
  });

  it('rolls over into the next segment, already preloaded', () => {
    const { engine, v } = setup();
    engine.seekPlay(D0 + 36290);
    v[0].land();
    Object.assign(v[0], { readyState: 4 });
    v[0].reach(300);
    engine.frame(performance.now() + 1000);
    expect(engine.t).toBe(D0 + 36300);
    expect(v[1].standby).toBe(false);
    v[1].land();
    expect(v[1].plays).toBe(1);
  });

  it('plays the room audio for the segment under the video', () => {
    const { engine, fetched } = setup();
    engine.seekPlay(D0 + 36100);
    expect(fetched).toEqual([`/audio/range?start=${D0 + 36000}&end=${D0 + 36300}`]);
  });

  it('fetches the room audio of a long segment a window at a time', () => {
    const { engine, fetched } = setup(undefined, { windowS: 120 });
    const at = Math.ceil((D0 + 36000) / 120) * 120 + 10; // 10 s into the segment's second window
    engine.seekPlay(at);
    expect(fetched).toEqual([`/audio/range?start=${at - 10}&end=${at + 110}`]);
  });

  describe('room audio ready before it is needed', () => {
    const range = (a: number, b: number) => `/audio/range?start=${D0 + a}&end=${D0 + b}`;

    it('has a paused spot’s room audio in the player before Play, so Play fetches nothing and starts it at once', async () => {
      vi.useFakeTimers();
      const { engine, v, audio, fetched } = setup();
      engine.jump(D0 + 36100); // paused
      vi.advanceTimersByTime(400);
      await vi.runAllTimersAsync();
      expect(fetched).toEqual([range(36000, 36300)]);
      expect(audio.src).toBe('blob:clip'); // already in the <audio>
      audio.readyState = 4;
      engine.seekPlay(D0 + 36100);
      v[0].land();
      v[0].playing();
      expect(fetched).toHaveLength(1); // nothing new
      expect(audio.paused).toBe(false); // the sound starts with the picture
      expect(audio.currentTime).toBeCloseTo(100, 1);
    });

    it('fetches only where clicking around while paused stops', async () => {
      vi.useFakeTimers();
      const { engine, fetched } = setup();
      engine.jump(D0 + 36100);
      vi.advanceTimersByTime(100);
      engine.jump(D0 + 36400);
      vi.advanceTimersByTime(100);
      engine.jump(D0 + 36050);
      vi.advanceTimersByTime(400);
      await vi.runAllTimersAsync();
      expect(fetched).toEqual([range(36000, 36300)]);
    });

    it('fetches the next stretch before playback reaches it, and uses it when it does', async () => {
      const { engine, v, audio, fetched } = setup();
      engine.seekPlay(D0 + 36280); // 20 s before this segment ends
      await settle();
      audio.readyState = 4;
      audio.emit('loadedmetadata');
      v[0].land();
      v[0].playing();
      engine.frame(performance.now() + 1000);
      expect(fetched).toEqual([range(36000, 36300), range(36300, 36600)]);
      engine.seekPlay(D0 + 36310); // into the next segment
      await settle();
      expect(fetched).toHaveLength(2); // kept, not fetched again
    });

    it('unlocks a preloaded <audio> inside the click, so a strict browser lets its sound start later (Safari)', async () => {
      vi.useFakeTimers();
      const { engine, audio } = setup();
      engine.jump(D0 + 36100);
      vi.advanceTimersByTime(400);
      await vi.runAllTimersAsync();
      expect(audio.src).toBe('blob:clip');
      const before = audio.plays;
      engine.seekPlay(D0 + 36100); // the camera is still seeking, so startRoom waits
      expect(audio.plays).toBe(before + 1); // played (and paused) inside the click
    });

    it('asks once for a stretch whose audio failed, not every frame', async () => {
      const { engine, v, audio, fetched } = setup(undefined, {
        blob: url =>
          url.includes(`start=${D0 + 36300}`) ? Promise.reject(new Error('404')) : Promise.resolve(new Blob(['a'])),
      });
      engine.seekPlay(D0 + 36280);
      await settle();
      audio.readyState = 4;
      audio.emit('loadedmetadata');
      v[0].land();
      v[0].playing();
      let now = performance.now();
      for (let i = 0; i < 5; i++) {
        now += 1000;
        engine.frame(now);
        await settle();
      }
      expect(fetched.filter(u => u === range(36300, 36600))).toHaveLength(1);
    });

    it('drops the paused spot it left while its audio was still downloading', async () => {
      vi.useFakeTimers();
      const signals: Record<string, AbortSignal | undefined> = {};
      const { engine } = setup(undefined, {
        blob: (url, signal) => {
          signals[url] = signal;
          return new Promise(() => {});
        },
      });
      engine.jump(D0 + 36100);
      vi.advanceTimersByTime(400);
      engine.jump(D0 + 39700);
      vi.advanceTimersByTime(400);
      expect(signals[range(36000, 36300)]?.aborted).toBe(true);
      expect(signals[range(39600, 39900)]?.aborted).toBe(false);
    });

    it("doesn't preload the last day's audio after another day is picked", async () => {
      vi.useFakeTimers();
      const { engine, fetched } = setup();
      engine.jump(D0 + 36100);
      engine.pickDay('2026-10-25'); // within the 300 ms
      vi.advanceTimersByTime(400);
      await vi.runAllTimersAsync();
      expect(fetched).toEqual([]);
    });

    it('fetches a stretch once, however often playback goes back to it', async () => {
      const { engine, fetched } = setup();
      engine.seekPlay(D0 + 36100);
      await settle();
      engine.seekPlay(D0 + 39700);
      await settle();
      engine.seekPlay(D0 + 36150);
      await settle();
      expect(fetched.filter(u => u === range(36000, 36300))).toHaveLength(1);
    });
  });

  it('plays the lines one after another where there is no video', async () => {
    const { engine, audio, fetched } = setup();
    engine.seekPlay(D0 + 38000);
    await settle();
    audio.emit('loadedmetadata');
    expect(fetched).toEqual([`/clip?ts_start=${D0 + 38000}`]);
    expect(audio.plays).toBe(1);
    engine.audioEnded();
    expect(fetched.at(-1)).toBe(`/clip?ts_start=${D0 + 38010}`);
    expect(engine.t).toBe(D0 + 38010);
  });

  it('skips ahead to the next video or speech across a gap', () => {
    const { engine } = setup();
    engine.seekPlay(D0 + 37000); // between the morning's video and the lines
    engine.frame(performance.now() + 1000);
    expect(engine.t).toBe(D0 + 38000);
  });

  it('with Skip the rest on, plays only the selected chips’ time and stops after it', () => {
    const { engine } = setup();
    engine.setChips(DAY, engine.ui.getState().gen, [
      { id: 'object:cup', kind: 'object', label: 'cup', spans: [[D0 + 36200, D0 + 36210]], total_s: 10, count: 1 },
    ]);
    engine.toggleChip('object:cup');
    engine.toggleSkip();
    engine.seekPlay(D0 + 36100);
    engine.frame(performance.now() + 1000);
    expect(engine.t).toBe(D0 + 36200);
    engine.seekPlay(D0 + 36250);
    engine.frame(performance.now() + 2000);
    expect(engine.ui.getState().playing).toBe(false);
  });

  it('keeps a camera-removed OR alternative known empty while another alternative overlaps the person', () => {
    const { engine } = setup();
    engine.setChips(DAY, engine.ui.getState().gen, [
      { id: 'person:alice', kind: 'person', label: 'Alice', spans: [[D0 + 36000, D0 + 36300]], total_s: 300, count: 1 },
      {
        id: 'event:a',
        kind: 'event',
        label: 'A',
        camera: 'other',
        spans: [[D0 + 36010, D0 + 36020]],
        total_s: 10,
        count: 1,
      },
      {
        id: 'event:b',
        kind: 'event',
        label: 'B',
        camera: 'cam0',
        spans: [[D0 + 36100, D0 + 36120]],
        total_s: 20,
        count: 1,
      },
    ]);
    engine.toggleChip('person:alice');
    engine.toggleChip('event:a');
    engine.toggleChip('event:b');
    engine.pickCameras(['cam0']);
    expect(engine.present).toEqual([[D0 + 36100, D0 + 36120]]);
    expect(engine.ui.getState().filterStatus).toBeNull();
  });

  it('stops Skip playback for a known empty intersection and refuses to play everything on the next click', () => {
    const { engine, v } = setup();
    engine.setChips(DAY, engine.ui.getState().gen, [
      { id: 'person:alice', kind: 'person', label: 'Alice', spans: [[D0 + 36000, D0 + 36010]], total_s: 10, count: 1 },
      { id: 'object:mug', kind: 'object', label: 'Mug', spans: [[D0 + 36100, D0 + 36110]], total_s: 10, count: 1 },
    ]);
    engine.toggleChip('person:alice');
    engine.toggleSkip();
    engine.seekPlay(D0 + 36005);
    engine.toggleChip('object:mug');
    expect(engine.present).toEqual([]);
    expect(engine.ui.getState()).toMatchObject({ playing: false, filterStatus: 'empty' });
    const played = v[0].plays + v[1].plays;
    engine.seekPlay(D0 + 36100);
    expect(v[0].plays + v[1].plays).toBe(played);
    expect(engine.playing).toBe(false);
  });

  it('keeps unknown timing neutral under Skip and restores confirmed status when that option is removed', () => {
    const { engine } = setup();
    engine.setChips(DAY, engine.ui.getState().gen, [
      { id: 'object:unknown', kind: 'object', label: 'Unknown timing', spans: [], total_s: 0, count: 1 },
    ]);
    engine.toggleChip('object:unknown');
    engine.toggleSkip();
    expect(engine.present).toBeNull();
    expect(engine.ui.getState().filterStatus).toBe('unknown');
    engine.seekPlay(D0 + 36100);
    engine.frame(performance.now() + 1000);
    expect(engine.playing).toBe(true);
    engine.clearChips();
    expect(engine.ui.getState().filterStatus).toBeNull();
  });

  it('pauses while scrubbing and plays on from where the scrub let go', () => {
    const { engine } = setup();
    engine.seekPlay(D0 + 36100);
    engine.scrubStart(D0 + 36150);
    expect(engine.ui.getState().playing).toBe(false);
    engine.scrubEnd(D0 + 36160);
    expect(engine.ui.getState()).toMatchObject({ playing: true });
    expect(engine.t).toBe(D0 + 36160);
  });

  it('stops playing on a plain seek, and keeps playing on a jump', () => {
    const { engine } = setup();
    engine.seekPlay(D0 + 36100);
    engine.jump(D0 + 36150);
    expect(engine.playing).toBe(true);
    engine.seek(D0 + 36160);
    expect(engine.playing).toBe(false);
  });

  it('goes on to the next seek when a file fails to load mid-seek', () => {
    const { engine, v } = setup();
    engine.seek(D0 + 36100); // the first segment, in the shown video: still seeking
    engine.seekPlay(D0 + 36400); // queued behind it: the second segment, preloaded in the hidden video
    v[0].error = { code: 4 } as MediaError;
    v[0].emit('error');
    expect(v[1].seeks).toEqual([100]);
  });

  it('with Skip the rest on, starts a line’s clip at the chip’s time instead of fetching it every frame', async () => {
    const { engine, audio, fetched } = setup();
    const clips = () => fetched.filter(u => u.startsWith('/clip'));
    // A chip whose time starts 1.5 s into the first line, where there's no video.
    engine.setChips(DAY, engine.ui.getState().gen, [
      {
        id: 'event:knock',
        kind: 'event',
        label: 'knock',
        spans: [[D0 + 38001.5, D0 + 38020]],
        total_s: 18.5,
        count: 1,
      },
    ]);
    engine.toggleChip('event:knock');
    engine.toggleSkip();
    engine.seekPlay(D0 + 37990);
    engine.frame(performance.now() + 1000); // skip jumps to the chip's start
    const fetches = clips().length;
    for (let i = 2; i <= 6; i++) engine.frame(performance.now() + i * 1000);
    expect(clips().length).toBe(fetches);
    expect(engine.t).toBe(D0 + 38001.5);
    await settle();
    audio.emit('loadedmetadata');
    expect(audio.currentTime).toBeCloseTo(1.5 + 0.4); // the clip starts 0.4 s before its line
  });

  it('plays the room audio once a line ends where video has begun', async () => {
    const late = { cam: 'cam0', url: url('cam0', D0 + 38005, D0 + 38305), start: D0 + 38005, end: D0 + 38305 };
    const { engine, audio, fetched } = setup(undefined, { more: { segments: [...data().segments, late] } });
    engine.seekPlay(D0 + 38000); // no video yet: the line plays
    await settle();
    audio.emit('loadedmetadata');
    audio.currentTime = 6.4; // 6 s into the line, and video has begun at 5 s
    engine.frame(performance.now() + 1000);
    engine.audioEnded();
    expect(fetched.at(-1)).toBe(`/audio/range?start=${D0 + 38005}&end=${D0 + 38305}`);
  });

  it('starts the room audio when the video starts playing, not before, at the video’s moment', async () => {
    const { engine, v, audio } = setup();
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata'); // the room audio is ready before the video's seek has landed
    expect(audio.plays).toBe(0);
    v[0].land(); // the video plays from 100 s into its segment
    expect(audio.plays).toBe(1);
    expect(audio.currentTime).toBe(100);
  });

  it('waits for the picture’s frames, not just play(), before starting the room audio', async () => {
    const { engine, v, audio } = setup();
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    v[0].deferPlaying = true;
    v[0].land(); // play() is called, but no frames yet
    expect(audio.plays).toBe(0);
    v[0].playing();
    expect(audio.plays).toBe(1);
  });

  it('never plays the last segment’s audio for a new one while the new one downloads', async () => {
    let release: (b: Blob) => void = () => {};
    const { engine, v, audio, fetched } = setup(undefined, {
      blob: url =>
        url.includes(`start=${D0 + 36300}`) ? new Promise(r => (release = r)) : Promise.resolve(new Blob(['room'])),
    });
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    v[0].land(); // the first segment's room audio plays
    expect(audio.plays).toBe(1);
    engine.seekPlay(D0 + 36400); // into the next segment, whose audio is still downloading
    v[1].land();
    expect(fetched.at(-1)).toBe(`/audio/range?start=${D0 + 36300}&end=${D0 + 36600}`);
    expect(audio.plays).toBe(1); // the old blob doesn't play at the new segment's offset
    release(new Blob(['room']));
    await settle();
    audio.emit('loadedmetadata');
    expect(audio.plays).toBe(2);
    expect(audio.currentTime).toBe(100);
  });

  it('lines the room audio up exactly when it arrives after the picture has started', async () => {
    let release: (b: Blob) => void = () => {};
    const { engine, v, audio } = setup(undefined, { blob: () => new Promise(r => (release = r)) });
    engine.seekPlay(D0 + 36000.25); // a quarter second into the segment
    v[0].land(); // the picture plays; the room audio is still on its way
    release(new Blob(['room']));
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata'); // the new element sits at 0, a quarter second behind
    expect(audio.currentTime).toBe(0.25);
  });

  it('pauses the room audio while the picture stalls', async () => {
    const { engine, v, audio } = setup();
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    v[0].land();
    engine.frame(performance.now() + 1000);
    v[0].land();
    expect(audio.paused).toBe(false);
    v[0].emit('waiting'); // the video ran out of data
    engine.frame(performance.now() + 2000);
    expect(audio.paused).toBe(true);
    v[0].playing(); // frames again
    expect(audio.paused).toBe(false);
  });

  it('seeks each camera once per Play, not again on the next frame', async () => {
    const { engine, v } = setup();
    const before = v[0].seeks.length + v[1].seeks.length;
    engine.seekPlay(D0 + 36100);
    v[0].land();
    v[1].land();
    let now = performance.now();
    for (let i = 0; i < 3; i++) {
      now += 1000;
      engine.frame(now);
      v[0].land();
      v[1].land();
    }
    await settle();
    expect(v[0].seeks.length + v[1].seeks.length - before).toBe(1);
  });

  it('plays every camera and the room audio at the chosen speed, easing around it', async () => {
    const { engine, v, audio } = setup();
    engine.setSpeed(2);
    expect([v[0].playbackRate, v[1].playbackRate, audio.playbackRate]).toEqual([2, 2, 2]);
    expect(v[0].defaultPlaybackRate).toBe(2); // kept when the next segment loads
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    v[0].land();
    v[0].playing();
    expect(audio.playbackRate).toBe(2);
    let now = performance.now();
    now += 1000;
    engine.frame(now);
    audio.currentTime = 99.9; // a little behind the picture
    engine.frame(now + 1000);
    expect(audio.playbackRate).toBeGreaterThan(2);
    expect(audio.playbackRate).toBeLessThan(2.2);
  });

  it('eases the room audio back in step by its rate, and snaps only a large offset', async () => {
    const { engine, v, audio } = setup();
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    v[0].land();
    let now = performance.now();
    const sync = () => {
      now += 1000;
      engine.frame(now);
      v[0].land(); // a resync can re-place the camera; its seek lands
    };
    sync();
    audio.currentTime = 99.9; // 0.1 s behind the picture
    sync();
    expect(audio.playbackRate).toBeGreaterThan(1);
    expect(audio.currentTime).toBe(99.9); // not snapped: no audible jump
    audio.currentTime = 100.2; // 0.2 s ahead
    sync();
    expect(audio.playbackRate).toBeLessThan(1);
    audio.currentTime = 100.01; // close enough
    sync();
    expect(audio.playbackRate).toBe(1);
    audio.currentTime = 101; // a second out: snap back
    sync();
    expect(audio.playbackRate).toBe(1);
    expect(audio.currentTime).toBe(100);
  });

  it('keeps the room audio it has when playback resumes in the same segment', async () => {
    const { engine, v, audio, fetched } = setup();
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    v[0].land(); // the video plays, and the room audio with it
    engine.togglePlay(); // pause
    engine.togglePlay(); // and resume
    v[0].land();
    engine.jump(D0 + 36150); // an arrow press while playing
    v[0].land();
    expect(fetched.filter(u => u.startsWith('/audio/range'))).toHaveLength(1);
    expect(audio.plays).toBe(3);
    expect(audio.paused).toBe(false);
  });

  it('leaves a line’s clip silent when playback pauses while it loads', async () => {
    const { engine, audio } = setup();
    engine.seekPlay(D0 + 38000); // a line, no video: its clip is fetched
    engine.togglePlay(); // paused before it arrives
    await settle();
    audio.emit('loadedmetadata');
    expect(audio.plays).toBe(0);
  });

  it('leaves a line’s clip silent when playback pauses while its metadata loads', async () => {
    const { engine, audio } = setup();
    engine.seekPlay(D0 + 38000);
    await settle(); // the clip arrived; the <audio> is loading it
    engine.togglePlay();
    audio.emit('loadedmetadata');
    expect(audio.plays).toBe(0);
  });

  it('drops an audio fetch the playhead has moved on from', () => {
    const signals: (AbortSignal | undefined)[] = [];
    const { engine } = setup(undefined, {
      blob: (_url, signal) => {
        signals.push(signal);
        return new Promise(() => {}); // still loading
      },
    });
    engine.seekPlay(D0 + 36100);
    engine.seekPlay(D0 + 36400); // the next segment, before the first one's audio arrived
    expect(signals.map(s => s?.aborted)).toEqual([true, false]);
  });

  it('says the browser blocked the sound, and logs why, when it refuses to play it', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { engine, audio } = setup();
    audio.refuse = new DOMException(
      "play() failed because the user didn't interact with the document first.",
      'NotAllowedError',
    );
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    await settle();
    expect(engine.ui.getState().notice).toBe('The browser blocked the sound; click Play again.');
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('refused'),
      'NotAllowedError',
      expect.stringContaining('interact'),
    );
  });

  it('logs a blocked sound once, not on every retry while it stays blocked', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { engine, v, audio } = setup();
    audio.refuse = new DOMException('blocked', 'NotAllowedError');
    engine.seekPlay(D0 + 36100);
    v[0].land();
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    for (let i = 1; i <= 4; i++) {
      engine.frame(performance.now() + i * 1000); // the room audio resyncs, and retries
      v[0].land();
    }
    await settle();
    expect(audio.plays).toBeGreaterThan(2);
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it('clears that notice once the sound plays', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { engine, v, audio } = setup();
    audio.refuse = new DOMException('blocked', 'NotAllowedError');
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    v[0].land();
    await settle();
    audio.refuse = null;
    engine.togglePlay(); // pause
    engine.togglePlay(); // and Play again
    v[0].land();
    await settle();
    expect(engine.ui.getState().notice).toBeNull();
  });

  it('clears any sound error once the sound plays', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { engine, v, audio } = setup();
    audio.refuse = new DOMException('The element has no supported sources.', 'NotSupportedError');
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    v[0].land();
    await settle();
    expect(engine.ui.getState().notice).toBe("Couldn't play the sound (NotSupportedError).");
    audio.refuse = null;
    engine.togglePlay();
    engine.togglePlay(); // the sound plays now
    v[0].land();
    await settle();
    expect(engine.ui.getState().notice).toBeNull();
  });

  it('only logs a play() that its own pause or load interrupted', async () => {
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => {});
    const { engine, audio } = setup();
    audio.refuse = new DOMException('The play() request was interrupted by a call to pause().', 'AbortError');
    engine.seekPlay(D0 + 36100);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    await settle();
    expect(engine.ui.getState().notice).toBeNull();
    expect(debug).toHaveBeenCalledWith(expect.any(String), 'AbortError', expect.any(String));
  });

  it('unlocks the <audio> inside the click, before the room audio has arrived', () => {
    const { engine, audio } = setup(undefined, { blob: () => new Promise(() => {}) }); // still downloading
    engine.seekPlay(D0 + 36100);
    expect(audio.unlocks).toBe(1);
    expect(audio.src).toMatch(/^data:audio\/wav/);
  });

  it('doesn’t take the unlocking silence ending for the line ending', () => {
    const { engine, fetched } = setup(undefined, { blob: () => new Promise(() => {}) });
    engine.seekPlay(D0 + 38000); // a line, no video: silence unlocks the <audio> while the clip downloads
    engine.audioEnded(); // the silence ends
    expect(fetched.filter(u => u.startsWith('/clip'))).toEqual([`/clip?ts_start=${D0 + 38000}`]);
    expect(engine.playing).toBe(true);
  });

  it('plays on with the kept room audio after the video ends, where nobody spoke', () => {
    const kept = { more: { spans: [{ start: D0 + 36600, end: D0 + 36900 }] } }; // 5 min of room audio, no video
    const { engine, v, fetched } = setup(undefined, kept);
    engine.seekPlay(D0 + 36590); // the end of the second segment
    v[0].land();
    Object.assign(v[0], { readyState: 4 });
    v[0].reach(300); // its video ends
    engine.frame(performance.now() + 1000);
    expect(engine.t).toBe(D0 + 36600); // not on to 39600, the next video
    expect(engine.playing).toBe(true);
    expect(fetched.at(-1)).toBe(`/audio/range?start=${D0 + 36600}&end=${D0 + 36900}`);
    expect(engine.ui.getState().roomOnly).toBe(true);
  });

  it('moves the playhead with that audio, and goes on to what plays next when it ends', async () => {
    const kept = { more: { spans: [{ start: D0 + 36600, end: D0 + 36900 }] } };
    const { engine, audio } = setup(undefined, kept);
    engine.seekPlay(D0 + 36700); // no video here, no lines: the room audio
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    expect(audio.plays).toBe(1);
    expect(audio.currentTime).toBe(100);
    audio.currentTime = 150;
    engine.frame(performance.now() + 1000);
    expect(engine.t).toBe(D0 + 36750);
    engine.audioEnded(); // the stretch is over
    expect(engine.t).toBe(D0 + 38000); // the day's next line, before the next video at 39600
    expect(engine.ui.getState().roomOnly).toBe(false);
  });

  it('plays the lines one by one where no room audio was kept, as before', async () => {
    const { engine, fetched } = setup(); // no spans
    engine.seekPlay(D0 + 38000);
    expect(fetched).toEqual([`/clip?ts_start=${D0 + 38000}`]);
  });

  it('still skips to the selected chips’ time through room audio', () => {
    const kept = { more: { spans: [{ start: D0 + 36600, end: D0 + 36900 }] } };
    const { engine } = setup(undefined, kept);
    engine.setChips(DAY, engine.ui.getState().gen, [
      { id: 'event:door', kind: 'event', label: 'door', spans: [[D0 + 36800, D0 + 36810]], total_s: 10, count: 1 },
    ]);
    engine.toggleChip('event:door');
    engine.toggleSkip();
    engine.seekPlay(D0 + 36700);
    engine.frame(performance.now() + 1000);
    expect(engine.t).toBe(D0 + 36800);
  });

  it('falls back to the lines when the kept audio won’t load', async () => {
    const kept = {
      more: { spans: [{ start: D0 + 37900, end: D0 + 38200 }] },
      blob: (url: string) =>
        url.startsWith('/audio/range') ? Promise.reject(new Error('500')) : Promise.resolve(new Blob(['clip'])),
    };
    const { engine, fetched } = setup(undefined, kept);
    engine.seekPlay(D0 + 38000);
    await settle();
    expect(fetched).toEqual([`/audio/range?start=${D0 + 37900}&end=${D0 + 38200}`, `/clip?ts_start=${D0 + 38000}`]);
    expect(engine.ui.getState().roomOnly).toBe(false);
  });

  it('goes from a line into kept audio that follows it', async () => {
    const kept = { more: { spans: [{ start: D0 + 38005, end: D0 + 38300 }] } }; // after the first line, before the second
    const { engine, fetched } = setup(undefined, kept);
    engine.seekPlay(D0 + 38000); // the first line, no kept audio there
    await settle(); // its clip replaces the unlocking silence
    engine.audioEnded(); // it ends at 38003
    expect(engine.t).toBe(D0 + 38005);
    expect(fetched.at(-1)).toBe(`/audio/range?start=${D0 + 38005}&end=${D0 + 38300}`);
  });

  it('pauses the room audio while a camera that starts mid-stretch loads, so nothing is heard twice', async () => {
    const late = { cam: 'cam0', url: url('cam0', D0 + 36700, D0 + 37000), start: D0 + 36700, end: D0 + 37000 };
    const kept = { more: { segments: [...data().segments, late], spans: [{ start: D0 + 36600, end: D0 + 36900 }] } };
    const { engine, audio } = setup(undefined, kept);
    engine.seekPlay(D0 + 36650); // after the second segment, before the late one: room audio only
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    expect(audio.paused).toBe(false);
    audio.currentTime = 100.5; // 36700.5: the late camera's video has begun
    engine.frame(performance.now() + 1000);
    expect(audio.paused).toBe(true);
  });

  it('plays on into the next kept stretch when it starts a moment after this one ends', async () => {
    const kept = {
      more: {
        spans: [
          { start: D0 + 36600, end: D0 + 36900 },
          { start: D0 + 36900.004, end: D0 + 37200 },
        ],
      },
    };
    const { engine, fetched } = setup(undefined, kept);
    engine.seekPlay(D0 + 36700);
    await settle(); // the room audio replaces the unlocking silence
    engine.audioEnded();
    expect(engine.t).toBe(D0 + 36900.004);
    expect(fetched.at(-1)).toBe(`/audio/range?start=${D0 + 36900.004}&end=${D0 + 37200}`);
  });

  it('tries the room audio again when the browser refused it, once allowed', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const kept = { more: { spans: [{ start: D0 + 36600, end: D0 + 36900 }] } };
    const { engine, audio } = setup(undefined, kept);
    audio.refuse = new DOMException('blocked', 'NotAllowedError');
    engine.seekPlay(D0 + 36700);
    await settle();
    audio.readyState = 4;
    audio.emit('loadedmetadata');
    await settle();
    audio.refuse = null; // the viewer clicked somewhere
    engine.frame(performance.now() + 1000);
    await settle();
    expect(audio.paused).toBe(false);
  });

  it('rolls over a short gap between videos without stopping for the room audio', () => {
    const near = { cam: 'cam0', url: url('cam0', D0 + 36602, D0 + 36900), start: D0 + 36602, end: D0 + 36900 };
    const kept = { more: { segments: [...data().segments, near], spans: [{ start: D0 + 36600, end: D0 + 36900 }] } };
    const { engine, v } = setup(undefined, kept);
    engine.seekPlay(D0 + 36590);
    v[0].land();
    Object.assign(v[0], { readyState: 4 });
    v[0].reach(300);
    engine.frame(performance.now() + 1000);
    expect(engine.t).toBe(D0 + 36602); // straight to the next video, 2 s on
  });

  it('holds the playhead where a seek sent it while that seek is on its way', () => {
    const { engine } = setup();
    engine.seek(D0 + 36100); // a seek into the first segment, still in flight
    engine.seekPlay(D0 + 36400); // then play from the second segment
    engine.frame(performance.now() + 1000);
    expect(engine.t).toBe(D0 + 36400);
  });
});

const chip = (id: string, spans: [number, number][]): Chip => ({
  id,
  kind: 'object',
  label: id,
  spans,
  total_s: 0,
  count: spans.length,
});

describe('chips', () => {
  const cup = chip('object:cup', [
    [D0 + 36050, D0 + 36060],
    [D0 + 39700, D0 + 39720],
  ]);
  const dog = chip('object:dog', [[D0 + 36400, D0 + 36410]]);

  it('steps through the selected chips’ time', () => {
    const { engine } = setup();
    const { gen } = engine.ui.getState();
    engine.setChips(DAY, gen, [cup, dog]);
    engine.toggleChip('object:cup');
    engine.seek(D0 + 36000);
    engine.stepChip(1);
    expect(engine.t).toBe(D0 + 36050);
    engine.stepChip(1);
    expect(engine.t).toBe(D0 + 39700);
    engine.stepChip(-1);
    expect(engine.t).toBe(D0 + 36050);
  });

  it('gives selected chips different colours', () => {
    const { engine } = setup();
    engine.setChips(DAY, engine.ui.getState().gen, [cup, dog]);
    engine.toggleChip('object:cup');
    engine.toggleChip('object:dog');
    const { colors } = engine.ui.getState();
    expect(Object.keys(colors)).toEqual(['object:cup', 'object:dog']);
    expect(colors['object:cup']).not.toBe(colors['object:dog']);
  });

  it('keeps the selection when the chips refresh, less any that went away', () => {
    const { engine } = setup();
    const { gen } = engine.ui.getState();
    engine.setChips(DAY, gen, [cup, dog]);
    engine.toggleChip('object:cup');
    engine.toggleChip('object:dog');
    engine.setChips(DAY, gen, [cup]);
    expect([...engine.ui.getState().sel]).toEqual(['object:cup']);
  });

  it('keeps selected filters when the day is picked again', () => {
    const { engine } = setup();
    engine.setChips(DAY, engine.ui.getState().gen, [cup]);
    engine.toggleChip('object:cup');
    engine.pickDay(DAY);
    engine.setChips(DAY, engine.ui.getState().gen, [cup]);
    expect([...engine.ui.getState().sel]).toEqual(['object:cup']);
  });

  it('keeps filters and their loaded marks while jumping to another day', () => {
    const { engine } = setup();
    engine.setChips(DAY, engine.ui.getState().gen, [cup]);
    engine.toggleChip('object:cup');
    engine.pickDay('2026-10-26');
    expect([...engine.ui.getState().sel]).toEqual(['object:cup']);
    expect(engine.present).toEqual(cup.spans);
    expect(engine.marks).toHaveLength(1);
  });

  it('remembers Skip the rest in this browser', () => {
    const { engine } = setup();
    engine.toggleSkip();
    expect(JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}')).toMatchObject({ reviewSkip: true });
    expect(new ReviewEngine({ cams: [], fetchNotes: async () => ({ cams: {} }) }).ui.getState().skip).toBe(true);
  });
});

describe('camera notes', () => {
  it('fetches the notes for the playhead’s 5 s bucket once, then reuses them', async () => {
    const notes: MomentNotes = { cams: { cam0: { ts: D0 + 36101, dt: 1, caption: 'A dog on the sofa', faces: [] } } };
    const fetchNotes = vi.fn(async (_bucket: number) => notes);
    const { engine } = setup(fetchNotes);
    vi.useFakeTimers();
    engine.seek(D0 + 36101);
    engine.frame(performance.now() + 1000);
    await vi.runAllTimersAsync();
    expect(engine.ui.getState().notes).toEqual(notes);
    engine.seek(D0 + 36200);
    engine.frame(performance.now() + 2000);
    engine.seek(D0 + 36103);
    engine.frame(performance.now() + 3000);
    await vi.runAllTimersAsync();
    expect(fetchNotes.mock.calls.map(c => c[0])).toEqual([(D0 + 36100) / 5, (D0 + 36200) / 5]);
    expect(engine.ui.getState().notes).toEqual(notes);
  });

  it('stops asking once the server says camera history is not shown here', async () => {
    const fetchNotes = vi.fn(async () => {
      throw new HttpError('camera history is only shown on this machine', 403);
    });
    const { engine } = setup(fetchNotes);
    vi.useFakeTimers();
    engine.seek(D0 + 36100);
    engine.frame(performance.now() + 1000);
    await vi.runAllTimersAsync();
    engine.seek(D0 + 36300);
    engine.frame(performance.now() + 2000);
    await vi.runAllTimersAsync();
    expect(fetchNotes).toHaveBeenCalledTimes(1);
    expect(engine.ui.getState().notes).toEqual({ cams: {}, forbidden: true });
  });
});

describe('Latest', () => {
  it('goes to the newest full minute of video and centres its transcript line', () => {
    const { engine } = setup();
    const setCur = vi.fn();
    engine.attachTranscript({ setCur });
    engine.seek(D0 + 36000);
    setCur.mockClear();
    engine.latest();
    expect(engine.t).toBe(D0 + 39840);
    expect(setCur).toHaveBeenLastCalledWith(1, true); // after both lines
  });
});

describe('keys', () => {
  const press = (key: string, target: EventTarget = document.body, more: KeyboardEventInit = {}) =>
    target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...more }));

  it('nudges, steps through lines, plays, and zooms', () => {
    const { engine } = setup();
    engine.start();
    engine.seek(D0 + 38005);
    engine.setView(D0 + 37700, D0 + 38300); // ten minutes across: a nudge is 4 s
    press('ArrowRight');
    expect(engine.t).toBe(D0 + 38009);
    press('k');
    expect(engine.t).toBe(D0 + 38010);
    press('j');
    expect(engine.t).toBe(D0 + 38000);
    press(' ');
    expect(engine.playing).toBe(true);
    press('-');
    engine.frame(performance.now() + 10_000); // past the zoom's glide
    expect(engine.view.t1 - engine.view.t0).toBe(1200);
    engine.stop();
  });

  it('halves or doubles where a glide is headed when + or - comes during it', () => {
    const { engine } = setup();
    engine.start();
    engine.seek(D0 + 38005);
    engine.setView(D0 + 37700, D0 + 38300); // ten minutes
    const t = performance.now();
    press('-'); // to twenty
    engine.frame(t + 50); // part way
    press('-'); // to forty, not twice wherever the glide had got to
    engine.frame(t + 10_000);
    expect(engine.view.t1 - engine.view.t0).toBe(2400);
    engine.stop();
  });

  describe('Space across Review', () => {
    const engines: ReviewEngine[] = [];
    const nodes: HTMLElement[] = [];
    const opened = () => {
      const result = setup();
      result.engine.start();
      engines.push(result.engine);
      return result;
    };
    const button = (kind: string) => {
      const node = document.body.appendChild(document.createElement('button'));
      if (kind === 'tab') node.setAttribute('role', 'tab');
      if (kind === 'Summary block') node.dataset.summary = '';
      if (kind === 'transcript line') node.dataset.reviewKeys = '';
      nodes.push(node);
      node.focus();
      return node;
    };
    afterEach(() => {
      for (const engine of engines.splice(0)) engine.stop();
      for (const node of nodes.splice(0)) node.remove();
      stopClip();
    });

    it.each(['tab', 'Summary block', 'play button', 'transcript line'])(
      'plays and pauses with focus on a %s, consuming Space without activating it',
      kind => {
        const { engine } = opened();
        const target = button(kind);
        expect(press(' ', target)).toBe(false);
        expect(engine.playing).toBe(true);
        expect(press(' ', target)).toBe(false);
        expect(engine.playing).toBe(false);
        expect(press('Enter', target)).toBe(true);
      },
    );

    it.each(['input', 'textarea', 'select', 'contenteditable'])('leaves Space in an Ask/typing %s untouched', kind => {
      const { engine } = opened();
      const target = document.body.appendChild(document.createElement(kind === 'contenteditable' ? 'div' : kind));
      nodes.push(target);
      if (kind === 'contenteditable') target.setAttribute('contenteditable', 'true');
      const child = kind === 'contenteditable' ? target.appendChild(document.createElement('span')) : target;
      expect(press(' ', child)).toBe(true);
      expect(engine.playing).toBe(false);
    });

    it('does not play behind an open modal, even when focus remains outside it', () => {
      const { engine } = opened();
      const dialog = document.body.appendChild(document.createElement('section'));
      dialog.setAttribute('role', 'dialog');
      dialog.setAttribute('aria-modal', 'true');
      nodes.push(dialog);
      expect(press(' ', button('tab'))).toBe(true);
      expect(engine.playing).toBe(false);
      expect(press(' ', dialog)).toBe(true);
      expect(engine.playing).toBe(false);
      dialog.hidden = true;
      expect(press(' ')).toBe(false);
      expect(engine.playing).toBe(true);
    });

    it('consumes held Space without toggling again, and ignores modified Space', () => {
      const { engine } = opened();
      const target = button('tab');
      press(' ', target);
      expect(press(' ', target, { repeat: true })).toBe(false);
      expect(engine.playing).toBe(true);
      expect(press(' ', target, { ctrlKey: true })).toBe(true);
      expect(engine.playing).toBe(true);
    });

    it.each<[string, Partial<DayData>, number]>([
      ['video', {}, D0 + 36000],
      ['room audio', { segments: [], spans: [{ start: D0 + 35000, end: D0 + 35100 }] }, D0 + 35000],
      ['transcript audio', { segments: [] }, D0 + 38000],
    ])('starts at the first %s when no time is picked', (_kind, more, first) => {
      const { engine } = setup(undefined, { more });
      engine.start();
      engines.push(engine);
      engine.setDayData(DAY, engine.ui.getState().gen, data({ entries: LINES, ...more }));
      engine.t = null;
      expect(press(' ')).toBe(false);
      expect(engine.t).toBe(first);
      expect(engine.playing).toBe(true);
    });

    it('starts at the beginning when Space arrives before the day loads', () => {
      const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
      engine.start();
      engines.push(engine);
      engine.pickDay(DAY);
      expect(press(' ')).toBe(false);
      engine.setDayData(DAY, engine.ui.getState().gen, data());
      expect(engine.t).toBe(D0 + 36000);
      expect(engine.playing).toBe(true);
    });

    it('cancels a queued start with a second Space, and keeps an empty day paused', () => {
      const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
      engine.start();
      engines.push(engine);
      engine.pickDay(DAY);
      press(' ');
      press(' ');
      engine.setDayData(DAY, engine.ui.getState().gen, data());
      expect(engine.playing).toBe(false);
      engine.stop();
      history.replaceState(null, '', '/review');
      const empty = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
      empty.start();
      engines.push(empty);
      empty.pickDay(DAY);
      press(' ');
      empty.setDayData(DAY, empty.ui.getState().gen, data({ segments: [] }));
      expect(empty.t).toBeNull();
      expect(empty.playing).toBe(false);
    });

    it.each(['body', 'tab', 'Summary block', 'play button'])(
      'toggles only the active shared clip with focus on %s',
      kind => {
        vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) {
          Object.defineProperty(this, 'paused', { value: false, configurable: true });
          this.dispatchEvent(new Event('play'));
          return Promise.resolve();
        });
        vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) {
          Object.defineProperty(this, 'paused', { value: true, configurable: true });
          this.dispatchEvent(new Event('pause'));
        });
        vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
        playClip('/clip.wav', 'clip');
        const { engine } = opened();
        const target = kind === 'body' ? document.body : button(kind);
        const clip = document.querySelector('audio');
        expect(press(' ', target)).toBe(false);
        expect(clip?.paused).toBe(true);
        expect(usePlayer.getState().key).toBe('clip');
        expect(engine.playing).toBe(false);
        expect(press(' ', target)).toBe(false);
        expect(clip?.paused).toBe(false);
        expect(engine.playing).toBe(false);
      },
    );
  });

  it('stops listening when Review closes', () => {
    const { engine } = setup();
    engine.start();
    engine.stop();
    const t = engine.t;
    press('ArrowRight');
    expect(engine.t).toBe(t);
  });
});

describe('event mark keys', () => {
  let engine: ReviewEngine;
  const key = (value: string, target: EventTarget = document.body, more: KeyboardEventInit = {}) =>
    target.dispatchEvent(new KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true, ...more }));
  beforeEach(() => {
    engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.start();
    engine.pickDay(DAY);
    engine.setDayData(DAY, engine.ui.getState().gen, data());
  });
  afterEach(() => engine.stop());

  it('marks in and out at the playhead, and normalizes reversed ends', () => {
    engine.jump(D0 + 36100);
    expect(key('i')).toBe(false);
    expect(engine.ui.getState().eventMarks).toEqual({ cam: 'cam0', in: D0 + 36100, out: null });
    expect(engine.ui.getState().eventSpan).toBeNull();
    engine.jump(D0 + 36050);
    key('o');
    expect(engine.ui.getState().eventSpan).toEqual({ cam: 'cam0', start: D0 + 36050, end: D0 + 36100 });
    key('Escape');
    expect(engine.ui.getState().eventMarks).toBeNull();
    expect(engine.ui.getState().eventSpan).toBeNull();
  });

  it('accepts out before in and moves a repeated mark before completing the range', () => {
    engine.jump(D0 + 36100);
    key('o');
    engine.jump(D0 + 36150);
    key('o');
    engine.jump(D0 + 36020);
    key('i');
    expect(engine.ui.getState().eventSpan).toEqual({ cam: 'cam0', start: D0 + 36020, end: D0 + 36150 });
    engine.pickDay('2026-10-28');
    expect(engine.ui.getState().eventMarks).toBeNull();
    expect(engine.ui.getState().eventSpan).toBeNull();
  });

  it('keeps a partial mark unchanged while a modal owns the keyboard', () => {
    engine.jump(D0 + 36100);
    key('i');
    const dialog = document.body.appendChild(document.createElement('section'));
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    try {
      engine.jump(D0 + 36120);
      expect(key('o', dialog)).toBe(true);
      expect(engine.ui.getState().eventMarks).toEqual({ cam: 'cam0', in: D0 + 36100, out: null });
      expect(engine.ui.getState().eventSpan).toBeNull();
    } finally {
      dialog.remove();
    }
  });

  it('leaves i and o in Ask and modified shortcuts alone', () => {
    const input = document.body.appendChild(document.createElement('textarea'));
    try {
      engine.jump(D0 + 36100);
      expect(key('i', input)).toBe(true);
      expect(key('o', input)).toBe(true);
      expect(key('i', document.body, { shiftKey: true })).toBe(true);
      expect(key('o', document.body, { ctrlKey: true })).toBe(true);
      expect(engine.ui.getState().eventMarks).toBeNull();
    } finally {
      input.remove();
    }
  });
});

describe('the link', () => {
  it('keeps the day and the playhead in the hash', () => {
    const clock = vi.spyOn(performance, 'now').mockReturnValue(0);
    const { engine } = setup();
    engine.seek(D0 + 36100.25);
    // Hash writes read the monotonic clock too, independently of the frame argument.
    clock.mockReturnValue(5000);
    engine.frame(performance.now());
    expect(location.hash).toBe(`#day=${DAY}&t=${D0 + 36100}`);
  });

  it('writes a moving playhead at most every 1.5 s, the latest rounded to a second, replacing the entry', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    hashTiming.everyMs = 1500;
    try {
      const { engine } = setup();
      engine.start();
      const steps = history.length;
      const opened = location.hash; // where the start landed, written at once
      engine.seek(D0 + 36100.4);
      engine.seek(D0 + 36200.6);
      expect(location.hash).toBe(opened);
      vi.advanceTimersByTime(1500);
      expect(location.hash).toBe(`#day=${DAY}&t=${D0 + 36201}`);
      expect(history.length).toBe(steps);
      engine.stop();
    } finally {
      hashTiming.everyMs = 0;
      vi.useRealTimers();
    }
  });

  it('pushes a camera or day change, so Back returns to it without a write of its own', () => {
    const { engine } = setup();
    const two = new ReviewEngine({ cams: ['cam0', 'cam1'], fetchNotes: async () => ({ cams: {} }) });
    engine.stop();
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36000}`);
    two.start();
    const steps = history.length;
    two.pickCamera('cam1');
    expect(location.hash).toBe(`#cam=cam1&day=${DAY}&t=${D0 + 36000}`);
    expect(history.length).toBe(steps + 1);
    two.pickDay('2026-10-26');
    expect(readHash().day).toBe('2026-10-26');
    expect(history.length).toBe(steps + 2);
    history.replaceState(null, '', `/review#day=${DAY}&t=${D0 + 36000}`); // what Back shows
    window.dispatchEvent(new PopStateEvent('popstate'));
    expect(two.ui.getState().camera).toBeUndefined();
    expect(two.ui.getState().day).toBe(DAY);
    expect(history.length).toBe(steps + 2); // following Back pushed nothing
    two.stop();
  });

  it('leaves the Settings entry alone while Review plays behind the drawer', () => {
    const { engine } = setup();
    const state = { settingsReturnTo: `/review#day=${DAY}`, settingsFromApp: true };
    history.pushState(state, '', '/settings');
    const steps = history.length;
    engine.pickDay('2026-10-26');
    engine.seek(D0 - 3600);
    expect(location.pathname + location.hash).toBe('/settings');
    expect(history.state).toEqual(state);
    expect(history.length).toBe(steps);
  });

  it('follows a hash that moves to another moment of the day', () => {
    const { engine } = setup();
    engine.start();
    history.replaceState(null, '', `#day=${DAY}&t=${D0 + 36200}`);
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(engine.t).toBe(D0 + 36200);
    engine.stop();
  });
});

describe('the visible range, for the Summary column', () => {
  it('is null until the day loads, then publishes once the view holds still', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.setView(D0, D0 + 600); // before the day's data: nothing to publish yet
    vi.advanceTimersByTime(1000);
    expect(engine.ui.getState().range).toBeNull();
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    vi.advanceTimersByTime(1000);
    const first = engine.ui.getState().range;
    expect(first).toEqual(engine.view);

    engine.setView(D0 + 3600, D0 + 7200, true);
    vi.advanceTimersByTime(RANGE_SETTLE_MS - 50);
    engine.setView(D0 + 3700, D0 + 7300, true); // still moving: waits on
    vi.advanceTimersByTime(RANGE_SETTLE_MS - 50);
    expect(engine.ui.getState().range).toBe(first);
    vi.advanceTimersByTime(100);
    expect(engine.ui.getState().range).toEqual({ t0: D0 + 3700, t1: D0 + 7300 });
  });

  it('publishes the new viewport on another day without dropping the continuous range', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    vi.advanceTimersByTime(1000);
    expect(engine.ui.getState().range).not.toBeNull();
    engine.setView(D0 + 100, D0 + 700, true); // its timer is still running when the day changes
    engine.pickDay('2026-01-03');
    vi.advanceTimersByTime(1000);
    const start = new Date('2026-01-03T00:00:00').getTime() / 1000;
    expect(engine.ui.getState().range).toEqual({ t0: start, t1: start + 86400 });
  });

  it('glideView changes the span by the same ratio each step and moves the centre evenly', () => {
    const a = { t0: 0, t1: 86400 };
    const b = { t0: 43200 - 300, t1: 43200 + 300 };
    expect(glideView(a, b, 0)).toEqual([0, 86400]);
    const [m0, m1] = glideView(a, b, 0.5);
    expect(m1 - m0).toBeCloseTo(Math.sqrt(86400 * 600)); // halfway in ratio: about 2 h, not about 12 h
    expect((m0 + m1) / 2).toBeCloseTo(43200);
    const [e0, e1] = glideView(a, b, 1);
    expect(e0).toBeCloseTo(b.t0);
    expect(e1).toBeCloseTo(b.t1);
  });

  it('zooms to a range with a glide, and centres one shorter than the closest zoom', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    const t = performance.now();
    engine.zoomTo(D0 + 3600, D0 + 5400);
    engine.frame(t + 60_000); // well past the glide
    expect(engine.view).toEqual({ t0: D0 + 3600, t1: D0 + 5400 });
    engine.zoomTo(D0 + 9010, D0 + 9000); // 10 s, given backwards
    engine.frame(t + 120_000);
    expect(engine.view).toEqual({ t0: D0 + 9005 - MIN_SPAN / 2, t1: D0 + 9005 + MIN_SPAN / 2 });
    vi.advanceTimersByTime(RANGE_SETTLE_MS);
    expect(engine.ui.getState().range).toEqual(engine.view);
  });

  it('glides Day, 1 h, and 10 min to their span over a few frames, landing exactly', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    engine.seek(D0 + 7200);
    engine.setView(D0, D0 + 86400, true);
    const t = performance.now();
    engine.zoomSpan(600);
    engine.frame(t + 100); // part way: narrower than the day, wider than 10 minutes
    const mid = engine.view.t1 - engine.view.t0;
    expect(mid).toBeLessThan(86400);
    expect(mid).toBeGreaterThan(600);
    engine.frame(t + 10_000);
    expect(engine.view).toEqual({ t0: D0 + 7200 - 300, t1: D0 + 7200 + 300 });
  });

  it("lands a seek made during a glide in view, rather than pulling the view back to the glide's old end", () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    engine.seek(D0 + 7200);
    engine.setView(D0, D0 + 86400, true);
    const t = performance.now();
    engine.zoomSpan(600);
    engine.frame(t + 100);
    engine.seek(D0 + 30000); // well outside where the glide was headed
    engine.frame(t + 10_000);
    expect(engine.view).toEqual({ t0: D0 + 30000 - 300, t1: D0 + 30000 + 300 });
  });

  it('jumps straight to the span when the viewer asks for reduced motion', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    vi.stubGlobal('matchMedia', (q: string) => ({ matches: q.includes('reduce') }) as MediaQueryList);
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    engine.seek(D0 + 7200);
    engine.zoomSpan(3600);
    expect(engine.view).toEqual({ t0: D0 + 5400, t1: D0 + 9000 });
    vi.unstubAllGlobals();
  });

  it("publishes nothing for a day's data that came too late, and doesn't wake subscribers for no change", () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    const stale = engine.ui.getState().gen;
    engine.pickDay(DAY); // picked again: the first load is stale
    engine.setDayData(DAY, stale, data());
    vi.advanceTimersByTime(1000);
    expect(engine.ui.getState().range).toBeNull();

    engine.setDayData(DAY, engine.ui.getState().gen, data());
    vi.advanceTimersByTime(1000);
    const r = engine.ui.getState().range;
    engine.setView(engine.view.t0, engine.view.t1); // the same view again
    vi.advanceTimersByTime(1000);
    expect(engine.ui.getState().range).toBe(r);
  });

  it('publishes a move at once when the tab hides, rather than dropping it', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
    engine.start();
    engine.pickDay(DAY);
    engine.setDayData(DAY, engine.ui.getState().gen, data());
    vi.advanceTimersByTime(1000);
    engine.setView(D0 + 3600, D0 + 7200, true);
    engine.stop(); // within the settle time
    expect(engine.ui.getState().range).toEqual({ t0: D0 + 3600, t1: D0 + 7200 });
    vi.unstubAllGlobals();
  });
});

describe("the day's lines read again", () => {
  it('takes the open day’s renamed lines, and nothing from another day or an earlier opening', () => {
    const { engine } = setup();
    const gen = engine.ui.getState().gen;
    const renamed = engine.entries.map(e => ({ ...e, speaker: 'Kara' }));
    engine.relabel('2026-01-03', gen, renamed); // another day
    engine.relabel(DAY, gen - 1, renamed); // an earlier opening
    expect(engine.entries.every(e => e.speaker !== 'Kara')).toBe(true);
    const t = engine.t;
    const redraw = vi.fn();
    engine.attachTimeline({ frame() {}, redraw });
    engine.relabel(DAY, gen, renamed);
    expect(engine.entries).toEqual(renamed);
    expect(redraw).toHaveBeenCalled(); // the timeline's speech lane shows the new names
    expect(engine.t).toBe(t); // nothing seeks
  });

  it('waits for the day to load before taking its lines read again', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.relabel(DAY, engine.ui.getState().gen, [LINES[0] as TranscriptEntry]);
    expect(engine.entries).toEqual([]); // setDayData opens the day; a re-read only renames what's there
  });
});

describe('roomWindow', () => {
  it.each([
    ['the start of a segment that starts mid-window', 1000, [1000, 1080]],
    ['a whole window inside the segment', 1100, [1080, 1200]],
    ['the end of a segment that ends mid-window', 1250, [1200, 1270]],
  ])('cuts %s to the segment', (_, ts, want) => {
    const w = roomWindow({ start: 1000, end: 1270 }, ts, 120);
    expect([w.start, w.end]).toEqual(want);
  });
});

it('seeks an activity source camera with fresh intent after a manual camera choice', () => {
  const engine = new ReviewEngine({ cams: ['cam0', 'cam1'], fetchNotes: async () => ({ cams: {} }) });
  engine.pickDay(DAY);
  engine.setDayData(DAY, engine.ui.getState().gen, { entries: [], segments: [], spans: [] });
  engine.setDisplayedCamera('cam1');
  const seek = vi.spyOn(engine, 'seek');
  engine.seekActivity(D0 + 100, 'cam0');
  const first = engine.ui.getState().cameraIntent;
  expect(engine.ui.getState().camera).toBe('cam0');
  engine.setDisplayedCamera('cam1');
  engine.seekActivity(D0 + 200, 'cam0');
  expect(engine.ui.getState().cameraIntent).toBe((first ?? 0) + 1);
  expect(seek).toHaveBeenLastCalledWith(D0 + 200, { play: false });
  engine.seekActivity(D0 + 300, 'not-configured');
  expect(engine.ui.getState().camera).toBe('cam0');
  expect(engine.ui.getState().cameraIntent).toBe((first ?? 0) + 1);
  expect(seek).toHaveBeenLastCalledWith(D0 + 300, { play: false });
  expect(engine.ui.getState().playing).toBe(false);
});

it('plays a clicked activity from its start every time: the same one again, after a pause, and another', () => {
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.pickDay(DAY);
  engine.setDayData(DAY, engine.ui.getState().gen, { entries: [], segments: [], spans: [] });
  engine.seekActivity(D0 + 100, 'cam0', { play: true });
  expect([engine.t, engine.ui.getState().playing]).toEqual([D0 + 100, true]);
  engine.seekActivity(D0 + 100, 'cam0', { play: true }); // the same row, already there and playing
  expect([engine.t, engine.ui.getState().playing]).toEqual([D0 + 100, true]);
  engine.pause();
  engine.seekActivity(D0 + 100, 'cam0', { play: true }); // the same row after a pause
  expect([engine.t, engine.ui.getState().playing]).toEqual([D0 + 100, true]);
  engine.seekActivity(D0 + 400, 'cam0', { play: true }); // another row
  expect([engine.t, engine.ui.getState().playing]).toEqual([D0 + 400, true]);
});

describe('activity seeks across recording days', () => {
  const clip = (start: number): DayData =>
    data({ entries: [], segments: [{ cam: 'cam0', start, end: start + 15, url: url('cam0', start, start + 15) }] });

  it.each([
    { start: D0 - 5, day: '2026-10-26' },
    { start: D0 + 86400 + 5, day: '2026-10-28' },
  ])('opens $day and retains the exact source start through its asynchronous load', ({ start, day }) => {
    const engine = new ReviewEngine({ cams: ['cam0', 'cam1'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    const oldGen = engine.ui.getState().gen;
    engine.setDayData(DAY, oldGen, data());
    engine.setDisplayedCamera('cam1');
    engine.seekActivity(start, 'cam0');
    const { gen, cameraIntent } = engine.ui.getState();
    expect(engine.ui.getState().day).toBe(day);
    expect(engine.t).toBe(start);
    expect(Number(new URLSearchParams(location.hash.slice(1)).get('t'))).toBe(start);
    engine.setDayData(DAY, oldGen, data());
    engine.setDayData(day, gen - 1, clip(start + 100));
    expect(engine.t).toBe(start);
    engine.setDayData(day, gen, clip(start));
    expect(engine.t).toBe(start);
    expect(engine.ui.getState()).toMatchObject({ day, camera: 'cam0', cameraIntent, playing: false });
  });

  it('ignores a source-day response superseded by a later activity selection', () => {
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.seekActivity(D0 - 5, 'cam0');
    const previous = engine.ui.getState().gen;
    const next = D0 + 86400 + 5;
    engine.seekActivity(next, 'not-configured');
    const current = engine.ui.getState().gen;
    engine.setDayData('2026-10-26', previous, clip(D0 - 5));
    expect(engine.t).toBe(next);
    engine.setDayData('2026-10-28', current, clip(next));
    expect(engine.t).toBe(next);
    expect(engine.ui.getState()).toMatchObject({ day: '2026-10-28', camera: 'cam0', playing: false });
  });

  it.each([
    { start: D0 + 100, day: DAY },
    { start: D0 - 5, day: '2026-10-26' },
  ])('cancels queued playback and start-from-beginning intent while $day is still loading', ({ start, day }) => {
    history.replaceState(null, '', `/review#day=${DAY}&play=1`);
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.start();
    try {
      engine.togglePlay(); // Queue Space's start-from-beginning intent before data arrives.
      engine.seekActivity(start, 'cam0');
      engine.setDayData(day, engine.ui.getState().gen, clip(start - 30));
      expect(engine.t).toBe(start);
      expect(engine.ui.getState().playing).toBe(false);
    } finally {
      engine.stop();
    }
  });

  it('keeps a late source start on a 25-hour local recording day', () => {
    vi.stubEnv('TZ', 'America/New_York');
    try {
      const start = new Date('2026-11-01T23:59:55').getTime() / 1000;
      const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
      engine.pickDay('2026-11-02');
      engine.seekActivity(start, 'cam0');
      engine.setDayData('2026-11-01', engine.ui.getState().gen, clip(start));
      expect(engine.ui.getState().day).toBe('2026-11-01');
      expect(engine.t).toBe(start);
    } finally {
      vi.unstubAllEnvs();
    }
  });
});

describe('authoritative listings', () => {
  it('replaces a changed source, preserving unchanged identities and failures', () => {
    const { engine } = setup();
    const gen = engine.ui.getState().gen;
    const original = engine.segs.cam0?.[0];
    if (!original) throw new Error('missing fixture segment');
    original.broken = 'cut';
    engine.growSegments(DAY, gen, data().segments);
    expect(engine.segs.cam0?.[0]).toBe(original);
    expect(engine.segs.cam0?.[0]?.broken).toBe('cut');
    const replacement = {
      cam: 'cam0',
      start: D0 + 39600,
      end: D0 + 39900,
      url: url('cam0', D0 + 39600, D0 + 39900),
    };
    engine.growSegments(DAY, gen, [replacement]);
    expect(engine.segs.cam0).toHaveLength(1);
    expect(engine.segs.cam0?.[0]).toMatchObject({ ...replacement, bytes: 0 });
    expect(engine.segs.cam0?.[0]?.broken).toBeUndefined();
    engine.growSegments(DAY, gen, []);
    expect(engine.segs.cam0).toBeUndefined();
    expect(engine.ui.getState().avail).toEqual([]);
    engine.stop();
  });
});

it('ignores a seek completion captured before an authoritative source replacement', () => {
  const { engine, v } = setup();
  engine.seek(D0 + 39800, { play: true });
  const oldCallbacks = v[0].callbacks('seeked');
  expect(oldCallbacks.length).toBeGreaterThan(0);
  const replacement = { cam: 'cam0', start: D0 + 39600, end: D0 + 39900, url: 'gateway:1:0:1' };
  engine.growSegments(DAY, engine.ui.getState().gen, [replacement]);
  const plays = v[0].plays;
  for (const callback of oldCallbacks) callback();
  expect(v[0].plays).toBe(plays);
  expect(engine.segs.cam0?.[0]?.url).toBe('gateway:1:0:1');
  v[0].land();
  expect(v[0].plays).toBeGreaterThan(plays);
  engine.stop();
});

describe('camera filters', () => {
  it('keeps cameras in links across day changes and reloads', () => {
    const engine = new ReviewEngine({ cams: ['cam0', 'cam1', 'cam2'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.pickCameras(['cam1', 'cam2']);
    expect(engine.visibleCams).toEqual(['cam1', 'cam2']);
    expect(new URLSearchParams(location.hash.slice(1)).getAll('cameras')).toEqual(['cam1', 'cam2']);
    engine.pickDay('2026-10-28');
    const reopened = new ReviewEngine({ cams: ['cam0', 'cam1', 'cam2'], fetchNotes: async () => ({ cams: {} }) });
    reopened.start();
    expect(reopened.visibleCams).toEqual(['cam1', 'cam2']);
    reopened.stop();
    engine.pickCameras([]);
    expect(engine.visibleCams).toEqual(['cam0', 'cam1', 'cam2']);
    expect(location.hash).not.toContain('cameras=');
  });

  it('filters activity marks and selected navigation by camera after selection', () => {
    const engine = new ReviewEngine({ cams: ['cam0', 'cam1'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.setChips(DAY, engine.ui.getState().gen, [
      {
        id: 'event:activity:crossing',
        kind: 'event',
        label: 'Walked past',
        count: 2,
        total_s: 20,
        spans: [
          [D0 + 100, D0 + 110],
          [D0 + 200, D0 + 210],
        ],
        activities: [
          {
            id: 'a',
            camera: 'cam0',
            start: D0 + 100,
            end: D0 + 110,
            kind: 'crossing',
            count: null,
            name: null,
            person: null,
            promoted: false,
          },
          {
            id: 'b',
            camera: 'cam1',
            start: D0 + 200,
            end: D0 + 210,
            kind: 'crossing',
            count: null,
            name: null,
            person: null,
            promoted: false,
          },
        ],
      },
    ]);
    engine.toggleChip('event:activity:crossing');
    engine.pickCameras(['cam1']);
    expect(engine.marks[0]?.spans).toEqual([[D0 + 200, D0 + 210]]);
    expect(engine.filteredChips[0]?.activities?.map(a => a.id)).toEqual(['b']);
    engine.stepChip(1);
    expect(engine.t).toBe(D0 + 200);
  });
});

it('lets go of the room audio blob an element held when the element goes', () => {
  const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  const audio = new FakeAudio();
  engine.setAudio(audio);
  audio.src = 'blob:room';
  engine.setAudio(null);
  expect(revoke).toHaveBeenCalledWith('blob:room');
  revoke.mockRestore();
});

it('retains room mute when the audio element is replaced', () => {
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.setMuted(true);
  const audio = document.createElement('audio');
  engine.setAudio(audio);
  expect(audio.muted).toBe(true);
  engine.setAudio(null);
  const replacement = document.createElement('audio');
  engine.setAudio(replacement);
  expect(replacement.muted).toBe(true);
  engine.setMuted(false);
  expect(replacement.muted).toBe(false);
});
