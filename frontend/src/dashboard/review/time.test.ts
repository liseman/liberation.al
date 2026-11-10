import { describe, expect, it } from 'vitest';
import type { TranscriptEntry } from '../live/types';
import {
  clampView,
  DAY_S,
  dayBounds,
  lanes,
  latestVideo,
  lineAt,
  midnights,
  nextMedia,
  segAt,
  tickSteps,
  ticks,
} from './time';
import type { Segment } from './types';

it('gives prototype-named cameras their own timeline lanes', () => {
  const result = lanes(['__proto__', 'constructor', 'toString']);
  expect(Object.keys(result.cams)).toEqual(['__proto__', 'constructor', 'toString']);
  expect(Object.getOwnPropertyDescriptor(result.cams, '__proto__')?.value).toEqual([62, 5]);
});

const seg = (start: number, end: number, more: Partial<Segment> = {}): Segment => ({
  cam: 'cam0',
  start,
  end,
  url: '',
  bytes: 0,
  ...more,
});
const line = (ts_start: number, ts_end = ts_start + 2): TranscriptEntry => ({ ts_start, ts_end, text: 'hi' });

describe('segAt', () => {
  const list = [seg(0, 300), seg(300, 600), seg(700, 1000)];
  it.each([
    [0, 0],
    [299.9, 0],
    [300, 300], // a boundary belongs to the segment that starts there
    [650, null], // the gap between segments
    [1000, null],
  ])('at %s is the segment from %s', (ts, start) => {
    expect(segAt(list, ts)?.start ?? null).toBe(start);
  });
});

describe('lineAt', () => {
  const lines = [line(10), line(20), line(30)];
  it('is the last line started by ts, with 0.3 s of slack, or -1 before the first', () => {
    expect(lineAt(lines, 5)).toBe(-1);
    expect(lineAt(lines, 9.8)).toBe(0);
    expect(lineAt(lines, 25)).toBe(1);
    expect(lineAt(lines, 99)).toBe(2);
  });
});

describe('clampView', () => {
  it('allows a continuous viewport to pan in either direction beyond its opening date', () => {
    expect(clampView(-100000, -96400)).toEqual({ t0: -100000, t1: -96400 });
    expect(clampView(900000, 903600)).toEqual({ t0: 900000, t1: 903600 });
  });
  const day = 1_000_000;
  it('widens a window under a minute around its centre', () => {
    expect(clampView(day + 100, day + 110, day)).toEqual({ t0: day + 75, t1: day + 135 });
  });
  it('narrows a window wider than the day plus two hours', () => {
    const v = clampView(day - 50_000, day + 150_000, day);
    expect(v.t1 - v.t0).toBe(DAY_S + 7200);
  });
  it('keeps the window within an hour of either end of the day', () => {
    expect(clampView(day - 9000, day - 5400, day)).toEqual({ t0: day - 3600, t1: day });
    expect(clampView(day + DAY_S + 3000, day + DAY_S + 6600, day)).toEqual({ t0: day + DAY_S, t1: day + DAY_S + 3600 });
  });
});

it.each([
  ['2026-03-08', 23, '2026-03-09'],
  ['2026-11-01', 25, '2026-11-02'],
])('keeps midnight markers and day chunks aligned across %s', (day, hours, next) => {
  const old = process.env.TZ;
  process.env.TZ = 'America/Los_Angeles';
  try {
    const bounds = dayBounds(day);
    const end = dayBounds(next).t0;
    expect(bounds.t1 - bounds.t0).toBe(hours * 3600);
    expect(bounds.t1).toBe(end);
    expect(midnights(bounds.t0, end + 3600)).toEqual([bounds.t0, end]);
    expect(ticks(bounds.t0, end + 3600, DAY_S)).toEqual([bounds.t0, end]);
  } finally {
    if (old === undefined) delete process.env.TZ;
    else process.env.TZ = old;
  }
});

describe('latestVideo', () => {
  it('opens a minute before the end of the newest finished video', () => {
    expect(latestVideo({ cam0: [seg(0, 300), seg(300, 600)] }, 10_000)).toBe(540);
  });
  it('opens in the stretch still being recorded: the gateway plays it to its end', () => {
    expect(latestVideo({ cam0: [seg(0, 300), seg(300, 590)] }, 600)).toBe(530);
  });
  it('skips segments that failed to load', () => {
    expect(latestVideo({ cam0: [seg(0, 300), seg(300, 600, { broken: 'cut' })] }, 10_000)).toBe(240);
  });
  it('prefers a moment more cameras can play a full minute from', () => {
    // cam1's newest minute starts at 840, but only cam1 has video then; at 540 both do.
    const segs = { cam0: [seg(0, 600)], cam1: [seg(0, 600, { cam: 'cam1' }), seg(620, 900, { cam: 'cam1' })] };
    expect(latestVideo(segs, 10_000)).toBe(540);
  });
  it('is null without a full minute of video', () => {
    expect(latestVideo({ cam0: [seg(0, 50)] }, 10_000)).toBeNull();
  });
});

describe('nextMedia', () => {
  it('is the earlier of the next segment start and the next line', () => {
    expect(nextMedia([seg(500, 800)], [line(10), line(300)], 20)).toBe(300);
    expect(nextMedia([seg(200, 800)], [line(10), line(300)], 20)).toBe(200);
    expect(nextMedia([seg(0, 100)], [line(10)], 20)).toBeNull();
  });
});

describe('ticks', () => {
  it('labels at least 70 px apart and puts minor ticks between them', () => {
    expect(tickSteps(3600, 1000)).toEqual({ major: 300, minor: 60 });
    expect(tickSteps(DAY_S, 1000)).toEqual({ major: 7200, minor: 3600 });
  });
  it('lands on the local clock', () => {
    const t0 = new Date(2026, 9, 27, 10, 7).getTime() / 1000;
    const at = ticks(t0, t0 + 3600, 1800).map(s => new Date(s * 1000).toTimeString().slice(0, 5));
    expect(at).toEqual(['10:30', '11:00']);
  });
});
