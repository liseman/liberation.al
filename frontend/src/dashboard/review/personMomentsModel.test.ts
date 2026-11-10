import { describe, expect, it } from 'vitest';
import type { TranscriptEntry } from '../live/types';
import type { Block, DetectedActivity } from './api';
import { type Moment, mentioning, mentions, momentsOf, playbackOf, visitsOf } from './personMomentsModel';
import type { Chip } from './types';

const T = 1_800_000_000;
const line = (ts: number, speaker: string, text = 'hi'): TranscriptEntry => ({
  ts_start: T + ts,
  ts_end: T + ts + 2,
  speaker,
  text,
});
const act = (start: number, person: string | null, name: string): DetectedActivity =>
  ({
    id: `a${start}`,
    camera: 'cam0',
    start: T + start,
    end: T + start + 5,
    count: null,
    name,
    kind: null,
    person: null,
    person_name: person,
    promoted: false,
  }) as DetectedActivity;
const chip = (label: string, cams: Record<string, [number, number][]>): Chip => ({
  id: `person:${label.toLowerCase()}`,
  kind: 'person',
  label,
  spans: [],
  total_s: 0,
  count: 0,
  cams: Object.fromEntries(Object.entries(cams).map(([c, s]) => [c, s.map(([a, b]) => [T + a, T + b])])),
});

describe('one person in Review (#2057)', () => {
  it('takes only their lines, the views their chip says saw them, and the actions that name them', () => {
    const moments = momentsOf(
      'alice',
      [line(0, 'Alice'), line(5, 'Dana'), line(400, 'Alice')],
      e => String(e.speaker),
      [chip('Alice', { cam0: [[10, 30]] }), chip('Dana', { cam0: [[0, 500]] })],
      [act(20, 'Alice', 'sit down'), act(30, 'Dana', 'stand up'), act(40, null, 'walk by')],
    );
    expect(moments.map(m => [m.kind, m.start - T, m.cam])).toEqual([
      ['line', 0, null],
      ['seen', 10, 'cam0'],
      ['action', 20, 'cam0'],
      ['line', 400, null],
    ]);
  });

  it('joins moments a minute apart or closer into one visit, and says what happened in it', () => {
    const visits = visitsOf(
      momentsOf(
        'Alice',
        [line(0, 'Alice'), line(80, 'Alice'), line(300, 'Alice')],
        e => String(e.speaker),
        [chip('Alice', { 'living-room': [[10, 70]], cam0: [[60, 65]] })],
        [act(20, 'Alice', 'sit down')],
      ),
    );
    expect(visits.map(v => [v.start - T, v.end - T, v.lines, v.seen_s, v.cams, v.actions])).toEqual([
      [0, 82, 2, 65, ['living-room', 'cam0'], ['sit down']], // 82 s: the second line's end; living-room saw them longest
      [300, 302, 1, 0, [], []], // more than a minute after: a visit of its own
    ]);
  });

  it.each([
    ['Alice was in the kitchen with Dana.', [[0, 5]]],
    [
      "Alice's tea, then alice left.",
      [
        [0, 5],
        [18, 23],
      ],
    ],
    ['Alicia and McAlice are other people.', []],
  ])('finds the name as a whole word in %j', (text, found) => {
    expect(mentions(text, 'Alice')).toEqual(found);
  });

  it('keeps the summary blocks that name them or are tagged with their chip', () => {
    const blocks: Block[] = [
      { start: 1, end: 2, text: 'Alice made tea.', tags: [] },
      { start: 3, end: 4, text: 'Someone talked.', tags: ['person:alice'] },
      { start: 5, end: 6, text: 'Dana worked.', tags: ['person:dana'] },
    ];
    expect(mentioning(blocks, 'Alice', 'person:alice').map(b => b.start)).toEqual([1, 3]);
  });
});

describe('person playback plans (#2057)', () => {
  const moment = (kind: Moment['kind'], start: number, end: number, cam: string | null = null): Moment => ({
    kind,
    start: T + start,
    end: T + end,
    cam,
    label: '',
  });

  it("plays only the selected speaker's exact line spans, back to back without unrelated room gaps", () => {
    const moments = momentsOf(
      'Alice',
      [line(40, 'Alice'), line(10, 'Dana'), line(0, 'Alice')],
      e => String(e.speaker),
      [chip('Alice', { 'front-door': [[5, 30]] })],
      [act(20, 'Alice', 'sit down')],
    );
    expect(playbackOf(moments, 'audio', T)).toEqual([
      { start: T, end: T + 2, cam: null },
      { start: T + 40, end: T + 42, cam: null },
    ]);
  });

  it('excludes lines whose ends are missing, zero-length, reversed, negative, or nonfinite', () => {
    const entries: TranscriptEntry[] = [
      { ts_start: T, speaker: 'Alice' },
      { ts_start: T + 1, ts_end: T + 1, speaker: 'Alice' },
      { ts_start: T + 3, ts_end: T + 2, speaker: 'Alice' },
      { ts_start: -2, ts_end: -1, speaker: 'Alice' },
      { ts_start: Number.NaN, ts_end: T + 5, speaker: 'Alice' },
      { ts_start: T + 6, ts_end: Number.POSITIVE_INFINITY, speaker: 'Alice' },
      { ts_start: Number.NEGATIVE_INFINITY, ts_end: T + 7, speaker: 'Alice' },
      line(10, 'Alice'),
    ];
    const moments = momentsOf('Alice', entries, e => String(e.speaker), [], []);
    expect(playbackOf(moments, 'audio', T)).toEqual([{ start: T + 10, end: T + 12, cam: null }]);
  });

  it("uses each video moment's own camera, clamps lead-in to the day, and gives point actions a tail", () => {
    const moments = [
      moment('seen', 50, 60, 'kitchen'),
      moment('line', 0, 2),
      moment('action', 20, 20, 'back-door'),
      moment('seen', 1, 4, 'front-door'),
    ];
    expect(playbackOf(moments, 'video', T)).toEqual([
      { start: T, end: T + 4, cam: 'front-door' },
      { start: T + 18, end: T + 22, cam: 'back-door' },
      { start: T + 48, end: T + 60, cam: 'kitchen' },
    ]);
    expect(moments[0]?.start).toBe(T + 50); // Sorting and merging never mutate the source moments.
  });

  it('merges overlapping consecutive windows on one camera and drops duplicate replay', () => {
    expect(
      playbackOf(
        [
          moment('seen', 30, 35, 'front-door'),
          moment('seen', 10, 20, 'front-door'),
          moment('action', 20, 25, 'front-door'),
          moment('seen', 10, 20, 'front-door'),
        ],
        'video',
        T,
      ),
    ).toEqual([
      { start: T + 8, end: T + 25, cam: 'front-door' },
      { start: T + 28, end: T + 35, cam: 'front-door' },
    ]);
  });

  it('keeps overlapping views from different cameras, in deterministic time and camera order', () => {
    const moments = [moment('seen', 10, 20, 'kitchen'), moment('seen', 10, 20, 'front-door')];
    const expected = [
      { start: T + 8, end: T + 20, cam: 'front-door' },
      { start: T + 8, end: T + 20, cam: 'kitchen' },
    ];
    expect(playbackOf(moments, 'video', T)).toEqual(expected);
    expect(playbackOf([...moments].reverse(), 'video', T)).toEqual(expected);
  });

  it('excludes invalid video spans and camera-less moments instead of choosing another camera', () => {
    expect(
      playbackOf(
        [
          moment('seen', 0, 3),
          moment('action', 1, 2, ''),
          moment('action', 2, 4, '  '),
          moment('seen', 4, 4, 'front-door'),
          moment('seen', 10, 9, 'front-door'),
          moment('action', 12, Number.NaN, 'front-door'),
          moment('seen', 13, Number.POSITIVE_INFINITY, 'front-door'),
          { kind: 'seen', start: -3, end: -1, cam: 'front-door', label: '' },
          moment('seen', -10, -5, 'front-door'),
          moment('line', 20, 22),
        ],
        'video',
        T,
      ),
    ).toEqual([]);
    expect(playbackOf([moment('seen', 0, 2, 'front-door')], 'video', Number.NaN)).toEqual([]);
  });
});
