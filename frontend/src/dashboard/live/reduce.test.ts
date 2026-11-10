import { describe, expect, it } from 'vitest';
import { EMPTY_TRANSCRIPT, MAX_LINES, reduce, reduceTranscript, resolveId, settle } from './reduce';
import type { Transcript, TranscriptEntry } from './types';

const line = (ts: number, over: Partial<TranscriptEntry> = {}): TranscriptEntry => ({
  ts_start: ts,
  ts_end: ts + 1,
  text: `line ${ts}`,
  speaker: 'Speaker 1',
  speaker_id: 1,
  ...over,
});
const shown = (t: Transcript) => t.entries.map(e => e.ts_start);

describe('reduceTranscript', () => {
  it('replaces every line on a reset and counts the reset', () => {
    const before = { entries: [line(1), line(2)], resets: 3 };
    const after = reduceTranscript(before, { entries: [line(10)], reset: true });
    expect(shown(after)).toEqual([10]);
    expect(after.resets).toBe(4);
  });

  it('keeps only the newest MAX_LINES on a reset', () => {
    const many = Array.from({ length: MAX_LINES + 5 }, (_, i) => line(i));
    const after = reduceTranscript(undefined, { entries: many, reset: true });
    expect(after.entries).toHaveLength(MAX_LINES);
    expect(after.entries[0]?.ts_start).toBe(5);
  });

  it('adds new lines in time order, marked new, even when they arrive out of order', () => {
    const before = { entries: [line(10), line(30)], resets: 1 };
    const after = reduceTranscript(before, { entries: [line(20, { assistant: true }), line(40)] });
    expect(shown(after)).toEqual([10, 20, 30, 40]);
    expect(after.entries.map(e => !!e._new)).toEqual([false, true, false, true]);
    expect(after.resets).toBe(1);
  });

  it('drops the oldest lines past MAX_LINES when lines arrive', () => {
    const full = { entries: Array.from({ length: MAX_LINES }, (_, i) => line(i)), resets: 0 };
    const after = reduceTranscript(full, { entries: [line(MAX_LINES)] });
    expect(after.entries).toHaveLength(MAX_LINES);
    expect(after.entries[0]?.ts_start).toBe(1);
    expect(after.entries.at(-1)?.ts_start).toBe(MAX_LINES);
  });

  it("swaps relabelled lines in place, keeping their highlight, and leaves the assistant's alone", () => {
    const before = {
      entries: [line(1, { _new: true }), line(2, { assistant: true, text: 'reply' }), line(3)],
      resets: 0,
    };
    const after = reduceTranscript(before, {
      entries: [line(1, { speaker: 'Dana' }), line(2, { text: 'not mine' }), line(99, { speaker: 'Nobody' })],
      replace: true,
    });
    expect(after.entries.map(e => [e.ts_start, e.speaker, e.text, e._new])).toEqual([
      [1, 'Dana', 'line 1', true],
      [2, 'Speaker 1', 'reply', undefined],
      [3, 'Speaker 1', 'line 3', undefined],
    ]);
  });

  it('ignores an event with no lines', () => {
    const before = { entries: [line(1)], resets: 2 };
    expect(reduceTranscript(before, { entries: [] })).toBe(before);
    expect(reduceTranscript(undefined, null)).toEqual(EMPTY_TRANSCRIPT);
  });
});

describe('settle', () => {
  it('clears every highlight', () => {
    const t = { entries: [line(1, { _new: true }), line(2)], resets: 0 };
    expect(settle(t).entries.map(e => e._new)).toEqual([false, undefined]);
  });
});

describe('reduce', () => {
  it('indexes speakers by id and keeps merges', () => {
    const s = reduce('speakers', undefined, {
      speakers: [
        { id: 4, label: 'Dana' },
        { id: 7, label: 'Speaker 7' },
      ],
      merged: { 9: 4 },
      next_id: 10,
    });
    expect(s.byId['4']?.label).toBe('Dana');
    expect(resolveId(s, 9)).toBe(4);
    expect(resolveId(s, 7)).toBe(7);
    expect(resolveId(s, null)).toBeNull();
  });

  it.each([
    ['scene', { not: 'a list' }, []],
    ['ambient', null, {}],
  ] as const)('falls back for an empty or malformed %s', (kind, data, expected) => {
    expect(reduce(kind, undefined, data)).toEqual(expected);
  });
});

describe('two microphones in one millisecond', () => {
  it("swaps in a relabelled line by its key, leaving the other microphone's line", () => {
    const a = { ts_start: 5, text: 'a', speaker: 'Joe', line: 'mic-a@5.000' };
    const b = { ts_start: 5, text: 'b', speaker: 'Joe', line: 'mic-b@5.000' };
    const state = reduce('transcript', undefined, { entries: [a, b], reset: true });
    const next = reduce('transcript', state, { entries: [{ ...b, speaker: 'Carly' }], replace: true });
    expect(next.entries.map(e => [e.text, e.speaker])).toEqual([
      ['a', 'Joe'],
      ['b', 'Carly'],
    ]);
  });
});
