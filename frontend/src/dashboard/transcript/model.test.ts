import { describe, expect, it } from 'vitest';
import type { Speakers, TranscriptEntry } from '../live/types';
import { initials, lineHits, lineSearch, turnKey, turnsOf } from './model';

const speakers: Speakers = {
  list: [],
  byId: { '1': { id: 1, label: 'Dana' }, '2': { id: 2, label: 'Speaker 2' } },
  merged: { '7': 1 },
};
const line = (ts: number, over: Partial<TranscriptEntry> = {}): TranscriptEntry => ({
  ts_start: ts,
  ts_end: ts + 2,
  text: `line ${ts}`,
  speaker_id: 1,
  ...over,
});
const shape = (list: TranscriptEntry[]) => turnsOf(list, speakers).map(t => [t.id, t.lines.map(l => l.e.ts_start)]);

describe('turnsOf', () => {
  it('joins one voice under two minutes apart, and splits on a gap or a new voice', () => {
    expect(shape([line(0), line(100), line(222 + 100), line(330, { speaker_id: 2 })])).toEqual([
      [1, [0, 100]],
      [1, [322]],
      [2, [330]],
    ]);
  });

  it('treats a merged voice as the one it joined', () => {
    expect(shape([line(0), line(5, { speaker_id: 7 })])).toEqual([[1, [0, 5]]]);
  });

  it("folds the assistant's log lines into one status turn however far apart, and keeps its replies apart", () => {
    expect(
      shape([
        line(0, { assistant: true, text: 'Please say your password' }),
        line(500, { status: true, text: 'thinking' }),
        line(510, { assistant: true, text: 'It is sunny.' }),
        line(520, { speaker_id: null }),
      ]),
    ).toEqual([
      ['status', [0, 500]],
      ['assistant', [510]],
      [null, [520]],
    ]);
  });

  it("splits lines no voice was matched to when the name they show changes, and keeps each line's index", () => {
    const tagged = line(10, { speaker_id: null, speaker: 'Eli' });
    const turns = turnsOf([line(0, { speaker_id: null }), line(5, { speaker_id: null }), tagged], speakers);
    expect(turns.map(t => t.lines.map(l => l.i))).toEqual([[0, 1], [2]]);
  });
});

describe('initials', () => {
  it.each([
    ['Dana Ruiz', 'DR'],
    ['dana', 'D'],
    ['Ana María Gómez', 'AM'],
    ['Speaker 261', '?'],
    ['unknown', '?'],
    ['Multiple speakers', '?'],
    ['', '?'],
  ])('%s is %s', (name, want) => expect(initials(name)).toBe(want));
});

describe('lineHits', () => {
  const asked = line(1, { text: 'Where is the café?', speaker_id: 2 });
  const answered = line(2, { text: 'In the garage, the dog walked off.' });
  const found = (q: string) => [...lineHits(lineSearch([asked, answered], speakers), q).keys()];
  it.each([
    [' CAFE ', ['1']], // case and accents don't matter, as in Review's search
    ['ｃａｆｅ', ['1']], // full-width letters fold too
    ['dana', ['2']], // the name a line shows counts
    ['unmatched', ['1']], // an unmatched voice is searched by what it shows...
    ['speaker 2', []], // ...never by its placeholder (#2580)
    ['walked dog', ['2']], // words in any order, as Review's results list them
    ['garaje', ['2']], // and with a typo
    ['absent', []],
    ['   ', []], // no search matches nothing
  ])('%j finds %j', (q, want) => expect(found(q)).toEqual(want));

  it('says where in the words and the name each match is, accents and all', () => {
    const hits = lineHits(lineSearch([asked, answered], speakers), 'cafe');
    const hit = hits.get('1');
    expect(hit?.text.map(([a, b]) => 'Where is the café?'.slice(a, b))).toEqual(['café']);
    expect(hit?.name).toEqual([]);
    const named = lineHits(lineSearch([asked, answered], speakers), 'dan').get('2');
    expect(named?.name.map(([a, b]) => 'Dana'.slice(a, b))).toEqual(['Dana']);
    expect(named?.text).toEqual([]);
  });
});

describe('turnKey', () => {
  it('keeps two turns that start in the same millisecond on two microphones apart (#247)', () => {
    const [a, b] = turnsOf(
      [line(0, { line: 'mic-a@0.000' }), line(0, { speaker_id: 2, line: 'mic-b@0.000' })],
      speakers,
    );
    expect(turnKey(a)).not.toBe(turnKey(b));
  });
});
