import { describe, expect, it } from 'vitest';
import { citedMoments } from './moments';

const ev = (id: string, ts: number, extra: Record<string, unknown> = {}) => ({
  id,
  ts,
  source: 'camera',
  camera: 'living-room',
  text: `note ${id}`,
  href: `/review#t=${ts}`,
  ...extra,
});

describe('citedMoments', () => {
  it('takes the cited items once each, oldest first, bare or bracketed', () => {
    const evidence = [ev('C6', 30), ev('J3', 20, { source: 'journal', camera: undefined }), ev('C9', 10)];
    const got = citedMoments('C6 (13:10) shows two; J3 notes sleep; [C6] again', evidence);
    expect(got.map(m => [m.code, m.ts, m.source, m.camera])).toEqual([
      ['J3', 20, 'journal', null],
      ['C6', 30, 'camera', 'living-room'],
    ]);
  });

  it('leaves out codes no item has, and items with no time or no link into Review', () => {
    const evidence = [ev('C1', 5), ev('C2', Number.NaN), ev('C3', 7, { href: undefined, link: undefined })];
    expect(citedMoments('C1, C2, C3, and C99', evidence).map(m => m.code)).toEqual(['C1']);
  });

  it('finds an item by its label, a link in `link`, and words in `quote`, on one line', () => {
    const evidence = [
      { label: 'T4', ts: 9, source: 'transcript', link: '/review#t=9', quote: 'Get on\n over   there.' },
    ];
    expect(citedMoments('As T4 says', evidence)).toEqual([
      {
        code: 'T4',
        ts: 9,
        href: '/review#t=9',
        source: 'transcript',
        camera: null,
        speaker: null,
        text: 'Get on over there.',
      },
    ]);
  });
});
