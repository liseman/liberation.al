import { describe, expect, it } from 'vitest';
import type { ActivityPromotion, DetectedActivity } from '../review/api';
import { kindRows, progress, recentDays, spotChecked, toCheck } from './recent';

const clip = (id: string, kind: string | null, feedback?: 'right' | 'wrong'): DetectedActivity => ({
  id,
  camera: 'cam0',
  start: 0,
  end: 5,
  count: null,
  name: null,
  kind,
  person: null,
  promoted: false,
  ...(feedback ? { feedback } : {}),
});
const promotion = (kind: string | null, status: ActivityPromotion['status'], right: number, left: number | null) => ({
  kind,
  camera: null,
  status,
  remaining_ratings: left,
  rated: right,
  right,
  precision: null,
});

describe('recentDays', () => {
  it('is the last seven local days, today first, across a month boundary', () => {
    expect(recentDays(new Date(2026, 10, 3, 9))).toEqual([
      '2026-11-03',
      '2026-11-02',
      '2026-11-01',
      '2026-10-31',
      '2026-10-30',
      '2026-10-29',
      '2026-10-28',
    ]);
  });
});

describe('kindRows', () => {
  it('lists counted kinds first, then the kinds with the most clips waiting, and counts only unanswered clips', () => {
    const rows = kindRows(
      [
        clip('a', 'drinking'),
        clip('b', 'drinking'),
        clip('c', 'squats'),
        clip('d', 'squats', 'right'),
        clip('e', null),
      ],
      [promotion('walked_past', 'counted', 20, 0), promotion('drinking', 'needs_ratings', 2, 4)],
    );
    expect(rows.map(r => [r.kind, r.waiting, r.counted])).toEqual([
      ['walked_past', 0, true],
      ['drinking', 2, false],
      ['squats', 1, false], // a tie on waiting goes by name
      [null, 1, false],
    ]);
    expect(rows[3]?.label).toBe('Unclassified activity');
  });
});

describe('progress', () => {
  it.each([
    [undefined, 0],
    [promotion('x', 'counted', 9, 0), 1],
    [promotion('x', 'needs_ratings', 6, 4), 0.6],
    [promotion('x', 'needs_ratings', 0, 0), 0],
    [promotion('x', 'needs_correct_ratings', 3, null), 0],
  ])('reads %o as %d', (p, want) => {
    expect(progress(p)).toBeCloseTo(want);
  });
});

describe('spotChecked', () => {
  it('picks the same clips every time, by their ids (worked out apart from this code)', () => {
    const ids = Array.from({ length: 20 }, (_, i) => `walk-${i}`);
    expect(ids.filter(spotChecked)).toEqual(['walk-7', 'walk-14']);
    expect(ids.filter(spotChecked)).toEqual(['walk-7', 'walk-14']);
  });

  it('keeps about 1 in 10', () => {
    expect(Array.from({ length: 10_000 }, (_, i) => `clip-${i}`).filter(spotChecked)).toHaveLength(995);
  });
});

describe('toCheck', () => {
  it("keeps unanswered clips, all of a learning kind's and only the spot-checked sample of a counted kind's", () => {
    const rows = [
      clip('walk-7', 'walked_past'), // sampled (see spotChecked)
      clip('walk-8', 'walked_past'),
      clip('d1', 'drinking'),
      clip('d2', 'drinking', 'right'),
      clip('u1', null),
    ].map(item => ({ day: '2026-11-05', item }));
    const kept = toCheck(rows, [
      promotion('walked_past', 'counted', 20, 0),
      promotion('drinking', 'needs_ratings', 2, 4),
    ]);
    expect(kept.map(r => r.item.id)).toEqual(['walk-7', 'd1', 'u1']);
  });

  it("samples nothing when only a camera's own promotion is counted", () => {
    const rows = [clip('walk-8', 'walked_past')].map(item => ({ day: '2026-11-05', item }));
    expect(toCheck(rows, [{ ...promotion('walked_past', 'counted', 20, 0), camera: 'cam0' }])).toHaveLength(1);
  });
});
