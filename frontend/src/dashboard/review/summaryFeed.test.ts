import { describe, expect, it } from 'vitest';
import { hm } from '../lib/format';
import type { Block, DetectedActivity } from './api';
import { activityWords, feedRows, foldRuns, spanWords, stepWords } from './summaryFeed';

const block = (start: number, end: number): Block => ({ start, end, text: `${start}`, tags: [] });
const activity = (id: string, start: number): DetectedActivity => ({
  id,
  camera: 'cam0',
  start,
  end: start + 5,
  count: null,
  name: null,
  kind: 'drinking',
  person: null,
  promoted: true,
});

describe('feedRows', () => {
  it('puts each activity in the stretch it starts in, and the rest in time order between them', () => {
    const rows = feedRows(
      [block(100, 200), block(300, 400)],
      [activity('late', 350), activity('gap', 250), activity('in', 150), activity('before', 50)],
    );
    expect(
      rows.map(r => (r.kind === 'block' ? `block ${r.at}: ${r.activities.map(a => a.id)}` : r.activity.id)),
    ).toEqual(['before', 'block 100: in', 'gap', 'block 300: late']);
  });

  it('counts an activity at a stretch end as outside it, and lists a stretch before an activity at its start', () => {
    const rows = feedRows([block(100, 200)], [activity('edge', 200)]);
    expect(rows.map(r => r.kind)).toEqual(['block', 'activity']);
    const tie = feedRows([block(100, 200), block(150, 250)], [activity('both', 160)]);
    expect(tie.map(r => (r.kind === 'block' ? r.activities.length : -1))).toEqual([1, 0]); // the first stretch only
  });

  it('is empty with nothing to show', () => {
    expect(feedRows([], [])).toEqual([]);
  });
});

describe('stepWords', () => {
  it.each([
    [3600, 'by the hour'],
    [7200, 'in 2-hour steps'],
    [600, 'in 10-minute steps'],
    [180, 'in 3-minute steps'],
    [30, 'in 30-second steps'],
  ])('reads level %i as %s', (level, words) => {
    expect(stepWords(level)).toBe(words);
  });
});

describe('spanWords', () => {
  it('says clock times under a day, else the whole day or a count of days', () => {
    expect(spanWords(3600, 7200)).toBe(`${hm(3600)} to ${hm(7200)}`);
    expect(spanWords(0, 86400)).toBe('the whole day');
    expect(spanWords(0, 82800)).toBe('the whole day'); // a 23-hour day across a clock change
    expect(spanWords(0, 3 * 86400)).toBe('3 days');
  });
});

describe('foldRuns (#2454 follow-up)', () => {
  const a = (id: string, name: string, promoted = false) => ({ id, name, promoted });
  const name = (x: { name: string }) => x.name;
  it('folds 3 or more unconfirmed of a kind where the kind first appears, and keeps the rest in order', () => {
    const items = [a('1', 'read'), a('2', 'walk', true), a('3', 'read'), a('4', 'sleep'), a('5', 'read')];
    expect(foldRuns(items, name)).toEqual([
      { kind: 'fold', name: 'read', items: [items[0], items[2], items[4]] },
      { kind: 'one', item: items[1] },
      { kind: 'one', item: items[3] },
    ]);
  });
  it('never folds confirmed ones, or fewer than 3', () => {
    const items = [a('1', 'read', true), a('2', 'read', true), a('3', 'read', true), a('4', 'read'), a('5', 'read')];
    expect(foldRuns(items, name).every(r => r.kind === 'one')).toBe(true);
  });
});

describe('activityWords (#2577 follow-up)', () => {
  const item = (more: Partial<DetectedActivity>): DetectedActivity => ({
    id: 'a',
    camera: 'cam0',
    start: 0,
    end: 1,
    count: null,
    name: null,
    kind: 'crossing',
    person: null,
    promoted: true,
    ...more,
  });
  it.each([
    ['who walked past', { person_name: 'Joe' }, 'Joe walked past'],
    ['nobody known', { person_name: null }, 'Walked past'],
    ['a blank name', { person_name: '  ' }, 'Walked past'],
    ['an owner-named activity', { name: 'Squats', person_name: 'Dana' }, 'Dana squats'],
    ['an acronym, kept', { name: 'TV on', person_name: 'Dana' }, 'Dana TV on'],
  ])('%s', (_case, more, words) => {
    expect(activityWords(item(more))).toBe(words);
  });
});
