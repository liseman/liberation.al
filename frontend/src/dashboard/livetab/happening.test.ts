import { describe, expect, it } from 'vitest';
import { type HappeningItem, KEEP, merge } from './happening';

const item = (id: string, ts: number): HappeningItem => ({
  id,
  ts,
  kind: 'activity',
  text: id,
  people: [],
  cam: null,
  place: null,
});

describe("merging a poll of what's happening", () => {
  it('adds the new rows in time order, each once, and moves on to the newest time', () => {
    const prev = { items: [item('a', 10), item('b', 20)], updated: 20 };
    const next = { items: [item('b', 20), item('c', 15), item('d', 30)], updated: 30 };
    expect(merge(prev, next)).toEqual({
      items: [item('a', 10), item('c', 15), item('b', 20), item('d', 30)],
      updated: 30,
    });
  });

  it('keeps the newest rows only', () => {
    const many = Array.from({ length: KEEP + 5 }, (_, i) => item(`r${i}`, i));
    const got = merge(undefined, { items: many, updated: KEEP + 4 });
    expect([got.items.length, got.items[0]?.id, got.items.at(-1)?.id]).toEqual([KEEP, 'r5', `r${KEEP + 4}`]);
  });

  it('keeps the last time when a poll brings nothing new', () => {
    expect(merge({ items: [item('a', 10)], updated: 10 }, { items: [], updated: null }).updated).toBe(10);
  });
});

it('retracts an anonymous episode after a named sample identifies that track', () => {
  const anonymous = item('a:unknown:1', 1);
  const named = item('a:person:alice:1', 1);
  const result = merge(
    { items: [anonymous], updated: 1 },
    { items: [{ ...anonymous, removed: true }, named], updated: 3 },
  );
  expect(result.items).toEqual([named]);
});
