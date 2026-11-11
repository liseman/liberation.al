import { describe, expect, it } from 'vitest';
import type { Sources } from '../api';
import { parseViews, UNTAGGED } from './load';
import {
  cameraState,
  countText,
  pointIndices,
  readHidden,
  toggled,
  viewMask,
  visibleViews,
  writeHidden,
} from './sources';

const view = (id: string, camera: string) => ({ id, camera, t: 1, points: 1, gaussians: 1 });
const SOURCES: Sources = {
  views: [view('a', 'porch'), view('b', 'hall'), view('c', 'porch')],
  cameras: [],
  splat_views: null,
};

describe('which views are drawn', () => {
  it('masks nothing while every view is shown, and gives the shown ones by index once one is hidden', () => {
    expect(visibleViews(SOURCES, new Set())).toBeNull();
    expect(visibleViews(SOURCES, new Set(['gone-from-this-map']))).toBeNull();
    expect(visibleViews(SOURCES, new Set(['a']))).toEqual(new Set([1, 2]));
    expect(visibleViews(SOURCES, new Set(['a', 'b', 'c']))).toEqual(new Set());
  });

  it("says a camera's box is on, off, or mixed by its views", () => {
    const porch = [SOURCES.views[0], SOURCES.views[2]].flatMap(v => (v ? [v] : []));
    expect(cameraState(porch, new Set(['b']))).toBe('on');
    expect(cameraState(porch, new Set(['a']))).toBe('mixed');
    expect(cameraState(porch, new Set(['a', 'c']))).toBe('off');
    expect(toggled(new Set(['a', 'b']), ['a', 'c'], true)).toEqual(new Set(['b']));
    expect(toggled(new Set(['b']), ['a'], false)).toEqual(new Set(['a', 'b']));
  });

  it("draws only the shown views' points, and always one of no known view", () => {
    const views = new Uint16Array([0, 1, UNTAGGED]);
    expect(Array.from(pointIndices(3, views, null))).toHaveLength(18);
    // point 0's square is corners 0 to 3; point 2's 8 to 11
    expect(Array.from(pointIndices(3, views, new Set([0])))).toEqual([0, 1, 2, 0, 2, 3, 8, 9, 10, 8, 10, 11]);
    expect(Array.from(pointIndices(3, null, new Set()))).toHaveLength(18); // a map that doesn't say: all drawn
  });

  it("marks each Gaussian drawn or hidden, rows of the texture's width", () => {
    const mask = viewMask(new Uint16Array([2, 0, UNTAGGED, 1]), new Set([1]), 3);
    expect(Array.from(mask)).toEqual([0, 0, 255, 255, 0, 0]);
    expect(Array.from(viewMask(new Uint16Array([5]), null, 2))).toEqual([255, 0]);
  });
});

describe('the views file', () => {
  it('reads little-endian views, and refuses bytes that cannot be one or are another count', () => {
    expect(Array.from(parseViews(new Uint8Array([1, 0, 0, 1]).buffer) ?? [])).toEqual([1, 256]);
    expect(parseViews(new Uint8Array([1, 0, 0]).buffer)).toBeNull();
    expect(parseViews(new Uint8Array([1, 0]).buffer, 2)).toBeNull();
  });
});

describe('what is kept for the session', () => {
  const store = () => {
    const kept = new Map<string, string>();
    return {
      getItem: (k: string) => kept.get(k) ?? null,
      setItem: (k: string, v: string) => void kept.set(k, v),
      removeItem: (k: string) => void kept.delete(k),
      kept,
    };
  };

  it('keeps hidden views per place, and nothing once all are shown', () => {
    const s = store();
    writeHidden('p1', new Set(['a']), s);
    expect(readHidden('p1', s)).toEqual(new Set(['a']));
    expect(readHidden('p2', s)).toEqual(new Set());
    writeHidden('p1', new Set(), s);
    expect(s.kept.size).toBe(0);
  });

  it('reads anything else kept there, or storage that refuses, as nothing hidden', () => {
    const s = store();
    s.setItem('places.cloud.hidden.p1', '{not json');
    expect(readHidden('p1', s)).toEqual(new Set());
    s.setItem('places.cloud.hidden.p1', '[1, "b"]');
    expect(readHidden('p1', s)).toEqual(new Set(['b']));
    const refusing = {
      getItem: () => {
        throw new Error('blocked');
      },
      setItem: () => {
        throw new Error('blocked');
      },
      removeItem: () => {},
    };
    expect(readHidden('p1', refusing)).toEqual(new Set());
    expect(() => writeHidden('p1', new Set(['a']), refusing)).not.toThrow();
  });
});

it('counts in short words', () => {
  expect(countText(1, 'points')).toBe('1 point');
  expect(countText(9_999, 'splats')).toBe('9,999 splats');
  expect(countText(123_456, 'points')).toBe('123k points');
  expect(countText(1_250_000, 'splats')).toBe('1.3M splats');
  expect(countText(null, 'splats')).toBe('');
});
