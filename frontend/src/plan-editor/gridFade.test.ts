import { expect, it } from 'vitest';
import { FADE_M, MARGIN_M, MASK_ID, maskMarkup, ramp, union } from './gridFade';

it('keeps the grid full a margin past the plan, then fades it out over the fade', () => {
  const { from, to, stops } = ramp(2, 10);
  expect(from).toBe(2 - MARGIN_M - FADE_M);
  expect(to).toBe(12 + MARGIN_M + FADE_M);
  // full strength begins one fade in from each end, which is a margin outside the plan
  expect(from + stops[0] * (to - from)).toBeCloseTo(2 - MARGIN_M);
  expect(from + stops[1] * (to - from)).toBeCloseTo(12 + MARGIN_M);
});

it('fits a box around what is drawn, and has none when nothing is', () => {
  expect(
    union([
      { x: 0, y: 0, width: 4, height: 3 },
      { x: 3, y: -1, width: 2, height: 1 },
      { x: 0, y: 0, width: 0, height: 0 }, // a layer with nothing in it yet
    ]),
  ).toEqual({ x: 0, y: -1, width: 5, height: 4 });
  expect(union([{ x: 0, y: 0, width: 0, height: 0 }])).toBeNull();
  expect(union([])).toBeNull();
});

it('makes a mask the stylesheet can name, faded both ways', () => {
  const markup = maskMarkup({ x: 0, y: 0, width: 9, height: 8 });
  expect(markup).toContain(`<mask id="${MASK_ID}"`);
  expect(markup).toContain(`url(#${MASK_ID}-x)`);
  expect(markup).toContain(`mask="url(#${MASK_ID}-v)"`);
  const doc = new DOMParser().parseFromString(
    `<svg xmlns="http://www.w3.org/2000/svg">${markup}</svg>`,
    'image/svg+xml',
  );
  expect(doc.querySelector('parsererror')).toBeNull();
});
