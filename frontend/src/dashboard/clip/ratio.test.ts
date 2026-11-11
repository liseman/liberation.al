import { expect, it } from 'vitest';
import { controlBand } from './ratio';

it('uses portrait side bands at the narrowest activity card width', () => {
  expect(controlBand(238, 240, 9 / 16)).toBe('side');
});

it('uses the spare height below a landscape clip without shrinking the picture', () => {
  expect(controlBand(314, 240, 16 / 9)).toBe('bottom');
  expect(controlBand(346, 240, 16 / 9)).toBe('bottom');
});

it('overlays controls when the image fills the box or the bands cannot fit them', () => {
  expect(controlBand(320, 240, 4 / 3)).toBe('overlay');
  expect(controlBand(240, 240, 1)).toBe('overlay');
  expect(controlBand(240, 240, 0.8)).toBe('overlay');
  expect(controlBand(400, 240, 16 / 9)).toBe('overlay');
});

it('keeps an unknown or unmeasured image usable with an overlay', () => {
  expect(controlBand(240, 240, null)).toBe('overlay');
  expect(controlBand(0, 0, 9 / 16)).toBe('overlay');
});
