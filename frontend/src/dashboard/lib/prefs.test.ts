import { expect, it } from 'vitest';
import { mirrored } from './prefs';

it.each(['__proto__', 'constructor', 'toString'])('uses only own mirror preferences for %s', cam => {
  expect(mirrored({ mirrorCams: {} }, cam)).toBe(false);
  expect(mirrored({ mirror: true, mirrorCams: {} }, cam)).toBe(true);
  expect(mirrored({ mirror: true, mirrorCams: JSON.parse(`{"${cam}":false}`) }, cam)).toBe(false);
});
