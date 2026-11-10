import { expect, it } from 'vitest';
import { deviceName } from './deviceName';

it.each([
  ['Elgato Facecam: Elgato Facecam', 'Elgato Facecam'],
  ['HD Pro Webcam C920: HD Pro Webcam', 'HD Pro Webcam C920: HD Pro Webcam'],
  ['FaceTime HD Camera', 'FaceTime HD Camera'],
  ['', ''],
])('shows %j as %j', (raw, shown) => {
  expect(deviceName(raw)).toBe(shown);
});
