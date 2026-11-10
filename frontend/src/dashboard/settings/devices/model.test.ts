import { describe, expect, it } from 'vitest';
import { cameraStatus, parsePlaces, withPlace } from './model';

describe('camera names (vision.places)', () => {
  it('reads entries the way vision does: "camN: name", "N: name", "Cam 0 = name", "cam00: name"', () => {
    expect(parsePlaces('cam0: front camera, 2: side camera')).toEqual({ cam0: 'front camera', cam2: 'side camera' });
    expect(parsePlaces('Cam 1 = den, cam03: hall')).toEqual({ cam1: 'den', cam3: 'hall' });
    expect(parsePlaces('')).toEqual({});
  });

  it("keeps another camera's hand-written name, and never names one twice", () => {
    expect(withPlace('cam 0: hall', 'cam1', 'den')).toBe('cam0: hall, cam1: den');
    expect(withPlace('cam00: hall', 'cam0', 'den')).toBe('cam0: den');
  });

  it('names one camera, keeps the others in order, and drops an emptied name', () => {
    expect(withPlace('cam2: side', 'cam0', 'front')).toBe('cam0: front, cam2: side');
    expect(withPlace('cam0: front, cam2: side', 'cam0', '')).toBe('cam2: side');
    expect(withPlace('', 'cam1', 'rear, left')).toBe('cam1: rear  left'); // a comma would split the list
  });
});

describe('cameraStatus', () => {
  const row = { present: true, selected: true };
  it.each([
    [{ ...row, present: false }, { state: 'on' }, 'Not plugged in'],
    [{ ...row, selected: false }, undefined, 'Not used'],
    [row, undefined, 'Off: capture is sending nothing from it'],
    [row, { state: 'retrying' }, 'Not responding, retrying'],
    [row, { state: 'starting' }, 'Starting…'],
    [row, { state: 'on', size: [1280, 720], fps: 30 }, 'On, 1280×720 at 30 fps'],
  ])('%o with %o says %s', (r, vision, text) => {
    expect(cameraStatus(r, vision).text).toBe(text);
  });
});

describe('native camera place names', () => {
  it('preserves exact configured IDs before interpreting legacy aliases', () => {
    expect(parsePlaces('entryway: Front, 0: Zero, cam00: Padded, CAM0: Upper', ['0', 'cam00', 'CAM0'])).toEqual({
      entryway: 'Front',
      '0': 'Zero',
      cam00: 'Padded',
      CAM0: 'Upper',
    });
  });
  it('round trips named cameras and sorts natural numeric portions deterministically', () => {
    expect(withPlace('rear10: Ten, rear2: Two, cam00: Padded', 'entryway', 'Front', ['cam00'])).toBe(
      'cam00: Padded, entryway: Front, rear2: Two, rear10: Ten',
    );
  });
});
