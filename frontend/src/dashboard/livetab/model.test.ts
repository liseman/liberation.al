import { describe, expect, it } from 'vitest';
import { faceGroup, knownIn, unknownFaces, where } from './model';

describe('faces', () => {
  it.each([
    [null, ''],
    [0, 'straight ahead'],
    [90, "at 3 o'clock"],
    [-90, "at 9 o'clock"],
    [350, 'straight ahead'],
  ])('bearing %s is %s', (deg, text) => {
    expect(where(deg)).toBe(text);
  });

  it('lists known names once, leaving out unnamed and unknown faces', () => {
    expect(knownIn([{ name: 'Dana' }, { name: 'Face 3' }, { name: 'unknown' }, { name: 'Dana' }, {}])).toEqual([
      'Dana',
    ]);
  });

  it('asks about unknown faces once each, and only ones big enough to be in the room', () => {
    const faces = [
      { name: 'Face 3', bbox: [0, 0, 30, 30] as [number, number, number, number] },
      { name: 'Face 3', bbox: [50, 0, 90, 40] as [number, number, number, number] },
      { name: 'Face 4', bbox: [0, 0, 23, 40] as [number, number, number, number] },
      { name: 'Dana', bbox: [0, 0, 40, 40] as [number, number, number, number] },
    ];
    expect(unknownFaces(faces).map(f => [f.name, f.bbox?.[0]])).toEqual([['Face 3', 0]]);
  });

  it('reads the face group number', () => {
    expect([faceGroup('Face 12'), faceGroup('Dana'), faceGroup(undefined)]).toEqual([12, null, null]);
  });
});
