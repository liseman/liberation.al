import { describe, expect, it } from 'vitest';
import { area, centre, type Point, pointAt, problem, simple } from './geometry';

const square: Point[] = [
  [0.1, 0.1],
  [0.3, 0.1],
  [0.3, 0.3],
  [0.1, 0.3],
];

describe('pointAt', () => {
  const rect = { left: 100, top: 50, width: 400, height: 200 };

  it('turns a click into a share of the picture', () => {
    expect(pointAt(rect, 200, 100)).toEqual([0.25, 0.25]);
  });

  it('clamps a click past the edge to the frame', () => {
    expect(pointAt(rect, 50, 300)).toEqual([0, 1]);
  });
});

describe('problem', () => {
  it.each([
    ['two corners', square.slice(0, 2), 'at least three'],
    [
      'a figure-eight',
      [
        [0.1, 0.1],
        [0.3, 0.3],
        [0.3, 0.1],
        [0.1, 0.3],
      ] as Point[],
      'crosses itself',
    ],
    [
      'a sliver',
      [
        [0.1, 0.1],
        [0.105, 0.1],
        [0.105, 0.105],
      ] as Point[],
      'too small',
    ],
    ['33 corners', Array.from({ length: 33 }, (_, i) => [i / 40, (i % 2) / 2] as Point), 'at most 32'],
  ])('says what is wrong with %s', (_, polygon, message) => {
    expect(problem(polygon)).toContain(message);
  });

  it('accepts a square', () => {
    expect(problem(square)).toBeNull();
  });
});

it('reads the area and centre of a square, and a square as simple', () => {
  expect([area(square), centre(square), simple(square)]).toEqual([
    expect.closeTo(0.04, 10),
    [expect.closeTo(0.2, 10), expect.closeTo(0.2, 10)],
    true,
  ]);
});
