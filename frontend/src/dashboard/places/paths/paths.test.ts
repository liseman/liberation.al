import { expect, it } from 'vitest';
import type { Point } from '../api';
import { examplePositions } from './fixture';
import { windowAround } from './PlacePaths';
import {
  arrowHead,
  COLOURS,
  colours,
  despiked,
  FAINTEST,
  fade,
  gapFor,
  latest,
  pathData,
  plainFrame,
  plainToScreen,
  planRoom,
  READ_POINTS,
  roomsAt,
  SPLIT_GAP_S,
  segments,
  spline,
  stepped,
  UNKNOWN_COLOUR,
  walkable,
} from './paths';

const pt = (t: number): Point => ({ t, x: t / 1000, y: 0, room: null, confidence: 1, source: 'fixed-camera' });

it('fades a point from solid at the scrubber to the faintest at the window start, and no further', () => {
  expect(fade(0, 3600)).toBe(1);
  expect(fade(1800, 3600)).toBeCloseTo(1 - (1 - FAINTEST) / 2);
  expect(fade(3600, 3600)).toBeCloseTo(FAINTEST);
  expect(fade(99999, 3600)).toBeCloseTo(FAINTEST);
});

const at = (t: number, x: number, y = 0): Point => ({ ...pt(t), x, y });
const ts = (parts: Point[][]) => parts.map(p => p.map(x => x.t));

it('drops a lone foot point that jumps out and straight back, but not a real turn', () => {
  expect(despiked([at(0, 0), at(1, 0.4), at(2, 0.4, 3), at(3, 0.8)]).map(p => p.t)).toEqual([0, 1, 3]);
  expect(despiked([at(0, 0), at(1, 2), at(2, 2, 2)]).map(p => p.t)).toEqual([0, 1, 2]); // round a corner
});

it('lets go of points closer than a step to the last kept, keeping where they ended up', () => {
  // 0.1 and 0.2 are a shuffle on the spot; 0.5 is a step; 0.55 is where they are, so it stands in for 0.5
  expect(stepped([at(0, 0), at(1, 0.1), at(2, 0.2), at(3, 0.5), at(4, 0.55)]).map(p => p.t)).toEqual([0, 4]);
  expect(stepped([at(0, 0), at(1, 0.1)]).map(p => p.t)).toEqual([0, 1]); // never fewer than its two ends
});

it('tells a walk from a jump nobody could make', () => {
  expect(walkable(at(0, 0), at(1, 3.5))).toBe(true); // 2.5 m/s and a metre of foot-point error
  expect(walkable(at(0, 0), at(1, 3.6))).toBe(false);
  expect(walkable(at(0, 0), at(10, 20))).toBe(true); // 20 m in 10 s, across the house
  expect(walkable(at(0, 0), at(2, 6.5))).toBe(false); // a foot point put in the next room
});

it('splits a path only where it went quiet for minutes or jumped, and draws no lone point', () => {
  const track = [at(0, 0), at(10, 1), at(20, 2), at(20 + SPLIT_GAP_S + 1, 2.5), at(220, 3.5), at(221, 9)];
  expect(ts(segments(track, 1000))).toEqual([
    [0, 10, 20],
    [20 + SPLIT_GAP_S + 1, 220],
  ]);
  expect(ts(segments(track, 15))).toEqual([[0, 10]]); // nothing past the scrubber
  expect(segments([at(0, 0)], 1000)).toEqual([]);
});

it('keeps a path whole across a stray foot point instead of splitting or kinking at it', () => {
  expect(ts(segments([at(0, 0), at(1, 0.4), at(2, 0.4, 3), at(3, 0.8)], 10))).toEqual([[0, 1, 3]]);
});

it('runs the spline through every point, straight where they are', () => {
  const pieces = spline([
    [0, 0],
    [1, 0],
    [2, 0],
  ]);
  // evenly spaced on a line: every control point a third of the way along, as a straight line's
  expect(pieces).toEqual([
    [
      [0, 0],
      [1 / 3, 0],
      [2 / 3, 0],
      [1, 0],
    ],
    [
      [1, 0],
      [4 / 3, 0],
      [5 / 3, 0],
      [2, 0],
    ],
  ]);
  expect(spline([[0, 0]])).toEqual([]);
});

it("doesn't overshoot a short step between long ones, as a uniform spline would", () => {
  // A long step, a 5 cm one, and a long one: centripetal keeps the short piece's controls near it, not looping out
  const [, short] = spline([
    [0, 0],
    [4, 0],
    [4.05, 0],
    [8, 0],
  ]);
  const xs = (short ?? []).map(([x]) => x);
  expect(Math.min(...xs)).toBeGreaterThanOrEqual(4 - 1e-9);
  expect(Math.max(...xs)).toBeLessThanOrEqual(4.05 + 1e-9);
});

it('writes the curve as an SVG path and points its arrowhead the way it ends', () => {
  const line = spline([
    [0, 0],
    [10, 0],
  ]);
  expect(pathData(line)).toBe('M0.0,0.0C3.3,0.0 6.7,0.0 10.0,0.0');
  expect(arrowHead(line, 4)).toBe('10.0,0.0 6.0,2.2 6.0,-2.2');
  const up = spline([
    [0, 10],
    [0, 0],
  ]);
  expect(arrowHead(up, 4)).toBe('0.0,0.0 2.2,4.0 -2.2,4.0'); // heading up the page, the tip at the top
  // on a curve 5 long, the arrowhead is 0.6 of it, not the 4 asked for, so the line still shows
  expect(
    arrowHead(
      spline([
        [0, 0],
        [5, 0],
      ]),
      4,
    ),
  ).toBe('5.0,0.0 2.0,1.7 2.0,-1.7');
  expect(pathData([])).toBe('');
  expect(arrowHead([], 4)).toBe('');
});

it("finds where someone was at the scrubber's time: not after it, and not long before", () => {
  expect(latest([pt(10), pt(20), pt(30)], 25)?.t).toBe(20);
  expect(latest([pt(10)], 5)).toBeNull();
  expect(latest([pt(10)], 10 + SPLIT_GAP_S)?.t).toBe(10);
  expect(latest([pt(10)], 10 + SPLIT_GAP_S + 1)).toBeNull(); // a track that ended a while ago has no dot
});

it("puts a zone-only span on the plan room it names, or the one whose name matches the zone's", () => {
  const rooms = [{ id: 'bedroom', name: 'Bedroom' }];
  const span = { entity: 'pet:miso', room: null, name: ' bedroom ', start: 0, end: 10, confidence: 1, source: 'zone' };
  expect(planRoom(span, rooms)).toBe('bedroom');
  expect(planRoom({ ...span, name: 'Porch' }, rooms)).toBeNull();
  expect(planRoom({ ...span, room: 'den' }, rooms)).toBe('den');
  expect(roomsAt([span], 10)).toEqual([span]); // its end counts: a visit going on ends at the window's end
  expect(roomsAt([span], 10.5)).toEqual([]);
});

it('centres the window on a linked moment, but never past now', () => {
  expect(windowAround(1000, 600, 5000)).toEqual({ start: 700, end: 1300 });
  expect(windowAround(4900, 600, 5000)).toEqual({ start: 4400, end: 5000 });
});

it('makes example positions inside the window it was asked for', () => {
  const p = examplePositions('example-flat', 1000, 4600);
  const ts = p.tracks.flatMap(t => t.points.map(x => x.t));
  expect(Math.min(...ts)).toBeGreaterThanOrEqual(1000);
  expect(Math.max(...ts)).toBeLessThanOrEqual(4600);
});

it('colours the people and pets the house knows, and everyone else one grey', () => {
  const c = colours([
    { id: 'cam0:1', name: null, kind: 'unknown' },
    { id: 'person:a', name: 'A', kind: 'person' },
    { id: 'pet:b', name: 'B', kind: 'pet' },
  ]);
  expect([c.get('person:a'), c.get('pet:b'), c.get('cam0:1')]).toEqual([COLOURS[0], COLOURS[1], UNKNOWN_COLOUR]);
});

it("keeps a week's thinned paths whole and its dots, where an hour's would split", () => {
  const week = 7 * 86400;
  const every = week / READ_POINTS; // ~302 s: how the server thins a week
  expect(gapFor(3600)).toBe(SPLIT_GAP_S);
  expect(gapFor(week)).toBeCloseTo(1.5 * every);
  const track = [0, 1, 2, 3].map(i => at(i * every, i * 0.5));
  const end = 3 * every;
  expect(ts(segments(track, end, gapFor(week)))).toEqual([track.map(p => p.t)]); // one segment, no splits
  expect(segments(track, end, gapFor(3600))).toEqual([]); // at an hour's gap, every point would stand alone
  expect(latest(track, end + every, gapFor(week))?.t).toBe(end); // a point one step back still has its dot
});

it('fits a plain frame to every point with half a metre around, and at least 3 m each way, centred', () => {
  const at = (x: number, y: number) => ({ x, y });
  // x spans 5 m: 6 m with the margin; y spans 1 m: 2 m with it, so the 3 m least, centred on 2.5
  const f = plainFrame([{ points: [at(0, 2), at(5, 3)] }, { points: [at(2, 2.5)] }]);
  expect(f).toEqual({ minX: -0.5, minY: 1, width: 6, height: 3 });
  expect(plainToScreen(f as NonNullable<typeof f>, [0, 2])).toEqual([20, 40]); // 40 px a metre
  expect(plainFrame([])).toBeNull();
  expect(plainFrame([{ points: [] }])).toBeNull();
});
