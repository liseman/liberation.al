// The 3D view's geometry, apart from the engine so it's tested without WebGL. Everything is drawn in the
// place map's frame (metres), where the cloud is: the plan comes in through the inverse of the revision's plan
// transform, the cameras through their poses, and height is measured from the floor plane.

import type { Scene } from '../../../plan-editor/protocol';
import { floorplan, parseDsl } from '../../../vendor/floorplan/index.js';
import { split } from '../edit/text';
import { level, under } from '../scene/draw';

export type V3 = [number, number, number];

/** A map point to the plan frame and back: the revision's plan transform is 3x4, row by row, map (x, y, z, 1) to plan
 *  (x, y, height). `toMap` inverts it, for drawing the plan's outline in the cloud's frame. */
export function planFrame(transform: readonly number[]): { toPlan: (p: V3) => V3; toMap: (p: V3) => V3 } {
  const [a, b, c, tx, d, e, f, ty, g, h, i, tz] = transform as [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) throw new Error('the plan transform has no inverse');
  // the inverse of the 3x3 part, by cofactors
  const inv = [
    (e * i - f * h) / det,
    (c * h - b * i) / det,
    (b * f - c * e) / det,
    (f * g - d * i) / det,
    (a * i - c * g) / det,
    (c * d - a * f) / det,
    (d * h - e * g) / det,
    (b * g - a * h) / det,
    (a * e - b * d) / det,
  ] as const;
  return {
    toPlan: ([x, y, z]) => [a * x + b * y + c * z + tx, d * x + e * y + f * z + ty, g * x + h * y + i * z + tz],
    toMap: ([x, y, z]) => {
      const p = [x - tx, y - ty, z - tz] as const;
      return [
        inv[0] * p[0] + inv[1] * p[1] + inv[2] * p[2],
        inv[3] * p[0] + inv[4] * p[1] + inv[5] * p[2],
        inv[6] * p[0] + inv[7] * p[1] + inv[8] * p[2],
      ];
    },
  };
}

/** How far `p` is above the floor plane (a, b, c, d: a unit normal and its offset, as spatial/cameras.py fits it). */
export const heightOf = (p: V3, floor: readonly number[]): number =>
  (floor[0] ?? 0) * p[0] + (floor[1] ?? 0) * p[1] + (floor[2] ?? 1) * p[2] + (floor[3] ?? 0);

/** A camera's view as a frustum in the map frame: its centre, and the four corners of its image `depth` metres out.
 *  `pose` is camera to map, 4x4 row by row; the camera looks along its +z with its principal point at the centre
 *  (a pinhole, as MASt3R fits it, and as memory/places.py `project` reads it). */
export function frustum(pose: readonly number[], focal: number, width: number, height: number, depth = 0.6): V3[] {
  const m = (r: number, c: number) => pose[4 * r + c] ?? 0;
  const toMap = ([x, y, z]: V3): V3 => [0, 1, 2].map(r => m(r, 0) * x + m(r, 1) * y + m(r, 2) * z + m(r, 3)) as V3;
  const corner = (u: number, v: number): V3 =>
    toMap([((u - width / 2) / focal) * depth, ((v - height / 2) / focal) * depth, depth]);
  return [toMap([0, 0, 0]), corner(0, 0), corner(width, 0), corner(width, height), corner(0, height)];
}

/** A colour for a height on a cool-to-warm ramp, low blue through green to high red: what "colour by height" shows. */
export function heightColour(h: number, low: number, high: number): V3 {
  const t = high > low ? Math.min(1, Math.max(0, (h - low) / (high - low))) : 0.5;
  // Four stops, linearly between: blue, cyan, yellow, red.
  const stops: V3[] = [
    [0.19, 0.3, 0.86],
    [0.13, 0.75, 0.8],
    [0.98, 0.84, 0.25],
    [0.86, 0.2, 0.17],
  ];
  const x = t * (stops.length - 1);
  const k = Math.min(stops.length - 2, Math.floor(x));
  const f = x - k;
  const a = stops[k] as V3;
  const b = stops[k + 1] as V3;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** The low and high of the middle 98% of `values`: a colour ramp that a few stray points (a ceiling light, a hole in
 *  the floor) don't flatten. */
export function robustRange(values: Float32Array | number[]): [number, number] {
  if (!values.length) return [0, 1];
  const sorted = Float32Array.from(values).sort();
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))] ?? 0;
  return [at(0.01), at(0.99)];
}

/** A stored plan's text as a Pascal scene, or null for one in the old DSL (a scene is JSON; DSL never starts with a
 *  brace). Throws for a scene that doesn't read, as the DSL reader does for its own. */
function sceneOf(planText: string): Scene | null {
  if (!planText.trimStart().startsWith('{')) return null;
  const scene = JSON.parse(planText) as Partial<Scene> | null;
  if (!scene || typeof scene.nodes !== 'object' || scene.nodes === null || !Array.isArray(scene.rootNodeIds))
    throw new Error('the plan is not a scene');
  return scene as Scene;
}

const pair = (v: unknown): [number, number] | null =>
  Array.isArray(v) && typeof v[0] === 'number' && typeof v[1] === 'number' ? [v[0], v[1]] : null;

/** A scene's rooms as outlines, and its walls as lines with their doorways cut out, in the plan frame. A window is
 *  wall; a door or a doorway with no door is a gap, as wide as it is, about its place along the wall. */
function sceneLines(scene: Scene): { rooms: [number, number][][]; walls: [[number, number], [number, number]][] } {
  const storey = level(scene);
  if (!storey) return { rooms: [], walls: [] };
  const rooms = under(scene, storey, 'zone')
    .map(zone => (Array.isArray(zone.polygon) ? zone.polygon.map(pair).filter(p => p !== null) : []))
    .filter(poly => poly.length >= 3);
  const walls: [[number, number], [number, number]][] = [];
  for (const wall of under(scene, storey, 'wall')) {
    const [a, b] = [pair(wall.start), pair(wall.end)];
    if (!a || !b) continue;
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (length === 0) continue;
    const gaps = under(scene, wall, 'door')
      .map(door => {
        const at = Array.isArray(door.position) && typeof door.position[0] === 'number' ? door.position[0] : length / 2;
        const half = (typeof door.width === 'number' ? door.width : 0.9) / 2;
        return [Math.max(0, at - half), Math.min(length, at + half)] as [number, number];
      })
      .sort((p, q) => p[0] - q[0]);
    const at = (d: number): [number, number] => [
      a[0] + ((b[0] - a[0]) * d) / length,
      a[1] + ((b[1] - a[1]) * d) / length,
    ];
    let done = 0;
    for (const [from, to] of gaps) {
      if (from - done > 1e-6) walls.push([at(done), at(from)]);
      done = Math.max(done, to);
    }
    if (length - done > 1e-6) walls.push([at(done), at(length)]);
  }
  return { rooms, walls };
}

/** The plan's rooms as outlines in the map frame, a little above the floor (`lift` metres), from its stored text: a
 *  Pascal scene's zones, or the rooms of a plan in the old DSL. */
export function planOutline(planText: string, transform: readonly number[], lift = 0.02): V3[][] {
  const { toMap } = planFrame(transform);
  const scene = sceneOf(planText);
  if (scene) return sceneLines(scene).rooms.map(poly => poly.map(([x, y]) => toMap([x, y, lift])));
  const { doc } = parseDsl(split(planText).base);
  return floorplan(doc).model.rooms.map(r => (r.room.poly as [number, number][]).map(([x, y]) => toMap([x, y, lift])));
}

/** The plan's walls as lines on the floor, in the map's frame, with its doorways cut out of them: what a walk
 *  through the place is stopped by. A door or a cased opening is a gap; a window is wall. Only straight walls: a
 *  curved one is left out, so it stops nothing. */
export function planWalls(planText: string, transform: readonly number[], lift = 0): [V3, V3][] {
  const { toMap } = planFrame(transform);
  const scene = sceneOf(planText);
  if (scene) return sceneLines(scene).walls.map(([a, b]) => [toMap([a[0], a[1], lift]), toMap([b[0], b[1], lift])]);
  const { doc } = parseDsl(split(planText).base);
  const { walls, openings } = floorplan(doc).model;
  const out: [V3, V3][] = [];
  for (const wall of walls) {
    if (wall.geometry.kind !== 'segment' || wall.to === wall.from) continue;
    const [ax, ay] = wall.start;
    const [bx, by] = wall.end;
    // where along the wall (0 at its start, 1 at its end) each doorway begins and ends
    const gaps = openings
      .filter(o => o.wall === wall && o.spec.type !== 'window')
      .map(
        o => [o.from, o.to].map(v => (v - wall.from) / (wall.to - wall.from)).sort((p, q) => p - q) as [number, number],
      )
      .sort((p, q) => p[0] - q[0]);
    const piece = (from: number, to: number) => {
      if (to - from < 1e-6) return;
      const at = (t: number) => toMap([ax + (bx - ax) * t, ay + (by - ay) * t, lift]);
      out.push([at(from), at(to)]);
    };
    let done = 0;
    for (const [from, to] of gaps) {
      piece(done, Math.max(done, Math.min(1, from)));
      done = Math.max(done, Math.min(1, to));
    }
    piece(done, 1);
  }
  return out;
}

/** The vertical field of view (degrees) that fills a view of `aspect` (width over height) with a fixed camera's own
 *  frame (`width` x `height` pixels at `focal`): its frame covers the view, cropped to the view's shape, so the view
 *  is filled and shows nothing past the frame's edges, nothing the camera never saw (#42). A view wider than
 *  the frame keeps its full width, a taller one its full height. */
export function coverFov(width: number, height: number, focal: number, aspect: number): number {
  const halfTan = Math.min(height, width / aspect) / 2 / focal;
  return (2 * Math.atan(halfTan) * 180) / Math.PI;
}
