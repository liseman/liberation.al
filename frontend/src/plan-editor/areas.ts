// Objects and unexplored areas in a scene: an object is a labelled box on the plan (a sofa, a desk), an unexplored area
// is a named shape the house hasn't seen yet. They live in the level's `metadata.liberational`, under `objects` and
// `unexplored` (spatial/plan_scene.py reads and moves them there), so the editor keeps them through every save and
// undoes a change to one like any other edit. These are the pure parts: reading and writing one in that record, and the
// geometry of moving and reshaping it. An object's label is its `label`, an unexplored area's its `name`; both keep a
// ring of points as their `poly`, which is what scene/draw.ts draws and what the plan's move turns.
import { level, type Pt, records } from '../dashboard/places/scene/draw';
import type { Scene } from './protocol';

export type AreaKind = 'object' | 'unexplored';

/** An object or an unexplored area: its label (an object's `label`, an unexplored area's `name`) and its ring. */
export interface Area {
  id: string;
  kind: AreaKind;
  label: string;
  poly: Pt[];
}

/** What a drag snaps to: metres. */
export const MOVE_SNAP_M = 0.05;
/** How big a new object or unexplored area is: metres, across the page and down it. */
export const NEW_SIZE_M = 1.5;

const round = (v: number, step = MOVE_SNAP_M): number => Math.round(Math.round(v / step) * step * 1000) / 1000;
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const snap = ([x, y]: Pt): Pt => [round(x), round(y)];

/** A length snapped to the drag's step. */
export const snapped = (v: number): number => round(v);

const record = (v: unknown): Record<string, unknown> =>
  v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {};

const point = (v: unknown): Pt | null => {
  if (!Array.isArray(v)) return null;
  const [x, y] = [num(v[0]), num(v[1])];
  return x === undefined || y === undefined ? null : [x, y];
};

const points = (v: unknown): Pt[] => (Array.isArray(v) ? v.map(point).filter((p): p is Pt => p !== null) : []);

/** A record's ring: its `poly`, its `rect` (x, y, width, height), or a box of `size` centred on `at`; empty when it
 *  has none. The same reading as scene/draw.ts's outline, so the editor and the Places page agree on a shape. */
export function outline(value: unknown): Pt[] {
  const area = record(value);
  const poly = points(area.poly);
  if (poly.length >= 3) return poly;
  if (Array.isArray(area.rect) && area.rect.length === 4) {
    const [x, y, w, h] = area.rect.map(num);
    if (x !== undefined && y !== undefined && w !== undefined && h !== undefined)
      return [
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
      ];
  }
  const at = point(area.at);
  const size = point(area.size);
  if (!at || !size) return [];
  const [hw, hd] = [size[0] / 2, size[1] / 2];
  return [
    [at[0] - hw, at[1] - hd],
    [at[0] + hw, at[1] - hd],
    [at[0] + hw, at[1] + hd],
    [at[0] - hw, at[1] + hd],
  ];
}

/** The key a kind's records live under in the level's metadata. */
const KEY: Record<AreaKind, 'objects' | 'unexplored'> = { object: 'objects', unexplored: 'unexplored' };
/** The field a kind's label lives in: an object's `label`, an unexplored area's `name`. */
const LABEL: Record<AreaKind, 'label' | 'name'> = { object: 'label', unexplored: 'name' };

/** An object's or an unexplored area's record read as one, or null when it has no ring of at least three points. */
export function readArea(id: string, kind: AreaKind, value: unknown): Area | null {
  const poly = outline(value);
  if (poly.length < 3) return null;
  const label = record(value)[LABEL[kind]];
  return { id, kind, label: typeof label === 'string' ? label : '', poly };
}

/** Every object and unexplored area in a scene, each kind in its own list. */
export function areas(scene: Scene): Record<AreaKind, Area[]> {
  const ours = records(scene);
  const found: Record<AreaKind, Area[]> = { object: [], unexplored: [] };
  for (const kind of ['object', 'unexplored'] as const)
    for (const [id, value] of Object.entries(record(ours[KEY[kind]]))) {
      const area = readArea(id, kind, value);
      if (area) found[kind].push(area);
    }
  return found;
}

/** An area as it is stored: its ring, and its label or name when it has one. */
function stored(area: Area): Record<string, unknown> {
  const out: Record<string, unknown> = { poly: area.poly.map(snap) };
  if (area.label) out[LABEL[area.kind]] = area.label;
  return out;
}

/** The level's `metadata` with `id` set to `area`, or taken away with null. Every other record, and every other
 *  object and area, is kept as it is; null when the scene has no level to hold one. */
export function withArea(scene: Scene, kind: AreaKind, id: string, area: Area | null): Record<string, unknown> | null {
  const storey = level(scene);
  if (!storey) return null;
  const ours = records(scene);
  const key = KEY[kind];
  const held = { ...record(ours[key]) };
  if (area) held[id] = stored(area);
  else delete held[id];
  return { ...(storey.metadata ?? {}), liberational: { ...ours, [key]: held } };
}

/** A ring moved bodily by `by` metres on the page. */
export const translated = (poly: Pt[], by: Pt): Pt[] => poly.map(([x, y]) => snap([x + by[0], y + by[1]]));

/** A ring with one corner moved to `to`, the rest as they were. */
export const reshaped = (poly: Pt[], corner: number, to: Pt): Pt[] => poly.map((p, i) => (i === corner ? snap(to) : p));

/** A rectangle centred on `at`, `size` metres across the page and down it. */
export function box(at: Pt, size = NEW_SIZE_M): Pt[] {
  const [hw, hd] = [size / 2, size / 2];
  return [
    snap([at[0] - hw, at[1] - hd]),
    snap([at[0] + hw, at[1] - hd]),
    snap([at[0] + hw, at[1] + hd]),
    snap([at[0] - hw, at[1] + hd]),
  ];
}

/** The middle of a ring: its corners' average, which is all a drag needs to keep it under the pointer. */
export const middleOf = (poly: Pt[]): Pt => {
  const n = Math.max(1, poly.length);
  return [poly.reduce((sum, p) => sum + p[0], 0) / n, poly.reduce((sum, p) => sum + p[1], 0) / n];
};

/** A new record's id: random, so it doesn't collide with the ids a reloaded plan already holds. */
export function newId(kind: AreaKind): string {
  const suffix = globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2, 10);
  return `${kind}-${suffix}`;
}
