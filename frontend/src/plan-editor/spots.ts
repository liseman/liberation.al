// Camera spots in a scene: where each fixed camera is on the plan, how high, and which way it looks. They live in
// the level's `metadata.liberational.cameras` (spatial/plan_scene.py reads them there), so the editor keeps them
// through every save and undoes a change to one like any other edit. These are the pure parts: reading and writing
// a spot in that record, and the geometry of a facing.
import { level, type Pt, records, type Spot, spots } from '../dashboard/places/scene/draw';
import type { Scene } from './protocol';

export type { Pt, Spot };
export { spots };

/** How far a spot's view is drawn from it (metres), where its turn handle sits. */
export const REACH_M = 1.5;
/** How wide a view is drawn when the spot gives no field of view (degrees). */
export const DEFAULT_FOV = 90;
/** What a drag snaps to: metres for a spot, degrees for a facing. */
export const MOVE_SNAP_M = 0.05;
export const TURN_SNAP_DEG = 1;

const round = (v: number, step: number) => Math.round(Math.round(v / step) * step * 1000) / 1000;

/** A length snapped to the drag's step. */
export const snapped = (v: number): number => round(v, MOVE_SNAP_M);

/** A facing as stored: tenths of a degree, from 0 up to 360. */
export const facing = (yaw: number): number => {
  const v = round(((yaw % 360) + 360) % 360, 0.1);
  return v === 360 ? 0 : v;
};

/** The unit direction a facing looks, on the page: 0 is up (y falls), 90 to the right. */
export const heading = (yaw: number): Pt => [Math.sin((yaw * Math.PI) / 180), -Math.cos((yaw * Math.PI) / 180)];

/** The facing from one plan point toward another, snapped to whole degrees. */
export const bearing = (from: Pt, to: Pt): number =>
  facing(round((Math.atan2(to[0] - from[0], -(to[1] - from[1])) * 180) / Math.PI, TURN_SNAP_DEG));

/** Where a spot's turn handle sits: at the end of its view, or up the page while it has no facing. */
export const tipOf = (spot: Spot): Pt => {
  const [dx, dy] = heading(spot.yaw ?? 0);
  return [spot.at[0] + dx * REACH_M, spot.at[1] + dy * REACH_M];
};

/** A view's outline as an SVG path in plan metres, or null for a spot with no facing. */
export function cone(spot: Spot): string | null {
  if (spot.yaw === undefined) return null;
  const fov = spot.fov ?? DEFAULT_FOV;
  const [ax, ay] = heading(spot.yaw - fov / 2);
  const [bx, by] = heading(spot.yaw + fov / 2);
  const f = (n: number) => n.toFixed(3);
  const [x, y] = spot.at;
  return `M${f(x)} ${f(y)} L${f(x + ax * REACH_M)} ${f(y + ay * REACH_M)} A${REACH_M} ${REACH_M} 0 ${fov > 180 ? 1 : 0} 1 ${f(x + bx * REACH_M)} ${f(y + by * REACH_M)} Z`;
}

/** A spot as it is stored: its position, and only the numbers it has. */
function stored(spot: Spot): Record<string, unknown> {
  const out: Record<string, unknown> = { at: [spot.at[0], spot.at[1]] };
  if (spot.height !== undefined) out.height = spot.height;
  if (spot.yaw !== undefined) out.yaw = facing(spot.yaw);
  if (spot.fov !== undefined) out.fov = Math.min(Math.max(spot.fov, 1), 359);
  return out;
}

/** The level's `metadata` with `camera`'s spot set to `spot`, or taken away with null. Every other record, and
 *  every other camera's spot, is kept as it is; null when the scene has no level to hold one. */
export function withSpot(scene: Scene, camera: string, spot: Spot | null): Record<string, unknown> | null {
  const storey = level(scene);
  if (!storey) return null;
  const ours = records(scene);
  const cameras = { ...((ours.cameras as Record<string, unknown> | undefined) ?? {}) };
  if (spot) cameras[camera] = { ...((cameras[camera] as Record<string, unknown> | undefined) ?? {}), ...stored(spot) };
  else delete cameras[camera];
  // a number taken away (a facing cleared) must go from the stored spot too, not linger from before
  if (spot)
    for (const key of ['height', 'yaw', 'fov'] as const)
      if (spot[key] === undefined) delete (cameras[camera] as Record<string, unknown>)[key];
  return { ...(storey.metadata ?? {}), liberational: { ...ours, cameras } };
}

/** The middle of the plan's rooms (or of its walls, with no room yet), where a new spot goes when it isn't clicked
 *  into place; null for a scene with neither. */
export function middle(scene: Scene): Pt | null {
  const storey = level(scene);
  if (!storey) return null;
  const pts: Pt[] = [];
  for (const node of Object.values(scene.nodes)) {
    if (node.parentId !== storey.id) continue;
    const found = node.type === 'zone' ? node.polygon : node.type === 'wall' ? [node.start, node.end] : [];
    for (const p of Array.isArray(found) ? found : [])
      if (Array.isArray(p) && typeof p[0] === 'number' && typeof p[1] === 'number') pts.push([p[0], p[1]]);
  }
  if (pts.length === 0) return null;
  const xs = pts.map(p => p[0]);
  const ys = pts.map(p => p[1]);
  return [snapped((Math.min(...xs) + Math.max(...xs)) / 2), snapped((Math.min(...ys) + Math.max(...ys)) / 2)];
}
