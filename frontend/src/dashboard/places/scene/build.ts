// A Pascal scene made from a short description of a one-storey plan: walls as centreline segments, openings as a
// point on a wall, and rooms as named outlines. Only the fields we know are written; the editor fills in the rest
// when it loads the scene. Coordinates are the plan frame's (metres, x east, y south on the page), which are the
// scene's x and z.
import type { Scene, SceneNode } from '../../../plan-editor/protocol';

export type Pt = [number, number];

export interface PlanSpec {
  /** Floor to ceiling, metres. */
  height: number;
  walls: { from: Pt; to: Pt; thickness: number }[];
  /** A door, a window, or a doorway with no door in it (`cased`), centred at `at` on the wall that runs through it. */
  openings: { kind: 'door' | 'window' | 'cased'; at: Pt; width: number }[];
  rooms: { id: string; name: string; outline: Pt[] }[];
  /** Our own records, kept under the level's `metadata.liberational`. */
  extras?: Record<string, unknown>;
}

const DOOR_HEIGHT = 2.1;
const WINDOW_HEIGHT = 1.2;
const WINDOW_SILL = 0.9;
/** How far the site reaches past the walls on every side (metres). */
const SITE_MARGIN = 2;
/** How far (metres) a point may be from a wall's centreline and still be on it. */
const ON_WALL = 0.01;

/** How far along the segment `from` to `to` the point `p` is (metres from `from`), or null when it isn't on it. */
export function along(from: Pt, to: Pt, p: Pt): number | null {
  const [dx, dy] = [to[0] - from[0], to[1] - from[1]];
  const length = Math.hypot(dx, dy);
  if (length === 0) return null;
  const t = ((p[0] - from[0]) * dx + (p[1] - from[1]) * dy) / length;
  const off = Math.abs((p[0] - from[0]) * dy - (p[1] - from[1]) * dx) / length;
  return off <= ON_WALL && t >= -ON_WALL && t <= length + ON_WALL ? t : null;
}

export function buildScene(spec: PlanSpec): Scene {
  const nodes: Record<string, SceneNode> = {};
  const add = (type: string, id: string, parentId: string | null, fields: Record<string, unknown>): SceneNode => {
    const node: SceneNode = { object: 'node', id, type, parentId, visible: true, metadata: {}, ...fields };
    nodes[id] = node;
    const siblings = parentId ? nodes[parentId]?.children : undefined;
    if (siblings) siblings.push(id);
    return node;
  };
  // the site is the plan's own extent and a margin: the editor's default is a 30 m square, which leaves a home small
  const xs = spec.walls.flatMap(wall => [wall.from[0], wall.to[0]]);
  const ys = spec.walls.flatMap(wall => [wall.from[1], wall.to[1]]);
  const [left, right] = [Math.min(...xs) - SITE_MARGIN, Math.max(...xs) + SITE_MARGIN];
  const [top, bottom] = [Math.min(...ys) - SITE_MARGIN, Math.max(...ys) + SITE_MARGIN];
  const site = spec.walls.length
    ? {
        polygon: {
          type: 'polygon',
          points: [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
          ],
        },
      }
    : {};
  add('site', 'site_main', null, { children: [], ...site });
  add('building', 'building_main', 'site_main', { children: [] });
  const level = add('level', 'level_ground', 'building_main', { children: [], level: 0, height: spec.height });
  if (spec.extras) level.metadata = { liberational: spec.extras };
  spec.walls.forEach((wall, i) => {
    add('wall', `wall_${i}`, level.id, {
      children: [],
      start: wall.from,
      end: wall.to,
      thickness: wall.thickness,
      height: spec.height,
    });
  });
  spec.openings.forEach((opening, i) => {
    const on = spec.walls.findIndex(wall => along(wall.from, wall.to, opening.at) !== null);
    const wall = spec.walls[on];
    if (!wall) throw new Error(`the ${opening.kind} at ${opening.at.join(', ')} is on no wall`);
    const distance = along(wall.from, wall.to, opening.at) as number;
    if (opening.kind === 'window')
      add('window', `window_${i}`, `wall_${on}`, {
        wallId: `wall_${on}`,
        position: [distance, WINDOW_SILL + WINDOW_HEIGHT / 2, 0],
        width: opening.width,
        height: WINDOW_HEIGHT,
      });
    else
      add('door', `door_${i}`, `wall_${on}`, {
        wallId: `wall_${on}`,
        position: [distance, DOOR_HEIGHT / 2, 0],
        width: opening.width,
        height: DOOR_HEIGHT,
        openingKind: opening.kind === 'cased' ? 'opening' : 'door',
      });
  });
  for (const room of spec.rooms) add('zone', `zone_${room.id}`, level.id, { name: room.name, polygon: room.outline });
  return { nodes, rootNodeIds: ['site_main'] };
}
