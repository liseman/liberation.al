// A place's plan drawn from its Pascal scene, as a small SVG: rooms with their names, walls at their thickness
// with doors and windows in them, the camera spots, and our objects and unexplored areas. This is the everyday
// Places view, so it is a few kilobytes of our own code and loads nothing of the editor. Paths are drawn over it
// through `surface` (paths/PlacePaths.tsx).
//
// The plan frame is the scene's x and z: metres, x east, y south on the page.
import type { Scene, SceneNode } from '../../../plan-editor/protocol';

export type Pt = [number, number];

/** What paths need of a drawn plan: where a plan point lands in the SVG, and the rooms to light up. */
export interface Surface {
  toScreen: (p: Pt) => Pt;
  rooms: { id: string; name: string; poly: Pt[] }[];
}

/** A camera's spot on a plan (the level's `metadata.liberational.cameras`). */
export interface Spot {
  at: Pt;
  /** Metres above the floor. */
  height?: number;
  /** Degrees clockwise from up the page, which way it looks. */
  yaw?: number;
  /** Horizontal field of view, degrees. */
  fov?: number;
}

export interface SceneDrawing {
  svg: string;
  surface: Surface;
  /** How many rooms the plan has, and their floor area on wall centrelines (m²). */
  rooms: number;
  area: number;
  /** The cameras with a spot on the plan. */
  cameras: Record<string, Spot>;
  /** Whether it draws an unexplored area (for the key). */
  unexplored: boolean;
}

/** SVG units per metre, and the bare floor left around the plan (metres). */
export const PX_PER_M = 60;
const MARGIN = 0.6;
/** How far a camera's view is drawn from its spot, and its field of view when the spot doesn't say (degrees). */
const VIEW_M = 1.2;
const DEFAULT_FOV = 90;

const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const point = (v: unknown): Pt | null => {
  if (!Array.isArray(v)) return null;
  const [x, y] = [num(v[0]), num(v[1])];
  return x === undefined || y === undefined ? null : [x, y];
};
const points = (v: unknown): Pt[] => (Array.isArray(v) ? v.map(point).filter((p): p is Pt => p !== null) : []);
const record = (v: unknown): Record<string, unknown> =>
  v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
const esc = (text: string): string =>
  text.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] ?? c);

/** The storey a plan is drawn on: the scene's lowest level. */
export function level(scene: Scene): SceneNode | null {
  const levels = Object.values(scene.nodes).filter(n => n.type === 'level');
  return levels.sort((a, b) => (num(a.level) ?? 0) - (num(b.level) ?? 0))[0] ?? null;
}

/** A node's children of one type, in its own order, then any that name it as parent but aren't listed. */
export function under(scene: Scene, parent: SceneNode, type: string): SceneNode[] {
  const listed = (parent.children ?? []).map(id => scene.nodes[id]).filter((n): n is SceneNode => n !== undefined);
  const seen = new Set(listed.map(n => n.id));
  const rest = Object.values(scene.nodes).filter(n => n.parentId === parent.id && !seen.has(n.id));
  return [...listed, ...rest].filter(n => n.type === type);
}

/** Our own records in a scene: the level's `metadata.liberational`. */
export function records(scene: Scene): Record<string, unknown> {
  return record(record(level(scene)?.metadata).liberational);
}

/** The camera spots on a scene that have a position. */
export function spots(scene: Scene): Record<string, Spot> {
  const out: Record<string, Spot> = {};
  for (const [name, value] of Object.entries(record(records(scene).cameras))) {
    const spot = record(value);
    const at = point(spot.at);
    if (!at) continue;
    out[name] = { at, height: num(spot.height), yaw: num(spot.yaw), fov: num(spot.fov) };
  }
  return out;
}

/** A polygon's area (m²) and the middle of its floor, by the shoelace sums. A sliver's middle is its corners'. */
export function centre(poly: Pt[]): { area: number; at: Pt } {
  let twice = 0;
  let cx = 0;
  let cy = 0;
  poly.forEach(([x0, y0], i) => {
    const [x1, y1] = poly[(i + 1) % poly.length] as Pt;
    const cross = x0 * y1 - x1 * y0;
    twice += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  });
  if (Math.abs(twice) < 1e-9) {
    const n = Math.max(1, poly.length);
    return { area: 0, at: [poly.reduce((s, p) => s + p[0], 0) / n, poly.reduce((s, p) => s + p[1], 0) / n] };
  }
  return { area: Math.abs(twice) / 2, at: [cx / (3 * twice), cy / (3 * twice)] };
}

/** An object's or an unexplored area's outline: its `poly`, its `rect` (x, y, width, height), or a box of `size`
 *  centred on `at`. */
function outline(value: Record<string, unknown>): Pt[] {
  const poly = points(value.poly);
  if (poly.length >= 3) return poly;
  if (Array.isArray(value.rect) && value.rect.length === 4) {
    const [x, y, w, h] = value.rect.map(num);
    if (x !== undefined && y !== undefined && w !== undefined && h !== undefined)
      return [
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
      ];
  }
  const at = point(value.at);
  const size = point(value.size);
  if (!at || !size) return [];
  const [hw, hd] = [size[0] / 2, size[1] / 2];
  return [
    [at[0] - hw, at[1] - hd],
    [at[0] + hw, at[1] - hd],
    [at[0] + hw, at[1] + hd],
    [at[0] - hw, at[1] + hd],
  ];
}

/** The plan of `scene` as an SVG titled `name`, or null for a scene with nothing to draw. */
export function drawScene(scene: Scene, name: string): SceneDrawing | null {
  const storey = level(scene);
  if (!storey) return null;
  const zones = under(scene, storey, 'zone')
    .map(z => ({
      id: z.id.replace(/^zone_/, ''),
      name: typeof z.name === 'string' ? z.name : '',
      poly: points(z.polygon),
    }))
    .filter(z => z.poly.length >= 3);
  const walls = under(scene, storey, 'wall')
    .map(w => ({ node: w, a: point(w.start), b: point(w.end), thickness: num(w.thickness) ?? 0.1 }))
    .filter((w): w is { node: SceneNode; a: Pt; b: Pt; thickness: number } => w.a !== null && w.b !== null);
  const ours = records(scene);
  const cameras = spots(scene);
  const areas = Object.values(record(ours.unexplored))
    .map(v => outline(record(v)))
    .filter(p => p.length >= 3);
  const objects = Object.values(record(ours.objects))
    .map(v => ({
      label: typeof record(v).label === 'string' ? (record(v).label as string) : '',
      poly: outline(record(v)),
    }))
    .filter(o => o.poly.length >= 3);
  const all: Pt[] = [
    ...zones.flatMap(z => z.poly),
    ...walls.flatMap(w => [w.a, w.b]),
    ...Object.values(cameras).map(c => c.at),
    ...areas.flat(),
  ];
  if (all.length === 0) return null;
  const minX = Math.min(...all.map(p => p[0])) - MARGIN;
  const minY = Math.min(...all.map(p => p[1])) - MARGIN;
  const width = (Math.max(...all.map(p => p[0])) + MARGIN - minX) * PX_PER_M;
  const height = (Math.max(...all.map(p => p[1])) + MARGIN - minY) * PX_PER_M;
  const toScreen = ([x, y]: Pt): Pt => [(x - minX) * PX_PER_M, (y - minY) * PX_PER_M];
  const xy = (p: Pt) =>
    toScreen(p)
      .map(v => v.toFixed(1))
      .join(',');
  const ring = (poly: Pt[]) => poly.map(xy).join(' ');
  const parts: string[] = [];
  for (const poly of areas) parts.push(`<polygon class="unexplored" points="${ring(poly)}"/>`);
  for (const zone of zones)
    parts.push(`<polygon class="room" data-room="${esc(zone.id)}" points="${ring(zone.poly)}"/>`);
  for (const object of objects)
    parts.push(`<polygon class="object" points="${ring(object.poly)}"><title>${esc(object.label)}</title></polygon>`);
  for (const wall of walls) {
    const [a, b] = [toScreen(wall.a), toScreen(wall.b)];
    const stroke = (wall.thickness * PX_PER_M).toFixed(1);
    parts.push(
      `<line class="wall" x1="${a[0].toFixed(1)}" y1="${a[1].toFixed(1)}" x2="${b[0].toFixed(1)}" y2="${b[1].toFixed(1)}" stroke-width="${stroke}"/>`,
    );
  }
  // openings go over the walls: a gap in the wall, and a thin line across a window's
  for (const wall of walls) {
    const length = Math.hypot(wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]);
    if (length === 0) continue;
    const at = (d: number): Pt => [
      wall.a[0] + ((wall.b[0] - wall.a[0]) * d) / length,
      wall.a[1] + ((wall.b[1] - wall.a[1]) * d) / length,
    ];
    const gap = ((wall.thickness + 0.04) * PX_PER_M).toFixed(1);
    for (const type of ['door', 'window'] as const)
      for (const opening of under(scene, wall.node, type)) {
        const middle = Array.isArray(opening.position) ? (num(opening.position[0]) ?? length / 2) : length / 2;
        const half = (num(opening.width) ?? 0.9) / 2;
        const [from, to] = [at(Math.max(0, middle - half)), at(Math.min(length, middle + half))];
        const line = `x1="${xy(from).replace(',', '" y1="')}" x2="${xy(to).replace(',', '" y2="')}"`;
        parts.push(`<line class="gap" ${line} stroke-width="${gap}"/>`);
        if (type === 'window') parts.push(`<line class="window" ${line}/>`);
      }
  }
  for (const zone of zones) {
    const { area, at } = centre(zone.poly);
    const [x, y] = toScreen(at);
    if (zone.name)
      parts.push(
        `<text class="name" x="${x.toFixed(1)}" y="${y.toFixed(1)}">${esc(zone.name)}</text>` +
          `<text class="area" x="${x.toFixed(1)}" y="${(y + 16).toFixed(1)}">${area.toFixed(1)} m²</text>`,
      );
  }
  for (const [camera, spot] of Object.entries(cameras)) {
    const [x, y] = toScreen(spot.at);
    if (spot.yaw !== undefined) {
      const reach = VIEW_M * PX_PER_M;
      const half = (((spot.fov ?? DEFAULT_FOV) / 2) * Math.PI) / 180;
      const facing = (spot.yaw * Math.PI) / 180;
      // a bearing clockwise from up the page: 0 looks up (y falls), 90 looks right
      const ray = (angle: number) =>
        `${(x + Math.sin(angle) * reach).toFixed(1)},${(y - Math.cos(angle) * reach).toFixed(1)}`;
      const wide = half * 2 > Math.PI ? 1 : 0;
      parts.push(
        `<path class="view" d="M${x.toFixed(1)},${y.toFixed(1)} L${ray(facing - half)} A${reach},${reach} 0 ${wide} 1 ${ray(facing + half)} Z"/>`,
      );
    }
    parts.push(
      `<circle class="camera" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5"><title>${esc(camera)}</title></circle>`,
    );
  }
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" class="scene-plan" role="img" aria-label="Floor plan of ${esc(name)}" ` +
    `width="${width.toFixed(0)}" height="${height.toFixed(0)}" viewBox="0 0 ${width.toFixed(1)} ${height.toFixed(1)}">${parts.join('')}</svg>`;
  return {
    svg,
    surface: { toScreen, rooms: zones },
    rooms: zones.length,
    area: zones.reduce((sum, zone) => sum + centre(zone.poly).area, 0),
    cameras,
    unexplored: areas.length > 0,
  };
}
