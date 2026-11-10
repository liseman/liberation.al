// Paths on a plan (#2831): what to draw for a window of positions at the scrubber's time. Each path is drawn as one
// smooth curve per segment (#3296), with an arrowhead where it ends showing which way they went: its points are cleaned
// (a foot point that jumps out and straight back is dropped, and points closer than a step to the last are let go, so
// standing still isn't a scribble), split only where the track went quiet for minutes or jumped further than anyone
// could have walked, and joined by a centripetal Catmull-Rom spline, which never loops or overshoots between close
// points the way a uniform one does. A segment fades with its age. Pure, so the drawing (PlacePaths) and the tests
// share it.
import type { Entity, Point, RoomSpan } from '../api';

/** Path colours, one per entity in the order they're listed: distinct in light and dark, none the plan's own. */
export const COLOURS = ['#e8590c', '#1c7ed6', '#2f9e44', '#ae3ec9', '#f08c00', '#0c8599', '#e64980', '#5c940d'];

/** The colour of everyone not identified yet: one quiet grey, since a camera can see hundreds of short tracks a day. */
export const UNKNOWN_COLOUR = '#868e96';

/** A path is split where its track goes quiet this long (seconds): a few minutes, so a pause on the couch stays one
 *  path but coming back hours later starts a new one. */
export const SPLIT_GAP_S = 180;

/** A path is split where it moves faster than this (metres a second, a fast walk), give or take JUMP_SLACK_M of
 *  foot-point error: two places nobody could have walked between in the time, joined, would cross walls. A foot point
 *  put in the wrong room jumps metres in a second or two, and joined, draws a line through a wall; a dog's sprint
 *  split in two only costs an extra arrowhead, so this errs towards splitting. */
export const MAX_SPEED_MPS = 2.5;
export const JUMP_SLACK_M = 1;

/** A point closer than this (metres) to the last one kept is let go: feet shuffle by a few centimetres standing still. */
export const MIN_STEP_M = 0.3;

/** A lone point this far (metres) from the points either side of it, while those two are close together, is a foot
 *  point put in the wrong place (a missed foot, a reflection), dropped. */
export const SPIKE_M = 1;

/** Points the server sends a window at most per entity (dashboard/places.py MAX_POINTS): a long window keeps every
 *  step someone moved, and while they stand still, one point every span / READ_POINTS seconds. */
export const READ_POINTS = 2000;

/** The gap (seconds) a path splits at, and past which a dot is stale, for a window `span` long: SPLIT_GAP_S, or more
 *  for a window the server thins coarser than that (a week is one point every ~5 min), so its paths don't split at
 *  every point and its dots don't all go. */
export const gapFor = (span: number): number => Math.max(SPLIT_GAP_S, 1.5 * (span / READ_POINTS));

/** The faintest a path gets, at the start of the window. */
export const FAINTEST = 0.15;

/** How opaque a point `age` seconds older than the scrubber is, over a window `span` seconds long. */
export function fade(age: number, span: number): number {
  if (span <= 0) return 1;
  return Math.max(FAINTEST, 1 - (1 - FAINTEST) * Math.min(1, Math.max(0, age / span)));
}

const far = (a: Point, b: Point): number => Math.hypot(b.x - a.x, b.y - a.y);

/** Whether someone could have got from `a` to `b` in the time between them. */
export const walkable = (a: Point, b: Point): boolean =>
  far(a, b) <= MAX_SPEED_MPS * Math.abs(b.t - a.t) + JUMP_SLACK_M;

/** `points` without the lone foot points that jump out and straight back (SPIKE_M). The first and last are kept. */
export function despiked(points: Point[]): Point[] {
  const out: Point[] = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i] as Point;
    const before = out.at(-1);
    const after = points[i + 1];
    if (before && after) {
      const out1 = far(before, p);
      const back = far(p, after);
      if (out1 > SPIKE_M && back > SPIKE_M && far(before, after) < Math.min(out1, back) / 2) continue;
    }
    out.push(p);
  }
  return out;
}

/** `points` with each one closer than MIN_STEP_M to the last kept let go; the last point always stays (it's where they
 *  are), in place of the one kept before it when the two are that close. */
export function stepped(points: Point[]): Point[] {
  const out: Point[] = [];
  for (const p of points) {
    const last = out.at(-1);
    if (!last || far(last, p) >= MIN_STEP_M) out.push(p);
  }
  const end = points.at(-1);
  if (end && out.at(-1) !== end) {
    if (out.length > 1) out[out.length - 1] = end;
    else out.push(end);
  }
  return out;
}

/** A track's points up to `at`, cleaned (despiked, stepped) and split where it went quiet longer than `gap` seconds or
 *  jumped further than anyone could walk: each segment has two points or more (a lone point draws no line). */
export function segments(points: Point[], at: number, gap = SPLIT_GAP_S): Point[][] {
  const quiet: Point[][] = [];
  let current: Point[] = [];
  for (const p of points) {
    if (p.t > at) break;
    const last = current.at(-1);
    if (last && p.t - last.t > gap) {
      quiet.push(current);
      current = [];
    }
    current.push(p);
  }
  if (current.length) quiet.push(current);
  const out: Point[][] = [];
  for (const piece of quiet) {
    let run: Point[] = [];
    for (const p of despiked(piece)) {
      const last = run.at(-1);
      if (last && !walkable(last, p)) {
        out.push(run);
        run = [];
      }
      run.push(p);
    }
    out.push(run);
  }
  return out.map(stepped).filter(s => s.length >= 2);
}

export type XY = [number, number];

/** The most of a curve's length its arrowhead takes. */
export const ARROW_SHARE = 0.6;

/** One cubic Bezier piece: its start, two control points, and end. */
export type Cubic = [XY, XY, XY, XY];

/** The centripetal Catmull-Rom spline through `pts` (alpha 1/2), as cubic Bezier pieces, one between each two points.
 *  The ends are given phantom neighbours mirrored through them, so a path starts and ends heading along its first
 *  and last step. Beziers keep their shape under the plan's projection (a similarity), so the pieces can be made in
 *  metres and their points moved onto the drawing. */
export function spline(pts: XY[]): Cubic[] {
  const out: Cubic[] = [];
  const at = (i: number): XY => {
    if (i < 0) {
      const [a, b] = [pts[0] as XY, pts[1] as XY];
      return [2 * a[0] - b[0], 2 * a[1] - b[1]];
    }
    if (i >= pts.length) {
      const [a, b] = [pts[pts.length - 1] as XY, pts[pts.length - 2] as XY];
      return [2 * a[0] - b[0], 2 * a[1] - b[1]];
    }
    return pts[i] as XY;
  };
  const dist = (a: XY, b: XY) => Math.hypot(b[0] - a[0], b[1] - a[1]);
  for (let i = 0; i + 1 < pts.length; i++) {
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)];
    const [d1, d2, d3] = [dist(p0, p1), dist(p1, p2), dist(p2, p3)];
    const [r1, r2, r3] = [Math.sqrt(d1), Math.sqrt(d2), Math.sqrt(d3)];
    const ctrl = (from: XY, to: XY, away: XY, dn: number, rn: number): XY => {
      // Yuksel, Schaefer, and Keyser (2011): the Bezier control point of a centripetal Catmull-Rom piece
      if (rn < 1e-9 || r2 < 1e-9) return from;
      const k = 2 * dn + 3 * rn * r2 + d2;
      const den = 3 * rn * (rn + r2);
      return [(dn * to[0] - d2 * away[0] + k * from[0]) / den, (dn * to[1] - d2 * away[1] + k * from[1]) / den];
    };
    out.push([p1, ctrl(p1, p2, p0, d1, r1), ctrl(p2, p1, p3, d3, r3), p2]);
  }
  return out;
}

/** An SVG path's `d` for `pieces`, each point to one decimal (pixels). */
export function pathData(pieces: Cubic[]): string {
  const f = ([x, y]: XY) => `${x.toFixed(1)},${y.toFixed(1)}`;
  const first = pieces[0];
  if (!first) return '';
  return `M${f(first[0])}${pieces.map(([, a, b, c]) => `C${f(a)} ${f(b)} ${f(c)}`).join('')}`;
}

/** The arrowhead at the end of `pieces`: a triangle `size` long pointing the way the curve leaves its last point, as an
 *  SVG polygon's points; '' for no curve. On a short curve it's at most ARROW_SHARE of the curve's length, so it never
 *  hides the line it ends. */
export function arrowHead(pieces: Cubic[], most: number): string {
  const last = pieces.at(-1);
  if (!last) return '';
  const length = pieces.reduce((sum, [a, , , b]) => sum + Math.hypot(b[0] - a[0], b[1] - a[1]), 0);
  const size = Math.min(most, ARROW_SHARE * length);
  const [start, , c2, tip] = last;
  let [dx, dy] = [tip[0] - c2[0], tip[1] - c2[1]];
  if (Math.hypot(dx, dy) < 1e-9) [dx, dy] = [tip[0] - start[0], tip[1] - start[1]];
  const len = Math.hypot(dx, dy);
  if (len < 1e-9) return '';
  const [ux, uy] = [dx / len, dy / len];
  const back: XY = [tip[0] - ux * size, tip[1] - uy * size];
  const w = size * 0.55;
  const pts: XY[] = [tip, [back[0] - uy * w, back[1] + ux * w], [back[0] + uy * w, back[1] - ux * w]];
  return pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
}

/** Where an entity was at `at`: its last point at or before it, if that's recent enough to still say so (within `gap`,
 *  where a path would split too: gapFor the window); else null. A track that ended an hour before the scrubber has no dot:
 *  live, a day's 1,000-odd short tracks each left one where it ended, all over the plan. */
export function latest(points: Point[], at: number, gap = SPLIT_GAP_S): Point | null {
  let found: Point | null = null;
  for (const p of points) {
    if (p.t > at) break;
    found = p;
  }
  return found && at - found.t <= gap ? found : null;
}

/** The room-only spans that cover `at`, their end included: a visit still going on ends at the window's end, where
 *  the scrubber starts. */
export const roomsAt = (rooms: RoomSpan[], at: number): RoomSpan[] => rooms.filter(r => r.start <= at && at <= r.end);

/** Each entity's colour: the people and pets the house knows by their place in the list, everyone else grey. */
export function colours(entities: Entity[]): Map<string, string> {
  const known = entities.filter(e => e.kind !== 'unknown');
  return new Map([
    ...known.map((e, i) => [e.id, COLOURS[i % COLOURS.length] as string] as const),
    ...entities.filter(e => e.kind === 'unknown').map(e => [e.id, UNKNOWN_COLOUR] as const),
  ]);
}

/** What to call an entity: its name, else what it is ("Someone", "A pet"). */
export function label(e: Entity): string {
  if (e.name) return e.name;
  return e.kind === 'pet' ? 'A pet' : e.kind === 'rig' ? 'The rig' : 'Someone';
}

/** The plan's room id a room-only span belongs on: its own, else the plan room whose name matches the zone's. */
export function planRoom(span: RoomSpan, rooms: { id: string; name: string }[]): string | null {
  if (span.room) return span.room;
  const name = span.name.trim().toLowerCase();
  return rooms.find(r => r.name.trim().toLowerCase() === name)?.id ?? null;
}

/** Pixels a metre on a plain frame: the plan library's own default scale, so a place with no plan reads at the size
 *  one with a plan does. */
export const PLAIN_SCALE = 40;

/** Metres of floor a plain frame leaves around the points, and the least it spans each way. */
export const PLAIN_MARGIN_M = 0.5;
export const PLAIN_MIN_M = 3;

/** Where a place with no plan yet draws its paths (#3169): plain floor metres around every point of the window, in
 *  the position layer's own floor frame (#3157), x to the right and y down as a plan's are. */
export interface PlainFrame {
  /** The floor point at the frame's top left, in metres. */
  minX: number;
  minY: number;
  /** The frame's size, in metres. */
  width: number;
  height: number;
}

/** The plain frame that holds every point of `tracks`, or null when there are none. */
export function plainFrame(tracks: { points: { x: number; y: number }[] }[]): PlainFrame | null {
  const xs = tracks.flatMap(t => t.points.map(p => p.x));
  const ys = tracks.flatMap(t => t.points.map(p => p.y));
  if (xs.length === 0) return null;
  const side = (lo: number, hi: number): [number, number] => {
    const size = Math.max(hi - lo + 2 * PLAIN_MARGIN_M, PLAIN_MIN_M);
    return [(lo + hi) / 2 - size / 2, size];
  };
  const [minX, width] = side(Math.min(...xs), Math.max(...xs));
  const [minY, height] = side(Math.min(...ys), Math.max(...ys));
  return { minX, minY, width, height };
}

/** A floor point's place on a plain frame's drawing, in pixels from its top left. */
export const plainToScreen = (f: PlainFrame, [x, y]: [number, number]): [number, number] => [
  (x - f.minX) * PLAIN_SCALE,
  (y - f.minY) * PLAIN_SCALE,
];
