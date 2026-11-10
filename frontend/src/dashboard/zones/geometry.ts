// Zone polygons in normalized frame coordinates (0 to 1, x right, y down), checked the way see/zones.py checks
// them on save, so the editor can say what's wrong before asking the server.

export type Point = [number, number];

export const MAX_POINTS = 32;
export const MIN_AREA = 1e-4;

/** The corner a click at (clientX, clientY) puts on a picture occupying `rect`, clamped to the frame and rounded. */
export function pointAt(rect: { left: number; top: number; width: number; height: number }, x: number, y: number) {
  const clamp = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 1e5) / 1e5;
  return [clamp((x - rect.left) / rect.width), clamp((y - rect.top) / rect.height)] as Point;
}

/** The polygon's area as a share of the frame. */
export function area(polygon: Point[]): number {
  let twice = 0;
  for (let i = 0; i < polygon.length; i++) {
    const [x1, y1] = polygon[i] as Point;
    const [x2, y2] = polygon[(i + 1) % polygon.length] as Point;
    twice += x1 * y2 - x2 * y1;
  }
  return Math.abs(twice) / 2;
}

const cross = (o: Point, a: Point, b: Point) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

/** True when no two edges that don't share a corner cross. */
export function simple(polygon: Point[]): boolean {
  const n = polygon.length;
  const edge = (i: number): [Point, Point] => [polygon[i] as Point, polygon[(i + 1) % n] as Point];
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      if (j === i + 1 || (i === 0 && j === n - 1)) continue;
      const [p1, p2] = edge(i);
      const [q1, q2] = edge(j);
      if (cross(q1, q2, p1) * cross(q1, q2, p2) < 0 && cross(p1, p2, q1) * cross(p1, p2, q2) < 0) return false;
    }
  return true;
}

/** Why the polygon can't be a zone yet, or null when it can. */
export function problem(polygon: Point[]): string | null {
  if (polygon.length < 3) return 'Click at least three corners.';
  if (polygon.length > MAX_POINTS) return `A zone has at most ${MAX_POINTS} corners.`;
  if (!simple(polygon)) return 'The outline crosses itself.';
  if (area(polygon) < MIN_AREA) return 'The zone is too small.';
  return null;
}

/** Where a zone's label goes: the average of its corners. */
export function centre(polygon: Point[]): Point {
  const n = polygon.length || 1;
  return [polygon.reduce((s, p) => s + p[0], 0) / n, polygon.reduce((s, p) => s + p[1], 0) / n];
}
