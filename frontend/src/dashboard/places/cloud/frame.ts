// The viewer's frame. SuperSplat's viewer (and its orbit, fly, and walk cameras) takes +y as up, while a place's
// map has whatever up its floor says (a levelled map's is +z). Everything the view hands the viewer goes through
// one rotation, the smallest that takes the map's up to +y: the splat and the overlays as their entities'
// rotation, and the cameras' positions as points. Plain maths, so it's tested without a canvas.
import type { V3 } from './geometry';

/** A rotation as a unit quaternion (x, y, z, w), the order PlayCanvas takes. */
export type Quat = [number, number, number, number];

/** The smallest rotation taking the unit vector `up` to +y. */
export function toViewerUp(up: V3): Quat {
  const [x, y, z] = up;
  // about up x (0, 1, 0) = (-z, 0, x), by the angle between them
  const sin = Math.hypot(x, z);
  if (sin < 1e-9) return y >= 0 ? [0, 0, 0, 1] : [1, 0, 0, 0]; // already up, or upside down: a half turn about x
  const half = Math.atan2(sin, y) / 2;
  const s = Math.sin(half) / sin;
  return [-z * s, 0, x * s, Math.cos(half)];
}

/** `p` turned by the rotation `q`. */
export function turned(p: V3, q: Quat): V3 {
  const [x, y, z, w] = q;
  // p + 2 w (u x p) + 2 u x (u x p), with u the quaternion's vector part
  const cx = y * p[2] - z * p[1];
  const cy = z * p[0] - x * p[2];
  const cz = x * p[1] - y * p[0];
  return [
    p[0] + 2 * (w * cx + y * cz - z * cy),
    p[1] + 2 * (w * cy + z * cx - x * cz),
    p[2] + 2 * (w * cz + x * cy - y * cx),
  ];
}
