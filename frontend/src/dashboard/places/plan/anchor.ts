// The anchor (#2797): how a plan's coordinates map into its place's point-cloud frame, so things measured in the
// frame (the walked path, fixed-camera person dots, object positions) land in the right room, and back.
//
// Plan coordinates are metres with y growing south (down the page, the library's convention). The frame is
// right-handed with Z up. A plan point (x, y) goes to the frame by flipping y (north becomes +Y), scaling, rotating
// counterclockwise by yaw about Z, and adding the origin; Z is the floor's height, origin[2].
import type { Anchor, Pt } from './ext';

const rad = (deg: number) => (deg * Math.PI) / 180;

/** A plan point in the place's frame, on the floor. */
export function planToPlace(anchor: Anchor, [x, y]: Pt): [number, number, number] {
  const s = anchor.scale ?? 1;
  const c = Math.cos(rad(anchor.yaw));
  const n = Math.sin(rad(anchor.yaw));
  const u = s * x;
  const v = -s * y;
  const [ox, oy, oz] = anchor.origin;
  return [ox + c * u - n * v, oy + n * u + c * v, oz];
}

/** A point in the place's frame, dropped onto the plan (its height is ignored). */
export function placeToPlan(anchor: Anchor, [X, Y]: readonly [number, number, ...number[]]): Pt {
  const s = anchor.scale ?? 1;
  const c = Math.cos(rad(anchor.yaw));
  const n = Math.sin(rad(anchor.yaw));
  const dx = X - anchor.origin[0];
  const dy = Y - anchor.origin[1];
  const u = c * dx + n * dy;
  const v = -n * dx + c * dy;
  return [u / s, -v / s];
}
