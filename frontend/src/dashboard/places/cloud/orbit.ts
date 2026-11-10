// The 3D view's orbit camera, as plain vector maths so it's tested without a canvas: where the eye goes when the
// pointer turns it about its target, moves in or out, or slides both sideways. The map's up is whatever the place's
// floor says, so nothing here assumes an axis.
import type { V3 } from './geometry';

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: V3): V3 => scale(a, 1 / (length(a) || 1));

/** `v` turned by `angle` (radians) about the unit `axis` (Rodrigues). */
function about(v: V3, axis: V3, angle: number): V3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1 - c)));
}

/** The nearest the eye comes to looking straight down or up at its target (radians from the up axis). */
const POLE = 0.02;

/** The eye after a drag of (`yaw`, `pitch`) radians about `target`: around `up`, and over the top toward it, never
 *  past the pole (so the view can't flip). */
export function turn(eye: V3, target: V3, up: V3, yaw: number, pitch: number): V3 {
  const axis = unit(up);
  let arm = about(sub(eye, target), axis, yaw);
  const side = cross(axis, arm);
  if (length(side) > 1e-9) {
    const polar = Math.acos(Math.max(-1, Math.min(1, dot(unit(arm), axis))));
    const wanted = Math.max(POLE, Math.min(Math.PI - POLE, polar + pitch));
    arm = about(arm, unit(side), wanted - polar);
  }
  return add(target, arm);
}

/** The eye moved toward its target (`factor` under 1) or away from it (over 1), never onto it. */
export function dolly(eye: V3, target: V3, factor: number): V3 {
  const arm = sub(eye, target);
  const reach = Math.max(0.02, length(arm) * factor);
  return add(target, scale(unit(arm), reach));
}

/** The eye and its target slid together across the view: `right` and `down` in metres at the target's distance. */
export function slide(eye: V3, target: V3, up: V3, right: number, down: number): { eye: V3; target: V3 } {
  const forward = unit(sub(target, eye));
  const across = unit(cross(forward, unit(up)));
  const over = cross(across, forward);
  const move = add(scale(across, -right), scale(over, down));
  return { eye: add(eye, move), target: add(target, move) };
}
