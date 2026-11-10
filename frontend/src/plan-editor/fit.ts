// Fitting the plan to the 2D canvas clear of the editor's floating controls. The editor frames a box by centring it
// and making the view as wide as the box plus EDITOR_PAD_M on each side (or as tall, for the canvas's shape); it has
// no way to be told that its level card covers the canvas's top and its tool bar the bottom. So the box it is given
// is worked out backwards: the view that puts the plan in the canvas less those bands, as a box the editor turns
// into that view.

type Vec3 = [number, number, number];
export interface Bounds {
  min: Vec3;
  max: Vec3;
}

/** The padding the editor adds around a box it frames (FLOORPLAN_PADDING in its floorplan panel), metres. */
export const EDITOR_PAD_M = 2;
/** Room left around the plan inside the clear part of the canvas, metres. */
export const MARGIN_M = 0.5;
/** The canvas's bands under the editor's controls, pixels: its level card at the top, its tool bar at the bottom. */
export const CLEAR_TOP_PX = 84;
export const CLEAR_BOTTOM_PX = 84;

/** The box to frame so the plan (`plan`, x and z in metres) sits in a `width` by `height` px canvas, clear of
 *  `top` and `bottom` px bands. Null when the canvas has no clear part, and the plain plan should be framed. */
export function framing(
  plan: Bounds,
  width: number,
  height: number,
  top = CLEAR_TOP_PX,
  bottom = CLEAR_BOTTOM_PX,
): Bounds | null {
  const clear = height - top - bottom;
  if (width <= 0 || clear <= 0) return null;
  const across = plan.max[0] - plan.min[0] + MARGIN_M * 2;
  const down = plan.max[2] - plan.min[2] + MARGIN_M * 2;
  const scale = Math.min(width / across, clear / down); // px a metre
  const view = width / scale; // the view's width, metres
  // The box's width that the editor pads out to that view. It is flat, so its padded height doesn't set the view;
  // for a plan so small that it would, the editor shows a wider view on the same centre, the plan still clear.
  const box = Math.max(view - EDITOR_PAD_M * 2, 0.01);
  const x = (plan.min[0] + plan.max[0]) / 2;
  // the plan's centre is at the clear part's centre, (top - bottom) / 2 px below the canvas's
  const z = (plan.min[2] + plan.max[2]) / 2 - (top - bottom) / 2 / scale;
  return { min: [x - box / 2, plan.min[1], z], max: [x + box / 2, plan.max[1], z] };
}
