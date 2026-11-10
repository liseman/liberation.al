import { expect, it } from 'vitest';
import { type Bounds, EDITOR_PAD_M, framing, MARGIN_M } from './fit';

/** Where the editor puts a framed box: centred, the view as wide as the box plus its padding, or as tall for the
 *  canvas's shape (its floorplan panel's frame handler). Returns px a metre and the box's centre, in metres. */
function editorView(box: Bounds, width: number, height: number) {
  const view = Math.max(
    box.max[0] - box.min[0] + EDITOR_PAD_M * 2,
    (box.max[2] - box.min[2] + EDITOR_PAD_M * 2) * (width / height),
  );
  return { scale: width / view, x: (box.min[0] + box.max[0]) / 2, z: (box.min[2] + box.max[2]) / 2 };
}

/** The plan's edges on the canvas, px, once the editor frames `box`. */
function onCanvas(plan: Bounds, box: Bounds, width: number, height: number) {
  const { scale, x, z } = editorView(box, width, height);
  return {
    left: width / 2 + (plan.min[0] - x) * scale,
    right: width / 2 + (plan.max[0] - x) * scale,
    top: height / 2 + (plan.min[2] - z) * scale,
    bottom: height / 2 + (plan.max[2] - z) * scale,
  };
}

const HOUSE: Bounds = { min: [0, 0, 0], max: [9, 2.4, 8.5] };

it.each([
  ['a wide canvas', 900, 480],
  ['a tall one', 480, 900],
  ['the 1440 editor', 640, 490],
])('puts the plan clear of the controls, a margin inside, on %s', (_, width, height) => {
  const box = framing(HOUSE, width, height, 84, 84);
  expect(box).not.toBeNull();
  const at = onCanvas(HOUSE, box as Bounds, width, height);
  expect(at.top).toBeGreaterThanOrEqual(84 - 0.01);
  expect(at.bottom).toBeLessThanOrEqual(height - 84 + 0.01);
  expect(at.left).toBeGreaterThanOrEqual(-0.01);
  expect(at.right).toBeLessThanOrEqual(width + 0.01);
  // and as large as that allows: one pair of edges sits a margin inside the clear part
  const { scale } = editorView(box as Bounds, width, height);
  const slack = Math.min(at.left, width - at.right, at.top - 84, height - 84 - at.bottom);
  expect(slack).toBeCloseTo(MARGIN_M * scale);
});

it('moves the plan off a deeper band at the top', () => {
  const at = onCanvas(HOUSE, framing(HOUSE, 900, 600, 150, 50) as Bounds, 900, 600);
  expect(at.top).toBeGreaterThanOrEqual(150 - 0.01);
  expect(at.bottom).toBeLessThanOrEqual(550 + 0.01);
});

it('frames the plain plan when the canvas has no clear part', () => {
  expect(framing(HOUSE, 600, 160, 84, 84)).toBeNull();
  expect(framing(HOUSE, 0, 600)).toBeNull();
});
