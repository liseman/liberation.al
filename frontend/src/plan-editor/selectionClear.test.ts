import { expect, it, vi } from 'vitest';

// the module subscribes to the editor's stores only when installed; these tests use its geometry alone
vi.mock('@pascal-app/core', () => ({ emitter: { emit: vi.fn() } }));
vi.mock('@pascal-app/editor', () => ({ useEditor: { getState: vi.fn(), subscribe: vi.fn() } }));
vi.mock('@pascal-app/viewer', () => ({ useViewer: { getState: vi.fn(), subscribe: vi.fn() } }));

const { clearing, GAP_PX } = await import('./selectionClear');

const box = (left: number, top: number, right: number, bottom: number) => ({ left, top, right, bottom });
const CANVAS = box(0, 0, 640, 490);
// the level card at the top left, the inspector and the action bar in the right-hand column, the tool bar below
const CONTROLS = [box(12, 12, 173, 71), box(306, 80, 626, 150), box(413, 158, 626, 238), box(133, 411, 509, 469)];

it('leaves a selection that nothing covers where it is', () => {
  expect(clearing(box(200, 250, 300, 350), CONTROLS, CANVAS)).toBeNull();
});

it('moves a selection under the inspector the least way out, a gap clear of it and of everything else', () => {
  const move = clearing(box(330, 100, 400, 180), CONTROLS, CANVAS);
  // under the inspector (beside the action bar, which ends left of it) is nearer than left of the inspector, and
  // above it is off the canvas
  expect(move).toEqual({ dx: 0, dy: 150 + GAP_PX - 100 });
});

it('takes the move that leaves the least covered when no move clears a big selection', () => {
  // taller than the room between the right-hand column and the tool bar, wider than the room left of the column
  const selection = box(320, 100, 500, 290);
  const move = clearing(selection, CONTROLS, CANVAS);
  expect(move).not.toBeNull();
  const covered = (dx: number, dy: number) =>
    CONTROLS.reduce((sum, b) => {
      const w = Math.min(selection.right + dx, b.right) - Math.max(selection.left + dx, b.left);
      const h = Math.min(selection.bottom + dy, b.bottom) - Math.max(selection.top + dy, b.top);
      return sum + Math.max(0, w) * Math.max(0, h);
    }, 0);
  expect(covered(move?.dx ?? 0, move?.dy ?? 0)).toBeLessThan(covered(0, 0));
});

it('never moves a selection off the canvas', () => {
  // the only way out from under the tool bar would be below the canvas's edge or up into the column: up it goes
  const move = clearing(box(200, 420, 260, 480), CONTROLS, CANVAS);
  expect(move).not.toBeNull();
  expect(480 + (move?.dy ?? 0)).toBeLessThanOrEqual(CANVAS.bottom);
  expect(420 + (move?.dy ?? 0)).toBeGreaterThanOrEqual(CANVAS.top);
});
