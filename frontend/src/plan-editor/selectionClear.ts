// What the editor shows for a selection stays off the selection. With a room or a wall picked, the editor floats
// its action bar (with a room's Floor height under it) just above the selection, and its inspector (the picked
// thing's name and kind) at the canvas's top right. Here the action bar is pinned under the inspector, in the same
// right-hand column (plan-editor.css places it from --plan-actions-top), and when the selection is then under any
// of the canvas's controls (the level card, that column, or the tool bar) the plan is panned, at the zoom it has,
// until the selection is clear, or as nearly as the room the controls leave allows. Nothing of the editor is
// patched: the pin is a style, and the pan is the editor's own frame request with a target and no box.
import { emitter } from '@pascal-app/core';
import { useEditor } from '@pascal-app/editor';
import { useViewer } from '@pascal-app/viewer';

export interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** The room kept between the selection and a control, px. */
export const GAP_PX = 8;
/** When to look again after a selection, ms: once its controls are up, and once the editor's glide has ended. */
const SETTLE_MS = [60, 600];

const moved = (r: Rect, dx: number, dy: number): Rect => ({
  left: r.left + dx,
  right: r.right + dx,
  top: r.top + dy,
  bottom: r.bottom + dy,
});

/** How much of `a` lies under `b`, px². */
const covered = (a: Rect, b: Rect): number =>
  Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) *
  Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));

/** The move (px) that takes `selection` out from under `blocks` and keeps it inside `canvas`: the smallest that
 *  clears it, or, when none can (a selection bigger than the room the controls leave), the one that leaves the
 *  least of it covered. Null when it is clear already, or when no move does better than staying. */
export function clearing(selection: Rect, blocks: Rect[], canvas: Rect): { dx: number; dy: number } | null {
  const under = (r: Rect) => blocks.reduce((sum, b) => sum + covered(r, b), 0);
  const now = under(selection);
  if (now === 0) return null;
  // each edge a block offers, on its own axis, and every pair of them across the two
  const dxs = [0, ...blocks.flatMap(b => [b.right + GAP_PX - selection.left, b.left - GAP_PX - selection.right])];
  const dys = [0, ...blocks.flatMap(b => [b.bottom + GAP_PX - selection.top, b.top - GAP_PX - selection.bottom])];
  let best: { dx: number; dy: number; under: number; far: number } | null = null;
  for (const dx of dxs)
    for (const dy of dys) {
      const r = moved(selection, dx, dy);
      if (r.left < canvas.left || r.right > canvas.right || r.top < canvas.top || r.bottom > canvas.bottom) continue;
      const left = under(r);
      const far = Math.hypot(dx, dy);
      if (!best || left < best.under || (left === best.under && far < best.far)) best = { dx, dy, under: left, far };
    }
  return best && best.under < now ? { dx: best.dx, dy: best.dy } : null;
}

const rectOf = (el: Element | null): Rect | null => {
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0 ? { left: r.left, top: r.top, right: r.right, bottom: r.bottom } : null;
};

/** The editor's inspector: the right-hand column's first part, when it shows something. */
const inspector = () => rectOf(document.querySelector('[data-right-stack] > :first-child'));

/** The selection's outline on screen, from the plan's own drawing of each picked node. */
function selectionRect(): Rect | null {
  // what the viewer has picked, and the room the editor has (a room is the editor's own selection, not the viewer's)
  const { selectedIds, zoneId } = useViewer.getState().selection;
  const room = useEditor.getState().room?.zoneId;
  const rects = [...selectedIds, ...[zoneId, room].filter((id): id is string => Boolean(id))]
    .flatMap(id => [...document.querySelectorAll(`g[data-floorplan-scene] [data-node-id="${CSS.escape(id)}"]`)])
    .map(rectOf)
    .filter((r): r is Rect => r !== null);
  if (!rects.length) return null;
  return {
    left: Math.min(...rects.map(r => r.left)),
    top: Math.min(...rects.map(r => r.top)),
    right: Math.max(...rects.map(r => r.right)),
    bottom: Math.max(...rects.map(r => r.bottom)),
  };
}

/** Pan the plan, at its zoom, so the selection is clear of the canvas's controls. */
function keepClear(): void {
  const scene = document.querySelector<SVGGraphicsElement>('g[data-floorplan-scene]');
  const canvas = rectOf(scene?.ownerSVGElement ?? null);
  const selection = selectionRect();
  const matrix = scene?.getScreenCTM();
  if (!(scene && canvas && selection && matrix)) return;
  const blocks = [
    rectOf(document.querySelector('[data-surface="level-selector"]')),
    rectOf(document.querySelector('[data-surface="tool-dock"]')),
    inspector(),
    rectOf(document.querySelector('[data-plan-actions]')),
  ].filter((r): r is Rect => r !== null);
  const move = clearing(selection, blocks, canvas);
  if (!move) return;
  // the view's centre moved the other way, in the plan's coordinates (the scene's metres: a plan's building sits at
  // the origin), carries the selection by (dx, dy) on screen
  const shifted = new DOMPoint(
    (canvas.left + canvas.right) / 2 - move.dx,
    (canvas.top + canvas.bottom) / 2 - move.dy,
  ).matrixTransform(matrix.inverse());
  emitter.emit('camera-controls:frame', {
    bounds: null,
    pose: { position: [shifted.x, 10, shifted.y], target: [shifted.x, 0, shifted.y] },
  });
}

/** Keep the action bar under the inspector and the selection clear of the controls, from now on. */
export function installSelectionClear(): void {
  // the action bar's place: under the inspector, or at the column's top without one
  let queued = false;
  const place = () => {
    queued = false;
    const top = inspector()?.bottom ?? rectOf(document.querySelector('[data-right-stack]'))?.top;
    if (top !== undefined) document.documentElement.style.setProperty('--plan-actions-top', `${top + GAP_PX}px`);
    // the editor's floating action bar, marked so plan-editor.css can pin it (its look and its anchor are its own)
    for (const el of document.querySelectorAll('div.fixed.z-30.w-max[style*="calc(-100% - 32px)"]'))
      el.setAttribute('data-plan-actions', '');
  };
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(place);
  }).observe(document.body, { childList: true, subtree: true });
  // A new selection: move it clear once its controls are drawn, and again once that move has settled (the editor
  // glides there, and the action bar can arrive a little after the selection).
  const picked = () => {
    for (const wait of SETTLE_MS)
      setTimeout(() => {
        place();
        keepClear();
      }, wait);
  };
  useViewer.subscribe((state, before) => {
    if (state.selection !== before.selection) picked();
  });
  useEditor.subscribe((state, before) => {
    if (state.room !== before.room && state.room) picked();
  });
}
