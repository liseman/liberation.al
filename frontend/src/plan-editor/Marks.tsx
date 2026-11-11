// The objects and unexplored areas drawn in the editor's 2D plan: an object as a box with its label, an unexplored
// area as a dashed outline. In the editor's select mode the outline drags to move it and its corner handles drag to
// reshape it (focused, the arrow keys do the same: 5 cm or 50 cm a press, with Shift), and an object or area picked in
// the Marks panel is put where the plan is clicked. This is mounted inside the editor's own plan
// (`floorplanSceneSlot`), whose coordinates are the plan's metres, so nothing here knows the zoom or the turn of the
// view. A change is one write to the level's metadata when the pointer lifts (useAreas.ts), so it undoes and saves
// like any other edit.
import { useEditor } from '@pascal-app/editor';
import { type KeyboardEvent, type PointerEvent, useRef, useState } from 'react';
import { type Area, box, middleOf, newId, reshaped, translated } from './areas';
import { isSelected, useAreasUi } from './areasUi';
import { useSpotsUi } from './spotsUi';
import { useAreas } from './useAreas';

/** The arrow keys' steps, metres: Shift takes the larger. */
const MOVE_M = [0.05, 0.5] as const;
/** Sizes on the plan, metres. */
const CORNER_M = 0.12;
const LABEL_M = 0.32;
/** The dashboard's accent, the plan's ground, and the dimmed colour an object or area is drawn in
 *  (plan-editor.css), as styles: an SVG attribute can't name a variable. */
const ACCENT = 'var(--plan-accent)';
const GROUND = 'var(--plan-ground)';
const OUTLINE = 'var(--mantine-color-dimmed)';

const arrow = (event: KeyboardEvent): [number, number] | null =>
  (
    ({ ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }) as Record<
      string,
      [number, number]
    >
  )[event.key] ?? null;

const ring = (poly: Area['poly']): string => poly.map(([x, y]) => `${x},${y}`).join(' ');

/** A drag in progress: the area as the pointer leaves it, and the corner being dragged (null when the whole moves). */
interface Drag {
  area: Area;
  corner: number | null;
}

/** `locked`: draw the objects and areas without letting them be moved, reshaped, or added (a scene opened to look at). */
export function Marks({ locked = false }: { locked?: boolean }) {
  const { areas, set } = useAreas();
  const selecting = useEditor(state => state.mode === 'select') && !locked;
  const { selected, adding, select, setAdding } = useAreasUi();
  const addingCamera = useSpotsUi(state => state.adding !== null);
  const group = useRef<SVGGElement>(null);
  const from = useRef<[number, number]>([0, 0]);
  const [drag, setDrag] = useState<Drag | null>(null);

  /** Where a pointer is on the plan, metres. */
  const planPoint = (event: PointerEvent): [number, number] | null => {
    const matrix = group.current?.getScreenCTM();
    if (!matrix) return null;
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return [p.x, p.y];
  };
  /** Begin a drag: from the pointer on the whole area, or from a corner to reshape it. */
  const start = (event: PointerEvent, area: Area, corner: number | null) => {
    const at = planPoint(event);
    if (!at) return;
    event.stopPropagation(); // the plan under it would start a selection box
    event.currentTarget.setPointerCapture(event.pointerId);
    from.current = corner === null ? at : (area.poly[corner] ?? at);
    select(area.kind, area.id);
  };
  const move = (event: PointerEvent, area: Area, corner: number | null) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const at = planPoint(event);
    if (!at) return;
    const poly =
      corner === null
        ? translated(area.poly, [at[0] - from.current[0], at[1] - from.current[1]])
        : reshaped(area.poly, corner, at);
    setDrag({ area: { ...area, poly }, corner });
  };
  const end = (event: PointerEvent) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
    if (drag) set(drag.area.kind, drag.area.id, drag.area);
    setDrag(null);
  };
  /** The arrow keys on a mark: the whole area by a step, or the focused corner. */
  const nudge = (event: KeyboardEvent, area: Area, corner: number | null) => {
    const d = arrow(event);
    if (!d) return;
    event.preventDefault();
    event.stopPropagation();
    const by = MOVE_M[event.shiftKey ? 1 : 0];
    const poly =
      corner === null
        ? translated(area.poly, [d[0] * by, d[1] * by])
        : reshaped(area.poly, corner, [
            (area.poly[corner]?.[0] ?? 0) + d[0] * by,
            (area.poly[corner]?.[1] ?? 0) + d[1] * by,
          ]);
    set(area.kind, area.id, { ...area, poly });
  };
  /** Put a new object or area where the plan was clicked. */
  const place = (event: PointerEvent) => {
    const at = planPoint(event);
    if (!at || !adding) return;
    event.stopPropagation();
    const area: Area = { id: newId(adding), kind: adding, label: '', poly: box(at) };
    set(area.kind, area.id, area);
    select(area.kind, area.id);
    setAdding(null);
  };
  const shownOf = (area: Area): Area => (drag?.area.kind === area.kind && drag.area.id === area.id ? drag.area : area);

  return (
    <g ref={group} data-plan-marks="" pointerEvents={selecting && !addingCamera ? undefined : 'none'}>
      {(['unexplored', 'object'] as const).flatMap(kind =>
        areas[kind].map(area => {
          const shown = shownOf(area);
          const on = isSelected(selected, kind, area.id);
          const name = area.label || kind;
          const [cx, cy] = middleOf(shown.poly);
          return (
            <g key={`${kind}:${area.id}`} data-area={`${kind}:${area.id}`}>
              {/* biome-ignore lint/a11y/useSemanticElements: a draggable mark on a drawing; no HTML control stands in for it */}
              <polygon
                role="button"
                tabIndex={selecting ? 0 : -1}
                aria-label={`${name}: move`}
                aria-pressed={on}
                points={ring(shown.poly)}
                strokeWidth={kind === 'unexplored' ? 1.5 : 1}
                strokeDasharray={kind === 'unexplored' ? '6 4' : undefined}
                vectorEffect="non-scaling-stroke"
                style={{
                  cursor: 'move',
                  fill: kind === 'unexplored' ? OUTLINE : ACCENT,
                  stroke: on ? ACCENT : OUTLINE,
                }}
                fillOpacity={kind === 'unexplored' ? (on ? 0.16 : 0.08) : 0}
                onPointerDown={event => start(event, shown, null)}
                onPointerMove={event => move(event, shown, null)}
                onPointerUp={end}
                onPointerCancel={() => setDrag(null)}
                onFocus={() => select(kind, area.id)}
                onKeyDown={event => nudge(event, shown, null)}
              >
                <title>{name}</title>
              </polygon>
              {area.label && (
                <text
                  x={cx}
                  y={cy}
                  fontSize={LABEL_M}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ fill: OUTLINE, pointerEvents: 'none' }}
                >
                  {area.label}
                </text>
              )}
              {on &&
                shown.poly.map(([x, y], i) => (
                  // biome-ignore lint/a11y/useSemanticElements: a draggable handle on a drawing; no HTML control stands in for it
                  <circle
                    key={`${x},${y}`}
                    data-corner={i}
                    role="button"
                    tabIndex={selecting ? 0 : -1}
                    aria-label={`${name}: corner ${i + 1}`}
                    cx={x}
                    cy={y}
                    r={CORNER_M}
                    strokeWidth={2}
                    vectorEffect="non-scaling-stroke"
                    style={{ cursor: 'grab', fill: GROUND, stroke: ACCENT }}
                    onPointerDown={event => start(event, shown, i)}
                    onPointerMove={event => move(event, shown, i)}
                    onPointerUp={end}
                    onPointerCancel={() => setDrag(null)}
                    onKeyDown={event => nudge(event, shown, i)}
                  />
                ))}
            </g>
          );
        }),
      )}
      {adding && selecting && (
        // an object or area picked to add: the next click on the plan is where it goes
        <rect
          data-area-place=""
          x={-1000}
          y={-1000}
          width={2000}
          height={2000}
          fill="transparent"
          style={{ cursor: 'crosshair' }}
          onPointerDown={place}
        />
      )}
    </g>
  );
}
