// The camera spots drawn in the editor's 2D plan: each a dot where the camera is and its view as a cone. In the
// editor's select mode the dot drags to move the spot and the handle at the cone's end drags to turn it (focused,
// the arrow keys do the same: 5 cm or 5 degrees a press, 50 cm or 15 degrees with Shift), and a camera picked in the
// Cameras panel is put where the plan is clicked. This is mounted inside the editor's own plan (`floorplanSceneSlot`),
// whose coordinates are the plan's metres, so nothing here knows the zoom or the turn of the view.
import { useEditor } from '@pascal-app/editor';
import { type KeyboardEvent, type PointerEvent, useRef, useState } from 'react';
import { bearing, cone, type Pt, type Spot, snapped, tipOf } from './spots';
import { cameraName, useSpotsUi } from './spotsUi';
import { useSpots } from './useSpots';

/** The arrow keys' steps: metres for a spot, degrees for a facing; Shift takes the larger. */
const MOVE_M = [0.05, 0.5] as const;
const TURN_DEG = [5, 15] as const;
/** Sizes on the plan, metres. */
const DOT_M = 0.14;
const HANDLE_M = 0.1;
/** The dashboard's accent and the plan's ground (plan-editor.css), as styles: an SVG attribute can't name a variable. */
const ACCENT = 'var(--plan-accent)';
const GROUND = 'var(--plan-ground)';

function arrow(event: KeyboardEvent): Pt | null {
  const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
  return (d as Pt | undefined) ?? null;
}

/** `locked`: draw the spots without letting them be moved, turned, or added (a scene opened to look at). */
export function CameraMarks({ locked = false }: { locked?: boolean }) {
  const { spots, set } = useSpots();
  const selecting = useEditor(state => state.mode === 'select') && !locked;
  const { selected, adding, cameras, select, setAdding } = useSpotsUi();
  const group = useRef<SVGGElement>(null);
  // a drag shows where the spot is going without writing it: one change to the scene when the pointer lifts
  const [drag, setDrag] = useState<{ camera: string; spot: Spot } | null>(null);

  /** Where a pointer is on the plan, metres. */
  const planPoint = (event: PointerEvent): Pt | null => {
    const matrix = group.current?.getScreenCTM();
    if (!matrix) return null;
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return [p.x, p.y];
  };
  const start = (event: PointerEvent, camera: string) => {
    event.stopPropagation(); // the plan under it would start a selection box
    event.currentTarget.setPointerCapture(event.pointerId);
    select(camera);
  };
  const move = (event: PointerEvent, camera: string, change: (at: Pt, spot: Spot) => Spot) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const at = planPoint(event);
    const spot = spots[camera];
    if (at && spot) setDrag({ camera, spot: change(at, spot) });
  };
  const end = (event: PointerEvent) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
    if (drag) set(drag.camera, drag.spot);
    setDrag(null);
  };

  return (
    <g ref={group} data-camera-spots="" pointerEvents={selecting ? undefined : 'none'}>
      {adding && selecting && (
        // a camera picked to add: the next click on the plan is where it is
        <rect
          data-camera-place=""
          x={-1000}
          y={-1000}
          width={2000}
          height={2000}
          fill="transparent"
          style={{ cursor: 'crosshair' }}
          onPointerDown={event => {
            const at = planPoint(event);
            if (!at) return;
            event.stopPropagation();
            set(adding, { at: [snapped(at[0]), snapped(at[1])] });
            select(adding);
            setAdding(null);
          }}
        />
      )}
      {Object.entries(spots).map(([camera, stored]) => {
        const spot = drag?.camera === camera ? drag.spot : stored;
        const view = cone(spot);
        const tip = tipOf(spot);
        const on = camera === selected;
        const name = cameraName(cameras, camera);
        return (
          <g key={camera} data-camera={camera}>
            {view && <path d={view} style={{ fill: ACCENT }} fillOpacity={on ? 0.28 : 0.14} pointerEvents="none" />}
            {on && (
              <line
                x1={spot.at[0]}
                y1={spot.at[1]}
                x2={tip[0]}
                y2={tip[1]}
                style={{ stroke: ACCENT }}
                strokeWidth={1.5}
                strokeDasharray={spot.yaw === undefined ? '4 4' : undefined}
                vectorEffect="non-scaling-stroke"
                pointerEvents="none"
              />
            )}
            {/* biome-ignore lint/a11y/useSemanticElements: a draggable mark on a drawing; no HTML control stands in for it */}
            <circle
              role="button"
              tabIndex={selecting ? 0 : -1}
              aria-label={`${name}: move`}
              aria-pressed={on}
              cx={spot.at[0]}
              cy={spot.at[1]}
              r={DOT_M}
              strokeWidth={on ? 3 : 1.5}
              vectorEffect="non-scaling-stroke"
              style={{ cursor: 'move', fill: ACCENT, stroke: GROUND }}
              onPointerDown={event => start(event, camera)}
              onPointerMove={event =>
                move(event, camera, (at, was) => ({ ...was, at: [snapped(at[0]), snapped(at[1])] }))
              }
              onPointerUp={end}
              onPointerCancel={() => setDrag(null)}
              onFocus={() => select(camera)}
              onKeyDown={event => {
                const d = arrow(event);
                if (!d) return;
                event.preventDefault();
                event.stopPropagation();
                const by = MOVE_M[event.shiftKey ? 1 : 0];
                set(camera, { ...stored, at: [snapped(stored.at[0] + d[0] * by), snapped(stored.at[1] + d[1] * by)] });
              }}
            >
              <title>{name}</title>
            </circle>
            {on && (
              <circle
                role="slider"
                tabIndex={selecting ? 0 : -1}
                aria-label={`${name}: facing, degrees clockwise from up the page`}
                aria-valuemin={0}
                aria-valuemax={359}
                aria-valuenow={Math.round(spot.yaw ?? 0)}
                cx={tip[0]}
                cy={tip[1]}
                r={HANDLE_M}
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
                style={{ cursor: 'grab', fill: GROUND, stroke: ACCENT }}
                onPointerDown={event => start(event, camera)}
                onPointerMove={event => move(event, camera, (at, was) => ({ ...was, yaw: bearing(was.at, at) }))}
                onPointerUp={end}
                onPointerCancel={() => setDrag(null)}
                onKeyDown={event => {
                  const d = arrow(event);
                  if (!d) return;
                  event.preventDefault();
                  event.stopPropagation();
                  const by = TURN_DEG[event.shiftKey ? 1 : 0] * (d[0] || d[1]);
                  set(camera, { ...stored, yaw: ((((stored.yaw ?? 0) + by) % 360) + 360) % 360 });
                }}
              />
            )}
          </g>
        );
      })}
    </g>
  );
}
