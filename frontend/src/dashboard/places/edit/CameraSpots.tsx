// The plan editor's camera spots: each `x-camera` line drawn on the plan as a dot with its view as a cone, and a small
// form for the one selected. The spot's dot drags to move it and the handle at the cone's tip drags to turn it (or,
// focused, the arrow keys: the dot moves 5 cm a press, 50 cm with Shift; the handle turns 5 degrees, 15 with Shift).
// A camera from the device manifest with no spot yet is added by picking it and clicking the plan, or put in the
// plan's middle. Every change is one `setSpot` on the plan's text (spots.ts), so it is a step in the editor's history
// like any other edit and is saved with the rest of the plan.
import { Button, Group, NumberInput, Select, Text, TextInput } from '@mantine/core';
import type { KeyboardEvent, PointerEvent } from 'react';
import type { Projection } from '../../../vendor/floorplan/index.js';
import { shell } from '../../shell';
import classes from '../places.module.css';
import { facing, heading, type Pt, type Spot, setSpot } from './spots';

/** How far the cone reaches, metres, and how wide it's drawn when the spot gives no field of view. */
export const REACH_M = 1.5;
const DEFAULT_FOV = 90;

/** The arrow keys' steps: metres for a spot, degrees for a facing; Shift takes the larger. */
const MOVE_M = [0.05, 0.5] as const;
const TURN_DEG = [5, 15] as const;

/** A camera's name as the page shows it: its friendly name when it has one. */
export const cameraName = (id: string) => shell().views[id] ?? id;

const f = (n: number) => Math.round(n * 100) / 100;

function cone(proj: Projection, s: Spot): string | null {
  if (s.yaw === undefined) return null;
  const fov = s.fov ?? DEFAULT_FOV;
  const [cx, cy] = proj.toScreen(s.at);
  const r = REACH_M * proj.scale;
  const [ax, ay] = heading(s.yaw - fov / 2);
  const [bx, by] = heading(s.yaw + fov / 2);
  return `M${f(cx)} ${f(cy)} L${f(cx + ax * r)} ${f(cy + ay * r)} A${f(r)} ${f(r)} 0 ${fov > 180 ? 1 : 0} 1 ${f(cx + bx * r)} ${f(cy + by * r)} Z`;
}

/** Where a spot's turn handle sits, plan metres: at the cone's tip, or up the page while it has no facing. */
export const tipOf = (s: Spot): Pt => {
  const [dx, dy] = heading(s.yaw ?? 0);
  return [s.at[0] + dx * REACH_M, s.at[1] + dy * REACH_M];
};

function arrow(e: KeyboardEvent): Pt | null {
  const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
  return (d as Pt | undefined) ?? null;
}

export function CameraMarks({
  spots,
  proj,
  text,
  selected,
  onSelect,
  onCommit,
  onPress,
}: {
  spots: Record<string, Spot>;
  proj: Projection;
  text: string;
  selected: string | null;
  onSelect: (camera: string) => void;
  onCommit: (text: string) => void;
  /** A press on a spot ("move") or its handle ("turn"), to drag. */
  onPress: (e: PointerEvent, camera: string, what: 'move' | 'turn') => void;
}) {
  return (
    <>
      {Object.entries(spots).map(([id, s]) => {
        const [cx, cy] = proj.toScreen(s.at);
        const path = cone(proj, s);
        const name = cameraName(id);
        const isSelected = id === selected;
        const [tx, ty] = proj.toScreen(tipOf(s));
        const step = (e: KeyboardEvent, steps: readonly [number, number]) => steps[e.shiftKey ? 1 : 0];
        return (
          <g key={id} data-camera={id}>
            {path ? (
              <path className={classes.cameraCone} data-selected={isSelected || undefined} d={path} />
            ) : (
              <circle className={classes.cameraNoFacing} cx={cx} cy={cy} r={REACH_M * proj.scale * 0.4} />
            )}
            {isSelected && (
              <>
                <line className={classes.cameraArm} x1={cx} y1={cy} x2={tx} y2={ty} />
                <circle
                  className={classes.cameraTurn}
                  cx={tx}
                  cy={ty}
                  r={8}
                  tabIndex={0}
                  role="slider"
                  aria-label={`Facing of ${name}`}
                  aria-valuemin={0}
                  aria-valuemax={360}
                  aria-valuenow={s.yaw ?? 0}
                  aria-valuetext={s.yaw === undefined ? 'no facing yet' : `${s.yaw} degrees`}
                  onPointerDown={e => onPress(e, id, 'turn')}
                  onKeyDown={e => {
                    const d = arrow(e);
                    if (!d) return;
                    e.preventDefault();
                    // Right and up turn clockwise, left and down back.
                    const by = (d[0] || -d[1]) * step(e, TURN_DEG);
                    onCommit(setSpot(text, id, { ...s, yaw: facing((s.yaw ?? 0) + by) }));
                  }}
                >
                  <title>{`Drag to turn ${name}`}</title>
                </circle>
              </>
            )}
            {/* biome-ignore lint/a11y/useSemanticElements: an SVG mark on the plan; a <button> can't be one */}
            <circle
              className={classes.cameraDot}
              data-selected={isSelected || undefined}
              cx={cx}
              cy={cy}
              r={7}
              tabIndex={0}
              role="button"
              aria-pressed={isSelected}
              aria-label={`Camera ${name}`}
              onPointerDown={e => {
                onSelect(id);
                onPress(e, id, 'move');
              }}
              onFocus={() => onSelect(id)}
              onKeyDown={e => {
                const d = arrow(e);
                if (!d) return;
                e.preventDefault();
                const m = step(e, MOVE_M);
                onCommit(setSpot(text, id, { ...s, at: [s.at[0] + d[0] * m, s.at[1] + d[1] * m] }));
              }}
            >
              <title>{`${name}: drag to move it`}</title>
            </circle>
          </g>
        );
      })}
    </>
  );
}

/** A number box that edits on leaving it (or Enter), as the room names do; empty clears an optional value. */
function Field({
  label,
  value,
  onSet,
  optional,
  step,
}: {
  label: string;
  value: number | undefined;
  onSet: (v: number | undefined) => void;
  optional?: boolean;
  step: number;
}) {
  return (
    <NumberInput
      key={`${label}-${value ?? ''}`}
      size="xs"
      label={label}
      defaultValue={value ?? ''}
      step={step}
      decimalScale={2}
      w={90}
      onBlur={e => {
        const raw = e.currentTarget.value.trim();
        if (raw === '') {
          if (optional && value !== undefined) onSet(undefined);
          return;
        }
        const n = Number(raw);
        if (Number.isFinite(n) && n !== value) onSet(n);
      }}
      onKeyDown={e => {
        if (e.key === 'Enter') e.currentTarget.blur();
      }}
    />
  );
}

export function CameraPanel({
  spots,
  text,
  selected,
  adding,
  onSelect,
  onAdding,
  onCommit,
  middle,
}: {
  spots: Record<string, Spot>;
  text: string;
  selected: string | null;
  /** The manifest camera picked to add, waiting for a click on the plan. */
  adding: string | null;
  onSelect: (camera: string | null) => void;
  onAdding: (camera: string | null) => void;
  onCommit: (text: string) => void;
  /** The plan's middle, plan metres, where "Put it in the middle" puts a new spot. */
  middle: Pt | null;
}) {
  const s = selected ? spots[selected] : undefined;
  const unplaced = shell().cams.filter(c => !(c in spots));
  const set = (change: Partial<Spot>) => {
    if (selected && s) onCommit(setSpot(text, selected, { ...s, ...change }));
  };
  return (
    <section aria-labelledby="edit-cameras-h">
      <h3 id="edit-cameras-h" className={classes.panelTitle}>
        Cameras
      </h3>
      <Group gap={6} mb={6} wrap="wrap">
        {Object.entries(spots).map(([id, spot]) => (
          <Button
            key={id}
            size="compact-xs"
            variant={id === selected ? 'filled' : 'default'}
            aria-pressed={id === selected}
            onClick={() => onSelect(id === selected ? null : id)}
          >
            {cameraName(id)}
            {spot.yaw === undefined ? ' (no facing)' : ''}
          </Button>
        ))}
      </Group>
      {selected && s && (
        <div className={classes.cameraForm}>
          <TextInput size="xs" label="Camera" value={cameraName(selected)} readOnly w={140} />
          <Group gap="xs" wrap="wrap">
            <Field label="X, m" value={s.at[0]} step={0.05} onSet={v => set({ at: [v ?? s.at[0], s.at[1]] })} />
            <Field label="Y, m" value={s.at[1]} step={0.05} onSet={v => set({ at: [s.at[0], v ?? s.at[1]] })} />
            <Field label="Height, m" value={s.height} step={0.05} optional onSet={height => set({ height })} />
            <Field
              label="Facing, °"
              value={s.yaw}
              step={5}
              optional
              onSet={yaw => set({ yaw: yaw === undefined ? undefined : facing(yaw) })}
            />
            <Field
              label="View, °"
              value={s.fov}
              step={5}
              optional
              onSet={fov => set({ fov: fov === undefined ? undefined : Math.min(Math.max(fov, 1), 359) })}
            />
          </Group>
          {s.yaw === undefined && (
            <Text size="xs" c="dimmed" mt={4}>
              It places nobody until it has a facing: drag the handle, or type one (0 is up the page, 90 to the right).
            </Text>
          )}
        </div>
      )}
      {unplaced.length > 0 && (
        <Group gap="xs" align="end" wrap="wrap" mt={6}>
          <Select
            size="xs"
            label="Add a camera"
            placeholder="Pick one"
            data={unplaced.map(c => ({ value: c, label: cameraName(c) }))}
            value={adding}
            onChange={onAdding}
            w={150}
          />
          {adding && (
            <Button
              size="xs"
              variant="light"
              disabled={!middle}
              onClick={() => {
                if (middle) onCommit(setSpot(text, adding, { at: middle }));
                onSelect(adding);
                onAdding(null);
              }}
            >
              Put it in the middle
            </Button>
          )}
        </Group>
      )}
      {adding && (
        <Text size="xs" c="dimmed" mt={4}>
          Click the plan where {cameraName(adding)} is.
        </Text>
      )}
      <Text size="xs" c="dimmed" mt={6}>
        People are placed from a saved spot at the next hourly positions run.
      </Text>
    </section>
  );
}
