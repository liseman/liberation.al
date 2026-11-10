// Correcting a place's plan by hand: the parts of the floorplan library's playground editor the owner needs,
// on Mantine. Drag a wall (along its own normal) or a corner, rename a room, add or remove a door. Each edit is a
// splice into the plan's DSL text and the drawing is made again from it, so the drawing and the text can't drift; a
// drag that makes the plan wrong turns its findings up rather than being blocked. Every edit is a step in a bounded
// history that Undo and Redo walk (Ctrl+Z, and Ctrl+Shift+Z or Ctrl+Y; Cmd on a Mac), except while typing in a field,
// where the field's own undo applies; the history goes with the editor on save or cancel. Saving writes the owner's
// correction as taught state, over the machine's plan, which is never edited (memory/places.py `correct_plan`).
// Camera spots (the `x-camera` lines) are moved, turned, and added here too (CameraSpots.tsx).
import { ActionIcon, Alert, Button, Group, NumberInput, Select, Text, TextInput, Tooltip } from '@mantine/core';
import { useHotkeys } from '@mantine/hooks';
import { IconTrash } from '@tabler/icons-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { type PointerEvent, useMemo, useRef, useState } from 'react';
import { messageOf } from '../../../shared/http';
import { MAC } from '../../../shared/platform';
import {
  dslSpliceAll,
  type Finding,
  type FloorplanResult,
  floorplan,
  isDslText,
  type OffsetHandle,
  parseDsl,
  projection,
  type RenderOptions,
  spliceAll,
  type VertexHandle,
  wallHandles,
} from '../../../vendor/floorplan/index.js';
import { savePlan } from '../api';
import classes from '../places.module.css';
import { parsePlaceDsl } from '../plan/dsl';
import { lintPlace } from '../plan/lint';
import { CameraMarks, CameraPanel } from './CameraSpots';
import { bearing, setSpot, spotsOf } from './spots';
import { addDoor, doorSpot, join, removeOpening, renameRoom, split } from './text';

/** How many edits Undo can go back through; older ones drop off the history. */
export const MAX_STEPS = 100;

/** The editor's shortcuts, as shown in its tooltips and named in `aria-keyshortcuts`. */
const KEYS = MAC
  ? { undo: '⌘Z', redo: '⇧⌘Z', undoAria: 'Meta+Z', redoAria: 'Meta+Shift+Z' }
  : { undo: 'Ctrl+Z', redo: 'Ctrl+Shift+Z or Ctrl+Y', undoAria: 'Control+Z', redoAria: 'Control+Shift+Z Control+Y' };

/** How far (screen pixels) a press must move before it's a drag, so a click on a wall changes nothing. */
const THRESHOLD_PX = 3;

type Pt = [number, number];

/** What a plan's text draws as, or why it doesn't. */
export function read(text: string, options: RenderOptions): { result?: FloorplanResult; error?: string } {
  try {
    const { doc } = parseDsl(split(text).base);
    return { result: floorplan(doc, { render: options, markFindings: 'none' }) };
  } catch (e) {
    return { error: messageOf(e) };
  }
}

/** The findings worth showing while correcting a plan: the geometry checks, errors and warnings only. */
export function findingsOf(text: string): Finding[] {
  try {
    return lintPlace(parsePlaceDsl(text)).filter(f => f.severity !== 'info');
  } catch {
    return [];
  }
}

/** Drags land on 5 cm, or on millimetres when free (Alt), as the library's own drags do (edit.js SNAP). */
const SNAP_M = 0.05;
const snap = (v: number, free: boolean) =>
  Math.round((free ? Math.round(v * 1000) / 1000 : Math.round(v / SNAP_M) * SNAP_M) * 1000) / 1000;

/** `edits` spliced into `text` in whichever syntax it's written. The library's applyHandle and applyVertexHandle
 *  splice as JSON only (they call spliceAll where applyDrag calls applyEdits, edit.js at 4644b5f6), which throws on
 *  a DSL plan, and every real plan is DSL; so the handles' own edits are applied here, the way applyDrag applies its
 *  own. Reported upstream; with that fixed, this can call applyHandle and applyVertexHandle directly. */
function splice(text: string, edits: { path: Array<string | number>; literal: string }[]): string {
  if (!edits.length) return text;
  return isDslText(text) ? dslSpliceAll(text, edits) : spliceAll(text, edits);
}

/** The plan's text after dragging `handle` from `from` to `to` (plan metres): a wall moves along its own normal by how
 *  far the pointer went along it, a corner by how far it went. Applied to `base` as it was when the drag began, snapped
 *  and clamped as the library's drags are. */
export function dragged(base: string, handle: OffsetHandle | VertexHandle, from: Pt, to: Pt, free = false): string {
  const d: Pt = [to[0] - from[0], to[1] - from[1]];
  if (handle.kind === 'offset') {
    const raw = handle.at + d[0] * handle.normal[0] + d[1] * handle.normal[1];
    const next = Math.min(
      Math.max(snap(Math.min(Math.max(raw, handle.min), handle.max), free), handle.min),
      handle.max,
    );
    return Math.abs(next - handle.at) < 1e-9 ? base : splice(base, handle.edits(next));
  }
  return splice(base, handle.edits([snap(handle.at[0] + d[0], free), snap(handle.at[1] + d[1], free)]));
}

interface Gesture {
  /** The plan's text with the pointer at `to` (plan metres), from the text as it was when the gesture began. */
  apply: (from: Pt, to: Pt, free: boolean) => string;
  from: Pt;
  screen: Pt;
  moved: boolean;
}

export function PlanEditor({
  place,
  text: initial,
  frame,
  options,
  onDone,
}: {
  place: string;
  /** The plan as it holds now: the owner's correction, else the machine's. */
  text: string;
  /** The revision whose frame that text is in: a correction is saved against it, so positions stay on it. */
  frame: string | null;
  options: RenderOptions;
  onDone: () => void;
}) {
  const client = useQueryClient();
  const [history, setHistory] = useState({ stack: [initial], at: 0 });
  /** The text while a drag is under way (not yet a step in the history). */
  const [live, setLive] = useState<string | null>(null);
  const text = live ?? (history.stack[history.at] as string);
  const { base, ext } = split(text);
  const drawn = useMemo(() => read(text, options), [text, options]);
  const findings = useMemo(() => findingsOf(text), [text]);
  const result = drawn.result;
  const [lastGood, setLastGood] = useState<FloorplanResult | undefined>(result);
  if (result && result !== lastGood) setLastGood(result);
  const shown = result ?? lastGood;
  const proj = useMemo(() => (shown ? projection(shown.model, options) : null), [shown, options]);
  const handles = useMemo(() => {
    if (!result) return [];
    try {
      return [...wallHandles(base, result.model).values()].filter(
        (h): h is OffsetHandle | VertexHandle => h.kind === 'offset' || h.kind === 'vertex',
      );
    } catch {
      return [];
    }
  }, [base, result]);
  const overlay = useRef<SVGSVGElement>(null);
  const gesture = useRef<Gesture | null>(null);
  const spots = useMemo(() => spotsOf(text), [text]);
  const [camera, setCamera] = useState<string | null>(null);
  const [adding, setAdding] = useState<string | null>(null);
  const [door, setDoor] = useState<{ a: string | null; b: string | null; width: number }>({
    a: null,
    b: null,
    width: 0.9,
  });

  /** A new step in the history, unless it's the step already there. Compared with the history's head, not `text`:
   *  during a drag `text` is the drag's own live text, which is exactly what a drag's release commits. */
  const commit = (next: string) => {
    setHistory(h => {
      if (next === h.stack[h.at]) return h;
      const stack = [...h.stack.slice(0, h.at + 1), next].slice(-(MAX_STEPS + 1));
      return { stack, at: stack.length - 1 };
    });
  };
  const canUndo = history.at > 0;
  const canRedo = history.at < history.stack.length - 1;
  // Not mid-drag: the drag's release would commit over the step undone.
  const undo = () => {
    if (live === null) setHistory(h => (h.at > 0 ? { ...h, at: h.at - 1 } : h));
  };
  const redo = () => {
    if (live === null) setHistory(h => (h.at < h.stack.length - 1 ? { ...h, at: h.at + 1 } : h));
  };
  // Ignored in an input, a text area, or a select (useHotkeys' default), where the field's own undo applies.
  useHotkeys([
    ['mod+Z', undo],
    ['mod+shift+Z', redo],
    ['mod+Y', redo],
  ]);
  const edit = (change: (base: string) => string) => {
    try {
      commit(join(change(base), ext));
    } catch {
      // an edit the text couldn't take (the plan doesn't parse just now): nothing changes
    }
  };

  const save = useMutation({
    mutationFn: () => savePlan(place, text, frame),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ['places'] });
      onDone();
    },
  });

  /** A pointer's place in the plan, metres. */
  const toPlan = (e: PointerEvent): Pt | null => {
    const svg = overlay.current;
    const ctm = svg?.getScreenCTM?.();
    if (!svg || !ctm || !proj) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    return proj.toModel(p.x, p.y);
  };

  const start = (e: PointerEvent, apply: Gesture['apply']) => {
    const from = toPlan(e);
    if (!from) return;
    e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    gesture.current = { apply, from, screen: [e.clientX, e.clientY], moved: false };
  };
  const press = (e: PointerEvent, handle: OffsetHandle | VertexHandle) =>
    start(e, (from, to, free) => join(dragged(base, handle, from, to, free), ext));
  /** A press on a camera's spot (to move it) or its turn handle (to turn it). */
  const pressCamera = (e: PointerEvent, id: string, what: 'move' | 'turn') => {
    const spot = spots[id];
    const was = text;
    if (!spot) return;
    start(e, (from, to, free) => {
      if (what === 'turn') {
        const yaw = bearing(spot.at, to);
        return setSpot(was, id, { ...spot, yaw: free ? yaw : Math.round(yaw) });
      }
      const at: Pt = [snap(spot.at[0] + to[0] - from[0], free), snap(spot.at[1] + to[1] - from[1], free)];
      return setSpot(was, id, { ...spot, at });
    });
  };
  /** A click on the plan while a camera is picked to add: its spot goes there. */
  const putSpot = (e: PointerEvent) => {
    const at = toPlan(e);
    if (!adding || !at) return;
    commit(setSpot(text, adding, { at: [snap(at[0], false), snap(at[1], false)] }));
    setCamera(adding);
    setAdding(null);
  };
  const drag = (e: PointerEvent) => {
    const g = gesture.current;
    if (!g) return;
    if (!g.moved && Math.hypot(e.clientX - g.screen[0], e.clientY - g.screen[1]) < THRESHOLD_PX) return;
    g.moved = true;
    const at = toPlan(e);
    if (!at) return;
    // Each move applies to the text as it was when the gesture began, so the wall lands where the pointer is.
    setLive(g.apply(g.from, at, e.altKey));
  };
  const release = () => {
    const g = gesture.current;
    gesture.current = null;
    if (g?.moved && live !== null) commit(live);
    setLive(null);
  };

  const rooms = (result?.model.rooms ?? []).map(r => ({ id: r.room.id, name: r.room.name }));
  const spaces = [...rooms.map(r => ({ value: r.id, label: r.name })), { value: 'exterior', label: 'Outside' }];
  const openings = result?.model.openings ?? [];
  const nameOf = (id: string) => rooms.find(r => r.id === id)?.name ?? (id === 'exterior' ? 'Outside' : id);
  const viewBox = shown?.svg.match(/viewBox="([^"]+)"/)?.[1];
  const [bx, by, bw, bh] = viewBox?.split(/[\s,]+/).map(Number) ?? [];
  const box = bx !== undefined && by !== undefined && bw !== undefined && bh !== undefined ? { bx, by, bw, bh } : null;
  // The middle of the rooms' extent, where "Put it in the middle" puts a new camera spot.
  const corners = (shown?.model.rooms ?? []).flatMap(r => (r.room.poly as Pt[] | undefined) ?? []);
  const middle: Pt | null = corners.length
    ? [
        (Math.min(...corners.map(c => c[0])) + Math.max(...corners.map(c => c[0]))) / 2,
        (Math.min(...corners.map(c => c[1])) + Math.max(...corners.map(c => c[1]))) / 2,
      ]
    : null;
  const errors = findings.filter(f => f.severity === 'error').length;

  return (
    <div className={classes.editor}>
      <Group justify="space-between" gap="xs" mb="xs" wrap="wrap">
        <Text size="sm" c="dimmed">
          Drag a wall, a corner, or a camera, and a selected camera's handle to turn it. Alt drags in millimetres. Your
          plan replaces the machine's, which is kept, and every version you save can be restored from the Editor view.
        </Text>
        <Group gap="xs">
          <Tooltip label={`Undo (${KEYS.undo})`}>
            <Button
              size="compact-sm"
              variant="default"
              disabled={!canUndo}
              onClick={undo}
              aria-keyshortcuts={KEYS.undoAria}
            >
              Undo
            </Button>
          </Tooltip>
          <Tooltip label={`Redo (${KEYS.redo})`}>
            <Button
              size="compact-sm"
              variant="default"
              disabled={!canRedo}
              onClick={redo}
              aria-keyshortcuts={KEYS.redoAria}
            >
              Redo
            </Button>
          </Tooltip>
          <Button size="compact-sm" variant="default" disabled={text === initial} onClick={() => commit(initial)}>
            Start over
          </Button>
        </Group>
      </Group>
      {drawn.error && (
        <Alert color="yellow" variant="light" mb="xs" title="This plan doesn't read right now">
          {drawn.error}
        </Alert>
      )}
      {shown && proj && viewBox && (
        <div className={classes.drawingWrap} data-stale={drawn.error ? '' : undefined}>
          <div
            className={classes.drawing}
            // biome-ignore lint/security/noDangerouslySetInnerHtml: the library's own SVG, which escapes every label
            dangerouslySetInnerHTML={{ __html: shown.svg }}
          />
          <svg
            ref={overlay}
            className={classes.handles}
            viewBox={viewBox}
            onPointerMove={drag}
            onPointerUp={release}
            onPointerCancel={release}
          >
            <title>Walls, corners, and cameras to drag</title>
            {handles.map(h => {
              if (h.kind === 'vertex') {
                const [x, y] = proj.toScreen(h.at);
                return (
                  <circle key={h.id} className={classes.corner} cx={x} cy={y} r={6} onPointerDown={e => press(e, h)}>
                    <title>{`Drag this corner: ${h.writes}`}</title>
                  </circle>
                );
              }
              const wall = result?.model.walls.find(w => w.id === h.wallId);
              if (!wall) return null;
              const [x1, y1] = proj.toScreen(wall.start);
              const [x2, y2] = proj.toScreen(wall.end);
              return (
                <line
                  key={h.id}
                  className={classes.wallHandle}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  onPointerDown={e => press(e, h)}
                >
                  <title>{`Drag this wall: ${h.writes}`}</title>
                </line>
              );
            })}
            {adding && box && (
              <rect
                className={classes.placeCatcher}
                x={box.bx}
                y={box.by}
                width={box.bw}
                height={box.bh}
                aria-label={`Put ${adding} here`}
                onPointerDown={putSpot}
              />
            )}
            <CameraMarks
              spots={spots}
              proj={proj}
              text={text}
              selected={camera}
              onSelect={setCamera}
              onCommit={commit}
              onPress={pressCamera}
            />
          </svg>
        </div>
      )}
      <div className={classes.editorPanels}>
        <section aria-labelledby="edit-rooms-h">
          <h3 id="edit-rooms-h" className={classes.panelTitle}>
            Rooms
          </h3>
          {rooms.map(r => (
            <TextInput
              key={`${r.id}-${r.name}`}
              size="xs"
              label="Name"
              aria-label={`Name of ${r.name}`}
              defaultValue={r.name}
              onBlur={e => {
                if (e.currentTarget.value.trim() && e.currentTarget.value.trim() !== r.name)
                  edit(b => renameRoom(b, r.id, e.currentTarget.value));
              }}
              onKeyDown={e => {
                if (e.key === 'Enter') e.currentTarget.blur();
              }}
              mb={6}
            />
          ))}
        </section>
        <section aria-labelledby="edit-doors-h">
          <h3 id="edit-doors-h" className={classes.panelTitle}>
            Doors and openings
          </h3>
          <ul className={classes.openings}>
            {openings.map(o => (
              <li key={o.spec.id}>
                <span>
                  {o.spec.type === 'cased' ? 'Opening' : o.spec.type === 'window' ? 'Window' : 'Door'}:{' '}
                  {nameOf(o.spec.between[0])} and {nameOf(o.spec.between[1])}, {Math.round(o.spec.width * 100) / 100} m
                </span>
                <ActionIcon
                  variant="subtle"
                  color="red"
                  size="sm"
                  aria-label={`Remove the ${o.spec.type} between ${nameOf(o.spec.between[0])} and ${nameOf(o.spec.between[1])}`}
                  onClick={() => edit(b => removeOpening(b, o.spec.index))}
                >
                  <IconTrash size={14} />
                </ActionIcon>
              </li>
            ))}
          </ul>
          <Group gap="xs" align="end" wrap="wrap">
            <Select
              size="xs"
              label="Door from"
              data={spaces}
              value={door.a}
              onChange={a => setDoor(d => ({ ...d, a }))}
              w={130}
            />
            <Select
              size="xs"
              label="to"
              data={spaces.filter(s => s.value !== door.a)}
              value={door.b}
              onChange={b => setDoor(d => ({ ...d, b }))}
              w={130}
            />
            <NumberInput
              size="xs"
              label="Width, m"
              value={door.width}
              onChange={w => setDoor(d => ({ ...d, width: Number(w) || 0.9 }))}
              min={0.5}
              max={4}
              step={0.1}
              decimalScale={2}
              w={90}
            />
            <Button
              size="xs"
              variant="light"
              disabled={!door.a || !door.b}
              onClick={() => {
                const { a, b } = door;
                // On the middle of the longest wall the two share, so a room with several outside walls isn't ambiguous.
                const spot = result
                  ? doorSpot(result.model.walls as Parameters<typeof doorSpot>[0], a ?? '', b ?? '')
                  : null;
                if (a && b) edit(t => addDoor(t, a, b, door.width, spot ?? undefined));
                setDoor(d => ({ ...d, a: null, b: null }));
              }}
            >
              Add door
            </Button>
          </Group>
        </section>
        <CameraPanel
          spots={spots}
          text={text}
          selected={camera}
          adding={adding}
          onSelect={setCamera}
          onAdding={setAdding}
          onCommit={commit}
          middle={middle}
        />
        <section aria-labelledby="edit-findings-h">
          <h3 id="edit-findings-h" className={classes.panelTitle}>
            {findings.length ? `To look at (${findings.length})` : 'Nothing to look at'}
          </h3>
          <ul className={classes.findings}>
            {findings.map((f, i) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: findings have no identity beyond their place
              <li key={i} data-severity={f.severity}>
                {f.message}
              </li>
            ))}
          </ul>
        </section>
      </div>
      {save.isError && (
        <Alert color="red" variant="light" mt="sm">
          Couldn't save the plan: {messageOf(save.error)}
        </Alert>
      )}
      <Group justify="flex-end" mt="sm" gap="xs">
        <Button variant="default" onClick={onDone} disabled={save.isPending}>
          Cancel
        </Button>
        <Button
          onClick={() => save.mutate()}
          loading={save.isPending}
          disabled={text === initial || !!drawn.error}
          title={errors ? 'The plan still has problems; you can save it and fix them later.' : undefined}
        >
          Save plan
        </Button>
      </Group>
    </div>
  );
}
