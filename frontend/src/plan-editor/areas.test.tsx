import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { buildScene } from '../dashboard/places/scene/build';
import { drawScene } from '../dashboard/places/scene/draw';
import type { Scene } from './protocol';

// The editor's own stores, stood in for: a scene store that holds nodes and takes `updateNode`, and an editor in its
// select mode. The real packages are megabytes of 3D editor that a test of our own panel has no use for.
const store = vi.hoisted(() => {
  const listeners = new Set<() => void>();
  const state = {
    nodes: {} as Record<string, Record<string, unknown>>,
    rootNodeIds: [] as string[],
    updateNode(id: string, data: Record<string, unknown>) {
      state.nodes = { ...state.nodes, [id]: { ...state.nodes[id], ...data } };
      for (const l of listeners) l();
    },
  };
  return { state, listeners, mode: 'select' };
});
vi.mock('@pascal-app/core', async () => {
  const { useSyncExternalStore } = await import('react');
  const subscribe = (l: () => void) => {
    store.listeners.add(l);
    return () => store.listeners.delete(l);
  };
  const useScene = <T,>(pick: (s: typeof store.state) => T) => useSyncExternalStore(subscribe, () => pick(store.state));
  useScene.getState = () => store.state;
  return { useScene };
});
vi.mock('@pascal-app/editor', () => ({
  useEditor: <T,>(pick: (s: { mode: string }) => T) => pick({ mode: store.mode }),
}));

import { areas, box, newId, outline, readArea, reshaped, translated, withArea } from './areas';
import { useAreasUi } from './areasUi';
import { Marks } from './Marks';
import { MarksTab } from './MarksTab';
import { useSpotsUi } from './spotsUi';

const plan = (extras?: Record<string, unknown>): Scene =>
  buildScene({
    height: 2.4,
    walls: [
      { from: [0, 0], to: [8, 0], thickness: 0.3 },
      { from: [8, 0], to: [8, 4], thickness: 0.3 },
    ],
    openings: [],
    rooms: [
      {
        id: 'den',
        name: 'Den',
        outline: [
          [0, 0],
          [8, 0],
          [8, 4],
          [0, 4],
        ],
      },
    ],
    extras,
  });
const MARKED = {
  cameras: { porch: { at: [1, 2], yaw: 90 } },
  objects: {
    sofa: {
      label: 'Sofa',
      poly: [
        [1, 1],
        [3, 1],
        [3, 2],
        [1, 2],
      ],
    },
  },
  unexplored: { back: { name: 'Back room', rect: [6, 3, 2, 1] } },
};

/** The scene as the editor's store holds it. */
const held = () => ({ nodes: store.state.nodes, rootNodeIds: store.state.rootNodeIds }) as unknown as Scene;

function open(scene: Scene) {
  store.state.nodes = scene.nodes as never;
  store.state.rootNodeIds = scene.rootNodeIds;
}

beforeEach(() => {
  store.mode = 'select';
  useAreasUi.setState({ selected: null, adding: null });
  useSpotsUi.setState({ selected: null, adding: null, cameras: [] });
});

it("reads an object's label and an unexplored area's name, and a ring from a poly, a rect, or a size box", () => {
  const object = readArea('sofa', 'object', {
    label: 'Sofa',
    poly: [
      [0, 0],
      [2, 0],
      [2, 1],
      [0, 1],
    ],
  });
  expect(object).toEqual({
    id: 'sofa',
    kind: 'object',
    label: 'Sofa',
    poly: [
      [0, 0],
      [2, 0],
      [2, 1],
      [0, 1],
    ],
  });
  expect(outline({ rect: [6, 3, 2, 1] })).toEqual([
    [6, 3],
    [8, 3],
    [8, 4],
    [6, 4],
  ]);
  expect(outline({ at: [4, 2], size: [2, 1] })).toEqual([
    [3, 1.5],
    [5, 1.5],
    [5, 2.5],
    [3, 2.5],
  ]);
  expect(readArea('back', 'unexplored', { name: 'Back room', rect: [6, 3, 2, 1] })).toMatchObject({
    label: 'Back room',
    poly: [
      [6, 3],
      [8, 3],
      [8, 4],
      [6, 4],
    ],
  });
  expect(readArea('bad', 'object', { label: 'Nowhere' })).toBeNull();
});

it("writes an object's label and ring into the level's records, keeping every other record and the other kind", () => {
  const scene = plan(MARKED);
  const written = withArea(scene, 'object', 'sofa', {
    id: 'sofa',
    kind: 'object',
    label: 'Long sofa',
    poly: [
      [1, 1],
      [4, 1],
      [4, 2],
      [1, 2],
    ],
  }) as { liberational: Record<string, unknown> };
  expect(written.liberational.cameras).toEqual(MARKED.cameras);
  expect(written.liberational.unexplored).toEqual(MARKED.unexplored);
  expect(written.liberational.objects).toEqual({
    sofa: {
      label: 'Long sofa',
      poly: [
        [1, 1],
        [4, 1],
        [4, 2],
        [1, 2],
      ],
    },
  });
  const added = withArea(scene, 'unexplored', 'attic', {
    id: 'attic',
    kind: 'unexplored',
    label: 'Attic',
    poly: [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
    ],
  }) as { liberational: { unexplored: object } };
  expect(added.liberational.unexplored).toHaveProperty('attic');
  expect(added.liberational.unexplored).toHaveProperty('back');
  const removed = withArea(scene, 'unexplored', 'back', null) as { liberational: { unexplored: object } };
  expect(removed.liberational.unexplored).toEqual({});
  expect(withArea({ nodes: {}, rootNodeIds: [] }, 'object', 'sofa', null)).toBeNull(); // no level to hold one
});

it('moves a ring bodily and reshapes one corner, snapping to the drag step', () => {
  const ring = box([4, 2], 2);
  expect(translated(ring, [0.123, -0.06])).toEqual([
    [3.1, 0.95],
    [5.1, 0.95],
    [5.1, 2.95],
    [3.1, 2.95],
  ]);
  expect(reshaped(ring, 2, [6.04, 4.06])).toEqual([
    [3, 1],
    [5, 1],
    [6.05, 4.05],
    [3, 3],
  ]);
  expect(areas(plan(MARKED))).toMatchObject({
    object: [{ id: 'sofa', kind: 'object', label: 'Sofa' }],
    unexplored: [{ id: 'back', kind: 'unexplored', label: 'Back room' }],
  });
  expect(newId('object')).toMatch(/^object-/);
});

it('draws a moved object and a reshaped unexplored area in the editor, and the Places page draws them after a save', () => {
  open(plan(MARKED));
  useAreasUi.setState({ selected: { kind: 'object', id: 'sofa' } });
  render(
    <svg>
      <title>plan</title>
      <Marks />
    </svg>,
  );
  // the object and the area are drawn in the editor's plan, with the object's label
  expect(document.querySelector('[data-area="object:sofa"] polygon')).not.toBeNull();
  expect(document.querySelector('[data-area="unexplored:back"] polygon')).not.toBeNull();
  expect(screen.getByText('Sofa', { selector: 'text' })).toBeTruthy();
  // the object's whole ring moves with the arrow keys, one change to the scene
  const dot = screen.getByRole('button', { name: 'Sofa: move' });
  fireEvent.focus(dot);
  fireEvent.keyDown(dot, { key: 'ArrowRight' });
  expect(areas(held()).object[0]?.poly).toEqual([
    [1.05, 1],
    [3.05, 1],
    [3.05, 2],
    [1.05, 2],
  ]);
  // and the Places page draws the object and the area from the same scene, after the save sends it
  const drawn = drawScene(held(), 'Home');
  expect(drawn?.svg).toContain('class="object"');
  expect(drawn?.svg).toContain('class="unexplored"');
  expect(drawn?.svg).toContain('<title>Sofa</title>');
});

it('reshapes an unexplored area by a corner, with the arrow keys', () => {
  open(plan(MARKED));
  useAreasUi.setState({ selected: { kind: 'unexplored', id: 'back' } });
  render(
    <svg>
      <title>plan</title>
      <Marks />
    </svg>,
  );
  const corner = screen.getByRole('button', { name: 'Back room: corner 2' });
  fireEvent.keyDown(corner, { key: 'ArrowLeft' });
  expect(areas(held()).unexplored[0]?.poly).toEqual([
    [6, 3],
    [7.95, 3],
    [8, 4],
    [6, 4],
  ]);
});

it("adds an object from the panel in the plan's middle, and removes it", () => {
  open(plan(MARKED));
  render(<MarksTab />);
  fireEvent.click(screen.getByRole('button', { name: 'Add an object' }));
  fireEvent.click(screen.getByRole('button', { name: 'Put it in the middle' }));
  expect(areas(held()).object).toHaveLength(2);
  expect(useAreasUi.getState().selected).toMatchObject({ kind: 'object' });
  const label = screen.getByLabelText('Label') as HTMLInputElement;
  fireEvent.change(label, { target: { value: 'Desk' } });
  expect(areas(held()).object.some(a => a.label === 'Desk')).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: 'Remove this object' }));
  expect(areas(held()).object).toHaveLength(1);
  expect(areas(held()).object[0]?.label).toBe('Sofa');
  expect(areas(held()).unexplored).toHaveLength(1);
});

it('leaves the marks alone while a camera is being added, and draws a locked scene without moving it', () => {
  open(plan(MARKED));
  useSpotsUi.setState({ adding: 'porch' });
  const { container, rerender } = render(
    <svg>
      <title>plan</title>
      <Marks />
    </svg>,
  );
  expect(container.querySelector('[data-plan-marks]')?.getAttribute('pointer-events')).toBe('none');
  useSpotsUi.setState({ adding: null });
  store.mode = 'build';
  rerender(
    <svg>
      <title>plan</title>
      <Marks />
    </svg>,
  );
  expect(container.querySelector('[data-plan-marks]')?.getAttribute('pointer-events')).toBe('none');
  store.mode = 'select';
  rerender(
    <svg>
      <title>plan</title>
      <Marks locked />
    </svg>,
  );
  expect(container.querySelector('[data-plan-marks]')?.getAttribute('pointer-events')).toBe('none');
  expect(screen.getByRole('button', { name: 'Sofa: move' }).getAttribute('tabindex')).toBe('-1');
});
