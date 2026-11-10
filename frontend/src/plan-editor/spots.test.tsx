import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { buildScene } from '../dashboard/places/scene/build';
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

import { CameraMarks } from './CameraMarks';
import { CamerasTab } from './CamerasTab';
import { bearing, cone, facing, heading, middle, spots, tipOf, withSpot } from './spots';
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
const SPOTTED = {
  cameras: { porch: { at: [1, 2], height: 2.2, yaw: 90, fov: 100 } },
  objects: { sofa: { label: 'Sofa' } },
};

/** The spots as the scene the panel wrote holds them. */
const held = () => spots({ nodes: store.state.nodes, rootNodeIds: store.state.rootNodeIds } as unknown as Scene);

function open(scene: Scene) {
  store.state.nodes = scene.nodes as never;
  store.state.rootNodeIds = scene.rootNodeIds;
}

beforeEach(() => {
  store.mode = 'select';
  useSpotsUi.setState({ selected: null, adding: null, cameras: [] });
});

it('keeps a facing between 0 and 360 in tenths, and reads a direction off the page as one', () => {
  expect([facing(0), facing(360), facing(-90), facing(725.04), facing(359.96)]).toEqual([0, 0, 270, 5, 0]);
  expect(heading(0)).toEqual([0, -1]); // up the page
  expect(heading(90)[0]).toBeCloseTo(1); // to the right
  // from a spot toward a point: right of it is 90, below it 180, left 270, above 0, snapped to whole degrees
  expect([bearing([1, 1], [3, 1]), bearing([1, 1], [1, 5]), bearing([1, 1], [0, 1]), bearing([1, 1], [1, 0])]).toEqual([
    90, 180, 270, 0,
  ]);
  expect(bearing([0, 0], [1, -1.01])).toBe(45); // 44.7 degrees snaps to 45
});

it("puts a spot's turn handle at the end of its view, up the page while it has no facing", () => {
  expect(tipOf({ at: [2, 3] })).toEqual([2, 1.5]);
  const [x, y] = tipOf({ at: [2, 3], yaw: 90 });
  expect([x, y]).toEqual([3.5, expect.closeTo(3)]);
  expect(cone({ at: [2, 3] })).toBeNull();
  expect(cone({ at: [0, 0], yaw: 0, fov: 180 })).toBe('M0.000 0.000 L-1.500 -0.000 A1.5 1.5 0 0 1 1.500 -0.000 Z');
  expect(cone({ at: [0, 0], yaw: 0, fov: 200 })).toContain('A1.5 1.5 0 1 1'); // past a half turn it is the long way round
});

it("writes a spot into the level's records, keeping every other record and clearing a number taken away", () => {
  const scene = plan(SPOTTED);
  const moved = withSpot(scene, 'porch', { at: [3, 1], height: 2.2, fov: 100 }) as {
    liberational: Record<string, unknown>;
  };
  expect(moved.liberational.cameras).toEqual({ porch: { at: [3, 1], height: 2.2, fov: 100 } }); // the facing is gone
  expect(moved.liberational.objects).toEqual({ sofa: { label: 'Sofa' } });
  const added = withSpot(scene, 'garage', { at: [5, 5], yaw: 365, fov: 400 }) as { liberational: { cameras: object } };
  expect(added.liberational.cameras).toEqual({
    porch: { at: [1, 2], height: 2.2, yaw: 90, fov: 100 },
    garage: { at: [5, 5], yaw: 5, fov: 359 }, // a facing wraps, and a view stays under a full turn
  });
  const removed = withSpot(scene, 'porch', null) as { liberational: { cameras: object } };
  expect(removed.liberational.cameras).toEqual({});
  expect(withSpot({ nodes: {}, rootNodeIds: [] }, 'porch', { at: [0, 0] })).toBeNull(); // no level to hold one
});

it("finds the plan's middle for a new spot, from its rooms and walls", () => {
  expect(middle(plan())).toEqual([4, 2]);
  expect(middle(buildScene({ height: 2.4, walls: [], openings: [], rooms: [] }))).toBeNull();
});

it('changes the selected spot from the form, one number at a time, and clears an optional one when emptied', () => {
  open(plan(SPOTTED));
  useSpotsUi.setState({ cameras: [{ id: 'porch', name: 'Front porch' }] });
  render(<CamerasTab />);
  fireEvent.click(screen.getByRole('button', { name: 'Front porch' }));
  const field = (label: string) => screen.getByLabelText(label) as HTMLInputElement;
  expect([field('X, m').value, field('Height, m').value, field('Facing, °').value]).toEqual(['1', '2.2', '90']);
  fireEvent.change(field('X, m'), { target: { value: '3.25' } });
  fireEvent.blur(field('X, m'));
  fireEvent.change(field('Facing, °'), { target: { value: '450' } });
  fireEvent.keyDown(field('Facing, °'), { key: 'Enter' });
  expect(held().porch).toEqual({ at: [3.25, 2], height: 2.2, yaw: 90, fov: 100 }); // 450 is 90 again
  fireEvent.change(field('Height, m'), { target: { value: '' } });
  fireEvent.blur(field('Height, m'));
  expect(held().porch?.height).toBeUndefined();
  fireEvent.change(field('Y, m'), { target: { value: '' } }); // a position can't be emptied: it goes back
  fireEvent.blur(field('Y, m'));
  expect(field('Y, m').value).toBe('2');
  expect(held().porch?.at).toEqual([3.25, 2]);
});

it('says a spot with no facing places nobody, and removes a spot', () => {
  open(plan({ cameras: { porch: { at: [1, 2] } } }));
  render(<CamerasTab />);
  fireEvent.click(screen.getByRole('button', { name: 'porch (no facing)' }));
  expect(screen.getByText(/places nobody until it has a facing/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Remove this spot' }));
  expect(held()).toEqual({});
  expect(screen.getByText('No camera has a spot on this plan yet.')).toBeTruthy();
});

it("offers the house's cameras that have no spot, and puts one in the plan's middle", () => {
  open(plan(SPOTTED));
  useSpotsUi.setState({
    cameras: [
      { id: 'porch', name: 'Front porch' },
      { id: 'garage', name: 'Garage camera' },
    ],
  });
  render(<CamerasTab />);
  const pick = screen.getByLabelText('Add a camera') as HTMLSelectElement;
  expect(
    within(pick)
      .getAllByRole('option')
      .map(o => o.textContent),
  ).toEqual(['Pick one', 'Garage camera']);
  fireEvent.change(pick, { target: { value: 'garage' } });
  expect(screen.getByText(/Click the plan where Garage camera is/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Put it in the middle' }));
  expect(held().garage).toEqual({ at: [4, 2], height: undefined, yaw: undefined, fov: undefined });
  expect(useSpotsUi.getState()).toMatchObject({ selected: 'garage', adding: null });
  expect(screen.queryByLabelText('Add a camera')).toBeNull(); // every camera has a spot now
});

it('moves a focused spot and turns its handle with the arrow keys, farther with Shift', () => {
  open(plan(SPOTTED));
  render(
    <svg>
      <title>plan</title>
      <CameraMarks />
    </svg>,
  );
  const dot = screen.getByRole('button', { name: 'porch: move' });
  fireEvent.focus(dot);
  fireEvent.keyDown(dot, { key: 'ArrowRight' });
  fireEvent.keyDown(dot, { key: 'ArrowDown', shiftKey: true });
  expect(held().porch?.at).toEqual([1.05, 2.5]);
  const handle = screen.getByRole('slider', { name: /porch: facing/ }); // shown for the selected spot
  fireEvent.keyDown(handle, { key: 'ArrowLeft' });
  fireEvent.keyDown(handle, { key: 'ArrowRight', shiftKey: true });
  expect(held().porch?.yaw).toBe(100); // 90 - 5 + 15
  expect(handle.getAttribute('aria-valuenow')).toBe('100');
});

it("leaves the spots alone while a drawing tool is in use, and shows where a picked camera's click goes", () => {
  open(plan(SPOTTED));
  store.mode = 'build';
  useSpotsUi.setState({ adding: 'garage' });
  const { container, rerender } = render(
    <svg>
      <title>plan</title>
      <CameraMarks />
    </svg>,
  );
  expect(container.querySelector('[data-camera-spots]')?.getAttribute('pointer-events')).toBe('none');
  expect(container.querySelector('[data-camera-place]')).toBeNull();
  store.mode = 'select';
  rerender(
    <svg>
      <title>plan</title>
      <CameraMarks />
    </svg>,
  );
  expect(container.querySelector('[data-camera-spots]')?.hasAttribute('pointer-events')).toBe(false);
  expect(container.querySelector('[data-camera-place]')).not.toBeNull();
});
