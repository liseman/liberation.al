import { act, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { editorUrl, fromHash, type Open, ours, type Scene } from '../../../plan-editor/protocol';
import { renderPage } from '../../../test/render';
import { along, buildScene } from './build';
import { EXAMPLE_SCENE } from './example';
import { PlanFrame } from './PlanFrame';

const of = (scene: Scene, type: string) => Object.values(scene.nodes).filter(n => n.type === type);

it('measures a point along a wall from its start, and says when it is off the wall', () => {
  expect(along([0, 0], [4, 0], [1.5, 0])).toBe(1.5);
  expect(along([4, 0], [0, 0], [1.5, 0])).toBe(2.5); // from the wall's own start
  expect(along([0, 0], [3, 4], [1.5, 2])).toBeCloseTo(2.5); // a wall at an angle
  expect(along([0, 0], [4, 0], [1.5, 0.2])).toBeNull(); // beside it
  expect(along([0, 0], [4, 0], [4.5, 0])).toBeNull(); // past its end
  expect(along([1, 1], [1, 1], [1, 1])).toBeNull(); // a wall with no length
});

it('makes a site, a building, and a level, with walls, rooms, and openings hung on their walls', () => {
  const scene = buildScene({
    height: 2.5,
    walls: [
      { from: [0, 0], to: [4, 0], thickness: 0.3 },
      { from: [4, 0], to: [4, 3], thickness: 0.3 },
    ],
    openings: [
      { kind: 'window', at: [1, 0], width: 1.2 },
      { kind: 'door', at: [4, 2], width: 0.8 },
      { kind: 'cased', at: [3, 0], width: 1 },
    ],
    rooms: [
      {
        id: 'den',
        name: 'Den',
        outline: [
          [0, 0],
          [4, 0],
          [4, 3],
          [0, 3],
        ],
      },
    ],
    extras: { cameras: { hall: { at: [1, 1], yaw: 90 } } },
  });
  expect(scene.rootNodeIds).toEqual(['site_main']);
  const level = scene.nodes.level_ground;
  expect(level?.parentId).toBe('building_main');
  expect(scene.nodes.building_main?.children).toEqual(['level_ground']);
  expect(level?.children).toEqual(['wall_0', 'wall_1', 'zone_den']);
  expect(level?.metadata).toEqual({ liberational: { cameras: { hall: { at: [1, 1], yaw: 90 } } } });
  expect(scene.nodes.wall_0).toMatchObject({ start: [0, 0], end: [4, 0], thickness: 0.3, height: 2.5 });
  // each opening is its wall's child, at its distance from the wall's start and its own height's middle
  expect(scene.nodes.wall_0?.children).toEqual(['window_0', 'door_2']);
  expect(scene.nodes.window_0).toMatchObject({ wallId: 'wall_0', position: [1, 1.5, 0], width: 1.2 });
  expect(scene.nodes.door_1).toMatchObject({ wallId: 'wall_1', position: [2, 1.05, 0], openingKind: 'door' });
  expect(scene.nodes.door_2).toMatchObject({ wallId: 'wall_0', position: [3, 1.05, 0], openingKind: 'opening' });
  expect(scene.nodes.zone_den).toMatchObject({ name: 'Den', parentId: 'level_ground' });
  // the site reaches 2 m past the walls
  expect(scene.nodes.site_main?.polygon).toEqual({
    type: 'polygon',
    points: [
      [-2, -2],
      [6, -2],
      [6, 5],
      [-2, 5],
    ],
  });
});

it('refuses an opening that is on no wall, saying which', () => {
  expect(() =>
    buildScene({
      height: 2.4,
      walls: [{ from: [0, 0], to: [4, 0], thickness: 0.3 }],
      openings: [{ kind: 'door', at: [2, 1], width: 0.8 }],
      rooms: [],
    }),
  ).toThrow('the door at 2, 1 is on no wall');
});

it('has the example flat as a scene: its six rooms, twelve openings, and camera spot', () => {
  expect(of(EXAMPLE_SCENE, 'zone').map(z => z.name)).toEqual([
    'Bedroom',
    'Hall',
    'Bathroom',
    'Living room',
    'Kitchen',
    'Study',
  ]);
  expect(of(EXAMPLE_SCENE, 'wall')).toHaveLength(9);
  expect(of(EXAMPLE_SCENE, 'window')).toHaveLength(6);
  expect(of(EXAMPLE_SCENE, 'door').map(d => d.openingKind)).toEqual([
    'door',
    'door',
    'opening',
    'opening',
    'door',
    'door',
  ]);
  expect(EXAMPLE_SCENE.nodes.level_ground?.metadata).toMatchObject({
    liberational: { cameras: { hall: { at: [0.4, 4.2], height: 2.2, yaw: 90 } } },
  });
  // every node's parent lists it, so the editor finds it
  for (const node of Object.values(EXAMPLE_SCENE.nodes))
    if (node.parentId) expect(EXAMPLE_SCENE.nodes[node.parentId]?.children).toContain(node.id);
});

it("writes the editor page's address and reads it back, falling back to the 2D editor in light", () => {
  expect(editorUrl('/static/app/', '3d', 'dark')).toBe('/static/app/plan-editor/index.html#view=3d&scheme=dark');
  for (const view of ['2d', '3d', 'split'] as const)
    for (const scheme of ['light', 'dark'] as const)
      expect(fromHash(new URL(editorUrl('/static/app/', view, scheme), 'http://x').hash)).toEqual({ view, scheme });
  expect(fromHash('')).toEqual({ view: '2d', scheme: 'light' });
  expect(fromHash('#view=4d&scheme=sepia')).toEqual({ view: '2d', scheme: 'light' });
});

it('takes a message only from its own origin and only of a type it asked for', () => {
  const event = (origin: string, data: unknown) => new MessageEvent('message', { origin, data });
  const ready = { type: 'plan-editor:ready' };
  expect(ours(event(location.origin, ready), ['plan-editor:ready'])).toEqual(ready);
  expect(ours(event('https://example.com', ready), ['plan-editor:ready'])).toBeNull();
  expect(ours(event(location.origin, ready), ['plan-editor:changed'])).toBeNull();
  expect(ours(event(location.origin, 'plan-editor:ready'), ['plan-editor:ready'])).toBeNull();
  expect(ours(event(location.origin, null), ['plan-editor:ready'])).toBeNull();
});

/** A message as the frame's own page would send it. */
function fromFrame(frame: HTMLIFrameElement, data: unknown, origin = location.origin) {
  act(() => {
    dispatchEvent(new MessageEvent('message', { origin, data, source: frame.contentWindow }));
  });
}

it("opens the editor's page in a frame, gives it the scene once it is up, and passes on each change", () => {
  const changed = vi.fn();
  renderPage(<PlanFrame scene={EXAMPLE_SCENE} view="2d" scheme="dark" onChange={changed} />);
  const frame = screen.getByTitle('Plan editor') as HTMLIFrameElement;
  expect(frame.getAttribute('src')).toBe(editorUrl(import.meta.env.BASE_URL, '2d', 'dark'));
  expect(screen.getByText('Loading the plan editor…')).toBeTruthy();
  const post = vi.spyOn(frame.contentWindow as Window, 'postMessage').mockImplementation(() => {});
  fromFrame(frame, { type: 'plan-editor:ready' });
  const open: Open = { type: 'plan-editor:open', scene: EXAMPLE_SCENE };
  expect(post).toHaveBeenCalledWith(open, location.origin);
  expect(screen.queryByText('Loading the plan editor…')).toBeNull();
  const edited = { ...EXAMPLE_SCENE, rootNodeIds: ['site_main'] };
  fromFrame(frame, { type: 'plan-editor:changed', scene: edited });
  expect(changed).toHaveBeenCalledWith(edited);
});

it('ignores a message that is not from its own frame, or from another origin', () => {
  const changed = vi.fn();
  renderPage(<PlanFrame scene={EXAMPLE_SCENE} view="3d" scheme="light" onChange={changed} />);
  const frame = screen.getByTitle('The plan in 3D') as HTMLIFrameElement;
  const post = vi.spyOn(frame.contentWindow as Window, 'postMessage').mockImplementation(() => {});
  const other = render(<iframe title="another frame" />).container.querySelector('iframe') as HTMLIFrameElement;
  act(() => {
    dispatchEvent(
      new MessageEvent('message', {
        origin: location.origin,
        data: { type: 'plan-editor:ready' },
        source: other.contentWindow,
      }),
    );
  });
  fromFrame(frame, { type: 'plan-editor:changed', scene: EXAMPLE_SCENE }, 'https://example.com');
  expect(post).not.toHaveBeenCalled();
  expect(changed).not.toHaveBeenCalled();
});
