import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { expect, it, vi } from 'vitest';
import { editorUrl, fromHash, type Open, ours, type Scene } from '../../../plan-editor/protocol';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { EXAMPLE_PLACES, type Place } from '../examples';
import { along, buildScene } from './build';
import { centre, drawScene, spots } from './draw';
import { EXAMPLE_SCENE } from './example';
import { PlaceEditor } from './PlaceEditor';
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

it('measures a room: its floor area and the middle of its floor', () => {
  expect(
    centre([
      [0, 0],
      [4, 0],
      [4, 3],
      [0, 3],
    ]),
  ).toEqual({ area: 12, at: [2, 1.5] });
  // an L: a 4x4 square less its 2x2 corner is 12 m², and its middle sits toward the full side
  const l = centre([
    [0, 0],
    [4, 0],
    [4, 2],
    [2, 2],
    [2, 4],
    [0, 4],
  ]);
  expect(l.area).toBe(12);
  expect(l.at[0]).toBeCloseTo(5 / 3);
  expect(l.at[1]).toBeCloseTo(5 / 3);
  expect(
    centre([
      [1, 1],
      [3, 1],
    ]),
  ).toEqual({ area: 0, at: [2, 1] }); // no floor: the corners' middle
});

it("draws a scene's rooms, walls, openings, and camera spots, and says where a plan point lands", () => {
  const drawing = drawScene(EXAMPLE_SCENE, 'Flat & "co"');
  if (!drawing) throw new Error('nothing drawn');
  expect(drawing.rooms).toBe(6);
  expect(drawing.area).toBeCloseTo(72); // the example's six rectangles
  expect(Object.keys(drawing.cameras)).toEqual(['hall']);
  const doc = new DOMParser().parseFromString(drawing.svg, 'image/svg+xml');
  expect(doc.documentElement.getAttribute('aria-label')).toBe('Floor plan of Flat & "co"'); // escaped, and read back
  expect(doc.querySelectorAll('.room')).toHaveLength(6);
  expect(doc.querySelectorAll('.wall')).toHaveLength(9);
  expect(doc.querySelectorAll('.gap')).toHaveLength(12); // every door, doorway, and window opens its wall
  expect(doc.querySelectorAll('.window')).toHaveLength(6);
  expect([...doc.querySelectorAll('.name')].map(t => t.textContent)).toContain('Living room');
  expect(doc.querySelectorAll('.camera')).toHaveLength(1);
  expect(doc.querySelectorAll('.view')).toHaveLength(1); // the spot has a facing
  // 60 units a metre, with 0.6 m of floor around the plan, whose corner is (0, 0)
  expect(drawing.surface.toScreen([0, 0])).toEqual([36, 36]);
  expect(drawing.surface.toScreen([8, 9])).toEqual([516, 576]);
  expect(doc.documentElement.getAttribute('viewBox')).toBe('0 0 552.0 612.0');
  expect(drawing.surface.rooms.map(r => r.id)).toContain('kitchen');
  // the front door is 0.9 m wide at (0, 4.8): its gap runs from 4.35 to 5.25 down the west wall
  const ends = [...doc.querySelectorAll('.gap')].map(g => ['x1', 'y1', 'x2', 'y2'].map(k => Number(g.getAttribute(k))));
  expect(
    ends.map(([x1, y1, x2, y2]) => [x1, Math.min(y1 ?? 0, y2 ?? 0), x2, Math.max(y1 ?? 0, y2 ?? 0)]),
  ).toContainEqual([36, 297, 36, 351]);
});

it('draws nothing for a scene with nothing on its level, and leaves out a spot with no position', () => {
  expect(drawScene({ nodes: {}, rootNodeIds: [] }, 'Empty')).toBeNull();
  const bare = buildScene({ height: 2.4, walls: [], openings: [], rooms: [] });
  expect(drawScene(bare, 'Bare')).toBeNull();
  const spotted = buildScene({
    height: 2.4,
    walls: [],
    openings: [],
    rooms: [],
    extras: { cameras: { a: { at: [1, 2] }, b: { height: 2 }, c: 'shelf' } },
  });
  expect(spots(spotted)).toEqual({ a: { at: [1, 2], height: undefined, yaw: undefined, fov: undefined } });
  const drawing = drawScene(spotted, 'Spot');
  expect(drawing?.svg).toContain('class="camera"');
  expect(drawing?.svg).not.toContain('class="view"'); // no facing, no view drawn
});

const HOME: Place = { id: 'p1', name: 'Home', lastVisit: null, plan: null, scene: EXAMPLE_SCENE, example: false };
const FLAT = EXAMPLE_PLACES.find(p => p.scene) as Place;

function serveScene(reply: Record<string, unknown> = {}) {
  server.use(
    http.get('/places/plan/scene', () =>
      HttpResponse.json({ place: 'p1', scene: EXAMPLE_SCENE, converted: false, left: [], base: 'r1', ...reply }),
    ),
  );
}

it("saves what was drawn as the place's plan, in the frame it was opened in, and says it is saved", async () => {
  serveScene();
  const sent: unknown[] = [];
  server.use(
    http.post('/places/plan', async ({ request }) => {
      sent.push(await request.json());
      return HttpResponse.json({ ok: true, plan: '{}', plan_source: 'owner' });
    }),
  );
  const saved = vi.fn();
  renderPage(<PlaceEditor place={HOME} view="2d" scheme="light" onSaved={saved} />);
  const frame = (await screen.findByTitle('Plan editor')) as HTMLIFrameElement;
  const save = screen.getByRole('button', { name: 'Save plan' }) as HTMLButtonElement;
  expect(save.disabled).toBe(true); // nothing drawn yet
  const edited = { ...EXAMPLE_SCENE, rootNodeIds: ['site_main'] };
  fromFrame(frame, { type: 'plan-editor:changed', scene: edited });
  expect(screen.getByText('Changes not saved yet.')).toBeTruthy();
  fireEvent.click(save);
  await waitFor(() => expect(saved).toHaveBeenCalled());
  expect(sent).toEqual([{ place: 'p1', plan: JSON.stringify(edited), base: 'r1' }]);
});

it('says a plan in the old format is converted by saving, with the lines the new one has no place for', async () => {
  serveScene({ converted: true, left: ['x-anchor place:p1 yaw 0 origin 0,0,0', 'x-source wall:w3 machine 0.4'] });
  renderPage(<PlaceEditor place={HOME} view="2d" scheme="light" />);
  expect(await screen.findByText(/stored in the old format. Saving it here stores it in the new one/)).toBeTruthy();
  expect(screen.getByText(/2 lines of the old plan have no place in the new format/).textContent).toContain(
    'x-source wall:w3 machine 0.4',
  );
  // it can be saved as it opened, with nothing drawn: that is the conversion
  const sent: unknown[] = [];
  server.use(
    http.post('/places/plan', async ({ request }) => {
      sent.push(await request.json());
      return HttpResponse.json({ ok: true, plan: '{}', plan_source: 'owner' });
    }),
  );
  fireEvent.click(screen.getByRole('button', { name: 'Save plan' }));
  await waitFor(() => expect(sent).toEqual([{ place: 'p1', plan: JSON.stringify(EXAMPLE_SCENE), base: 'r1' }]));
});

it('opens the stored plan again on Discard, and refuses to save a plan with no room', async () => {
  serveScene();
  renderPage(<PlaceEditor place={HOME} view="2d" scheme="light" />);
  const frame = (await screen.findByTitle('Plan editor')) as HTMLIFrameElement;
  const roomless = buildScene({
    height: 2.4,
    walls: [{ from: [0, 0], to: [4, 0], thickness: 0.2 }],
    openings: [],
    rooms: [],
  });
  fromFrame(frame, { type: 'plan-editor:changed', scene: roomless });
  expect(screen.getByText('A plan needs at least one room before it can be saved.')).toBeTruthy();
  expect((screen.getByRole('button', { name: 'Save plan' }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: 'Discard changes' }));
  expect(screen.queryByText(/at least one room/)).toBeNull();
  expect(screen.getByTitle('Plan editor')).not.toBe(frame); // a new frame, on the stored plan
});

it("says why when a plan can't be opened or saved", async () => {
  server.use(
    http.get('/places/plan/scene', () =>
      HttpResponse.json({ error: "this plan can't be opened in the editor: the plan has no rooms" }, { status: 422 }),
    ),
  );
  const first = renderPage(<PlaceEditor place={HOME} view="2d" scheme="light" />);
  expect(await screen.findByText(/can't be opened in the editor: the plan has no rooms/)).toBeTruthy();
  first.unmount();
  serveScene();
  server.use(
    http.post('/places/plan', () =>
      HttpResponse.json({ ok: false, error: 'the plan has no room in it' }, { status: 400 }),
    ),
  );
  renderPage(<PlaceEditor place={HOME} view="2d" scheme="light" />);
  const frame = (await screen.findByTitle('Plan editor')) as HTMLIFrameElement;
  fromFrame(frame, { type: 'plan-editor:changed', scene: EXAMPLE_SCENE });
  fireEvent.click(screen.getByRole('button', { name: 'Save plan' }));
  expect(await screen.findByText(/Couldn't save the plan: the plan has no room in it/)).toBeTruthy();
  expect(screen.getByText('Changes not saved yet.')).toBeTruthy(); // nothing drawn is lost
});

it('opens an example without asking the server, and offers nothing to save', () => {
  renderPage(<PlaceEditor place={FLAT} view="3d" scheme="dark" />);
  expect(screen.getByTitle('The plan in 3D')).toBeTruthy();
  expect(screen.getByText('An example flat. Nothing drawn here is saved.')).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Save plan' })).toBeNull();
});
