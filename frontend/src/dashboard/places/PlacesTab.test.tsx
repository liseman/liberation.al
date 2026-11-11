import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { hm } from '../lib/format';
import type { PlaceSummary, Positions } from './api';
import { EXAMPLE_PLACES, type Place } from './examples';
import { lastVisit, linkedFrom, PlacesTab, sceneSummary } from './PlacesTab';
import { buildScene } from './scene/build';
import { drawScene } from './scene/draw';

const drawing = () => screen.getByRole('img', { name: /^Floor plan/ });
const rows = () => within(screen.getByRole('navigation', { name: 'Places' })).getAllByRole('button');
const row = (name: RegExp) => within(screen.getByRole('navigation', { name: 'Places' })).getByRole('button', { name });

/** A made-up home as a Pascal scene: a kitchen and a den, a door between them, and one camera spot. */
const SCENE = buildScene({
  height: 2.4,
  walls: [
    { from: [0, 0], to: [7, 0], thickness: 0.2 },
    { from: [4, 0], to: [4, 3], thickness: 0.1 },
  ],
  openings: [{ kind: 'door', at: [4, 1.5], width: 0.8 }],
  rooms: [
    {
      id: 'kitchen',
      name: 'Kitchen',
      outline: [
        [0, 0],
        [4, 0],
        [4, 3],
        [0, 3],
      ],
    },
    {
      id: 'den',
      name: 'Den',
      outline: [
        [4, 0],
        [7, 0],
        [7, 3],
        [4, 3],
      ],
    },
  ],
  extras: { cameras: { porch: { at: [1, 1], yaw: 90 } } },
});
const HOME_PLAN = JSON.stringify(SCENE);

function serve(places: PlaceSummary[] = [], positions?: (q: URLSearchParams) => Positions) {
  server.use(
    http.get('/places', () => HttpResponse.json({ places })),
    // no earlier owner plans, unless a test serves its own
    http.get('/places/plan/history', ({ request }) =>
      HttpResponse.json({ place: new URL(request.url).searchParams.get('place'), versions: [], previous: null }),
    ),
    http.get('/places/positions', ({ request }) => {
      const q = new URL(request.url).searchParams;
      return positions ? HttpResponse.json(positions(q)) : HttpResponse.json({ error: 'none' }, { status: 404 });
    }),
  );
}

beforeEach(() => {
  history.replaceState(null, '', '/places');
});

it("sums a plan's rooms and floor area on wall centrelines", () => {
  // The example flat's six rectangles: 3x3.6 + 3x2.4 + 3x3 + 5x4.4 + 5x2.2 + 5x2.4 = 72 m².
  const flat = EXAMPLE_PLACES.find(p => p.id === 'example-flat');
  const drawing = flat?.scene && drawScene(flat.scene, 'Flat');
  expect(drawing && sceneSummary(drawing)).toBe('6 rooms · 72.0 m²');
});

const HOME: PlaceSummary = {
  id: 'p1',
  name: 'Home',
  named: true,
  plan: HOME_PLAN,
  plan_source: 'machine',
  machine_newer: false,
  revision: { id: 'r1', at: 1762500000, producer: 'footprints-v1' },
  plan_revision: 'r1',
  machine_plan: null,
  last_visit: null,
};
const OFFICE: PlaceSummary = { ...HOME, id: 'p2', name: 'Office' };

it('lists the places the house knows, without the examples, and draws the one picked', async () => {
  serve([HOME, OFFICE]);
  renderPage(<PlacesTab />);
  await screen.findByRole('button', { name: /^Home/ });
  expect(rows().map(b => b.textContent)).toEqual([expect.stringContaining('Home'), expect.stringContaining('Office')]);
  expect(drawing().getAttribute('aria-label')).toBe('Floor plan of Home');
  fireEvent.click(row(/Office/));
  expect(drawing().getAttribute('aria-label')).toBe('Floor plan of Office');
  expect(row(/Office/).getAttribute('aria-current')).toBe('true');
});

it('shows the example plans while no place has a plan, after the places waiting for one', async () => {
  serve([{ ...HOME, name: 'Place 1', plan: null }]);
  renderPage(<PlacesTab />);
  expect(await screen.findByText(/No place has a floorplan yet/)).toBeTruthy();
  expect(rows().map(b => b.textContent)).toEqual([
    expect.stringContaining('Place 1'),
    expect.stringContaining('Example flat'),
    expect.stringContaining('Example cabin'),
  ]);
});

it("says whose plan it is: the machine's, with its revision's date, or the owner's correction", async () => {
  serve([HOME, { ...HOME, id: 'p3', name: 'Studio', plan_source: 'owner', machine_newer: true }]);
  renderPage(<PlacesTab />);
  expect(await screen.findByText(/^Machine-made plan, /)).toBeTruthy();
  fireEvent.click(await screen.findByRole('button', { name: /^Studio/ }));
  expect(screen.getByText(/^Your corrected plan \(a newer machine plan exists\)/)).toBeTruthy();
});

it('says when a place has no plan yet, or that its plan is in a format no longer read', async () => {
  serve([
    {
      ...HOME,
      name: 'Place 1',
      named: false,
      plan: null,
      plan_source: null,
      last_visit: Date.now() / 1000 - 3 * 86400,
    },
    { ...HOME, id: 'p2', name: 'Place 2', named: false, plan: 'frobnicate' },
  ]);
  renderPage(<PlacesTab />);
  expect(await screen.findByText(/No floorplan yet/)).toBeTruthy();
  expect(row(/Place 1/).textContent).toContain('Visited 3 d ago');
  fireEvent.click(row(/Place 2/));
  expect(screen.getByText(/can't be shown: it is in an old format that is no longer read/)).toBeTruthy();
});

it('keys the marks a plan draws, and shows no key for a plan with none', async () => {
  serve();
  renderPage(<PlacesTab />);
  const key = await screen.findByRole('list', { name: 'Key' });
  expect([...key.querySelectorAll('li')].map(li => li.textContent)).toEqual(['Camera']);
  fireEvent.click(row(/Example cabin/));
  expect(screen.queryByRole('list', { name: 'Key' })).toBeNull();
});

it("draws each person's and pet's path, and lights up the room of one known only by its room", async () => {
  serve();
  const { container } = renderPage(<PlacesTab />);
  await screen.findByRole('list', { name: 'Who' });
  const paths = () => container.querySelector('svg[class*="paths"]') as SVGSVGElement;
  // Dana and Sam are walking now (footprints); Biscuit's track ended earlier, so it has a path but no dot; Miso is
  // only known to be in the bedroom (a zone).
  expect(paths().querySelectorAll('circle')).toHaveLength(2);
  expect(paths().querySelectorAll('g').length).toBeGreaterThanOrEqual(3);
  expect(paths().querySelector('polygon title')?.textContent).toMatch(/^Miso in the bedroom/);
  const who = screen.getByRole('list', { name: 'Who' });
  expect(within(who).getByRole('button', { name: /Miso/ }).textContent).toContain('in the bedroom');
  fireEvent.click(within(who).getByRole('button', { name: /Dana/ }));
  expect(within(who).getByRole('button', { name: /Dana/ }).getAttribute('aria-pressed')).toBe('false');
  expect(paths().querySelectorAll('circle')).toHaveLength(1);
});

it('draws the paths of a place with no plan yet on plain floor fitted to them', async () => {
  // A lone camera's people have x and y on its floor without a plan
  serve([{ ...HOME, name: 'Place 3', named: false, plan: null, plan_source: null }], q => {
    const end = Number(q.get('end'));
    return {
      place: 'p1',
      start: Number(q.get('start')),
      end,
      entities: [{ id: 'pet:c', name: 'Cat', kind: 'pet' }],
      tracks: [
        {
          entity: 'pet:c',
          points: [
            { t: end - 90, x: 1, y: 1, room: null, confidence: 0.9, source: 'fixed-camera' },
            { t: end - 60, x: 2, y: 1, room: null, confidence: 0.9, source: 'fixed-camera' },
          ],
        },
      ],
      rooms: [],
    };
  });
  const { container } = renderPage(<PlacesTab />);
  expect(await screen.findByRole('button', { name: /Cat/ })).toBeTruthy();
  expect(screen.getByText(/No floorplan yet/)).toBeTruthy();
  // 3 m by 3 m of floor around the points (from 0, -0.5), 40 px a metre: (1, 1) is at 40, 60 and (2, 1) at 80, 60
  expect(screen.getByRole('img', { name: /Floor, a line every metre/ }).getAttribute('viewBox')).toBe('0 0 120 120');
  // one smooth curve from (40, 60) to (80, 60), straight between two points, with its arrowhead at (80, 60)
  const line = container.querySelector('svg[class*="paths"] path');
  expect(line?.getAttribute('d')).toBe('M40.0,60.0C53.3,60.0 66.7,60.0 80.0,60.0');
  expect(container.querySelector('svg[class*="paths"] polygon')?.getAttribute('points')).toMatch(/^80\.0,60\.0 /);
});

it("reads a real place's positions for the window and draws them", async () => {
  const asked: URLSearchParams[] = [];
  serve([HOME], q => {
    asked.push(q);
    const start = Number(q.get('start'));
    const end = Number(q.get('end'));
    return {
      place: 'p1',
      start,
      end,
      entities: [{ id: 'person:a', name: 'Alice', kind: 'person' }],
      tracks: [
        {
          entity: 'person:a',
          points: [
            { t: end - 90, x: 3, y: 1, room: 'kitchen', confidence: 0.9, source: 'fixed-camera' },
            { t: end - 60, x: 4.5, y: 1.5, room: 'den', confidence: 0.9, source: 'fixed-camera' },
          ],
        },
      ],
      rooms: [],
    };
  });
  const { container } = renderPage(<PlacesTab />);
  expect(await screen.findByRole('button', { name: /Alice/ })).toBeTruthy();
  const q = asked[0] as URLSearchParams;
  expect(q.get('place')).toBe('p1');
  expect(Number(q.get('end')) - Number(q.get('start'))).toBe(86400); // a day, by default
  expect(container.querySelectorAll('svg[class*="paths"] path').length).toBeGreaterThan(0);
  expect(container.querySelector('svg[class*="paths"] g > title')?.textContent).toMatch(/^Alice, /); // named on hover
});

it('opens the place and moment a link names', async () => {
  serve([HOME, OFFICE]);
  const t = Math.floor(Date.now() / 1000) - 5 * 3600;
  history.replaceState(null, '', `/places?place=p2&t=${t}`);
  renderPage(<PlacesTab />);
  await screen.findByRole('button', { name: /^Office/ });
  expect(drawing().getAttribute('aria-label')).toBe('Floor plan of Office');
  expect(screen.getByRole('status').textContent).toContain(hm(t)); // with its weekday: a day's two ends read alike
});

it('reads a link from the address, ignoring a time that is not one', () => {
  expect(linkedFrom('?place=p1&t=1762500000')).toEqual({ place: 'p1', t: 1762500000 });
  expect(linkedFrom('?t=soon')).toEqual({ place: null, t: null });
  expect(linkedFrom('')).toEqual({ place: null, t: null });
});

it('shows the examples it is given as the empty state', async () => {
  serve();
  const [flat] = EXAMPLE_PLACES as [Place];
  renderPage(<PlacesTab examples={[{ ...flat, name: 'Only this' }]} />);
  await screen.findByText(/No place has a floorplan yet/);
  expect(rows().map(b => b.textContent)).toEqual([expect.stringContaining('Only this')]);
});

it('reads a window once: scrubbing and toggling people never read it again, however long it stays open', async () => {
  const asked: string[] = [];
  serve([HOME], q => {
    asked.push(q.toString());
    const start = Number(q.get('start'));
    const end = Number(q.get('end'));
    return {
      place: 'p1',
      start,
      end,
      entities: [{ id: 'person:a', name: 'Alice', kind: 'person' }],
      tracks: [
        {
          entity: 'person:a',
          points: [
            { t: end - 90, x: 3, y: 1, room: 'kitchen', confidence: 0.9, source: 'fixed-camera' },
            { t: end - 60, x: 4.5, y: 1.5, room: 'den', confidence: 0.9, source: 'fixed-camera' },
          ],
        },
      ],
      rooms: [],
    };
  });
  renderPage(<PlacesTab />);
  const alice = await screen.findByRole('button', { name: /Alice/ });
  await new Promise(r => setTimeout(r, 1100)); // a second goes by: a window that followed the clock would move
  const slider = screen.getByRole('slider', { name: 'Time' });
  fireEvent.keyDown(slider, { key: 'ArrowLeft' });
  fireEvent.keyDown(slider, { key: 'ArrowLeft' });
  fireEvent.click(alice);
  fireEvent.click(alice);
  await new Promise(r => setTimeout(r, 50));
  expect(asked).toHaveLength(1);
});

it('lists everyone not identified yet as one legend entry that hides or shows them all', async () => {
  serve([HOME], q => {
    const end = Number(q.get('end'));
    const strangers = Array.from({ length: 40 }, (_, i) => `cam0:${i}`);
    return {
      place: 'p1',
      start: Number(q.get('start')),
      end,
      entities: [
        { id: 'person:a', name: 'Alice', kind: 'person' },
        ...strangers.map(id => ({ id, name: null, kind: 'unknown' as const })),
      ],
      tracks: [
        ...['person:a', ...strangers].map(entity => ({
          entity,
          points: [
            { t: end - 120, x: 1, y: 1, room: 'kitchen', confidence: 0.9, source: 'fixed-camera' },
            { t: end - 60, x: 1.5, y: 1.2, room: 'kitchen', confidence: 0.9, source: 'fixed-camera' },
          ],
        })),
      ],
      rooms: [],
    };
  });
  const { container } = renderPage(<PlacesTab />);
  const who = await screen.findByRole('list', { name: 'Who' });
  await within(who).findByRole('button', { name: /Alice/ });
  expect(
    within(who)
      .getAllByRole('button')
      .map(b => b.textContent),
  ).toEqual(['Alice', 'Not identified yet · 40 tracks']);
  const dots = () => container.querySelectorAll('svg[class*="paths"] circle').length;
  expect(dots()).toBe(41);
  fireEvent.click(within(who).getByRole('button', { name: /Not identified yet/ }));
  expect(dots()).toBe(1);
});

it("opens the plan editor from Edit plan, the one way in, and closes it back to the plan's paths", async () => {
  serve([HOME]);
  server.use(http.get('/places/plan/scene', () => HttpResponse.json({ place: 'p1', scene: SCENE, base: 'r1' })));
  const { container } = renderPage(<PlacesTab />);
  await screen.findByRole('img', { name: 'Floor plan of Home' });
  expect(container.querySelector('iframe')).toBeNull(); // nothing of the editor loads until it is asked for
  expect(screen.getAllByRole('button', { name: 'Edit plan' })).toHaveLength(1);
  fireEvent.click(screen.getByRole('button', { name: 'Edit plan' }));
  expect((await screen.findByTitle('Plan editor')).getAttribute('src')).toContain('#view=2d');
  expect(screen.getByRole('button', { name: 'Save plan' })).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Edit plan' })).toBeNull();
  // and the view switch goes while it is open: Close editor is the one way out
  expect(screen.queryByRole('radiogroup', { name: 'Show the place as' })).toBeNull();
  // the facts stay where they were, under the place's name
  expect(screen.getByText(/Machine-made plan.*2 rooms · 21.0 m²/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Close editor' }));
  expect(await screen.findByRole('img', { name: 'Floor plan of Home' })).toBeTruthy();
  expect(container.querySelector('iframe')).toBeNull();
});

it('asks before leaving the editor with unsaved changes, for another place or by Close editor', async () => {
  serve([SCENED, OFFICE]);
  server.use(http.get('/places/plan/scene', () => HttpResponse.json({ place: 'p1', scene: SCENE, base: 'r1' })));
  renderPage(<PlacesTab />);
  fireEvent.click(await screen.findByRole('button', { name: 'Edit plan' }));
  const frame = (await screen.findByTitle('Plan editor')) as HTMLIFrameElement;
  act(() => {
    dispatchEvent(
      new MessageEvent('message', {
        origin: location.origin,
        data: { type: 'plan-editor:changed', scene: SCENE },
        source: frame.contentWindow,
      }),
    );
  });
  expect(await screen.findByText('Unsaved changes')).toBeTruthy();
  fireEvent.click(row(/Office/));
  const dialog = await screen.findByRole('dialog', { name: 'Discard your unsaved changes?' });
  fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  expect(screen.getByTitle('Plan editor')).toBe(frame); // still editing, with the changes
  fireEvent.click(screen.getByRole('button', { name: 'Close editor' }));
  fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Discard changes' }));
  await waitFor(() => expect(screen.queryByTitle('Plan editor')).toBeNull());
});

it("shows when a newer machine plan exists, compares the two, and takes the machine's on confirm", async () => {
  const machine = HOME_PLAN.replace('"Kitchen"', '"Galley"'); // the machine's own plan, with one room named otherwise
  const posted: unknown[] = [];
  serve([{ ...HOME, plan_source: 'owner', machine_newer: true, machine_plan: machine }]);
  server.use(
    http.post('/places/plan', async ({ request }) => {
      posted.push(await request.json());
      return HttpResponse.json({ ok: true, plan: machine, plan_source: 'machine' });
    }),
  );
  renderPage(<PlacesTab />);
  expect(await screen.findByText('A newer machine plan exists')).toBeTruthy();
  fireEvent.click(screen.getByRole('radio', { name: "The machine's" }));
  expect(screen.getByText(/The newest machine plan, to compare/)).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Edit plan' })).toBeDisabled(); // you edit your own, not the machine's
  fireEvent.click(screen.getByRole('button', { name: "Use the machine's plan" }));
  const dialog = await screen.findByRole('dialog', { name: "Use the machine's plan?" });
  expect(dialog.textContent).toContain('stays in its versions');
  fireEvent.click(within(dialog).getByRole('button', { name: "Use the machine's plan" }));
  await screen.findByText('A newer machine plan exists'); // the page reads the places again
  await new Promise(r => setTimeout(r, 0));
  expect(posted).toEqual([{ place: 'p1', plan: '' }]);
});

it("switches a real place between its plan and its map in 3D, but not an example's", async () => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as RenderingContext);
  serve([HOME]);
  server.use(http.get('/places/scene', () => HttpResponse.json({ place: 'p1', revision: null })));
  renderPage(<PlacesTab />);
  await screen.findByRole('button', { name: 'Edit plan' });
  fireEvent.click(screen.getByRole('radio', { name: '3D map' }));
  expect(await screen.findByText(/Nothing to show in 3D here yet/)).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Edit plan' })).toBeTruthy(); // the way into editing stays put
  fireEvent.click(screen.getByRole('radio', { name: 'Plan' }));
  expect(await screen.findByRole('button', { name: 'Edit plan' })).toBeTruthy();
  vi.restoreAllMocks();
});

it('offers no map in 3D for an example plan', async () => {
  serve([]);
  renderPage(<PlacesTab />);
  await screen.findAllByRole('img', { name: /^Floor plan/ });
  fireEvent.click(row(/Example cabin/));
  expect(screen.queryByRole('radio', { name: '3D map' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Edit plan' })).toBeTruthy();
});

it('starts an empty plan in the editor for a place that has none, to draw one from nothing', async () => {
  serve([{ ...HOME, plan: null, plan_source: null }, OFFICE]);
  server.use(
    http.get('/places/plan/scene', () => HttpResponse.json({ error: 'this place has no plan' }, { status: 404 })),
  );
  renderPage(<PlacesTab />);
  expect(await screen.findByText(/No floorplan yet/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Edit plan' }));
  expect(await screen.findByTitle('Plan editor')).toBeTruthy();
  expect(screen.getByText('No plan yet. Draw rooms and walls, and put doors and windows in them.')).toBeTruthy();
  expect((screen.getByRole('button', { name: 'Save plan' }) as HTMLButtonElement).disabled).toBe(true);
});

it('opens the example flat in the plan editor, with nothing to save and no view to switch to', async () => {
  serve([]);
  const { container } = renderPage(<PlacesTab />);
  await screen.findAllByRole('img', { name: /^Floor plan/ });
  expect(screen.queryByRole('radiogroup', { name: 'Show the place as' })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Edit plan' }));
  expect(screen.getByTitle('Plan editor').getAttribute('src')).toContain('plan-editor/index.html#view=2d');
  expect(screen.getByText(/Nothing drawn here is saved/)).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Save plan' })).toBeNull(); // an example has nothing to save to
  fireEvent.click(screen.getByRole('button', { name: 'Close editor' }));
  expect(container.querySelector('iframe')).toBeNull();
  expect(drawing()).toBeTruthy();
});

const SCENED: PlaceSummary = { ...HOME, plan_source: 'owner' };

it('draws a plan stored as a scene from the scene, with its rooms, area, and camera key', async () => {
  serve([SCENED]);
  renderPage(<PlacesTab />);
  expect(await screen.findByRole('img', { name: 'Floor plan of Home' })).toBeTruthy();
  // 4x3 + 3x3 = 21 m², in the list row and under the place's name
  expect(row(/Home/).textContent).toContain('2 rooms · 21.0 m²');
  expect(screen.getByText(/Your corrected plan · 2 rooms · 21.0 m²/)).toBeTruthy();
  expect(within(screen.getByRole('list', { name: 'Key' })).getByText('Camera')).toBeTruthy();
});

it('says a failed positions read with a way to try again', async () => {
  let reads = 0;
  serve([HOME]);
  server.use(
    http.get('/places/positions', () => {
      reads += 1;
      return HttpResponse.json({ error: 'the reader is busy' }, { status: 503 });
    }),
  );
  renderPage(<PlacesTab />);
  expect(await screen.findByText("Couldn't load where people were")).toBeTruthy();
  const before = reads;
  fireEvent.click(screen.getByRole('button', { name: /Retry|Try again/ }));
  await waitFor(() => expect(reads).toBeGreaterThan(before));
});

it('keeps the view switch for looking: the plan and the map in 3D, and nothing that edits', async () => {
  serve([HOME]);
  renderPage(<PlacesTab />);
  await screen.findByRole('button', { name: 'Edit plan' });
  const views = within(screen.getByRole('radiogroup', { name: 'Show the place as' })).getAllByRole('radio');
  expect(views.map(r => (r as HTMLInputElement).value)).toEqual(['plan', 'map']);
});

it("says when a place was last in use: the owner's visit, or anyone its cameras saw since", () => {
  // A place only fixed cameras watch has no visits, and read "No visits yet" with hundreds of tracks.
  const now = Date.now() / 1000;
  const at = (visit: number | null, activity: number | null) =>
    lastVisit({ id: 'p', name: 'P', scene: null, example: false, lastVisit: visit, lastActivity: activity });
  expect(at(null, now - 120)).toBe('Activity 2 min ago');
  expect(at(now - 3 * 86400, now - 120)).toBe('Activity 2 min ago');
  expect(at(now - 120, now - 3 * 86400)).toBe('Visited 2 min ago');
  expect(at(null, null)).toBe('Nothing seen yet');
});

type Note = { camera: string; by: 'spot' | 'map'; missing: string[] };
const empty =
  (cameras: number, camera_notes: Note[] = [], map = true) =>
  (q: URLSearchParams) => ({
    map,
    place: q.get('place') ?? '',
    start: Number(q.get('start')),
    end: Number(q.get('end')),
    entities: [],
    tracks: [],
    rooms: [],
    cameras,
    camera_notes,
  });

it('says nobody can be placed while no camera has a spot or a pose, and only then', async () => {
  serve([HOME], empty(0));
  const first = renderPage(<PlacesTab />);
  expect(
    await screen.findByText(/No camera has a spot on this place's plan or a fixed pose in its map yet/),
  ).toBeTruthy();
  // with nobody placeable there is no time to move through: the window and the slider are not shown
  expect(screen.queryByRole('radiogroup', { name: 'How far back' })).toBeNull();
  expect(screen.queryByRole('slider')).toBeNull();
  first.unmount();
  serve([HOME], empty(2));
  renderPage(<PlacesTab />);
  expect(await screen.findByText('Nobody was placed here in this window.')).toBeTruthy();
  expect(screen.queryByText(/No camera has a spot/)).toBeNull();
  expect(screen.getByRole('radiogroup', { name: 'How far back' })).toBeTruthy();
});

it('says what each camera with a spot is missing, and not that no camera has a spot', async () => {
  serve(
    [HOME],
    empty(1, [
      { camera: 'hall', by: 'spot', missing: [] },
      { camera: 'porch', by: 'spot', missing: ['facing'] },
      { camera: 'attic', by: 'spot', missing: ['height', 'tilt'] },
    ]),
  );
  renderPage(<PlacesTab />);
  expect(await screen.findByText(/porch can't place anyone yet: its spot on the plan has no facing/)).toBeTruthy();
  expect(
    screen.getByText(/attic can't place anyone yet: its spot on the plan has no height; it hasn't seen/),
  ).toBeTruthy();
  expect(screen.queryByText(/hall can't place/)).toBeNull();
  expect(screen.queryByText(/No camera has a spot/)).toBeNull();
  expect(screen.getByText('Nobody was placed here in this window.')).toBeTruthy();
});

it('says a place has no map yet when it has none and no camera has a spot', async () => {
  serve([HOME], empty(0, [], false));
  renderPage(<PlacesTab />);
  expect(await screen.findByText(/This place has no map yet, and no camera has a spot on its plan/)).toBeTruthy();
});

it('says every camera is missing something when none can place anyone', async () => {
  serve([HOME], empty(0, [{ camera: 'porch', by: 'spot', missing: ['facing'] }]));
  renderPage(<PlacesTab />);
  expect(
    await screen.findByText(/Nobody can be placed here yet: each camera of this place is missing something/),
  ).toBeTruthy();
  expect(screen.getByText(/porch can't place anyone yet/)).toBeTruthy();
});
