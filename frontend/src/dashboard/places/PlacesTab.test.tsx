import { fireEvent, screen, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { hm } from '../lib/format';
import type { PlaceSummary, Positions } from './api';
import { EXAMPLE_PLACES, type Place } from './examples';
import { drawPlan, lastVisit, linkedFrom, PlacesTab, planSummary } from './PlacesTab';

const drawing = () => screen.getByRole('img', { name: /^Floor plan/ });
const rows = () => within(screen.getByRole('navigation', { name: 'Places' })).getAllByRole('button');
const row = (name: RegExp) => within(screen.getByRole('navigation', { name: 'Places' })).getByRole('button', { name });

const HOME_PLAN =
  'plan "Home" walls 0.2/0.1\nroom kitchen "Kitchen" kitchen rect 0,0 4x3\nroom den "Den" living rect 4,0 3x3\n';

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

it("sums a plan's rooms and interior area on wall centrelines", () => {
  // The example flat's six rectangles: 3x3.6 + 3x2.4 + 3x3 + 5x4.4 + 5x2.2 + 5x2.4 = 72 m².
  const flat = EXAMPLE_PLACES.find(p => p.id === 'example-flat');
  expect(planSummary(drawPlan(flat?.plan ?? {}, 'light').result)).toBe('6 rooms · 72.0 m²');
});

it("draws the plan in the app's own light or dark, not the system's", () => {
  const plan = EXAMPLE_PLACES[0]?.plan ?? {};
  expect(drawPlan(plan, 'dark').svg).toContain('theme-dark');
  expect(drawPlan(plan, 'light').svg).toContain('theme-light');
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
const OFFICE: PlaceSummary = { ...HOME, id: 'p2', name: 'Office', plan: HOME_PLAN.replace('"Home"', '"Office"') };

it('lists the places the house knows, without the examples, and draws the one picked', async () => {
  serve([HOME, OFFICE]);
  renderPage(<PlacesTab />);
  await screen.findByRole('button', { name: /^Home/ });
  expect(rows().map(b => b.textContent)).toEqual([expect.stringContaining('Home'), expect.stringContaining('Office')]);
  expect(drawing().getAttribute('aria-label')).toMatch(/^Floor plan Home/);
  fireEvent.click(row(/Office/));
  expect(drawing().getAttribute('aria-label')).toMatch(/^Floor plan Office/);
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

it("says when a place has no plan yet, or why its plan couldn't be read", async () => {
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
  expect(screen.getByText(/couldn't be read: line 1/)).toBeTruthy();
});

it('says which revision a plan is, and keys the marks its overlay draws', async () => {
  serve();
  renderPage(<PlacesTab />);
  expect(await screen.findByText(/Revision 2, from 2 visits/)).toBeTruthy();
  const key = screen.getByRole('list', { name: 'Key' });
  expect([...key.querySelectorAll('li')].map(li => li.textContent)).toEqual([
    'You corrected',
    'Machine, unsure',
    'Partly seen',
    'Not seen yet',
    'Camera',
  ]);
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
  expect(drawing().getAttribute('aria-label')).toMatch(/^Floor plan Office/);
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

it('opens the editor on a machine plan, and Cancel goes back to the paths', async () => {
  serve([HOME]);
  renderPage(<PlacesTab />);
  fireEvent.click(await screen.findByRole('button', { name: 'Edit plan' }));
  expect(screen.getByRole('button', { name: 'Save my plan' })).toBeTruthy();
  expect(screen.queryByRole('list', { name: 'Who' })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(screen.queryByRole('button', { name: 'Save my plan' })).toBeNull();
});

it("shows when a newer machine plan exists, compares the two, and reverts to the machine's on confirm", async () => {
  const machine = HOME_PLAN.replace('"Kitchen"', '"Galley"');
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
  fireEvent.click(await screen.findByRole('button', { name: "Revert to the machine's plan" }));
  fireEvent.click(await screen.findByRole('button', { name: 'Revert' }));
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
  fireEvent.click(screen.getByRole('radio', { name: '3D' }));
  expect(await screen.findByText(/No 3D map of this place yet/)).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Edit plan' })).toBeNull();
  fireEvent.click(screen.getByRole('radio', { name: 'Plan' }));
  expect(await screen.findByRole('button', { name: 'Edit plan' })).toBeTruthy();
  vi.restoreAllMocks();
});

it("offers no map in 3D for an example plan, and no editor for one that isn't a scene yet", async () => {
  serve([]);
  renderPage(<PlacesTab />);
  await screen.findAllByRole('img', { name: /^Floor plan/ });
  fireEvent.click(row(/Example cabin/));
  expect(screen.queryByRole('radio', { name: '3D' })).toBeNull();
  expect(screen.queryByRole('radio', { name: 'Editor' })).toBeNull();
});

it('opens the example flat in the new plan editor and in its 3D view, each in a frame made only then', async () => {
  serve([]);
  const { container } = renderPage(<PlacesTab />);
  await screen.findAllByRole('img', { name: /^Floor plan/ });
  expect(container.querySelector('iframe')).toBeNull(); // nothing of the editor loads until it is asked for
  fireEvent.click(screen.getByRole('radio', { name: 'Editor' }));
  expect(screen.getByTitle('Plan editor').getAttribute('src')).toContain('plan-editor/index.html#view=2d');
  expect(screen.getByText(/nothing drawn here is saved/)).toBeTruthy();
  fireEvent.click(screen.getByRole('radio', { name: '3D' }));
  expect(screen.getByTitle('The plan in 3D').getAttribute('src')).toContain('plan-editor/index.html#view=3d');
  fireEvent.click(screen.getByRole('radio', { name: 'Plan' }));
  expect(container.querySelector('iframe')).toBeNull();
  expect(drawing()).toBeTruthy();
});

it('offers the new plan editor on an example beside real plans, as a preview that can be hidden again', async () => {
  serve([HOME]);
  const { container } = renderPage(<PlacesTab />);
  await screen.findByRole('button', { name: 'Edit plan' });
  expect(rows()).toHaveLength(1); // the real place alone: no example is listed until it is asked for
  fireEvent.click(screen.getByRole('button', { name: 'Try the new plan editor on an example' }));
  expect(rows().map(r => r.textContent)).toEqual([
    expect.stringContaining('Home'),
    expect.stringContaining('Example flat'),
  ]);
  expect(screen.getByTitle('Plan editor').getAttribute('src')).toContain('plan-editor/index.html#view=2d');
  expect(screen.getByText(/doesn't touch your own plans, and nothing drawn here is saved/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Hide the example' }));
  expect(rows()).toHaveLength(1);
  expect(container.querySelector('iframe')).toBeNull();
  expect(await screen.findByRole('button', { name: 'Edit plan' })).toBeTruthy(); // back on the real place's plan
});

it('offers no preview button while the examples are already listed', async () => {
  serve([]);
  renderPage(<PlacesTab />);
  await screen.findAllByRole('img', { name: /^Floor plan/ });
  expect(screen.queryByRole('button', { name: /Try the new plan editor/ })).toBeNull();
});

it("says when a place was last in use: the owner's visit, or anyone its cameras saw since", () => {
  // A place only fixed cameras watch has no visits, and read "No visits yet" with hundreds of tracks.
  const now = Date.now() / 1000;
  const at = (visit: number | null, activity: number | null) =>
    lastVisit({ id: 'p', name: 'P', plan: null, example: false, lastVisit: visit, lastActivity: activity });
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
  first.unmount();
  serve([HOME], empty(2));
  renderPage(<PlacesTab />);
  expect(await screen.findByText('Nobody was placed here in this window.')).toBeTruthy();
  expect(screen.queryByText(/No camera has a spot/)).toBeNull();
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

it('says a camera on something that moves is left unplaced, and why', async () => {
  serve([HOME], empty(0, [{ camera: 'cart', by: 'map', missing: ['moving'] }]));
  renderPage(<PlacesTab />);
  expect(await screen.findByText(/cart can't place anyone yet: it's on something that moves/)).toBeTruthy();
});

it('says every camera is missing something when none can place anyone', async () => {
  serve([HOME], empty(0, [{ camera: 'porch', by: 'spot', missing: ['facing'] }]));
  renderPage(<PlacesTab />);
  expect(
    await screen.findByText(/Nobody can be placed here yet: each camera of this place is missing something/),
  ).toBeTruthy();
  expect(screen.getByText(/porch can't place anyone yet/)).toBeTruthy();
});
