// The 3D view's splat states, with the viewer itself stood in for (jsdom can't draw): what the page says and what
// it asks the viewer to draw in each, and that none of them is left to the console (#246).
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { SCENE_IDLE_MS, SCENE_LEARNING_MS, sceneQuery } from '../api';
import CloudView from './CloudView';
import type { Built, View } from './scene';

const build = vi.hoisted(() => vi.fn());
vi.mock('./scene', () => ({ build, ABOVE: '@above' }));

/** What the stood-in viewer was asked to show, newest last, and how its splat's load ends. */
let shown: Parameters<Built['show']>[0][] = [];
/** The cameras the stood-in viewer was asked to look from. */
let lookedFrom: string[] = [];
let splatLoaded: () => Promise<void> = () => Promise.resolve();
/** The splat's own load is only followed for a viewer built around one, as scene.ts settles it at once without. */
type BuildArgs = [HTMLDivElement, object, object, ArrayBuffer | null, string | View | null];

beforeEach(() => {
  shown = [];
  lookedFrom = [];
  splatLoaded = () => Promise.resolve();
  build.mockReset();
  build.mockImplementation(
    async (...args: BuildArgs): Promise<Built> => ({
      view: () => LOOKING,
      home: () => {},
      lookFrom: name => {
        lookedFrom.push(name);
      },
      show: s => {
        shown.push(s);
      },
      splatLoaded: args[3] ? splatLoaded() : Promise.resolve(),
      cameraNames: [],
      dispose: () => {},
    }),
  );
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as RenderingContext);
});

/** Where the stood-in viewer says the person is looking from. */
const LOOKING: View = { position: [1, 2, 3], target: [1, 2, 0], fov: 60, mode: 'fly' };

/** A map with no points: enough for the view to build. */
const MAP = new TextEncoder().encode(
  'ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n',
);
/** Bytes that pass for a SOG: a zip's first signature and its end record. */
const SOG = new Uint8Array([0x50, 0x4b, 3, 4, 0, 0, 0x50, 0x4b, 5, 6, ...new Array<number>(18).fill(0)]);
const bytes = (b: Uint8Array) =>
  new HttpResponse(b.slice().buffer, { headers: { 'content-length': String(b.length) } });

const SPLAT_URL = '/places/splat?place=p1&revision=r1&format=sog';
const scene = (more: object) => ({
  place: 'p1',
  revision: 'r2',
  map: '/places/map?place=p1&revision=r2',
  map_format: 'ply',
  map_bytes: MAP.length,
  cameras: {},
  ...more,
});
const withSplat = (more: object = {}) =>
  scene({
    splat: SPLAT_URL,
    splat_format: 'sog',
    splat_bytes: SOG.length,
    splat_revision: 'r1',
    splat_state: 'current',
    splat_from: 'cameras',
    ...more,
  });

function serve(reply: object, splat?: Parameters<typeof http.get>[1]) {
  server.use(
    http.get('/places/scene', () => HttpResponse.json(reply)),
    http.get('/places/map', () => bytes(MAP)),
    ...(splat ? [http.get('/places/splat', splat)] : []),
  );
}

/** The splat the viewer was last built around: its bytes, or null for the points alone. */
const lastSplat = () => build.mock.calls.at(-1)?.[3] as ArrayBuffer | null | undefined;

it('says a place has no splat yet, draws its points, and logs no error', async () => {
  const errors = vi.spyOn(console, 'error');
  const warnings = vi.spyOn(console, 'warn');
  let asked = 0;
  serve(scene({ splat: null, splat_state: 'none' }), () => {
    asked += 1;
    return new HttpResponse(null, { status: 404 });
  });
  renderPage(<CloudView place="p1" />);
  expect((await screen.findByRole('status')).textContent).toBe('No splat of this place yet.');
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(false));
  expect(lastSplat()).toBeNull();
  expect(screen.getByText('Point size')).toBeTruthy();
  // nothing offers a splat that isn't there, and nothing asks for one
  expect(screen.queryByText('Splat')).toBeNull();
  expect(screen.queryByRole('alert')).toBeNull();
  expect(asked).toBe(0);
  expect(errors).not.toHaveBeenCalled();
  expect(warnings).not.toHaveBeenCalled();
});

it("draws an earlier map's splat that still lines up, labelled as one", async () => {
  serve(
    withSplat({
      splat_state: 'earlier',
      splat_at: 1.8e9,
    }),
    () => bytes(SOG),
  );
  renderPage(<CloudView place="p1" />);
  expect((await screen.findByText(/^This splat was made for an earlier map of this place/)).textContent).toMatch(
    /it lines up with the current one\. A splat of the current map isn't made yet\.$/,
  );
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(true));
  expect(new Uint8Array(lastSplat() ?? new ArrayBuffer(0))).toEqual(SOG);
  expect(screen.getByText('Splat')).toBeTruthy();
});

const learning = (more: object = {}) => ({ on: true, cameras: 1, ...more });

it('says no camera has a point cloud yet, and that learning is off, for a place with nothing to draw', async () => {
  serve({ place: 'p1', revision: null, learning: learning({ on: false, cameras: 0 }), views: [] });
  renderPage(<CloudView place="p1" />);
  expect(
    await screen.findByText(/^No camera has a point cloud of this place yet\. Turn on Learn surroundings in/),
  ).toBeTruthy();
  expect(screen.queryByRole('combobox', { name: 'Camera' })).toBeNull();
  expect(build).not.toHaveBeenCalled();
});

// One camera's own cloud (#417): picked by camera, drawn as points alone, from that camera's viewpoint.
const view = (camera: string, more: object = {}) => ({
  camera,
  cloud: true,
  at: 1.8e9,
  points: 1024,
  size: 'approximate',
  why: null,
  ...more,
});
const VIEWS = [view('hall'), view('porch', { cloud: false, at: null, points: null, size: null, why: 'people' })];
const cloudOf = (camera: string, more: object = {}) =>
  scene({
    revision: `c-${camera}`,
    map: `/places/map?place=p1&revision=c-${camera}&camera=${camera}`,
    splat: null,
    cameras: {
      [camera]: {
        pose: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1.5, 0, 0, 0, 1],
        focal: 300,
        width: 512,
        height: 288,
        at: 1,
      },
    },
    learning: learning(),
    views: VIEWS,
    view: { camera, size: 'approximate', floor_found: true, points: 1024, at: 1.8e9 },
    ...more,
  });

/** Serve each camera's scene by the camera asked for; with none asked for, the first camera's. */
function serveCameras() {
  const asked: (string | null)[] = [];
  server.use(
    http.get('/places/scene', ({ request }) => {
      const camera = new URL(request.url).searchParams.get('camera');
      asked.push(camera);
      if (camera === 'porch')
        return HttpResponse.json({ place: 'p1', revision: null, learning: learning(), views: VIEWS, view: { camera } });
      return HttpResponse.json(cloudOf('hall'));
    }),
    http.get('/places/map', () => bytes(MAP)),
  );
  return asked;
}

it("opens on the first camera's own cloud, from that camera, as points with no splat or plan offered", async () => {
  serveCameras();
  renderPage(<CloudView place="p1" />);
  await waitFor(() => expect(build).toHaveBeenCalledTimes(1));
  expect(build.mock.calls[0]?.[3]).toBeNull(); // no splat
  expect(build.mock.calls[0]?.[4]).toBe('hall'); // from where the camera stands
  expect(
    await screen.findByText(/^What hall sees on its own\. 1,024 points, .+ Approximate size: the model's own estimate/),
  ).toBeTruthy();
  expect(screen.getByLabelText('Camera mark')).not.toBeChecked(); // its own mark would lie along the picture's edges
  expect(screen.queryByText('Splat')).toBeNull();
  expect(screen.queryByLabelText('Plan')).toBeNull();
  expect(screen.queryByText(/No splat/)).toBeNull();
});

it('shows the camera picked, and says why one with no cloud has none', async () => {
  const asked = serveCameras();
  renderPage(<CloudView place="p1" />);
  await waitFor(() => expect(build).toHaveBeenCalledTimes(1));
  await userEvent.click(screen.getByRole('combobox', { name: 'Camera' }));
  await userEvent.click(await screen.findByRole('option', { name: 'porch (no cloud yet)' }));
  expect(
    await screen.findByText(
      'No point cloud from porch yet. Someone was in view each time it looked. It needs about 20 seconds with nobody in view.',
    ),
  ).toBeTruthy();
  expect(asked).toEqual([null, 'porch']);
  // and back to the one that has a cloud
  await userEvent.click(screen.getByRole('combobox', { name: 'Camera' }));
  await userEvent.click(await screen.findByRole('option', { name: 'hall' }));
  await waitFor(() => expect(asked.at(-1)).toBe('hall'));
  expect(await screen.findByText(/1,024 points/)).toBeTruthy();
});

it("says why a camera's cloud isn't newer while learning is off", async () => {
  serve(cloudOf('hall', { learning: learning({ on: false, cameras: 0 }), views: [view('hall', { why: 'off' })] }));
  renderPage(<CloudView place="p1" />);
  const live = await screen.findByRole('link', { name: 'Live' });
  expect(live.getAttribute('href')).toBe('/');
  expect(live.closest('p')?.textContent).toBe(
    "Learn surroundings is off, so this picture won't update. Turn it on in Live.",
  );
});

it('asks for the scene again every few seconds while learning is on, and seldom while it is off', () => {
  const every = sceneQuery('p1').refetchInterval as (q: { state: { data?: object } }) => number;
  expect(every({ state: { data: withSplat({ learning: learning() }) } })).toBe(SCENE_LEARNING_MS);
  expect(every({ state: { data: withSplat({ learning: learning({ on: false }) }) } })).toBe(SCENE_IDLE_MS);
  expect(every({ state: {} })).toBe(SCENE_IDLE_MS);
});

it('draws a newer splat once all of it is here, from where the person was looking, without a reload', async () => {
  const NEWER = '/places/splat?place=p1&revision=r3&format=sog';
  let reply = withSplat({ learning: learning() });
  let release = () => {};
  const held = new Promise<void>(resolve => {
    release = resolve;
  });
  server.use(
    http.get('/places/scene', () => HttpResponse.json(reply)),
    http.get('/places/map', () => bytes(MAP)),
    http.get('/places/splat', async ({ request }) => {
      if (new URL(request.url).searchParams.get('revision') === 'r3') await held;
      return bytes(SOG);
    }),
  );
  const { client } = renderPage(<CloudView place="p1" />);
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(true));
  const before = build.mock.calls.length;

  // the next read of the scene names a newer map and splat, with one more view
  reply = withSplat({
    revision: 'r3',
    map: '/places/map?place=p1&revision=r3',
    splat: NEWER,
    splat_revision: 'r3',
    learning: learning(),
  });
  await client.invalidateQueries({ queryKey: ['places', 'scene', 'p1'] });
  expect(await screen.findByText(/^A newer version of this map is loading/)).toBeTruthy();
  // the one before it stays drawn while the newer one's bytes come
  expect(build.mock.calls.length).toBe(before);
  expect(shown.at(-1)?.splat).toBe(true);

  release();
  await waitFor(() => expect(build.mock.calls.length).toBe(before + 1));
  const [, drawn, , file, opening] = build.mock.calls.at(-1) as BuildArgs;
  expect((drawn as { revision: string }).revision).toBe('r3');
  expect(new Uint8Array(file ?? new ArrayBuffer(0))).toEqual(SOG);
  expect(opening).toEqual(LOOKING);
  await waitFor(() => expect(screen.queryByText(/^A newer version of this map is loading/)).toBeNull());
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(true));
});

it("doesn't draw the map again when only what is said about learning changed", async () => {
  let reply = withSplat({ learning: learning() });
  server.use(
    http.get('/places/scene', () => HttpResponse.json(reply)),
    http.get('/places/map', () => bytes(MAP)),
    http.get('/places/splat', () => bytes(SOG)),
  );
  const { client } = renderPage(<CloudView place="p1" />);
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(true));
  const before = build.mock.calls.length;
  reply = withSplat({ splat_state: 'current', learning: learning({ on: false }) });
  await client.invalidateQueries({ queryKey: ['places', 'scene', 'p1'] });
  await waitFor(() =>
    expect(client.getQueryData<{ learning?: { on: boolean } }>(['places', 'scene', 'p1', null])?.learning?.on).toBe(
      false,
    ),
  );
  expect(build.mock.calls.length).toBe(before);
});

it("names a splat that doesn't line up with the map without offering it", async () => {
  serve(
    scene({
      splat: null,
      splat_state: 'stale',
      splat_stale: { revision: 'r0', at: 1.8e9 },
    }),
  );
  renderPage(<CloudView place="p1" />);
  expect((await screen.findByRole('status')).textContent).toMatch(
    /doesn't line up with this one, so it isn't shown\. A new one isn't made yet\.$/,
  );
  await waitFor(() => expect(lastSplat()).toBeNull());
  expect(screen.queryByText('Splat')).toBeNull();
});

it('says how much of the splat has loaded, with the points drawn meanwhile', async () => {
  // half of an 8 MB splat arrives, and the rest never does
  const total = 8_000_000;
  serve(withSplat({ splat_bytes: total }), () => {
    const half = new ReadableStream<Uint8Array>({
      start: c => c.enqueue(new Uint8Array(total / 2)),
    });
    return new HttpResponse(half, { headers: { 'content-length': String(total) } });
  });
  renderPage(<CloudView place="p1" />);
  // once the map is up, the loading line says the points are what's drawn for now
  expect(await screen.findByText('Loading the splat: 50% of 8.0 MB The points are drawn meanwhile.')).toBeTruthy();
  expect(screen.getByRole('progressbar', { name: 'How much of the splat has loaded' })).toBeTruthy();
  // the map doesn't wait for the splat: the viewer is up with the points, and Points can be chosen
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(false));
  expect(lastSplat()).toBeNull();
  await userEvent.click(screen.getByText('Points'));
  expect(await screen.findByText('Point size')).toBeTruthy();
  expect(screen.getByText(/^Loading the splat: 50% of 8\.0 MB$/)).toBeTruthy();
});

it("draws the splat once it's loaded, and says nothing more about it", async () => {
  serve(withSplat(), () => bytes(SOG));
  renderPage(<CloudView place="p1" />);
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(true));
  expect(new Uint8Array(lastSplat() ?? new ArrayBuffer(0))).toEqual(SOG);
  await waitFor(() => expect(screen.queryByRole('status')).toBeNull());
  expect(screen.queryByRole('alert')).toBeNull();
});

it("says why a splat's bytes can't be drawn, keeps the points, and reads them afresh on Retry", async () => {
  // the browser's kept copy is an older encoding; a read past the cache gets the real file
  const fetched = vi.spyOn(globalThis, 'fetch');
  let asked = 0;
  serve(withSplat(), () => {
    asked += 1;
    return bytes(asked > 1 ? SOG : new Uint8Array([0x1f, 0x8b, 8, 0, 0, 0, 0, 0]));
  });
  renderPage(<CloudView place="p1" />);
  const alert = await screen.findByRole('alert');
  expect(alert.textContent).toMatch(
    /^Couldn't show the splat: the file isn't a splat this view can draw .*\. The points are drawn instead\./,
  );
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(false));
  expect(lastSplat()).toBeNull(); // the viewer was never handed the bad bytes
  expect(screen.getByText('Point size')).toBeTruthy();
  await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(true));
  const splatReads = fetched.mock.calls.filter(([url]) => String(url).startsWith('/places/splat'));
  expect(splatReads.map(([, init]) => init?.cache)).toEqual(['default', 'reload']);
  expect(screen.queryByRole('alert')).toBeNull();
});

it("says why when the viewer can't draw a splat that read fine, and goes back to the points", async () => {
  splatLoaded = () => Promise.reject(new Error('the graphics device was lost'));
  serve(withSplat(), () => bytes(SOG));
  renderPage(<CloudView place="p1" />);
  expect((await screen.findByRole('alert')).textContent).toMatch(
    /^Couldn't show the splat: the graphics device was lost\. The points are drawn instead\./,
  );
  expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy();
  // built around the splat first, then again around the points alone
  await waitFor(() => expect(lastSplat()).toBeNull());
  expect(build.mock.calls.some(call => call[3] !== null)).toBe(true);
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(false));
  expect(screen.getByText('Point size')).toBeTruthy();
});

it("says when the splat's file can't be fetched at all", async () => {
  serve(withSplat(), () => new HttpResponse(null, { status: 500 }));
  renderPage(<CloudView place="p1" />);
  expect((await screen.findByRole('alert')).textContent).toMatch(/^Couldn't show the splat: the server answered 500\./);
  await waitFor(() => expect(lastSplat()).toBeNull());
});

it('turns Floor, Cameras, and Plan on and off without building the view again, including before it is ready', async () => {
  const pending: Array<(built: Built) => void> = [];
  let open = false;
  const built: Built = {
    view: () => LOOKING,
    home: () => {},
    lookFrom: () => {},
    show: s => {
      shown.push(s);
    },
    splatLoaded: Promise.resolve(),
    cameraNames: ['Hall', 'Study'],
    dispose: () => {},
  };
  build.mockImplementation(
    () =>
      new Promise<Built>(resolve => {
        if (open) resolve(built);
        else pending.push(resolve);
      }),
  );
  serve(withSplat({ cameras: { Hall: {}, Study: {} } }), () => bytes(SOG));
  renderPage(<CloudView place="p1" />);
  const floor = await screen.findByRole('checkbox', { name: 'Floor' });
  const cameras = await screen.findByRole('checkbox', { name: 'Cameras (2)' });
  const plan = await screen.findByRole('checkbox', { name: 'Plan' });
  // before the viewer exists: the choice is kept and applied once it does, and the view is still built once
  await userEvent.click(floor);
  await userEvent.click(cameras);
  await userEvent.click(plan);
  expect(shown).toEqual([]);
  expect(floor).not.toBeChecked();
  expect(cameras).not.toBeChecked();
  expect(plan).not.toBeChecked();
  const before = build.mock.calls.length;
  expect(before).toBeGreaterThan(0);
  open = true;
  for (const resolve of pending.splice(0)) resolve(built);
  await waitFor(() => expect(shown.at(-1)).toMatchObject({ floor: false, cameras: false, plan: false }));
  const settled = build.mock.calls.length;

  await userEvent.click(floor);
  await userEvent.click(floor);
  await userEvent.click(cameras);
  await userEvent.click(cameras);
  await userEvent.click(plan);
  await userEvent.click(plan);
  await waitFor(() => expect(shown.at(-1)).toMatchObject({ floor: false, cameras: false, plan: false }));
  expect(build.mock.calls.length).toBe(settled);
  expect(screen.queryByRole('alert')).toBeNull();

  // the same checkboxes in the points view
  await userEvent.click(screen.getByText('Points'));
  await userEvent.click(floor);
  await userEvent.click(cameras);
  await userEvent.click(plan);
  await waitFor(() => expect(shown.at(-1)).toMatchObject({ splat: false, floor: true, cameras: true, plan: true }));
  expect(build.mock.calls.length).toBe(settled);
  expect(screen.queryByRole('alert')).toBeNull();
});

it('keeps Floor, Cameras, and Plan across a newer map replacing the one on screen', async () => {
  const newer = '/places/splat?place=p1&revision=r3&format=sog';
  let reply = withSplat({ cameras: { Hall: {} } });
  let release = () => {};
  const held = new Promise<void>(resolve => {
    release = resolve;
  });
  server.use(
    http.get('/places/scene', () => HttpResponse.json(reply)),
    http.get('/places/map', () => bytes(MAP)),
    http.get('/places/splat', async ({ request }) => {
      if (new URL(request.url).searchParams.get('revision') === 'r3') await held;
      return bytes(SOG);
    }),
  );
  const { client } = renderPage(<CloudView place="p1" />);
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(true));
  await userEvent.click(screen.getByRole('checkbox', { name: 'Floor' }));
  await waitFor(() => expect(shown.at(-1)?.floor).toBe(false));
  const before = build.mock.calls.length;

  reply = withSplat({
    revision: 'r3',
    map: '/places/map?place=p1&revision=r3',
    splat: newer,
    splat_revision: 'r3',
    cameras: { Hall: {} },
  });
  await client.invalidateQueries({ queryKey: ['places', 'scene', 'p1'] });
  expect(await screen.findByText(/^A newer version of this map is loading/)).toBeTruthy();
  await userEvent.click(screen.getByRole('checkbox', { name: 'Plan' }));
  await userEvent.click(screen.getByRole('checkbox', { name: 'Cameras (1)' }));
  expect(build.mock.calls.length).toBe(before);

  release();
  await waitFor(() => expect(build.mock.calls.length).toBe(before + 1));
  await waitFor(() => expect(shown.at(-1)).toMatchObject({ splat: true, floor: false, cameras: false, plan: false }));
  expect(screen.queryByRole('alert')).toBeNull();
});

it('shows the shared error and Retry when the view fails, then builds it again', async () => {
  let fail = true;
  build.mockImplementation(
    async (): Promise<Built> => ({
      view: () => LOOKING,
      home: () => {},
      lookFrom: () => {},
      show: s => {
        if (fail) throw new Error('the overlay could not be drawn');
        shown.push(s);
      },
      splatLoaded: Promise.resolve(),
      cameraNames: [],
      dispose: () => {},
    }),
  );
  serve(withSplat(), () => bytes(SOG));
  renderPage(<CloudView place="p1" />);
  const alert = await screen.findByRole('alert');
  expect(alert.textContent).toMatch(/Couldn't show the map/);
  expect(alert.textContent).toMatch(/Nothing was lost/);
  await userEvent.click(screen.getByText('Show details'));
  expect(screen.getByText('the overlay could not be drawn')).toBeTruthy();
  const before = build.mock.calls.length;
  fail = false;
  await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
  await waitFor(() => expect(shown.at(-1)).toMatchObject({ floor: true, cameras: true, plan: true }));
  expect(screen.queryByRole('alert')).toBeNull();
  expect(build.mock.calls.length).toBe(before + 1);
});

// Every camera's cloud placed in the place (#445): asked for with All cameras, drawn from above, a layer a camera.
const placedRow = (camera: string, more: object = {}) => ({
  camera,
  points: 1024,
  placed: true,
  how: 'spot',
  size: 'approximate',
  against: [],
  shift_m: 0,
  turn_deg: 0,
  suggest: null,
  ...more,
});
const allCameras = (cameras: object[], pairs: object[] = []) =>
  scene({
    revision: 's1',
    map: '/places/map?place=p1&revision=s1&scene=1',
    splat: null,
    plan_transform: [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0],
    learning: learning(),
    views: VIEWS,
    combined: { place: 'p1', id: 's1', cameras, pairs },
  });

function serveAll(cameras: object[], pairs: object[] = []) {
  const asked: URLSearchParams[] = [];
  server.use(
    http.get('/places/scene', ({ request }) => {
      const q = new URL(request.url).searchParams;
      asked.push(q);
      return HttpResponse.json(q.get('all') === '1' ? allCameras(cameras, pairs) : cloudOf('hall'));
    }),
    http.get('/places/map', () => bytes(MAP)),
  );
  return asked;
}

const toAll = async () => {
  await waitFor(() => expect(build).toHaveBeenCalledTimes(1));
  await userEvent.click(screen.getByRole('radio', { name: 'All cameras' }));
};

it('shows every camera placed in one scene from above, each saying how it is placed', async () => {
  const asked = serveAll(
    [
      placedRow('hall', { how: 'adjusted', against: ['porch'], shift_m: 0.16, turn_deg: 1.7 }),
      placedRow('porch', { how: 'adjusted', against: ['hall'], shift_m: 0.16, turn_deg: 1.2 }),
      placedRow('yard', { placed: false, how: 'no_spot' }),
    ],
    [{ a: 'hall', b: 'porch', state: 'tightened', off_m: 0.34, off_deg: 0.5, measured: null, latest: null }],
  );
  const editPlan = vi.fn();
  renderPage(<CloudView place="p1" onEditPlan={editPlan} />);
  await toAll();
  await waitFor(() => expect(build).toHaveBeenCalledTimes(2));
  expect(asked.at(-1)?.get('all')).toBe('1');
  expect(build.mock.calls[1]?.[4]).toBe('@above'); // looking down at the plan
  expect(
    await screen.findByText(/adjusted against porch by what their pictures share: 16 cm and 2° from the spot/),
  ).toBeTruthy();
  expect(screen.getByText(/Not in the scene: it has no spot on your plan/)).toBeTruthy();
  // what the pair's pictures said is detail of its two cameras' rows
  expect(screen.getAllByText(/hall and porch see the same space and are adjusted/)).toHaveLength(2);
  expect(screen.getByText('Each camera shows only what it sees, so expect gaps and seams.')).toBeTruthy();
  // each row leads with its state in a word, and the camera that needs a spot says so and opens the plan editor
  const rows = screen.getAllByRole('listitem');
  expect(rows.map(row => within(row).getByText(/^(Placed|Adjusted|Moved|Not placed|No spot)$/).textContent)).toEqual([
    'Adjusted',
    'Adjusted',
    'No spot',
  ]);
  await userEvent.click(within(rows[2] as HTMLElement).getByRole('button', { name: 'Set its spot' }));
  expect(editPlan).toHaveBeenCalledTimes(1);
  // the sentence behind the word opens under the row
  const more = within(rows[0] as HTMLElement).getByRole('button', { name: 'Details of hall' });
  expect(more.getAttribute('aria-expanded')).toBe('false');
  await userEvent.click(more);
  expect(more.getAttribute('aria-expanded')).toBe('true');
  // one list of cameras: looking from one is the row's own action
  await userEvent.click(within(rows[0] as HTMLElement).getAllByRole('button', { name: 'Look from' })[0] as HTMLElement);
  expect(lookedFrom).toEqual(['hall']);
  expect(screen.getByLabelText("Show yard's points")).toBeDisabled(); // nothing of it to show
  expect(screen.queryByRole('combobox', { name: 'Camera' })).toBeNull(); // the picker is One camera's
});

it("says in a row's detail how its cloud was levelled and squared, that its walls bend, and when its picture was taken", async () => {
  serveAll([
    placedRow('hall', {
      at: new Date(2026, 5, 1, 15, 53).getTime() / 1000,
      level: 'walls',
      tilt_deg: 12.5,
      roll_deg: 0.6,
      squared: true,
      square: { state: 'squared', turn_deg: -2.73, right_m: 0, ahead_m: 0, slide: 'beyond', bend_deg: 2.7 },
    }),
  ]);
  renderPage(<CloudView place="p1" />);
  await toAll();
  const row = (await screen.findAllByRole('listitem'))[0] as HTMLElement;
  expect(
    within(row).getByText(/^Squared to your plan by its walls: turned 2\.7°\. Not slid onto the plan's walls/),
  ).toBeTruthy();
  expect(
    within(row).getByText('No floor in view, so levelled from its walls: the camera looks 13° down and is rolled 1°.'),
  ).toBeTruthy();
  expect(
    within(row).getByText('Walls in this cloud bend by about 3°: that is the depth estimate, not the placement.'),
  ).toBeTruthy();
  expect(within(row).getByText(/^From a picture taken .* 15:53\.$/)).toBeTruthy();
});

it("hides a camera's points when its box is cleared, and the points outside the plan until asked for", async () => {
  serveAll([placedRow('hall'), placedRow('porch')]);
  renderPage(<CloudView place="p1" />);
  await toAll();
  await waitFor(() => expect(build).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(shown.at(-1)?.inside).toBe(true)); // left out by default, and the box says so
  expect(screen.getByLabelText(/^Only inside the plan/)).toBeChecked();
  expect([...(shown.at(-1)?.hidden ?? [])]).toEqual([]);
  await userEvent.click(screen.getByLabelText("Show porch's points"));
  await waitFor(() => expect([...(shown.at(-1)?.hidden ?? [])]).toEqual([1]));
  await userEvent.click(screen.getByLabelText(/^Only inside the plan/));
  await waitFor(() => expect(shown.at(-1)?.inside).toBe(false));
  expect(build).toHaveBeenCalledTimes(2); // nothing was loaded or built again
});

it("offers to move a moved camera's spot, asks first, and can take it back", async () => {
  serveAll([placedRow('hall', { how: 'moved', shift_m: 0.5, turn_deg: 35.2, suggest: [3.4, -0.9, 212.4] })]);
  const posted: object[] = [];
  const undone: object[] = [];
  server.use(
    http.post('/places/scene/spot', async ({ request }) => {
      posted.push((await request.json()) as object);
      return HttpResponse.json({ ok: true, spot: { at: [3.4, -0.9], yaw: 212.4 }, previous: 'plan-v1' });
    }),
    http.post('/places/scene/spot/undo', async ({ request }) => {
      undone.push((await request.json()) as object);
      return HttpResponse.json({ ok: true });
    }),
  );
  renderPage(<CloudView place="p1" />);
  await toAll();
  expect(await screen.findByText(/This camera was moved: by its pictures, 50 cm and 35° from its spot/)).toBeTruthy();
  expect(screen.getByText('Moved')).toBeTruthy();
  // it changes the owner's plan, so it asks first, saying what changes and that the earlier version stays
  await userEvent.click(screen.getByRole('button', { name: 'Move its spot' }));
  const asking = await screen.findByRole('dialog');
  expect(within(asking).getByText("Move hall's spot on your plan?")).toBeTruthy();
  expect(
    within(asking).getByText(
      "It moves 50 cm and turns 35 degrees to where its pictures now put it. The plan's earlier version stays in Versions.",
    ),
  ).toBeTruthy();
  expect(posted).toEqual([]);
  await userEvent.click(within(asking).getByRole('button', { name: 'Move its spot' }));
  await waitFor(() => expect(posted).toEqual([{ place: 'p1', camera: 'hall' }]));
  // done, and how to take it back
  expect(await screen.findByText("Moved hall's spot on your plan.")).toBeTruthy();
  await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(undone).toEqual([{ place: 'p1', camera: 'hall', record: 'plan-v1' }]));
});

it('says so when no camera can be placed, with why for each', async () => {
  server.use(
    http.get('/places/scene', ({ request }) =>
      HttpResponse.json(
        new URL(request.url).searchParams.get('all') === '1'
          ? {
              place: 'p1',
              revision: null,
              learning: learning(),
              views: VIEWS,
              combined: {
                place: 'p1',
                id: 's0',
                cameras: [placedRow('hall', { placed: false, how: 'no_facing' })],
                pairs: [],
              },
            }
          : cloudOf('hall'),
      ),
    ),
    http.get('/places/map', () => bytes(MAP)),
  );
  renderPage(<CloudView place="p1" />);
  await toAll();
  expect(
    await screen.findByText(/No camera can be placed yet: each needs a spot with a facing on your plan/),
  ).toBeTruthy();
  expect(screen.getByText(/Not in the scene: its spot has no facing/)).toBeTruthy();
});
