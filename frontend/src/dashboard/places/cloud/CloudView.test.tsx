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
/** What the stood-in viewer was asked to draw in place, newest last, and how many viewers were taken down. */
let updated: Parameters<Built['update']>[] = [];
let disposed = 0;
/** The splat's own load is only followed for a viewer built around one, as scene.ts settles it at once without. */
type BuildArgs = [HTMLDivElement, object, object, ArrayBuffer | null, string | View | null];

beforeEach(() => {
  localStorage.clear(); // which scene was last chosen is remembered per browser
  shown = [];
  lookedFrom = [];
  updated = [];
  disposed = 0;
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
      update: (data, cloud) => {
        updated.push([data, cloud]);
      },
      splatLoaded: args[3] ? splatLoaded() : Promise.resolve(),
      cameraNames: [],
      dispose: () => {
        disposed += 1;
      },
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
    expect(
      client.getQueryData<{ learning?: { on: boolean } }>(['places', 'scene', 'p1', null, true])?.learning?.on,
    ).toBe(false),
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
    update: () => {},
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
      update: () => {},
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
    combined: { place: 'p1', id: 's1', cameras, pairs, covered: { floor_m2: 33.5, plan_m2: 88.5 } },
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
  // how far along the map is has a line of its own, first, with each number joined to what it counts
  const covers = screen.getByText(/^The map covers about 38 percent of your plan's floor so far \(34 of 89 m²\)\.$/);
  expect(covers.textContent).toContain('89\u00a0m²'); // the unit can't be left on a line of its own
  const limits = screen.getByText(/^One map of what the cameras have seen, each surface once\. .* so expect gaps\.$/);
  expect(covers.compareDocumentPosition(limits) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  // each row leads with its state in a word, and the camera that needs a spot says so and opens the plan editor
  const rows = screen.getAllByRole('listitem');
  expect(
    rows.map(row => within(row).getByText(/^(Placed|Adjusted|Following|Not placed|No spot)$/).textContent),
  ).toEqual(['Adjusted', 'Adjusted', 'No spot']);
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
  // the lines are in the row's detail from the start, shown when it is opened
  expect(
    await screen.findByText(/^Squared to your plan by its walls: turned 2\.7°\. Not slid onto the plan's walls/),
  ).toBeTruthy();
  expect(
    screen.getByText('No floor in view, so levelled from its walls: the camera looks 13° down and is rolled 1°.'),
  ).toBeTruthy();
  expect(
    screen.getByText('Walls in this cloud bend by about 3°: that is the depth estimate, not the placement.'),
  ).toBeTruthy();
  expect(screen.getByText(/^From a picture taken .* 15:53\.$/)).toBeTruthy();
});

const FITTED = {
  state: 'fitted',
  scale: 0.966,
  up: 0.966,
  dx: 0.01,
  dy: 0.012,
  walls: [
    { off_deg: 0.3, before_m: 0.241, after_m: 0.038 },
    { off_deg: 1.1, before_m: 0.064, after_m: -0.133 },
  ],
  seen: 2,
};

it('draws fitted clouds with flat walls and floor unless the box is cleared, and says what a flat wall is', async () => {
  const asked = serveAll([placedRow('hall', { fitted: true, fit: FITTED }), placedRow('porch')]);
  renderPage(<CloudView place="p1" />);
  await toAll();
  const box = await screen.findByLabelText('Flat walls and floor');
  expect(box).toBeChecked();
  expect(screen.getByText("A flat wall is your plan's wall with what the camera saw painted on it.")).toBeTruthy();
  expect(asked.at(-1)?.get('flat')).toBeNull(); // flat is what the scene is unless asked otherwise
  expect(
    await screen.findByText(
      "Fitted to the room: resized by 3 percent (smaller), moved 2 cm, matched to 2 walls. Before, its walls stood up to 24 cm from your plan's.",
    ),
  ).toBeTruthy();
  await userEvent.click(box);
  await waitFor(() => expect(asked.at(-1)?.get('flat')).toBe('0'));
  expect(screen.getByLabelText('Flat walls and floor')).not.toBeChecked();
});

it('offers no flat walls where no cloud is fitted to its room', async () => {
  serveAll([placedRow('hall'), placedRow('porch', { fit: { ...FITTED, state: 'no_match', walls: [] } })]);
  renderPage(<CloudView place="p1" />);
  await toAll();
  expect(
    await screen.findByText(/^Not fitted to the room: the wall it sees isn't where a wall of your plan is/),
  ).toBeTruthy();
  expect(screen.queryByLabelText('Flat walls and floor')).toBeNull();
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

it('says a turned camera was placed again without asking, and offers the one way back', async () => {
  const since = Date.now() / 1000 - 20;
  const followed = { by: 'turned', since, lost: '', pinned: false };
  let rows: object[] = [
    placedRow('hall', { how: 'following', shift_m: 0.004, turn_deg: 90.2, follow: followed }),
    placedRow('porch', { placed: false, how: 'lost', follow: { ...followed, by: '', lost: 'ambiguous' } }),
  ];
  const posted: object[] = [];
  server.use(
    http.get('/places/scene', ({ request }) =>
      HttpResponse.json(new URL(request.url).searchParams.get('all') === '1' ? allCameras(rows) : cloudOf('hall')),
    ),
    http.get('/places/map', () => bytes(MAP)),
    http.post('/places/scene/follow', async ({ request }) => {
      const body = (await request.json()) as { camera: string; use: string };
      posted.push(body);
      rows =
        body.use === 'spot'
          ? [placedRow('hall', { follow: { ...followed, by: '', pinned: true } }), rows[1] as object]
          : [
              placedRow('hall', { how: 'following', shift_m: 0.004, turn_deg: 90.2, follow: followed }),
              rows[1] as object,
            ];
      return HttpResponse.json({ ok: true });
    }),
  );
  renderPage(<CloudView place="p1" onEditPlan={vi.fn()} />);
  await toAll();
  // what was done is said plainly, and nothing was asked: no dialog, no button in the row that changes the plan
  expect(
    await screen.findByText(
      "Turned 90° from your plan's spot, a moment ago. Placed where it stood, by the walls it now sees. Your plan is unchanged.",
    ),
  ).toBeTruthy();
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Move its spot' })).toBeNull();
  const [hall, porch] = screen.getAllByRole('listitem') as [HTMLElement, HTMLElement];
  expect(within(hall).getByText('Following')).toBeTruthy();
  // a view that can't be placed says why, isn't offered as a layer, and doesn't send the owner to the plan editor
  expect(within(porch).getByText('Not placed')).toBeTruthy();
  expect(within(porch).getByText(/the walls it sees fit your plan in more than one place/)).toBeTruthy();
  expect((within(porch).getByRole('checkbox') as HTMLInputElement).disabled).toBe(true);
  expect(within(porch).queryByRole('button', { name: 'Set its spot' })).toBeNull();
  // its one fix is on the row itself, where a camera with no spot has Set its spot
  expect(within(porch).getByRole('button', { name: "Use the plan's spot" })).toBeTruthy();
  // the way back is in the row's detail, and is itself taken back from the message that says it was done
  expect(within(hall).queryByRole('button', { name: "Use the plan's spot" })).toBeNull(); // not in the row itself
  await userEvent.click(within(hall).getByRole('button', { name: 'Details of hall' }));
  await userEvent.click(await within(hall).findByRole('button', { name: "Use the plan's spot" }));
  await waitFor(() => expect(posted).toEqual([{ place: 'p1', camera: 'hall', use: 'spot' }]));
  expect(await screen.findByText("hall is back on your plan's spot.")).toBeTruthy();
  expect(await within(hall).findByText(/Placed from its spot on your plan, as you asked/)).toBeTruthy();
  expect(within(hall).getByRole('button', { name: 'Follow it again' })).toBeTruthy();
  await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(posted.at(-1)).toEqual({ place: 'p1', camera: 'hall', use: 'cloud' }));
  expect(await within(hall).findByText(/^Turned 90° from your plan's spot/)).toBeTruthy();
});

it('shows a moving camera as Tracking, with where it is now, and asks for the scene more often meanwhile', async () => {
  const at = Date.now() / 1000 - 1;
  const track = {
    state: 'tracking',
    since: at - 30,
    at,
    held: at - 1,
    pose: [1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 1.4, 0, 0, 0, 1],
    went_m: 0.9,
    from_m: 0.62,
    from_deg: 41.2,
    added: 4200,
    added_at: at - 1,
  };
  serveAll([placedRow('hall', { track }), placedRow('porch')]);
  renderPage(<CloudView place="p1" />);
  await toAll();
  const [hall, porch] = (await screen.findAllByRole('listitem')) as [HTMLElement, HTMLElement];
  expect(await within(hall).findByText('Tracking')).toBeTruthy();
  // the seconds since the last points went in are whatever the clock makes them by the time the row is drawn
  const said = /This move has added 4,200 points to the map, the last \d+ s ago\.$/;
  const line = await within(hall).findByText(said);
  expect(line.textContent).toMatch(
    /^It is moving, now 62 cm away and turned 41° from where its points were made, a moment ago\. Held to your plan's walls\. /,
  );
  expect(within(hall).getByLabelText(said)).toBeTruthy(); // and its row says so without being opened
  expect(within(porch).getByText('Placed')).toBeTruthy();
  expect(within(porch).queryByText(/^It is moving/)).toBeNull();
});

it('opens a place with several cameras on all of them, and after that on what was last chosen here', async () => {
  const two = [view('hall'), view('porch')];
  const asked: (string | null)[] = [];
  server.use(
    http.get('/places/scene', ({ request }) => {
      const q = new URL(request.url).searchParams;
      asked.push(q.get('all'));
      const reply = q.get('all') === '1' ? allCameras([placedRow('hall'), placedRow('porch')]) : cloudOf('hall');
      return HttpResponse.json({ ...reply, views: two });
    }),
    http.get('/places/map', () => bytes(MAP)),
  );
  const first = renderPage(<CloudView place="p1" />);
  // nothing was chosen in this browser yet: two cameras have clouds, so the place's map is what opens
  await waitFor(() => expect(asked.at(-1)).toBe('1'));
  expect(((await screen.findByRole('radio', { name: 'All cameras' })) as HTMLInputElement).checked).toBe(true);
  await userEvent.click(screen.getByRole('radio', { name: 'One camera' }));
  first.unmount();
  const before = asked.length;
  renderPage(<CloudView place="p1" />); // the page opened again: the choice made here holds
  expect(((await screen.findByRole('radio', { name: 'One camera' })) as HTMLInputElement).checked).toBe(true);
  await waitFor(() => expect(build).toHaveBeenCalled());
  expect(asked.slice(before)).not.toContain('1'); // and the scene of all cameras isn't asked for
});

it('opens a place whose one camera has a cloud on that camera', async () => {
  const asked = serveAll([placedRow('hall')]);
  renderPage(<CloudView place="p1" />);
  await waitFor(() => expect(build).toHaveBeenCalledTimes(1));
  expect(((await screen.findByRole('radio', { name: 'One camera' })) as HTMLInputElement).checked).toBe(true);
  expect(asked.every(q => q.get('all') === null)).toBe(true);
});

it("says what the map took of a camera's picture, and marks one that was left out of it", async () => {
  const held = {
    points: 18204,
    state: 'added',
    why: '',
    added: 3100,
    off_m: 0.03,
    moved_m: 0,
    turned_deg: 0,
    resized: 1,
  };
  serveAll([
    placedRow('hall', { map: held }),
    placedRow('porch', { map: { ...held, points: 0, state: 'refused', why: 'apart', added: 0, off_m: 0.45 } }),
  ]);
  renderPage(<CloudView place="p1" />);
  await toAll();
  const [hall, porch] = (await screen.findAllByRole('listitem')) as [HTMLElement, HTMLElement];
  expect(
    await within(hall).findByText('The map holds 18,204 points from this camera: 3,100 from its latest picture.'),
  ).toBeTruthy();
  expect(within(hall).getByText('Placed')).toBeTruthy();
  expect(within(porch).getByText('Left out')).toBeTruthy();
  // what the Left out badge is about leads that camera's detail
  const why = within(porch).getByText(
    /^Its latest picture is left out: it doesn't line up with what the map already holds/,
  );
  const how = within(porch).getByText('Placed from its spot on your plan.');
  expect(why.compareDocumentPosition(how) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  const holds = within(hall).getByText('The map holds 18,204 points from this camera: 3,100 from its latest picture.');
  const stands = within(hall).getByText('Placed from its spot on your plan.');
  expect(stands.compareDocumentPosition(holds) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
});

it('puts its One camera / All cameras switch where the page gives a place for it, and nowhere else', async () => {
  serveAll([placedRow('hall')]);
  const slot = document.createElement('div');
  document.body.append(slot);
  const page = renderPage(<CloudView place="p1" switchIn={slot} />);
  await waitFor(() => expect(build).toHaveBeenCalledTimes(1));
  const radios = await screen.findAllByRole('radio', { name: 'All cameras' });
  expect(radios).toHaveLength(1);
  expect(slot.contains(radios[0] as HTMLElement)).toBe(true); // on the page's line, not at the head of the view
  page.unmount();
  slot.remove();
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
  expect(await screen.findByText(/Not in the scene: its spot has no facing/)).toBeTruthy();
});

it("takes the window's height from where the view starts, and measures again when a render moves that", async () => {
  serve(scene({ splat: null }));
  const { container } = renderPage(<CloudView place="p1" />);
  const floor = await screen.findByRole('checkbox', { name: 'Floor' });
  const frame = container.querySelector<HTMLElement>('[data-fills-page]');
  expect(frame).toBeTruthy();
  // The page gives up the room it kept under a view that fills the window, and the view's own CSS takes the window's
  // height less where the view starts and the page frame's gutter (jsdom lays nothing out, so that starts at nothing).
  expect(frame?.style.getPropertyValue('--fills-top')).toBe('0px');
  const measure = vi.spyOn(frame as HTMLElement, 'getBoundingClientRect').mockReturnValue({ top: 268 } as DOMRect);
  await userEvent.click(floor);
  await waitFor(() => expect(frame?.style.getPropertyValue('--fills-top')).toBe('268px'));
  measure.mockRestore();
});

/** A scene's points as the server writes them (spatial/ply.py): x, y, z, colour, whose each is, and outside or not. */
function scenePly(points: [number, number, number][]): Uint8Array {
  const head = new TextEncoder().encode(
    `ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\n` +
      'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty ushort view\nproperty ushort outside\nend_header\n',
  );
  const body = new DataView(new ArrayBuffer(points.length * 19));
  points.forEach(([x, y, z], i) => {
    body.setFloat32(i * 19, x, true);
    body.setFloat32(i * 19 + 4, y, true);
    body.setFloat32(i * 19 + 8, z, true);
    body.setUint8(i * 19 + 12, 100);
  });
  const out = new Uint8Array(head.length + body.byteLength);
  out.set(head);
  out.set(new Uint8Array(body.buffer), head.length);
  return out;
}

/** What changed between two scenes' points (spatial/cloud_scene.py `changed`): runs of the earlier points kept
 *  (where from, how many), -1 for the new ones, then the new ones. */
function sceneChange(total: number, runs: [number, number][], fresh: [number, number, number][]): Uint8Array {
  const added = scenePly(fresh);
  const out = new Uint8Array(16 + runs.length * 8 + added.length);
  out.set(new TextEncoder().encode('LMC1'));
  const head = new DataView(out.buffer);
  head.setUint32(4, total, true);
  head.setUint32(8, runs.length, true);
  head.setUint32(12, fresh.length, true);
  runs.forEach(([start, count], i) => {
    head.setInt32(16 + i * 8, start, true);
    head.setInt32(20 + i * 8, count, true);
  });
  out.set(added, 16 + runs.length * 8);
  return out;
}

it('draws a map that grew while a camera moved into the view it has, asking only for what changed', async () => {
  const cam = (x: number) => ({
    pose: [1, 0, 0, x, 0, 1, 0, 3, 0, 0, 1, 1.4, 0, 0, 0, 1],
    focal: 300,
    width: 512,
    height: 288,
    at: 1,
  });
  const track = (x: number, added: number) => ({
    state: 'tracking',
    since: Date.now() / 1000 - 30,
    at: Date.now() / 1000 - 1,
    held: Date.now() / 1000 - 2,
    pose: cam(x).pose,
    went_m: x,
    from_m: x,
    from_deg: 0,
    added,
    added_at: Date.now() / 1000 - 1,
  });
  // the scene as it is read first, then with the hall camera half a metre on and two more points in the map,
  // one of the first three gone
  let now = {
    ...allCameras([placedRow('hall', { track: track(0.1, 3) }), placedRow('porch')]),
    cameras: { hall: cam(0.1) },
  };
  const later = {
    ...allCameras([placedRow('hall', { track: track(0.6, 5) }), placedRow('porch')]),
    revision: 's2',
    map: '/places/map?place=p1&revision=s2&scene=1',
    cameras: { hall: cam(0.6) },
  };
  const first: [number, number, number][] = [
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
  ];
  const whole = scenePly([
    [0, 0, 0],
    [2, 0, 0],
    [3, 0, 0],
    [4, 0, 0],
  ]);
  const change = sceneChange(
    4,
    [
      [0, 1],
      [2, 1],
      [-1, 2],
    ],
    [
      [3, 0, 0],
      [4, 0, 0],
    ],
  );
  const maps: { revision: string | null; since: string | null; sent: number }[] = [];
  server.use(
    http.get('/places/scene', ({ request }) =>
      HttpResponse.json(new URL(request.url).searchParams.get('all') === '1' ? now : cloudOf('hall')),
    ),
    http.get('/places/map', ({ request }) => {
      const q = new URL(request.url).searchParams;
      if (q.get('scene') !== '1') return bytes(MAP);
      const body = q.get('since') ? change : q.get('revision') === 's1' ? scenePly(first) : whole;
      maps.push({ revision: q.get('revision'), since: q.get('since'), sent: body.length });
      return new HttpResponse(body.slice().buffer, {
        headers: { 'content-type': q.get('since') ? 'application/x-map-change' : 'application/octet-stream' },
      });
    }),
  );
  const { client } = renderPage(<CloudView place="p1" />);
  await toAll();
  await waitFor(() => expect((build.mock.calls.at(-1)?.[2] as { count: number } | undefined)?.count).toBe(3));
  const builds = build.mock.calls.length;
  const taken = disposed;
  const canvas = screen.getByRole('img');
  const [hall] = screen.getAllByRole('listitem') as [HTMLElement];
  await userEvent.click(within(hall).getByRole('button', { name: /^Details of / }));
  await userEvent.click(screen.getByRole('checkbox', { name: 'Floor' })); // a setting changed from how it opens
  expect(
    within(hall)
      .getByRole('button', { name: /^Details of / })
      .getAttribute('aria-expanded'),
  ).toBe('true');

  now = later;
  await client.invalidateQueries({ queryKey: ['places', 'scene', 'p1'] });
  await waitFor(() => expect(updated.at(-1)?.[1]?.count).toBe(4));
  // the points the view now has are the later scene's, made from the ones it held and the two that came
  const [scene2, cloud] = updated.at(-1) as Parameters<Built['update']>;
  expect(Array.from(cloud?.positions ?? [])).toEqual([0, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]);
  expect(scene2.cameras?.hall?.pose[3]).toBe(0.6); // and the camera's mark is where the camera now is
  // only what changed was asked for and sent, not the points whole
  expect(maps).toEqual([
    { revision: 's1', since: null, sent: scenePly(first).length },
    { revision: 's2', since: 's1', sent: change.length },
  ]);
  expect(change.length).toBe(
    16 +
      3 * 8 +
      scenePly([
        [3, 0, 0],
        [4, 0, 0],
      ]).length,
  ); // two points, and where the rest are
  // the view was not built again or taken down, the page is the same page, and nothing said it was loading
  expect(build.mock.calls.length).toBe(builds);
  expect(disposed).toBe(taken);
  expect(screen.getByRole('img')).toBe(canvas);
  expect(screen.getAllByRole('listitem')[0]).toBe(hall);
  expect(
    within(hall)
      .getByRole('button', { name: /^Details of / })
      .getAttribute('aria-expanded'),
  ).toBe('true');
  expect((screen.getByRole('checkbox', { name: 'Floor' }) as HTMLInputElement).checked).toBe(false);
  expect(screen.queryByText(/Loading|newer version/)).toBeNull();
  expect(within(hall).getByText(/^\+5 points, /)).toBeTruthy(); // its row has the move's new count

  // a camera that only moved, with the map as it was, is drawn where it is without reading any points
  now = { ...later, cameras: { hall: cam(0.9) } };
  await client.invalidateQueries({ queryKey: ['places', 'scene', 'p1'] });
  await waitFor(() => expect(updated.at(-1)?.[0].cameras?.hall?.pose[3]).toBe(0.9));
  expect(updated.at(-1)?.[1]).toBeNull();
  expect(maps).toHaveLength(2);
  expect(build.mock.calls.length).toBe(builds);
});

it('reads the points whole when the server sends them in answer to a question about what changed', async () => {
  let now = allCameras([placedRow('hall'), placedRow('porch')]);
  const later = { ...now, revision: 's2', map: '/places/map?place=p1&revision=s2&scene=1' };
  server.use(
    http.get('/places/scene', ({ request }) =>
      HttpResponse.json(new URL(request.url).searchParams.get('all') === '1' ? now : cloudOf('hall')),
    ),
    http.get('/places/map', ({ request }) => {
      const q = new URL(request.url).searchParams;
      if (q.get('scene') !== '1') return bytes(MAP);
      return bytes(
        scenePly(
          q.get('revision') === 's1'
            ? [[0, 0, 0]]
            : [
                [5, 5, 5],
                [6, 6, 6],
              ],
        ),
      );
    }),
  );
  const { client } = renderPage(<CloudView place="p1" />);
  await toAll();
  await waitFor(() => expect((build.mock.calls.at(-1)?.[2] as { count: number } | undefined)?.count).toBe(1));
  const builds = build.mock.calls.length;
  now = later;
  await client.invalidateQueries({ queryKey: ['places', 'scene', 'p1'] });
  await waitFor(() => expect(Array.from(updated.at(-1)?.[1]?.positions ?? [])).toEqual([5, 5, 5, 6, 6, 6]));
  expect(build.mock.calls.length).toBe(builds);
});
