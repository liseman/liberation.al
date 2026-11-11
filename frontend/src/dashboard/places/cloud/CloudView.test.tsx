// The 3D view's splat states, with the viewer itself stood in for (jsdom can't draw): what the page says and what
// it asks the viewer to draw in each, and that none of them is left to the console (#246).
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { SCENE_IDLE_MS, SCENE_LEARNING_MS, sceneQuery } from '../api';
import CloudView from './CloudView';
import type { Built, View } from './scene';

const build = vi.hoisted(() => vi.fn());
vi.mock('./scene', () => ({ build }));

/** What the stood-in viewer was asked to show, newest last, and how its splat's load ends. */
let shown: Parameters<Built['show']>[0][] = [];
/** What the stood-in viewer was asked to show of each view, newest last. */
let viewsShown: Parameters<Built['showViews']>[] = [];
let splatLoaded: () => Promise<void> = () => Promise.resolve();
/** The splat's own load is only followed for a viewer built around one, as scene.ts settles it at once without. */
type BuildArgs = [HTMLDivElement, object, object, ArrayBuffer | null, string | View | null];

beforeEach(() => {
  shown = [];
  viewsShown = [];
  sessionStorage.clear();
  splatLoaded = () => Promise.resolve();
  build.mockReset();
  build.mockImplementation(
    async (...args: BuildArgs): Promise<Built> => ({
      view: () => LOOKING,
      home: () => {},
      lookFrom: () => {},
      show: s => {
        shown.push(s);
      },
      showViews: (visible, splatViews) => {
        viewsShown.push([visible, splatViews]);
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

it.each([
  ['pass', /^No splat of this place yet\. It's made from .* a pass that runs at 01:00\.$/],
  ['few_frames', /^No splat of this place yet\. It's waiting for daylight: .* It tries again at 01:00\.$/],
  ['no_cameras', /^No splat of this place yet\. This map has no cameras registered yet/],
  ['failed', /^No splat of this place yet\. Making it failed when it was last tried/],
])('says a place has no splat yet and why (%s), draws its points, and logs no error', async (reason, text) => {
  const errors = vi.spyOn(console, 'error');
  const warnings = vi.spyOn(console, 'warn');
  let asked = 0;
  serve(scene({ splat: null, splat_state: 'none', splat_wait: { reason, tried: 1.8e9, pass_at: '01:00' } }), () => {
    asked += 1;
    return new HttpResponse(null, { status: 404 });
  });
  renderPage(<CloudView place="p1" />);
  expect((await screen.findByRole('status')).textContent).toMatch(text);
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
      splat_wait: { reason: 'few_frames', tried: 1.8e9 + 3600, pass_at: '01:00' },
    }),
    () => bytes(SOG),
  );
  renderPage(<CloudView place="p1" />);
  expect((await screen.findByText(/^This splat was made for an earlier map of this place/)).textContent).toMatch(
    /it lines up with the current one\. A splat of the current map isn't made yet\. It's waiting for daylight/,
  );
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(true));
  expect(new Uint8Array(lastSplat() ?? new ArrayBuffer(0))).toEqual(SOG);
  expect(screen.getByText('Splat')).toBeTruthy();
});

const learning = (more: object = {}) => ({
  on: true,
  cameras: 1,
  improving: false,
  views: 42,
  improved: 1.8e9,
  ...more,
});

it('draws a splat that is still improving, and says it is', async () => {
  serve(withSplat({ splat_state: 'improving', learning: learning({ improving: true }) }), () => bytes(SOG));
  renderPage(<CloudView place="p1" />);
  expect(
    await screen.findByText(/^Learning is on, and this map is still improving: .* Built from 42 views/),
  ).toBeTruthy();
  await waitFor(() => expect(shown.at(-1)?.splat).toBe(true));
  expect(screen.queryByRole('alert')).toBeNull();
});

it('says a place with no map has none, and what would start one', async () => {
  serve({ place: 'p1', revision: null, learning: learning({ on: false, cameras: 0, views: null, improved: null }) });
  renderPage(<CloudView place="p1" />);
  expect(
    await screen.findByText(/^No 3D map of this place yet\. Learning is off\. Turn on Learn surroundings/),
  ).toBeTruthy();
  expect(build).not.toHaveBeenCalled();
});

it('asks for the scene again every few seconds while learning is on, and seldom while it is off', () => {
  const every = sceneQuery('p1').refetchInterval as (q: { state: { data?: object } }) => number;
  expect(every({ state: { data: withSplat({ learning: learning() }) } })).toBe(SCENE_LEARNING_MS);
  expect(every({ state: { data: withSplat({ learning: learning({ on: false }) }) } })).toBe(SCENE_IDLE_MS);
  expect(every({ state: {} })).toBe(SCENE_IDLE_MS);
});

it('draws a newer splat once all of it is here, from where the person was looking, without a reload', async () => {
  const NEWER = '/places/splat?place=p1&revision=r3&format=sog';
  let reply = withSplat({ learning: learning({ improving: true }) });
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
    learning: learning({ improving: true, views: 43 }),
  });
  await client.invalidateQueries({ queryKey: ['places', 'scene', 'p1'] });
  expect(await screen.findByText(/^A newer version of this map is loading/)).toBeTruthy();
  expect(await screen.findByText(/Built from 43 views/)).toBeTruthy();
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
  let reply = withSplat({ learning: learning({ improving: true }) });
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
  expect(await screen.findByText(/^Learning is off, so this map stays as it is/)).toBeTruthy();
  expect(build.mock.calls.length).toBe(before);
});

it("names a splat that doesn't line up with the map without offering it", async () => {
  serve(
    scene({
      splat: null,
      splat_state: 'stale',
      splat_stale: { revision: 'r0', at: 1.8e9 },
      splat_wait: { reason: 'pass', tried: null, pass_at: '01:00' },
    }),
  );
  renderPage(<CloudView place="p1" />);
  expect((await screen.findByRole('status')).textContent).toMatch(
    /doesn't line up with this one, so it isn't shown\. A new one isn't made yet\. It's made from the fixed cameras' daytime frames/,
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

const SOURCES = {
  views: [
    { id: 'k1', camera: 'kitchen', t: 1_700_000_000, points: 1200, gaussians: 900 },
    { id: 's1', camera: 'study', t: 1_700_003_600, points: 800, gaussians: 700 },
    { id: 'k2', camera: 'kitchen', t: 1_700_007_200, points: 300, gaussians: 100 },
  ],
  cameras: [
    { name: 'kitchen', points: 1500, gaussians: 1000, views: [0, 2] },
    { name: 'study', points: 800, gaussians: 700, views: [1] },
  ],
  splat_views: '/places/splat?place=p1&revision=r1&format=views',
};
/** Each Gaussian's view, as the views file has them. */
const SPLAT_VIEWS = new Uint8Array([0, 0, 1, 0, 2, 0]);

function serveTagged(more: object = {}) {
  serve(withSplat({ sources: SOURCES, ...more }), ({ request }) =>
    new URL(request.url).searchParams.get('format') === 'views' ? bytes(SPLAT_VIEWS) : bytes(SOG),
  );
}

it("hides a camera's points and splats at once, and its views one by one, without building again", async () => {
  serveTagged();
  renderPage(<CloudView place="p1" />);
  const kitchen = await screen.findByRole('checkbox', { name: 'kitchen · 1,000 splats' });
  await waitFor(() => expect(viewsShown.at(-1)?.[1]).toEqual(new Uint16Array([0, 1, 2])));
  expect(viewsShown.at(-1)?.[0]).toBeNull(); // everything shown
  const builds = build.mock.calls.length;
  await userEvent.click(kitchen);
  await waitFor(() => expect(viewsShown.at(-1)?.[0]).toEqual(new Set([1])));
  expect((kitchen as HTMLInputElement).checked).toBe(false);
  expect(screen.getByText('2 pictures hidden')).toBeTruthy();
  // one of its views back: the camera's box is half on
  await userEvent.click(screen.getByRole('button', { name: '2 pictures' }));
  await userEvent.click(await screen.findByRole('checkbox', { name: /· 100 splats$/ }));
  await waitFor(() => expect(viewsShown.at(-1)?.[0]).toEqual(new Set([1, 2])));
  expect((kitchen as HTMLInputElement).indeterminate).toBe(true);
  await userEvent.click(screen.getByRole('button', { name: 'None' }));
  await waitFor(() => expect(viewsShown.at(-1)?.[0]).toEqual(new Set()));
  await userEvent.click(screen.getByRole('button', { name: 'All' }));
  await waitFor(() => expect(viewsShown.at(-1)?.[0]).toBeNull());
  expect(build.mock.calls.length).toBe(builds);
});

it('counts points in the Points view, and keeps what is hidden for the place this session', async () => {
  serveTagged();
  const first = renderPage(<CloudView place="p1" />);
  await userEvent.click(await screen.findByRole('radio', { name: 'Points' }));
  await userEvent.click(await screen.findByRole('checkbox', { name: 'study · 800 points' }));
  await waitFor(() => expect(viewsShown.at(-1)?.[0]).toEqual(new Set([0, 2])));
  first.unmount();
  renderPage(<CloudView place="p1" />);
  const study = await screen.findByRole('checkbox', { name: 'study · 700 splats' });
  expect((study as HTMLInputElement).checked).toBe(false);
  await waitFor(() => expect(viewsShown.at(-1)?.[0]).toEqual(new Set([0, 2])));
});

it('shows the cameras unavailable, with why, for a map made before maps said where their points came from', async () => {
  serve(scene({ sources: null, cameras: { kitchen: CAMERA, study: CAMERA } }));
  renderPage(<CloudView place="p1" />);
  const kitchen = await screen.findByRole('checkbox', { name: 'kitchen' });
  expect((kitchen as HTMLInputElement).disabled).toBe(true);
  expect(screen.getByText(/It will after its next rebuild\./)).toBeTruthy();
  expect((screen.getByRole('button', { name: 'All' }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole('button', { name: 'None' }) as HTMLButtonElement).disabled).toBe(true);
});

const CAMERA = { pose: new Array<number>(16).fill(0), focal: 100, width: 64, height: 36, at: 1 };
