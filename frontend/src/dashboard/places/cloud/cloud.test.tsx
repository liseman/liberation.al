import { screen } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { afterEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import CloudView from './CloudView';
import { coverFov, frustum, heightColour, heightOf, planFrame, planOutline, robustRange, type V3 } from './geometry';
import { canDraw, canDrawSplat, openingCamera, parseCloud } from './load';

afterEach(() => vi.restoreAllMocks());

const near = (a: readonly number[], b: readonly number[]) => {
  for (const [i, x] of a.entries()) expect(x).toBeCloseTo(b[i] ?? 0, 9);
};

// Map to plan: a quarter turn about the floor's normal, then a shift: plan x = -map y + 2, plan y = map x - 1, height
// = map z.
const TURN = [0, -1, 0, 2, 1, 0, 0, -1, 0, 0, 1, 0];

it("takes a map point into the plan frame and back out by the transform's inverse", () => {
  const { toPlan, toMap } = planFrame(TURN);
  near(toPlan([1, 0, 0.5]), [2, 0, 0.5]);
  near(toPlan([3, 4, 0]), [-2, 2, 0]);
  near(toMap([2, 0, 0.5]), [1, 0, 0.5]);
  near(toMap(toPlan([-0.7, 2.3, 1.1])), [-0.7, 2.3, 1.1]);
});

it('refuses a plan transform that flattens the map', () => {
  expect(() => planFrame([1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0])).toThrow(/no inverse/);
});

it('measures height from the floor plane, wherever it is and whichever way it faces', () => {
  expect(heightOf([5, 5, 1.2], [0, 0, 1, 0.3])).toBeCloseTo(1.5);
  // a floor whose normal is the map's -y, at y = 2: a point at y = 0.5 is 1.5 above it
  expect(heightOf([0, 0.5, 9], [0, -1, 0, 2])).toBeCloseTo(1.5);
});

it("draws a camera's view from its centre out to its image's corners", () => {
  // At (1, 2, 3), turned so its +z looks along the map's +x (its x along map -z, its y along map +y).
  const pose = [0, 0, 1, 1, 0, 1, 0, 2, -1, 0, 0, 3, 0, 0, 0, 1];
  const [o, topLeft, , bottomRight] = frustum(pose, 100, 200, 100, 1) as V3[];
  near(o as V3, [1, 2, 3]);
  // the top-left pixel is 1 to the camera's left (x -1) and 0.5 up (y -0.5) at a depth of 1
  near(topLeft as V3, [2, 1.5, 4]);
  near(bottomRight as V3, [2, 2.5, 2]);
});

it("fills the view with a camera's frame, cropped to the view's shape", () => {
  const deg = (halfTan: number) => (2 * Math.atan(halfTan) * 180) / Math.PI;
  // a 16:9 camera (512 x 288 at a focal of 256) in a view of its own shape: its own vertical field of view
  expect(coverFov(512, 288, 256, 16 / 9)).toBeCloseTo(deg(144 / 256), 9);
  // in a wider view (2:1) its full width spans the view, so the view shows less of its height
  expect(coverFov(512, 288, 256, 2)).toBeCloseTo(deg(128 / 256), 9);
  // in a taller view (1:1) its full height spans the view, and its sides are cropped
  expect(coverFov(512, 288, 256, 1)).toBeCloseTo(deg(144 / 256), 9);
  // a portrait phone frame (768 x 1024) in a wide view keeps its full width: a band across its middle
  expect(coverFov(768, 1024, 700, 2)).toBeCloseTo(deg(192 / 700), 9);
});

it('colours the lowest heights blue and the highest red, and clamps beyond them', () => {
  const blue = heightColour(0, 0, 2);
  const red = heightColour(2, 0, 2);
  expect(blue[2]).toBeGreaterThan(blue[0]);
  expect(red[0]).toBeGreaterThan(red[2]);
  expect(heightColour(-5, 0, 2)).toEqual(blue);
  expect(heightColour(9, 0, 2)).toEqual(red);
  // a flat cloud gets one middling colour, not a division by zero
  expect(heightColour(1, 1, 1).every(Number.isFinite)).toBe(true);
});

it('ranges the colours over the middle of the heights, so a few strays do not flatten them', () => {
  const heights = [...Array.from({ length: 200 }, (_, i) => i / 100), -40, 90];
  const [low, high] = robustRange(heights);
  expect(low).toBeGreaterThanOrEqual(0);
  expect(high).toBeLessThanOrEqual(2);
  expect(robustRange([])).toEqual([0, 1]);
});

it("outlines the plan's rooms in the map frame, just above the floor", () => {
  const rings = planOutline('plan "Home" walls 0.2/0.1\nroom den "Den" living rect 0,0 2x1\n', TURN, 0.05);
  expect(rings).toHaveLength(1);
  const { toPlan } = planFrame(TURN);
  const back = (rings[0] ?? []).map(p => toPlan(p).map(x => Math.round(x * 1000) / 1000));
  expect(back.map(p => p[2])).toEqual(back.map(() => 0.05));
  expect(new Set(back.map(p => `${p[0]},${p[1]}`))).toEqual(new Set(['0,0', '2,0', '2,1', '0,1']));
});

/** A two-point binary PLY as spatial/ply.py writes it: positions, colours, and a count the loader ignores. */
function ply(): ArrayBuffer {
  const header =
    'ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\n' +
    'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty ushort count\nend_header\n';
  const head = new TextEncoder().encode(header);
  const body = new DataView(new ArrayBuffer(2 * 17));
  [
    [1, 2, 3, 255, 0, 0],
    [-1, 0.5, 0, 0, 0, 255],
  ].forEach(([x, y, z, r, g, b], i) => {
    const o = i * 17;
    body.setFloat32(o, x ?? 0, true);
    body.setFloat32(o + 4, y ?? 0, true);
    body.setFloat32(o + 8, z ?? 0, true);
    body.setUint8(o + 12, r ?? 0);
    body.setUint8(o + 13, g ?? 0);
    body.setUint8(o + 14, b ?? 0);
    body.setUint16(o + 15, 7, true);
  });
  const out = new Uint8Array(head.length + body.byteLength);
  out.set(head);
  out.set(new Uint8Array(body.buffer), head.length);
  return out.buffer;
}

it("reads a place map's points and colours from its PLY", () => {
  const g = parseCloud(ply(), 'ply');
  expect(Array.from(g.getAttribute('position').array)).toEqual([1, 2, 3, -1, 0.5, 0]);
  // kept as bytes the GPU scales to 0 to 1, a quarter of the memory of floats
  const c = g.getAttribute('color');
  expect(c.normalized).toBe(true);
  expect([0, 1].flatMap(i => [c.getX(i), c.getY(i), c.getZ(i)])).toEqual([1, 0, 0, 0, 0, 1]);
});

it('draws only the formats it can read', () => {
  expect(canDraw('ply')).toBe(true);
  expect(canDraw('spz')).toBe(false);
  expect(canDraw(null)).toBe(false);
});

it('draws splats only in the formats Spark is given', () => {
  expect(canDrawSplat('spz')).toBe(true);
  expect(canDrawSplat('ply')).toBe(false);
  expect(canDrawSplat(null)).toBe(false);
});

it('offers the splat, and a look from each camera, when the revision has one', async () => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as RenderingContext);
  const cam = { pose: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], focal: 300, width: 512, height: 288, at: 0 };
  server.use(
    http.get('/places/scene', () =>
      HttpResponse.json({
        place: 'p1',
        revision: 'r1',
        map: '/places/map?place=p1&revision=r1',
        map_format: 'ply',
        splat: '/places/splat?place=p1&revision=r1',
        splat_format: 'spz',
        splat_bytes: 4_400_000,
        cameras: { cam0: cam, hall: cam },
      }),
    ),
    // the files never arrive: jsdom can't draw, so the view is looked at before it builds
    http.get('/places/map', () => new Promise<never>(() => {})),
    http.get('/places/splat', () => new Promise<never>(() => {})),
  );
  renderPage(<CloudView place="p1" />);
  expect(await screen.findByText('Splat')).toBeTruthy();
  expect(screen.getByText('Points')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'cam0' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'hall' })).toBeTruthy();
  // the points' own settings wait for the points
  expect(screen.queryByText('Point size')).toBeNull();
});

it("falls back to the points, saying why, when the splat can't be read", async () => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as RenderingContext);
  server.use(
    http.get('/places/scene', () =>
      HttpResponse.json({
        place: 'p3',
        revision: 'r3',
        map: '/places/map?place=p3&revision=r3',
        map_format: 'ply',
        splat: '/places/splat?place=p3&revision=r3',
        splat_format: 'spz',
      }),
    ),
    http.get('/places/map', () => new Promise<never>(() => {})),
    http.get('/places/splat', () => HttpResponse.json({ error: 'no such splat' }, { status: 404 })),
  );
  renderPage(<CloudView place="p3" />);
  expect(await screen.findByText(/Couldn't show the splat, so the points are drawn/)).toBeTruthy();
  expect(screen.getByText('Point size')).toBeTruthy();
  expect(screen.queryByText('Splat')).toBeNull();
});

it('names a splat from another map frame as out of date instead of drawing it', async () => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as RenderingContext);
  server.use(
    http.get('/places/scene', () =>
      HttpResponse.json({
        place: 'p4',
        revision: 'r4',
        map: '/places/map?place=p4&revision=r4',
        map_format: 'ply',
        splat: null,
        splat_stale: { revision: 'r1', at: 1_800_000_000 },
      }),
    ),
    http.get('/places/map', () => new Promise<never>(() => {})),
  );
  renderPage(<CloudView place="p4" />);
  expect(await screen.findByText(/is out of date: the map has moved since/)).toBeTruthy();
  expect(screen.queryByText('Splat')).toBeNull();
});

it("says so when the browser can't draw 3D", async () => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  renderPage(<CloudView place="p1" />);
  expect(await screen.findByText(/no WebGL/)).toBeTruthy();
});

it('says when a place has no 3D map yet, or one in a format it cannot draw', async () => {
  // a context that's never used: the view stops before drawing
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as RenderingContext);
  server.use(http.get('/places/scene', () => HttpResponse.json({ place: 'p1', revision: null })));
  const { unmount } = renderPage(<CloudView place="p1" />);
  expect(await screen.findByText(/No 3D map of this place yet/)).toBeTruthy();
  unmount();
  server.use(
    http.get('/places/scene', () =>
      HttpResponse.json({ place: 'p2', revision: 'r2', map: '/places/map?place=p2&revision=r2', map_format: 'spz' }),
    ),
  );
  renderPage(<CloudView place="p2" />);
  expect(await screen.findByText(/format the view can't draw yet \(spz\)/)).toBeTruthy();
});

it('opens a splat from a view it trained on: a walk splat from the walk, else a fixed camera', () => {
  const names = ['dining', 'walk-0m00s', 'porch', 'walk-0m11s'];
  expect(openingCamera(names, 'walk')).toBe('walk-0m00s');
  expect(openingCamera(names, 'cameras')).toBe('dining');
  expect(openingCamera(names, null)).toBe('dining');
  expect(openingCamera(['dining'], 'walk')).toBeNull();
  expect(openingCamera([], 'cameras')).toBeNull();
});
