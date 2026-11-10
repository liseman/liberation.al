import { screen } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { afterEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import CloudView from './CloudView';
import { collisionMesh, glb, glbUrl } from './collision';
import { toViewerUp, turned } from './frame';
import {
  coverFov,
  frustum,
  heightColour,
  heightOf,
  planFrame,
  planOutline,
  planWalls,
  robustRange,
  type V3,
} from './geometry';
import { canDraw, canDrawSplat, openingCamera, parseCloud, readBytes, splatFault } from './load';
import { loadingText, splatNote } from './status';

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
  const cloud = parseCloud(ply(), 'ply');
  expect(cloud.count).toBe(2);
  expect(Array.from(cloud.positions)).toEqual([1, 2, 3, -1, 0.5, 0]);
  // kept as bytes the GPU scales to 0 to 1, a quarter of the memory of floats; the counts after them are skipped
  expect(Array.from(cloud.colours ?? [])).toEqual([255, 0, 0, 0, 0, 255]);
});

it('draws only the formats it can read', () => {
  expect(canDraw('ply')).toBe(true);
  expect(canDraw('spz')).toBe(false);
  expect(canDraw(null)).toBe(false);
});

it('draws splats only in the format PlayCanvas is given', () => {
  expect(canDrawSplat('sog')).toBe(true);
  expect(canDrawSplat('spz')).toBe(false);
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
        splat_format: 'sog',
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
        splat_format: 'sog',
      }),
    ),
    http.get('/places/map', () => new Promise<never>(() => {})),
    http.get('/places/splat', () => HttpResponse.json({ error: 'no such splat' }, { status: 404 })),
  );
  renderPage(<CloudView place="p3" />);
  expect(
    await screen.findByText("Couldn't show the splat: the server answered 404. The points are drawn instead."),
  ).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy();
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
        splat_state: 'stale',
        splat_stale: { revision: 'r1', at: 1_800_000_000 },
        splat_wait: { reason: 'pass', tried: null, pass_at: '01:00' },
      }),
    ),
    http.get('/places/map', () => new Promise<never>(() => {})),
  );
  renderPage(<CloudView place="p4" />);
  expect(await screen.findByText(/doesn't line up with this one, so it isn't shown/)).toBeTruthy();
  expect(screen.queryByText('Splat')).toBeNull();
});

/** The smallest bytes that pass for a zip: a local file header's signature, then an empty directory's end record. */
const zipLike = () => new Uint8Array([0x50, 0x4b, 3, 4, 0, 0, 0x50, 0x4b, 5, 6, ...new Array<number>(18).fill(0)]);

it("tells a splat's bytes from ones the viewer would choke on, with the reason", () => {
  expect(splatFault(zipLike().buffer)).toBeNull();
  expect(splatFault(new ArrayBuffer(0))).toBe('the file is empty');
  // a zip whose end is missing: the download stopped part way
  expect(splatFault(zipLike().slice(0, 6).buffer)).toBe('the file is cut short');
  // gzip's magic, as an SPZ copy kept by the browser from before the view moved to SOG starts
  expect(splatFault(new Uint8Array([0x1f, 0x8b, 8, 0, 0, 0, 0, 0]).buffer)).toMatch(/isn't a splat this view can draw/);
});

it("reads a response's bytes whole, saying how much has come", async () => {
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(new Uint8Array([1, 2, 3]));
      c.enqueue(new Uint8Array([4, 5]));
      c.close();
    },
  });
  const seen: [number, number][] = [];
  const bytes = await readBytes(new Response(body), 5, (received, total) => seen.push([received, total]));
  expect(Array.from(new Uint8Array(bytes))).toEqual([1, 2, 3, 4, 5]);
  // no length on the response, so the scene's size is the total
  expect(seen).toEqual([
    [0, 5],
    [3, 5],
    [5, 5],
  ]);
});

it('says how much of a splat has loaded, then that it is being drawn', () => {
  expect(loadingText(0, 0, false)).toBe('Loading the splat…');
  expect(loadingText(6_771_297, 13_542_594, false)).toBe('Loading the splat: 50% of 13.5 MB');
  expect(loadingText(2_000_000, 0, false)).toBe('Loading the splat: 2.0 MB…');
  expect(loadingText(13_542_594, 13_542_594, true)).toBe('Drawing the splat…');
});

it("says nothing about a splat that is the map's own, and why there is none otherwise", () => {
  const scene = { place: 'p1', revision: 'r2' };
  const wait = { tried: null, pass_at: '01:00' };
  expect(splatNote({ ...scene, splat: '/s', splat_state: 'current' })).toBeNull();
  expect(splatNote({ ...scene, splat: '/s', splat_state: 'improving' })).toMatch(/^This splat is still improving/);
  expect(splatNote({ ...scene, splat_state: 'none', splat_wait: { ...wait, reason: 'pass' } })).toBe(
    "No splat of this place yet. It's made from the fixed cameras' daytime frames, by a pass that runs at 01:00.",
  );
  const dark = splatNote({
    ...scene,
    splat_state: 'none',
    splat_wait: { ...wait, reason: 'few_frames', tried: 1.8e9 },
  });
  expect(dark).toMatch(/^No splat of this place yet\. It's waiting for daylight: .* too few when it was last tried \(/);
  expect(dark).toMatch(/It tries again at 01:00\.$/);
  expect(splatNote({ ...scene, splat_state: 'none', splat_wait: { ...wait, reason: 'no_cameras' } })).toMatch(
    /no cameras registered yet to make one from/,
  );
  expect(splatNote({ ...scene, splat_state: 'none', splat_wait: { ...wait, reason: 'no_runtime' } })).toMatch(
    /trainer isn't installed/,
  );
  expect(splatNote({ ...scene, splat_state: 'none', splat_wait: { ...wait, reason: 'failed' } })).toMatch(
    /Making it failed when it was last tried\. It tries again at 01:00\./,
  );
  // with no pass scheduled, it says so and promises no next try
  expect(splatNote({ ...scene, splat_state: 'none', splat_wait: { reason: 'pass', tried: null, pass_at: null } })).toBe(
    'No splat of this place yet. None has been made for this map, and nothing makes splats on a schedule yet.',
  );
  expect(
    splatNote({ ...scene, splat_state: 'none', splat_wait: { reason: 'failed', tried: null, pass_at: null } }),
  ).toBe('No splat of this place yet. Making it failed when it was last tried.');
});

it("labels an earlier map's splat as one, and names one that no longer lines up without offering it", () => {
  const scene = { place: 'p1', revision: 'r2', splat_wait: { reason: 'pass' as const, tried: null, pass_at: '01:00' } };
  const earlier = splatNote({ ...scene, splat: '/s', splat_state: 'earlier', splat_at: 1.8e9 });
  expect(earlier).toMatch(
    /^This splat was made for an earlier map of this place \(.+\); it lines up with the current one\./,
  );
  expect(earlier).toMatch(
    /A splat of the current map isn't made yet\. It's made from the fixed cameras' daytime frames/,
  );
  const stale = splatNote({ ...scene, splat_state: 'stale', splat_stale: { revision: 'r1', at: 1.8e9 } });
  expect(stale).toMatch(
    /doesn't line up with this one, so it isn't shown\. A new one isn't made yet\. It's made from the fixed cameras/,
  );
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

it('opens a splat from the first camera it trained on, and from the overview with none', () => {
  expect(openingCamera(['dining', 'porch'])).toBe('dining');
  expect(openingCamera([])).toBeNull();
});

it("turns the map so its floor's up is the viewer's up, by the smallest turn", () => {
  // a levelled map (z up): x stays, the map's up becomes y, and its y becomes the viewer's -z
  const level = toViewerUp([0, 0, 1]);
  near(turned([0, 0, 1], level), [0, 1, 0]);
  near(turned([1, 0, 0], level), [1, 0, 0]);
  near(turned([0, 1, 0], level), [0, 0, -1]);
  // a map that's already y up isn't turned, and an upside-down one is turned over
  near(turned([3, 4, 5], toViewerUp([0, 1, 0])), [3, 4, 5]);
  near(turned([0, -1, 0], toViewerUp([0, -1, 0])), [0, 1, 0]);
  // a tilted floor: its normal still lands on +y, and lengths are kept
  const tilt = toViewerUp([0.6, 0, 0.8]);
  near(turned([0.6, 0, 0.8], tilt), [0, 1, 0]);
  expect(Math.hypot(...turned([1, 2, 3], tilt))).toBeCloseTo(Math.hypot(1, 2, 3), 9);
});

// A two-room plan, 4 m by 3 m each, side by side, with a door in the wall between them.
const TWO_ROOMS = [
  'plan "Home" walls 0.2/0.1',
  'room a "A" living rect 0,0 4x3',
  'room b "B" living rect 4,0 4x3',
  'door a>b w0.9',
].join('\n');
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

it("stops a walk at the plan's walls and lets it through their doorways", () => {
  const walls = planWalls(TWO_ROOMS, IDENTITY);
  const length = ([a, b]: [V3, V3]) => Math.hypot(b[0] - a[0], b[1] - a[1]);
  // the wall between the rooms (x = 4, 3 m long) is two pieces with a 0.9 m gap between them
  const between = walls.filter(([a, b]) => Math.abs(a[0] - 4) < 1e-6 && Math.abs(b[0] - 4) < 1e-6);
  expect(between).toHaveLength(2);
  expect(between.reduce((sum, w) => sum + length(w), 0)).toBeCloseTo(3 - 0.9, 6);
  // the outside walls are whole: the two rooms' outline is 8 + 3 + 8 + 3 m
  const outside = walls.filter(w => !between.includes(w));
  expect(outside.reduce((sum, w) => sum + length(w), 0)).toBeCloseTo(22, 6);
});

it('builds the walk a floor to stand on and walls with two faces', () => {
  const floor: V3[] = [
    [0, 0, 0],
    [4, 0, 0],
    [4, 0, 3],
    [0, 0, 3],
  ];
  const mesh = collisionMesh(
    floor,
    [
      [
        [1, 0, 1],
        [3, 0, 1],
      ],
    ],
    [0, 1, 0],
    2.4,
  );
  // four corners each for the floor and the wall; two triangles a face, both faces
  expect(mesh.positions).toHaveLength(8 * 3);
  expect(mesh.indices).toHaveLength(2 * 12);
  // the wall stands 2.4 m up from its line on the floor
  expect(Array.from(mesh.positions.slice(12))).toEqual([1, 0, 1, 3, 0, 1, 3, 2.4, 1, 1, 2.4, 1].map(Math.fround));
});

it('writes the collision as a GLB the viewer will take for one: a data URL ending in .glb', () => {
  const mesh = collisionMesh(
    [
      [0, 0, 0],
      [1, 0, 0],
      [1, 0, 1],
      [0, 0, 1],
    ],
    [],
    [0, 1, 0],
  );
  const bytes = glb(mesh);
  const view = new DataView(bytes.buffer);
  expect(new TextDecoder().decode(bytes.subarray(0, 4))).toBe('glTF');
  expect(view.getUint32(8, true)).toBe(bytes.length);
  const jsonLength = view.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)));
  expect(json.accessors[0]).toMatchObject({ count: 4, type: 'VEC3', min: [0, 0, 0], max: [1, 0, 1] });
  expect(json.accessors[1].count).toBe(12);
  // the buffer holds the positions, the indices, and four spare bytes no accessor reaches
  expect(json.buffers[0].byteLength).toBe(4 * 12 + 12 * 4 + 4);
  const first = new Float32Array(bytes.buffer.slice(20 + jsonLength + 8, 20 + jsonLength + 8 + 48));
  expect(Array.from(first)).toEqual(Array.from(mesh.positions));
  const url = glbUrl(bytes);
  // what the viewer checks: the last piece of the URL's path after a dot
  expect(new URL(url).pathname.split('.').pop()).toBe('glb');
  expect(url.startsWith('data:model/gltf-binary,%67%6c%54%46')).toBe(true);
});
