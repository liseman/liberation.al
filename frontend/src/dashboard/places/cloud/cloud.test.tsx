import { screen } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { afterEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import type { SceneFit, ScenePair, ScenePlaced, SceneSquare } from '../api';
import { buildScene } from '../scene/build';
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
import { outsideCounts, pointIndices, TINT_SHARE, tinted, tintOf } from './layers';
import { applyChange, canDraw, canDrawSplat, openingCamera, parseCloud, readBytes, splatFault } from './load';
import {
  addedNote,
  bendLine,
  coveredLine,
  fitLine,
  learningNote,
  levelLine,
  loadingText,
  mapLine,
  pairLine,
  pictureLine,
  placedLine,
  placedState,
  planNote,
  splatNote,
  squareLine,
  trackLine,
  viewNote,
  viewWhy,
  wayBack,
} from './status';

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

/** What changed since some points: runs of them kept (where from, how many; -1 for the new ones), then `fresh`. */
function change(total: number, runs: [number, number][], fresh: ArrayBuffer, said = 2): ArrayBuffer {
  const out = new Uint8Array(16 + runs.length * 8 + fresh.byteLength);
  out.set(new TextEncoder().encode('LMC1'));
  const head = new DataView(out.buffer);
  head.setUint32(4, total, true);
  head.setUint32(8, runs.length, true);
  head.setUint32(12, said, true);
  runs.forEach(([start, count], i) => {
    head.setInt32(16 + i * 8, start, true);
    head.setInt32(20 + i * 8, count, true);
  });
  out.set(new Uint8Array(fresh), 16 + runs.length * 8);
  return out.buffer;
}

it('makes the points of a later map from the ones held and what changed, leaving the ones held as they were', () => {
  const was = { ...parseCloud(ply(), 'ply'), views: new Uint16Array([4, 5]), outside: new Uint8Array([0, 1]) };
  // the second point held stays, then the two new ones, then the first held
  const now = applyChange(
    was,
    change(
      4,
      [
        [1, 1],
        [-1, 2],
        [0, 1],
      ],
      ply(),
    ),
  );
  expect(now.count).toBe(4);
  expect(Array.from(now.positions)).toEqual([-1, 0.5, 0, 1, 2, 3, -1, 0.5, 0, 1, 2, 3]);
  expect(Array.from(now.colours ?? [])).toEqual([0, 0, 255, 255, 0, 0, 0, 0, 255, 255, 0, 0]);
  expect(Array.from(now.views ?? [])).toEqual([5, 0, 0, 4]); // the new points here say nothing of whose they are
  expect(Array.from(now.outside ?? [])).toEqual([1, 0, 0, 0]);
  expect(Array.from(was.positions)).toEqual([1, 2, 3, -1, 0.5, 0]);
});

it.each([
  ['a change that takes points the held map does not have', change(1, [[5, 1]], ply())],
  ['one whose runs do not add up to the points it says', change(3, [[0, 1]], ply())],
  ['one that says more new points than it brings', change(2, [[-1, 2]], ply(), 3)],
  ['bytes that are no change at all', ply()],
])('refuses %s', (_, bytes) => {
  expect(() => applyChange(parseCloud(ply(), 'ply'), bytes)).toThrow(/change/);
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

it("says nothing about a splat that is the map's own, and that there is none otherwise", () => {
  const scene = { place: 'p1', revision: 'r2' };
  expect(splatNote({ ...scene, splat: '/s', splat_state: 'current' })).toBeNull();
  expect(splatNote({ ...scene, splat_state: 'none' })).toBe('No splat of this place yet.');
});

it("labels an earlier map's splat as one, and names one that no longer lines up without offering it", () => {
  const scene = { place: 'p1', revision: 'r2' };
  const earlier = splatNote({ ...scene, splat: '/s', splat_state: 'earlier', splat_at: 1.8e9 });
  expect(earlier).toMatch(
    /^This splat was made for an earlier map of this place \(.+\); it lines up with the current one\./,
  );
  expect(earlier).toMatch(/A splat of the current map isn't made yet\.$/);
  const stale = splatNote({ ...scene, splat_state: 'stale', splat_stale: { revision: 'r1', at: 1.8e9 } });
  expect(stale).toMatch(/doesn't line up with this one, so it isn't shown\. A new one isn't made yet\.$/);
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
  expect(await screen.findByText(/Nothing to show in 3D here yet/)).toBeTruthy();
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
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

it("outlines a plan's rooms and stops a walk at its walls, with its doorways open and its windows shut", () => {
  const scene = JSON.stringify(
    buildScene({
      height: 2.4,
      walls: [
        { from: [0, 0], to: [8, 0], thickness: 0.3 },
        { from: [4, 0], to: [4, 3], thickness: 0.1 },
      ],
      openings: [
        { kind: 'window', at: [2, 0], width: 1.2 },
        { kind: 'door', at: [4, 1], width: 0.9 },
        { kind: 'cased', at: [4, 2.5], width: 1 },
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
    }),
  );
  const rings = planOutline(scene, TURN, 0.05);
  expect(rings).toHaveLength(1);
  const { toPlan } = planFrame(TURN);
  const back = (rings[0] ?? []).map(p => toPlan(p).map(x => Math.round(x * 1000) / 1000));
  expect(new Set(back.map(p => `${p[0]},${p[1]},${p[2]}`))).toEqual(
    new Set(['0,0,0.05', '4,0,0.05', '4,3,0.05', '0,3,0.05']),
  );
  const walls = planWalls(scene, IDENTITY);
  const spans = walls.map(([a, b]) => [a[0], a[1], b[0], b[1]].map(v => Math.round(v * 1000) / 1000));
  // the outside wall is whole (its window is wall); the inner one is cut by the door (0.55 to 1.45) and the doorway
  // (2 to 3, which reaches the wall's end)
  expect(spans).toEqual([
    [0, 0, 8, 0],
    [4, 0, 4, 0.55],
    [4, 1.45, 4, 2],
  ]);
  expect(() => planOutline('{"nodes": 3}', IDENTITY)).toThrow('the plan is not a scene');
  expect(() => planWalls('plan "Home" walls 0.2/0.1', IDENTITY)).toThrow(); // the old format is not read
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

// The line while there is nothing to draw: that no camera has a point cloud yet, and what would make one.
const learning = (more: object = {}) => ({ on: false, cameras: 0, ...more });

it.each([
  [
    'off',
    learning(),
    'No camera has a point cloud of this place yet. Turn on Learn surroundings in {Live}, and each camera chosen there gets a cloud of what it sees.',
  ],
  [
    'on',
    learning({ on: true, cameras: 1 }),
    'No camera has a point cloud of this place yet. Each chosen camera gets one once it has had about 20 seconds with nobody in view.',
  ],
  [
    'on with no camera',
    learning({ on: true }),
    'No camera has a point cloud of this place yet. No camera is chosen to learn from. Choose one beside Learn surroundings in {Live}.',
  ],
])('says what would make a point cloud for a place where no camera has one: %s', (_, said, text) => {
  expect(learningNote({ place: 'p1', revision: null, learning: said })).toBe(text);
});

it("says nothing more about a place's map once it has one, or when the scene doesn't say", () => {
  expect(learningNote({ place: 'p1', revision: 'r2', learning: learning({ on: true, cameras: 2 }) })).toBeNull();
  expect(learningNote({ place: 'p1', revision: 'r2' })).toBeNull();
});

it("leads a camera's cloud with whose view it is, then its points, when it was made, and that its size is a guess", () => {
  const now = new Date(2026, 10, 10, 9, 0);
  const at = new Date(2026, 10, 10, 1, 40).getTime() / 1000;
  const own = {
    place: 'p1',
    revision: 'c1',
    view: { camera: 'cam0', points: 123554, at, size: 'approximate' },
  } as const;
  expect(viewNote(own, 'Hall', now)).toBe(
    "What Hall sees on its own. 123,554 points, today 01:40. Approximate size: the model's own estimate of distances, which can be a tenth off or more.",
  );
  expect(viewNote({ ...own, view: { ...own.view, at: at - 7 * 3600 } }, 'Hall', now)).toMatch(
    /123,554 points, yesterday 18:40\./,
  );
  expect(
    viewNote({ ...own, view: { ...own.view, level: 'walls', tilt_deg: 12.5, roll_deg: 0.6 } }, 'Hall', now),
  ).toMatch(
    /today 01:40\. No floor in view, so levelled from its walls: the camera looks 13° down and is rolled 1°\. Approximate size/,
  );
  expect(viewNote({ place: 'p1', revision: 'r2' })).toBeNull(); // the place's own map
  expect(viewNote({ place: 'p1', revision: null, view: { camera: 'hall' } })).toBeNull(); // no cloud yet
  expect(splatNote({ ...own, splat: null })).toBeNull(); // no splat goes with a camera's cloud
});

// The scene of all the cameras' clouds (#445): how each camera is placed, what each pair's pictures said, and which
// points are drawn.
const placedAs = (how: ScenePlaced['how'], more: Partial<ScenePlaced> = {}): ScenePlaced => ({
  camera: 'hall',
  points: 100,
  placed: !['no_spot', 'no_facing', 'lost'].includes(how),
  how,
  size: 'approximate',
  against: [],
  shift_m: 0,
  turn_deg: 0,
  ...more,
});

const NOW = 1_762_500_000;
const followed = (more: Partial<NonNullable<ScenePlaced['follow']>> = {}): ScenePlaced['follow'] => ({
  by: 'turned',
  since: NOW - 30,
  lost: '',
  pinned: false,
  ...more,
});

it.each([
  [placedAs('spot'), 'Placed from its spot on your plan.'],
  [
    placedAs('adjusted', { against: ['porch'], shift_m: 0.161, turn_deg: 1.7 }),
    'Placed from its spot, adjusted against porch by what their pictures share: 16 cm and 2° from the spot.',
  ],
  [
    placedAs('following', { shift_m: 0.004, turn_deg: 90.2, follow: followed() }),
    "Turned 90° from your plan's spot, a moment ago. Placed where it stood, by the walls it now sees. Your plan is unchanged.",
  ],
  [
    placedAs('following', { shift_m: 4.53, turn_deg: 20, follow: followed({ by: 'walls', since: NOW - 240 }) }),
    "Turned 20°, moved 4.5 m from your plan's spot, 4 min ago. Placed by the walls it now sees, where they fit your plan. Your plan is unchanged.",
  ],
  [
    placedAs('following', { shift_m: 0.5, turn_deg: 0.2, follow: followed({ by: 'pictures', since: NOW - 7200 }) }),
    "Moved 50 cm from your plan's spot, 2.0 h ago. Placed by what its new picture shares with an earlier one or another camera's. Your plan is unchanged.",
  ],
  [
    placedAs('spot', { follow: followed({ by: '', pinned: true }) }),
    'Placed from its spot on your plan, as you asked. It is followed again when its view next changes.',
  ],
  [
    placedAs('following', { shift_m: 0.8, turn_deg: 54, follow: followed({ by: 'tracking' }) }),
    "Turned 54°, moved 80 cm from your plan's spot, a moment ago. Placed where it was tracked to as it moved. Your plan is unchanged.",
  ],
  [placedAs('no_spot'), 'Not in the scene: it has no spot on your plan. Give it one in the plan editor.'],
  [
    placedAs('lost', { follow: followed({ by: '', lost: 'no_walls' }) }),
    "Its view changed, and there is no wall in view to place it by. It isn't drawn, keeps where it last stood, and is tried again with each new picture.",
  ],
  [
    placedAs('lost', { follow: followed({ lost: 'ambiguous' }) }),
    "Its view changed, and the walls it sees fit your plan in more than one place. It isn't drawn, keeps where it last stood, and is tried again with each new picture.",
  ],
  [
    placedAs('lost', { follow: followed({ lost: 'no_match' }) }),
    "Its view changed, and the walls it sees don't fit your plan anywhere well enough. It isn't drawn, keeps where it last stood, and is tried again with each new picture.",
  ],
  [placedAs('lost', { follow: followed({ lost: 'pending' }) }), 'Its view changed. It is being placed again.'],
])('says how a camera is placed in the scene, or why it is not in it', (camera, text) => {
  expect(placedLine(camera, c => c, NOW)).toBe(text);
});

const tracked = (more: Partial<NonNullable<ScenePlaced['track']>> = {}): ScenePlaced['track'] => ({
  state: 'tracking',
  since: NOW - 40,
  at: NOW - 1,
  held: NOW - 2,
  pose: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  went_m: 0.9,
  from_m: 0.62,
  from_deg: 41.2,
  added: 12239,
  added_at: NOW - 3,
  ...more,
});

it.each([
  [
    tracked(),
    "It is moving, now 62 cm away and turned 41° from where its points were made, a moment ago. Held to your plan's walls. This move has added 12,239 points to the map, the last 3 s ago.",
  ],
  [
    tracked({ held: NOW - 13, added: 1, added_at: NOW - 200 }),
    "It is moving, now 62 cm away and turned 41° from where its points were made, a moment ago. Last held to your plan's walls 12 s before, so it may have drifted since. This move has added 1 point to the map, the last 3 min ago.",
  ],
  [
    tracked({ held: null, from_m: 0, from_deg: 12, added: 0, added_at: null }),
    'It is moving, now turned 12° from where its points were made, a moment ago. No wall has held it to your plan yet, so it may be off. This move has added nothing to the map yet.',
  ],
  [tracked({ pose: null, held: null }), 'It began to move. Its first frames are being placed.'],
  [
    tracked({ state: 'ended', at: NOW - 300 }),
    "It came to rest 5 min ago, 62 cm away and turned 41° from where its points were made. This move added 12,239 points to the map. It's placed for good after about 20 seconds with nobody in view.",
  ],
])('says where a moving camera is now and how far that can be trusted', (track, text) => {
  expect(trackLine(placedAs('spot', { track }), NOW)).toBe(text);
});

it("says on a moving camera's row what this move has added to the map, and nothing once it rests", () => {
  expect(addedNote(placedAs('spot', { track: tracked() }), NOW)).toEqual({
    wide: '+12,239 points, 3 s ago',
    narrow: '+12,239, 3 s',
    label: 'This move has added 12,239 points to the map, the last 3 s ago.',
  });
  expect(addedNote(placedAs('spot', { track: tracked({ added: 0, added_at: null }) }), NOW)).toEqual({
    wide: 'No points added yet',
    narrow: '+0',
    label: 'This move has added nothing to the map yet.',
  });
  expect(addedNote(placedAs('spot', { track: tracked({ state: 'ended' }) }), NOW)).toBeNull();
  expect(addedNote(placedAs('spot'), NOW)).toBeNull();
});

it("reads a tracked camera's own placement as background to where it is now", () => {
  const camera = placedAs('following', { shift_m: 0.07, turn_deg: 0.2, follow: followed(), track: tracked() });
  expect(placedLine(camera, c => c, NOW)).toBe(
    "Before it began to move: moved 7 cm from your plan's spot, a moment ago. Placed where it stood, by the walls it now sees. Your plan is unchanged.",
  );
});

const mapped = (more: Partial<NonNullable<ScenePlaced['map']>> = {}): ScenePlaced['map'] => ({
  points: 18204,
  state: 'added',
  why: '',
  added: 3100,
  left_out: 0,
  off_m: 0.03,
  moved_m: 0,
  turned_deg: 0,
  resized: 1,
  ...more,
});

it.each([
  [mapped(), 'The map holds 18,204 points from this camera: 3,100 from its latest picture.'],
  [mapped({ added: 0 }), 'The map holds 18,204 points from this camera: its latest picture showed nothing new.'],
  [mapped({ added: 18204 }), 'The map holds 18,204 points from this camera, all from its latest picture.'],
  [
    mapped({ moved_m: 0.98, turned_deg: 14.2, added: 5200 }),
    'The map holds 18,204 points from this camera: 5,200 from its latest picture. It was moved 98 cm and turned 14° to line up with what the map holds there.',
  ],
  [
    mapped({ off_m: 0.2, added: 900, left_out: 4200 }),
    "The map holds 18,204 points from this camera: 900 from its latest picture. It couldn't be lined up with what the map holds there (20 cm out). 4,200 of its points are left out, where better-placed pictures see empty space.",
  ],
  [
    mapped({ state: 'refused', why: 'apart', added: 0, points: 1 }),
    "Its latest picture is left out: it doesn't line up with what the map already holds, so it would draw the room twice. It's tried again with each new picture. The map holds 1 point from this camera.",
  ],
  [
    mapped({ state: 'refused', why: 'full', added: 0 }),
    "Its latest picture isn't in the map: the map holds as many points as it keeps, so newer pictures add nothing for now. The map holds 18,204 points from this camera.",
  ],
])("says what the place's map took of a camera's latest picture", (map, text) => {
  expect(mapLine(placedAs('spot', { map }))).toBe(text);
});

it.each([
  [
    { floor_m2: 33.5, plan_m2: 88.5 },
    "The map covers about 38\u00a0percent of your plan's floor so far (34\u00a0of\u00a089\u00a0m²).",
  ],
  [
    { floor_m2: 0, plan_m2: 88.5 },
    "The map covers about 0\u00a0percent of your plan's floor so far (0\u00a0of\u00a089\u00a0m²).",
  ],
  [
    { floor_m2: 90, plan_m2: 88.5 },
    "The map covers about 100\u00a0percent of your plan's floor so far (90\u00a0of\u00a089\u00a0m²).",
  ],
  [{ floor_m2: 3, plan_m2: 0 }, null],
  [null, null],
  [undefined, null],
])("says how much of the plan's floor the map covers so far", (covered, text) => {
  expect(coveredLine(covered)).toBe(text);
});

it("says nothing of the map before a camera's cloud was offered to it, and marks one that was left out", () => {
  expect(mapLine(placedAs('spot'))).toBeNull();
  expect(placedState('spot', null, true)).toEqual({ label: 'Left out', tone: 'warn' });
  expect(placedState('lost', null, true)).toEqual({ label: 'Not placed', tone: 'warn' });
  expect(placedState('spot', 'tracking', true)).toEqual({ label: 'Tracking', tone: 'plain' });
});

it('says nothing of tracking for a camera that is not moving, and names a moving one first', () => {
  expect(trackLine(placedAs('spot'))).toBeNull();
  expect(placedState('spot', 'tracking')).toEqual({ label: 'Tracking', tone: 'plain' });
  expect(placedState('following', 'ended')).toEqual({ label: 'Moved', tone: 'plain' });
});

it.each([
  [placedAs('following', { follow: followed() }), { use: 'spot', label: "Use the plan's spot" }],
  [placedAs('lost', { follow: followed({ lost: 'no_walls' }) }), { use: 'spot', label: "Use the plan's spot" }],
  [placedAs('spot', { follow: followed({ pinned: true }) }), { use: 'cloud', label: 'Follow it again' }],
  [placedAs('lost', { follow: followed({ lost: 'pending' }) }), null],
  [placedAs('spot'), null],
  [placedAs('adjusted'), null],
  [placedAs('no_spot'), null],
])('offers one way back, only where a camera was placed again or could not be', (camera, way) => {
  expect(wayBack(camera)).toEqual(way);
});

const pairOf = (state: ScenePair['state'], more: Partial<ScenePair> = {}): ScenePair => ({
  a: 'hall',
  b: 'porch',
  state,
  off_m: 0.34,
  off_deg: 0.5,
  measured: null,
  latest: null,
  ...more,
});

const squareOf = (more: Partial<SceneSquare> = {}): SceneSquare => ({
  state: 'squared',
  turn_deg: 0,
  right_m: 0,
  ahead_m: 0,
  slide: '',
  bend_deg: 0,
  ...more,
});

it.each([
  ['floor', 14.4, 0.1, 'Levelled from its own floor: the camera looks 14° down.'],
  ['floor', 28.3, 15.3, 'Levelled from its own floor: the camera looks 28° down and is rolled 15°.'],
  ['walls', 12.5, -1.3, 'No floor in view, so levelled from its walls: the camera looks 13° down and is rolled 1°.'],
  ['none', 0, 0, 'No floor or wall in view to level it by: it stands as if the camera were upright, so it may lean.'],
] as const)('says what a cloud was levelled by, and the camera tilt that gives: %s', (level, tilt, roll, line) => {
  expect(levelLine(level, tilt, roll)).toBe(line);
});

it('says nothing of the level for a scene that does not say', () => {
  expect(levelLine()).toBeNull();
});

it.each([
  [
    squareOf({ turn_deg: -3.5, right_m: 0.12, ahead_m: 0.16 }),
    'Squared to your plan by its walls: turned 3.5° and moved 20 cm.',
  ],
  [squareOf({ turn_deg: 0.2 }), "Its walls already run along your plan's."],
  [
    squareOf({ turn_deg: -2.73, slide: 'beyond' }),
    "Squared to your plan by its walls: turned 2.7°. Not slid onto the plan's walls: they are farther off than a slide goes.",
  ],
  [squareOf({ turn_deg: 1.9, slide: 'no_counterpart' }), 'Squared to your plan by its walls: turned 1.9°.'],
])('says how a cloud was squared to the plan: %#', (square, line) => {
  expect(squareLine(placedAs('spot', { square, squared: true }))).toBe(line);
});

const fitOf = (more: Partial<SceneFit> = {}): SceneFit => ({
  state: 'fitted',
  scale: 1,
  up: 1,
  dx: 0,
  dy: 0,
  walls: [{ off_deg: 0.2, before_m: 0.02, after_m: 0.0 }],
  seen: 1,
  ...more,
});

it.each([
  [
    fitOf({
      scale: 0.966,
      dx: 0.01,
      dy: 0.012,
      walls: [
        { off_deg: 0.3, before_m: 0.241, after_m: 0.04 },
        { off_deg: 7.6, before_m: -0.06, after_m: -0.13 },
      ],
    }),
    "Fitted to the room: resized by 3 percent (smaller), moved 2 cm, matched to 2 walls. Before, its walls stood up to 24 cm from your plan's and ran up to 8° off them.",
  ],
  [
    fitOf({ scale: 1.042 }),
    'Fitted to the room: resized by 4 percent (larger), matched to 1 wall. One wall gives its size, not where it stands across it.',
  ],
  [
    fitOf({
      scale: 0.998,
      walls: [
        { off_deg: 0.2, before_m: 0.04, after_m: 0 },
        { off_deg: 1.7, before_m: -0.026, after_m: 0 },
      ],
    }),
    'Fitted to the room: its size kept, matched to 2 walls.',
  ],
  [fitOf({ state: 'no_walls', walls: [] }), 'Not fitted to the room: no wall in view.'],
  [
    fitOf({ state: 'no_match', walls: [] }),
    "Not fitted to the room: the wall it sees isn't where a wall of your plan is. It may be an island, a half wall, or furniture that the plan doesn't draw.",
  ],
  [
    fitOf({ state: 'beyond', scale: 1.12, dx: -0.7, dy: 0 }),
    'Not fitted to the room: its walls ask for it to be resized by 12 percent (larger) and moved 70 cm, more than a fit may do. Check where its spot is.',
  ],
])('says how a cloud was fitted to the room, or in plain words why not: %#', (fit, line) => {
  expect(fitLine(fit)).toBe(line);
});

it('leaves the slide out of the squaring line of a fitted cloud, which the fit slides instead', () => {
  const square = squareOf({ turn_deg: -2.7, right_m: 0.2, ahead_m: 0.1, slide: 'beyond' });
  expect(squareLine(placedAs('spot', { square, squared: true, fitted: true }))).toBe(
    'Squared to your plan by its walls: turned 2.7°.',
  );
});

it("says a fitted camera's own cloud is fitted, in place of calling its size approximate", () => {
  const now = new Date(2026, 5, 2, 9, 0);
  const at = new Date(2026, 5, 2, 1, 40).getTime() / 1000;
  const view = { camera: 'hall', points: 1000, at, fit: fitOf({ scale: 0.966 }) };
  expect(viewNote({ place: 'p1', revision: 'r1', view }, 'Hall', now)).toBe(
    'What Hall sees on its own. 1,000 points, today 01:40. Fitted to the room: resized by 3 percent (smaller), matched to 1 wall. One wall gives its size, not where it stands across it.',
  );
});

it('says why a cloud was not squared, and nothing for a camera placed some other way', () => {
  expect(squareLine(placedAs('spot', { square: squareOf({ state: 'beyond', turn_deg: -25.2 }) }))).toBe(
    'Not squared: its walls line up with your plan only 25° from the facing you set, farther than a cloud is turned. Check which way its spot faces.',
  );
  expect(squareLine(placedAs('spot', { square: squareOf({ state: 'no_walls' }) }))).toBe(
    'No wall in view to square it to your plan by: it faces the way its spot says.',
  );
  expect(squareLine(placedAs('spot', { square: squareOf({ state: 'no_plan' }) }))).toBe(
    'No wall of your plan near it to square it to: it faces the way its spot says.',
  );
  expect(squareLine(placedAs('following', { square: squareOf({ turn_deg: 3 }), squared: false }))).toBeNull();
  expect(squareLine(placedAs('no_spot'))).toBeNull();
});

it('says that walls bend from the depth estimate only where they do', () => {
  expect(bendLine(placedAs('spot', { square: squareOf({ bend_deg: 7.9 }) }))).toBe(
    'Walls in this cloud bend by about 8°: that is the depth estimate, not the placement.',
  );
  expect(bendLine(placedAs('spot', { square: squareOf({ bend_deg: 1.4 }) }))).toBeNull();
  expect(bendLine(placedAs('spot'))).toBeNull();
});

it("says when a camera's picture was taken", () => {
  const now = new Date(2026, 5, 2, 9, 0);
  const at = new Date(2026, 5, 1, 15, 53).getTime() / 1000;
  expect(pictureLine(placedAs('spot', { at }), now)).toBe('From a picture taken yesterday 15:53.');
  expect(pictureLine(placedAs('spot'), now)).toBeNull();
});

it('says two squared cameras are checked by their pictures, not moved by them', () => {
  const pair: ScenePair = {
    a: 'hall',
    b: 'porch',
    state: 'checked',
    off_m: 0.21,
    off_deg: 3.2,
    measured: null,
    latest: null,
  };
  expect(pairLine(pair)).toBe(
    "hall and porch see the same space. Both are squared to your plan, which their pictures don't move: the pictures put the two 21 cm and 3° from where they stand.",
  );
});

it.each([
  ['spot', 'Placed', 'plain'],
  ['adjusted', 'Adjusted', 'plain'],
  ['following', 'Following', 'plain'],
  ['lost', 'Not placed', 'warn'],
  ['no_spot', 'No spot', 'plain'],
  ['no_facing', 'No spot', 'plain'],
] as const)("names a camera's state in a word, and marks the ones that want the owner: %s", (how, label, tone) => {
  expect(placedState(how)).toEqual({ label, tone });
});

it('says what an overlapping pair of cameras came to, and nothing for a pair that shares no space', () => {
  expect(pairLine(pairOf('tightened'))).toBe(
    'hall and porch see the same space and are adjusted to what their pictures share: the spots were 34 cm and 1° from it.',
  );
  // a correction measured by day is kept through a night look that didn't match, and says so
  expect(pairLine(pairOf('tightened', { latest: 'apart' }))).toMatch(
    /A later look didn't match well enough \(pictures at night often don't\), so that measurement is kept\.$/,
  );
  expect(pairLine(pairOf('disagrees', { off_m: 2.5, off_deg: 50 }))).toBe(
    'hall and porch: their pictures put them 2.5 m and 50° from where their spots do. One of the two spots is off. Nothing was moved.',
  );
  expect(pairLine(pairOf('unsure'))).toMatch(/placed from their spots only/);
  expect(pairLine(pairOf('apart'))).toBeNull();
  expect(pairLine(pairOf('moved'))).toBeNull();
});

it('draws only the points of the cameras shown, and leaves out those outside the plan when asked', () => {
  const views = new Uint16Array([0, 0, 1, 1]);
  const outside = new Uint8Array([0, 1, 0, 0]);
  const square = (i: number) => [i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3];
  expect([...pointIndices(4, views, outside, new Set(), false)]).toEqual([0, 1, 2, 3].flatMap(square));
  expect([...pointIndices(4, views, outside, new Set([1]), false)]).toEqual([0, 1].flatMap(square));
  expect([...pointIndices(4, views, outside, new Set(), true)]).toEqual([0, 2, 3].flatMap(square));
  expect(pointIndices(4, views, outside, new Set([0, 1]), true)).toHaveLength(0);
  // a cloud that says nothing of its points (one camera's own) is drawn whole
  expect(pointIndices(2, null, null, new Set([0]), true)).toHaveLength(12);
  expect(outsideCounts(views, outside)).toEqual([1]);
});

it("mixes each camera's colour into its points, and gives a point with no colour the camera's alone", () => {
  const views = new Uint16Array([0, 1]);
  const mixed = tinted(2, new Uint8Array([100, 100, 100, 200, 200, 200]), views);
  const [a, b] = [tintOf(0), tintOf(1)];
  expect([...mixed.slice(0, 3)]).toEqual(a.map(t => Math.round(100 * (1 - TINT_SHARE) + t * TINT_SHARE)));
  expect([...mixed.slice(3)]).toEqual(b.map(t => Math.round(200 * (1 - TINT_SHARE) + t * TINT_SHARE)));
  expect([...tinted(1, null, new Uint16Array([1]))]).toEqual([...b]);
});

it("reads whose each point is and whether it lies outside the plan from a scene's points", () => {
  const head = [
    'ply',
    'format binary_little_endian 1.0',
    'element vertex 2',
    'property float x',
    'property float y',
    'property float z',
    'property ushort view',
    'property ushort outside',
    'end_header',
    '',
  ].join('\n');
  const bytes = new ArrayBuffer(head.length + 2 * 16);
  new Uint8Array(bytes).set(new TextEncoder().encode(head));
  const body = new DataView(bytes, head.length);
  body.setUint16(12, 3, true);
  body.setUint16(16 + 12, 1, true);
  body.setUint16(16 + 14, 1, true);
  const cloud = parseCloud(bytes, 'ply');
  expect([...(cloud.views ?? [])]).toEqual([3, 1]);
  expect([...(cloud.outside ?? [])]).toEqual([0, 1]);
  expect(parseCloud(ply(), 'ply').views).toBeNull(); // a cloud with no such tags
});

it.each([
  ['people', /nobody in view/],
  ['gpu', /GPU memory/],
  ['no_runtime', /isn't set up on this machine/],
  ['not_chosen', /Choose it beside Learn surroundings in \{Live\}/],
] as const)('says why a camera has no cloud: %s', (why, text) => {
  expect(viewWhy({ why, cloud: false })).toMatch(text);
});

it('says what learning being off means for a cloud that is showing, and for a camera with none', () => {
  expect(viewWhy({ why: 'off', cloud: true })).toBe(
    "Learn surroundings is off, so this picture won't update. Turn it on in {Live}.",
  );
  expect(viewWhy({ why: 'off', cloud: false })).toBe(
    'Turn on Learn surroundings in {Live}, and each camera chosen there gets a cloud of what it sees.',
  );
});

it('says nothing stands in the way of a camera whose cloud is current', () => {
  expect(viewWhy({ why: null, cloud: true })).toBeNull();
  expect(viewWhy(undefined)).toBeNull();
});

it("says the owner's plan isn't drawn on a map not lined up with it, and nothing once it is", () => {
  const owned = { place: 'p1', revision: 'r2', plan: 'plan "Home"', plan_source: 'owner' } as const;
  expect(planNote({ ...owned, plan_aligned: false, plan_transform: null })).toBe(
    "Your plan isn't drawn on this map: the map isn't lined up with it. The Plan view shows your plan as you drew it.",
  );
  expect(planNote({ ...owned, plan_aligned: true })).toBeNull();
});

it('says nothing about a plan for a place with only a machine plan, or with no map', () => {
  expect(planNote({ place: 'p1', revision: 'r2', plan_source: 'machine', plan_aligned: true })).toBeNull();
  expect(planNote({ place: 'p1', revision: null, plan_source: 'owner', plan_aligned: false })).toBeNull();
});
