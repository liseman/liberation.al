// What the 3D view draws (#76, #216). The view is PlayCanvas's own SuperSplat viewer
// (@playcanvas/supersplat-viewer, MIT), embedded whole: its orbit, fly, and walk cameras, its pointer, touch,
// and key handling, its mode bar, help panel, settings, info, and fullscreen all run as it ships them. This file
// is only the glue: it hands the viewer the place's splat, turns it to the viewer's frame (frame.ts), adds our
// overlays to the viewer's PlayCanvas app (the map's points, the floor grid, the plan's outline, and the cameras'
// views), gives it the floor and the plan's walls to walk on and be stopped by (collision.ts), and names each
// camera to look from as one of the viewer's annotations, which is how the viewer flies to a view.
//
// The viewer reports a splat it can't load only to the console, and counts itself created before the load ends, so
// `splatLoaded` is how the page learns: it settles when the splat has drawn, or with why it couldn't (the engine's
// asset error, or a lost graphics device). The viewer is asked for WebGPU only where the browser has an adapter:
// asked without one, the engine logs an error before it falls back.
//
// A place with no splat still gets the viewer: its content is then a thin, see-through copy of the map's points,
// there only so the viewer knows the place's size, and the points themselves are drawn by our overlay.
//
// WebGPU has no point size, so the points are small screen-facing squares (two triangles each) grown from each
// point's centre in the vertex shader, on both backends. Lines are one pixel wide. The two materials are our own
// shaders, in GLSL for WebGL2 and WGSL for WebGPU.
import { defaultSettings } from '@playcanvas/supersplat-viewer/settings';
import { createViewer } from '@playcanvas/supersplat-viewer/viewer';
import '@playcanvas/supersplat-viewer/viewer.css';
import {
  BLEND_NORMAL,
  CULLFACE_NONE,
  Entity,
  type GraphicsDevice,
  Mesh,
  MeshInstance,
  PRIMITIVE_LINES,
  PRIMITIVE_TRIANGLES,
  SEMANTIC_COLOR,
  SEMANTIC_POSITION,
  SEMANTIC_TEXCOORD0,
  ShaderMaterial,
  TYPE_UINT8,
} from 'playcanvas';
import type { Scene } from '../api';
import { collisionMesh, FLOOR_MARGIN, glb, glbUrl } from './collision';
import { type Quat, toViewerUp, turned } from './frame';
import { coverFov, frustum, heightColour, heightOf, planOutline, planWalls, robustRange, type V3 } from './geometry';
import { pointIndices, tinted, tintOf } from './layers';
import type { Cloud } from './load';

/** How the points are coloured: their own colours, their height above the floor, or their own with their camera's
 *  tint mixed in (a scene of several cameras' clouds, layers.ts). */
export type Colour = 'rgb' | 'height' | 'camera';

/** An opening view that isn't a camera's: the whole scene from above, looking down at the plan. */
export const ABOVE = '@above';

/** Where the viewer keeps whether its controls panel is open (its own key). */
const CONTROLS_OPEN = 'controlsPanelOpen';

/** Where the view looks from, in the viewer's frame, and the viewer's camera mode: what a view built again around
 *  a newer map of the same place opens with, so it stays where the person had it. */
export interface View {
  position: V3;
  target: V3;
  fov: number;
  mode: string;
}

/** Everything the view draws, built once per map (and splat), and what the page asks of it. */
export interface Built {
  /** Where the view looks from now, or null before the viewer has drawn. */
  view: () => View | null;
  /** The overview: the whole place in view, in the viewer's orbit mode. */
  home: () => void;
  /** Fly to where a camera stands, looking where it looks. */
  lookFrom: (name: string) => void;
  /** What's shown: the splat in place of the points, the points' size and colouring, and the overlays. */
  show: (s: {
    splat: boolean;
    size: number;
    colour: Colour;
    floor: boolean;
    cameras: boolean;
    plan: boolean;
    /** The cameras whose points aren't drawn, by their place in the scene (a scene of several cameras' clouds). */
    hidden?: ReadonlySet<number>;
    /** Leave out the points that lie outside the plan. */
    inside?: boolean;
  }) => void;
  /** Settles once the splat has drawn its first frame, or rejects with why it couldn't be drawn; settled at once
   *  when the place was built without a splat. */
  /** Draw a later reading of the same scene in place: its cameras where they now are and, with `cloud`, its points
   *  as they now are. The view (where it looks from, how, and every setting) is left exactly as it is. */
  update: (data: Scene, cloud: Cloud | null) => void;
  splatLoaded: Promise<void>;
  /** The cameras drawn, by name. */
  cameraNames: string[];
  /** Free everything and take the viewer out of the page. */
  dispose: () => void;
}

const POINT_GLSL = {
  vertex: /* glsl */ `
    attribute vec3 vertex_position;
    attribute vec4 vertex_color;
    attribute vec2 vertex_corner;
    uniform mat4 matrix_model;
    uniform mat4 matrix_viewProjection;
    uniform vec2 view_size;
    uniform float point_size;
    varying vec4 vColor;
    void main(void) {
      vec4 clip = matrix_viewProjection * matrix_model * vec4(vertex_position, 1.0);
      clip.xy += vertex_corner * point_size / view_size * clip.w;
      gl_Position = clip;
      vColor = vertex_color;
    }`,
  fragment: /* glsl */ `
    varying vec4 vColor;
    void main(void) {
      gl_FragColor = vec4(vColor.rgb, 1.0);
    }`,
};

const POINT_WGSL = {
  vertex: /* wgsl */ `
    attribute vertex_position: vec3f;
    attribute vertex_color: vec4f;
    attribute vertex_corner: vec2f;
    uniform matrix_model: mat4x4f;
    uniform matrix_viewProjection: mat4x4f;
    uniform view_size: vec2f;
    uniform point_size: f32;
    varying vColor: vec4f;
    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
      var output: VertexOutput;
      var clip = uniform.matrix_viewProjection * uniform.matrix_model * vec4f(input.vertex_position, 1.0);
      clip = vec4f(clip.xy + input.vertex_corner * uniform.point_size / uniform.view_size * clip.w, clip.zw);
      output.position = clip;
      output.vColor = input.vertex_color;
      return output;
    }`,
  fragment: /* wgsl */ `
    varying vColor: vec4f;
    @fragment
    fn fragmentMain(input: FragmentInput) -> FragmentOutput {
      var output: FragmentOutput;
      output.color = vec4f(input.vColor.rgb, 1.0);
      return output;
    }`,
};

const LINE_GLSL = {
  vertex: /* glsl */ `
    attribute vec3 vertex_position;
    uniform mat4 matrix_model;
    uniform mat4 matrix_viewProjection;
    void main(void) {
      gl_Position = matrix_viewProjection * matrix_model * vec4(vertex_position, 1.0);
    }`,
  fragment: /* glsl */ `
    uniform vec4 line_color;
    void main(void) {
      gl_FragColor = line_color;
    }`,
};

const LINE_WGSL = {
  vertex: /* wgsl */ `
    attribute vertex_position: vec3f;
    uniform matrix_model: mat4x4f;
    uniform matrix_viewProjection: mat4x4f;
    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
      var output: VertexOutput;
      output.position = uniform.matrix_viewProjection * uniform.matrix_model * vec4f(input.vertex_position, 1.0);
      return output;
    }`,
  fragment: /* wgsl */ `
    uniform line_color: vec4f;
    @fragment
    fn fragmentMain(input: FragmentInput) -> FragmentOutput {
      var output: FragmentOutput;
      output.color = uniform.line_color;
      return output;
    }`,
};

/** A CSS colour (a custom property's value, or `fallback`) as 0 to 1 red, green, and blue. */
function cssColour(host: HTMLElement, name: string, fallback: string): [number, number, number] {
  const probe = document.createElement('span');
  probe.style.color = `var(${name}, ${fallback})`;
  host.appendChild(probe);
  const rgb = getComputedStyle(probe).color.match(/[\d.]+/g);
  probe.remove();
  const [r = 255, g = 255, b = 255] = (rgb ?? []).map(Number);
  return [r / 255, g / 255, b / 255];
}

/** Line segments (pairs of points) in one colour, as an entity; `alpha` under 1 blends them over what's behind. */
function lines(
  device: GraphicsDevice,
  app: Entity['_app'],
  name: string,
  segments: V3[],
  rgb: [number, number, number],
  alpha = 1,
): Entity {
  const material = new ShaderMaterial({
    uniqueName: 'PlaceLines',
    vertexGLSL: LINE_GLSL.vertex,
    fragmentGLSL: LINE_GLSL.fragment,
    vertexWGSL: LINE_WGSL.vertex,
    fragmentWGSL: LINE_WGSL.fragment,
    attributes: { vertex_position: SEMANTIC_POSITION },
  });
  material.setParameter('line_color', [...rgb, alpha]);
  if (alpha < 1) {
    material.blendType = BLEND_NORMAL;
    material.depthWrite = false;
  }
  material.update();
  const entity = new Entity(name, app);
  if (!segments.length) return entity; // nothing to draw: a place with no cameras, or no plan
  const mesh = new Mesh(device);
  mesh.setPositions(new Float32Array(segments.flat()));
  mesh.update(PRIMITIVE_LINES);
  entity.addComponent('render', { meshInstances: [new MeshInstance(mesh, material)], castShadows: false });
  return entity;
}

/** The corners a point's square grows to, in its own half-sizes. */
const CORNERS = [-1, -1, 1, -1, 1, 1, -1, 1];

/** Each point's value four times over, once a corner: `width` values a point in, `width` (and a filler, when
 *  `pad` is given) a corner out. */
function perCorner<T extends Float32Array | Uint8Array>(
  out: T,
  values: ArrayLike<number>,
  width: number,
  pad?: number,
): T {
  const step = width + (pad === undefined ? 0 : 1);
  for (let i = 0, n = values.length / width; i < n; i++)
    for (let c = 0; c < 4; c++) {
      const o = (i * 4 + c) * step;
      for (let k = 0; k < width; k++) out[o + k] = values[i * width + k] ?? 0;
      if (pad !== undefined) out[o + width] = pad;
    }
  return out;
}

/** At most this many of the map's points stand in for a splat the place doesn't have (see the top). */
const STAND_IN_POINTS = 20_000;

/** A 3DGS PLY of see-through specks at some of the map's points: content for the viewer when there's no splat. */
function standIn(cloud: Cloud): ArrayBuffer {
  const step = Math.max(1, Math.ceil(cloud.count / STAND_IN_POINTS));
  const count = Math.ceil(cloud.count / step);
  const names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2'];
  names.push('rot_0', 'rot_1', 'rot_2', 'rot_3');
  const head = new TextEncoder().encode(
    `ply\nformat binary_little_endian 1.0\nelement vertex ${count}\n${names.map(n => `property float ${n}\n`).join('')}end_header\n`,
  );
  const rows = new Float32Array(count * names.length);
  for (let i = 0; i < count; i++) {
    const from = i * step * 3;
    // opacity is a logit and scale a log: far below anything drawn
    rows.set(
      [
        cloud.positions[from] ?? 0,
        cloud.positions[from + 1] ?? 0,
        cloud.positions[from + 2] ?? 0,
        0,
        0,
        0,
        -12,
        -6,
        -6,
        -6,
        1,
        0,
        0,
        0,
      ],
      i * names.length,
    );
  }
  const out = new Uint8Array(head.length + rows.byteLength);
  out.set(head);
  out.set(new Uint8Array(rows.buffer), head.length);
  return out.buffer;
}

/** The field of view the viewer wants for a camera's frame filling a view of `aspect`: of the view's longer side,
 *  in degrees (it takes its field of view across whichever side is longer). */
function viewerFov(cam: { width: number; height: number; focal: number }, aspect: number): number {
  const vertical = coverFov(cam.width, cam.height, cam.focal, aspect);
  if (aspect <= 1) return vertical;
  return (2 * Math.atan(Math.tan((vertical * Math.PI) / 360) * aspect) * 180) / Math.PI;
}

/** The viewer's own bounds on a field of view (its CAMERA_FOV_RANGE). */
const clampFov = (fov: number) => Math.max(10, Math.min(120, fov));

/** Whether the browser can give the viewer a WebGPU device; without one it's asked for WebGL2 outright. */
async function hasWebGPU(): Promise<boolean> {
  try {
    const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    return !!(await gpu?.requestAdapter());
  } catch {
    return false;
  }
}

/** An engine error as a reason to show: its message, without the stack. */
const reasonOf = (err: unknown): string =>
  (err instanceof Error ? err.message : String(err ?? '')).split('\n')[0]?.trim() || 'the viewer gave no reason';

export async function build(
  host: HTMLDivElement,
  data: Scene,
  cloud: Cloud,
  splat: ArrayBuffer | null,
  opening: string | View | null,
): Promise<Built> {
  const floorPlane = data.floor ?? [0, 0, 1, 0];
  const [nx = 0, ny = 0, nz = 1] = floorPlane;
  const upLength = Math.hypot(nx, ny, nz) || 1;
  const up: V3 = [nx / upLength, ny / upLength, nz / upLength];
  // Map to viewer: the rotation the splat's and the overlays' entities get, and the points' positions go through.
  const turn: Quat = toViewerUp(up);
  const inViewer = (p: readonly number[]): V3 => turned([p[0] ?? 0, p[1] ?? 0, p[2] ?? 0], turn);

  // What the view frames, and the floor reaches under: everything drawn (the cloud, the plan, and the cameras).
  const lo: V3 = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const hi: V3 = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const grow = (p: readonly number[]) => {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k] ?? 0, p[k] ?? 0);
      hi[k] = Math.max(hi[k] ?? 0, p[k] ?? 0);
    }
  };
  const { count, positions } = cloud;
  for (let i = 0; i < count; i++) grow([positions[i * 3] ?? 0, positions[i * 3 + 1] ?? 0, positions[i * 3 + 2] ?? 0]);

  // Each camera's view: lines from its centre to the four corners of its image, and the image's edge.
  const views: V3[] = [];
  const viewOf: Record<string, V3[]> = {};
  const marksOf = (scene: Scene): Record<string, V3[]> => {
    const out: Record<string, V3[]> = {};
    for (const [name, cam] of Object.entries(scene.cameras ?? {})) {
      const [o, a, b, c, d] = frustum(cam.pose, cam.focal, cam.width, cam.height) as [V3, V3, V3, V3, V3];
      out[name] = [o, a, o, b, o, c, o, d, a, b, b, c, c, d, d, a];
    }
    // a camera that is being carried also has where it went drawn, at the height it is held, in its own colour
    for (const placed of scene.combined?.cameras ?? []) {
      const mark = out[placed.camera];
      const path = placed.track?.path ?? [];
      if (!mark?.[0] || path.length < 2) continue;
      const height = mark[0][2];
      for (let i = 1; i < path.length; i++) {
        const [ax = 0, ay = 0] = path[i - 1] ?? [];
        const [bx = 0, by = 0] = path[i] ?? [];
        mark.push([ax, ay, height], [bx, by, height]);
      }
    }
    return out;
  };
  Object.assign(viewOf, marksOf(data));
  for (const segments of Object.values(viewOf)) views.push(...segments);
  views.forEach(grow);

  // The plan's rooms, outlined on the floor, and its walls, for the walk to be stopped by.
  const outline: V3[] = [];
  let walls: [V3, V3][] = [];
  if (data.plan && data.plan_transform) {
    try {
      for (const ring of planOutline(data.plan, data.plan_transform))
        ring.forEach((p, i) => {
          outline.push(p, ring[(i + 1) % ring.length] as V3);
        });
      walls = planWalls(data.plan, data.plan_transform);
    } catch {
      // a plan that doesn't read is left off; the cloud still draws, and the walk has only the floor
    }
  }
  outline.forEach(grow);

  const empty = !Number.isFinite(lo[0]);
  const centre: V3 = empty ? [0, 0, 0] : [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const radius = empty ? 5 : Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;

  // Two directions across the floor, and a point on it under the middle of everything.
  const drop = heightOf(centre, floorPlane);
  const onFloor: V3 = [centre[0] - up[0] * drop, centre[1] - up[1] * drop, centre[2] - up[2] * drop];
  const seed: V3 = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = (a: V3): V3 => {
    const n = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / n, a[1] / n, a[2] / n];
  };
  const along = unit(cross(up, seed));
  const across = unit(cross(up, along));
  const onGrid = (s: number, t: number): V3 => [
    onFloor[0] + along[0] * s + across[0] * t,
    onFloor[1] + along[1] * s + across[1] * t,
    onFloor[2] + along[2] * s + across[2] * t,
  ];

  // What the walk stands on and is stopped by, in the viewer's frame.
  const far = radius + FLOOR_MARGIN;
  const ground = [onGrid(-far, -far), onGrid(far, -far), onGrid(far, far), onGrid(-far, far)].map(inViewer);
  const blocking = walls.map(([a, b]) => [inViewer(a), inViewer(b)] as [V3, V3]);
  const collisionUrl = glbUrl(glb(collisionMesh(ground, blocking, [0, 1, 0])));

  // The viewer's settings: our page's background, no animation, each camera a view to fly to, and the view a splat
  // opens from (one whose frames it trained on, #44).
  const aspect = Math.max(1, host.clientWidth) / Math.max(1, host.clientHeight);
  const names = Object.keys(data.cameras ?? {});
  const poseOf = (name: string) => {
    const cam = data.cameras?.[name];
    if (!cam) return null;
    // Camera to map, row by row; the camera looks along its +z (OpenCV), so its centre and forward are columns 3 and 2.
    const p = cam.pose as number[];
    const eye: V3 = [p[3] ?? 0, p[7] ?? 0, p[11] ?? 0];
    const ahead: V3 = [eye[0] + (p[2] ?? 0), eye[1] + (p[6] ?? 0), eye[2] + (p[10] ?? 1)];
    return { position: inViewer(eye), target: inViewer(ahead), fov: clampFov(viewerFov(cam, aspect)) };
  };
  const settings = defaultSettings();
  settings.background.color = cssColour(host, '--lib-card', '#ffffff');
  settings.animTracks = [];
  settings.startMode = 'default';
  settings.annotations = names.flatMap(name => {
    const pose = poseOf(name);
    return pose ? [{ position: pose.position, title: name, text: '', camera: { initial: pose } }] : [];
  });
  const kept = typeof opening === 'string' ? null : opening;
  // From above: over the middle of everything, high enough to take it all in, a hair off straight down so the
  // viewer's orbit has a way up.
  const lift = Math.max(4, radius * 3.2); // far enough that the whole plan is in view, whatever the pane's shape
  const above = {
    position: inViewer([
      onFloor[0] + up[0] * lift - across[0] * 0.02 * lift,
      onFloor[1] + up[1] * lift - across[1] * 0.02 * lift,
      onFloor[2] + up[2] * lift - across[2] * 0.02 * lift,
    ]),
    target: inViewer(onFloor),
    fov: 50,
  };
  const fromAbove = opening === ABOVE;
  const first = fromAbove ? above : typeof opening === 'string' ? poseOf(opening) : null;
  const initial = kept ? { position: kept.position, target: kept.target, fov: kept.fov } : first;
  settings.cameras = initial ? [{ initial }] : [];

  // With its length, so the viewer's loading bar knows when it's done. The viewer gets the file only once we're
  // listening for its load to fail (`hand`), so a failure can't come before anyone hears it.
  const file = splat ?? standIn(cloud);
  let hand = () => {};
  const listening = new Promise<void>(resolve => {
    hand = resolve;
  });
  // The viewer opens its controls panel (the keys of the mode it's in) for anyone who hasn't closed it, over a
  // quarter of the view, and most of a phone's. Here it starts closed; its keyboard button opens it, and the viewer
  // remembers that choice in the same place.
  try {
    if (localStorage.getItem(CONTROLS_OPEN) === null) localStorage.setItem(CONTROLS_OPEN, 'false');
  } catch {
    // no storage: the viewer shows its panel, as it does for any page denied storage
  }
  const viewer = await createViewer({
    container: host,
    settings,
    renderer: (await hasWebGPU()) ? 'webgpu' : 'webgl',
    contentUrl: splat ? 'splat.sog' : 'points.ply',
    contentFilename: splat ? 'splat.sog' : 'points.ply',
    contents: listening.then(() => new Response(file, { headers: { 'content-length': String(file.byteLength) } })),
    collisionUrl,
    // The viewer always has a camera tour of its own making; it waits for its play button.
    noanim: true,
  });
  const app = viewer.app;
  const device = app.graphicsDevice;
  let gone = false;
  const draw = () => {
    if (!gone) app.renderNextFrame = true;
  };

  // Whether the content loads: the viewer says when it has drawn, the engine when the file can't be read, and
  // the device when it's lost.
  const splatLoaded = new Promise<void>((resolve, reject) => {
    if (viewer.state.loaded) resolve();
    else viewer.events.once('loaded:changed', () => resolve());
    app.assets.on('error', (err: unknown, asset?: { type?: string }) => {
      if (asset?.type === 'gsplat') reject(new Error(reasonOf(err)));
    });
    device.on('devicelost', () => reject(new Error('the graphics device was lost')));
  });
  splatLoaded.catch(() => {}); // the page listens too; a place nobody is looking at any more isn't an error
  hand();

  // The viewer turns its content half a turn about z, as a 3DGS file wants; ours is in the map's frame, so its
  // entity gets the map's rotation the moment it joins the scene, before the viewer measures it.
  let content: Entity | null = null;
  let wantSplat = true;
  const adopt = () => {
    const found = app.root.findByName('gsplat') as Entity | null;
    if (!found || found === content) return;
    content = found;
    found.setLocalRotation(...turn);
  };
  app.root.on('childinsert', adopt);
  adopt();

  // Our overlays, in the map's frame under one turned parent.
  const overlays = new Entity('place-overlays', app);
  overlays.setLocalRotation(...turn);
  app.root.addChild(overlays);

  const pointMaterial = new ShaderMaterial({
    uniqueName: 'PlacePoints',
    vertexGLSL: POINT_GLSL.vertex,
    fragmentGLSL: POINT_GLSL.fragment,
    vertexWGSL: POINT_WGSL.vertex,
    fragmentWGSL: POINT_WGSL.fragment,
    attributes: { vertex_position: SEMANTIC_POSITION, vertex_color: SEMANTIC_COLOR, vertex_corner: SEMANTIC_TEXCOORD0 },
  });
  pointMaterial.cull = CULLFACE_NONE;
  pointMaterial.setParameter('point_size', 5);
  pointMaterial.setParameter('view_size', [1, 1]);
  pointMaterial.update();
  const points = new Entity('points', app);
  // The points drawn, and how: which cameras' are hidden, whether those outside the plan are, and what they are
  // coloured by. These outlive the points themselves, which are laid again in place when the map gains some
  // (`update`): the same view, the same settings, other points.
  let laid = cloud;
  let pointMesh: Mesh | null = null;
  let hiddenNow: ReadonlySet<number> = new Set();
  let insideNow = false;
  let colourNow: Colour = 'rgb';
  let splatShown = false;
  let anyPoint = false;
  let colours: { height: Uint8Array; rgb: Uint8Array | null; camera: Uint8Array | null } | null = null;
  const paint = () => {
    if (!pointMesh || !colours) return;
    const by =
      colourNow === 'camera' && colours.camera ? 'camera' : colourNow !== 'height' && colours.rgb ? 'rgb' : 'height';
    pointMesh.setVertexStream(SEMANTIC_COLOR, colours[by] ?? colours.height, 4, laid.count * 4, TYPE_UINT8, true);
  };
  // Only the squares of the points shown are drawn, by building the index buffer again. Nothing is loaded again.
  const pick = () => {
    if (!pointMesh) return;
    const indices = pointIndices(laid.count, laid.views, laid.outside, hiddenNow, insideNow);
    anyPoint = indices.length > 0;
    if (anyPoint) pointMesh.setIndices(indices);
  };
  const layPoints = (next: Cloud) => {
    laid = next;
    const n = next.count;
    // Colour by height: each point's height above the floor on a ramp over the middle 98% of heights.
    const heights = new Float32Array(n);
    for (let i = 0; i < n; i++)
      heights[i] = heightOf(
        [next.positions[i * 3] ?? 0, next.positions[i * 3 + 1] ?? 0, next.positions[i * 3 + 2] ?? 0],
        floorPlane,
      );
    const [low, high] = robustRange(heights);
    const ramp = new Uint8Array(n * 3);
    for (let i = 0; i < n; i++)
      ramp.set(
        heightColour(heights[i] ?? 0, low, high).map(v => Math.round(v * 255)),
        i * 3,
      );
    colours = {
      height: perCorner(new Uint8Array(n * 16), ramp, 3, 255),
      rgb: next.colours ? perCorner(new Uint8Array(n * 16), next.colours, 3, 255) : null,
      camera: next.views ? perCorner(new Uint8Array(n * 16), tinted(n, next.colours, next.views), 3, 255) : null,
    };
    pointMesh = null;
    anyPoint = false;
    const instances: MeshInstance[] = [];
    if (n) {
      const mesh = new Mesh(device);
      mesh.setPositions(perCorner(new Float32Array(n * 12), next.positions, 3));
      const corners = new Float32Array(n * 8);
      for (let i = 0; i < n; i++) corners.set(CORNERS, i * 8);
      mesh.setVertexStream(SEMANTIC_TEXCOORD0, corners, 2);
      mesh.setIndices(pointIndices(n, null, null, new Set(), false));
      pointMesh = mesh;
      pick();
      paint();
      mesh.update(PRIMITIVE_TRIANGLES);
      instances.push(new MeshInstance(mesh, pointMaterial));
    }
    // the component lets go of the points it drew before, and their buffers with them
    if (points.render) points.render.meshInstances = instances;
    else points.addComponent('render', { meshInstances: instances, castShadows: false });
    points.enabled = !splatShown && anyPoint;
  };
  const layersShown = (hidden: ReadonlySet<number>, inside: boolean) => {
    const same = inside === insideNow && hidden.size === hiddenNow.size && [...hidden].every(i => hiddenNow.has(i));
    if (same) return;
    hiddenNow = new Set(hidden);
    insideNow = inside;
    pick();
    pointMesh?.update(PRIMITIVE_TRIANGLES);
  };
  const colourBy = (want: Colour) => {
    if (want === colourNow) return;
    colourNow = want;
    paint();
    pointMesh?.update(PRIMITIVE_TRIANGLES);
  };
  layPoints(cloud);
  overlays.addChild(points);
  // The squares are sized in the page's pixels, so they follow the view's size.
  const sized = () =>
    pointMaterial.setParameter('view_size', [Math.max(1, host.clientWidth), Math.max(1, host.clientHeight)]);
  app.on('update', sized);

  // In a scene of several cameras' clouds each camera's view is drawn in its own colour (layers.ts), whatever the
  // points are coloured by, so the colour beside its name always points at something. A camera that moves has its
  // view drawn again where it now is (`update`); nothing else is touched.
  const order = data.combined?.cameras.map(c => c.camera);
  const layCameras = (marks: Record<string, V3[]>): Entity => {
    if (!order)
      return lines(
        device,
        app,
        'cameras',
        Object.values(marks).flat(),
        cssColour(host, '--mantine-primary-color-filled', '#228be6'),
      );
    const all = new Entity('cameras', app);
    for (const [name, segments] of Object.entries(marks)) {
      const [r, g, b] = tintOf(Math.max(0, order.indexOf(name)));
      all.addChild(lines(device, app, `camera-${name}`, segments, [r / 255, g / 255, b / 255]));
    }
    return all;
  };
  let cameras = layCameras(viewOf);
  overlays.addChild(cameras);
  const plan = lines(device, app, 'plan', outline, [0xe8 / 255, 0x59 / 255, 0x0c / 255]);
  overlays.addChild(plan);

  // The floor: a grid on the floor plane under all of it.
  const reach = Math.max(4, Math.ceil(radius * 2.2)) / 2;
  const gridLines: V3[] = [];
  for (let i = 0; i <= 24; i++) {
    const t = -reach + (2 * reach * i) / 24;
    gridLines.push(onGrid(t, -reach), onGrid(t, reach), onGrid(-reach, t), onGrid(reach, t));
  }
  const floor = lines(device, app, 'floor', gridLines, [0.6, 0.6, 0.6], 0.45);
  overlays.addChild(floor);

  // The viewer's keys are the window's (it listens there), so it has them only while the pointer is over it or the
  // focus is in it: typing anywhere else on the page isn't a camera command.
  viewer.state.inputEnabled = false;
  const mine = () => {
    viewer.state.inputEnabled = true;
  };
  const notMine = () => {
    viewer.state.inputEnabled = host.contains(document.activeElement) && document.activeElement !== document.body;
  };
  host.addEventListener('pointerenter', mine);
  host.addEventListener('pointerdown', mine);
  host.addEventListener('pointerleave', notMine);

  // A splat opens from a view it trained on, standing there: in the viewer's walk mode where the place is large
  // enough for it, else flying. A place opened without such a view starts as an overview in orbit.
  const ready = (run: () => void) => {
    if (gone) return;
    if (viewer.state.loaded) run();
    else viewer.events.once('loaded:changed', () => !gone && run());
  };
  const home = () =>
    ready(() => {
      viewer.frameScene();
      draw();
    });
  if (first && !fromAbove)
    ready(() => {
      viewer.state.cameraMode = viewer.state.walkAllowed ? 'walk' : 'fly';
      draw();
    });
  else if (fromAbove)
    ready(draw); // the viewer's own orbit, from the opening view over the plan
  else if (kept)
    // A view kept from the map before this one: where it was, moving as it was (orbit is the viewer's own start).
    ready(() => {
      if (kept.mode === 'fly' || (kept.mode === 'walk' && viewer.state.walkAllowed))
        viewer.state.cameraMode = kept.mode;
      draw();
    });
  else home();

  return {
    home,
    view: () => {
      // The viewer has no call for where its camera is, so it's read from the engine: the viewer's own camera
      // entity. Anything unexpected there means no view is kept, and the next map opens as a first one does.
      try {
        const camera = app.root.findByName('camera') as Entity | null;
        if (gone || !viewer.state.loaded || !camera?.camera) return null;
        const at = camera.getPosition();
        const ahead = camera.forward;
        const centreNow = inViewer(centre);
        // The target is the point ahead that orbiting turns about: as far off as the middle of the place.
        const reach = Math.max(0.5, Math.hypot(at.x - centreNow[0], at.y - centreNow[1], at.z - centreNow[2]));
        const view: View = {
          position: [at.x, at.y, at.z],
          target: [at.x + ahead.x * reach, at.y + ahead.y * reach, at.z + ahead.z * reach],
          fov: camera.camera.fov,
          mode: viewer.state.cameraMode,
        };
        return [...view.position, ...view.target, view.fov].every(Number.isFinite) ? view : null;
      } catch {
        return null;
      }
    },
    lookFrom: name =>
      ready(() => {
        const at = names.indexOf(name);
        if (at >= 0) viewer.selectAnnotation(at);
        draw();
      }),
    show: s => {
      // Overlays stay in the scene for the life of the view. A checkbox only enables or disables them, and a
      // call after the view was torn down does nothing: the page may still be flushing one.
      if (gone) return;
      wantSplat = s.splat;
      const showSplat = s.splat && !!splat;
      splatShown = showSplat;
      layersShown(s.hidden ?? new Set(), !!s.inside);
      points.enabled = !showSplat && anyPoint;
      // The viewer counts itself loaded once its content has drawn a frame, so the content is only hidden after
      // that; the stand-in for a missing splat is see-through and stays.
      if (splat)
        ready(() => {
          if (gone) return;
          if (content) content.enabled = wantSplat;
          draw();
        });
      pointMaterial.setParameter('point_size', s.size);
      colourBy(s.colour);
      floor.enabled = s.floor;
      cameras.enabled = s.cameras;
      plan.enabled = s.plan;
      draw();
    },
    update: (next, nextCloud) => {
      if (gone) return;
      if (nextCloud) layPoints(nextCloud);
      const shown = cameras.enabled;
      overlays.removeChild(cameras);
      cameras.destroy();
      cameras = layCameras(marksOf(next));
      cameras.enabled = shown;
      overlays.addChild(cameras);
      draw();
    },
    splatLoaded: splat ? splatLoaded : Promise.resolve(),
    cameraNames: names,
    dispose: () => {
      if (gone) return;
      gone = true;
      host.removeEventListener('pointerenter', mine);
      host.removeEventListener('pointerdown', mine);
      host.removeEventListener('pointerleave', notMine);
      try {
        viewer.destroy();
      } catch {
        // The device can already be gone when the page tears the view down a second time.
      }
    },
  };
}
