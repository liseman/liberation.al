// What the 3D view draws, on PlayCanvas (MIT; #76): the map's points, the floor grid, the plan's outline, the
// cameras' views, and the place's Gaussian splat, in the map's own frame (metres, up along the floor's normal).
// PlayCanvas draws on WebGPU where the browser has it and on WebGL2 otherwise, and reads the splat's SOG file
// itself. Drawing is on demand: a frame is drawn when the view changes or the splat's sort finishes, never in a
// loop, so an open view costs next to nothing while it's still.
//
// WebGPU has no point size, so the points are small screen-facing squares (two triangles each) grown from each
// point's centre in the vertex shader, on both backends. Lines are one pixel wide, as the plan and frustums were
// before. The two materials are our own shaders, in GLSL for WebGL2 and WGSL for WebGPU.
import {
  AppBase,
  AppOptions,
  Asset,
  BLEND_NORMAL,
  CameraComponentSystem,
  Color,
  CULLFACE_NONE,
  createGraphicsDevice,
  Entity,
  FILLMODE_NONE,
  type GraphicsDevice,
  GSplatComponentSystem,
  GSplatHandler,
  Mat4,
  Mesh,
  MeshInstance,
  PRIMITIVE_LINES,
  PRIMITIVE_TRIANGLES,
  RESOLUTION_AUTO,
  RenderComponentSystem,
  SEMANTIC_COLOR,
  SEMANTIC_POSITION,
  SEMANTIC_TEXCOORD0,
  ShaderMaterial,
  TextureHandler,
  TYPE_UINT8,
  Vec3,
  Vec4,
} from 'playcanvas';
import type { Scene } from '../api';
import { coverFov, frustum, heightColour, heightOf, planOutline, robustRange, type V3 } from './geometry';
import type { Cloud } from './load';
import { dolly, slide, turn } from './orbit';

export type Colour = 'rgb' | 'height';

/** Everything the view draws, built once per map, and what the page asks of it. */
export interface Built {
  /** The overview: everything drawn, from above and to one side. */
  home: () => void;
  /** Put the view where a camera stands, looking where it looks, filled with its frame. */
  lookFrom: (name: string) => void;
  /** Fit the drawing to its host's size, and draw. */
  resize: () => void;
  /** Draw one frame. */
  render: () => void;
  /** What's shown: the splat in place of the points, the points' size and colouring, and the overlays. */
  show: (s: { splat: boolean; size: number; colour: Colour; floor: boolean; cameras: boolean; plan: boolean }) => void;
  /** Put a splat (a SOG file's bytes) in the scene, replacing any there; resolves once it can be drawn. */
  setSplat: (bytes: ArrayBuffer) => Promise<void>;
  /** Take the splat out of the scene and free it. */
  dropSplat: () => void;
  /** The cameras drawn, by name. */
  cameraNames: string[];
  /** Which backend is drawing. */
  backend: 'webgpu' | 'webgl2';
  /** Free everything and take the canvas out of the page. */
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
function lines(device: GraphicsDevice, name: string, segments: V3[], rgb: [number, number, number], alpha = 1): Entity {
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
  const entity = new Entity(name);
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

export async function build(host: HTMLDivElement, data: Scene, cloud: Cloud): Promise<Built> {
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  canvas.style.touchAction = 'none';
  host.appendChild(canvas);
  // No multisampling: PlayCanvas draws splats several times slower into a multisampled target.
  const device = await createGraphicsDevice(canvas, { deviceTypes: ['webgpu', 'webgl2'], antialias: false });
  device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);
  const options = new AppOptions();
  options.graphicsDevice = device;
  options.componentSystems = [RenderComponentSystem, CameraComponentSystem, GSplatComponentSystem];
  options.resourceHandlers = [GSplatHandler, TextureHandler];
  const app = new AppBase(canvas);
  app.init(options);
  app.setCanvasFillMode(FILLMODE_NONE);
  app.setCanvasResolution(RESOLUTION_AUTO);
  app.autoRender = false;
  app.start();

  const floorPlane = data.floor ?? [0, 0, 1, 0];
  const [nx = 0, ny = 0, nz = 1] = floorPlane;
  const upLength = Math.hypot(nx, ny, nz) || 1;
  const up: V3 = [nx / upLength, ny / upLength, nz / upLength];

  // The points. Colour by height: each point's height above the floor on a ramp over the middle 98% of heights.
  const { count, positions } = cloud;
  const heights = new Float32Array(count);
  for (let i = 0; i < count; i++)
    heights[i] = heightOf([positions[i * 3] ?? 0, positions[i * 3 + 1] ?? 0, positions[i * 3 + 2] ?? 0], floorPlane);
  const [low, high] = robustRange(heights);
  const ramp = new Uint8Array(count * 3);
  for (let i = 0; i < count; i++)
    ramp.set(
      heightColour(heights[i] ?? 0, low, high).map(v => Math.round(v * 255)),
      i * 3,
    );
  const colours = {
    height: perCorner(new Uint8Array(count * 16), ramp, 3, 255),
    rgb: cloud.colours ? perCorner(new Uint8Array(count * 16), cloud.colours, 3, 255) : null,
  };
  const pointMaterial = new ShaderMaterial({
    uniqueName: 'PlacePoints',
    vertexGLSL: POINT_GLSL.vertex,
    fragmentGLSL: POINT_GLSL.fragment,
    vertexWGSL: POINT_WGSL.vertex,
    fragmentWGSL: POINT_WGSL.fragment,
    attributes: { vertex_position: SEMANTIC_POSITION, vertex_color: SEMANTIC_COLOR, vertex_corner: SEMANTIC_TEXCOORD0 },
  });
  pointMaterial.cull = CULLFACE_NONE;
  pointMaterial.setParameter('point_size', 2.5);
  pointMaterial.setParameter('view_size', [1, 1]);
  pointMaterial.update();
  const pointMesh = new Mesh(device);
  pointMesh.setPositions(perCorner(new Float32Array(count * 12), positions, 3));
  const corners = new Float32Array(count * 8);
  for (let i = 0; i < count; i++) corners.set(CORNERS, i * 8);
  pointMesh.setVertexStream(SEMANTIC_TEXCOORD0, corners, 2);
  const indices = new Uint32Array(count * 6);
  for (let i = 0; i < count; i++) indices.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], i * 6);
  pointMesh.setIndices(indices);
  let coloured: Colour | null = null;
  const colourBy = (want: Colour) => {
    const by = want === 'rgb' && colours.rgb ? 'rgb' : 'height';
    if (by === coloured) return;
    coloured = by;
    pointMesh.setVertexStream(SEMANTIC_COLOR, colours[by] ?? colours.height, 4, count * 4, TYPE_UINT8, true);
    pointMesh.update(PRIMITIVE_TRIANGLES);
  };
  colourBy('rgb');
  const points = new Entity('points');
  points.addComponent('render', { meshInstances: [new MeshInstance(pointMesh, pointMaterial)], castShadows: false });
  app.root.addChild(points);

  // What home frames: everything drawn but the grid (the cloud, the plan, and the cameras).
  const lo: V3 = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const hi: V3 = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const grow = (p: readonly number[]) => {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k] ?? 0, p[k] ?? 0);
      hi[k] = Math.max(hi[k] ?? 0, p[k] ?? 0);
    }
  };
  for (let i = 0; i < count; i++) grow([positions[i * 3] ?? 0, positions[i * 3 + 1] ?? 0, positions[i * 3 + 2] ?? 0]);

  // Each camera's view: lines from its centre to the four corners of its image, and the image's edge.
  const accent = cssColour(host, '--mantine-primary-color-filled', '#228be6');
  const views: V3[] = [];
  for (const cam of Object.values(data.cameras ?? {})) {
    const [o, a, b, c, d] = frustum(cam.pose, cam.focal, cam.width, cam.height) as [V3, V3, V3, V3, V3];
    views.push(o, a, o, b, o, c, o, d, a, b, b, c, c, d, d, a);
  }
  views.forEach(grow);
  const cameras = lines(device, 'cameras', views, accent);
  app.root.addChild(cameras);

  // The plan's rooms, outlined on the floor.
  const outline: V3[] = [];
  if (data.plan && data.plan_transform) {
    try {
      for (const ring of planOutline(data.plan, data.plan_transform))
        ring.forEach((p, i) => {
          outline.push(p, ring[(i + 1) % ring.length] as V3);
        });
    } catch {
      // a plan that doesn't read is left off; the cloud still draws
    }
  }
  outline.forEach(grow);
  const plan = lines(device, 'plan', outline, [0xe8 / 255, 0x59 / 255, 0x0c / 255]);
  app.root.addChild(plan);

  const empty = !Number.isFinite(lo[0]);
  const centre: V3 = empty ? [0, 0, 0] : [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const radius = empty ? 5 : Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;

  // The floor: a grid on the floor plane under all of it, through the point of it nearest the middle.
  const drop = heightOf(centre, floorPlane);
  const onFloor: V3 = [centre[0] - up[0] * drop, centre[1] - up[1] * drop, centre[2] - up[2] * drop];
  const reach = Math.max(4, Math.ceil(radius * 2.2)) / 2;
  const seed: V3 = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const along = new Vec3().cross(new Vec3(...up), new Vec3(...seed)).normalize();
  const across = new Vec3().cross(new Vec3(...up), along).normalize();
  const gridLines: V3[] = [];
  const onGrid = (s: number, t: number): V3 => [
    onFloor[0] + along.x * s + across.x * t,
    onFloor[1] + along.y * s + across.y * t,
    onFloor[2] + along.z * s + across.z * t,
  ];
  for (let i = 0; i <= 24; i++) {
    const t = -reach + (2 * reach * i) / 24;
    gridLines.push(onGrid(t, -reach), onGrid(t, reach), onGrid(-reach, t), onGrid(reach, t));
  }
  const floor = lines(device, 'floor', gridLines, [0.6, 0.6, 0.6], 0.45);
  app.root.addChild(floor);

  const FOV = 55;
  const eyeEntity = new Entity('eye');
  const [r, g, b] = cssColour(host, '--lib-card', '#ffffff');
  eyeEntity.addComponent('camera', { fov: FOV, nearClip: 0.02, farClip: 500, clearColor: new Color(r, g, b) });
  app.root.addChild(eyeEntity);
  const lens = eyeEntity.camera;
  if (!lens) throw new Error("The 3D view couldn't make its camera.");

  // The orbit: an eye and the point it looks at and turns about.
  let eye: V3 = [centre[0], centre[1], centre[2] + radius * 2.5];
  let target: V3 = centre;
  // Looking from a camera, the view is filled with its frame (`coverFov`: cropped to the view's shape, never past
  // its edges, where a splat is only smear) until the view goes home. Null: the view's own field of view.
  let frame: { width: number; height: number; focal: number } | null = null;
  let gone = false;
  const aspect = () => Math.max(1, canvas.clientWidth) / Math.max(1, canvas.clientHeight);
  const render = () => {
    if (gone) return;
    lens.fov = frame ? coverFov(frame.width, frame.height, frame.focal, aspect()) : FOV;
    eyeEntity.setPosition(...eye);
    eyeEntity.lookAt(new Vec3(...target), new Vec3(...up));
    app.renderNextFrame = true;
  };
  // A splat's order is worked out off the frame that asked for it, so that frame is followed by another.
  app.scene.on('gsplat:sorted', render);
  // A few frames in a row: a splat that has just arrived, or a jump to a new view, settles over several.
  const settle = (frames = 12) => {
    if (gone) return;
    render();
    if (frames > 1) requestAnimationFrame(() => settle(frames - 1));
  };

  const resize = () => {
    const w = host.clientWidth;
    const h = host.clientHeight;
    app.resizeCanvas(w, h);
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    pointMaterial.setParameter('view_size', [Math.max(1, w), Math.max(1, h)]);
    render();
  };

  // Home: looking down at it from above and to one side, with everything drawn just in view. A sphere around a flat
  // room seen at a slant wastes most of the frame, so the box's corners are projected and the distance scaled until
  // the widest of them reaches the edge (a few steps settle it).
  const boxCorners = [0, 1, 2, 3, 4, 5, 6, 7].map(
    i => new Vec4(i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2], 1),
  );
  const home = () => {
    const side = new Vec3(1, 0, 0);
    const upV = new Vec3(...up);
    side.sub(upV.clone().mulScalar(side.dot(upV))).normalize();
    const from = upV.clone().mulScalar(0.8).add(side.mulScalar(0.6)).normalize();
    const middle = new Vec3(...centre);
    const at = (distance: number) => middle.clone().add(from.clone().mulScalar(distance));
    let distance = radius * 2.5;
    const projection = new Mat4().setPerspective(FOV, aspect(), 0.02, 500);
    for (let step = 0; step < 4 && !empty; step++) {
      const view = new Mat4().setLookAt(at(distance), middle, upV).invert();
      const both = new Mat4().mul2(projection, view);
      const widest = Math.max(
        ...boxCorners.map(c => {
          const p = both.transformVec4(c, new Vec4());
          return Math.max(Math.abs(p.x / p.w), Math.abs(p.y / p.w));
        }),
      );
      distance *= Math.min(4, Math.max(0.25, widest / 0.92));
    }
    const p = at(distance);
    eye = [p.x, p.y, p.z];
    target = centre;
    frame = null;
    settle();
  };
  const lookFrom = (name: string) => {
    const cam = data.cameras?.[name];
    if (!cam) return;
    // Camera to map, row by row; the camera looks along its +z (OpenCV), so its centre and forward are columns 3 and 2.
    const p = cam.pose as number[];
    const forward = new Vec3(p[2], p[6], p[10]).normalize();
    // Its own frame, filling the view at whatever shape the view has (`render`).
    frame = { width: cam.width, height: cam.height, focal: cam.focal };
    eye = [p[3] ?? 0, p[7] ?? 0, p[11] ?? 0];
    // Orbit about a point just ahead, so a small drag looks around from where the camera stands.
    target = [eye[0] + forward.x * 0.5, eye[1] + forward.y * 0.5, eye[2] + forward.z * 0.5];
    settle();
  };

  // The pointer: drag to turn, shift-drag or the right button to slide, the wheel or a pinch to move in and out.
  const held = new Map<number, { x: number; y: number }>();
  const down = (e: PointerEvent) => {
    canvas.setPointerCapture(e.pointerId);
    held.set(e.pointerId, { x: e.clientX, y: e.clientY });
  };
  const move = (e: PointerEvent) => {
    const was = held.get(e.pointerId);
    if (!was) return;
    const dx = e.clientX - was.x;
    const dy = e.clientY - was.y;
    if (held.size === 2) {
      // a pinch: the change in the two fingers' distance moves the eye in or out
      const other = [...held.entries()].find(([id]) => id !== e.pointerId)?.[1];
      if (other) {
        const before = Math.hypot(was.x - other.x, was.y - other.y);
        const after = Math.hypot(e.clientX - other.x, e.clientY - other.y);
        if (before > 0 && after > 0) eye = dolly(eye, target, before / after);
      }
    } else if (e.shiftKey || e.buttons & 2 || e.buttons & 4) {
      const span = Math.hypot(eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]);
      const perPixel = (2 * span * Math.tan((lens.fov * Math.PI) / 360)) / Math.max(1, canvas.clientHeight);
      ({ eye, target } = slide(eye, target, up, dx * perPixel, dy * perPixel));
    } else {
      const k = (2 * Math.PI) / Math.max(1, canvas.clientHeight);
      eye = turn(eye, target, up, -dx * k, -dy * k);
    }
    held.set(e.pointerId, { x: e.clientX, y: e.clientY });
    render();
  };
  const release = (e: PointerEvent) => {
    held.delete(e.pointerId);
  };
  const wheel = (e: WheelEvent) => {
    e.preventDefault();
    eye = dolly(eye, target, Math.exp(e.deltaY * 0.001));
    render();
  };
  const menu = (e: Event) => e.preventDefault();
  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('wheel', wheel, { passive: false });
  canvas.addEventListener('contextmenu', menu);

  let splat: { entity: Entity; asset: Asset } | null = null;
  let wantSplat = true;
  const dropSplat = () => {
    if (!splat) return;
    splat.entity.destroy();
    app.assets.remove(splat.asset);
    splat.asset.unload();
    splat = null;
  };
  const setSplat = (bytes: ArrayBuffer) =>
    new Promise<void>((resolve, reject) => {
      dropSplat();
      // The file's name is what tells PlayCanvas it's a SOG; its bytes are already here.
      const asset = new Asset('splat', 'gsplat', { url: 'splat.sog', contents: bytes } as unknown as { url: string });
      asset.on('error', (err: unknown) => reject(new Error(String(err))));
      asset.ready(() => {
        if (gone) return resolve();
        const entity = new Entity('splat');
        entity.addComponent('gsplat', { asset });
        entity.enabled = wantSplat;
        app.root.addChild(entity);
        splat = { entity, asset };
        settle();
        resolve();
      });
      app.assets.add(asset);
      app.assets.load(asset);
    });

  return {
    home,
    lookFrom,
    resize,
    render,
    show: s => {
      wantSplat = s.splat;
      const showSplat = s.splat && !!splat;
      points.enabled = !showSplat;
      if (splat) splat.entity.enabled = showSplat;
      pointMaterial.setParameter('point_size', s.size);
      colourBy(s.colour);
      floor.enabled = s.floor;
      cameras.enabled = s.cameras;
      plan.enabled = s.plan;
      render();
    },
    setSplat,
    dropSplat,
    cameraNames: Object.keys(data.cameras ?? {}),
    backend: device.isWebGPU ? 'webgpu' : 'webgl2',
    dispose: () => {
      gone = true;
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', release);
      canvas.removeEventListener('pointercancel', release);
      canvas.removeEventListener('wheel', wheel);
      canvas.removeEventListener('contextmenu', menu);
      dropSplat();
      app.destroy();
      canvas.remove();
    },
  };
}
