// A place's map in 3D: the point cloud the house built of it, with the floor, the fixed cameras' views, and
// the plan's outline on the floor, in an orbit view. three.js (MIT) draws it; this chunk loads only when the 3D view
// opens (PlacesTab lazy-loads it), since three.js is most of its weight. Drawing is on demand: a frame is rendered
// when the camera moves or a setting changes, never in a loop, so an open view costs nothing while it's still.
// A revision with a splat can be drawn as the splat instead of the points: Spark draws it in the same scene,
// and asks for a frame (onDirty) when its sort or decode finishes. A splat of a few fixed cameras looks right only
// from where they stand, so each camera has a button that puts the view there, filled with that camera's frame.
import { Button, Checkbox, Group, SegmentedControl, Slider, Text } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { messageOf } from '../../../shared/http';
import { type Scene, sceneQuery } from '../api';
import classes from '../places.module.css';
import { coverFov, frustum, heightColour, heightOf, planOutline, robustRange, type V3 } from './geometry';
import { canDraw, canDrawSplat, type LoadedSplat, loadSplat, openingCamera, parseCloud } from './load';

type Colour = 'rgb' | 'height';
type Draw = 'points' | 'splat';

/** Whether this browser can draw WebGL at all (not jsdom, not a GPU-less fallback). */
function hasWebGL(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') ?? c.getContext('webgl'));
  } catch {
    return false;
  }
}

const css = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

/** The map's bytes, read once per revision (the server lets them be cached for good). */
function useMapBytes(scene: Scene | undefined) {
  return useQuery({
    queryKey: ['places', 'map', scene?.place, scene?.revision],
    queryFn: async ({ signal }) => {
      const r = await fetch(scene?.map ?? '', { signal });
      if (!r.ok) throw new Error(`The map couldn't be read (${r.status}).`);
      return r.arrayBuffer();
    },
    enabled: !!scene?.map && canDraw(scene.map_format),
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 10 * 60_000,
  });
}

/** The splat's bytes, read once per revision, only once the view is asked to draw it. */
function useSplatBytes(scene: Scene | undefined, wanted: boolean) {
  return useQuery({
    queryKey: ['places', 'splat', scene?.place, scene?.revision],
    queryFn: async ({ signal }) => {
      const r = await fetch(scene?.splat ?? '', { signal });
      if (!r.ok) throw new Error(`The splat couldn't be read (${r.status}).`);
      return r.arrayBuffer();
    },
    enabled: wanted && !!scene?.splat && canDrawSplat(scene.splat_format),
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 10 * 60_000,
  });
}

/** Everything the scene draws, built once per map and kept between renders. */
interface Built {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  points: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;
  colours: { rgb: THREE.BufferAttribute | null; height: THREE.BufferAttribute };
  floor: THREE.Object3D;
  cameras: THREE.Object3D;
  plan: THREE.Object3D;
  home: () => void;
  /** Put the view where a fixed camera stands, looking where it looks, with its field of view. */
  lookFrom: (name: string) => void;
  render: () => void;
  /** The splat, once it's in the scene. */
  splat: LoadedSplat | null;
  /** The fixed cameras drawn, by name. */
  cameraNames: string[];
}

function build(host: HTMLDivElement, data: Scene, bytes: ArrayBuffer): Built {
  const geometry = parseCloud(bytes, 'ply');
  const pos = geometry.getAttribute('position') as THREE.BufferAttribute;
  const floorPlane = data.floor ?? [0, 0, 1, 0];
  const up = new THREE.Vector3(floorPlane[0], floorPlane[1], floorPlane[2]).normalize();

  // Colour by height: each point's height above the floor on a ramp over the middle 98% of heights.
  const heights = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i++) heights[i] = heightOf([pos.getX(i), pos.getY(i), pos.getZ(i)], floorPlane);
  const [low, high] = robustRange(heights);
  const ramp = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) ramp.set(heightColour(heights[i] ?? 0, low, high), i * 3);
  const height = new THREE.BufferAttribute(ramp, 3);
  const rgb = (geometry.getAttribute('color') as THREE.BufferAttribute | undefined) ?? null;
  geometry.setAttribute('color', rgb ?? height);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(css('--lib-card', '#ffffff'));

  const material = new THREE.PointsMaterial({ size: 2.5, sizeAttenuation: false, vertexColors: true });
  const points = new THREE.Points(geometry, material);
  scene.add(points);

  geometry.computeBoundingSphere();
  const sphere = geometry.boundingSphere ?? new THREE.Sphere(new THREE.Vector3(), 5);

  // Each fixed camera's view: lines from its centre to the four corners of its image, and the image's edge.
  const cameras = new THREE.Group();
  const accent = new THREE.Color(css('--mantine-primary-color-filled', '#228be6'));
  for (const cam of Object.values(data.cameras ?? {})) {
    const [o, a, b, c, d] = frustum(cam.pose, cam.focal, cam.width, cam.height);
    const lines = [o, a, o, b, o, c, o, d, a, b, b, c, c, d, d, a] as V3[];
    const g = new THREE.BufferGeometry().setFromPoints(lines.map(p => new THREE.Vector3(...p)));
    cameras.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: accent })));
  }
  scene.add(cameras);

  // The plan's rooms, outlined on the floor.
  const plan = new THREE.Group();
  if (data.plan && data.plan_transform) {
    try {
      for (const ring of planOutline(data.plan, data.plan_transform)) {
        const g = new THREE.BufferGeometry().setFromPoints(ring.map(p => new THREE.Vector3(...p)));
        plan.add(new THREE.LineLoop(g, new THREE.LineBasicMaterial({ color: 0xe8590c })));
      }
    } catch {
      // a plan that doesn't read is left off; the cloud still draws
    }
  }
  scene.add(plan);

  // What home frames: everything drawn but the grid (the cloud, the plan, and the cameras).
  const drawn = new THREE.Box3();
  for (const o of [points, plan, cameras]) drawn.expandByObject(o);
  const bounds = drawn.isEmpty() ? sphere : drawn.getBoundingSphere(new THREE.Sphere());

  // The floor: a grid on the floor plane under all of it, through the point of it nearest the middle.
  const onFloor = bounds.center.clone().addScaledVector(up, -heightOf(bounds.center.toArray() as V3, floorPlane));
  const grid = new THREE.GridHelper(Math.max(4, Math.ceil(bounds.radius * 2.2)), 24, 0x888888, 0xbbbbbb);
  grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);
  grid.position.copy(onFloor);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.45;
  scene.add(grid);

  const FOV = 55;
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.02, 500);
  camera.up.copy(up);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  // Looking from a fixed camera, the view is filled with its frame (`coverFov`: cropped to the view's shape, never
  // past its edges, where a splat is only smear) until the view goes home. Null: the view's own field of view.
  let frame: { width: number; height: number; focal: number } | null = null;
  const size = new THREE.Vector2();
  const render = () => {
    renderer.getSize(size);
    camera.aspect = size.x / Math.max(1, size.y);
    if (frame) camera.fov = coverFov(frame.width, frame.height, frame.focal, camera.aspect);
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  };
  controls.addEventListener('change', render);
  // Home: looking down at it from above and to one side, with everything drawn just in view. A sphere around a flat
  // room seen at a slant wastes most of the frame, so the box's corners are projected and the distance scaled until
  // the widest of them reaches the edge (a few steps settle it).
  const corners = [0, 1, 2, 3, 4, 5, 6, 7].map(
    i =>
      new THREE.Vector3(
        i & 1 ? drawn.max.x : drawn.min.x,
        i & 2 ? drawn.max.y : drawn.min.y,
        i & 4 ? drawn.max.z : drawn.min.z,
      ),
  );
  const home = () => {
    const side = new THREE.Vector3(1, 0, 0).projectOnPlane(up).normalize();
    const from = up.clone().multiplyScalar(0.8).addScaledVector(side, 0.6).normalize();
    let distance = bounds.radius * 2.5;
    for (let step = 0; step < 4 && !drawn.isEmpty(); step++) {
      camera.position.copy(bounds.center).addScaledVector(from, distance);
      camera.lookAt(bounds.center);
      camera.updateMatrixWorld();
      const reach = Math.max(
        ...corners.map(c => {
          const p = c.clone().project(camera);
          return Math.max(Math.abs(p.x), Math.abs(p.y));
        }),
      );
      distance *= Math.min(4, Math.max(0.25, reach / 0.92));
    }
    camera.position.copy(bounds.center).addScaledVector(from, distance);
    controls.target.copy(bounds.center);
    camera.fov = FOV;
    frame = null;
    controls.update();
    render();
  };
  const lookFrom = (name: string) => {
    const cam = data.cameras?.[name];
    if (!cam) return;
    // Camera to map, row by row; the camera looks along its +z (OpenCV), so its centre and forward are columns 3 and 2.
    const p = cam.pose;
    const centre = new THREE.Vector3(p[3], p[7], p[11]);
    const forward = new THREE.Vector3(p[2], p[6], p[10]).normalize();
    // Its own frame, filling the view at whatever shape the view has (`render`).
    frame = { width: cam.width, height: cam.height, focal: cam.focal };
    camera.position.copy(centre);
    // Orbit about a point just ahead, so a small drag looks around from where the camera stands.
    controls.target.copy(centre).addScaledVector(forward, 0.5);
    controls.update();
    render();
  };
  return {
    renderer,
    scene,
    camera,
    controls,
    points,
    colours: { rgb, height },
    floor: grid,
    cameras,
    plan,
    home,
    lookFrom,
    render,
    splat: null,
    cameraNames: Object.keys(data.cameras ?? {}),
  };
}

/** Take the splat out of the scene and free it: when its bytes change, or before the scene itself goes. */
function dropSplat(b: Built) {
  const loaded = b.splat;
  b.splat = null;
  if (!loaded) return;
  b.scene.remove(loaded.spark, loaded.mesh);
  loaded.mesh.dispose();
  loaded.spark.dispose();
}

function dispose(b: Built) {
  b.controls.dispose();
  b.scene.traverse(o => {
    const any = o as THREE.Mesh;
    any.geometry?.dispose?.();
    const m = any.material as THREE.Material | THREE.Material[] | undefined;
    for (const x of Array.isArray(m) ? m : m ? [m] : []) x.dispose();
  });
  b.renderer.dispose();
  b.renderer.domElement.remove();
}

export default function CloudView({ place }: { place: string }) {
  const webgl = useMemo(hasWebGL, []);
  const scene = useQuery({ ...sceneQuery(place), enabled: webgl });
  const data = scene.data;
  const bytes = useMapBytes(data);
  const hasSplat = !!data?.splat && canDrawSplat(data.splat_format);
  const [chosen, setChosen] = useState<Draw | null>(null);
  const [splatFailed, setSplatFailed] = useState<string | null>(null);
  const wantSplat = (chosen ?? (hasSplat ? 'splat' : 'points')) === 'splat';
  const splatBytes = useSplatBytes(data, wantSplat);
  const splatFrom = data?.splat_from;
  // A splat that can't be read or drawn leaves the points: the map still shows, with a note.
  const splatError = splatFailed ?? (splatBytes.isError ? messageOf(splatBytes.error) : null);
  const draw: Draw = wantSplat && !splatError ? 'splat' : 'points';
  const [splatReady, setSplatReady] = useState(false);
  const host = useRef<HTMLDivElement>(null);
  const built = useRef<Built | null>(null);
  const [colour, setColour] = useState<Colour>('rgb');
  const [size, setSize] = useState(2.5);
  const [show, setShow] = useState({ floor: true, cameras: true, plan: true });
  const [failed, setFailed] = useState<string | null>(null);

  // Build the scene once its map's bytes are here; tear it down when they change or the view closes.
  useEffect(() => {
    const el = host.current;
    if (!webgl || !el || !data || !bytes.data) return;
    let b: Built;
    try {
      b = build(el, data, bytes.data);
    } catch (e) {
      setFailed(messageOf(e));
      return;
    }
    built.current = b;
    const resize = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      b.renderer.setSize(w, h, false);
      b.renderer.domElement.style.width = '100%';
      b.renderer.domElement.style.height = '100%';
      b.render(); // sets the camera's aspect, and its field of view when it looks from a camera
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();
    b.home();
    return () => {
      observer.disconnect();
      built.current = null;
      dropSplat(b);
      setSplatReady(false);
      dispose(b);
    };
  }, [webgl, data, bytes.data]);

  // The splat joins the scene once its bytes are here (and the scene is built), and leaves it with them.
  // biome-ignore lint/correctness/useExhaustiveDependencies: bytes.data and data rebuild the scene, which the splat joins again
  useEffect(() => {
    const b = built.current;
    const data = splatBytes.data;
    if (!b || !data) return;
    let gone = false;
    let frame = 0;
    // Spark may ask for a frame several times a frame; one is drawn.
    const dirty = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        if (!gone) b.render();
      });
    };
    loadSplat(data, b.renderer, dirty).then(
      loaded => {
        if (gone) {
          loaded.mesh.dispose();
          loaded.spark.dispose();
          return;
        }
        b.scene.add(loaded.spark, loaded.mesh);
        b.splat = loaded;
        setSplatReady(true);
        // A splat looks right from where its frames were taken, and like floaters from above: it opens from the
        // first of the views it trained on, a walk's or a fixed camera's ("Back to the start" goes to the overview).
        const first = openingCamera(b.cameraNames, splatFrom);
        if (first) b.lookFrom(first);
      },
      e => {
        if (!gone) setSplatFailed(messageOf(e));
      },
    );
    return () => {
      gone = true;
      cancelAnimationFrame(frame);
      dropSplat(b);
      setSplatReady(false);
    };
  }, [splatBytes.data, bytes.data, data, splatFrom]);

  // Settings change what's drawn, then one frame is drawn.
  useEffect(() => {
    const b = built.current;
    if (!b) return;
    const showSplat = draw === 'splat' && splatReady;
    b.points.visible = !showSplat;
    if (b.splat) b.splat.mesh.visible = showSplat;
    b.points.material.size = size;
    const want = colour === 'rgb' ? (b.colours.rgb ?? b.colours.height) : b.colours.height;
    if (b.points.geometry.getAttribute('color') !== want) b.points.geometry.setAttribute('color', want);
    b.floor.visible = show.floor;
    b.cameras.visible = show.cameras;
    b.plan.visible = show.plan;
    b.render();
  });

  if (!webgl)
    return (
      <Text size="sm" c="dimmed">
        This browser can't draw 3D here (no WebGL), so the map can only be shown as the plan.
      </Text>
    );
  if (scene.isError || bytes.isError || failed)
    return (
      <Text size="sm" c="red">
        Couldn't show the map: {failed ?? messageOf(scene.error ?? bytes.error)}
      </Text>
    );
  if (data && !data.revision)
    return (
      <Text size="sm" c="dimmed">
        No 3D map of this place yet. It appears once the house has built one.
      </Text>
    );
  if (data && !canDraw(data.map_format))
    return (
      <Text size="sm" c="dimmed">
        This map is in a format the view can't draw yet ({data.map_format}).
      </Text>
    );
  const loading = !data || bytes.isPending;
  const splatLoading = draw === 'splat' && !splatReady;
  const cams = Object.keys(data?.cameras ?? {});
  return (
    <div>
      <div
        className={classes.cloud}
        ref={host}
        aria-label="The place's map in 3D: drag to turn, scroll to zoom"
        role="img"
      >
        {loading ? (
          <Text size="sm" c="dimmed" className={classes.cloudNote}>
            {data?.map_bytes ? `Loading the map (${(data.map_bytes / 1e6).toFixed(1)} MB)…` : 'Loading the map…'}
          </Text>
        ) : (
          splatLoading && (
            <Text size="sm" c="dimmed" className={classes.cloudNote}>
              {data?.splat_bytes
                ? `Loading the splat (${(data.splat_bytes / 1e6).toFixed(1)} MB)…`
                : 'Loading the splat…'}
            </Text>
          )
        )}
      </div>
      <Group justify="space-between" gap="sm" mt="sm" wrap="wrap">
        {hasSplat && !splatError && (
          <SegmentedControl
            size="xs"
            value={draw}
            onChange={v => setChosen(v as Draw)}
            data={[
              { value: 'splat', label: 'Splat' },
              { value: 'points', label: 'Points' },
            ]}
            aria-label="Draw the map as"
          />
        )}
        {draw === 'points' && (
          <>
            <SegmentedControl
              size="xs"
              value={colour}
              onChange={v => setColour(v as Colour)}
              data={[
                { value: 'rgb', label: 'Colour' },
                { value: 'height', label: 'Height' },
              ]}
              aria-label="Colour the points by"
            />
            <Group gap="xs" align="center" wrap="nowrap">
              <Text size="xs" c="dimmed">
                Point size
              </Text>
              <Slider value={size} onChange={setSize} min={1} max={8} step={0.5} w={120} thumbLabel="Point size" />
            </Group>
          </>
        )}
        <Group gap="md">
          <Checkbox
            size="xs"
            label="Floor"
            checked={show.floor}
            onChange={e => setShow(s => ({ ...s, floor: e.currentTarget.checked }))}
          />
          <Checkbox
            size="xs"
            label={`Cameras${cams.length ? ` (${cams.length})` : ''}`}
            checked={show.cameras}
            onChange={e => setShow(s => ({ ...s, cameras: e.currentTarget.checked }))}
          />
          <Checkbox
            size="xs"
            label="Plan"
            checked={show.plan}
            onChange={e => setShow(s => ({ ...s, plan: e.currentTarget.checked }))}
          />
        </Group>
        <Text
          size="xs"
          c="dimmed"
          component="button"
          type="button"
          className={classes.cloudHome}
          onClick={() => built.current?.home()}
        >
          Back to the start
        </Text>
      </Group>
      {!hasSplat && data?.splat_stale && (
        <Text size="xs" c="dimmed" mt="xs">
          The splat from{' '}
          {new Date(data.splat_stale.at * 1000).toLocaleString(undefined, {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
          })}{' '}
          is out of date: the map has moved since. A new one comes with the next nightly run.
        </Text>
      )}
      {splatError && (
        <Text size="xs" c="red" mt="xs">
          Couldn't show the splat, so the points are drawn: {splatError}
        </Text>
      )}
      {cams.length > 0 && (
        <Group gap="xs" mt="xs" wrap="wrap">
          <Text size="xs" c="dimmed">
            Look from
          </Text>
          {cams.map(name => (
            <Button key={name} size="compact-xs" variant="light" onClick={() => built.current?.lookFrom(name)}>
              {name}
            </Button>
          ))}
        </Group>
      )}
    </div>
  );
}
