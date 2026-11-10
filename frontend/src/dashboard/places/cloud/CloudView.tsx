// A place's map in 3D: the point cloud the house built of it, with the floor, the cameras' views, and the plan's
// outline on the floor, in an orbit view. PlayCanvas (MIT) draws it (scene.ts), on WebGPU or WebGL2; this chunk
// loads only when the 3D view opens (PlacesTab lazy-loads it), since the engine is most of its weight. Drawing is
// on demand: a frame is drawn when the view moves or a setting changes, never in a loop.
// A revision with a splat can be drawn as the splat instead of the points, in the same scene. A splat looks right
// only from where its frames were taken, so each camera has a button that puts the view there, filled with that
// camera's frame.
import { Button, Checkbox, Group, SegmentedControl, Slider, Text } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useRef, useState } from 'react';
import { messageOf } from '../../../shared/http';
import { type Scene, sceneQuery } from '../api';
import classes from '../places.module.css';
import { canDraw, canDrawSplat, openingCamera, parseCloud } from './load';
import { type Built, build, type Colour } from './scene';

type Draw = 'points' | 'splat';

/** Whether this browser can draw the view at all: WebGL2 (not jsdom, not a GPU-less fallback). A browser with
 *  WebGPU has WebGL2 too. */
function hasWebGL(): boolean {
  try {
    return !!document.createElement('canvas').getContext('webgl2');
  } catch {
    return false;
  }
}

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
  const [ready, setReady] = useState(0);
  useEffect(() => {
    const el = host.current;
    if (!webgl || !el || !data || !bytes.data) return;
    let gone = false;
    let made: Built | null = null;
    let observer: ResizeObserver | null = null;
    (async () => build(el, data, parseCloud(bytes.data, 'ply')))().then(
      b => {
        if (gone) return b.dispose();
        made = b;
        built.current = b;
        observer = new ResizeObserver(b.resize);
        observer.observe(el);
        b.resize();
        b.home();
        setReady(n => n + 1); // the splat and the settings wait for the scene
      },
      e => {
        if (!gone) setFailed(messageOf(e));
      },
    );
    return () => {
      gone = true;
      observer?.disconnect();
      built.current = null;
      setSplatReady(false);
      made?.dispose();
    };
  }, [webgl, data, bytes.data]);

  // The splat joins the scene once its bytes are here (and the scene is built), and leaves it with them.
  // biome-ignore lint/correctness/useExhaustiveDependencies: `ready` says the scene was rebuilt, which the splat joins again
  useEffect(() => {
    const b = built.current;
    const file = splatBytes.data;
    if (!b || !file) return;
    let gone = false;
    b.setSplat(file).then(
      () => {
        if (gone) return;
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
      b.dropSplat();
      setSplatReady(false);
    };
  }, [splatBytes.data, ready, splatFrom]);

  // Settings change what's drawn, then one frame is drawn.
  // biome-ignore lint/correctness/useExhaustiveDependencies: `ready` says the scene was rebuilt and wants its settings
  useEffect(() => {
    built.current?.show({ splat: draw === 'splat' && splatReady, size, colour, ...show });
  }, [draw, splatReady, size, colour, show, ready]);

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
