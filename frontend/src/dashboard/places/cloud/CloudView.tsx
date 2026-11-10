// A place's map in 3D: the point cloud the house built of it, with the floor, the cameras' views, and the plan's
// outline on the floor, and its splat when it has one. PlayCanvas's SuperSplat viewer draws it and moves through
// it (scene.ts): orbit, fly, and walk modes, with its own mode bar, help, settings, and fullscreen. This chunk
// loads only when the 3D view opens (PlacesTab lazy-loads it), since the viewer and its engine are most of its
// weight. The view fills its pane down to the bottom of the window, with our own settings in a row under it.
// A splat looks right only from where its frames were taken, so each camera has a button that flies the view
// there.
import { Button, Checkbox, Group, SegmentedControl, Slider, Text } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
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
  const wantSplat = (chosen ?? (hasSplat ? 'splat' : 'points')) === 'splat';
  // The viewer is built around its splat, so a place that has one waits for it, whichever way it's drawn first.
  const splatBytes = useSplatBytes(data, hasSplat);
  // A splat that can't be read or drawn leaves the points: the map still shows, with a note.
  const splatError = splatBytes.isError ? messageOf(splatBytes.error) : null;
  const draw: Draw = wantSplat && !splatError ? 'splat' : 'points';
  const [splatReady, setSplatReady] = useState(false);
  const host = useRef<HTMLDivElement>(null);
  const built = useRef<Built | null>(null);
  const [colour, setColour] = useState<Colour>('rgb');
  const [size, setSize] = useState(2.5);
  const [show, setShow] = useState({ floor: true, cameras: true, plan: true });
  const [failed, setFailed] = useState<string | null>(null);

  // Build the scene once its map's bytes are here, and its splat's when it has one (or they've failed); tear it
  // down when they change or the view closes.
  const [ready, setReady] = useState(0);
  const splatSettled = !hasSplat || !!splatBytes.data || splatBytes.isError;
  useEffect(() => {
    const el = host.current;
    if (!webgl || !el || !data || !bytes.data || !splatSettled) return;
    let gone = false;
    let made: Built | null = null;
    const file = splatBytes.data ?? null;
    // A splat looks right from where its frames were taken, and like floaters from above: it opens from the first of
    // the views it trained on ("Back to the start" goes to the overview).
    const opening = file ? openingCamera(Object.keys(data.cameras ?? {})) : null;
    (async () => build(el, data, parseCloud(bytes.data, 'ply'), file, opening))().then(
      b => {
        if (gone) return b.dispose();
        made = b;
        built.current = b;
        setSplatReady(!!file);
        setReady(n => n + 1); // the settings wait for the scene
      },
      e => {
        if (!gone) setFailed(messageOf(e));
      },
    );
    return () => {
      gone = true;
      built.current = null;
      setSplatReady(false);
      made?.dispose();
    };
  }, [webgl, data, bytes.data, splatBytes.data, splatSettled]);

  // Settings change what's drawn, then one frame is drawn.
  // biome-ignore lint/correctness/useExhaustiveDependencies: `ready` says the scene was rebuilt and wants its settings
  useEffect(() => {
    built.current?.show({ splat: draw === 'splat' && splatReady, size, colour, ...show });
  }, [draw, splatReady, size, colour, show, ready]);

  // The view takes what's left of the window under where it starts: measured, since what's above it (the place's
  // header, the tabs) is the page's to decide, and measured again when the window changes.
  const frame = useRef<HTMLDivElement>(null);
  const [top, setTop] = useState(0);
  useLayoutEffect(() => {
    const measure = () => {
      const el = frame.current;
      if (el) setTop(Math.round(el.getBoundingClientRect().top + window.scrollY));
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

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
    <div
      ref={frame}
      className={classes.cloudFrame}
      style={{ height: `calc(100dvh - ${top}px - var(--mantine-spacing-md))` }}
    >
      <div
        className={classes.cloud}
        ref={host}
        aria-label="The place's map in 3D: orbit, fly, or walk through it; press H for the controls"
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
