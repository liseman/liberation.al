// A place's map in 3D: the point cloud the house built of it, with the floor, the cameras' views, and the plan's
// outline on the floor, and its splat when it has one. PlayCanvas's SuperSplat viewer draws it and moves through
// it (scene.ts): orbit, fly, and walk modes, with its own mode bar, help, settings, and fullscreen. This chunk
// loads only when the 3D view opens (PlacesTab lazy-loads it), since the viewer and its engine are most of its
// weight. The view fills its pane down to the bottom of the window, with our own settings in a row under it.
// A splat looks right only from where its frames were taken, so each camera has a button that flies the view
// there.
//
// Whatever state the place's splat is in is said under the view, never left to the console (#246): none yet and
// why, one made for an earlier map, loading with how much has come, or a load that failed with its reason and a
// Retry. The points are drawn in every one of them: the map shows as soon as its own bytes are here, and the splat
// takes over once it has drawn.
//
// A map learned from the cameras changes while the view is open (#259): the scene is asked for again every few
// seconds, and a newer map or splat is drawn once all of its bytes are here, the one before it staying drawn until
// then, from where the person was looking. A line under the view says whether learning is on, how many views the
// map was built from, and when it last improved (status.ts).
//
// Under the settings, a box for each camera (SourceFilter.tsx) shows or hides what it added, in both the points and
// the splat, at once and without loading anything again; under each, its views by when they were taken. What's
// hidden is kept for the place this session (sources.ts), and a newer map is drawn with it hidden too.
import { Button, Checkbox, Group, Progress, SegmentedControl, Slider, Text } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { messageOf } from '../../../shared/http';
import { type Scene, sceneQuery } from '../api';
import classes from '../places.module.css';
import { canDraw, canDrawSplat, openingCamera, parseCloud, parseViews, readBytes, splatFault } from './load';
import { SourceFilter } from './SourceFilter';
import { type Built, build, type Colour, type View } from './scene';
import { readHidden, visibleViews, writeHidden } from './sources';
import { learningNote, loadingText, planNote, splatNote } from './status';

type Draw = 'points' | 'splat';

/** What's drawn: a scene with its map's bytes and its splat's (null with none, or one that can't be drawn). */
interface Drawn {
  scene: Scene;
  map: ArrayBuffer;
  file: ArrayBuffer | null;
}

/** The part of a scene the viewer is built from. What the scene says about learning and the splat's state changes
 *  from one read to the next without anything to draw again, so it's left out; the same drawing is the same text. */
function drawnPart(scene: Scene): string {
  const { place, revision, map, map_format, map_bytes, splat, splat_format, splat_bytes, splat_revision } = scene;
  const { floor, plan_transform, cameras, plan } = scene;
  return JSON.stringify({
    place,
    revision,
    map,
    map_format,
    map_bytes,
    splat,
    splat_format,
    splat_bytes,
    splat_revision,
    floor,
    plan_transform,
    cameras,
    plan,
  });
}

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

/** The splat's bytes, read once per splat and try, with how much of them has come. A try after the first (Retry)
 *  goes past the browser's cache, which may be what kept bytes the view can't draw. Bytes that can't be a splat are
 *  an error here, with the reason, before the viewer is given them. */
function useSplatBytes(scene: Scene | undefined, wanted: boolean, attempt: number) {
  const [got, setGot] = useState({ received: 0, total: 0 });
  const query = useQuery({
    queryKey: ['places', 'splat', scene?.place, scene?.splat_revision ?? scene?.revision, scene?.splat_format, attempt],
    queryFn: async ({ signal }) => {
      setGot({ received: 0, total: scene?.splat_bytes ?? 0 });
      const r = await fetch(scene?.splat ?? '', { signal, cache: attempt ? 'reload' : 'default' });
      if (!r.ok) throw new Error(`the server answered ${r.status}`);
      const bytes = await readBytes(r, scene?.splat_bytes, (received, total) => setGot({ received, total }));
      const fault = splatFault(bytes);
      if (fault) throw new Error(fault);
      return bytes;
    },
    enabled: wanted && !!scene?.splat && canDrawSplat(scene.splat_format),
    retry: false, // the page says why and offers Retry
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 10 * 60_000,
  });
  return { ...query, got };
}

/** Each of the splat's Gaussians' views, read once per splat; null while there's none to read, or when the bytes
 *  can't be one. */
function useSplatViews(url: string | null | undefined, wanted: boolean) {
  return useQuery({
    queryKey: ['places', 'splat-views', url],
    queryFn: async ({ signal }) => {
      const r = await fetch(url ?? '', { signal });
      if (!r.ok) throw new Error(`the server answered ${r.status}`);
      return parseViews(await r.arrayBuffer());
    },
    enabled: wanted && !!url,
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 10 * 60_000,
  });
}

export default function CloudView({ place }: { place: string }) {
  const webgl = useMemo(hasWebGL, []);
  const scene = useQuery({ ...sceneQuery(place), enabled: webgl });
  const read = scene.data;
  // The scene as far as drawing goes: the same object from one read to the next while nothing drawn changed.
  const part = read ? drawnPart(read) : null;
  // biome-ignore lint/correctness/useExhaustiveDependencies: `part` is what of `read` is drawn
  const data = useMemo(() => read, [part]);
  const bytes = useMapBytes(data);
  const hasSplat = !!data?.splat && canDrawSplat(data.splat_format);
  const [chosen, setChosen] = useState<Draw | null>(null);
  const wantSplat = (chosen ?? (hasSplat ? 'splat' : 'points')) === 'splat';
  const [attempt, setAttempt] = useState(0);
  const splatBytes = useSplatBytes(data, hasSplat, attempt);
  // A splat the viewer couldn't draw (its bytes read, and looked right): which bytes, and why.
  const [drawFault, setDrawFault] = useState<{ file: ArrayBuffer; reason: string } | null>(null);
  // A splat that can't be read or drawn leaves the points: the map still shows, with why and a Retry.
  const splatError = splatBytes.isError
    ? messageOf(splatBytes.error)
    : drawFault && drawFault.file === splatBytes.data
      ? drawFault.reason
      : null;
  const draw: Draw = wantSplat && hasSplat && !splatError ? 'splat' : 'points';
  const [splatReady, setSplatReady] = useState(false);
  const host = useRef<HTMLDivElement>(null);
  const built = useRef<Built | null>(null);
  const [colour, setColour] = useState<Colour>('rgb');
  const [size, setSize] = useState(2.5);
  const [show, setShow] = useState({ floor: true, cameras: true, plan: true });
  const [hidden, setHidden] = useState<Set<string>>(() => readHidden(place));
  const hide = (next: Set<string>) => {
    setHidden(next);
    writeHidden(place, next);
  };
  const [failed, setFailed] = useState<string | null>(null);

  // Build the scene once its map's bytes are here, with the points alone until the splat's bytes are (and when
  // they can't be read or drawn), then again around the splat: the viewer is built around its content. It's torn
  // down when they change or the view closes.
  const [ready, setReady] = useState(0);
  const file = splatError ? null : (splatBytes.data ?? null);
  // A newer scene (a learned map's next splat) takes the place of the one drawn only once all of it is here, its
  // splat's bytes too, or the reason they can't be used: until then the one before it stays drawn.
  const settled = !hasSplat || !!file || !!splatError;
  const [drawn, setDrawn] = useState<Drawn | null>(null);
  const mapBytes = bytes.data;
  useEffect(() => {
    if (!data || !mapBytes) return;
    setDrawn(was => {
      if (was && was.scene === data && was.map === mapBytes && was.file === file) return was;
      if (was && was.scene !== data && !settled) return was;
      return { scene: data, map: mapBytes, file };
    });
  }, [data, mapBytes, file, settled]);
  const newer = !!drawn && !!data && drawn.scene !== data;
  // Where the person was looking when a drawn splat was taken down for a newer one.
  const carried = useRef<View | null>(null);
  useEffect(() => {
    const el = host.current;
    if (!webgl || !el || !drawn) return;
    const { scene: data, map, file } = drawn;
    let gone = false;
    let made: Built | null = null;
    // A splat looks right from where its frames were taken, and like floaters from above: it opens from the first of
    // the views it trained on ("Back to the start" goes to the overview). A newer splat of the same place opens
    // from where the one before it was being looked at.
    const opening = carried.current ?? (file ? openingCamera(Object.keys(data.cameras ?? {})) : null);
    carried.current = null;
    (async () => build(el, data, parseCloud(map, 'ply'), file, opening))().then(
      b => {
        if (gone) return b.dispose();
        made = b;
        built.current = b;
        setReady(n => n + 1); // the settings wait for the scene
        if (!file) return;
        b.splatLoaded.then(
          () => {
            if (!gone) setSplatReady(true);
          },
          e => {
            if (!gone) setDrawFault({ file, reason: messageOf(e) });
          },
        );
      },
      e => {
        if (!gone) setFailed(messageOf(e));
      },
    );
    return () => {
      gone = true;
      built.current = null;
      setSplatReady(false);
      // only a splat's view is carried: the points drawn while a first splat loads give way to its opening view
      carried.current = file && made ? made.view() : null;
      made?.dispose();
    };
  }, [webgl, drawn]);

  // The views shown: the drawn map's, by what's hidden. The splat's are read beside it, and applied once it's here.
  const sources = drawn?.scene.sources ?? null;
  const visible = useMemo(() => (sources ? visibleViews(sources, hidden) : null), [sources, hidden]);
  const splatViews = useSplatViews(sources?.splat_views, hasSplat).data ?? null;
  // biome-ignore lint/correctness/useExhaustiveDependencies: `ready` says the scene was rebuilt and wants its views
  useEffect(() => {
    built.current?.showViews(visible, splatViews);
  }, [visible, splatViews, ready, splatReady]);

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
  // A read that fails while a map is drawn leaves it drawn: the next read may well answer.
  if ((scene.isError && !read) || (bytes.isError && !drawn) || failed)
    return (
      <Text size="sm" c="red">
        Couldn't show the map: {failed ?? messageOf(scene.error ?? bytes.error)}
      </Text>
    );
  if (data && !data.revision)
    return (
      <Text size="sm" c="dimmed">
        {(read && learningNote(read)) ?? 'No 3D map of this place yet. It appears once the house has built one.'}
      </Text>
    );
  if (data && !canDraw(data.map_format))
    return (
      <Text size="sm" c="dimmed">
        This map is in a format the view can't draw yet ({data.map_format}).
      </Text>
    );
  const loading = !drawn;
  // Loading is said whichever way the map is drawn meanwhile: the points show under it.
  const splatLoading = hasSplat && !splatError && !splatReady && !newer;
  const note = read ? splatNote(read) : null;
  const learning = read ? learningNote(read) : null;
  const plan = read ? planNote(read) : null;
  const retry = () => {
    setDrawFault(null);
    setAttempt(n => n + 1);
  };
  const cams = Object.keys((drawn?.scene ?? data)?.cameras ?? {});
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
        {loading && (
          <Text size="sm" c="dimmed" className={classes.cloudNote}>
            {data?.map_bytes ? `Loading the map (${(data.map_bytes / 1e6).toFixed(1)} MB)…` : 'Loading the map…'}
          </Text>
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
      {drawn && (
        <SourceFilter
          sources={sources}
          cameras={cams}
          hidden={hidden}
          onChange={hide}
          noun={draw === 'splat' ? 'splats' : 'points'}
          splatUntold={draw === 'splat' && !!sources && !sources.splat_views}
        />
      )}
      {splatLoading && (
        <Group gap="xs" mt="xs" wrap="nowrap" role="status">
          <Text size="xs" c="dimmed">
            {loadingText(splatBytes.got.received, splatBytes.got.total, !!splatBytes.data)}
            {draw === 'splat' && !loading ? ' The points are drawn meanwhile.' : ''}
          </Text>
          {!splatBytes.data && splatBytes.got.total > 0 && (
            <Progress
              value={(splatBytes.got.received / splatBytes.got.total) * 100}
              size="sm"
              w={120}
              aria-label="How much of the splat has loaded"
            />
          )}
        </Group>
      )}
      {newer && (
        <Text size="xs" c="dimmed" mt="xs" role="status">
          A newer version of this map is loading. This one stays until it's ready.
        </Text>
      )}
      {splatError && (
        <Group gap="xs" mt="xs" role="alert">
          <Text size="xs" c="red">
            Couldn't show the splat: {splatError}. The points are drawn instead.
          </Text>
          <Button size="compact-xs" variant="light" onClick={retry}>
            Retry
          </Button>
        </Group>
      )}
      {note && (
        <Text size="xs" c="dimmed" mt="xs" role="status">
          {note}
        </Text>
      )}
      {learning && (
        <Text size="xs" c="dimmed" mt="xs">
          {learning}
        </Text>
      )}
      {plan && (
        <Text size="xs" c="dimmed" mt="xs">
          {plan}
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
