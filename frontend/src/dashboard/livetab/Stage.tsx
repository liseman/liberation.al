// Live opens multiple cameras as preview cards in a grid of equal cells; a portrait picture fits its cell.
// A selected exact camera ID gets the full video and
// face/object controls; the #camera hash preserves that selection across reloads. The caption sits below pixels.
import { Anchor, Button, NativeSelect, Switch, Text, Tooltip, UnstyledButton } from '@mantine/core';
import { IconEqual } from '@tabler/icons-react';
import { useQuery } from '@tanstack/react-query';
import { type RefObject, useEffect, useMemo, useRef, useState } from 'react';
import { cropLiveObject } from '../../objects/handoff';
import { type Box, type BoxPick, liveBoxes } from '../camera/boxes';
import { CameraControls } from '../camera/CameraControls';
import { currentBoxes } from '../camera/gateway';
import { LiveBoxes } from '../camera/LiveBoxes';
import { LiveCam, useCameraAspect } from '../camera/LiveCam';
import { PreviewCam } from '../camera/PreviewCam';
import { hm } from '../lib/format';
import { mirrored, setMirrored, setPref, usePrefs } from '../lib/prefs';
import { useLive } from '../live/hooks';
import { useLiveStatus } from '../live/status';
import type { VisionCamera } from '../live/types';
import { FacePicker } from '../people/FacePicker';
import { type LiveFaceDraft, prepareLiveFace } from '../people/liveFace';
import { getPets, PETS_KEY, type PetsView, petHref } from '../people/petLinks';
import { setSkeletons, useSkeletons } from '../pose/prefs';
import { usePausedCameras } from '../privacy';
import { unnamed, viewName } from '../review/views';
import { href, linkClick, navigate, tabOf } from '../router';
import { DEVICE_CARD } from '../settings/devices/cards';
import { useActiveCameras, useCameraNames } from '../shell';
import { VISION_FRESH_S } from '../status';
import { WHOS_THIS } from '../whosthis/labels';
import type { LiveScene } from './Rail';
import { CAMERA_SERVICES, StartButton } from './Setup';
import { LearnSurroundings } from './Surroundings';
import './cameras.css';

/** Mirror this camera's picture left to right, in this browser only (the boxes are drawn here, so they mirror
 *  too). Settings' Mirror is the same choice. */
function FlipButton({ cam }: { cam: string }) {
  const on = usePrefs(s => mirrored(s, cam));
  return (
    <Tooltip label="Mirror the camera picture" events={{ hover: true, focus: true, touch: false }}>
      <UnstyledButton
        className={`flip-btn${on ? ' on' : ''}`}
        aria-pressed={on}
        onClick={() => setMirrored(cam, !on)}
        aria-label="Mirror the camera picture"
      >
        {'⇆'}
      </UnstyledButton>
    </Tooltip>
  );
}

/** Draw people's skeletons over the large camera, in this browser (Review's Show skeletons is the same choice). */
export function SkeletonButton() {
  const on = useSkeletons();
  return (
    <Tooltip label="Show people's skeletons" events={{ hover: true, focus: true, touch: false }}>
      <UnstyledButton
        className={`flip-btn skel-btn${on ? ' on' : ''}`}
        aria-pressed={on}
        onClick={() => setSkeletons(!on)}
        aria-label="Show people's skeletons"
      >
        Skeletons
      </UnstyledButton>
    </Tooltip>
  );
}

type CamState = 'live' | VisionCamera['state'] | 'off' | 'paused';

const TROUBLE: Partial<Record<CamState, string>> = {
  retrying: 'Camera not responding, retrying',
  starting: 'Starting…',
  paused: 'Paused for privacy',
};

/** On the large view, an unknown face says "Who's this?" (its box links there). */
export const relabel = (b: Box): Box =>
  b.kind === 'face' && b.tone === 'unknown' ? { ...b, label: "Who's this?" } : b;

/** Follow a large-view box: faces open People; an animal (a taught pet's name, or a species) opens Pets, at that pet
 *  or for that species; and any other object opens a cropped Objects draft. */
export function follow(pick: BoxPick, camera?: string, canvas?: HTMLCanvasElement | null, pets?: PetsView): void {
  const animal = pick.kind === 'object' ? petHref(pick.label, pets) : null;
  if (animal) {
    navigate(animal);
    return;
  }
  if (pick.kind === 'object') {
    if (!camera || !canvas) throw new Error('No displayed picture is available.');
    cropLiveObject(canvas, pick.box, camera, pick.label);
    navigate('/objects');
    return;
  }
  navigate(pick.tone === 'unknown' ? `/people#${WHOS_THIS}` : `/people#card=${encodeURIComponent(pick.label)}`);
}

function Large({
  cam,
  state,
  last,
  onFace,
}: {
  cam: string;
  state: CamState;
  last?: LiveScene;
  onFace: (draft: LiveFaceDraft) => void;
}) {
  const mirror = usePrefs(s => mirrored(s, cam));
  const skeletons = useSkeletons();
  const name = viewName(cam);
  const figure = useCameraAspect<HTMLElement>(cam);
  const [error, setError] = useState('');
  const pets = useQuery({ queryKey: PETS_KEY, queryFn: ({ signal }) => getPets(signal), staleTime: 60_000 }).data;
  const pick = (value: BoxPick) => {
    if (value.kind === 'face' && value.tone !== 'unknown') return follow(value);
    if (value.kind === 'object' && petHref(value.label, pets)) return follow(value, cam, null, pets);
    try {
      const canvas = figure.current?.querySelector<HTMLCanvasElement>('canvas.live-canvas:not(.stale):not(.off)');
      if (!canvas) throw new Error('No displayed picture is available.');
      if (value.kind === 'face') onFace(prepareLiveFace(value, canvas));
      else follow(value, cam, canvas);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not prepare the selected picture.');
    }
  };
  return (
    <figure ref={figure} className={`cam-large${mirror ? ' mirror' : ''}`} aria-label={`${name}, large`}>
      <div className="cam-picture">
        {state === 'live' ? (
          <>
            <LiveCam cam={cam} alt={last?.caption || `Live video from ${name}`} />
            <LiveBoxes cam={cam} mirror={mirror} onPick={pick} pickObjects relabel={relabel} skeletons={skeletons} />
          </>
        ) : (
          <div className="cam-dark">{TROUBLE[state] ?? 'Camera off'}</div>
        )}
        <span className="cam-name" title={name}>
          {state === 'live' && <span className="cam-dot" aria-hidden="true" />}
          <span className="cam-name-text">{name}</span>
          <CameraControls camera={cam} name={name} />
        </span>
        {state === 'live' && <FlipButton cam={cam} />}
        {state === 'live' && <SkeletonButton />}
      </div>
      {error && <Text role="alert">{error}</Text>}
      {state === 'live' && last?.caption && (
        <figcaption className="cam-caption">
          {last.unchanged && (
            <span
              className="cam-unchanged"
              role="img"
              aria-label={last.since ? `Unchanged since ${hm(last.since)}` : 'Unchanged'}
              title={last.since ? `Unchanged since ${hm(last.since)}` : 'Unchanged'}
            >
              <IconEqual size={14} stroke={2.5} aria-hidden="true" />
            </span>
          )}
          {last.caption}
        </figcaption>
      )}
    </figure>
  );
}

/** Play every camera in the overview live, or show a picture every few seconds (lighter on a slow link). */
function LiveTilesSwitch() {
  const on = usePrefs(s => s.liveTiles ?? true);
  return (
    <Switch
      size="xs"
      label="Live tiles"
      checked={on}
      onChange={e => setPref('liveTiles', e.currentTarget.checked)}
      aria-label="Play every camera live in the overview"
    />
  );
}

function Small({ cam, state, onShow }: { cam: string; state: CamState; onShow: () => void }) {
  const name = viewName(cam);
  const mirror = usePrefs(s => mirrored(s, cam));
  // Live tiles: the same JPEG stream Live's large camera shows; off, a sparse preview every few s.
  const live = usePrefs(s => s.liveTiles ?? true);
  const tile = useCameraAspect<HTMLDivElement>(cam);
  return (
    <div ref={tile} className={`cam-tile${mirror ? ' mirror' : ''}`}>
      <UnstyledButton className="cam-tile-open" aria-label={`Open ${name} camera`} onClick={onShow}>
        <span className="cam-tile-picture">
          {state === 'live' ? (
            live ? (
              <LiveCam cam={cam} alt={`Live video from ${name}`} />
            ) : (
              <PreviewCam cam={cam} alt={`Preview from ${name}`} />
            )
          ) : (
            <span className="cam-dark">{TROUBLE[state] ?? 'Camera off'}</span>
          )}
        </span>
      </UnstyledButton>
      <span className="cam-tile-label">
        <span className="cam-name-text">{name}</span>
        <CameraControls camera={cam} name={name} />
      </span>
    </div>
  );
}

/** A camera with no name goes by its id on every page, so the heading offers to name it. It reads only the
 *  page shell, so it is there from the first paint, in the waiting heading too, and moves nothing when it lands. */
function NameNudge({ cams }: { cams: readonly string[] }) {
  const missing = unnamed(cams, useCameraNames()); // a name saved in Settings takes it away at once
  if (!missing.length) return null;
  return (
    <Anchor
      className="cams-name-nudge"
      href={`${href('/settings')}#${DEVICE_CARD.cameras}`}
      onClick={linkClick}
      title="Name the view each camera shows, like front or desk, in Settings"
    >
      {missing.length === 1 ? `Name ${missing[0]}` : `Name ${missing.length} cameras`}
    </Anchor>
  );
}

function cameraInHash(): string | null {
  return new URLSearchParams(location.hash.slice(1)).get('camera') || null;
}

function writeCamera(cam: string | null): void {
  const hash = new URLSearchParams(location.hash.slice(1));
  if (cam) hash.set('camera', cam);
  else hash.delete('camera');
  history.replaceState(history.state, '', location.pathname + location.search + (hash.size ? `#${hash}` : ''));
}

/** The cameras' last height in this browser, so their place is held at it on the next load. */
const STAGE_H_KEY = 'liberational.live.stageHeight';
function rememberedStageHeight(): number | undefined {
  try {
    const h = Number(localStorage.getItem(STAGE_H_KEY));
    return h > 0 ? h : undefined;
  } catch {
    return undefined;
  }
}
function rememberHeight(root: RefObject<HTMLElement | null>) {
  return (el: HTMLElement | null) => {
    root.current = el;
    if (!el || typeof ResizeObserver !== 'function') return;
    const ro = new ResizeObserver(() => {
      const h = Math.round(el.getBoundingClientRect().height);
      try {
        if (h > 0) localStorage.setItem(STAGE_H_KEY, String(h));
      } catch {
        // not remembered this time: the next load holds a 16:9 place instead
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  };
}

export function Stage({
  last,
  live,
  trouble = null,
  naming,
  sceneKnown = true,
}: {
  last: Record<string, LiveScene>;
  live: string[];
  trouble?: 'starting' | 'retrying' | null;
  sceneKnown?: boolean; // the first scene read has answered: before it, no camera is known to be off
  naming?: RefObject<(() => boolean) | null>;
}) {
  const vision = useLive('vision');
  const now = useLiveStatus(s => s.now);
  const [chosen, setChosen] = useState<string | null>(cameraInHash);
  const root = useRef<HTMLElement>(null);
  const stageRef = useMemo(() => rememberHeight(root), []);
  const [faceDraft, setFaceDraft] = useState<LiveFaceDraft | null>(null);
  const [error, setError] = useState('');
  const cams = useActiveCameras();
  const paused = usePausedCameras();
  const fresh = !!vision?.ts && now - vision.ts <= VISION_FRESH_S;
  // Vision's own word on a camera wins while fresh: starting or retrying, else on (its picture, even before its
  // first caption, as while the VLM warms up). Without it, a camera with a recent scene entry is live.
  const stateOf = (cam: string): CamState => {
    if (paused.has(cam)) return 'paused'; // capture is stopping or has stopped it: never show a last frame
    const said = fresh ? vision?.cameras?.[cam]?.state : undefined;
    if (said === 'starting' || said === 'retrying') return said;
    return said === 'on' || live.includes(cam) ? 'live' : 'off';
  };
  // Keep a failed camera visible alongside the others, so a missing picture never silently removes a source.
  const shown = cams.some(c => stateOf(c) !== 'off') ? cams : [];
  const large = chosen && shown.includes(chosen) ? chosen : shown.length === 1 ? shown[0] : undefined;
  const choose = (cam: string | null) => {
    setChosen(cam);
    writeCamera(cam);
  };
  useEffect(() => {
    const follow = () => {
      if (tabOf(location.pathname) === 'live') setChosen(cameraInHash());
    };
    window.addEventListener('hashchange', follow);
    window.addEventListener('popstate', follow);
    return () => {
      window.removeEventListener('hashchange', follow);
      window.removeEventListener('popstate', follow);
    };
  }, []);
  useEffect(() => {
    if (!naming) return;
    const open = () => {
      const canvases = [
        ...(root.current?.querySelectorAll<HTMLCanvasElement>('canvas.live-canvas:not(.stale):not(.off)') ?? []),
      ];
      // Prefer the view already large, then the other displayed cameras; never change the selected view.
      const ordered = [...canvases].sort((a, b) => Number(b.dataset.cam === large) - Number(a.dataset.cam === large));
      for (const canvas of ordered) {
        const frame = currentBoxes(canvas.dataset.cam ?? '');
        const face =
          frame && liveBoxes(frame).find(box => box.kind === 'face' && box.tone === 'unknown' && box.faceSource);
        if (!face) continue;
        try {
          setFaceDraft(prepareLiveFace({ ...face, left: 0, top: 0, width: 0, height: 0 }, canvas));
          setError('');
        } catch (reason) {
          setError(reason instanceof Error ? reason.message : 'Could not prepare the selected picture.');
        }
        return true;
      }
      return false;
    };
    naming.current = open;
    return () => {
      if (naming.current === open) naming.current = null;
    };
  }, [naming, large]);
  const picker = faceDraft && (
    <FacePicker
      key={`${faceDraft.source.track}:${faceDraft.source.at}`}
      draft={faceDraft}
      onClose={() => setFaceDraft(null)}
    />
  );
  // The hash and configured cameras are known before live status; keep these controls in place while it loads.
  const selected = chosen && cams.includes(chosen) && cams.length > 1 ? chosen : null;
  const header = (
    <div className="cams-head">
      <h2>Cameras</h2>
      <span>{cams.length === 1 ? '1 camera' : `${cams.length} cameras`}</span>
      <NameNudge cams={cams} />
      <LearnSurroundings />
      {selected && (
        <div className="cam-controls">
          <Button variant="default" size="xs" onClick={() => choose(null)}>
            All cameras
          </Button>
          <NativeSelect
            aria-label="Choose camera"
            value={selected}
            onChange={e => choose(e.currentTarget.value)}
            data={cams.map(cam => ({ value: cam, label: viewName(cam) }))}
          />
        </div>
      )}
    </div>
  );
  // Until vision and the scene have both answered, nothing is known to be off, so the cameras' place is held
  // rather than saying "Cameras are off" and then swapping the cameras in over it, which moved everything
  // under them. The overview's place is the overview itself, with empty tiles, so the grid's own CSS sizes it
  // exactly; one large camera's place is its height last time in this browser, else a capped 16:9 picture.
  if (!shown.length && (vision === undefined || !sceneKnown)) {
    const overview = cams.length > 1 && !(chosen && cams.includes(chosen));
    return (
      // Its own key, as the cameras' section has: React would otherwise reuse this section's nodes for the
      // real one by position, and the grid's node turning into the heading counts as a shift though nothing moves.
      <section key="stage-wait" className="live-cams stage-wait" aria-label="Cameras" aria-busy="true">
        {header}
        {overview ? (
          <>
            <div className="cams-overview-bar">
              <Text size="sm">Open a camera for face names and object selections.</Text>
              <LiveTilesSwitch />
            </div>
            <div className={`cams-grid${cams.length > 4 ? ' many' : ''}`}>
              {cams.map(cam => (
                <div key={cam} className="cam-tile cam-tile-wait" />
              ))}
            </div>
          </>
        ) : (
          <div className="cam-wait" style={{ height: rememberedStageHeight() }} />
        )}
      </section>
    );
  }
  if (!shown.length)
    return (
      <section ref={root} className="stage stage-off" aria-label="Cameras">
        {picker}
        {trouble === 'retrying' ? (
          <Text c="dimmed">Camera not responding, retrying.</Text>
        ) : trouble === 'starting' ? (
          <Text c="dimmed">Starting cameras…</Text>
        ) : (
          <>
            <Text c="dimmed">Cameras are off, so Liberational can't see who's here.</Text>
            <StartButton name={CAMERA_SERVICES} label="Start cameras" />
          </>
        )}
      </section>
    );
  return (
    <section key="stage" ref={stageRef} className="live-cams" aria-label="Cameras">
      {picker}
      {error && <Text role="alert">{error}</Text>}
      {header}
      {large ? (
        <div className="cams-single">
          <Large
            key={large}
            cam={large}
            state={stateOf(large)}
            last={last[large]}
            onFace={draft => {
              setError('');
              setFaceDraft(draft);
            }}
          />
        </div>
      ) : (
        <>
          <div className="cams-overview-bar">
            <Text size="sm">Open a camera for face names and object selections.</Text>
            <LiveTilesSwitch />
          </div>
          <div className={`cams-grid${shown.length > 4 ? ' many' : ''}`}>
            {shown.map(cam => (
              <Small key={cam} cam={cam} state={stateOf(cam)} onShow={() => choose(cam)} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}
