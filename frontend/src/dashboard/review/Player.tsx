// The Review player (R2, #327): one camera large with what it noticed as a caption, the others as tiles beside it
// (click one to swap it in), each named like "Front camera", and "No video now" on a camera with no video at the
// playhead; then one control bar (play/pause, the time, the speed, the sound, the timeline's tools, and "?", #2454)
// over the timeline (#2577). Each camera has two <video>s that the engine drives directly (one shows, the other holds the next segment). They mount once and never move: which camera is
// large is only where the grid puts it.
// A noninteractive hint for the engine's [ and ] moment keys floats at the large video's bottom center; captions
// sit above it. The hint stays visible without hover or focus on every screen size.
import { ActionIcon, Button, Checkbox, Chip, Group, Kbd, Skeleton, Text, Tooltip, UnstyledButton } from '@mantine/core';
import { IconEqual, IconPlayerPauseFilled, IconPlayerPlayFilled, IconVolume, IconVolumeOff } from '@tabler/icons-react';
import { useQuery } from '@tanstack/react-query';
import { type CSSProperties, type ReactNode, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { evidenceAt, journalEvidenceQuery, toMedia } from '../journal/dayEvidence';
import { SourceMedia } from '../journal/evidence';
import { hms } from '../lib/format';
import { setPref, usePrefs } from '../lib/prefs';
import { useLive } from '../live/hooks';
import { micName } from '../livetab/Mics';
import { FacePicker } from '../people/FacePicker';
import type { LiveFaceDraft } from '../people/liveFace';
import { useEngine, useReview } from './context';
import type { VideoLike } from './engine';
import { GatewayVideo } from './gatewayVideo';
import classes from './player.module.css';
import { ReviewBoxes } from './ReviewBoxes';
import { ReviewHelp } from './ReviewHelp';
import { nextSpanStart, prevSpanStart } from './spans';
import { TimelineTools } from './Timeline';
import { segAt } from './time';
import type { CamNote, Face } from './types';
import { viewName } from './views';

const SPEEDS = [1, 1.5, 2] as const;
/** Read the hotkeys' own moment list; all navigation stays with the engine's keyboard listener. */
function MomentNavigationHint() {
  const engine = useEngine();
  const t = useReview(s => s.t);
  useReview(s => s.sel);
  useReview(s => s.dataVersion);
  const moments = engine.present;
  if (!moments || moments.length <= 1) return null;
  const at = engine.t ?? t ?? engine.view.t0;
  const next = nextSpanStart(moments, at);
  // Before the first moment none has started; in a gap retain the last moment's position.
  const position = next == null ? moments.length : moments.findIndex(([start]) => start === next);
  const previous = position > 1 && prevSpanStart(moments, at) != null;
  return (
    <p className={classes.momentNavigation} role="note" aria-label="Moment navigation shortcuts">
      {previous && (
        <>
          <span>
            <Kbd className={classes.momentKey}>[</Kbd> prev
          </span>{' '}
        </>
      )}
      <span>
        {position} of {moments.length}
      </span>
      {next != null && (
        <>
          {' '}
          <span>
            next <Kbd className={classes.momentKey}>]</Kbd>
          </span>
        </>
      )}
    </p>
  );
}

/** Who a camera recognized: the names, then how many it didn't. Null when face ID was off. */
export function seen(faces: Face[] | null): string | null {
  if (faces == null) return null;
  const known = (n: string | null | undefined): n is string => !!n && n !== 'unknown' && !/^Face \d+$/.test(n);
  const named = [...new Set(faces.map(f => f.name).filter(known))];
  const others = faces.filter(f => !known(f.name)).length;
  const parts = [...named];
  if (others) parts.push(others === 1 ? 'one person not recognized' : `${others} people not recognized`);
  return parts.length ? parts.join(', ') : 'Nobody';
}

/** A camera note as /history/at sends it since #379 (types.ts's CamNote predates it). */
type Note = Omit<CamNote, 'caption'> & { caption: string | null; unchanged?: boolean; since?: number | null };

/** A time as HH:MM. */
const hm = (ts: number): string => hms(ts).slice(0, 5);

/** The next speed after this one, round again after the last. */
export const nextSpeed = (s: number): number =>
  SPEEDS[(SPEEDS.indexOf(s as (typeof SPEEDS)[number]) + 1) % SPEEDS.length] ?? 1;

/** What plays a camera's recorded video on a canvas: the gateway's player, or a fake in tests. */
export type MakeVideo = (canvas: HTMLCanvasElement) => VideoLike & { close(): void };
export const gatewayVideo: MakeVideo = canvas => new GatewayVideo(canvas);

/** loading: the day is loading, so a placeholder covers the video (the canvases stay mounted: the engine binds them
 *  once). makeVideo: what plays each canvas (the gateway by default). */
export function Player({
  loading = false,
  makeVideo = gatewayVideo,
  compact = false,
  timeline,
  inShowOnly = false,
  empty = null,
}: {
  loading?: boolean;
  makeVideo?: MakeVideo;
  compact?: boolean;
  timeline?: ReactNode; // Review's timeline, under the control bar, whose tools it gets (#2577)
  inShowOnly?: boolean; // the page's Show only holds the video markers and Skip quiet time, so the bar leaves them out
  empty?: string | null; // the page's day has no video at all: this one line, and no tiles
}) {
  const engine = useEngine();
  const cams = engine.cams;
  const cameraFilter = useReview(s => s.cameras);
  const shown = cameraFilter.length ? cams.filter(c => cameraFilter.includes(c)) : cams;
  const avail = useReview(s => s.avail);
  const linkedCamera = useReview(s => s.camera);
  const cameraIntent = useReview(s => s.cameraIntent);
  const t = useReview(s => s.t);
  const gap = useReview(s => s.gap);
  const unplayable = useReview(s => s.unplayable);
  const unplayableCodecs = useReview(s => s.unplayableCodecs);
  const roomOnly = useReview(s => s.roomOnly); // no picture, but the room's kept audio plays (#196)
  const notes = useReview(s => s.notes);
  const playing = useReview(s => s.playing);
  const speed = useReview(s => s.speed);
  const [pick, setPick] = useState<{ camera: string; intent: number | undefined } | null>(null);
  const markers = useMarkers();
  const showBoxes = markers.length > 0;
  const knownOnly = usePrefs(s => s.reviewObjectsKnownOnly !== false);
  const preferred = pick && pick.intent === cameraIntent ? pick.camera : linkedCamera;
  const big =
    preferred && shown.includes(preferred) ? preferred : (avail.find(c => shown.includes(c)) ?? shown[0] ?? '');
  // A day with no video at all says so once, in the large camera, with no row of empty tiles beside it (#2454).
  const tiles = empty ? [] : shown.filter(c => c !== big);
  const canvases = useRef<Record<string, (HTMLCanvasElement | null)[]>>({});
  const [naming, setNaming] = useState<LiveFaceDraft | null>(null);
  // Which camera's caption was opened in full: a new caption, or another camera, starts at two lines again.
  const [openCaption, setOpenCaption] = useState<string | null>(null);

  // Marks belong to what the person sees, including tile swaps and the available-camera fallback. Register after
  // layout so keyboard input always reads the camera rendered large, rather than only the last linked camera.
  useLayoutEffect(() => {
    engine.setDisplayedCamera(big);
    return () => engine.setDisplayedCamera(null);
  }, [engine, big]);
  // Stable ref callbacks: React calls each once on mount, and its cleanup on unmount. Each canvas plays a camera's
  // recorded video through Continuum's gateway (#549), standing in for the <video> the engine drives.
  const refs = useMemo(
    () =>
      Object.fromEntries(
        cams.map(c => [
          c,
          [0, 1].map(i => (el: HTMLCanvasElement | null) => {
            if (!el) return;
            canvases.current[c] ??= [];
            canvases.current[c][i] = el;
            const v = makeVideo(el);
            const off = engine.bindVideo(c, i === 0 ? 0 : 1, v);
            return () => {
              if (canvases.current[c]?.[i] === el) canvases.current[c][i] = null;
              off();
              v.close();
            };
          }),
        ]),
      ),
    [engine, cams, makeVideo],
  );
  // The camera's note (#379): its last real caption, with unchanged and since when the view hasn't changed; no caption
  // at all while the vision model warmed up. Who was seen still shows without one.
  const note = notes?.cams?.[big] as Note | undefined;
  const caption = note?.caption || null;
  const captionKey = `${big}\n${caption ?? ''}`;
  const captionOpen = openCaption === captionKey;
  // A note from a while before or after the playhead says so, so it doesn't read as now.
  const off =
    note && Math.abs(note.dt) >= 3 ? ` (${Math.round(Math.abs(note.dt))} s ${note.dt < 0 ? 'before' : 'after'})` : '';
  const who = note ? seen(note.faces) : null;
  // Expiry is the shown camera's own (#1748): another camera's aged-out video says nothing about this one.
  const expired = t != null && segAt(engine.segs[big] ?? [], t)?.broken === 'expired';
  // Once a camera's video has aged out, what the journal cited there is kept (#2808): a promoted clip or still plays in
  // its place, or the citation says its footage expired, with what was seen (#2838). Read only once it's needed.
  const day = useReview(s => s.day);
  const cited = useQuery({ ...journalEvidenceQuery(day ?? ''), enabled: expired && !!day });
  const kept = expired && t != null && cited.data ? evidenceAt(cited.data, big, t) : null;
  const nothing = empty
    ? empty
    : t == null
      ? 'Nothing recorded this day.'
      : expired
        ? `The video around ${hms(t)} has expired and was deleted. What was seen and heard in it is still kept.`
        : gap === 'recording'
          ? `The video around ${hms(t)} is still being recorded. It can be played once its five minutes are up.`
          : gap === 'cut'
            ? `The video around ${hms(t)} was cut off when the camera service restarted, so it can't be played.`
            : avail.length
              ? 'No video from this camera now.'
              : `No video at ${hms(t)}.${roomOnly ? " Playing the room's sound." : ''}`;

  return (
    <section className={classes.player} aria-label="Video">
      {naming && (
        <FacePicker key={`${naming.source.track}:${naming.source.at}`} draft={naming} onClose={() => setNaming(null)} />
      )}
      <div
        className={classes.stage}
        data-tiles={tiles.length > 0 || undefined}
        style={{ '--tiles': Math.max(1, tiles.length) } as CSSProperties}
      >
        {cams.map(c => {
          const large = c === big;
          const name = viewName(c);
          const has = avail.includes(c);
          return (
            <div
              key={c}
              className={large ? classes.big : classes.tile}
              style={
                {
                  display: shown.includes(c) && (large || !empty) ? undefined : 'none',
                  '--row': large ? `1 / span ${Math.max(1, tiles.length)}` : `${tiles.indexOf(c) + 1}`,
                } as CSSProperties
              }
              data-video={has || undefined}
            >
              <canvas ref={refs[c]?.[0]} className="live-canvas" role="img" aria-label={`${name} video`} />
              <canvas ref={refs[c]?.[1]} className="live-canvas" aria-hidden />
              <span className={classes.label}>{name}</span>
              {large ? (
                <>
                  {has && showBoxes && !compact && (
                    <ReviewBoxes
                      kinds={markers}
                      knownOnly={knownOnly}
                      cam={c}
                      onFace={setNaming}
                      getCanvas={() =>
                        canvases.current[c]?.find(canvas => canvas && !canvas.hasAttribute('data-standby')) ?? null
                      }
                    />
                  )}
                  {!has && kept && (kept.stage === 'clip' || kept.stage === 'still') ? (
                    <figure className={classes.kept} data-stage={kept.stage}>
                      <SourceMedia source={toMedia(kept)} />
                      <figcaption className={classes.keptCaption}>
                        {kept.stage === 'clip' ? 'Kept clip' : 'Kept still'}
                        {kept.caption ? `: ${kept.caption}` : ''}
                      </figcaption>
                    </figure>
                  ) : (
                    !has && (
                      <p className={classes.nothing}>
                        {nothing}
                        {kept?.stage === 'expired' && kept.caption ? ` The journal cites it: ${kept.caption}` : ''}
                      </p>
                    )
                  )}
                  {has && unplayable.includes(c) && t != null && (
                    <p className={classes.nothing}>
                      {unplayableCodecs[c] === 'av1'
                        ? `The video around ${hms(t)} was archived as AV1, which this browser can't play. Chrome, Edge, Firefox, and Safari on recent Apple devices can.`
                        : `The video around ${hms(t)} is in a format this browser can't play.`}
                    </p>
                  )}
                  {notes?.forbidden ? (
                    <p className={classes.caption}>
                      What the cameras saw is only shown on the machine Liberational runs on. To see it here, turn on
                      Remote camera history in Settings (dashboard.remote_camera_history).
                    </p>
                  ) : (
                    note &&
                    (caption || who) && (
                      <p className={classes.caption}>
                        {caption && (
                          // Two lines over the picture, the notes on the same line; a tap shows the rest. The button
                          // keeps its identity as captions change, so keyboard focus stays on it (#2454 follow-up).
                          <span className={classes.captionLine} data-open={captionOpen || undefined}>
                            <UnstyledButton
                              className={classes.captionText}
                              aria-expanded={captionOpen}
                              onClick={() => setOpenCaption(captionOpen ? null : captionKey)}
                            >
                              {caption}
                            </UnstyledButton>
                            {note.unchanged && note.since != null && (
                              <span className={classes.same} title={`Unchanged since ${hm(note.since)}`}>
                                <IconEqual size={14} stroke={2.5} aria-hidden="true" />
                                <span className={classes.srOnly}>{`Unchanged since ${hm(note.since)}`}</span>
                              </span>
                            )}
                            {off && <span className={classes.off}>{off}</span>}
                          </span>
                        )}
                        {who && <span className={classes.who}>Seen: {who}</span>}
                      </p>
                    )
                  )}
                  {!compact && !loading && <MomentNavigationHint />}
                </>
              ) : (
                <>
                  {!has && <span className={classes.noneTile}>No video now</span>}
                  <UnstyledButton
                    className={classes.swap}
                    aria-label={`Show the ${name.toLowerCase()} large`}
                    onClick={() => {
                      setPick({ camera: c, intent: cameraIntent });
                      engine.pickCamera(c);
                    }}
                  />
                </>
              )}
            </div>
          );
        })}
        {loading && <Skeleton className={classes.cover} radius={0} aria-hidden data-testid="player-skeleton" />}
      </div>
      <div className={classes.controls}>
        <ActionIcon
          size={32}
          radius="xl"
          className={classes.play}
          onClick={() => engine.togglePlay()}
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? <IconPlayerPauseFilled size={14} aria-hidden /> : <IconPlayerPlayFilled size={14} aria-hidden />}
        </ActionIcon>
        <time className={classes.clock} ref={el => engine.setClock(el)} />
        <Button
          variant="default"
          size="compact-md"
          className={classes.speed}
          onClick={() => engine.setSpeed(nextSpeed(speed))}
          aria-label={`Speed ${speed}×. Change to ${nextSpeed(speed)}×`}
        >
          {speed}×
        </Button>
        <RoomSound />
        {!compact && (
          <>
            <span className={classes.spacer} />
            {timeline && <TimelineTools mark />}
            {!inShowOnly && <VideoMarkers />}
            {!inShowOnly && <SkipQuiet />}
            <ReviewHelp timeline={!!timeline} />
          </>
        )}
      </div>
      {timeline}
    </section>
  );
}

/** Which boxes Review draws on the large camera: the viewer's choice, else people when the old switch was on. */
function useMarkers(): ('people' | 'pets' | 'objects')[] {
  const legacyBoxes = usePrefs(s => s.reviewBoxes !== false);
  const markerPref = usePrefs(s => s.reviewMarkers);
  return markerPref ?? (legacyBoxes ? ['people'] : []);
}

/** The boxes to draw on the video (people, pets, objects) and, with objects, Known only. On Review's bar for an
 *  embedded clip; Review's own page puts them in Show only (#2454). */
export function VideoMarkers() {
  const markers = useMarkers();
  const knownOnly = usePrefs(s => s.reviewObjectsKnownOnly !== false);
  return (
    <>
      <Chip.Group
        multiple
        value={markers}
        onChange={value =>
          setPref(
            'reviewMarkers',
            value.filter(
              (kind): kind is 'people' | 'pets' | 'objects' =>
                kind === 'people' || kind === 'pets' || kind === 'objects',
            ),
          )
        }
      >
        <Group gap={6} role="group" aria-label="Review markers">
          <Chip size="xs" value="people">
            People
          </Chip>
          <Chip size="xs" value="pets">
            Pets
          </Chip>
          <Chip size="xs" value="objects">
            Objects
          </Chip>
        </Group>
      </Chip.Group>
      {markers.includes('objects') && (
        <Tooltip
          label="Only the objects you asked to remember; off adds the camera's labels (chair, car)"
          events={{ hover: true, focus: true, touch: false }}
        >
          <Checkbox
            size="xs"
            label="Known only"
            checked={knownOnly}
            onChange={e => setPref('reviewObjectsKnownOnly', e.currentTarget.checked)}
          />
        </Tooltip>
      )}
    </>
  );
}

/** Skip quiet time: while playing, jump over the time none of what Show only picks is there. */
export function SkipQuiet() {
  const engine = useEngine();
  const skip = useReview(s => s.skip);
  return (
    <Tooltip
      label="While playing, skip the time when none of what Show only picks is there (s)"
      events={{ hover: true, focus: true, touch: false }}
    >
      <Checkbox label="Skip quiet time" checked={skip} onChange={() => engine.toggleSkip()} />
    </Tooltip>
  );
}

/** What Review plays aloud (#2041): the room's microphone, never a camera (the recordings carry no sound), with a
 *  mute that keeps the video playing. */
function RoomSound() {
  const engine = useEngine();
  const muted = useReview(s => s.muted);
  const mics = Object.values(useLive('ambient')?.devices ?? {});
  const name = mics.length === 1 && mics[0] ? micName(mics[0].name) : null;
  return (
    <Group gap={4} wrap="nowrap">
      <Text size="xs" c="dimmed" title={`Sound: the room microphone${name ? ` (${name})` : ''}`}>
        Room mic
      </Text>
      <ActionIcon
        variant="subtle"
        size="sm"
        aria-pressed={muted}
        aria-label={muted ? 'Unmute the room microphone' : 'Mute the room microphone'}
        onClick={() => engine.setMuted(!muted)}
      >
        {muted ? <IconVolumeOff size={16} aria-hidden /> : <IconVolume size={16} aria-hidden />}
      </ActionIcon>
    </Group>
  );
}
