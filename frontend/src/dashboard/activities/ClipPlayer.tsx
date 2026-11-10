// One activity clip's own small player: a poster until it's asked to play, then Review's player in an embedded
// engine of its own, bounded to the clip (clip/clip.ts, shared with People's videos). Only a clip someone
// plays loads any video; before that the poster shows the clip's start frame, a prepared still (clipThumb.ts), with
// a skeleton while it's being cut. The expand button (or a click on the playing video) opens the clip bigger in the
// shared floating window (clip/ClipFloat.tsx), an engine of its own over the same segments, started where the card
// was and playing if it was; closing hands its time back to the card.
import { ActionIcon, Alert, UnstyledButton } from '@mantine/core';
import { IconArrowsMaximize } from '@tabler/icons-react';
import { type CSSProperties, type MouseEvent, useEffect, useRef, useState } from 'react';
import { ClipFloat } from '../clip/ClipFloat';
import { ClipPoster } from '../clip/ClipPoster';
import { type LoadSegments, type Moment, useClip } from '../clip/clip';
import { cameraRatio, learnRatio, ratioOf, WIDE } from '../clip/ratio';
import { clipLength, hm } from '../lib/format';
import { EngineContext } from '../review/context';
import { type MakeVideo, Player } from '../review/Player';
import classes from './activities.module.css';
import { useClipThumb, useNearScreen } from './clipThumb';

export const LEAD_S = 1; // a second of run-up before the clip, and after it

const shape = (ratio: number | null): CSSProperties => ({ '--clip-ratio': ratio ?? WIDE }) as CSSProperties;

export interface ClipPlayerProps {
  cam: string;
  start: number;
  end: number;
  label: string; // what the clip is, for the play button's name
  makeVideo?: MakeVideo;
  loadSegments?: LoadSegments;
}

function Embedded({
  cam,
  start,
  end,
  label,
  makeVideo,
  loadSegments,
  ratio,
  onRatio,
}: ClipPlayerProps & { ratio: number | null; onRatio: (ratio: number) => void }) {
  const frame = useRef<HTMLDivElement>(null);
  const { engine, clip, loading, error } = useClip({ cam, start: start - LEAD_S, end: end + LEAD_S, loadSegments });
  const [popup, setPopup] = useState<Moment | null>(null);
  // The gateway sizes its canvas to the video's own frame: once it has painted, that's the camera's true shape.
  useEffect(() => {
    const root = frame.current;
    if (!root) return;
    const read = () => {
      for (const canvas of root.querySelectorAll('canvas')) {
        const found = ratioOf(canvas.width, canvas.height);
        if (found && canvas.width > 300) return onRatio(found); // past a fresh canvas's default 300 x 150
      }
    };
    const observer = new MutationObserver(read);
    observer.observe(root, { attributes: true, attributeFilter: ['width', 'height'], subtree: true });
    return () => observer.disconnect();
  }, [onRatio]);
  const expand = () => {
    if (!clip) return;
    const moment = { t: engine.t ?? clip.at, play: engine.playing };
    engine.pause();
    setPopup(moment);
  };
  const back = (moment: Moment) => {
    setPopup(null);
    engine.seek(moment.t, { play: moment.play });
  };
  if (error)
    return (
      <Alert color="gray" className={classes.clipBox} style={shape(ratio)}>
        {error}
      </Alert>
    );
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: a pointer shortcut; the expand button is the keyboard way
    // biome-ignore lint/a11y/useKeyWithClickEvents: as above, the expand button is focusable and labelled
    <div
      ref={frame}
      className={classes.clipFrame}
      style={shape(ratio)}
      onClick={(e: MouseEvent) => {
        // React bubbles the popup's clicks through its portal to here: only the card's own video expands.
        if (!popup && e.target instanceof HTMLCanvasElement && e.currentTarget.contains(e.target)) expand();
      }}
    >
      <EngineContext value={engine}>
        <Player loading={loading} compact makeVideo={makeVideo} />
      </EngineContext>
      {clip && (
        <ActionIcon
          className={classes.expand}
          variant="filled"
          color="dark"
          radius="xl"
          aria-label={`Open ${label} bigger`}
          onClick={expand}
        >
          <IconArrowsMaximize size={16} aria-hidden />
        </ActionIcon>
      )}
      {clip && popup && frame.current && (
        <ClipFloat
          cam={cam}
          start={clip.at}
          end={end + LEAD_S}
          label={label}
          origin={frame.current}
          clip={clip}
          moment={popup}
          full
          makeVideo={makeVideo}
          onClose={back}
        />
      )}
    </div>
  );
}

export function ClipPlayer({ cam, start, end, label, makeVideo, loadSegments }: ClipPlayerProps) {
  const [open, setOpen] = useState(false);
  const [ratio, setRatio] = useState<number | null>(() => cameraRatio(cam));
  const learn = useRef((found: number) => {
    learnRatio(cam, found);
    setRatio(previous => (previous !== null && Math.abs(previous - found) < 0.01 ? previous : found));
  }).current;
  const poster = useRef<HTMLButtonElement>(null);
  const near = useNearScreen(poster);
  const thumb = useClipThumb(cam, start, end, near && !open);
  if (open)
    return (
      <Embedded
        cam={cam}
        start={start}
        end={end}
        label={label}
        makeVideo={makeVideo}
        loadSegments={loadSegments}
        ratio={ratio}
        onRatio={learn}
      />
    );
  return (
    <UnstyledButton
      ref={poster}
      type="button"
      className={classes.clipBox}
      // its name ends with the length the poster shows, so what's seen is what's said (axe)
      aria-label={`Play ${label} at ${hm(start)}, ${clipLength(Math.max(1, end - start))}`}
      aria-busy={thumb.waiting || undefined}
      style={shape(ratio)}
      onClick={() => setOpen(true)}
    >
      <ClipPoster
        src={thumb.state === 'ready' ? thumb.url : null}
        seconds={Math.max(1, end - start)}
        waiting={thumb.waiting}
        ratio={ratio}
        onShape={learn}
      />
    </UnstyledButton>
  );
}
