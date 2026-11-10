// A clip opened bigger (#2476): a floating window that grows out of what was clicked (a People card's video, an
// Activities card's clip) and shrinks back into it on close, sized to the camera's own shape. It plays the clip in
// an engine of its own (useClip), keeps the poster until the first frame paints, offers Retry when that's slow, and
// links to the same moment in Review. Closing hands back where it was and whether it was playing.
import { ActionIcon, Button, FocusTrap, getDefaultZIndex, Loader, Portal, Text } from '@mantine/core';
import { IconArrowUpRight, IconX } from '@tabler/icons-react';
import { type CSSProperties, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useStore } from 'zustand';
import { SafeImage } from '../../shared/SafeImage';
import { isoDay } from '../lib/format';
import { PageError } from '../lib/page';
import { EngineContext } from '../review/context';
import { gatewayVideo, type MakeVideo, Player } from '../review/Player';
import { reviewURL } from '../review/url';
import { navigate } from '../router';
import { type Clip, type LoadSegments, type Moment, useClip } from './clip';
import classes from './clipFloat.module.css';
import { cameraRatio, learnRatio, ratioOf, WIDE } from './ratio';

const DURATION = 320;
// Just under Mantine's modal layer (200; popovers, menus, and tooltips are 300), so what the player opens over
// itself (the help popover, face naming's modal) sits above the window rather than behind it (#2507).
export const FLOAT_Z = getDefaultZIndex('modal') - 10;
const FIRST_FRAME_WAIT_MS = 15_000;
const COMPACT_CHROME = 84; // the padding and the compact control row around the stage
const FULL_CHROME = 128; // Review's full control bar, which can wrap to two rows

interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Keep the whole player, controls included, inside a width by height window, with its stage in the camera's shape
 *  (ratio, width over height). */
export function floatingRect(width: number, height: number, ratio = WIDE, chrome = COMPACT_CHROME): Rect {
  const w = Math.max(160, Math.min(960, width - 32, (height - 36 - chrome) * ratio));
  const h = (w - 24) / ratio + chrome;
  return { left: (width - w) / 2, top: Math.max(16, (height - h) / 2), width: w, height: h };
}

function frame(rect: Rect, base: Rect, radius: number): Keyframe {
  const x = rect.width / base.width;
  const y = rect.height / base.height;
  return {
    transform: `translate(${rect.left - base.left}px, ${rect.top - base.top}px) scale(${x}, ${y})`,
    clipPath: `inset(0 round ${radius / x}px / ${radius / y}px)`,
  };
}

function contentFrame(rect: Rect, base: Rect): string {
  const scale = rect.width / (base.width - 24);
  const x = scale / (rect.width / base.width);
  const y = scale / (rect.height / base.height);
  return `translate(${-12 * x}px, ${-12 * y}px) scale(${x}, ${y})`;
}

function motion(element: HTMLElement, from: Rect, to: Rect, reduced: boolean, base = to, hold = false): Promise<void> {
  if (!element.animate) return Promise.resolve();
  const keys = reduced
    ? [{ opacity: hold ? 1 : 0 }, { opacity: hold ? 0 : 1 }]
    : [frame(from, base, hold ? 8 : 6), frame(to, base, hold ? 6 : 8)];
  const animation = element.animate(keys, {
    duration: reduced ? 120 : DURATION,
    easing: 'cubic-bezier(.2,.75,.2,1)',
    fill: 'both',
  });
  return animation.finished
    .then(() => {
      if (!hold) animation.cancel();
    })
    .catch(() => {});
}

export interface ClipFloatProps {
  cam: string;
  start: number;
  end: number;
  label: string; // the window's name for screen readers: "Video of Dana", "squats"
  origin: HTMLElement; // what it grows out of, and shrinks back into
  onClose: (moment: Moment) => void;
  poster?: string | null; // a still to show until the first frame paints
  person?: string; // Review's person filter for Open in Review
  clip?: Clip | null; // already listed by the view it opens from
  moment?: Moment | null; // where to start, and whether to play
  full?: boolean; // Review's full control bar (boxes and help) rather than the compact one
  makeVideo?: MakeVideo;
  loadSegments?: LoadSegments;
}

export function ClipFloat({
  cam,
  start,
  end,
  label,
  origin,
  onClose,
  poster = null,
  person,
  clip = null,
  moment = null,
  full = false,
  makeVideo = gatewayVideo,
  loadSegments,
}: ClipFloatProps) {
  const [attempt, setAttempt] = useState(0);
  const { engine, loading, error, cancel } = useClip({ cam, start, end, loadSegments, clip, moment, attempt });
  const [ratio, setRatio] = useState(() => cameraRatio(cam) ?? WIDE);
  const [view, setView] = useState(() => ({ width: innerWidth, height: innerHeight }));
  const bounds = floatingRect(view.width, view.height, ratio, full ? FULL_CHROME : COMPACT_CHROME);
  const [ready, setReady] = useState(false);
  const [slow, setSlow] = useState(false);
  const [closing, setClosing] = useState(false);
  const t = useStore(engine.ui, state => state.t);
  const unplayable = useStore(engine.ui, state => state.unplayable.length > 0);
  const [panel, setPanel] = useState<HTMLDivElement | null>(null);
  const content = useRef<HTMLDivElement>(null);
  const mounted = useRef(true);
  const returnFocus = useRef(true);
  const reduced = useRef(matchMedia('(prefers-reduced-motion: reduce)').matches);
  const initial = useRef(origin.getBoundingClientRect());
  const opened = useRef(false);

  const trackedVideo = useCallback<MakeVideo>(
    canvas => {
      // GatewayVideo sizes the visible canvas only when it transfers a decoded bitmap. A seeked event can
      // arrive earlier, while its worker still decodes. Start at a sentinel size and keep the poster until paint.
      canvas.width = 1;
      canvas.height = 1;
      let painted = false;
      const observer = new MutationObserver(records => {
        if (records.some(record => record.attributeName === 'width' || record.attributeName === 'height')) {
          painted = canvas.width > 1 || canvas.height > 1;
          // Past a fresh canvas's default 300 x 150, the painted size is the camera's true shape.
          const found = canvas.width > 300 ? ratioOf(canvas.width, canvas.height) : null;
          if (found && mounted.current) {
            learnRatio(cam, found);
            setRatio(previous => (Math.abs(previous - found) < 0.01 ? previous : found));
          }
        }
        if (painted && !canvas.hasAttribute('data-standby') && mounted.current) setReady(true);
      });
      observer.observe(canvas, { attributes: true, attributeFilter: ['width', 'height', 'data-standby'] });
      const media = makeVideo(canvas);
      const close = media.close.bind(media);
      media.close = () => {
        observer.disconnect();
        close();
      };
      return media;
    },
    [makeVideo, cam],
  );

  useLayoutEffect(() => {
    const el = panel;
    if (!el || opened.current) return;
    opened.current = true;
    void motion(el, initial.current, bounds, reduced.current);
    // FLIP keeps the layout fixed; only the window transform, crop, and image compensation move.
    if (!reduced.current && content.current?.animate) {
      const image = content.current.animate(
        [{ transform: contentFrame(initial.current, bounds) }, { transform: 'none' }],
        { duration: DURATION, easing: 'cubic-bezier(.2,.75,.2,1)', fill: 'both' },
      );
      void image.finished.then(() => image.cancel()).catch(() => {});
    }
    el.dataset.positioned = 'true';
  }, [bounds, panel]);

  useEffect(() => {
    mounted.current = true;
    const resize = () => setView({ width: innerWidth, height: innerHeight });
    window.addEventListener('resize', resize);
    return () => {
      mounted.current = false;
      window.removeEventListener('resize', resize);
    };
  }, []);

  useEffect(
    () => () => {
      if (returnFocus.current && origin.isConnected) origin.focus({ preventScroll: true });
    },
    [origin],
  );

  // Listing, connection, seek acknowledgement, and decoding all precede the first bitmap. A successful seek
  // alone must not dismiss this state. Keep the current attempt alive so a late frame can still recover it.
  useEffect(() => {
    if (ready || error || unplayable || closing || slow) return;
    const timer = setTimeout(() => setSlow(true), FIRST_FRAME_WAIT_MS);
    return () => clearTimeout(timer);
  }, [ready, error, unplayable, closing, slow]);

  const retry = () => {
    setReady(false);
    setSlow(false);
    setAttempt(value => value + 1);
  };

  const here = (): Moment => ({ t: engine.t ?? moment?.t ?? start, play: engine.playing });

  const close = () => {
    if (closing) return;
    const last = here();
    setClosing(true);
    cancel();
    engine.pause();
    const el = panel;
    if (!el) {
      onClose(last);
      return;
    }
    const target = origin.isConnected ? origin.getBoundingClientRect() : initial.current;
    if (!reduced.current && content.current?.animate) {
      content.current.animate(
        [{ transform: getComputedStyle(content.current).transform }, { transform: contentFrame(target, bounds) }],
        { duration: DURATION, easing: 'cubic-bezier(.2,.75,.2,1)', fill: 'forwards' },
      );
    }
    void motion(el, el.getBoundingClientRect(), target, reduced.current, bounds, true).then(() => {
      if (mounted.current) onClose(last);
    });
  };

  // Escape closes it wherever focus is: a click on the video leaves focus on the page, not in the window.
  const closeRef = useRef(close);
  closeRef.current = close;
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      event.preventDefault();
      event.stopPropagation();
      closeRef.current();
    };
    window.addEventListener('keydown', key, true);
    return () => window.removeEventListener('keydown', key, true);
  }, []);

  const reviewAt = (at: number) => reviewURL({ person, cam, day: isoDay(new Date(at * 1000)), t: at, play: true });
  return (
    <Portal>
      <div className={classes.backdrop} style={{ zIndex: FLOAT_Z }} onClick={close} aria-hidden="true" />
      <FocusTrap active={!closing} innerRef={setPanel}>
        <div
          role="dialog"
          aria-modal="true"
          aria-label={label}
          className={classes.panel}
          style={{ ...bounds, zIndex: FLOAT_Z + 1 }}
          data-closing={closing || undefined}
        >
          <div
            ref={content}
            className={classes.content}
            style={{ width: bounds.width, '--clip-ratio': ratio } as CSSProperties}
          >
            <EngineContext value={engine}>
              <Player key={attempt} loading={loading} compact={!full} makeVideo={trackedVideo} />
            </EngineContext>
            {poster && !unplayable && (
              <SafeImage
                src={poster}
                alt=""
                className={classes.poster}
                data-ready={ready || undefined}
                style={{ opacity: ready && !closing ? 0 : 1 }}
              />
            )}
            {!ready && !error && !unplayable && (
              <div className={classes.loading} role="status">
                <Loader size="sm" />
                <Text size="sm">{slow ? 'Still loading video' : 'Loading video'}</Text>
                {slow && (
                  <Button size="xs" onClick={retry} aria-label="Retry video">
                    Retry
                  </Button>
                )}
              </div>
            )}
            {error && (
              <div className={classes.error}>
                <PageError title={error}>{null}</PageError>
              </div>
            )}
          </div>
          <div className={classes.actions}>
            <ActionIcon
              component="a"
              href={reviewAt(t ?? start)}
              variant="filled"
              size="lg"
              aria-label="Open in Review"
              onClick={event => {
                if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                event.preventDefault();
                returnFocus.current = false;
                const last = here();
                cancel();
                engine.stop();
                onClose(last);
                navigate(reviewAt(last.t));
              }}
            >
              <IconArrowUpRight size={20} />
            </ActionIcon>
            <ActionIcon variant="filled" size="lg" aria-label="Close video" onClick={close} data-autofocus>
              <IconX size={20} />
            </ActionIcon>
          </div>
        </div>
      </FocusTrap>
    </Portal>
  );
}
