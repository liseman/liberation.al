// Live's Enhance view: a toggle and a look picker over the large camera, and the transparent WebGL2 canvas that
// draws the highlight (renderer.ts) over the plain picture, which this layer steadies with a CSS translate. Off by
// default and remembered in this browser; off, nothing here runs: no canvas, no GPU context, no frame tap, and the
// plain picture is left as it is.
import { SegmentedControl, Tooltip, UnstyledButton } from '@mantine/core';
import { useReducedMotion } from '@mantine/hooks';
import { useEffect, useRef, useState } from 'react';
import { create } from 'zustand';
import { tapFrames, useLiveNow } from '../camera/mux';
import { setPref, usePrefs } from '../lib/prefs';
import { type EnhanceCanvas, Enhancer, type Look, TooSlow } from './renderer';
import './enhance.css';

/** Why this browser can't run Enhance, once it has tried; undefined until then. */
export const useEnhanceUnsupported = create<{ reason?: string }>(() => ({}));

const LOOKS: { value: Look; label: string; tip: string }[] = [
  { value: 'subtle', label: 'Subtle', tip: 'The picture as it is, with a soft glow where something changes' },
  { value: 'contrast', label: 'Contrast', tip: 'Everything still is dimmed, so small changes stand out' },
  { value: 'colour', label: 'Colour', tip: 'Shows tiny colour changes too faint to see' },
];
const CALM_TIP = 'Off because your system asks for reduced motion';
const STOPPED = 'it stopped with an error; reload to try again';

/** The look in use: Colour amplifies a pulse, which is motion a reduced-motion setting asks not to see. */
export function useLook(): Look {
  const look = usePrefs(s => s.enhanceLook) ?? 'subtle';
  const calm = useReducedMotion();
  return calm && look === 'colour' ? 'subtle' : look;
}

export function EnhanceButton() {
  const on = usePrefs(s => !!s.enhance);
  const unsupported = useEnhanceUnsupported(s => s.reason);
  const label = unsupported
    ? `Enhance isn't available in this browser: ${unsupported}`
    : 'Enhance: steady the picture and highlight what changes. Stays on in this browser until you turn it off.';
  return (
    <Tooltip label={label} events={{ hover: true, focus: true, touch: false }}>
      <UnstyledButton
        className={`flip-btn${on && !unsupported ? ' on' : ''}`}
        aria-pressed={on && !unsupported}
        aria-disabled={!!unsupported}
        onClick={() => {
          if (!unsupported) setPref('enhance', !on);
        }}
        aria-label={label}
      >
        Enhance
      </UnstyledButton>
    </Tooltip>
  );
}

/** Subtle (the picture as it is, a soft glow where something changes), Contrast (the rest dimmed and greyed so small
 *  changes stand out), or Colour (tiny colour changes amplified; not with reduced motion). */
export function EnhanceLook() {
  const look = useLook();
  const calm = useReducedMotion();
  return (
    <SegmentedControl
      className="enhance-look"
      size="xs"
      radius="xl"
      aria-label="Enhance look"
      value={look}
      onChange={value => setPref('enhanceLook', value as Look)}
      data={LOOKS.map(({ value, label, tip }) => {
        const off = !!calm && value === 'colour';
        return {
          value,
          disabled: off,
          label: (
            <Tooltip label={off ? CALM_TIP : tip} events={{ hover: true, focus: true, touch: false }}>
              <span>{label}</span>
            </Tooltip>
          ),
        };
      })}
    />
  );
}

/** The plain picture a layer sits over: the camera's canvas beside it (camera/LiveCam.tsx). */
function plainOf(layer: HTMLElement): HTMLCanvasElement | null {
  return layer.parentElement?.querySelector<HTMLCanvasElement>('canvas.live-canvas') ?? null;
}

/** Put the plain picture back as it was. */
function release(plain: HTMLCanvasElement | null): void {
  if (!plain) return;
  delete plain.dataset.enhance;
  plain.style.removeProperty('--enhance-x');
  plain.style.removeProperty('--enhance-y');
}

/** The highlight for cam, over its plain picture (which stays the video, and is what face and object picks read).
 *  While the highlight shows, the plain picture carries data-enhance (the look) and --enhance-x and --enhance-y
 *  (how far to move it, in percent of its box); enhance.css does the rest. */
export function EnhanceLayer({ cam }: { cam: string }) {
  const ref = useRef<EnhanceCanvas>(null);
  const enhancer = useRef<Enhancer | null>(null);
  const look = useLook();
  const calm = useReducedMotion();
  const live = useLiveNow(s => s[cam]);
  const wanted = useRef({ look, calm: !!calm });
  wanted.current = { look, calm: !!calm };
  // Whether this camera's enhanced picture has a frame to show. It is state, not a class set on the element: a class
  // set by hand is dropped the next time the class list is rendered (frames stopping and coming back, a reload onto
  // a camera whose frames haven't started), and the layer then drew every frame without ever showing one.
  const [drawn, setDrawn] = useState(false);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    let drawer: Enhancer;
    try {
      drawer = new Enhancer(canvas);
    } catch (error) {
      useEnhanceUnsupported.setState({ reason: error instanceof Error ? error.message : 'no WebGL2' });
      return;
    }
    drawer.setLook(wanted.current.look, wanted.current.calm);
    enhancer.current = drawer;
    let shown = false;
    // The layer's box, to turn a shift in fractions of the frame into percent of the box (they differ where the
    // picture is letterboxed)
    const box = { width: 0, height: 0 };
    const sized =
      typeof ResizeObserver === 'undefined'
        ? undefined
        : new ResizeObserver(([entry]) => {
            if (entry) Object.assign(box, { width: entry.contentRect.width, height: entry.contentRect.height });
          });
    sized?.observe(canvas);
    let plain: HTMLCanvasElement | null = null;
    const placed = { x: Number.NaN, y: Number.NaN };
    const place = () => {
      if (!plain?.isConnected) {
        plain = plainOf(canvas);
        placed.x = Number.NaN;
      }
      if (!plain) return;
      let kx = 1;
      let ky = 1;
      if (box.width > 0 && box.height > 0 && canvas.width > 0 && canvas.height > 0) {
        const across = Math.min(box.width, (box.height * canvas.width) / canvas.height);
        kx = across / box.width;
        ky = (across * canvas.height) / canvas.width / box.height;
      }
      const x = -100 * kx * drawer.shift.x;
      const y = -100 * ky * drawer.shift.y;
      if (x === placed.x && y === placed.y) return; // a still camera writes nothing
      placed.x = x;
      placed.y = y;
      plain.style.setProperty('--enhance-x', `${x.toFixed(4)}%`);
      plain.style.setProperty('--enhance-y', `${y.toFixed(4)}%`);
    };
    // Giving up: the plain picture underneath is the fallback, so this layer's last frame must not stay over it
    const giveUp = (reason: string) => {
      stop();
      setDrawn(false);
      useEnhanceUnsupported.setState({ reason });
    };
    const stop = tapFrames(cam, (bitmap, at, flip, small) => {
      try {
        drawer.draw(bitmap, at, flip, small);
      } catch (error) {
        giveUp(error instanceof TooSlow ? error.message : STOPPED);
        return;
      }
      place();
      if (!shown) {
        shown = true;
        setDrawn(true); // until a first frame, the plain picture shows through
      }
    });
    const lost = (e: Event) => {
      e.preventDefault();
      giveUp('the graphics context was lost; reload to try again');
    };
    canvas.addEventListener('webglcontextlost', lost);
    return () => {
      stop();
      sized?.disconnect();
      canvas.removeEventListener('webglcontextlost', lost);
      release(plain);
      drawer.dispose();
      enhancer.current = null;
      setDrawn(false);
    };
  }, [cam]);

  useEffect(() => {
    enhancer.current?.setLook(look, !!calm);
  }, [look, calm]);

  // The plain picture is moved and filtered only while the highlight shows over it
  useEffect(() => {
    const plain = ref.current && plainOf(ref.current);
    if (!plain || !drawn || live === false) return;
    plain.dataset.enhance = look;
    return () => {
      delete plain.dataset.enhance;
    };
  }, [drawn, live, look]);

  // Frames stopped: the plain picture underneath shows the last one, dimmed, as it does without Enhance
  return <canvas ref={ref} className={`enhance-canvas${drawn ? ' drawn' : ''}${live === false ? ' paused' : ''}`} />;
}
