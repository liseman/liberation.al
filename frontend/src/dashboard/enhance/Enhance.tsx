// Live's Enhance view: a toggle and a look picker over the large camera, and the WebGL2 canvas that draws the
// stabilised picture with change highlighted (renderer.ts) on top of the plain one. Off by default and remembered in
// this browser; off, nothing here runs: no canvas, no GPU context, no frame tap.
import { SegmentedControl, Tooltip, UnstyledButton } from '@mantine/core';
import { useReducedMotion } from '@mantine/hooks';
import { useEffect, useRef, useState } from 'react';
import { create } from 'zustand';
import { tapFrames, useLiveNow } from '../camera/mux';
import { setPref, usePrefs } from '../lib/prefs';
import { type EnhanceCanvas, Enhancer, type Look } from './renderer';
import './enhance.css';

/** Why this browser can't run Enhance, once it has tried; undefined until then. */
export const useEnhanceUnsupported = create<{ reason?: string }>(() => ({}));

const LOOKS: { value: Look; label: string; tip: string }[] = [
  { value: 'subtle', label: 'Subtle', tip: 'The picture as it is, with a soft glow where something changes' },
  { value: 'contrast', label: 'Contrast', tip: 'Everything still is dimmed, so small changes stand out' },
  { value: 'colour', label: 'Colour', tip: 'Shows tiny colour changes too faint to see' },
];
const CALM_TIP = 'Off because your system asks for reduced motion';

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

/** The enhanced picture of cam, over its plain one (which stays underneath for face and object picks). */
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
    // Giving up: the plain picture underneath is the fallback, so this layer's last frame must not stay over it
    const giveUp = (reason: string) => {
      stop();
      setDrawn(false);
      useEnhanceUnsupported.setState({ reason });
    };
    const stop = tapFrames(cam, (bitmap, at, flip) => {
      try {
        drawer.draw(bitmap, at, flip);
      } catch {
        giveUp('it stopped with an error; reload to try again');
        return;
      }
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
      canvas.removeEventListener('webglcontextlost', lost);
      drawer.dispose();
      enhancer.current = null;
      setDrawn(false);
    };
  }, [cam]);

  useEffect(() => {
    enhancer.current?.setLook(look, !!calm);
  }, [look, calm]);

  // Frames stopped: the plain picture underneath shows the last one, dimmed, as it does without Enhance
  return <canvas ref={ref} className={`enhance-canvas${drawn ? ' drawn' : ''}${live === false ? ' paused' : ''}`} />;
}
