// A live wave of what the microphones hear, full width behind the Live tab. Its height follows the current level
// from the /audio/levels stream (the loudest microphone), rising fast and falling slower; layered lines at different
// speeds keep it moving, and the peak shows as a fainter outer wave. It turns orange while the VAD hears speech.
import { useEffect, useRef } from 'react';
import { subscribeLevels } from '../sources/levels';
import './camera.css';

const GATE = 6; // dB above the room's noise floor before the wave moves at all
const SPAN = 30; // dB above that gate drawn full height
const FLOOR_RISE = 0.5; // dB a second the floor estimate creeps up through steady noise (it drops at once)
const ATTACK = 0.45; // share of the gap to a louder level closed each frame
const RELEASE = 0.06; // ... to a quieter level
const MIN = 0.004; // nearly flat in silence

const TINT = '#9db8ff'; // fallbacks for --wave-tint and --wave-speech (camera.css)
const SPEECH = '#ff9a3c';

/** Blend two #rrggbb colours, t from 0 (a) to 1 (b). */
export function mix(a: string, b: string, t: number): string {
  const hex = /^#[0-9a-f]{6}$/i;
  if (!hex.test(a) || !hex.test(b)) return t > 0.5 ? b : a;
  const p = (c: string) => [1, 3, 5].map(i => Number.parseInt(c.slice(i, i + 2), 16));
  const y = p(b);
  return `rgb(${p(a)
    .map((v, i) => Math.round(v + ((y[i] ?? v) - v) * t))
    .join(',')})`;
}

/** Height above the noise floor, curved so small sounds stay small and speech still fills it. */
export const level = (db: number, floor: number): number => Math.max(0, Math.min(1, (db - floor - GATE) / SPAN)) ** 1.6;
const follow = (cur: number, target: number) => cur + (target - cur) * (target > cur ? ATTACK : RELEASE);

// [cycles across the width, speed in radians a second, share of the height, opacity]
const LINES = [
  [1.5, 1.1, 1.0, 0.9],
  [2.5, -1.7, 0.7, 0.55],
  [4.0, 2.6, 0.45, 0.35],
] as const;

export function Waveform() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || typeof EventSource === 'undefined') return;
    // The theme's accent, so it reads in light and dark; strength comes from CSS opacity.
    const color = (name: string, fallback: string) =>
      getComputedStyle(canvas).getPropertyValue(name).trim() || fallback;
    let rmsT = 0; // targets from the stream
    let peakT = 0;
    let speaking = false;
    let floor: number | null = null; // the room's noise floor in dBFS, learned as levels arrive
    let floorAt = performance.now() / 1000;
    let warm = 0; // 0 to 1, eases toward speaking so the colour fades rather than flips
    let rms = MIN; // what's drawn, eased toward the targets
    let peak = MIN;
    const off = subscribeLevels(levels => {
      let r = -120;
      let p = -120;
      speaking = false;
      for (const v of Object.values(levels)) {
        r = Math.max(r, v.rms);
        p = Math.max(p, v.peak);
        speaking ||= !!v.speech;
      }
      const now = performance.now() / 1000;
      floor = floor === null || r < floor ? r : Math.min(r, floor + FLOOR_RISE * (now - floorAt));
      floorAt = now;
      rmsT = level(r, floor);
      peakT = level(p, floor);
    });
    const wave = (w: number, mid: number, amp: number, t: number, cycles: number, speed: number, phase: number) => {
      ctx.beginPath();
      for (let x = 0; x <= w; x += 4) {
        const u = x / w;
        const taper = Math.sin(Math.PI * u); // calm at the edges, full in the middle
        const y = mid + amp * taper * Math.sin(u * cycles * 2 * Math.PI + t * speed + phase);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    const draw = (now: number) => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      rms = Math.max(MIN, follow(rms, rmsT));
      peak = Math.max(rms, follow(peak, peakT));
      const t = now / 1000;
      const mid = h / 2;
      const full = h * 0.42;
      warm += ((speaking ? 1 : 0) - warm) * 0.15;
      ctx.strokeStyle = mix(color('--wave-tint', TINT), color('--wave-speech', SPEECH), warm);
      ctx.lineCap = 'round';
      LINES.forEach(([cycles, speed, share, alpha], i) => {
        ctx.globalAlpha = alpha * 0.35;
        ctx.lineWidth = 1.5;
        wave(w, mid, full * peak * share, t, cycles, speed, i * 1.7);
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 3;
        wave(w, mid, full * rms * share, t, cycles, speed, i * 1.7);
      });
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(draw);
    };
    let raf = requestAnimationFrame(draw);
    return () => {
      off();
      cancelAnimationFrame(raf);
    };
  }, []);
  return <canvas className="waveform" ref={ref} aria-hidden />;
}
