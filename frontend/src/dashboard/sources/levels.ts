// Live microphone meters. GET /audio/levels streams {"ts", "levels": {slug: {"rms", "peak"}}} in dBFS about 25 times
// a second while the microphone service runs.
import { type RefObject, useEffect } from 'react';

export const FLOOR = -90; // dBFS shown as an empty meter
export const FALL_DB_S = 20; // how fast a held peak falls
// Meters move at most this often: every other frame at 60 Hz, about the stream's 25 a second. Any change in a frame
// makes the browser rebuild the page's compositing layers, so moving them every frame cost twice as much for nothing
// new to show.
export const METER_MS = 32;
const NO_PEAK = -120;

/** A level in dBFS as a meter width: FLOOR is 0%, 0 dBFS is 100%, clamped in between. */
export const levelWidth = (db: number): string =>
  `${Math.max(0, Math.min(100, ((db - FLOOR) / -FLOOR) * 100)).toFixed(1)}%`;

/** A level as a share of the meter, FLOOR 0 to 0 dBFS 1, to three places. */
const share = (db: number): number => +Math.max(0, Math.min(1, (db - FLOOR) / -FLOOR)).toFixed(3);

/** The live bar's transform: a full-width bar scaled to the level. */
export const barTransform = (db: number): string => `scaleX(${share(db)})`;

/** The held peak mark's transform: a full-width box moved along by the level, its mark drawn at its left edge. */
export const markTransform = (db: number): string => `translateX(${+(share(db) * 100).toFixed(1)}%)`;

/** The held peak after dt seconds: the new peak, or the old one fallen by FALL_DB_S a second, whichever is higher. */
export const holdPeak = (peak: number, held: number | undefined, dt: number): number =>
  Math.max(peak, (held ?? NO_PEAK) - FALL_DB_S * dt);

export interface Level {
  rms: number;
  peak: number;
  speech?: boolean; // listen's fresh energy-gate state; absent when unavailable
}

type Listener = (levels: Record<string, Level>) => void;

const CLOSED = 2; // EventSource.CLOSED, without needing EventSource to exist (tests, old browsers)

// One /audio/levels connection for every meter on the page (Sources, Live's mic chips, the waveform): a page can only
// hold a few connections to the dashboard open. It opens with the first listener and closes with the last.
const listeners = new Set<Listener>();
let stream: EventSource | null = null;

/** Call listener with each update's levels ({} when the stream closes); returns a function that stops. */
export function subscribeLevels(listener: Listener): () => void {
  if (stream && stream.readyState === CLOSED) stream = null; // given up on: the next listener opens a new one
  listeners.add(listener);
  if (!stream && typeof EventSource !== 'undefined') {
    const es = new EventSource('/audio/levels');
    stream = es;
    es.onmessage = e => {
      let d: { levels?: Record<string, Level> };
      try {
        d = JSON.parse(e.data) as { levels?: Record<string, Level> };
      } catch {
        return; // a broken event: keep the last levels
      }
      if (d.levels) for (const l of listeners) l(d.levels as Record<string, Level>);
    };
    es.onerror = () => {
      if (es.readyState !== CLOSED) return; // the browser reconnects by itself
      if (stream === es) stream = null;
      for (const l of listeners) l({});
    };
  }
  return () => {
    listeners.delete(listener);
    if (!listeners.size && stream) {
      stream.close();
      stream = null;
    }
  };
}

/**
 * While mounted, stream /audio/levels and move the meters with [data-level-key] inside root directly, not through
 * React (25 updates a second would re-render the whole panel). A meter's first child is the level bar and its last
 * the peak mark; data-live marks it as streaming, and a meter the stream stops reporting (it closed, or the meter's
 * key went) goes back to its status level.
 *
 * They move by transform, at most every METER_MS, written only when it changes: a width or left set every frame laid
 * out and repainted the whole page every frame, over half of Live's main-thread time. The CSS for [data-live]
 * makes both full width.
 */
export function useLiveLevels(root: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    let latest: Record<string, Level> = {};
    const peaks: Record<string, number> = {};
    let last = performance.now();
    const live = new Set<HTMLElement>();
    const still = (el: HTMLElement) => {
      // back to the status level: the CSS shows --level again without data-live
      el.removeAttribute('data-live');
      for (const part of [el.firstElementChild, el.lastElementChild] as (HTMLElement | null)[])
        part?.style.removeProperty('transform');
      live.delete(el);
    };
    const off = subscribeLevels(levels => {
      latest = levels;
    });
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      if (now - last < METER_MS) return;
      const dt = (now - last) / 1000;
      last = now;
      const shown = new Set<HTMLElement>();
      for (const el of root.current?.querySelectorAll<HTMLElement>('[data-level-key]') ?? []) {
        const key = el.dataset.levelKey ?? '';
        const v = latest[key];
        if (!v) continue;
        shown.add(el);
        const hold = holdPeak(v.peak, peaks[key], dt);
        peaks[key] = hold;
        const bar = el.firstElementChild as HTMLElement | null;
        const mark = el.lastElementChild as HTMLElement | null;
        const b = barTransform(v.rms);
        const m = markTransform(hold);
        if (bar && bar.style.transform !== b) bar.style.transform = b;
        if (mark && mark.style.transform !== m) mark.style.transform = m;
        if (el.dataset.live == null) el.dataset.live = '';
        live.add(el);
      }
      for (const el of live) if (!shown.has(el)) still(el);
      for (const key of Object.keys(peaks)) if (!latest[key]) delete peaks[key];
    };
    let raf = requestAnimationFrame(frame);
    return () => {
      off();
      cancelAnimationFrame(raf);
    };
  }, [root]);
}
