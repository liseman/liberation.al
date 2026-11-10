// Review's time logic, with no DOM: which segment or line covers a moment, where a day opens, the visible window's
// limits, and the timeline's ticks and lanes. Times are Unix seconds; lists are sorted by start.
import type { TranscriptEntry } from '../live/types';
import type { AudioSpan, Segment, Span } from './types';

export const DAY_S = 86400;
export const MIN_SPAN = 60; // the closest zoom: one minute across
export const RULER = 20; // px of time ruler at the top of the timeline: dragging there pans
const STEPS = [1, 5, 10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, DAY_S, 2 * DAY_S];

export interface View {
  t0: number;
  t1: number;
}

/** Local midnight to the next local midnight, including 23/25-hour clock-change days. */
export function dayBounds(day: string): View {
  const start = new Date(`${day}T00:00:00`);
  const end = new Date(start);
  end.setDate(end.getDate() + 1);
  return { t0: start.getTime() / 1000, t1: end.getTime() / 1000 };
}

/** Local date boundaries within the viewport; advancing dates keeps midnight correct across DST. */
export function midnights(t0: number, t1: number): number[] {
  const d = new Date(t0 * 1000);
  d.setHours(0, 0, 0, 0);
  if (d.getTime() / 1000 < t0) d.setDate(d.getDate() + 1);
  const out: number[] = [];
  while (d.getTime() / 1000 < t1) {
    out.push(d.getTime() / 1000);
    d.setDate(d.getDate() + 1);
  }
  return out;
}

/** The segment in a camera's list that covers ts, or null. */
export function segAt<T extends { start: number; end: number }>(list: readonly T[], ts: number): T | null {
  let lo = 0;
  let hi = list.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((list[mid]?.start ?? 0) <= ts) lo = mid + 1;
    else hi = mid;
  }
  const s = list[lo - 1];
  return s && ts < s.end ? s : null;
}

/** Index of the last line that starts at or before ts (0.3 s slack), or -1. */
export function lineAt(entries: readonly TranscriptEntry[], ts: number): number {
  let lo = 0;
  let hi = entries.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((entries[mid]?.ts_start ?? 0) <= ts + 0.3) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

/** When a line ends: its own end, or its start when it has none. */
export const lineEnd = (e: TranscriptEntry): number => e.ts_end ?? e.ts_start;

/** A bounded zoom with free horizontal pan. Embedded day views may supply their local bound. */
export function clampView(t0: number, t1: number, dayStart?: number): View {
  const span = Math.min(dayStart === undefined ? 7 * DAY_S : DAY_S + 7200, Math.max(MIN_SPAN, t1 - t0));
  let a = span !== t1 - t0 ? (t0 + t1) / 2 - span / 2 : t0;
  if (dayStart !== undefined) {
    const lo = dayStart - 3600;
    const hi = dayStart + DAY_S + 3600;
    a = Math.max(lo, Math.min(a, hi - span));
  }
  return { t0: a, t1: a + span };
}

/** The earliest start and latest end across the day's lines, video, and audio, or null for an empty day. */
export function dataEdges(
  entries: readonly TranscriptEntry[],
  segs: readonly Segment[],
  spans: readonly AudioSpan[],
): [number, number] | null {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const e of entries) {
    lo = Math.min(lo, e.ts_start);
    hi = Math.max(hi, lineEnd(e));
  }
  for (const s of [...segs, ...spans]) {
    lo = Math.min(lo, s.start);
    hi = Math.max(hi, s.end);
  }
  return Number.isFinite(lo) ? [lo, hi] : null;
}

/** Where Review opens: the latest moment with at least minS seconds of video after it, so pressing play shows a full
 *  minute (opening at the very end of the data otherwise showed "No video"). A camera's newest segment counts while
 *  it's still growing: the gateway plays it to its end. Null if the day has no stretch of video that long. */
export function latestVideo(segs: Record<string, readonly Segment[]>, _now: number, minS = 60): number | null {
  const runs: Span[][] = []; // per camera: its unbroken stretches of finished video, each at least minS long
  for (const list of Object.values(segs)) {
    const rs: Span[] = [];
    list.forEach(s => {
      if (s.broken) return;
      const last = rs.at(-1);
      if (last && s.start - last[1] < 5) last[1] = Math.max(last[1], s.end);
      else rs.push([s.start, s.end]);
    });
    runs.push(rs.filter(r => r[1] - r[0] >= minS));
  }
  // Cameras are not in step (a restart shifts one), so each camera's own latest minute is a candidate, and the one
  // the most cameras can play a full minute from wins; among those, the latest.
  const covers = (t: number) => runs.filter(rs => rs.some(r => r[0] <= t && t + minS <= r[1])).length;
  let best: number | null = null;
  let bestN = 0;
  for (const rs of runs) {
    const r = rs.at(-1);
    if (!r) continue;
    const t = r[1] - minS;
    const n = covers(t);
    if (n > bestN || (n === bestN && best != null && t > best)) {
      best = t;
      bestN = n;
    }
  }
  return best;
}

/** The next moment after ts with any video, speech, or kept room audio (spans), for skipping gaps while playing. */
export function nextMedia(
  segs: readonly Segment[],
  entries: readonly TranscriptEntry[],
  ts: number,
  spans: readonly AudioSpan[] = [],
): number | null {
  let best: number | null = null;
  for (const s of [...segs, ...spans]) if (s.start > ts + 0.05 && (best == null || s.start < best)) best = s.start;
  const next = entries[lineAt(entries, ts) + 1];
  if (next && (best == null || next.ts_start < best)) best = next.ts_start;
  return best;
}

/** Index of the first item that could reach ts: items are sorted by start and none lasts over an hour. */
export function firstVisible(items: readonly { ts_start: number }[], ts: number): number {
  let lo = 0;
  let hi = items.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((items[mid]?.ts_start ?? 0) < ts - 3600) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** The timeline's tick steps for a window span seconds across and width px wide: labels at least 70 px apart, and
 *  minor ticks between them when they can be 8 px apart. */
export function tickSteps(span: number, width: number): { major: number; minor: number | null } {
  const major = STEPS.find(s => (s / span) * width >= 70) ?? STEPS[STEPS.length - 1] ?? 21600;
  const minor = STEPS.filter(s => s < major && major % s === 0 && (s / span) * width >= 8).pop() ?? null;
  return { major, minor };
}

/** The tick times from t0 to t1 every step seconds, on local clock time (a 1-hour step lands on the hour). */
export function ticks(t0: number, t1: number, step: number): number[] {
  if (step >= DAY_S)
    return midnights(t0, t1).filter(ts => {
      const d = new Date(ts * 1000);
      return (Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) / 1000 / DAY_S) % Math.round(step / DAY_S) === 0;
    });
  const tz = new Date(t0 * 1000).getTimezoneOffset() * 60;
  const out: number[] = [];
  for (let s = Math.ceil((t0 - tz) / step) * step + tz; s < t1; s += step) out.push(s);
  return out;
}

/** A tick's label: HH:MM, with seconds when the step is under a minute. */
export function tickLabel(ts: number, step: number): string {
  const d = new Date(ts * 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return step < 60 ? `${hm}:${pad(d.getSeconds())}` : hm;
}

export interface Lanes {
  marks: [number, number];
  speech: [number, number];
  cams: Record<string, [number, number]>;
  audio: [number, number];
  height: number;
}

/** The timeline's lanes as [y, height], top to bottom: ruler, the selected chips (always there, so picking chips
 *  never changes the height), speech, one thin lane per camera, audio. */
export function lanes(cams: readonly string[]): Lanes {
  const out: Lanes = {
    marks: [RULER + 4, 8],
    speech: [RULER + 16, 20],
    cams: Object.create(null),
    audio: [0, 4],
    height: 0,
  };
  let y = RULER + 42;
  for (const c of cams) {
    out.cams[c] = [y, 5];
    y += 8;
  }
  out.audio = [y, 4];
  out.height = y + 10;
  return out;
}

/** The line under px on a timeline, within 3 px, from those near ts. x maps a time to px. */
export function lineNear(
  entries: readonly TranscriptEntry[],
  ts: number,
  px: number,
  x: (t: number) => number,
): TranscriptEntry | null {
  let hit: TranscriptEntry | null = null;
  for (let i = firstVisible(entries, ts - 2); i < entries.length; i++) {
    const e = entries[i];
    if (!e || e.ts_start > ts + 2) break;
    const a = x(e.ts_start);
    const b = x(lineEnd(e));
    if (px >= a - 3 && px <= Math.max(b, a + 2) + 3) hit = e;
  }
  return hit;
}
