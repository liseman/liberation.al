// The Review timeline: a compact combined strip by default (#2111), expandable into labelled lanes (R3, #328):
// "Who spoke" (speech
// coloured per person), one lane per camera angle (recorded video in grey, a bar per person in view), the audio kept,
// and Events as tick marks, under hour labels, with a playhead you can drag. People picked in Show only stay at full
// strength and everyone else dims; picked things and events mark their time. The canvas redraws only when the view
// or data changes; the playhead moves by transform every animation frame, so scrubbing stays smooth.
import { Button, Group, Skeleton, Text } from '@mantine/core';
import { type KeyboardEvent, type ReactNode, useCallback, useEffect, useMemo, useRef } from 'react';
import { daysWords, hms, isoDay } from '../lib/format';
import { setPref, usePrefs } from '../lib/prefs';
import { EMPTY_SPEAKERS, shownName, speakerName } from '../lib/speakers';
import { useLive } from '../live/hooks';
import { chipColor } from './chipStyle';
import { useEngine, useReview } from './context';
import {
  camAt,
  cameraChips,
  compactLayout,
  eventTicks,
  hitAt,
  hitTitle,
  layout,
  markBands,
  type PresenceChip,
  personColor,
  pickedPeople,
  presenceRows,
  type Said,
  shownPerson,
  UNKNOWN_COLOR,
  UNKNOWN_FACES,
} from './lanes';
import classes from './review.module.css';
import so from './showOnly.module.css';
import {
  DAY_S,
  dayBounds,
  firstVisible,
  lineEnd,
  lineNear,
  midnights,
  RULER,
  tickLabel,
  tickSteps,
  ticks,
} from './time';
import type { Chip, EventSpan } from './types';
import { viewName } from './views';
import { WheelShuttle } from './wheelShuttle';

const DIMMED = 0.18; // people not picked in Show only, while someone is

const COLORS = ['line', 'dim', 'mute', 'acc', 'bg', 'bg2', 'ink', 'warn'] as const;
type Colors = Record<(typeof COLORS)[number], string>;

/** The days in view still loading or failed, on one line that keeps its height when empty, so days arriving one by
 *  one never move what is under the timeline (#2435): "Loading 2026-11-05 and 2026-11-04." */
export function ChunkLine({
  chunkStatus,
  range,
  notes,
}: {
  chunkStatus: Record<string, string>;
  range: { t0: number; t1: number } | null | undefined;
  notes?: ReactNode; // the page's own notes about the timeline (markers truncated or unavailable), on the same line
}) {
  const shown = Object.entries(chunkStatus).filter(
    ([day, status]) => status !== 'ready' && (!range || (dayBounds(day).t1 > range.t0 && dayBounds(day).t0 < range.t1)),
  );
  const failed = shown.filter(([, status]) => status === 'error').map(([day]) => day);
  const loading = shown.filter(([, status]) => status !== 'error').map(([day]) => day);
  return (
    <Text
      size="xs"
      c="dimmed"
      role="status"
      className={classes.chunkLine}
      title={
        [
          loading.length ? `Loading ${daysWords(loading)}.` : '',
          failed.length ? `Could not load ${daysWords(failed)}.` : '',
        ]
          .filter(Boolean)
          .join(' ') || undefined
      }
    >
      {loading.length > 0 && `Loading ${daysWords(loading)}.`}
      {loading.length > 0 && failed.length > 0 && ' '}
      {failed.length > 0 && (
        <Text span inherit c="red">
          Could not load {daysWords(failed)}.
        </Text>
      )}
      {(loading.length > 0 || failed.length > 0) && notes ? ' ' : null}
      {notes}
    </Text>
  );
}

/** loading: the day is loading, so a placeholder covers the track (its canvas stays mounted). marks: moments to
 *  point out on the ruler (Ask's cited moments, R4). */
const NO_MARKS: readonly number[] = []; // one default, so a render alone never counts as new marks

export type { EventSpan } from './types';

const MIN_SPAN_S = 1; // a shorter drag is taken as a slip, not a span

/** onSpan: a span picked with the explicit Mark an event pointer tool. I/O marks live in the engine. notes: the
 *  page's notes about the timeline, shown on its status line, which keeps its height (#2435). */
export function Timeline({
  loading = false,
  marks = NO_MARKS,
  onSpan,
  notes,
}: {
  loading?: boolean;
  marks?: readonly number[];
  onSpan?: (s: EventSpan) => void;
  notes?: ReactNode;
} = {}) {
  const engine = useEngine();
  const mode = usePrefs(s => (s.reviewTimeline === 'expanded' ? 'expanded' : 'compact'));
  const compact = mode === 'compact';
  const cameras = useReview(s => s.cameras);
  const cams = useMemo(
    () => (cameras.length ? engine.cams.filter(c => cameras.includes(c)) : engine.cams),
    [engine, cameras],
  );
  const t = useReview(s => s.t);
  const range = useReview(s => s.range);
  const chunkStatus = useReview(s => s.chunkStatus);
  const eventMarks = useReview(s => s.eventMarks);
  const span = useMemo(
    () =>
      eventMarks
        ? {
            cam: eventMarks.cam,
            start: eventMarks.in ?? eventMarks.out ?? 0,
            end: eventMarks.out ?? eventMarks.in ?? 0,
          }
        : null,
    [eventMarks],
  );
  const sel = useReview(s => s.sel); // changes whenever the day's chips arrive, too (setChips sets it)
  const colors = useReview(s => s.colors);
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  const speakers = useLive('speakers') ?? EMPTY_SPEAKERS;
  const speakersRef = useRef(speakers);
  const redraw = useRef<() => void>(() => {});
  // Mark an event (#287): the next drag in a camera's lane picks a span instead of scrubbing. The control bar's Mark
  // turns it on (#2454), so it lives in the engine's store.
  const marking = useReview(s => s.markingTool);
  const markingRef = useRef(false);
  markingRef.current = marking;
  const onSpanRef = useRef(onSpan);
  onSpanRef.current = onSpan;
  /** The day's lines as hit testing needs them: when, and who said each. */
  const said = useCallback(
    (): Said[] =>
      engine.entries.map(e => ({ ts_start: e.ts_start, ts_end: e.ts_end, who: speakerName(speakersRef.current, e) })),
    [engine],
  );
  const allChips = engine.chips;
  const chips = useMemo(() => cameraChips(allChips, cameras) as PresenceChip[], [allChips, cameras]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: sel stands for engine.chips, which setChips replaces with it
  const L = useMemo(() => {
    const colorOf = (c: Chip) => (c.id === UNKNOWN_FACES ? UNKNOWN_COLOR : personColor(c.label, speakers));
    const rows = presenceRows(chips, cams, colorOf);
    return compact ? compactLayout(cams, rows) : layout(cams, rows, cam => viewName(cam));
  }, [cams, chips, sel, speakers, compact]);
  const picked = useMemo(() => pickedPeople(chips, sel), [chips, sel]);
  const ticksAt = useMemo(
    () =>
      eventTicks(
        chips,
        c => colors[c.id] ?? chipColor(c.id),
        item =>
          item.person_name
            ? personColor(item.person_name, speakers)
            : chipColor(`track:${item.camera}:${item.track_id ?? item.id}`),
      ),
    [chips, colors, speakers],
  );
  const bands = useMemo(
    () =>
      markBands(chips, compact ? new Set(chips.filter(c => c.kind !== 'event').map(c => c.id)) : sel, c =>
        c.id === UNKNOWN_FACES
          ? UNKNOWN_COLOR
          : c.kind === 'person'
            ? personColor(c.label, speakers)
            : (colors[c.id] ?? chipColor(c.id)),
      ),
    [chips, sel, colors, speakers, compact],
  );
  // What the canvas reads, through a ref: the drawing and pointer setup below runs once per engine, so a new layout
  // (chips arriving, speakers changing) never tears down a drag in progress.
  const shown = useRef({ L, picked, ticksAt, marks, bands, span, eventMarks });
  shown.current = { L, picked, ticksAt, marks, bands, span, eventMarks };

  useEffect(() => {
    speakersRef.current = speakers;
    redraw.current();
  }, [speakers]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: redraw when any of what the canvas reads through shown changes
  useEffect(() => redraw.current(), [L, picked, ticksAt, marks, bands, span]);

  useEffect(() => {
    const el = box.current;
    const cv = canvas.current;
    const ph = head.current;
    const tp = tip.current;
    if (!el || !cv || !ph || !tp) return;
    const E = engine;
    const colors = Object.fromEntries(COLORS.map(k => [k, ''])) as Colors;
    let dirty = true;
    let W = 0;
    let font = 'sans-serif';

    const readColors = () => {
      const st = getComputedStyle(el);
      for (const k of COLORS) colors[k] = st.getPropertyValue(`--rv-${k}`).trim();
      font = st.getPropertyValue('--mantine-font-family').trim() || font;
      dirty = true;
    };
    const x = (ts: number) => ((ts - E.view.t0) / (E.view.t1 - E.view.t0)) * W;
    const tsAt = (px: number) => E.view.t0 + (px / W) * (E.view.t1 - E.view.t0);

    function draw(g: CanvasRenderingContext2D, H: number) {
      const { t0, t1 } = E.view;
      // One continuous strip. Loading and failed chunks stay visible as unknown intervals.
      g.fillStyle = colors.bg2;
      g.fillRect(0, RULER, W, H - RULER);
      for (const [day, status] of Object.entries(E.ui.getState().chunkStatus)) {
        if (status === 'ready') continue;
        const { t0: a, t1: b } = dayBounds(day);
        if (a >= t1 || b <= t0) continue;
        g.fillStyle = status === 'error' ? colors.warn : colors.line;
        g.globalAlpha = 0.25;
        g.fillRect(x(a), RULER, x(b) - x(a), H - RULER);
        g.globalAlpha = 1;
      }
      // ticks
      const { major, minor } = tickSteps(t1 - t0, W);
      g.font = `11px ${font}`;
      g.textBaseline = 'top';
      if (minor) {
        g.fillStyle = colors.line;
        for (const s of ticks(t0, t1, minor)) g.fillRect(Math.round(x(s)), RULER - 4, 1, 4);
      }
      for (const s of ticks(t0, t1, major)) {
        const px = Math.round(x(s));
        g.fillStyle = colors.line;
        g.fillRect(px, RULER - 8, 1, H - RULER + 8);
        g.fillStyle = colors.dim;
        const d = new Date(s * 1000);
        if (d.getHours() || d.getMinutes() || d.getSeconds()) g.fillText(tickLabel(s, major), px + 3, 2);
      }
      for (const at of midnights(t0, t1)) {
        const px = Math.round(x(at));
        g.fillStyle = colors.dim;
        g.fillRect(px, RULER, 1, H - RULER);
        g.fillText(new Date(at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }), px + 3, 2);
      }
      const { L, picked, ticksAt, marks, bands } = shown.current;
      // Who spoke: speech coloured per person; separate lines show as their own blocks
      const [sy, sh] = L.speech;
      const lines = E.entries;
      const spk = speakersRef.current;
      const colorOf = new Map<string, string>();
      for (let i = firstVisible(lines, t0); i < lines.length; i++) {
        const e = lines[i];
        if (!e || e.ts_start > t1) break;
        if (lineEnd(e) < t0) continue;
        const a = x(e.ts_start);
        const w = Math.max(2, x(Math.max(lineEnd(e), e.ts_start + 0.2)) - a);
        const who = speakerName(spk, e);
        const c = colorOf.get(who) ?? personColor(who, spk);
        colorOf.set(who, c);
        g.fillStyle = c;
        g.globalAlpha = shownPerson(who, picked) ? 1 : DIMMED;
        g.fillRect(a, sy, w > 6 ? w - 1 : w, sh);
      }
      g.globalAlpha = 1;
      // a lane per camera angle: recorded video in grey, segment edges once they're wide enough to tell apart, and a
      // presence bar per person on top
      for (const lane of L.cams) {
        const c = lane.cam;
        const [vy, vh] = lane.video;
        g.fillStyle = colors.mute;
        g.globalAlpha = 0.35;
        for (const s of E.segs[c] ?? []) {
          if (s.end < t0 || s.start > t1) continue;
          const a = x(s.start);
          const w = Math.max(1, x(s.end) - a);
          if (s.broken) {
            g.fillStyle = colors.warn; // still recording, or cut off: shown but not playable
            g.fillRect(a, vy + 1, w, vh - 2);
            g.fillStyle = colors.mute;
            continue;
          }
          g.fillRect(a, vy, w, vh);
          if (w > 40) {
            g.globalAlpha = 0.6;
            g.fillStyle = colors.ink;
            g.fillRect(Math.round(a), vy - 2, 1, vh + 4);
            g.fillStyle = colors.mute;
            g.globalAlpha = 0.35;
          }
        }
        for (const { row, band } of lane.rows) {
          g.fillStyle = row.color;
          g.globalAlpha = row.id === 'others' || shownPerson(row.label, picked) ? 0.95 : DIMMED;
          for (const [a0, b0] of row.spans) {
            if (b0 < t0 || a0 > t1) continue;
            g.fillRect(x(a0), band[0], Math.max(2, x(b0) - x(a0)), band[1]);
          }
        }
        g.globalAlpha = 1;
      }
      const [ay, ah] = L.audio;
      g.fillStyle = colors.acc;
      g.globalAlpha = 0.55;
      for (const s of E.spans)
        if (s.end >= t0 && s.start <= t1) g.fillRect(x(s.start), ay, Math.max(1, x(s.end) - x(s.start)), ah);
      g.globalAlpha = 1;
      // Events: subtle duration markers for activities, legacy ticks for owner events.
      const [ey, eh] = L.events;
      for (const k of ticksAt) {
        if ((k.end ?? k.t) < t0 || k.t > t1) continue;
        g.fillStyle = k.color;
        if (k.activity) {
          const selectedEvents = E.chips.some(c => c.kind === 'event' && E.ui.getState().sel.has(c.id));
          g.globalAlpha = selectedEvents && k.chipId && !E.ui.getState().sel.has(k.chipId) ? 0.12 : 0.55;
          const left = Math.max(0, Math.round(x(k.t)));
          const width = Math.max(5, Math.min(W, x(k.end ?? k.t)) - left);
          g.strokeStyle = k.color;
          g.setLineDash(k.activity.promoted ? [] : [2, 2]);
          g.strokeRect(left, ey + 1, width, eh - 2);
          if (!k.activity.promoted && width >= 8) {
            g.fillText('?', left + 2, ey + eh - 2);
          }
          g.setLineDash([]);
          g.globalAlpha = 1;
        } else g.fillRect(Math.round(x(k.t)) - 1, ey, 2, eh);
      }
      // the things and events picked in Show only: each its own colour, a band in the marks lane and a faint wash
      // down the lanes (picked people show through their own speech and presence instead)
      const things = bands;
      if (things.length) {
        const [my, mh] = L.marks;
        const rh = Math.max(2, Math.floor(mh / things.length));
        things.forEach((m, i) => {
          g.fillStyle = m.color;
          for (const [a0, b0] of m.spans) {
            if (b0 < t0 || a0 > t1) continue;
            const a = x(a0);
            const w = Math.max(2, x(b0) - a);
            g.globalAlpha = 0.9;
            g.fillRect(a, my + Math.min(i * rh, mh - rh), w, rh);
            g.globalAlpha = 0.1;
            g.fillRect(a, RULER, w, H - RULER);
          }
        });
        g.globalAlpha = 1;
      }
      // moments pointed out (Ask's citations, R4): a notch on the ruler and a faint line down the lanes
      for (const m of marks) {
        if (m < t0 || m > t1) continue;
        const px = Math.round(x(m));
        g.fillStyle = colors.acc;
        g.beginPath?.();
        g.moveTo?.(px - 5, 0);
        g.lineTo?.(px + 5, 0);
        g.lineTo?.(px, 7);
        g.fill?.();
        g.globalAlpha = 0.35;
        g.fillRect(px, RULER, 1, H - RULER);
        g.globalAlpha = 1;
      }
      // the span being marked as an event (#287): a band over its camera's lane, edged at both ends
      const sp = drag ?? shown.current.span;
      const lane = sp ? L.cams.find(l => l.cam === sp.cam) : undefined;
      if (sp && lane && sp.end >= t0 && sp.start <= t1) {
        const a = x(sp.start);
        const w = Math.max(2, x(sp.end) - a);
        const [ly, lh] = lane.video;
        g.fillStyle = colors.acc;
        g.globalAlpha = 0.28;
        g.fillRect(a, ly, w, lh);
        g.globalAlpha = 1;
        g.fillRect(a, ly, 2, lh);
        g.fillRect(a + w - 2, ly, 2, lh);
        // Small ruler handles make even a single mark visible before its other boundary is set.
        const points = drag ? { in: drag.start, out: drag.end } : shown.current.eventMarks;
        for (const [label, ts] of [
          ['I', points?.in],
          ['O', points?.out],
        ] as const) {
          if (ts == null) continue;
          const px = x(ts);
          g.fillRect(px - 3, RULER - 8, 6, 8);
          g.fillText?.(label, px + 5, RULER - 9);
        }
      }
      // Every lane shares the selected facets' temporal intersection, even while playback's Skip is off.
      // Unknown timing stays neutral; a known empty intersection dims the whole loaded strip.
      if (E.ui.getState().sel.size && E.present !== null) {
        g.fillStyle = colors.bg;
        g.globalAlpha = 0.55;
        let from = t0;
        for (const [a0, b0] of E.present) {
          if (b0 < t0) continue;
          if (a0 > from) g.fillRect(x(from), RULER, x(Math.min(a0, t1)) - x(from), H - RULER);
          from = Math.max(from, b0);
          if (from >= t1) break;
        }
        if (from < t1) g.fillRect(x(from), RULER, W - x(from), H - RULER);
        g.globalAlpha = 1;
      }
    }

    function paint() {
      dirty = false;
      if (!el || !cv) return;
      const dpr = window.devicePixelRatio || 1;
      W = el.clientWidth;
      const H = shown.current.L.height;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
        cv.width = Math.round(W * dpr);
        cv.height = Math.round(H * dpr);
        cv.style.height = `${H}px`;
      }
      const g = cv.getContext('2d');
      if (g && W > 0) {
        g.setTransform?.(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        draw(g, H);
      }
      placeHead();
    }

    function placeHead() {
      if (!ph) return;
      if (E.t == null) {
        ph.style.display = 'none';
        return;
      }
      const px = x(E.t);
      ph.style.display = px < -2 || px > W + 2 ? 'none' : '';
      ph.style.transform = `translateX(${px}px)`;
    }

    redraw.current = () => {
      dirty = true;
    };
    E.attachTimeline({
      frame() {
        if (dirty) paint();
        else placeHead();
      },
      redraw() {
        dirty = true;
      },
    });

    // ---- pointer input: scrub (lanes, playhead), pan (ruler, or a middle drag anywhere), click to seek, pinch to zoom ----
    const pts = new Map<number, number>();
    let mode: 'scrub' | 'pan' | 'pending' | 'pinch' | 'span' | null = null;
    let drag: EventSpan | null = null; // a span being dragged out (#287)
    let spanFrom = 0;
    let startX = 0;
    let startY = 0;
    let startCY = 0; // clientY at the press, for how far a gesture moved
    let moved = false; // more than a few pixels either way: a drag, never a click
    let startView = { ...E.view };
    let pinch0 = { dist: 1, mid: 0, view: { ...E.view } };
    const local = (e: PointerEvent | WheelEvent) => e.clientX - el.getBoundingClientRect().left;
    const width = () => W || el.clientWidth || 1;

    function down(e: PointerEvent) {
      try {
        el?.setPointerCapture(e.pointerId);
      } catch {
        // pointers that can't be captured (synthetic) still work while over the track
      }
      W = width();
      pts.set(e.pointerId, local(e));
      // A middle drag pans from anywhere on the timeline, and its release never seeks (#2354).
      if (e.button === 1) {
        e.preventDefault();
        startX = local(e);
        startCY = e.clientY;
        moved = false;
        startView = { ...E.view };
        mode = 'pan';
        if (el) el.dataset.panning = '';
        if (tp) tp.hidden = true;
        return;
      }
      if (pts.size === 2) {
        const [a = 0, b = 0] = [...pts.values()];
        pinch0 = { dist: Math.abs(a - b) || 1, mid: tsAt((a + b) / 2), view: { ...E.view } };
        if (mode === 'scrub') E.scrubEnd();
        mode = 'pinch';
        return;
      }
      const px = local(e);
      const y = e.clientY - (el?.getBoundingClientRect().top ?? 0);
      startX = px;
      startY = y;
      startCY = e.clientY;
      moved = false;
      startView = { ...E.view };
      // Only the explicit Mark an event pointer tool selects a range; Shift-drag scrubs normally. A drag anywhere on
      // the timeline marks: over a camera's lane, for that camera; elsewhere, for the camera the I and O keys use.
      const cam = markingRef.current ? (camAt(shown.current.L, y) ?? E.markingCamera()) : null;
      if (cam) {
        mode = 'span';
        spanFrom = tsAt(px);
        drag = { cam, start: spanFrom, end: spanFrom };
        dirty = true;
        if (tp) tp.hidden = true;
        return;
      }
      const nearHead = E.t != null && Math.abs(px - x(E.t)) <= 10;
      const onHead = e.target instanceof Node && (e.target === ph || !!ph?.contains(e.target));
      if (onHead || (y >= RULER && nearHead)) {
        mode = 'scrub';
        E.scrubStart(tsAt(px));
      } else mode = 'pending';
      if (tp) tp.hidden = true;
    }
    function move(e: PointerEvent) {
      const px = local(e);
      if (!pts.has(e.pointerId)) return hover(e, px);
      pts.set(e.pointerId, px);
      if (Math.hypot(px - startX, e.clientY - startCY) > 4) moved = true;
      if (mode === 'pinch' && pts.size === 2) {
        const [a = 0, b = 0] = [...pts.values()];
        const k = pinch0.dist / (Math.abs(a - b) || 1);
        const span = (pinch0.view.t1 - pinch0.view.t0) * k;
        E.setView(pinch0.mid - ((a + b) / 2 / W) * span, pinch0.mid + (1 - (a + b) / 2 / W) * span, true);
      } else if (mode === 'span' && drag) {
        const to = tsAt(px);
        drag = { cam: drag.cam, start: Math.min(spanFrom, to), end: Math.max(spanFrom, to) };
        dirty = true;
        return;
      } else if (mode === 'scrub') E.scrubMove(tsAt(px));
      else if (mode === 'pending' && moved) {
        if (startY < RULER) {
          mode = 'pan';
          if (el) el.dataset.panning = '';
        } else {
          mode = 'scrub';
          E.scrubStart(tsAt(startX));
          E.scrubMove(tsAt(px));
        }
      }
      if (mode === 'pan') {
        const dt = ((px - startX) / W) * (startView.t1 - startView.t0);
        E.setView(startView.t0 - dt, startView.t1 - dt, true);
      }
    }
    function up(e: PointerEvent) {
      if (el) delete el.dataset.panning;
      const px = local(e);
      pts.delete(e.pointerId);
      if (mode === 'pinch') {
        if (pts.size === 0) mode = null;
        return;
      }
      if (mode === 'span') {
        const picked = drag;
        drag = null;
        mode = null;
        dirty = true;
        if (picked && picked.end - picked.start >= MIN_SPAN_S) {
          E.setMarkingTool(false);
          onSpanRef.current?.(picked);
        }
        return;
      }
      // The marks in the lanes are only drawn: a click anywhere (ruler, lanes, or a mark) seeks to where it landed,
      // and a press that moved 4 px or less is still a click.
      const ts = tsAt(px);
      if (mode === 'scrub') E.scrubEnd(ts);
      else if (mode === 'pending') E.jump(ts);
      mode = null;
    }
    /** A gesture the browser took over (a touch turned scroll, a lost pointer): end it without acting on it. */
    function cancel(e: PointerEvent) {
      if (el) delete el.dataset.panning;
      pts.delete(e.pointerId);
      if (mode === 'scrub') E.scrubEnd();
      if (mode === 'span') {
        drag = null;
        dirty = true;
      }
      if (pts.size === 0) mode = null;
    }
    function hover(e: PointerEvent, px: number) {
      // The time ruler pans when dragged, so it shows a hand there; the lanes below seek, with the arrow.
      if (el) el.dataset.zone = e.clientY - el.getBoundingClientRect().top < RULER ? 'ruler' : '';
      if (e.pointerType === 'touch' || !tp) return;
      // A hover can come before the first paint measured the track: with W still 0, tsAt is no time, and the
      // tooltip's clock threw "Invalid time value".
      W = width();
      const ts = tsAt(px);
      const activityHit = hitAt(
        shown.current.L,
        e.clientY - (el?.getBoundingClientRect().top ?? 0),
        ts,
        (6 / width()) * (E.view.t1 - E.view.t0),
        E.filteredChips as PresenceChip[],
        said(),
      );
      const hit = lineNear(E.entries, ts, px, x);
      tp.textContent =
        activityHit?.kind === 'activity'
          ? `${hms(activityHit.start)} · ${hitTitle(activityHit)} · ${activityHit.activity.count == null ? '' : `Count: ${activityHit.activity.count} · `}${activityHit.activity.promoted ? 'Detected' : '? Unconfirmed'}`
          : shown.current.L.compact && activityHit
            ? `${hms(ts)} · ${hitTitle(activityHit)}`
            : hms(ts) +
              (hit
                ? `  ${shownName(speakerName(speakersRef.current, hit))}: ${String(hit.text ?? '').slice(0, 80)}`
                : '');
      tp.hidden = false;
      tp.style.transform = `translateX(${Math.min(Math.max(0, px - tp.offsetWidth / 2), W - tp.offsetWidth)}px)`;
    }
    const leave = () => {
      if (tp) tp.hidden = true;
    };
    const shuttle = new WheelShuttle();
    let shuttleFrame: number | null = null;
    let shuttleTarget: { time: number; gen: number; navigation: number } | null = null;
    // Wheel zooms or pans; Shift shuttles gently, accelerating as the scrolling continues.
    function wheel(e: WheelEvent) {
      W = width();
      const px = local(e);
      const { t0, t1 } = E.view;
      const span = t1 - t0;
      if (e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (E.t == null) return;
        const seconds = shuttle.step({
          deltaX: e.deltaX,
          deltaY: e.deltaY,
          deltaMode: e.deltaMode,
          width: W,
          span,
          now: performance.now(),
        });
        if (!seconds) return;
        e.preventDefault();
        const gen = E.ui.getState().gen;
        const navigation = E.navigationVersion;
        shuttleTarget = {
          time:
            (shuttleTarget?.gen === gen && shuttleTarget.navigation === navigation ? shuttleTarget.time : E.t) +
            seconds,
          gen,
          navigation,
        };
        if (shuttleFrame == null) {
          shuttleFrame = requestAnimationFrame(() => {
            shuttleFrame = null;
            const target = shuttleTarget;
            shuttleTarget = null;
            if (target?.gen === E.ui.getState().gen && target.navigation === E.navigationVersion) E.jump(target.time);
          });
        }
        return;
      }
      e.preventDefault();
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY) && !e.ctrlKey) {
        const dt = (e.deltaX / W) * span;
        E.setView(t0 + dt, t1 + dt, true);
        return;
      }
      const k = Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
      const at = tsAt(px);
      E.setView(at - (at - t0) * k, at + (t1 - at) * k, true);
    }
    // A middle press's mouse events are cancelled too, so the browser's middle-click autoscroll never starts.
    const middle = (e: MouseEvent) => {
      if (e.button === 1) e.preventDefault();
    };
    el.addEventListener('pointerdown', down);
    el.addEventListener('mousedown', middle);
    el.addEventListener('auxclick', middle);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', cancel);
    el.addEventListener('pointerleave', leave);
    el.addEventListener('wheel', wheel, { passive: false });
    // The colours come from CSS variables that follow the colour scheme, so read them again when it changes.
    const mq = matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', readColors);
    const themeOb = new MutationObserver(readColors);
    themeOb.observe(document.documentElement, { attributes: true, attributeFilter: ['data-mantine-color-scheme'] });
    const ro = new ResizeObserver(() => {
      dirty = true;
    });
    ro.observe(el);
    readColors();
    return () => {
      ro.disconnect();
      mq.removeEventListener('change', readColors);
      themeOb.disconnect();
      el.removeEventListener('pointerdown', down);
      el.removeEventListener('mousedown', middle);
      el.removeEventListener('auxclick', middle);
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', cancel);
      el.removeEventListener('pointerleave', leave);
      el.removeEventListener('wheel', wheel);
      if (shuttleFrame != null) cancelAnimationFrame(shuttleFrame);
      redraw.current = () => {};
      E.attachTimeline(null);
    };
  }, [engine, said]);

  /** Escape stops marking. */
  const keyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Escape' || !marking) return;
    e.preventDefault();
    engine.setMarkingTool(false);
    engine.closeEvent();
  };
  return (
    <div className={classes.scrub}>
      {(marking || eventMarks) && (
        <Text size="xs" c="dimmed" role="status" mb={4}>
          {marking
            ? "Drag across the timeline, from where it starts to where it ends (over a camera's lane, for that camera). Escape cancels."
            : `${eventMarks?.in != null ? `Mark in: ${hms(eventMarks.in)}. ` : ''}${eventMarks?.out != null ? `Mark out: ${hms(eventMarks.out)}. ` : ''}Escape clears the marks.`}
        </Text>
      )}
      <div className={classes.trackRow}>
        {L.labels.length > 0 && (
          // The lanes' names in a column to their left, so the lanes pack tight (#2577).
          <div className={so.laneLabels} style={{ height: L.height }} aria-hidden>
            {L.labels.map(l => (
              <span
                key={`${l.text}-${l.y}`}
                className={so.laneLabel}
                title={l.text}
                style={{ top: l.y, height: l.h, lineHeight: `${l.h}px` }}
              >
                {l.text}
              </span>
            ))}
          </div>
        )}
        <div
          className={classes.track}
          data-layout={mode}
          data-marking={marking || undefined}
          ref={box}
          style={{ height: L.height }}
          onKeyDown={keyDown}
          role="slider"
          tabIndex={0}
          aria-label="Timeline: click or drag to seek, Shift scroll to shuttle, scroll to zoom"
          aria-valuemin={Math.min(range?.t0 ?? engine.view.t0, t ?? engine.view.t0)}
          aria-valuemax={Math.max(range?.t1 ?? engine.view.t1, t ?? engine.view.t1)}
          aria-valuenow={t ?? undefined}
          aria-valuetext={t != null ? `${isoDay(new Date(t * 1000))} ${hms(t)}` : 'Nothing recorded'}
          aria-busy={loading || undefined}
        >
          <canvas ref={canvas} />
          <div className={classes.playhead} ref={head}>
            <span className={classes.knob} />
          </div>
          <div className={classes.tip} ref={tip} hidden />
          {loading && <Skeleton className={classes.cover} radius={0} aria-hidden data-testid="timeline-skeleton" />}
        </div>
      </div>
      <ChunkLine chunkStatus={chunkStatus} range={range} notes={notes} />
    </div>
  );
}

/** How the timeline reads and how to move through it, for the control bar's "?" (#2454): each colour beside what it
 *  means, then the pointer's gestures beside what they do, in the "?" table's two columns (#2577). The colours mean
 *  the same in the compact strip and the camera lanes. */
export function TimelineKey() {
  const swatches: [className: string | undefined, text: string][] = [
    [so.lgPeople, 'Speech, and who each camera saw (a colour per person)'],
    [classes.lgVideo, 'Video recorded'],
    [classes.lgAudio, 'Audio kept'],
    [classes.lgCut, "Video that won't play: still recording, or cut off"],
    [so.lgEvent, 'Events (a dashed ? is not yet confirmed)'],
    [classes.lgMark, 'Picked in Show only'],
  ];
  const gestures: [how: string, does: string][] = [
    ['Click or drag', 'Seek'],
    ['Drag the times', 'Pan'],
    ['Scroll or pinch', 'Zoom'],
    ['Shift and scroll', 'Shuttle'],
  ];
  return (
    <table className={`${classes.helpTable} ${classes.legend}`} aria-label="Timeline key">
      <tbody>
        <tr>
          <th colSpan={2} scope="rowgroup" className={classes.helpGroup}>
            The timeline
          </th>
        </tr>
        {swatches.map(([className, text]) => (
          <tr key={text}>
            <td className={classes.helpLead}>
              <i className={className} />
            </td>
            <td>{text}</td>
          </tr>
        ))}
      </tbody>
      <tbody>
        <tr>
          <th colSpan={2} scope="rowgroup" className={classes.helpGroup}>
            On the timeline
          </th>
        </tr>
        {gestures.map(([how, does]) => (
          <tr key={how}>
            <td className={`${classes.helpLead} ${classes.helpHow}`}>{how}</td>
            <td>{does}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** The timeline's tools on Review's control bar (#2454): how much of the day it spans, Latest, the Mark pointer tool,
 *  and whether it shows a lane per camera. mark: offer Mark (the page handles the span it picks). */
export function TimelineTools({ mark = false }: { mark?: boolean }) {
  const engine = useEngine();
  const marking = useReview(s => s.markingTool);
  const lanes = usePrefs(s => s.reviewTimeline === 'expanded');
  const zoom = (label: string, name: string, span: number) => (
    <Button variant="subtle" size="compact-sm" aria-label={name} onClick={() => engine.zoomSpan(span)}>
      {label}
    </Button>
  );
  return (
    <>
      <Group gap={0} role="group" aria-label="Zoom the timeline" wrap="nowrap">
        {zoom('Day', 'Show the whole day', DAY_S)}
        {zoom('1 h', 'Show 1 hour', 3600)}
        {zoom('10 min', 'Show 10 minutes', 600)}
      </Group>
      <Button variant="default" size="compact-sm" onClick={() => engine.latest()}>
        Latest
      </Button>
      {mark && (
        <Button
          variant={marking ? 'light' : 'default'}
          size="compact-sm"
          aria-pressed={marking}
          aria-label="Mark an event"
          onClick={() => engine.setMarkingTool(!marking)}
          title="Drag across a camera's lane, or use I and O at the playhead"
        >
          Mark
        </Button>
      )}
      <Button
        variant={lanes ? 'light' : 'default'}
        size="compact-sm"
        aria-pressed={lanes}
        onClick={() => setPref('reviewTimeline', lanes ? 'compact' : 'expanded')}
        aria-label="Camera lanes"
        title="One lane per camera, with who each one saw"
      >
        Lanes
      </Button>
    </>
  );
}
