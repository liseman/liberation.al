// The labelled timeline's layout and what goes in its lanes (R3): "Who spoke", one lane per camera (recorded
// video in grey with a bar per person in view), the audio kept, and Events as tick marks. The cameras are angles of
// one rig (a backpack), not rooms, so a lane says who that angle saw. Pure, so the canvas drawing
// in Timeline.tsx stays thin and this is tested on its own.

import { activityHint, activityName } from '../lib/activity';
import { spkColor } from '../lib/speakers';
import { resolveId } from '../live/reduce';
import type { Speakers } from '../live/types';
import type { ActivityPromotion, DetectedActivity } from './api';
import { RULER } from './time';
import type { Chip, Span } from './types';
import { viewName } from './views';

export const UNKNOWN_FACES = 'person:unknown'; // the chip for faces nobody has named
export const UNKNOWN_COLOR = 'hsl(220 10% 62%)'; // unknown faces: a person, but nobody's colour
export const MAX_ROWS = 5; // presence rows per camera; the rest share one "Others" row
const LANE_MIN = 14; // a labelled lane's least height: one line of its label, which sits to its left
const LANE_GAP = 2; // between lanes
const ROW = 5; // one presence row: a 4 px bar and a 1 px gap
const OTHERS = 'others';

export type Band = [y: number, h: number];

/** A chip from GET /review/presence, with a person's recorded presence per camera when the server sends it. Declared
 *  here rather than in types.ts (shared with R1, R2, and R4) since only the lanes read it. */
export type PresenceChip = Chip & { cams?: Record<string, Span[]> };

export interface PresenceRow {
  id: string; // the chip's id, or 'others'
  label: string;
  color: string;
  spans: Span[];
}

export interface CamLane {
  cam: string;
  video: Band; // the grey recorded video
  rows: { row: PresenceRow; band: Band }[];
}

export interface Layout {
  compact?: true;
  marks: Band; // the selected things and events
  speech: Band;
  cams: CamLane[];
  audio: Band;
  events: Band;
  labels: { text: string; y: number; h: number }[]; // each lane's label, beside its row: its top and height
  height: number;
}

/** One colour per person everywhere (their Show only dot, speech, and presence): their voice's colour when a voice
 *  has their name, else one from the name, so a person with no voice yet still keeps it. */
export function personColor(name: string, speakers: Speakers): string {
  const key = name.trim().toLowerCase();
  if (!key || key === 'unknown') return spkColor(null); // speech nobody is known for: grey, never a person's colour
  const voice = speakers.list.find(s => s.label.trim().toLowerCase() === key);
  return voice ? spkColor(resolveId(speakers, voice.id)) : spkColor(key || null);
}

/** The people seen on each camera, as presence rows, most time first. Rows past MAX_ROWS - 1 fold into "Others". */
export function presenceRows(
  chips: readonly PresenceChip[],
  cams: readonly string[],
  colorOf: (c: Chip) => string,
): Record<string, PresenceRow[]> {
  const out: Record<string, PresenceRow[]> = Object.create(null);
  for (const cam of cams) {
    const seen = chips
      .filter(c => c.kind === 'person' && c.cams && Object.hasOwn(c.cams, cam) && (c.cams[cam]?.length ?? 0) > 0)
      .map(c => {
        const spans = c.cams?.[cam] ?? [];
        return { c, spans, total: spans.reduce((s, [a, b]) => s + Math.max(0, b - a), 0) };
      })
      .sort((a, b) => b.total - a.total || a.c.label.localeCompare(b.c.label));
    const rows: PresenceRow[] = seen
      .slice(0, seen.length > MAX_ROWS ? MAX_ROWS - 1 : MAX_ROWS)
      .map(({ c, spans }) => ({ id: c.id, label: c.label, color: colorOf(c), spans }));
    if (seen.length > MAX_ROWS)
      rows.push({
        id: OTHERS,
        label: `${seen.length - (MAX_ROWS - 1)} others`,
        color: 'hsl(220 8% 55%)', // a canvas colour, not a CSS variable
        spans: seen.slice(MAX_ROWS - 1).flatMap(s => s.spans),
      });
    out[cam] = rows;
  }
  return out;
}

/** Where each lane goes, top to bottom, each a row with its label to its left. names gives a camera's label. */
export function layout(
  cams: readonly string[],
  rows: Record<string, PresenceRow[]>,
  names: (cam: string) => string = cam => cam,
): Layout {
  const labels: Layout['labels'] = [];
  let y = RULER + 3;
  const marks: Band = [y, 6];
  y += 6 + LANE_GAP;
  // A labelled lane: a row at least a label's line tall, its band centred in it, then a small gap to the next.
  const lane = (text: string, h: number): Band => {
    const rh = Math.max(LANE_MIN, h);
    const top = y;
    labels.push({ text, y: top, h: rh });
    y += rh + LANE_GAP;
    return [top + Math.floor((rh - h) / 2), h];
  };
  const speech = lane('Who spoke', 12);
  const lanes: CamLane[] = [];
  for (const cam of cams) {
    const mine = (Object.hasOwn(rows, cam) ? rows[cam] : undefined) ?? [];
    const [vy, vh] = lane(names(cam), Math.max(LANE_MIN, 3 + mine.length * ROW));
    lanes.push({ cam, video: [vy, vh], rows: mine.map((row, i) => ({ row, band: [vy + 2 + i * ROW, ROW - 1] })) });
  }
  const audio = lane('Audio kept', 4);
  const events = lane('Events', 10);
  return { marks, speech, cams: lanes, audio, events, labels, height: y - LANE_GAP + 3 };
}

/** One shared strip, independent of camera and person counts. The bands retain the expanded drawing contract. */
export function compactLayout(cams: readonly string[], rows: Record<string, PresenceRow[]>): Layout {
  const video: Band = [RULER + 4, 38];
  const presence: Band = [RULER + 13, 8];
  return {
    compact: true,
    marks: [RULER + 24, 5],
    speech: [RULER + 32, 8],
    cams: cams.map(cam => ({
      cam,
      video,
      rows: ((Object.hasOwn(rows, cam) ? rows[cam] : undefined) ?? []).map(row => ({ row, band: presence })),
    })),
    audio: [RULER + 42, 6],
    events: [RULER + 4, 38],
    labels: [],
    height: RULER + 52, // 72 px with the ruler: the main way through the day, with room to aim at a marker
  };
}

/** The camera whose lane a height on the track is in, else null. */
export function camAt(L: Layout, y: number): string | null {
  if (L.compact) return null;
  const lane = L.cams.find(l => y >= l.video[0] && y < l.video[0] + l.video[1]);
  return lane?.cam ?? null;
}

/** Whether a person's speech and presence show at full strength: yes when no person is picked in Show only, else
 *  only the picked people. */
export function shownPerson(name: string, picked: ReadonlySet<string>): boolean {
  return picked.size === 0 || picked.has(name.trim().toLowerCase());
}

/** The lowercased names of the people picked in Show only (person chips, not "Unknown faces"). */
export function pickedPeople(chips: readonly Chip[], sel: ReadonlySet<string>): Set<string> {
  return new Set(
    chips.filter(c => c.kind === 'person' && c.id !== UNKNOWN_FACES && sel.has(c.id)).map(c => c.label.toLowerCase()),
  );
}

export function activityKind(item: DetectedActivity): string {
  return item.kind ? activityName(item.kind) : 'Unclassified activity';
}

export function activitySubject(item: DetectedActivity): string | null {
  return item.person_name ?? null;
}

export function activityTrack(track: string): string {
  return track.length > 16 ? `${track.slice(0, 8)}…` : track;
}

export function activityTitle(item: DetectedActivity): string {
  const subject = activitySubject(item);
  const person = subject ? ` · ${subject}` : '';
  const track = item.track_id ? ` · track ${activityTrack(item.track_id)}` : '';
  const hint = activityHint(item.kind);
  return `${item.name ?? activityKind(item)}${person}${track}${hint ? `. ${hint}` : ''}`;
}

/** Kind-level filters remain bounded; every marker retains its own source identity and evidence. */
export function activityChips(
  items: readonly DetectedActivity[],
  promotion: readonly ActivityPromotion[] = [],
  cameraName: (camera: string) => string = viewName,
): Chip[] {
  const groups = new Map<string, Chip>();
  for (const item of items) {
    const id = `event:activity:${item.kind ?? 'movement'}`;
    const chip = groups.get(id) ?? {
      id,
      kind: 'event',
      label: activityKind(item),
      spans: [],
      total_s: 0,
      count: 0,
      notes: [],
      activities: [],
    };
    chip.spans.push([item.start, item.end]);
    chip.total_s += Math.max(0, item.end - item.start);
    chip.count++;
    chip.activities?.push(item);
    groups.set(id, chip);
  }
  for (const chip of groups.values()) {
    const kind = chip.activities?.[0]?.kind ?? null;
    const angles = new Set(chip.activities?.map(item => item.camera));
    const base = promotion.find(row => row.kind === kind && row.camera === null);
    const overrides = promotion.filter(row => row.kind === kind && row.camera !== null && angles.has(row.camera));
    const summary = promotionText(base);
    const parts = overrides.length
      ? [`Default: ${summary}`, ...overrides.map(row => `${cameraName(row.camera ?? '')}: ${promotionText(row)}`)]
      : [summary];
    const hint = activityHint(kind);
    chip.notes = [`${hint ? `${hint} ` : ''}${parts.join('. ')}.`];
  }
  return [...groups.values()];
}

export function promotionText(row: ActivityPromotion | undefined): string {
  if (!row) return 'Counting status unavailable';
  switch (row.status) {
    case 'counted':
      return 'Counted in Ask; clips answered No in Known, Activities are left out';
    case 'needs_correct_ratings':
      return 'Not counted in Ask: more Yes answers in Known, Activities';
    case 'disabled':
      return 'Not counted in Ask';
    case 'needs_ratings': {
      const n = row.remaining_ratings;
      return typeof n === 'number' && Number.isInteger(n) && n > 0
        ? `Needs ${n} more ${n === 1 ? 'Yes or No answer' : 'Yes or No answers'} in Known, Activities`
        : 'Not counted in Ask: more Yes or No answers in Known, Activities';
    }
    default:
      return 'Counting status unavailable';
  }
}

/** Legacy events are ticks; detected activities retain their duration and individual identity. */
export function eventTicks(
  chips: readonly Chip[],
  colorOf: (c: Chip) => string,
  activityColor?: (item: DetectedActivity) => string,
): { t: number; color: string; end?: number; activity?: DetectedActivity; chipId?: string }[] {
  return chips
    .filter(c => c.kind === 'event')
    .flatMap(c =>
      c.activities
        ? c.activities.map(activity => ({
            t: activity.start,
            end: activity.end,
            color: activityColor?.(activity) ?? colorOf(c),
            activity,
            chipId: c.id,
          }))
        : c.spans.map(([a]) => ({ t: a, color: colorOf(c) })),
    )
    .sort((a, b) => a.t - b.t);
}

/** The bands for what's picked in Show only: things and events, plus any person (or Unknown faces) the server sent no
 *  per-camera presence for, whose time would otherwise show nowhere. People with presence show in their cameras' lanes. */
export function markBands(
  chips: readonly PresenceChip[],
  sel: ReadonlySet<string>,
  colorOf: (c: Chip) => string,
): { color: string; spans: Span[] }[] {
  return chips
    .filter(c => sel.has(c.id) && (c.kind !== 'person' || !Object.values(c.cams ?? {}).some(s => s.length)))
    .map(c => ({ color: colorOf(c), spans: c.spans }));
}

// ---- what the pointer is over on the timeline, for the hover tooltip ----

export const RUN_GAP_S = 10; // lines less than this apart are one stretch of talk

/** A line of the transcript, as much of it as hit testing needs. */
export interface Said {
  ts_start: number;
  ts_end?: number | null;
  who: string;
}

/** What the pointer is over: an event, camera-backed presence, an unscoped thing, or a stretch of talk. */
export type Hit =
  | { kind: 'activity'; activity: DetectedActivity; start: number; end: number }
  | { kind: 'event'; chip: Chip; start: number; end: number }
  | { kind: 'presence'; chip: PresenceChip; cam: string; start: number; end: number }
  | { kind: 'thing'; chip: PresenceChip; start: number; end: number }
  | { kind: 'speech'; names: string[]; start: number; end: number };

const within = (y: number, [top, h]: Band, pad = 2) => y >= top - pad && y <= top + h + pad;
const endOf = (l: Said) => Math.max(l.ts_end ?? l.ts_start, l.ts_start + 0.2);
/** How far ts is from [a, b]: 0 inside it. */
const off = (ts: number, a: number, b: number) => (ts < a ? a - ts : ts > b ? ts - b : 0);
/** The index of the item nearest ts, within tol, or -1. */
function nearest<T>(items: readonly T[], ts: number, tol: number, span: (it: T) => readonly [number, number]): number {
  let best = -1;
  let gap = tol;
  items.forEach((it, i) => {
    const [a, b] = span(it);
    const d = off(ts, a, b);
    if (d <= gap) {
      best = i;
      gap = d;
      if (d === 0) gap = -1; // inside one: keep it
    }
  });
  return best;
}

/** The stretch of talk around line i: the lines either side less than RUN_GAP_S apart, and who spoke in it. */
export function speechRun(lines: readonly Said[], i: number): Hit | null {
  const at = lines[i];
  if (!at) return null;
  let a = i;
  let b = i;
  while (a > 0 && (lines[a] as Said).ts_start - endOf(lines[a - 1] as Said) < RUN_GAP_S) a--;
  while (b < lines.length - 1 && (lines[b + 1] as Said).ts_start - endOf(lines[b] as Said) < RUN_GAP_S) b++;
  const run = lines.slice(a, b + 1);
  const names = [...new Set(run.map(l => l.who))];
  return { kind: 'speech', names, start: (lines[a] as Said).ts_start, end: Math.max(...run.map(endOf)) };
}

/** Events keep their source start; activities can be picked anywhere in their actual duration. */
function eventHit(ts: number, tol: number, chips: readonly PresenceChip[]): Hit | null {
  let best: Hit | null = null;
  let gap = tol;
  for (const c of chips)
    if (c.kind === 'event' && c.activities) {
      for (const activity of c.activities) {
        const distance = off(ts, activity.start, activity.end);
        if (distance <= gap) {
          gap = distance;
          best = { kind: 'activity', activity, start: activity.start, end: activity.end };
        }
      }
    } else if (c.kind === 'event')
      for (const [a, b] of c.spans)
        if (Math.abs(a - ts) <= gap) {
          gap = Math.abs(a - ts);
          best = { kind: 'event', chip: c, start: a, end: b };
        }
  return best;
}

/** Overlaid evidence is picked by time, not by its tiny drawing band or folded presence row. */
function compactHit(
  L: Layout,
  ts: number,
  tol: number,
  chips: readonly PresenceChip[],
  lines: readonly Said[],
  eventTolerance = tol,
): Hit | null {
  const cameras = L.cams.map(lane => lane.cam);
  const visible = cameraChips(chips, cameras);
  const event = eventHit(ts, eventTolerance, visible);
  if (event) return event;
  const spans: Hit[] = [];
  for (const chip of visible) {
    if (chip.kind === 'event') continue;
    if (chip.cams && Object.keys(chip.cams).length) {
      for (const [cam, seen] of Object.entries(chip.cams))
        for (const [start, end] of seen) spans.push({ kind: 'presence', chip, cam, start, end });
    } else {
      for (const [start, end] of chip.spans)
        spans.push(
          chip.camera ? { kind: 'presence', chip, cam: chip.camera, start, end } : { kind: 'thing', chip, start, end },
        );
    }
  }
  const i = nearest(spans, ts, tol, hit => [hit.start, hit.end]);
  if (i >= 0) return spans[i] ?? null;
  const line = nearest(lines, ts, tol, item => [item.ts_start, endOf(item)]);
  return line < 0 ? null : speechRun(lines, line);
}

/** What sits at (ts, y) on the timeline, tol seconds either side in time (a few pixels), or null for empty lane. */
export function hitAt(
  L: Layout,
  y: number,
  ts: number,
  tol: number,
  chips: readonly PresenceChip[],
  lines: readonly Said[],
): Hit | null {
  if (L.compact) return compactHit(L, ts, tol, chips, lines);
  if (within(y, L.events, 4)) {
    return eventHit(ts, tol, chips);
  }
  if (within(y, L.speech)) {
    const i = nearest(lines, ts, tol, l => [l.ts_start, endOf(l)]);
    return i < 0 ? null : speechRun(lines, i);
  }
  for (const lane of L.cams) {
    if (!within(y, lane.video)) continue;
    const row = lane.rows.find(r => within(y, r.band, 1));
    if (!row || row.row.id === 'others') return null;
    const span = row.row.spans[nearest(row.row.spans, ts, tol, s => s)];
    const chip = chips.find(c => c.id === row.row.id);
    return span && chip ? { kind: 'presence', chip, cam: lane.cam, start: span[0], end: span[1] } : null;
  }
  return null;
}

/** Names as a phrase: "Kara", "Kara and Luca", "Kara, Luca, and Dana". */
export function nameList(names: readonly string[]): string {
  if (names.length <= 2) return names.join(' and ');
  return `${names.slice(0, -1).join(', ')}, and ${names.at(-1)}`;
}

/** What the hover tooltip calls it: "Door opened", "Kara in view", "Kara and Luca talking". */
export function hitTitle(h: Hit): string {
  if (h.kind === 'activity') return activityTitle(h.activity);
  if (h.kind === 'event') return h.chip.label;
  if (h.kind === 'thing') return h.chip.label;
  if (h.kind === 'presence') return `${h.chip.label} in view`;
  return `${nameList(h.names)} talking`;
}

/** Distinct covered time, sorted, without counting speech/visit or multi-camera overlaps twice. */
function unionSpans(spans: readonly Span[]): Span[] {
  const out: Span[] = [];
  for (const [start, end] of [...spans].sort((a, b) => a[0] - b[0])) {
    const last = out.at(-1);
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else out.push([start, end]);
  }
  return out;
}

/** Restrict evidence with a known camera; speech and camera-less events remain available. */
export function cameraChips(chips: readonly PresenceChip[], cameras: readonly string[]): Chip[] {
  if (!cameras.length) return [...chips];
  return chips.flatMap(c => {
    if (c.camera && !cameras.includes(c.camera)) return [];
    if (c.activities) {
      const activities = c.activities.filter(a => cameras.includes(a.camera));
      return activities.length
        ? [{ ...c, activities, spans: activities.map(a => [a.start, a.end] as Span), count: activities.length }]
        : [];
    }
    if (c.cams) {
      const cams = Object.fromEntries(Object.entries(c.cams).filter(([cam]) => cameras.includes(cam)));
      const spans = unionSpans([...(c.speech ?? []), ...Object.values(cams).flat()]);
      return [{ ...c, cams, spans, count: spans.length, total_s: spans.reduce((sum, [a, b]) => sum + b - a, 0) }];
    }
    return [c];
  });
}
