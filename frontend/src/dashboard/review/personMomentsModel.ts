// One person's day in Review (#2057): their lines, the camera views that saw them, and the actions put down to them,
// merged into visits; and the summary blocks that mention them. Pure, so the panel and its tests share it.

import type { TranscriptEntry } from '../live/types';
import type { Block, DetectedActivity } from './api';
import { lineEnd } from './time';
import type { Chip, Span } from './types';

/** Moments closer than this are one visit. */
export const VISIT_GAP_S = 60;
export const LEAD_S = 2;
const POINT_TAIL_S = 2;

export interface Moment {
  kind: 'line' | 'seen' | 'action';
  start: number;
  end: number;
  cam: string | null; // the camera view, for a sighting or an action; null for a line (room audio)
  label: string; // a line's text, an action's name
}

/** One bounded item in a person's playback queue; null cam means room audio. */
export interface PlaybackMoment {
  start: number;
  end: number;
  cam: string | null;
}

export interface Visit {
  start: number;
  end: number;
  cams: string[]; // the views that saw them, most seen first
  lines: number;
  seen_s: number; // how long the cameras saw them, in seconds
  actions: string[];
  moments: Moment[]; // in time order
}

const same = (a: string | null | undefined, b: string) =>
  !!a && a.trim().toLocaleLowerCase() === b.trim().toLocaleLowerCase();

/** The person's moments of the day: their lines (`name` gives a line's speaker as the transcript shows it), the
 *  views their person chip says saw them, and the actions that name them. */
export function momentsOf(
  person: string,
  entries: readonly TranscriptEntry[],
  name: (e: TranscriptEntry) => string,
  chips: readonly Chip[],
  activities: readonly DetectedActivity[],
): Moment[] {
  const out: Moment[] = [];
  for (const e of entries)
    if (same(name(e), person))
      out.push({ kind: 'line', start: e.ts_start, end: lineEnd(e), cam: null, label: e.text ?? '' });
  const chip = chips.find(c => c.kind === 'person' && same(c.label, person));
  for (const [cam, spans] of Object.entries(chip?.cams ?? {}))
    for (const [start, end] of spans) out.push({ kind: 'seen', start, end, cam, label: '' });
  for (const a of activities)
    if (same(a.person_name, person) || same(a.person, person))
      out.push({ kind: 'action', start: a.start, end: a.end, cam: a.camera, label: a.name ?? a.kind ?? 'activity' });
  return out.sort((a, b) => a.start - b.start || a.end - b.end);
}

/** A person's selected media only. Audio keeps the exact positive line spans, with no room gaps added.
 *  Video adds two seconds of lead-in, gives point actions two seconds after their timestamp, and joins
 *  overlapping consecutive windows on the same camera so that footage is not replayed. */
export function playbackOf(moments: readonly Moment[], mode: 'audio' | 'video', dayStart: number): PlaybackMoment[] {
  if (!Number.isFinite(dayStart)) return [];
  const windows: PlaybackMoment[] = [];
  for (const m of moments) {
    if (!Number.isFinite(m.start) || !Number.isFinite(m.end) || m.start < 0 || m.end < m.start) continue;
    if (mode === 'audio') {
      if (m.kind === 'line' && m.end > m.start) windows.push({ start: m.start, end: m.end, cam: null });
    } else if (m.kind !== 'line' && m.cam?.trim()) {
      const start = Math.max(dayStart, m.start - LEAD_S);
      const end = m.kind === 'action' && m.start === m.end ? m.end + POINT_TAIL_S : m.end;
      if (m.end === m.start && m.kind !== 'action') continue;
      if (Number.isFinite(end) && end > start) windows.push({ start, end, cam: m.cam });
    }
  }
  windows.sort((a, b) => a.start - b.start || a.end - b.end || (a.cam ?? '').localeCompare(b.cam ?? ''));
  if (mode === 'audio') return windows;
  const queue: PlaybackMoment[] = [];
  for (const window of windows) {
    const previous = queue.at(-1);
    if (previous && previous.cam === window.cam && window.start <= previous.end)
      previous.end = Math.max(previous.end, window.end);
    else queue.push(window);
  }
  return queue;
}

/** Moments within VISIT_GAP_S of each other, as visits, in time order. */
export function visitsOf(moments: readonly Moment[], gap = VISIT_GAP_S): Visit[] {
  const visits: Visit[] = [];
  for (const m of moments) {
    const v = visits.at(-1);
    if (v && m.start - v.end <= gap) {
      v.end = Math.max(v.end, m.end);
      v.moments.push(m);
    } else visits.push({ start: m.start, end: m.end, cams: [], lines: 0, seen_s: 0, actions: [], moments: [m] });
  }
  for (const v of visits) {
    const byCam = new Map<string, number>();
    for (const m of v.moments) {
      if (m.kind === 'line') v.lines++;
      if (m.kind === 'seen') v.seen_s += m.end - m.start;
      if (m.kind === 'action' && !v.actions.includes(m.label)) v.actions.push(m.label);
      if (m.cam) byCam.set(m.cam, (byCam.get(m.cam) ?? 0) + Math.max(1, m.end - m.start));
    }
    v.cams = [...byCam].sort((a, b) => b[1] - a[1]).map(([cam]) => cam);
  }
  return visits;
}

/** Where the person's name is in a block's text, as [start, end) character offsets, whole words only. */
export function mentions(text: string, person: string): Span[] {
  const name = person.trim();
  if (!name) return [];
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const found: Span[] = [];
  for (const m of text.matchAll(new RegExp(`(?<![\\p{L}\\p{N}])${escaped}(?![\\p{L}\\p{N}])`, 'giu')))
    found.push([m.index, m.index + m[0].length]);
  return found;
}

/** The summary blocks that mention the person: by name in the text, or tagged with their person chip. */
export function mentioning(blocks: readonly Block[], person: string, chipId: string | null): Block[] {
  return blocks.filter(b => mentions(b.text, person).length > 0 || (!!chipId && b.tags.includes(chipId)));
}
