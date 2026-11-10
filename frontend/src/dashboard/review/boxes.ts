// The boxes Review draws over the large camera: who and what that camera saw at the playhead, from GET
// /review/boxes, a minute of one camera at a time. People are body boxes from see-actions' pose tracks, 10 a second,
// and pets come 5 a second: both move smoothly between samples and end at a gap. Faces come about once a
// second; they're drawn only for a person without a body box (older recordings), and naming an unknown body uses its
// track's face. The camera's object labels (chair, car) come from groundings every several seconds, so each carries
// the moment it stops standing ("until"); remembered things are held a while after each sample. Show only
// keeps only what it picks.
import { queryOptions } from '@tanstack/react-query';
import { request } from '../../shared/http';
import type { Box } from '../camera/boxes';
import type { Chip } from './types';

export const BUCKET_S = 60; // one request covers this much of one camera
export const HOLD_S = 1; // a face's box stays up this long after its sample (faces log about once a second)
export const GAP_S = 0.5; // a body's or pet's box moves between samples this close, and ends at a longer gap
const FACE_GAP_S = 1.5; // faces log about once a second, so their samples are this far apart and still one face
const AHEAD_S = 0.05; // a sample this close after the frame counts as on it
export const THING_HOLD_S = 6; // a thing's, longer: object memory samples about every 5 s
export const LABEL_HOLD_S = 15; // a label without "until" (the server always sends one): review_boxes.LABEL_HOLD_S

/** One box as the server sends it. */
export interface ReviewBox {
  ts: number;
  id: string;
  kind: 'person' | 'face' | 'pet' | 'object' | 'label';
  box: number[];
  until?: number; // a label's last moment: its camera's next grounding, or a short hold
  label: string;
  tone: 'known' | 'unknown';
  name?: string;
  species?: string;
  pet_id?: string;
  track?: string; // a face's original track, a person's pose track (the same string its faces carry)
  group?: number; // an unknown face's face group, for naming it in Review
}

export const boxesQuery = (cam: string, bucket: number, recent: boolean) =>
  queryOptions({
    queryKey: ['review', 'boxes', cam, bucket] as const,
    queryFn: async ({ signal }) => {
      const start = bucket * BUCKET_S;
      const r = await request<{ boxes?: ReviewBox[] }>(
        `/review/boxes?cam=${encodeURIComponent(cam)}&start=${start}&end=${start + BUCKET_S}`,
        undefined,
        signal,
      );
      return r.boxes ?? [];
    },
    staleTime: recent ? 30_000 : Number.POSITIVE_INFINITY, // the last few minutes are still being scanned
  });

/** The boxes to draw at t: the labels standing at t; each person and pet where it is between its samples (moved
 *  in a straight line while they're at most GAP_S apart, else held until GAP_S after the last); each face the same
 *  way on its once-a-second cadence, unless its track has a body box; and the things of the newest object sample held
 *  THING_HOLD_S. */
export function boxesAt(list: readonly ReviewBox[], t: number): ReviewBox[] {
  const out: ReviewBox[] = list.filter(
    b => b.kind === 'label' && b.ts <= t + AHEAD_S && t < (b.until ?? b.ts + LABEL_HOLD_S),
  );
  const bodies = new Set(list.flatMap(b => (b.kind === 'person' && b.track ? [b.track] : [])));
  const samples = new Map<string, ReviewBox[]>(); // each moving box's samples, in time order
  for (const b of list) {
    if (b.kind !== 'person' && b.kind !== 'pet' && b.kind !== 'face') continue;
    if (b.kind === 'face' && b.track && bodies.has(b.track)) continue; // the body box shows this person
    const key = `${b.kind}\u0000${b.id}`;
    const run = samples.get(key);
    if (run) run.push(b);
    else samples.set(key, [b]);
  }
  for (const [key, run] of samples) {
    const face = key.startsWith('face\u0000');
    const at = placed(run, t, face ? FACE_GAP_S : GAP_S, face ? HOLD_S : GAP_S);
    if (at) out.push(at);
  }
  let moment: number | null = null; // things: the newest object sample within its hold, and everything from it
  for (const b of list) {
    if (b.kind !== 'object' || b.ts > t + AHEAD_S || b.ts < t - THING_HOLD_S) continue;
    if (moment == null || b.ts > moment) moment = b.ts;
  }
  if (moment != null) {
    const newest = moment;
    out.push(...list.filter(b => b.kind === 'object' && Math.abs(b.ts - newest) <= AHEAD_S));
  }
  return out;
}

/** One box's place at t from its samples (in time order): between two samples at most `gap` apart, moved in a
 *  straight line from the earlier to the later; else the last sample, held `hold`; null before it was seen or after. */
function placed(run: readonly ReviewBox[], t: number, gap: number, hold: number): ReviewBox | null {
  let prev: ReviewBox | undefined; // the last sample at or before t
  let next: ReviewBox | undefined; // and the first after it
  for (const b of run) {
    if (b.ts <= t) prev = b;
    else {
      next = b;
      break;
    }
  }
  if (!prev) return next && next.ts <= t + AHEAD_S ? next : null;
  if (next && next.ts - prev.ts <= gap) {
    const f = (t - prev.ts) / (next.ts - prev.ts);
    const to = next.box;
    return { ...prev, box: prev.box.map((v, k) => v + f * ((to[k] ?? v) - v)) };
  }
  if (next && next.ts <= t + AHEAD_S) return next;
  return t - prev.ts <= hold ? prev : null;
}

/** The face to name an unknown person or face from, placed on the frame at t: its track's newest unknown face
 *  sample with a face group at most HOLD_S before t. Faces are sampled about once a second while the person moves,
 *  so the sample's box is carried to t: moved as the track's body box moved since the sample when the track has
 *  one, else between face samples as Review draws them. */
export function faceToName(list: readonly ReviewBox[], track: string, t: number): ReviewBox | null {
  const faces = list.filter(b => b.kind === 'face' && b.track === track && b.tone === 'unknown' && b.group != null);
  let sample: ReviewBox | null = null;
  for (const b of faces) {
    if (b.ts > t + AHEAD_S || b.ts < t - HOLD_S) continue;
    if (!sample || b.ts > sample.ts) sample = b;
  }
  if (!sample) return null;
  const bodies = list.filter(b => b.kind === 'person' && b.track === track);
  const then = placed(bodies, sample.ts, GAP_S, GAP_S);
  const now = placed(bodies, t, GAP_S, GAP_S);
  if (then && now) {
    const dx = (now.box[0] ?? 0) + (now.box[2] ?? 0) - (then.box[0] ?? 0) - (then.box[2] ?? 0);
    const dy = (now.box[1] ?? 0) + (now.box[3] ?? 0) - (then.box[1] ?? 0) - (then.box[3] ?? 0);
    return { ...sample, box: sample.box.map((v, k) => v + (k % 2 === 0 ? dx : dy) / 2) };
  }
  const placedFace = placed(faces, t, FACE_GAP_S, HOLD_S);
  return placedFace && placedFace.id === sample.id ? { ...sample, box: placedFace.box } : sample;
}

/** What Show only lets through: everything while nothing is picked; else the picked people's faces, any unknown face
 *  or body when Unknown faces is picked, and the picked things. */
export function shownBoxes(list: readonly ReviewBox[], chips: readonly Chip[], sel: ReadonlySet<string>): ReviewBox[] {
  if (!sel.size) return [...list];
  const picked = chips.filter(c => sel.has(c.id));
  const people = new Set(picked.filter(c => c.kind === 'person').map(c => c.label.toLowerCase()));
  // A thing chip names a kind ("mug", presence's normalized label); a remembered thing's name is free text ("Red mug"),
  // so it matches when it has every word of the chip, plurals aside.
  const things = picked.filter(c => c.kind === 'object').map(c => words(c.label));
  const unknown = picked.some(c => c.id === 'person:unknown');
  return list.filter(b =>
    b.kind !== 'face' && b.kind !== 'person'
      ? things.some(w => [...w].every(x => words(b.label).has(x)))
      : b.tone === 'known'
        ? people.has((b.name ?? b.label).toLowerCase())
        : unknown,
  );
}

/** The words of a label, lowercased and without a plural s. */
const words = (s: string): Set<string> =>
  new Set(
    s
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(Boolean)
      .map(w => (w.length > 3 && w.endsWith('s') ? w.slice(0, -1) : w)),
  );

/** As the shared box renderer draws them: people's boxes (bodies and faces) as faces (green named, amber unknown), things are objects,
 *  and the camera's labels are objects without a tone, as Live draws them, so remembered things stand out. */
export const asBox = (b: ReviewBox): Box =>
  b.kind === 'label'
    ? { id: b.id, box: b.box, label: b.label, kind: 'object' }
    : { id: b.id, box: b.box, label: b.label, kind: b.kind === 'person' ? 'face' : b.kind, tone: b.tone };
