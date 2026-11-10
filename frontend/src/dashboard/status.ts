// Whether the microphone service and the cameras are up, from the live feed. "Stale" means the last update is late
// but recent enough that it's probably a hiccup; after STALE_LIMIT_S it counts as off. A camera is on while vision
// says its frames are arriving; captions only stand in when vision's status is stale, since a camera can go a
// minute between captions while the captioner works through the others.
import type { Ambient, SceneEntry, VisionStatus } from './live/types';

export const MIC_FRESH_S = 15; // the microphone service rewrites its status every few seconds
export const CAM_FRESH_S = 30;
export const VISION_FRESH_S = 10; // the dashboard sends the camera status at least every 5 s; older means it stopped
export const STALE_LIMIT_S = 300;

export type Health = 'on' | 'some' | 'stale' | 'off';

export interface MicStatus {
  state: Exclude<Health, 'some'>;
  age: number | null; // seconds since the last status, or null when there never was one
}

export interface CamStatus {
  state: Health;
  live: string[]; // cameras sending frames (or, while vision's status is stale, with a caption in the last CAM_FRESH_S)
  paused: string[]; // cameras paused for privacy: neither on nor quiet
  age: number | null; // seconds since the newest scene entry from any camera
}

export function micStatus(ambient: Ambient | undefined, now: number): MicStatus {
  if (!ambient?.ts) return { state: 'off', age: null };
  const age = Math.max(0, now - ambient.ts);
  return { state: age < MIC_FRESH_S ? 'on' : age < STALE_LIMIT_S ? 'stale' : 'off', age };
}

export function camStatus(
  scene: SceneEntry[] | undefined,
  cams: string[],
  now: number,
  vision?: VisionStatus,
  paused: readonly string[] = [],
): CamStatus {
  if (!cams.length) return { state: 'off', live: [], paused: [], age: null };
  const newest: Record<string, number> = Object.create(null);
  for (const e of scene ?? []) newest[e.cam] = Math.max(newest[e.cam] ?? 0, e.ts);
  const seen = cams.map(c => newest[c]).filter((t): t is number => t != null);
  const age = seen.length ? Math.max(0, now - Math.max(...seen)) : null;
  const off = cams.filter(c => paused.includes(c));
  const expected = cams.filter(c => !paused.includes(c));
  const frames = vision?.ts != null && now - vision.ts <= VISION_FRESH_S;
  const sending = (c: string) => {
    const said = vision?.cameras && Object.hasOwn(vision.cameras, c) ? vision.cameras[c] : undefined;
    return said?.state === 'on';
  };
  const captioned = (c: string) => newest[c] != null && now - (newest[c] ?? 0) <= CAM_FRESH_S;
  const live = expected.filter(frames ? sending : captioned);
  let state: Health;
  if (!expected.length) state = 'off';
  else if (live.length === expected.length) state = 'on';
  else if (live.length) state = 'some';
  else state = !frames && age != null && age < STALE_LIMIT_S ? 'stale' : 'off';
  return { state, live, paused: off, age };
}

/** A short age: "12 s", "4 min", "2 h". */
export function shortAge(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${Math.round(seconds / 3600)} h`;
}

export function micText(s: MicStatus): string {
  if (s.state === 'on') return 'Listening';
  if (s.state === 'stale') return `Microphone is late: last update ${shortAge(s.age ?? 0)} ago`;
  return s.age == null ? 'Not listening' : `Not listening: last update ${shortAge(s.age)} ago`;
}

/** What vision says about cameras that aren't delivering yet: "retrying" when one is being reopened, "starting"
 *  when one hasn't had its first frame, else null (on, or vision isn't running). */
export function camTrouble(vision: VisionStatus | undefined, now: number): 'starting' | 'retrying' | null {
  if (!vision?.ts || now - vision.ts > VISION_FRESH_S) return null;
  const states = Object.values(vision.cameras ?? {}).map(c => c.state);
  return states.includes('retrying') ? 'retrying' : states.includes('starting') ? 'starting' : null;
}

export function camText(s: CamStatus, cams: string[], trouble: 'starting' | 'retrying' | null = null): string {
  const n = cams.length;
  if (n && s.paused.length === n) return 'Paused for privacy';
  if (s.state !== 'on' && trouble === 'retrying') return 'Camera not responding, retrying';
  if (s.state !== 'on' && trouble === 'starting') return 'Starting cameras…';
  // Quiet: no frames arriving. A paused camera says so instead.
  const quiet = cams.filter(c => !s.live.includes(c) && !s.paused.includes(c));
  const notes = [s.paused.length && `${s.paused.join(', ')} paused`, quiet.length && `${quiet.join(', ')} quiet`]
    .filter(Boolean)
    .join('; ');
  if (s.state === 'on' && !s.paused.length) return n === 1 ? 'Camera on' : `All ${n} cameras on`;
  if (s.state === 'on' || s.state === 'some') return `${s.live.length} of ${n} cameras on (${notes})`;
  if (s.state === 'stale') return `Cameras are late: last scene ${shortAge(s.age ?? 0)} ago`;
  return 'Cameras off';
}
