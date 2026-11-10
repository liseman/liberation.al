// One shared audio player for clips, from any tab: play(url, key) plays, and the same key again stops it. The
// reactive state says what's playing (key), which transcript line to highlight (ts), and a status line. Space
// pauses and resumes outside text fields.
//
// Audio the dashboard plays comes out of the speakers, so while it plays the server is told (/playback/start) to keep
// the room microphone from transcribing it back in. One window is open at a time, closed when the audio stops or ends.
import { create } from 'zustand';
import type { TranscriptEntry } from '../live/types';
import { hms } from './format';

export interface PlayerState {
  key: string | null;
  ts: number | null;
  paused: boolean;
  status: string;
  position: number; // seconds into what's playing, for a player that shows where it is (Calls board)
  duration: number | null; // its length in seconds, once known
}

const IDLE: PlayerState = { key: null, ts: null, paused: false, status: '', position: 0, duration: null };

export const usePlayer = create<PlayerState>(() => IDLE);

let audio: HTMLAudioElement | null = null;
let playingKey: string | null = null; // set at once; the store lags the media events
let guardToken: string | null = null;

const set = (p: Partial<PlayerState>) => usePlayer.setState(p);

/** The server's reason a clip can't play (its JSON error), or the browser's. */
async function whyNot(url: string, fallback: string): Promise<string> {
  try {
    const r = await fetch(url, { cache: 'no-store' });
    if ((r.headers.get('Content-Type') ?? '').includes('json')) {
      const body: unknown = await r.json();
      if (body && typeof body === 'object' && 'error' in body && typeof body.error === 'string') return body.error;
    }
  } catch {
    // fall back to the browser's reason
  }
  return fallback;
}

const post = (url: string, body: unknown) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

function guardOpen(seconds: number) {
  if (guardToken) return;
  post('/playback/start', { seconds: Math.ceil(seconds || 10) })
    .then(r => r.json())
    .then((j: { ok?: boolean; token?: string }) => {
      if (j.ok && j.token) guardToken = j.token;
    })
    .catch(() => {});
}

function guardClose() {
  if (!guardToken) return;
  const token = guardToken;
  guardToken = null;
  post('/playback/stop', { token }).catch(() => {});
}

// Inputs a person types into. Any other input (a checkbox, switch, radio, or slider) is a control.
const TYPED = new Set([
  '',
  'text',
  'search',
  'email',
  'number',
  'password',
  'tel',
  'url',
  'date',
  'datetime-local',
  'month',
  'time',
  'week',
]);

const isSpace = (e: KeyboardEvent) => e.key === ' ' || e.code === 'Space';

/** Text editing, modified/consumed keys, and visible modal dialogs own their keyboard input. With `spaceOverControls`
 *  (Review), Space belongs to playback from any focused control but a text field: a focused chip, switch, or
 *  checkbox no longer toggles; only typing keeps it. */
export function playbackKeyBlocked(e: KeyboardEvent, { spaceOverControls = false } = {}): boolean {
  if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return true;
  const target = e.target instanceof Element ? e.target : null;
  const field = target?.closest(
    'input, textarea, select, [contenteditable]:not([contenteditable="false"]), [data-review-search]',
  );
  const control = field instanceof HTMLInputElement && !TYPED.has(field.type);
  // A checkbox in a [data-review-keys] area (Show only's pills) keeps Space and Enter for itself and lets the
  // playback keys through, as that area's buttons do.
  const passes =
    (spaceOverControls && control && isSpace(e)) ||
    (field instanceof HTMLInputElement &&
      field.type === 'checkbox' &&
      field.closest('[data-review-keys]') !== null &&
      !isSpace(e) &&
      e.key !== 'Enter');
  if ((field && !passes) || (target as HTMLElement | null)?.isContentEditable) return true;
  return [...document.querySelectorAll<HTMLElement>('[role="dialog"][aria-modal="true"], dialog[open]')].some(
    dialog => {
      if (typeof dialog.checkVisibility === 'function') return dialog.checkVisibility();
      // jsdom and older browsers lack checkVisibility; a kept-mounted drawer can be hidden by an ancestor.
      for (let el: HTMLElement | null = dialog; el; el = el.parentElement) {
        const style = getComputedStyle(el);
        if (el.hidden || style.display === 'none' || style.visibility === 'hidden') return false;
      }
      return true;
    },
  );
}

function onKey(e: KeyboardEvent) {
  const target = e.target instanceof Element ? e.target : null;
  if ((e.key !== ' ' && e.code !== 'Space') || playbackKeyBlocked(e) || target?.closest('button')) return;
  if (e.repeat) {
    if (playingKey) e.preventDefault();
    return;
  }
  if (toggle()) e.preventDefault();
}

function element(): HTMLAudioElement {
  if (audio) return audio;
  const a = document.createElement('audio');
  a.setAttribute('playsinline', '');
  a.preload = 'auto';
  a.hidden = true;
  document.body.append(a);
  a.addEventListener('ended', () => stop());
  a.addEventListener('error', async () => {
    if (!playingKey) return; // clearing src after a stop raises an error too
    const url = a.src;
    stop(true);
    set({ status: `Can't play this: ${await whyNot(url, 'the clip failed to load')}` });
  });
  a.addEventListener('pause', () => {
    if (playingKey && !a.ended) set({ paused: true });
  });
  a.addEventListener('play', () => set({ paused: false }));
  a.addEventListener('timeupdate', () => {
    if (playingKey) set({ position: a.currentTime });
  });
  a.addEventListener('loadedmetadata', () => {
    if (!playingKey) return;
    const d = a.duration;
    set({ duration: Number.isFinite(d) ? d : null });
    const ts = usePlayer.getState().ts;
    set({ status: (ts ? hms(ts) : 'Playing') + (Number.isFinite(d) ? `, ${d.toFixed(1)} s` : '') });
  });
  a.addEventListener('playing', () => guardOpen(Math.max(10, (a.duration || 10) - a.currentTime)));
  a.addEventListener('pause', guardClose);
  a.addEventListener('ended', guardClose);
  document.addEventListener('keydown', onKey);
  audio = a;
  return a;
}

/** Stop playing; keepStatus leaves the status line (an error) in place. */
export function stop(keepStatus = false): void {
  playingKey = null;
  if (audio) {
    try {
      audio.pause();
    } catch {
      // nothing was playing
    }
    audio.removeAttribute('src');
    try {
      audio.load();
    } catch {
      // nothing to reset
    }
  }
  usePlayer.setState({ ...IDLE, status: keepStatus ? usePlayer.getState().status : '' });
}

/** Play url. key names what's playing so the same control stops it; ts marks the transcript line to highlight. */
export function play(url: string, key: string, ts: number | null = null): void {
  if (playingKey === key) {
    stop();
    return;
  }
  const a = element();
  playingKey = key;
  a.src = url;
  usePlayer.setState({ key, ts, paused: false, status: 'Loading', position: 0, duration: null });
  a.play()?.catch(async (e: unknown) => {
    if (playingKey !== key) return; // stopped or replaced meanwhile
    // A DOMException isn't always an Error (jsdom's isn't), so read its name off any object.
    const name = e && typeof e === 'object' && 'name' in e && typeof e.name === 'string' ? e.name : String(e);
    const why = name === 'NotAllowedError' ? 'the browser blocked autoplay; click again' : await whyNot(url, name);
    set({ status: `Can't play this: ${why}` });
    stop(true);
  });
}

/** What playLine plays a line under: its key, so the other microphone's line in that millisecond isn't it. */
export const lineKey = (e: TranscriptEntry): string => (e.line ? `line:${e.line}` : `ts:${e.ts_start}`);

/** Play one transcript line's audio: by its key when it has one, else by its time as before keys. */
export const playLine = (e: TranscriptEntry): void =>
  play(
    e.line ? `/clip?line=${encodeURIComponent(e.line)}` : `/clip?ts_start=${encodeURIComponent(e.ts_start)}`,
    lineKey(e),
    e.ts_start,
  );

/** Move the highlight while key plays (Review follows the playhead along the transcript). */
export function setPlayingTs(key: string, ts: number): void {
  if (playingKey === key) set({ ts });
}

/** Move what's playing under key to `seconds` in (a player's position bar). Nothing happens for another key. */
export function seek(key: string, seconds: number): void {
  if (!audio || playingKey !== key || !Number.isFinite(seconds)) return;
  audio.currentTime = Math.max(0, seconds);
  set({ position: audio.currentTime });
}

/** Pause or resume what's playing; false when nothing is. */
export function toggle(): boolean {
  if (!audio || !playingKey) return false;
  if (audio.paused) audio.play()?.catch(() => {});
  else audio.pause();
  return true;
}
