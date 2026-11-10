// Settings' Devices section (#384): the pure parts. Settings' draft holds list settings as text joined with
// ", " (vision.places "cam0: front camera, cam1: rear camera"; vision.cams "0, 2"),
// or as JSON when an item holds a comma (model.ts listText).
import { listItems, listText } from '../model';

// Exact manifest names win before legacy numbered aliases, as in vision/places.py.
const ENTRY = /^\s*(.*?)\s*[:=]\s*(.*?)\s*$/s;
const LEGACY = /^(?:cam)?\s*(\d+)$/i;
const CAMERA_ID = /^[A-Za-z0-9_-]+$/;

/** vision.places keyed by exact configured ID, retaining compatibility with legacy numbered aliases. */
export function parsePlaces(
  text: string | boolean | undefined,
  cameraIds: Iterable<string> = [],
): Record<string, string> {
  const known = new Set(cameraIds);
  const out: Record<string, string> = Object.create(null);
  for (const item of listItems(text)) {
    const m = item.match(ENTRY);
    if (!m?.[1] || !m[2]) continue;
    let cam = m[1];
    const legacy = cam.match(LEGACY);
    if (!known.has(cam) && legacy) cam = `cam${Number(legacy[1])}`;
    if (CAMERA_ID.test(cam)) out[cam] = m[2].split(/\s+/).join(' ');
  }
  return out;
}

/** vision.places text with camera cam named name (an empty name drops it). */
export function withPlace(
  text: string | boolean | undefined,
  cam: string,
  name: string,
  cameraIds: Iterable<string> = [],
): string {
  const places = { ...parsePlaces(text, cameraIds), [cam]: name.replace(/,/g, ' ').trim() };
  return listText(
    Object.entries(places)
      .filter(([, n]) => n)
      .sort(([a], [b]) => a.localeCompare(b, 'en', { numeric: true }) || (a < b ? -1 : a > b ? 1 : 0))
      .map(([c, n]) => `${c}: ${n}`),
  );
}

/** A camera's status line, from capture's word on it (its stream on the broker, fresh) and the picker's row. */
export function cameraStatus(
  row: { present: boolean; selected: boolean },
  vision: { state?: string; size?: number[] | null; fps?: number | null } | undefined,
): { text: string; tone: 'ok' | 'warn' | 'dim' } {
  if (!row.present) return { text: 'Not plugged in', tone: 'warn' };
  if (!row.selected) return { text: 'Not used', tone: 'dim' };
  if (!vision) return { text: 'Off: capture is sending nothing from it', tone: 'dim' };
  if (vision.state === 'retrying') return { text: 'Not responding, retrying', tone: 'warn' };
  if (vision.state === 'starting') return { text: 'Starting…', tone: 'dim' };
  const [w, h] = vision.size ?? [];
  return {
    text: w && h ? `On, ${w}×${h}${vision.fps ? ` at ${vision.fps} fps` : ''}` : 'On',
    tone: 'ok',
  };
}
