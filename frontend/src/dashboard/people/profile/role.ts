// Text about a person: their role phrase, the roster's meta line, and the profile's dates and delete summary. Ported
// from static/dashboard/tab-people.js (roleText, personMeta, when, removedText).
import { HttpError, messageOf } from '../../../shared/http';
import { capitalize, dateTime, plural, seenStatus } from '../../lib/format';
import type { Person } from '../api';

/** A person's role and organization as one phrase ("CEO at Acme"), or '' when neither is known. */
export function roleLine(p: { role?: string | null; organization?: string | null }): string {
  if (p.role && p.organization) return `${p.role} at ${p.organization}`;
  return p.role || (p.organization ? `at ${p.organization}` : '');
}

/** One line under a name: their role, then what Liberational has heard and seen of them. */
export function personMeta(p: Person): string {
  const voices = p.voices?.length ?? 0;
  const heard = p.lines
    ? `${plural(p.lines, 'line', 'lines')} heard`
    : p.enrolled
      ? 'Voice learned from a recording'
      : voices
        ? 'Voice known'
        : p.faces
          ? 'Face known'
          : 'Nothing heard yet';
  return [
    roleLine(p),
    heard,
    (p.faces ?? 0) > 0 && (p.lines || voices) ? 'face known' : '',
    seenStatus(p),
    p.number ? `phone ${p.number}` : '',
  ]
    .filter(Boolean)
    .join(', ');
}

/** A time as a medium date and a short time, or "never". */
export const when = (ts: number | null | undefined): string => dateTime(ts);

/** A time as the card and the list say it: "today at 18:42", "yesterday at 20:12", "Oct 26 at 09:10", with the
 * year added when it isn't this year's. Local time, 24-hour. */
export function dayTime(ts: number, now = new Date()): string {
  const d = new Date(ts * 1000);
  const hm = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  const midnight = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((midnight(now) - midnight(d)) / 86400000);
  if (days === 0) return `today at ${hm}`;
  if (days === 1) return `yesterday at ${hm}`;
  const date = d.toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    ...(d.getFullYear() === now.getFullYear() ? {} : { year: 'numeric' }),
  });
  return `${date} at ${hm}`;
}

/** A time in a few characters, for small tiles: "today 19:30", "yesterday 08:05", "Mon 19:30" within the week, else
 * "Oct 12 19:30" (with the year when it isn't this year's). */
export function shortWhen(ts: number, now = new Date()): string {
  const d = new Date(ts * 1000);
  const hm = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  const midnight = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((midnight(now) - midnight(d)) / 86400000);
  if (days === 0) return `today ${hm}`;
  if (days === 1) return `yesterday ${hm}`;
  if (days > 1 && days < 7) return `${d.toLocaleDateString([], { weekday: 'short' })} ${hm}`;
  const date = d.toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    ...(d.getFullYear() === now.getFullYear() ? {} : { year: 'numeric' }),
  });
  return `${date} ${hm}`;
}

/** Where someone is, or was last seen, and which camera saw them: "Here now, seen by Front camera", "Last seen
 * yesterday at 20:12 by cam1". The cameras are views of one rig, not rooms, so the camera is named, never a place.
 * '' when there is nothing to say. */
export function whereabouts(p: {
  here_now?: { cam: string | null; place?: string | null } | null;
  last_seen?: number | null;
  last_seen_cam?: string | null;
  last_seen_place?: string | null;
}): string {
  if (p.here_now) {
    const by = cameraName(p.here_now.cam, p.here_now.place);
    return by ? `Here now, seen by ${by}` : 'Here now';
  }
  if (!p.last_seen) return '';
  const by = cameraName(p.last_seen_cam, p.last_seen_place);
  return `Last seen ${dayTime(p.last_seen)}${by ? ` by ${by}` : ''}`;
}

/** A camera as people read it: its name from Settings (vision.places, #313) with a capital, "Front camera", else its
 *  id, "cam0". The server sends the id as the name when there is none. '' with no camera. */
export function cameraName(cam: string | null | undefined, name?: string | null): string {
  const n = name?.trim();
  if (n && n !== cam) return capitalize(n);
  return cam ?? '';
}

/** What a delete removed, in one short sentence. */
export function removedText(
  name: string,
  r: { voices?: number; faces?: number; photos?: number; sightings?: number },
): string {
  const bits = [
    r.voices ? plural(r.voices, 'voice', 'voices') : '',
    r.faces ? plural(r.faces, 'face sample', 'face samples') : '',
    r.photos ? plural(r.photos, 'photo', 'photos') : '',
    r.sightings ? plural(r.sightings, 'recorded sighting', 'recorded sightings') : '',
  ].filter(Boolean);
  return `Forgot ${name}${bits.length ? `: ${bits.join(', ')}.` : '.'}`;
}

/** Why a request failed, for showing: the server's own message (or fallback when it gave none), else the error's. */
export const failure = (e: unknown, fallback: string): string =>
  e instanceof HttpError ? e.reason || fallback : messageOf(e);

/** The error a 200 reply carries when it says no, or fallback. */
export const refusal = (reply: unknown, fallback: string): string => {
  const error = reply && typeof reply === 'object' && 'error' in reply ? reply.error : null;
  return typeof error === 'string' && error ? error : fallback;
};
