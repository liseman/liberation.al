// Formatting shared by the dashboard's tabs. Times are Unix seconds.

// One formatter each, made once: toLocaleTimeString builds a new one per call, and Live formats every line's time on
// every render (a quarter of its script time).
const HMS = new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
const HM = new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit', hour12: false });

/** A time of day as 24-hour HH:MM:SS, in local time. */
export const hms = (ts: number): string => HMS.format(ts * 1000);

/** A time of day as 24-hour HH:MM, in local time. */
export const hm = (ts: number): string => HM.format(ts * 1000);

const DAY = new Intl.DateTimeFormat('en-GB', { weekday: 'short', day: 'numeric', month: 'short' });

/** When something was seen, the dashboard's usual words: "today 18:40", "yesterday 18:40", or "Mon 26 Oct 18:40". */
export function seenAt(ts: number, now = Date.now() / 1000): string {
  const d = new Date(ts * 1000);
  const day = (t: Date) => new Date(t.getFullYear(), t.getMonth(), t.getDate()).getTime();
  const days = Math.round((day(new Date(now * 1000)) - day(d)) / 86_400_000);
  if (days === 0) return `today ${hm(ts)}`;
  if (days === 1) return `yesterday ${hm(ts)}`;
  return `${DAY.format(d)} ${hm(ts)}`;
}

/** A span of time in seenAt's words: one moment when it is within a minute, the day once when it is one day
 *  ("today 13:14 to 13:18"). */
export function seenSpan(first: number, last: number, now = Date.now() / 1000): string {
  const from = seenAt(first, now);
  if (from === seenAt(last, now)) return from;
  const sameDay = new Date(first * 1000).toDateString() === new Date(last * 1000).toDateString();
  return `${from} to ${sameDay ? hm(last) : seenAt(last, now)}`;
}

/** How long ago ts was: "12 s ago", "4 min ago", "2.5 h ago", "3 d ago"; "-" for no time. */
export function ago(ts: number | null | undefined, now = Date.now() / 1000): string {
  if (!ts) return '-';
  const s = Math.max(0, now - ts);
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${(s / 3600).toFixed(1)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

/** "1 voice", "3 voices". */
export const plural = (n: number, one: string, many: string): string => `${n} ${n === 1 ? one : many}`;

/** A word or phrase with a capital first letter: "front camera" reads "Front camera". */
export const capitalize = (s: string): string => (s[0]?.toUpperCase() ?? '') + s.slice(1);

/** A length of time in seconds, the way a person says it: "45 s", "12 min", "2 h 5 min", "3 d 4 h". Minutes are
 *  rounded before they're split into hours, so it never says "1 h 60 min". */
export function secondsText(seconds: number): string {
  const s = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  if (Math.round(s) < 60) return `${Math.round(s)} s`;
  const minutes = Math.round(s / 60);
  if (minutes < 60) return `${minutes} min`;
  if (minutes < 24 * 60) {
    const h = Math.floor(minutes / 60);
    const m = minutes % 60;
    return m ? `${h} h ${m} min` : `${h} h`;
  }
  const hours = Math.round(s / 3600);
  const d = Math.floor(hours / 24);
  const h = hours % 24;
  return h ? `${d} d ${h} h` : `${d} d`;
}

/** A length of time given in minutes, as secondsText says it ("12 s", "1 h 5 min"); nothing at all is "0 min". */
export const minutesText = (minutes: number): string =>
  Number.isFinite(minutes) && minutes > 0 ? secondsText(minutes * 60) : '0 min';

/** A moment as a medium date and a short time ("Nov 4, 2026, 1:31 PM"), or `none` when there isn't one. */
export const dateTime = (ts: number | null | undefined, none = 'never'): string =>
  ts ? new Date(ts * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : none;

export const mb = (bytes: number): string => `${(bytes / 1048576).toFixed(0)} MB`;
export const gb = (bytes: number): string => `${(bytes / 1073741824).toFixed(1)} GB`;

/** A local calendar day as YYYY-MM-DD (toISOString would give the UTC day). */
export function isoDay(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

export interface SeenFields {
  last_seen?: number | null;
  last_seen_error?: unknown;
  last_seen_pending?: unknown;
}

/** When a person was last seen. Identity rematching in progress is not evidence that someone was never seen. */
export function seenStatus(person: SeenFields, format: (ts: number) => string = ago, prefix = 'seen'): string {
  if (person.last_seen_error) return 'Face history unavailable';
  if (person.last_seen_pending) return 'Face history updating';
  return person.last_seen ? `${prefix} ${format(person.last_seen)}` : '';
}

/** A clip's length as m:ss, or h:mm:ss from an hour up. */
export function clipLength(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const pad = (n: number) => String(n).padStart(2, '0');
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`;
}

/** A day as people say it: "today", "yesterday", else "Thu 5 Nov". day is YYYY-MM-DD in local time. */
export function dayWords(day: string, now = new Date()): string {
  if (day === isoDay(now)) return 'today';
  const before = new Date(now);
  before.setDate(before.getDate() - 1);
  if (day === isoDay(before)) return 'yesterday';
  return new Date(`${day}T12:00:00`).toLocaleDateString(undefined, {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
  });
}

/** A moment as people say it, in 24-hour time: "Today, 13:31", "Yesterday, 09:05", or "Wed 4 Nov, 13:31". */
export function momentWords(ts: number, now = new Date()): string {
  return `${capitalize(dayWords(isoDay(new Date(ts * 1000)), now))}, ${hm(ts)}`;
}

/** Days as a phrase: "today", "yesterday and today", "Mon 2 Nov, yesterday, and today". */
export function daysWords(days: readonly string[], now = new Date()): string {
  const words = days.map(d => dayWords(d, now));
  if (words.length <= 2) return words.join(' and ');
  return `${words.slice(0, -1).join(', ')}, and ${words.at(-1)}`;
}
