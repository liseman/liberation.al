// What a call row says (the Calls notes, #2621 board): which way the call went and with whom, when, and how long,
// in the app's words. Pure, so the row stays thin and this is tested on its own.
import { capitalize, dayWords, hm, isoDay, secondsText } from '../lib/format';
import type { Call, Person } from '../people/api';

/** Which way a call went: one the assistant placed, one it answered, or null for a call from before that was saved. */
export function callDirection(c: Call): 'out' | 'in' | null {
  return c.direction === 'out' || c.direction === 'in' ? c.direction : null;
}

const digits = (n: string): string => n.replace(/[^\d+]/g, '');

/** Who a number is: the person in People with that number, else the number itself; null for no number. */
export function callWho(number: string | null | undefined, people: readonly Person[]): string | null {
  const n = number?.trim();
  if (!n) return null;
  return people.find(p => p.number && digits(p.number) === digits(n))?.name ?? n;
}

/** The row's title: "Called Dana", "Dana called", or, with no direction saved, who it was with. */
export function callTitle(c: Call, people: readonly Person[]): string {
  const way = callDirection(c);
  if (way === 'out') return `Called ${c.to?.name?.trim() || callWho(c.to?.number, people) || 'someone'}`;
  const from = callWho(c.from, people);
  if (way === 'in') return `${from ?? 'Someone'} called`;
  return from ?? 'Unknown caller';
}

/** When the call started, in seconds, or null. */
export const callStart = (c: Call): number | null => {
  const t = c.started ? Date.parse(c.started) : Number.NaN;
  return Number.isFinite(t) ? t / 1000 : null;
};

/** The row's second line: "13:02 · 3 min · with Dana". The day is its heading's. Who the title names already
 *  ("Dana called") isn't said again. */
export function callMeta(c: Call, title = ''): string {
  const start = callStart(c);
  const who = Object.keys(c.speakers ?? {}).filter(name => !title.includes(name));
  return [
    start == null ? null : hm(start),
    c.durationSeconds ? secondsText(c.durationSeconds) : null,
    who.length ? `with ${who.join(', ')}` : null,
  ]
    .filter(Boolean)
    .join(' · ');
}

/** The local day a call is listed under (YYYY-MM-DD), from its start, else the day it was saved under. */
export const callDay = (c: Call): string => {
  const start = callStart(c);
  return start == null ? c.day : isoDay(new Date(start * 1000));
};

/** A day's heading: "Today", "Yesterday", "Mon 3 Nov", and the year only when it isn't this one. */
export function dayHeading(day: string, now = new Date()): string {
  const words = capitalize(dayWords(day, now));
  const year = Number(day.slice(0, 4));
  return year && year !== now.getFullYear() ? `${words} ${year}` : words;
}
