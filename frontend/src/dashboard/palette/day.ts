// The command palette's "go to a day in Review": a typed day read as a date, as people say it. "today",
// "yesterday", a weekday ("monday" is the last one, today included), an ISO date ("2026-11-03"), or a month and day
// ("nov 3", "3 november", "11/3"), taken as this year's unless that's still ahead, then last year's.
import { isoDay } from '../lib/format';

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];
const WEEKDAYS = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];

const monthOf = (word: string): number => (word.length >= 3 ? MONTHS.indexOf(word.slice(0, 3)) : -1);

/** A real calendar day (no 31 Feb), or null. */
function made(year: number, month: number, day: number): Date | null {
  const d = new Date(year, month, day, 12);
  return d.getFullYear() === year && d.getMonth() === month && d.getDate() === day ? d : null;
}

/** A month and day without a year: this year's, or last year's when this year's is still to come. */
function recent(month: number, day: number, now: Date): Date | null {
  const d = made(now.getFullYear(), month, day);
  if (!d) return null;
  // By day, not by time: today's date typed before noon is today, not still ahead (the review).
  return isoDay(d) > isoDay(now) ? made(now.getFullYear() - 1, month, day) : d;
}

/** The day `query` names, as YYYY-MM-DD, or null when it names none (or one still ahead). */
export function dayOf(query: string, now: Date = new Date()): string | null {
  const q = query.trim().toLowerCase().replace(/,/g, ' ').replace(/\s+/g, ' ');
  if (!q) return null;
  let d: Date | null = null;
  if (q === 'today') d = now;
  else if (q === 'yesterday') d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1, 12);
  else if (WEEKDAYS.some(w => w.startsWith(q) && q.length >= 3)) {
    const want = WEEKDAYS.findIndex(w => w.startsWith(q));
    const back = (now.getDay() - want + 7) % 7;
    d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - back, 12);
  } else {
    const iso = q.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
    const slash = q.match(/^(\d{1,2})\/(\d{1,2})$/);
    const monthFirst = q.match(/^([a-z]+) (\d{1,2})$/);
    const dayFirst = q.match(/^(\d{1,2}) ([a-z]+)$/);
    if (iso) d = made(Number(iso[1]), Number(iso[2]) - 1, Number(iso[3]));
    else if (slash) d = recent(Number(slash[1]) - 1, Number(slash[2]), now);
    else if (monthFirst && monthOf(monthFirst[1] ?? '') >= 0)
      d = recent(monthOf(monthFirst[1] ?? ''), Number(monthFirst[2]), now);
    else if (dayFirst && monthOf(dayFirst[2] ?? '') >= 0)
      d = recent(monthOf(dayFirst[2] ?? ''), Number(dayFirst[1]), now);
  }
  if (!d) return null;
  const day = isoDay(d);
  return day > isoDay(now) ? null : day; // ISO days sort as dates
}
