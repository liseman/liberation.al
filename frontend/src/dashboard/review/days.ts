// The days Review can show something on (GET /review/days): video, room audio, or transcript lines. Previous day and
// Next day step between them, and Review opens on the latest when today has nothing yet. Days are YYYY-MM-DD, so they
// sort and compare as strings.

export interface RecordedDay {
  day: string;
  video: boolean;
  audio: boolean;
  transcript: boolean;
}

/** The calendar day before (-1) or after (+1) day, for when the recorded days aren't known. */
export function calendarStep(day: string, dir: -1 | 1): string {
  const d = new Date(`${day}T12:00:00`); // noon: a daylight-saving change never moves it to another day
  d.setDate(d.getDate() + dir);
  return [d.getFullYear(), d.getMonth() + 1, d.getDate()].map((n, i) => String(n).padStart(i ? 2 : 4, '0')).join('-');
}

/** The nearest recorded day before (-1) or after (+1) day, or null when there is none that way. With the recorded days
 *  unknown (still loading, or the list failed), the calendar day. */
export function stepDay(days: readonly string[] | undefined, day: string, dir: -1 | 1): string | null {
  if (!days) return calendarStep(day, dir);
  if (dir < 0) return days.findLast(d => d < day) ?? null;
  return days.find(d => d > day) ?? null;
}

/** Where Review should open instead of day, an empty today: the latest recorded day up to upTo (today, the later of
 *  the browser's and the server's), or a later one only when the clock is behind every recording. Null to stay: day
 *  has something, or nothing is recorded at all. */
export function landingDay(days: readonly string[], day: string, upTo = day): string | null {
  if (!days.length || days.includes(day)) return null;
  return days.findLast(d => d <= upTo) ?? days[days.length - 1] ?? null;
}

/** What a recorded day has, for its label in the picker: "video, room audio, and transcript". */
export function kinds(d: RecordedDay): string {
  const k = [d.video && 'video', d.audio && 'room audio', d.transcript && 'transcript'].filter((x): x is string => !!x);
  return k.length < 3 ? k.join(' and ') : `${k.slice(0, -1).join(', ')}, and ${k[k.length - 1]}`;
}
