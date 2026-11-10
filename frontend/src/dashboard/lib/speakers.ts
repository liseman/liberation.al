// Speakers: a stable colour per voice, what a speaker's badge says, and who said a line.
import { resolveId } from '../live/reduce';
import type { Speakers, TranscriptEntry } from '../live/types';

export const PALETTE = [
  'hsl(32 90% 62%)',
  'hsl(172 55% 58%)',
  'hsl(48 85% 60%)',
  'hsl(355 75% 68%)',
  'hsl(200 70% 65%)',
  'hsl(280 55% 70%)',
  'hsl(130 50% 60%)',
  'hsl(12 85% 64%)',
  'hsl(90 50% 62%)',
  'hsl(220 60% 72%)',
];
const UNKNOWN = 'hsl(220 10% 70%)';

/** A stable colour per speaker id: numeric ids index the palette, other ids are hashed. */
export function spkColor(id: number | string | null | undefined): string {
  if (id == null) return UNKNOWN;
  let n = Number.parseInt(String(id), 10);
  if (Number.isNaN(n)) {
    n = 0;
    for (const c of String(id)) n = (n * 31 + c.charCodeAt(0)) >>> 0;
  }
  return PALETTE[Math.abs(n) % PALETTE.length] ?? UNKNOWN;
}

/** What a badge says: the number for an unnamed voice ("Speaker 4" is "4"), "?" when unknown, else the name. */
export function badgeText(label: string | null | undefined): string {
  const l = String(label ?? '').trim();
  const m = /^Speaker (\d+)$/.exec(l);
  if (m?.[1]) return m[1];
  return !l || /^(unknown|multiple speakers)$/i.test(l) ? '?' : l;
}

/** What a voice no person was matched to is called: never its placeholder ("Speaker 261"), "unknown", or "multiple
 *  speakers" (#2580). */
export const UNMATCHED = 'Unmatched';

/** Whether a label is someone's name, not a placeholder for a voice nobody was matched to. */
export function isNamed(label: string | null | undefined): boolean {
  const l = String(label ?? '').trim();
  return !!l && badgeText(l) === l;
}

/** The name to show for a voice: the person's, else Unmatched. Live, Review, the timeline, and Ask all show this. */
export const shownName = (label: string | null | undefined): string =>
  isNamed(label) ? String(label).trim() : UNMATCHED;

/** The voices with one of them (after merges) called `name`: a voice just named shows its name at once, before the
 *  live feed sends the voices again. */
export function relabel(speakers: Speakers, id: number | string, name: string): Speakers {
  const key = String(resolveId(speakers, id));
  const was = speakers.byId[key];
  if (!was || was.label === name) return speakers;
  const now = { ...was, label: name };
  return {
    ...speakers,
    list: speakers.list.map(s => (s === was ? now : s)),
    byId: { ...speakers.byId, [key]: now },
  };
}

export const EMPTY_SPEAKERS: Speakers = { list: [], byId: {}, merged: {} };

/** Who said a line: the voice's current name (after merges), else the name on the line, else "unknown". */
export function speakerName(speakers: Speakers, e: TranscriptEntry): string {
  const id = resolveId(speakers, e.speaker_id);
  const s = id == null ? undefined : speakers.byId[String(id)];
  return s ? s.label : e.speaker || 'unknown';
}
