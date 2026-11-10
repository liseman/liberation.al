// The transcript's pure logic, shared by Live and Review: lines grouped into turns, and the words a turn shows.
import { isNamed, shownName, speakerName } from '../lib/speakers';
import { resolveId } from '../live/reduce';
import type { Speakers, TranscriptEntry } from '../live/types';
import { createSearchIndex, type SearchIndex, searchIndex } from '../review/search';

export type TurnId = number | string | null | 'status' | 'assistant';

/** A line with its index in the list it came from (a day on Review, the recent feed on Live). */
export interface Line {
  e: TranscriptEntry;
  i: number;
}

export interface Turn {
  id: TurnId; // the (merged) voice id, null for an unknown voice, or 'status' and 'assistant' for the assistant
  lines: Line[];
  end: number;
}

// The assistant's unspoken log lines: they fold into one status line instead of a turn.
const AUTH =
  /^(That request expired|Please say your password|Password (not )?accepted|Too many attempts|Please wait one minute|Say yes confirm|Cancelled\.|Confirm this exact request)/;

export const TURN_GAP_S = 120;

const endOf = (e: TranscriptEntry) => e.ts_end ?? e.ts_start;

/** Consecutive lines from one speaker, less than two minutes apart, read as one turn. Lines no voice was matched to
 *  are one speaker while they show one name: a line tagged with a name starts a turn of its own. The assistant's
 *  spoken lines are its own turns; its unspoken log lines fold into one status turn. */
export function turnsOf(entries: readonly TranscriptEntry[], speakers: Speakers): Turn[] {
  const turns: Turn[] = [];
  entries.forEach((e, i) => {
    const auth = e.assistant && AUTH.test(String(e.text ?? ''));
    const id: TurnId = e.status || auth ? 'status' : e.assistant ? 'assistant' : resolveId(speakers, e.speaker_id);
    const t = turns.at(-1);
    const prev = t?.lines.at(-1)?.e;
    const sameName = id != null || !prev || speakerName(speakers, prev) === speakerName(speakers, e);
    if (t && t.id === id && (id === 'status' || (sameName && e.ts_start - t.end < TURN_GAP_S))) {
      t.lines.push({ e, i });
      t.end = endOf(e);
    } else turns.push({ id, lines: [{ e, i }], end: endOf(e) });
  });
  return turns;
}

/** What a speaker's avatar says: up to two initials ("Dana Ruiz" is "DR"), or "?" for a voice nobody was matched to
 *  (#2580). */
export function initials(name: string): string {
  if (!isNamed(name)) return '?';
  const words = name.trim().split(/\s+/).filter(Boolean);
  return words
    .slice(0, 2)
    .map(w => [...w][0]?.toLocaleUpperCase() ?? '')
    .join('');
}

/** A name the transcript can be corrected away from: someone's, not "unknown" or a numbered "Speaker 4". */
export const named = isNamed;

/** What a search reads of a line: the name it shows, then its words ("Dana: Coffee's ready"), the same for Review's
 *  search results and for the transcript dimmed under them (#2580). `label` is the line's speaker as stored; an
 *  unmatched voice is searched by what it shows, never by its placeholder. */
export const lineSearchText = (label: string | null | undefined, text: string): string =>
  label ? `${shownName(label)}: ${text}` : text;

/** A half-open range of UTF-16 offsets, as String.slice takes them. */
export type Range = readonly [number, number];
/** Where a search matched in a line: in the name it shows, and in its words. */
export interface LineHit {
  name: readonly Range[];
  text: readonly Range[];
}

/** A transcript's lines indexed for searching, built once per set of lines (not once per key typed). */
export interface LineSearch {
  index: SearchIndex;
  keys: readonly string[];
  /** Where each line's words start in its search text, after "Name: ". */
  starts: readonly number[];
}

export function lineSearch(entries: readonly TranscriptEntry[], speakers: Speakers): LineSearch {
  const keys: string[] = [];
  const starts: number[] = [];
  const documents = entries.map((e, i) => {
    const words = String(e.text ?? '');
    const text = lineSearchText(speakerName(speakers, e), words);
    keys.push(lineKey(e));
    starts.push(text.length - words.length);
    return { id: String(i), kind: 'transcript' as const, ts: e.ts_start, text };
  });
  return { index: createSearchIndex(documents), keys, starts };
}

/**
 * The lines a search matches, by line key, with where: Review's search (review/search.ts) over the same text its
 * results show, so a line it lists is never dimmed below them, and its typo-tolerant, accent-insensitive matches are
 * the words highlighted. Empty for no search.
 */
export function lineHits(search: LineSearch | null, query: string): ReadonlyMap<string, LineHit> {
  const out = new Map<string, LineHit>();
  if (!search || !query.trim()) return out;
  for (const hit of searchIndex(search.index, query, search.keys.length)) {
    const i = Number(hit.document.id);
    const key = search.keys[i];
    const start = search.starts[i] ?? 0;
    if (key === undefined) continue;
    const name = hit.ranges.filter(([a]) => a < start).map(([a, b]): Range => [a, Math.min(b, start - 2)]);
    const text = hit.ranges.filter(([a]) => a >= start).map(([a, b]): Range => [a - start, b - start]);
    const was = out.get(key); // two lines can share a key: keep both lines' matches
    out.set(key, was ? { name: [...was.name, ...name], text: [...was.text, ...text] } : { name, text });
  }
  return out;
}

/** A line's key: its id, or its start for a line without one. */
export const lineKey = (e: TranscriptEntry): string => String(e.line ?? e.ts_start);

/** A turn's key: its first line's key, which stays as lines are added. Two microphones can start lines in the same
 *  millisecond (#247), so the start alone isn't enough (Codex on #2516). */
export const turnKey = (t: Turn | undefined): string => {
  const first = t?.lines[0]?.e;
  return first ? lineKey(first) : '';
};

/** Whether two turns hold the very same lines. A feed keeps a line's object until the line changes, so a new line
 *  elsewhere re-renders only the turn it's in (#440). */
export const sameTurn = (a: Turn, b: Turn): boolean =>
  a.id === b.id && a.lines.length === b.lines.length && a.lines.every((l, k) => l.e === b.lines[k]?.e);
