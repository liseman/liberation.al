// A research report's caveats and sources in plain words, for the Inbox reader (#2457). The server writes its
// warnings for the model that answers (and for whoever built the pipeline); these say the same thing to the person
// who asked. A warning this doesn't know is shown as the server wrote it.

// Each known warning, in plain words, and whether it holds for every report (how the sources work) or says something
// about this one. Only this report's own caveats can make it "Partly sure"; the rest fold away (#2674).
const PLAIN: [RegExp, (m: RegExpMatchArray) => string, 'always' | 'this'][] = [
  [
    /sparse VLM observations/i,
    () => 'Based on camera snapshots every few minutes, so short moments can be missed.',
    'always',
  ],
  [
    /Frame decode or local VLM unavailable/i,
    () => "Some snapshots couldn't be read, so what happened then is unknown.",
    'this',
  ],
  [
    /^(\d+) gate-skipped\/error rows excluded/i,
    m => `${m[1]} ${m[1] === '1' ? 'snapshot was' : 'snapshots were'} skipped or unreadable and left out.`,
    'this',
  ],
  [/^Journal entries are summaries/i, () => 'Journal entries are summaries a model wrote, and can be wrong.', 'always'],
  [
    /^Transcripts are sampled ASR/i,
    () => 'Who said what comes from automatic transcripts, and can be wrong.',
    'always',
  ],
  [
    /^Playback windows retain only one day/i,
    () => 'Speech played through the speaker more than a day ago may be mixed in.',
    'always',
  ],
];

/** A warning in plain words, and whether it holds for every report; one the list doesn't know is about this one. */
function plainOne(w: string): { line: string; always: boolean; skipped?: number } {
  for (const [re, say, kind] of PLAIN) {
    const m = w.match(re);
    if (m)
      return { line: say(m), always: kind === 'always', ...(/gate-skipped/i.test(w) ? { skipped: Number(m[1]) } : {}) };
  }
  return { line: w.trim(), always: false };
}

/** Each warning in plain words, once each, in the order given. */
export function plainWarnings(warnings: readonly unknown[]): string[] {
  const out: string[] = [];
  for (const w of warnings) {
    if (typeof w !== 'string' || !w.trim()) continue;
    const { line } = plainOne(w);
    if (!out.includes(line)) out.push(line);
  }
  return out;
}

export interface Verdict {
  sure: 'Fairly sure' | 'Partly sure';
  why: string | null; // this report's main caveat, beside the verdict
  rest: string[]; // the other caveats, folded
}

/** "How sure is this?" in one line (#2674): "Partly sure" when the answer came out partial or this report has a caveat of
 *  its own (snapshots it couldn't read), with the main one beside it; else "Fairly sure". The caveats that hold for
 *  every report (snapshots every few minutes, transcripts can be wrong) go in the fold, so they don't read as news. */
export function verdict(status: unknown, warnings: readonly unknown[]): Verdict {
  const all = warnings.filter((w): w is string => typeof w === 'string' && !!w.trim()).map(plainOne);
  const seen = new Set<string>();
  const lines = all.filter(w => !seen.has(w.line) && seen.add(w.line));
  const own = lines.filter(w => !w.always);
  const skipped = own.find(w => w.skipped !== undefined)?.skipped;
  const unreadable = "Some snapshots couldn't be read, so what happened then is unknown.";
  let why: string | null = null;
  let used: string[] = [];
  if (skipped !== undefined) {
    why = `${skipped} ${skipped === 1 ? "snapshot couldn't" : "snapshots couldn't"} be read, so some of it is missing.`;
    used = own.filter(w => w.skipped !== undefined || w.line === unreadable).map(w => w.line);
  } else if (own[0]) {
    why = own[0].line;
    used = [own[0].line];
  } else if (typeof status === 'string' && status !== 'ok') {
    why = "Some of what it needed wasn't there.";
  }
  const partly = why !== null;
  return {
    sure: partly ? 'Partly sure' : 'Fairly sure',
    why,
    rest: lines.map(w => w.line).filter(line => !used.includes(line)),
  };
}

const NAMES: Record<string, [string, string]> = {
  camera: ['camera note', 'camera notes'],
  journal: ['journal entry', 'journal entries'],
  transcript: ['transcript line', 'transcript lines'],
  facts: ['fact', 'facts'],
  timeline: ['timeline entry', 'timeline entries'],
  detections: ['face sighting', 'face sightings'],
};

/** "20 camera notes", "1 journal entry", or "3 other" for a source this doesn't name. */
export function sourceCount(kind: string, n: number): string {
  const [one, many] = NAMES[kind] ?? [kind || 'other', kind || 'other'];
  return `${n} ${n === 1 ? one : many}`;
}

/** The evidence fold's one line: "Looked at 24 sources: 20 camera notes, 3 journal entries, and 1 transcript line". */
export function lookedLine(total: number, sources: Record<string, number>): string {
  const parts = Object.entries(sources)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([k, n]) => sourceCount(k, n));
  const head = `Looked at ${total} ${total === 1 ? 'source' : 'sources'}`;
  if (!parts.length) return head;
  const list =
    parts.length === 1
      ? parts[0]
      : parts.length === 2
        ? `${parts[0]} and ${parts[1]}`
        : `${parts.slice(0, -1).join(', ')}, and ${parts.at(-1)}`;
  return `${head}: ${list}`;
}
