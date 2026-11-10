// The Debug tab's buffer of service log lines from GET /logs/stream (dashboard/logtail.py), kept in the query cache
// under DEBUG_KEY. Pure functions, so the tab and its tests share them.

export const DEBUG_KEY = ['debug', 'logs'] as const;
export const CAP = 5000; // lines kept in the browser, so a chatty service can't grow the buffer without bound
export const AROUND = 10; // lines shown on each side of a clicked timestamp

/**
 * Log levels, lowest first. Every service logs "<stamp> [service] LEVEL message", the stamp an ISO
 * local date and time with offset (2026-11-03T02:20:15-08:00) or, in lines from before it, "HH:MM:SS".
 */
export const LEVELS = ['debug', 'info', 'warn', 'error'] as const;
export type Level = (typeof LEVELS)[number];

const RANK: Record<Level, number> = { debug: 0, info: 1, warn: 2, error: 3 };
// Python's own names for two of them, which its loggers may still print.
const NAMES: Readonly<Record<string, Level>> = {
  DEBUG: 'debug',
  INFO: 'info',
  WARN: 'warn',
  WARNING: 'warn',
  ERROR: 'error',
  CRITICAL: 'error',
};
const STAMP = String.raw`(?:\d{4}-\d\d-\d\dT)?\d\d:\d\d:\d\d(?:[+-]\d\d:\d\d)?`;
const LEVEL_AT = new RegExp(String.raw`^(?:${STAMP} )?(?:\[[^\]]*\] )?([A-Z]+)(?: |$)`);
const STAMPED = new RegExp(String.raw`^${STAMP}(?![\d:])`); // a line of its own; one without a time continues the line before

/** The level a line states, after its optional time and [service] tag, or null when it states none. */
export function levelOf(text: string): Level | null {
  const word = LEVEL_AT.exec(text)?.[1];
  return word === undefined ? null : (NAMES[word] ?? null);
}

export const isLevel = (v: unknown): v is Level => typeof v === 'string' && (LEVELS as readonly string[]).includes(v);

/** A line as the server sends it: service, text, and when the server read it (null in the backlog). */
export type WireLine = { s: string; l: string; t: number | null };
/** A line in the buffer: its level, and whether that came from an earlier line of the service (see receive). */
export type LogLine = WireLine & { id: number; lv: Level; inherited: boolean };
export type Hello = { ts: number; services: string[]; missing: string[] };

export type LogState = {
  services: string[]; // every service streamed, in services.toml order
  missing: string[]; // services with no log file on this machine yet
  lines: LogLine[]; // shown, oldest first, at most CAP
  held: LogLine[]; // arrived while paused, at most CAP
  paused: boolean;
  dropped: number; // lines the cap let go, so the tab can say the buffer is full
  connects: number; // hello events so far: a backlog after a reconnect adds only lines not already here
  nextId: number;
  last: Readonly<Record<string, Level>>; // each service's latest level, for the lines after it that state none
};

export const EMPTY: LogState = {
  services: [],
  missing: [],
  lines: [],
  held: [],
  paused: false,
  dropped: 0,
  connects: 0,
  nextId: 0,
  last: {},
};

function capped(lines: LogLine[]): [LogLine[], number] {
  return lines.length > CAP ? [lines.slice(-CAP), lines.length - CAP] : [lines, 0];
}

export function hello(state: LogState, h: Hello): LogState {
  return { ...state, services: h.services, missing: h.missing, connects: state.connects + 1 };
}

const OVERLAP = 50; // lines compared when matching a reconnect's backlog to what is already shown

/**
 * A batch without the backlog lines (t null) a reconnect sent again. For each service, the backlog is matched to
 * the lines already shown by the longest run that ends with the last one shown, and only what follows that run is
 * new. Matching on the last line alone would drop new output that repeats it (the same error after a new try). A
 * tie goes to the earliest match, so when it can't be told, a line shows twice rather than goes missing.
 */
function unseen(state: LogState, batch: WireLine[]): WireLine[] {
  const have = [...state.lines, ...state.held];
  const keep = new Set<number>();
  const backlog = new Map<string, number[]>(); // service: indexes of its backlog lines in batch
  batch.forEach((w, i) => {
    if (w.t !== null) keep.add(i);
    else backlog.set(w.s, [...(backlog.get(w.s) ?? []), i]);
  });
  for (const [service, at] of backlog) {
    const shown = have
      .filter(x => x.s === service)
      .slice(-OVERLAP)
      .map(x => x.l);
    const sent = at.map(i => batch[i]?.l);
    let end = -1;
    let longest = 0;
    for (let j = 0; j < sent.length; j++) {
      let m = 0;
      while (m < shown.length && m <= j && sent[j - m] === shown[shown.length - 1 - m]) m++;
      if (m > longest) [end, longest] = [j, m];
    }
    at.forEach((i, k) => {
      if (k > end) keep.add(i);
    });
  }
  return batch.filter((_, i) => keep.has(i));
}

/**
 * Add a batch of lines: shown at once, or held while paused. A line that states no level but starts with a time (a
 * service's own print, an old line) is info. One with no time either (a traceback, third-party output) continues the
 * service's line before it and takes its level, so a traceback after an error is an error; a service's first such
 * line is info.
 */
export function receive(state: LogState, batch: WireLine[]): LogState {
  const fresh = state.connects > 1 ? unseen(state, batch) : batch;
  const last: Record<string, Level> = { ...state.last };
  const added = fresh.map((w, i): LogLine => {
    const own = levelOf(w.l) ?? (STAMPED.test(w.l) ? 'info' : null);
    const lv = own ?? last[w.s] ?? 'info';
    last[w.s] = lv;
    return { s: w.s, l: w.l, t: w.t, id: state.nextId + i, lv, inherited: own === null };
  });
  const nextId = state.nextId + added.length;
  // A service that started writing after the stream opened has a log now; one the hello didn't name gets a chip.
  const heard = new Set(batch.map(w => w.s));
  const missing = state.missing.filter(s => !heard.has(s));
  const services = [...state.services, ...[...heard].filter(s => !state.services.includes(s))];
  if (state.paused) {
    const [held, lost] = capped([...state.held, ...added]);
    return { ...state, held, missing, services, dropped: state.dropped + lost, nextId, last };
  }
  const [lines, lost] = capped([...state.lines, ...added]);
  return { ...state, lines, missing, services, dropped: state.dropped + lost, nextId, last };
}

export const pause = (state: LogState): LogState => ({ ...state, paused: true });

export function resume(state: LogState): LogState {
  const [lines, lost] = capped([...state.lines, ...state.held]);
  return { ...state, lines, held: [], paused: false, dropped: state.dropped + lost };
}

export const clear = (state: LogState): LogState => ({ ...state, lines: [], held: [], dropped: 0 });

export type View = { hidden: ReadonlySet<string>; query: string; mode: 'filter' | 'highlight'; min: Level };

export const matches = (line: WireLine, query: string): boolean =>
  !!query && line.l.toLowerCase().includes(query.toLowerCase());

/** The lines the service toggles and the search let through, whatever their level. */
function unfiltered(lines: LogLine[], view: View): LogLine[] {
  const q = view.query.trim();
  return lines.filter(x => !view.hidden.has(x.s) && (view.mode !== 'filter' || !q || matches(x, q)));
}

/** The lines to show: services not hidden, at least the minimum level, and in filter mode only the ones matching. */
export function visible(lines: LogLine[], view: View): LogLine[] {
  return unfiltered(lines, view).filter(x => RANK[x.lv] >= RANK[view.min]);
}

/** How many lines of each level the other filters let through, so the level control can say what each one shows. */
export function levelCounts(lines: LogLine[], view: View): Record<Level, number> {
  const counts: Record<Level, number> = { debug: 0, info: 0, warn: 0, error: 0 };
  for (const x of unfiltered(lines, view)) counts[x.lv]++;
  return counts;
}

/** The lines around one (every service, ignoring the filter), with it in the middle. */
export function around(lines: LogLine[], id: number, radius = AROUND): LogLine[] {
  const i = lines.findIndex(x => x.id === id);
  return i < 0 ? [] : lines.slice(Math.max(0, i - radius), i + radius + 1);
}

const PALETTE = [
  'blue',
  'teal',
  'grape',
  'orange',
  'cyan',
  'pink',
  'lime',
  'indigo',
  'yellow',
  'red',
  'violet',
  'green',
];

/** A Mantine color for a service: by its place in the service list, so it stays the same while the page is open. */
export function colorOf(services: string[], s: string): string {
  const i = services.indexOf(s);
  const n = i >= 0 ? i : [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 0);
  return PALETTE[n % PALETTE.length] ?? 'gray';
}

/** A line's time as HH:MM:SS, or "earlier" for a backlog line (read before the page connected). */
export function clock(t: number | null): string {
  if (t === null) return 'earlier';
  return new Date(t * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

/** The Debug tab's service chips, grouped in sections by what each service does. A service no section
 *  names, a new one, lands in "Other", so none is ever dropped. */
export const SERVICE_GROUPS: readonly { title: string; services: readonly string[] }[] = [
  { title: 'Capture and cameras', services: ['continuum-capture', 'video-window'] },
  {
    title: 'Vision',
    services: ['see-decode', 'see-actions', 'see-zones', 'see-objects', 'see-captions', 'surroundings'],
  },
  {
    title: 'Audio and speech',
    services: ['listen', 'whisper', 'whisper-names', 'deep-transcribe', 'voice', 'phone-bridge'],
  },
  { title: 'People and memory', services: ['identity', 'journal'] },
  { title: 'Ask and research', services: ['ask-ollama', 'research'] },
  { title: 'Imports', services: ['ingest', 'ingest-analyze'] },
  {
    title: 'Broker and storage',
    services: ['continuum-broker', 'continuum-gateway', 'continuum-tier-worker', 'continuum-vector-worker'],
  },
  { title: 'Devices', services: ['bluetooth', 'network-devices'] },
  { title: 'Dashboard', services: ['dashboard'] },
];

/** `services` in their sections: each section's in its listed order, empty sections left out, and the rest in
 *  "Other" in the order given. */
export function groupServices(services: readonly string[]): { title: string; services: string[] }[] {
  const have = new Set(services);
  const placed = new Set(SERVICE_GROUPS.flatMap(g => g.services));
  const out = SERVICE_GROUPS.map(g => ({ title: g.title, services: g.services.filter(s => have.has(s)) }));
  out.push({ title: 'Other', services: services.filter(s => !placed.has(s)) });
  return out.filter(g => g.services.length > 0);
}
