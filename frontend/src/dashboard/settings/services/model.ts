// Settings' Services data (once the top bar's Services drawer, now drawn as cards), and the rules for what
// each row says.

export const SERVICES_KEY = ['services'] as const;

/** One service from GET /services. */
export interface Service {
  name: string;
  description: string;
  group?: string; // its section: core, workers, dashboard, or extras (services.toml); anything else is Other
  state: string; // the service manager's word: active, running, activating, failed, inactive, not running, ...
  installed: boolean;
  scheduled?: boolean;
  pids: number[];
  selected: boolean; // enabled for this machine in pack.toml
  default?: boolean; // runs unless services.disable names it
  required?: boolean; // another setting needs it (Whisper for ASR), so it can't be turned off here
  restarts?: number | null; // how often the service manager restarted it; null when it didn't say
  up_s?: number | null; // how long its current run has lasted, when the manager says
  run_id?: string | null; // stable across polls, changes on a new service-manager run
  issue_acknowledged?: boolean; // browser-local acknowledgement of this occurrence while the service runs
  // The newest ERROR line in its log, "HH:MM:SS", and its time in epoch seconds when the log dates it. For a running
  // service, only one logged since its current start and within the last ten minutes.
  last_error?: { id?: string; at: string; ts?: number | null; text: string } | null;
}

/** A crash loop: restarted at least CRASH_LOOP times, and up less than CRASH_RECENT_S since the last one.
 *  The count alone can't say it, since it only resets when the service is started by hand: a service that looped
 *  last week and has run since is fine. */
export const CRASH_LOOP = 3;
export const CRASH_RECENT_S = 600;

/** Whether a service is in a crash loop now. With no uptime from the manager, the count alone decides. */
export function looping(s: Service): boolean {
  return (
    !s.issue_acknowledged &&
    s.installed &&
    (s.restarts ?? 0) >= CRASH_LOOP &&
    (s.up_s == null || s.up_s < CRASH_RECENT_S)
  );
}

/** Acknowledging a running service's history never hides a failed, stopped, or starting state. */
export function dismissible(s: Service): boolean {
  return (s.state === 'active' || s.state === 'running') && (looping(s) || !!s.last_error);
}

/** The displayed occurrence, independent of changing uptime and later unrelated log writes. */
export function issueKey(s: Service): string {
  return JSON.stringify([
    s.run_id ?? [...s.pids].sort((a, b) => a - b),
    s.restarts ?? 0,
    s.last_error?.id ?? s.last_error ?? null,
  ]);
}

/** GET /services, and the body of a POST /services reply. */
export interface ServicesView {
  ok: boolean;
  control: boolean; // whether this viewer may switch services
  platform?: string;
  error?: string | null;
  note?: string | null;
  services: Service[];
}

/** What can be done to one service. Its switch stops and starts it now (or turns on one that isn't installed):
 *  a stopped service starts again when the computer restarts. Disable takes it out of this machine's services in
 *  pack.toml and uninstalls it, so it stays off until Enable. */
export type ServiceAction = 'on' | 'stop' | 'start' | 'enable' | 'disable';

/** POST /services: one service's action, or Repair or All off every one. */
export type ServicesPost = { name: string; action: ServiceAction } | { all: BulkAction };

/** Each action's name, and what it does in words. */
export const ACTIONS: Record<ServiceAction, { label: string; does: string }> = {
  on: { label: 'Turn on', does: 'Installs and starts it. It runs at every login.' },
  stop: { label: 'Stop', does: 'Stops it now. It starts again when the computer restarts, or on Repair.' },
  start: { label: 'Start', does: 'Starts it again now.' },
  enable: { label: 'Enable', does: 'Runs it on this machine, now and after every restart.' },
  disable: { label: 'Disable', does: 'Stops it and keeps it off, even after a restart, until you enable it.' },
};

const UP = new Set(['active', 'running', 'activating', 'reloading']);

/** Whether a service is running, or on its way up: its switch's position. */
export function isUp(s: Service): boolean {
  return (s.installed && UP.has(s.state)) || s.pids.length > 0;
}

/** What flipping a service's switch does: on starts one installed (or installs one that isn't); off stops it. */
export function toggleAction(s: Service, on: boolean): ServiceAction {
  if (!on) return 'stop';
  return s.installed ? 'start' : 'on';
}

/** Stopped and not enabled for this machine: listed apart, one line each, under Turned off. */
export function turnedOff(s: Service): boolean {
  return !s.selected && !isUp(s);
}

/** The sections of the list, in order, each with a line on what its services do; a service with any other group,
 *  or none, goes under Other. */
export const GROUPS: readonly { key: string; label: string; blurb?: string }[] = [
  {
    key: 'core',
    label: 'Continuum core',
    blurb: 'Records the cameras and microphones, and keeps everything. The workers need these.',
  },
  {
    key: 'workers',
    label: 'Workers',
    blurb: 'Turn the recordings into words, faces, and people, and answer when spoken to.',
  },
  { key: 'dashboard', label: 'Dashboard', blurb: 'This page. It stays on while you use it.' },
  { key: 'extras', label: 'Extras', blurb: 'Optional additions, each turned on by choice.' },
];

/** The name of a service's section, for the tag on a Turned off row. */
export function groupLabel(s: Service): string {
  return GROUPS.find(g => g.key === s.group)?.label ?? 'Other';
}

export interface Group {
  key: string;
  label: string;
  blurb?: string;
  services: Service[];
}

/** The services the main list shows (all but the turned-off ones), by section, leaving out empty sections. */
export function grouped(list: readonly Service[]): Group[] {
  const shown = list.filter(s => !turnedOff(s));
  const known = new Set(GROUPS.map(g => g.key));
  return [
    ...GROUPS.map(g => ({ ...g, services: shown.filter(s => s.group === g.key) })),
    { key: 'other', label: 'Other', services: shown.filter(s => !known.has(s.group ?? '')) },
  ].filter(g => g.services.length > 0);
}

/** A group card's count: "4 of 5 running" ("Running" for a group of one), red while one of them is down. */
export function groupCount(services: readonly Service[]): { text: string; tone: Tone } {
  const up = services.filter(s => describe(s).tone === 'ok').length;
  const text = services.length === 1 ? (up ? 'Running' : 'Not running') : `${up} of ${services.length} running`;
  return { text, tone: up === services.length ? 'ok' : 'bad' };
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const two = (n: number) => String(n).padStart(2, '0');

/** When an error was logged, in words: "09:02" today, "Oct 29 09:02" on another day, or "" when the log didn't say. */
export function errorTime(e: { at: string; ts?: number | null }, now: Date = new Date()): string {
  if (e.ts == null) return e.at.slice(0, 5);
  const d = new Date(e.ts * 1000);
  const clock = `${two(d.getHours())}:${two(d.getMinutes())}`;
  return d.toDateString() === now.toDateString() ? clock : `${MONTHS[d.getMonth()]} ${d.getDate()} ${clock}`;
}

/** A POST /services reply: the fresh view, unless the request was refused before anything ran. */
export type ServicesReply = Partial<ServicesView> & { ok: boolean; error?: string | null; note?: string | null };

export type BulkAction = 'repair' | 'off';

/** Whether a reply carries the fresh view, which then replaces the cached one. */
export function isView(r: ServicesReply): r is ServicesView {
  return Array.isArray(r.services) && typeof r.control === 'boolean';
}

/** A dot's colour: ok (green), warn (amber), bad (red), or none (grey). */
export type Tone = 'ok' | 'warn' | 'bad' | 'none';

/** A switch in flight: the service and which way it is going. */
export interface Busy {
  name: string;
  on: boolean;
}

export interface Described {
  label: string;
  tone: Tone;
  on: boolean; // whether its switch is on
}

export const TONE_COLOR: Record<Tone, string> = {
  ok: 'var(--lib-ok)',
  warn: 'var(--lib-warn)',
  bad: 'var(--lib-bad)',
  none: 'var(--mantine-color-dimmed)',
};

/** A tone as small text: green darkened to pass AA contrast (--lib-ok-text); the others already pass. */
export const TONE_TEXT: Record<Tone, string> = { ...TONE_COLOR, ok: 'var(--lib-ok-text)' };

/** A tone as a ground: the halo round a summary's dot. */
export const TONE_TINT: Record<Tone, string> = {
  ok: 'var(--lib-ok-tint)',
  warn: 'var(--lib-warn-tint)',
  bad: 'var(--lib-bad-tint)',
  none: 'transparent',
};

/** What a service row says: label, dot, and whether its switch is on. */
export function describe(s: Service): Described {
  const st = s.state;
  if (s.installed) {
    // A crash loop, up now or coming up again (systemd's "activating (auto-restart)").
    if (UP.has(st) && looping(s)) return { label: 'keeps restarting', tone: 'bad', on: true };
    if (st === 'active' || st === 'running')
      return { label: s.scheduled ? 'scheduled' : 'running', tone: 'ok', on: true };
    if (st === 'activating' || st === 'reloading') return { label: 'starting', tone: 'warn', on: true };
    if (st === 'failed') return { label: 'failed', tone: 'bad', on: true };
    return { label: st === 'inactive' || st === 'not running' ? 'stopped' : st, tone: 'warn', on: true };
  }
  if (s.pids.length) return { label: 'running by hand', tone: 'ok', on: false };
  return { label: 'off', tone: 'none', on: false };
}

/** The summary's dot: red when status is unreadable or a service failed or keeps restarting, amber while an
 *  enabled one is stopped or starting, else green. */
export function overall(view: ServicesView | undefined): Tone {
  if (!view) return 'none';
  if (!view.ok || view.services.some(troubled)) return 'bad';
  const mine = view.services.filter(s => s.selected);
  return mine.every(s => describe(s).tone === 'ok') ? 'ok' : 'warn';
}

/** What the summary card counts: the services running, the enabled ones that need looking at (failed or in a crash
 *  loop), the other enabled ones not running (stopped or starting), how many are enabled here, and how many are
 *  turned off. */
export interface Summary {
  running: number;
  attention: Service[];
  stopped: number;
  enabled: number;
  off: number;
}

export function summary(view: ServicesView): Summary {
  const list = view.services;
  const attention = list.filter(troubled);
  return {
    running: list.filter(s => describe(s).tone === 'ok').length,
    attention,
    stopped: list.filter(s => s.selected && !attention.includes(s) && describe(s).tone !== 'ok').length,
    enabled: list.filter(s => s.selected).length,
    off: list.filter(turnedOff).length,
  };
}

/** What Start n stopped turns on: enabled here and not up (installed but stopped or failed included), never the
 * dashboard itself, one already running by hand, or one up or coming up (a crash loop's next start is the manager's). */
export function startable(list: Service[]): Service[] {
  return list.filter(s => s.selected && s.name !== 'dashboard' && !isUp(s));
}

/** A row's dot and words: the switch's direction while a switch is in flight (polls mid-switch can't flip it). */
export function rowStatus(s: Service, busy: Busy | null): Omit<Described, 'on'> {
  return busy ? { label: busy.on ? 'starting' : 'stopping', tone: 'warn' } : describe(s);
}

/** While Repair or All off runs, the way each service it touches is going. The dashboard and the rows it leaves
 * alone get null. */
export function swept(s: Service, bulk: BulkAction | null): Busy | null {
  if (s.name === 'dashboard' || !bulk) return null;
  if (bulk === 'repair') return s.selected && (s.installed || !s.pids.length) ? { name: s.name, on: true } : null;
  return s.installed ? { name: s.name, on: false } : null;
}

/** Why a row's switch can't be used, or null when it can. */
export function locked(s: Service, control: boolean): string | null {
  if (!control) return 'Services can only be switched from this machine';
  if (!s.selected) return 'Not enabled for this machine: enable it first';
  if (s.name === 'dashboard') return 'This page runs on the dashboard; use pack-services in a terminal';
  return null;
}

/** The view to show when GET /services itself failed. */
export function unreadable(message: string): ServicesView {
  return { ok: false, control: false, error: `Could not read service status: ${message}`, services: [] };
}

/** The confirm question before Repair or All off. */
export const ASK: Record<BulkAction, string> = {
  repair:
    'Remove and reinstall every enabled service except the dashboard? Each one restarts, so the microphone and cameras drop out for a moment.',
  off: 'Turn off every service except the dashboard? Nothing is listened to, watched, or recorded until you turn them back on.',
};

/** An enabled service that needs looking at: failed, or in a crash loop. */
export function troubled(s: Service): boolean {
  return s.selected && (describe(s).tone === 'bad' || looping(s));
}

/** The Settings menu's badge for Services: what needs looking at, or nothing. */
export function servicesBadge(view: ServicesView | undefined): { text: string; tone: 'warn' | 'bad' } | null {
  if (!view) return null;
  if (!view.ok) return { text: 'Unreadable', tone: 'warn' };
  const n = view.services.filter(troubled).length;
  return n ? { text: `${n} ${n === 1 ? 'issue' : 'issues'}`, tone: 'bad' } : null;
}

/** The services Also run (off: the ones off by default) or Don't run (on: the ones on by default) offers. The
 *  dashboard and any service another setting needs are never offered. */
export function pickable(list: readonly Service[], by: 'on' | 'off'): Service[] {
  return list.filter(s => s.name !== 'dashboard' && !s.required && !!s.default === (by === 'on'));
}

/** Recorded lifecycle history. Unknown intervals never count as stopped or healthy. */
export interface HistorySpan {
  start: number;
  end: number;
  state: 'up' | 'stopped' | 'unknown';
}
export interface HistoryEvent {
  at: number;
  kind: 'started' | 'stopped' | 'restart' | 'crashed' | 'start-failed';
  reason: string | null;
}
export interface ServiceHistory {
  name: string;
  spans: HistorySpan[];
  events: HistoryEvent[];
}
export interface HistoryView {
  start: number;
  end: number;
  source: 'journal' | 'sampled' | 'unavailable';
  available: boolean;
  sample_s: number | null;
  truncated: boolean;
  note: string | null;
  services: ServiceHistory[];
}
export function observedUptime(row: ServiceHistory): number | null {
  const known = row.spans.filter(s => s.state !== 'unknown');
  const total = known.reduce((n, s) => n + Math.max(0, s.end - s.start), 0);
  return total > 0
    ? (100 * known.filter(s => s.state === 'up').reduce((n, s) => n + Math.max(0, s.end - s.start), 0)) / total
    : null;
}
/** A unit crash or failed start job is trouble, but only the unit crash proves a process failure. */
export function isHistoryFailure(event: HistoryEvent): boolean {
  return event.kind === 'crashed' || event.kind === 'start-failed';
}
export function historyOrder(rows: readonly ServiceHistory[], services: readonly Service[]): ServiceHistory[] {
  const current = new Map(services.map(s => [s.name, s]));
  const score = (row: ServiceHistory) => {
    const service = current.get(row.name);
    return [
      Number(!!service && troubled(service)),
      row.events.filter(isHistoryFailure).length,
      row.events.filter(e => e.kind === 'restart').length,
      row.spans.filter(s => s.state === 'stopped').reduce((n, s) => n + Math.max(0, s.end - s.start), 0),
    ];
  };
  return [...rows]
    .filter(row => {
      const s = current.get(row.name);
      return !!s && (s.selected || s.installed || isUp(s) || row.events.length > 0);
    })
    .sort((a, b) => {
      const aa = score(a),
        bb = score(b);
      for (let i = 0; i < aa.length; i++) {
        const diff = (bb[i] ?? 0) - (aa[i] ?? 0);
        if (diff) return diff;
      }
      return a.name.localeCompare(b.name);
    });
}
