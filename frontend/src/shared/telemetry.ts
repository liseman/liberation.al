// The page's own error log and trace, kept on this machine: what goes wrong in a browser, and which page
// loads are slow and why, batched to the dashboard's POST /frontend/log, which appends it to logs/frontend.log.
//
// What it reports: uncaught errors and promise rejections, chunks that failed to load (the dashboard's stale-chunk
// handler reports them, and only when reloading for the new build can't recover), error boundaries, requests that
// failed or took over SLOW_REQUEST_MS, and navigations that took over SLOW_NAV_MS or had a request fail, each with the
// requests and code chunks it waited on. Paths only: a route or URL loses its query and
// hash (#card=<name> names a person), and no body is ever sent. At most MAX_PER_MINUTE events a minute, the same one
// at most once in DEDUPE_MS, sent at most every FLUSH_MS and when the page is hidden. The log's own POST goes by
// sendBeacon, never through request(), so logging can't log itself.

export const ENDPOINT = '/frontend/log';
export const FLUSH_MS = 10_000;
export const MAX_PER_MINUTE = 30;
export const DEDUPE_MS = 30_000;
export const SLOW_REQUEST_MS = 3_000;
export const SLOW_NAV_MS = 2_000;
const SETTLE_MS = 500; // a navigation is done this long after its last request ends
const SPAN_MAX_MS = 15_000; // or after this long, whatever is still loading
const MAX_ITEMS = 20; // requests or chunks kept for one navigation
const MAX_BATCH = 20;

type EventType = 'error' | 'rejection' | 'chunk' | 'boundary' | 'request' | 'navigation';
interface Item {
  url: string;
  status?: number;
  ms: number;
}
export interface LogEvent {
  type: EventType;
  at: number;
  route: string;
  message?: string;
  stack?: string;
  name?: string;
  url?: string;
  method?: string;
  status?: number;
  ms?: number;
  requests?: Item[];
  chunks?: Item[];
}

/** An in-app path with no query or hash, from a URL or a path. */
export function pathOf(url: string): string {
  try {
    return new URL(url, location.href).pathname;
  } catch {
    return url.split(/[?#]/, 1)[0] ?? '';
  }
}

const route = (): string => location.pathname;

let queue: LogEvent[] = [];
const kept: number[] = []; // when this minute's events were queued
const recent = new Map<string, number>(); // an event's identity, and when it was last queued

/** Queue an event for the next batch, unless the minute's cap is spent or the same one went out just now. */
export function report(event: Omit<LogEvent, 'at' | 'route'> & { route?: string }, now = Date.now()): boolean {
  const key = `${event.type}|${event.message ?? ''}|${event.url ?? ''}|${event.route ?? route()}`;
  const last = recent.get(key);
  if (last !== undefined && now - last < DEDUPE_MS) return false;
  while (kept.length && now - (kept[0] ?? 0) >= 60_000) kept.shift();
  if (kept.length >= MAX_PER_MINUTE) return false;
  kept.push(now);
  recent.set(key, now);
  if (recent.size > 200) recent.clear();
  queue.push({ route: route(), ...event, at: now });
  if (queue.length >= MAX_BATCH) flush();
  return true;
}

/** An Error (or anything thrown) as the fields a log line keeps. */
function described(error: unknown): Pick<LogEvent, 'message' | 'stack' | 'name'> {
  if (error instanceof Error)
    return { message: error.message.slice(0, 500), stack: error.stack?.slice(0, 2000), name: error.name };
  return { message: String(error).slice(0, 500) };
}

export function reportError(type: 'error' | 'rejection' | 'chunk' | 'boundary', error: unknown, name?: string): void {
  const fields = described(error);
  if (/ResizeObserver loop/.test(fields.message ?? '')) return; // the browser's harmless layout warning
  report({ type, ...fields, ...(name ? { name } : {}) });
}

type Send = (body: string) => void;

function beacon(body: string): void {
  const blob = new Blob([body], { type: 'application/json' });
  if (typeof navigator !== 'undefined' && navigator.sendBeacon?.(ENDPOINT, blob)) return;
  void fetch(ENDPOINT, {
    method: 'POST',
    body,
    headers: { 'Content-Type': 'application/json' },
    keepalive: true,
  }).catch(() => {});
}

/** Send what's queued, in batches, and empty the queue. */
export function flush(send: Send = beacon): void {
  while (queue.length) {
    const batch = queue.slice(0, MAX_BATCH);
    queue = queue.slice(MAX_BATCH);
    try {
      send(JSON.stringify({ events: batch }));
    } catch {
      // a log that can't be sent is dropped: it must never break the page
    }
  }
}

// ---- requests and navigations ----

interface Span {
  route: string;
  start: number;
  requests: Item[];
  chunks: Item[];
  failed: boolean;
  settle?: ReturnType<typeof setTimeout>;
  cap?: ReturnType<typeof setTimeout>;
}
let span: Span | null = null;
let inFlight = 0;

function endSpan(now = performance.now()): void {
  const done = span;
  if (!done) return;
  span = null;
  clearTimeout(done.settle);
  clearTimeout(done.cap);
  const ms = now - done.start;
  if (ms < SLOW_NAV_MS && !done.failed) return;
  report({ type: 'navigation', route: done.route, ms, requests: done.requests, chunks: done.chunks });
}

function settleSoon(): void {
  if (!span || inFlight > 0) return;
  clearTimeout(span.settle);
  span.settle = setTimeout(() => endSpan(), SETTLE_MS);
}

/** A navigation began (a new route, or the first load): its requests and chunks are traced until it settles. */
export function startNavigation(to: string = route(), now = performance.now()): void {
  endSpan(now);
  span = { route: pathOf(to), start: now, requests: [], chunks: [], failed: false };
  span.cap = setTimeout(() => endSpan(), SPAN_MAX_MS);
  settleSoon();
}

export function requestStarted(): void {
  inFlight++;
  if (span) clearTimeout(span.settle);
}

/** A request ended: `status` is null when no answer came. Failed and slow ones are reported; every one joins the
 *  current navigation's trace. An aborted request is the page's own choice, and isn't an error. */
export function requestEnded(url: string, method: string, status: number | null, ms: number, aborted = false): void {
  inFlight = Math.max(0, inFlight - 1);
  const path = pathOf(url);
  if (span && !aborted && span.requests.length < MAX_ITEMS)
    span.requests.push({ url: path, ms: Math.round(ms), ...(status != null ? { status } : {}) });
  const failed = !aborted && (status == null || status >= 400);
  if (span && failed) span.failed = true;
  if (failed || (!aborted && ms >= SLOW_REQUEST_MS))
    report({ type: 'request', url: path, method, ms: Math.round(ms), ...(status != null ? { status } : {}) });
  settleSoon();
}

function chunkTimes(): () => void {
  if (typeof PerformanceObserver === 'undefined') return () => {};
  const observer = new PerformanceObserver(list => {
    for (const entry of list.getEntries()) {
      const path = pathOf(entry.name);
      if (!span || !/^\/static\/app\/assets\/.+\.(js|css)$/.test(path) || span.chunks.length >= MAX_ITEMS) continue;
      span.chunks.push({ url: path, ms: Math.round(entry.duration) });
    }
  });
  try {
    observer.observe({ type: 'resource', buffered: false });
  } catch {
    return () => {};
  }
  return () => observer.disconnect();
}

/** Starts the log for this page: its error listeners, the navigation trace, and the periodic send. */
export function startTelemetry(target: Window = window): () => void {
  let shown = route();
  const onError = (e: ErrorEvent) => reportError('error', e.error ?? e.message);
  const onRejection = (e: PromiseRejectionEvent) => reportError('rejection', e.reason);
  const onRoute = () => {
    if (route() === shown) return;
    shown = route();
    startNavigation(shown);
  };
  const onHide = () => {
    if (document.visibilityState === 'hidden') flush();
  };
  const onLeave = () => flush();
  target.addEventListener('error', onError);
  target.addEventListener('unhandledrejection', onRejection);
  target.addEventListener('popstate', onRoute);
  target.addEventListener('hashchange', onRoute); // the router's navigate() announces every in-app move this way
  target.addEventListener('pagehide', onLeave);
  document.addEventListener('visibilitychange', onHide);
  const stopChunks = chunkTimes();
  const timer = setInterval(() => flush(), FLUSH_MS);
  startNavigation(shown, 0); // the first load, from navigation start
  return () => {
    target.removeEventListener('error', onError);
    target.removeEventListener('unhandledrejection', onRejection);
    target.removeEventListener('popstate', onRoute);
    target.removeEventListener('hashchange', onRoute);
    target.removeEventListener('pagehide', onLeave);
    document.removeEventListener('visibilitychange', onHide);
    stopChunks();
    clearInterval(timer);
    flush();
  };
}

/** For tests: forget everything queued, kept, and traced. */
export function resetTelemetry(): void {
  queue = [];
  kept.length = 0;
  recent.clear();
  if (span) {
    clearTimeout(span.settle);
    clearTimeout(span.cap);
  }
  span = null;
  inFlight = 0;
}

/** For tests: what's queued. */
export const queued = (): readonly LogEvent[] => queue;
