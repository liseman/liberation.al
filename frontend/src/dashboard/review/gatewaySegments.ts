// A day's recorded video, as Review's segments, from Continuum's gateway (#549): every camera stream's stretches
// with frames (the gateway's `spans`), each a segment GatewayVideo plays. A camera's stream is `<cam>.video`, and on
// a machine with v1's recordings imported, `<cam>.video_v1` too.

import { defaultGatewayUrl, type StreamSummary } from '../player/protocol';
import { type Connection, connect as connectGateway, type Handlers, type Reply } from '../player/transport';
import { gatewaySegmentUrl, parseSegmentUrl } from './gatewayVideo';
import type { RawSegment } from './types';

type Connect = (url: string, handlers: Handlers) => Promise<Connection>;

/** How long a day's spans may take before Review gives up on them. */
export const SPANS_TIMEOUT_MS = 45000;

/** The camera a recorded video stream is of: `cam0` for `cam0.video` or `cam0.video_v1`. */
export function cameraOf(stream: string): string | null {
  const m = /^(.+)\.video(?:_v1)?$/.exec(stream);
  return m?.[1] ?? null;
}

/** How long the connection stays open after its last listing: Review refreshes today's every few seconds, and a
 *  fresh connection each time cost a handshake and a hello (#917). */
export const IDLE_CLOSE_MS = 30000;

/** A day's listing is bounded independently of each gateway reply. */
export const MAX_SPANS_PARTS = 128;
export const MAX_DAY_SPANS_REQUESTS = 1024;
export const MIN_SPANS_WINDOW_US = 1_000_000;
export const MAX_DAY_SEGMENTS = 8192;
/** How many unjoined segments a listing gathers before joining what it has (it joins once at the end otherwise). */
const RAW_SEGMENTS = 4 * MAX_DAY_SEGMENTS;
/** The gap the gateway may close when it joins a window's spans: a camera that records in short bursts (a few seconds
 *  apart) lists as one stretch to play through, not thousands of slivers (#2348). Longer gaps stay gaps. */
export const SPANS_GAP_US = 10_000_000;
/** Ever wider gaps (seconds) a listing over MAX_DAY_SEGMENTS joins before it gives up, so a choppy day still plays. */
const COALESCE_GAPS_S = [30, 120, 600, 3600];
type Window = { stream: number; cam: string; start: number; end: number };
const safeTime = (n: unknown): n is number => typeof n === 'number' && Number.isSafeInteger(n) && n >= 0;

/** The listed spans clipped to the window they were asked for. */
function windowSegments(window: Window, numeric: [number, number][]): RawSegment[] {
  return numeric
    .map(([a, b]) => [Math.max(a, window.start), Math.min(b, window.end)] as const)
    .filter(([a, b]) => b > a)
    .map(([a, b]) => ({
      cam: window.cam,
      start: a / 1e6,
      end: b / 1e6,
      url: gatewaySegmentUrl(window.stream, a, b),
    }));
}

/** The segments sorted and rejoined, `gapS` apart or closer; a listing over `maxSegments` joins wider gaps
 *  (COALESCE_GAPS_S) until it fits, so a fragmented day plays as fewer, longer stretches instead of failing. */
export function normalizeSegments(input: RawSegment[], maxSegments = MAX_DAY_SEGMENTS): RawSegment[] {
  let result = joinSegments(input, 0);
  for (const gap of COALESCE_GAPS_S) {
    if (result.length <= maxSegments) break;
    result = joinSegments(result, gap);
  }
  if (result.length > maxSegments)
    throw new Error('The day video listing exceeded its safe bound. Choose a smaller range.');
  return result;
}

function joinSegments(input: readonly RawSegment[], gapS: number): RawSegment[] {
  // Each URL is parsed once, not in every comparison: a day is thousands of segments (the Review audit).
  const keyed = input.map(segment => ({ segment, stream: parseSegmentUrl(segment.url)?.stream ?? null }));
  keyed.sort(
    (a, b) =>
      (a.segment.cam < b.segment.cam ? -1 : a.segment.cam > b.segment.cam ? 1 : 0) ||
      (a.stream ?? 0) - (b.stream ?? 0) ||
      a.segment.start - b.segment.start,
  );
  const result: RawSegment[] = [];
  let last: RawSegment | null = null;
  let lastStream: number | null = null;
  let grown = false;
  const seal = () => {
    if (last && grown && lastStream !== null)
      last.url = gatewaySegmentUrl(lastStream, Math.round(last.start * 1e6), Math.round(last.end * 1e6));
  };
  for (const { segment, stream } of keyed) {
    // Rejoin a span that a split listing window cut in two. Spans further apart than gapS stay apart, and expired
    // video joins only expired video: a purge can leave thousands of expired slivers on a day (#2348's follow-up).
    if (
      last &&
      stream !== null &&
      stream === lastStream &&
      !!last.expired === !!segment.expired &&
      last.cam === segment.cam &&
      last.end + gapS >= segment.start
    ) {
      if (segment.end > last.end) {
        last.end = segment.end;
        grown = true;
      }
      continue;
    }
    seal();
    last = { ...segment };
    lastStream = stream;
    grown = false;
    result.push(last);
  }
  seal();
  return result;
}

/** One gateway connection that lists segments, a listing at a time, kept open between listings while they come. */
class Lister {
  private conn: Promise<Connection> | null = null;
  private active: ((r: Reply) => void) | null = null;
  private failActive: ((e: Error) => void) | null = null;
  private queue: Promise<unknown> = Promise.resolve();
  private idle: ReturnType<typeof setTimeout> | undefined;

  private readonly connect: Connect;
  private readonly url: string;

  constructor(connect: Connect, url: string) {
    this.connect = connect;
    this.url = url;
  }

  /** Runs one listing after any before it finishes. */
  list(startUs: number, endUs: number, signal?: AbortSignal, cams?: readonly string[]): Promise<RawSegment[]> {
    const run = this.queue.then(() => this.run(startUs, endUs, signal, cams));
    this.queue = run.catch(() => {});
    return run;
  }

  private open(): Promise<Connection> {
    if (this.conn) return this.conn;
    let hello: () => void = () => {};
    const said = new Promise<void>(r => {
      hello = r;
    });
    const opened = this.connect(this.url, {
      onReply: r => (r.type === 'hello' ? hello() : this.active?.(r)),
      onBinary: () => {},
      onClose: reason => this.drop(new Error(`the gateway closed: ${reason}`)),
    }).then(async c => {
      await said;
      return c;
    });
    this.conn = opened;
    opened.catch(() => {
      if (this.conn === opened) this.conn = null;
    });
    return opened;
  }

  /** Forgets the connection (closed, or left in an unknown state), failing the listing on it. */
  private drop(err: Error): void {
    const was = this.conn;
    this.conn = null;
    was?.then(c => c.close()).catch(() => {});
    this.failActive?.(err);
  }

  private run(startUs: number, endUs: number, signal?: AbortSignal, cams?: readonly string[]): Promise<RawSegment[]> {
    clearTimeout(this.idle);
    return new Promise<RawSegment[]>((resolve, reject) => {
      const queue: Window[] = [];
      let current: Window | null = null;
      let requested = 0;
      const requestsByStream = new Map<number, number>();
      let finished = false;
      let out: RawSegment[] = [];
      const abort = () => this.drop(new DOMException('aborted', 'AbortError'));
      /** `keep`: the listing failed on its own data (a bound, a malformed reply), not the connection, which stays
       *  open for the next listing, another day's included (#2348). */
      const finish = (err?: Error, keep = false) => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        signal?.removeEventListener('abort', abort);
        this.active = null;
        this.failActive = null;
        if (err) {
          if (keep) this.idle = setTimeout(() => this.drop(new Error('idle')), IDLE_CLOSE_MS);
          else this.drop(err);
          reject(err);
        } else {
          this.idle = setTimeout(() => this.drop(new Error('idle')), IDLE_CLOSE_MS);
          try {
            resolve(normalizeSegments(out));
          } catch (error) {
            reject(error);
          }
        }
      };
      const timer = setTimeout(
        () => finish(new Error('the gateway took too long to list the video')),
        SPANS_TIMEOUT_MS,
      );
      const next = () => {
        current = queue.shift() ?? null;
        if (!current) {
          finish();
          return;
        }
        const streamRequests = (requestsByStream.get(current.stream) ?? 0) + 1;
        requestsByStream.set(current.stream, streamRequests);
        if (++requested > MAX_DAY_SPANS_REQUESTS || streamRequests > MAX_SPANS_PARTS) {
          finish(new Error('The video listing exceeded its safe window bound.'), true);
          return;
        }
        const window = current;
        void this.conn?.then(c => {
          if (!finished && current === window)
            c.send({
              type: 'spans',
              stream: window.stream,
              start_us: window.start,
              end_us: window.end,
              gap_us: SPANS_GAP_US,
              media_context: true,
            });
        });
      };
      this.failActive = e => finish(e);
      signal?.addEventListener('abort', abort);
      if (signal?.aborted) {
        finish(new DOMException('aborted', 'AbortError'));
        return;
      }
      this.active = r => {
        try {
          if (r.type === 'streams') {
            for (const stream of r.streams as StreamSummary[]) {
              const cam = stream.name ? cameraOf(stream.name) : null;
              if (cam && (!cams || cams.includes(cam)))
                queue.push({ stream: stream.id, cam, start: startUs, end: endUs });
            }
            next();
          } else if (r.type === 'spans') {
            const window = current;
            if (!window || r.stream !== window.stream || r.start_us !== window.start || r.end_us !== window.end) return;
            if (
              !Array.isArray(r.spans) ||
              r.spans.length > MAX_DAY_SEGMENTS ||
              r.spans.some(
                s => !Array.isArray(s) || s.length !== 2 || !safeTime(s[0]) || !safeTime(s[1]) || s[1] < s[0],
              )
            )
              throw new Error('The gateway returned invalid video spans.');
            if (r.partial === true) {
              if (window.end - window.start <= MIN_SPANS_WINDOW_US)
                throw new Error('The video listing is incomplete for the smallest safe window. Retry later.');
              const middle = Math.floor((window.start + window.end) / 2);
              queue.unshift({ ...window, end: middle }, { ...window, start: middle });
              next();
              return;
            }
            const expired = r.expired ?? [];
            if (
              !Array.isArray(expired) ||
              expired.length > MAX_DAY_SEGMENTS ||
              expired.some(
                s => !Array.isArray(s) || s.length !== 2 || !safeTime(s[0]) || !safeTime(s[1]) || s[1] < s[0],
              )
            )
              throw new Error('The gateway returned invalid expired spans.');
            // Video aged out past the history window (#1748) shows as expired, never as unrecorded time.
            const gone: RawSegment[] = (expired as [number, number][]).map(([a, b]) => ({
              cam: window.cam,
              start: a / 1e6,
              end: b / 1e6,
              url: gatewaySegmentUrl(window.stream, a, b),
              expired: true,
            }));
            // Appended, and normalized once when the listing is done; joined early only if a listing grows very long.
            out.push(...windowSegments(window, r.spans as [number, number][]), ...gone);
            if (out.length > RAW_SEGMENTS) out = joinSegments(out, 0);
            next();
          } else if (r.type === 'error') finish(new Error(r.message), true);
        } catch (error) {
          finish(error instanceof Error ? error : new Error(String(error)), true);
        }
      };
      this.open().then(
        c => {
          if (!finished) c.send({ type: 'streams' });
        },
        e => finish(e instanceof Error ? e : new Error(String(e))),
      );
    });
  }
}

const listers = new WeakMap<Connect, Map<string, Lister>>();

function listerFor(connect: Connect, url: string): Lister {
  const byUrl = listers.get(connect) ?? new Map<string, Lister>();
  listers.set(connect, byUrl);
  const l = byUrl.get(url) ?? new Lister(connect, url);
  byUrl.set(url, l);
  return l;
}

/** Every camera's video segments over `[startS, endS)` (Unix seconds), in no particular order, over one gateway
 *  connection per URL that stays open IDLE_CLOSE_MS after the last listing. `cams` lists only those cameras'
 *  streams (one clip's camera, #2552), rather than asking the gateway about every stream. */
export function daySegments(
  startS: number,
  endS: number,
  opts: { connect?: Connect; url?: string; signal?: AbortSignal; cams?: readonly string[] } = {},
): Promise<RawSegment[]> {
  const connect = opts.connect ?? connectGateway;
  const url = opts.url ?? defaultGatewayUrl(location);
  return listerFor(connect, url).list(Math.floor(startS * 1e6), Math.ceil(endS * 1e6), opts.signal, opts.cams);
}
