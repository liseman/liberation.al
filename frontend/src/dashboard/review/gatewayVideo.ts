// A camera's recorded video for Review through Continuum's gateway (#549): the parts of a <video> the review engine
// uses (VideoLike), played on a canvas by the player's WebCodecs worker from the gateway's `range` windows.
//
// A segment's URL names the stream and the stretch it covers (`gateway:<stream>:<start_us>:<end_us>`, from the
// gateway's `spans`). Setting currentTime seeks: the window at the new position is asked for, and `seeked` fires once
// it has arrived. While playing, a clock moves the playhead at playbackRate, the worker draws the frame at or before
// it, and the next window is asked for before this one runs out; at the segment's end, `ended`. Each object holds one
// active gateway connection. Seeking away from an unfinished reply replaces it; removing src closes it.
//
// The gateway going away (a restart) or answering an error isn't the recording's fault: the window is asked for again
// after a short wait, so playback picks up once the gateway is back, and nothing is reported as an error (Review marks
// a segment that errors as cut off). A window whose frames are all in a codec this browser can't decode sets
// `unplayable` (and `unplayableCodec`, which one), so Review can say why the picture is empty. Recorded video is H.264,
// which every browser decodes, until the opt-in archive pass makes it AV1 (#1479), which older Safari can't.

import type { FrameMetadata } from '../camera/geometry';
import { decodeHeader, PayloadType } from '../player/header';
import { MediaPackets } from '../player/media';
import { defaultGatewayUrl } from '../player/protocol';
import {
  type Connection,
  connect as connectGateway,
  connectWebSocket,
  type Handlers,
  type Reply,
} from '../player/transport';
import type { WorkerLike } from '../player/worker';
import type { VideoLike } from './engine';

const US = 1_000_000;
/** A segment as the player plays it: one stream's window, in microseconds UTC. */
type Seg = { stream: number; startUs: number; endUs: number };

/** The URL of a stretch of a stream, as GatewayVideo plays it. */
export const gatewaySegmentUrl = (stream: number, startUs: number, endUs: number): string =>
  `gateway:${stream}:${startUs}:${endUs}`;

/** A segment URL's parts, or null for anything else. */
export function parseSegmentUrl(url: string): { stream: number; startUs: number; endUs: number } | null {
  const m = /^gateway:(\d+):(\d+):(\d+)$/.exec(url);
  if (!m) return null;
  const [stream, startUs, endUs] = [Number(m[1]), Number(m[2]), Number(m[3])];
  return endUs > startUs ? { stream, startUs, endUs } : null;
}

type Connect = (url: string, handlers: Handlers) => Promise<Connection>;

export interface GatewayVideoDeps {
  connect?: Connect;
  makeWorker?: () => WorkerLike;
  /** The gateway's WebSocket address. */
  url?: string;
  /** How much each `range` asks for. */
  windowUs?: number;
  raf?: (cb: (now: number) => void) => number;
  cancelRaf?: (id: number) => void;
  /** Wall-clock milliseconds, for telling a drag's seeks apart (DRAG_GAP_MS). */
  now?: () => number;
}

/** How long playback waits at the end of a segment that has been growing before it ends there: Review lists today's
 *  video again every few seconds (#806), so the stretch being recorded grows on while it plays. */
const GROW_WAIT_MS = 15_000;

/** A seek asks for a short window around its point, not a playing window: the frame showing at the point (the
 *  newest at or before it) and a few after. A 4 s window was 120 MJPEG frames and 28 MB, all decoded before the next
 *  scrub position could start (2026-10-30). Playing on from a seek asks for the next full window as usual; for
 *  H.264 the gateway sends whole GOPs from the keyframe anyway. */
const SEEK_LEAD_US = 250_000;
const SEEK_WINDOW_US = 500_000;
/** How far outside a wanted window a frame still belongs to it: a GOP's frames start at its keyframe, which can be
 *  before the window, and run on past its end (Tier 2 H.264 GOPs are a couple of seconds). */
const GOP_SLACK_US = 10_000_000;

/** The first wait before a window is asked for again, and the longest; it doubles in between. */
const RETRY_MS = [500, 5000] as const;
/** Seeks closer together than this are a drag (the timeline scrubbed): one that would give up a request still out
 *  waits until they stop, instead of replacing the connection every step (binary replies carry no request ID, so a
 *  request is given up by closing its connection). */
const DRAG_GAP_MS = 300;
/** How long a drag must hold still before its latest seek is asked for. */
const SEEK_SETTLE_MS = 100;
/** With request ids (a gateway that lists the `req` feature), a drag sends its newest position at most this often,
 *  cancelling the one before on the same connection, so the picture keeps catching up (about 8 a second). */
const SCRUB_MS = 120;

const realWorker = (): WorkerLike => new Worker(new URL('../player/worker.ts', import.meta.url), { type: 'module' });

/** A MediaError-shaped failure, for the engine's error handling. */
const failure = (message: string): MediaError => ({ code: 4, message }) as unknown as MediaError;

export class GatewayVideo implements VideoLike {
  readonly canvas: HTMLCanvasElement;
  preload = 'auto';
  playbackRate = 1;
  defaultPlaybackRate = 1;
  readyState = 0;
  duration = Number.NaN;
  error: MediaError | null = null;
  seeking = false;
  paused = true;
  ended = false;
  /** The current window's frames are all in codecs this browser can't decode. */
  unplayable = false;
  /** Which codec, while `unplayable` (`av1` for archived video): the first of the window's. */
  unplayableCodec: string | null = null;

  private readonly deps: Required<Omit<GatewayVideoDeps, 'url'>> & { url: string };
  private readonly events = new EventTarget();
  private srcUrl = '';
  private accepting = false;
  private sentContext: number | null = null;
  private readonly packets = new MediaPackets((buffer, media) => {
    if (!this.acceptPacket(buffer)) return;
    const worker = this.ensureWorker();
    if (media && this.sentContext !== media.context_id) {
      worker.postMessage({ type: 'media', media });
      this.sentContext = media.context_id;
    }
    worker.postMessage({ type: 'message', buffer, epoch: this.generation }, [buffer]);
  });

  private acceptPacket(buffer: ArrayBuffer): boolean {
    if (!this.accepting) return false;
    const h = decodeHeader(buffer);
    if (h.streamId !== this.seg?.stream || h.live) return false;
    const us = h.captureUs;
    // A frame of a window a later seek gave up on (the gateway answers a range whole, in order) is not decoded. A
    // GOP that starts in the window comes whole, so its tail runs past the window's end: those frames are the next
    // window's first ones, and dropping them froze the picture at every window's end for up to a GOP (#917).
    if (
      this.wanted !== null &&
      this.wantedEnd !== null &&
      (us < this.wanted - GOP_SLACK_US || us >= this.wantedEnd + GOP_SLACK_US)
    )
      return false;
    // While a scrub position waits for its frame, only that frame (at or before it) is wanted.
    if (this.wanted === null && this.framing !== null && (us > this.framing || us < this.framing - GOP_SLACK_US))
      return false;
    if (this.wanted !== null && (h.payloadType === PayloadType.video || h.payloadType === PayloadType.jpeg))
      this.receivedMax = Math.max(this.receivedMax ?? us, us);
    return true;
  }

  private clearFrame(): void {
    delete this.canvas.dataset.capturedAt;
    delete this.canvas.dataset.frameMetadata;
    this.canvas.getContext('2d')?.clearRect?.(0, 0, this.canvas.width, this.canvas.height);
  }
  private seg: Seg | null = null;
  private pos = 0; // seconds into the segment
  private conn: Connection | null = null;
  private opening: Promise<Connection | null> | null = null;
  /** Socket callbacks can arrive after close; only the current connection attempt owns this state. */
  private connectionEpoch = 0;
  private worker: WorkerLike | null = null;
  private closed = false;
  /** The window loaded last, and where the gateway said to continue (null when it sent all of it). */
  private loaded: { start: number; end: number; next: number | null; after: number | null } | null = null;
  /** The window asked for and not yet done. */
  private wanted: number | null = null;
  /** A continuation frontier commits only when its complete range reply arrives. */
  private wantedAfter: number | null = null;
  private receivedMax: number | null = null;
  private clock = 0;
  private last = 0;
  private generation = 0; // decoded-pixel ownership, invalidated when the worker surface resets
  private sourceEpoch = 0; // source metadata and connections survive an empty-picture reset
  private seekEpoch = 0; // deferred settlement belongs to only the newest seek
  private retryMs: number = RETRY_MS[0];
  private contextTimer: ReturnType<typeof setTimeout> | undefined;
  private retryTimer: ReturnType<typeof setTimeout> | undefined;
  /** A drag's latest seek, waiting for the drag to settle (DRAG_GAP_MS). */
  private seekTimer: ReturnType<typeof setTimeout> | undefined;
  private lastSeekAt = Number.NEGATIVE_INFINITY;
  /** When a drag's position last went out (SCRUB_MS). */
  private lastSentAt = Number.NEGATIVE_INFINITY;
  /** The connection's gateway tags replies with request ids (its hello lists `req`), so a request can be cancelled
   *  on it instead of closing it. */
  private canReq = false;
  private reqSeq = 0;
  /** The request id of the window or frame still coming, when it was sent with one; null when untagged or none. */
  private outReq: number | null = null;
  /** Codecs the worker said it can't decode, and the codecs of the last window. */
  private readonly unsupported = new Set<string>();
  private codecs: string[] = [];
  /** The time a `frame` request (a scrub position) asked for and hasn't had answered. */
  private framing: number | null = null;
  /** The end of the window asked for and not yet done. */
  private wantedEnd: number | null = null;
  /** When the segment last grew (performance.now()), or null for one that never has. */
  private grewAt: number | null = null;

  constructor(canvas: HTMLCanvasElement, deps: GatewayVideoDeps = {}) {
    this.canvas = canvas;
    this.deps = {
      connect: deps.connect ?? connectGateway,
      makeWorker: deps.makeWorker ?? realWorker,
      url: deps.url ?? defaultGatewayUrl(typeof location === 'undefined' ? { hostname: '', protocol: '' } : location),
      windowUs: deps.windowUs ?? 4 * US,
      raf: deps.raf ?? (cb => requestAnimationFrame(cb)),
      cancelRaf: deps.cancelRaf ?? (id => cancelAnimationFrame(id)),
      now: deps.now ?? Date.now,
    };
  }

  get src(): string {
    return this.srcUrl;
  }
  set src(url: string) {
    if (url === this.srcUrl || (this.closed && url)) return;
    const replacing = !!this.srcUrl;
    this.srcUrl = url;
    this.sourceEpoch++;
    this.seekEpoch++;
    this.accepting = false;
    if (replacing && this.conn) {
      this.connectionEpoch++;
      this.conn?.close();
      this.conn = null;
      this.opening = null;
    }
    this.clearFrame();
    this.packets.clear();
    clearTimeout(this.contextTimer);
    this.generation++;
    this.stopClock();
    this.paused = true;
    this.ended = false;
    this.error = null;
    this.loaded = null;
    this.wanted = null;
    this.pos = 0;
    this.unplayable = false;
    this.codecs = [];
    this.grewAt = null;
    this.framing = null;
    clearTimeout(this.retryTimer);
    this.sentContext = null;
    this.worker?.postMessage({ type: 'reset' });
    if (!url) {
      this.seg = null;
      return;
    }
    this.seg = parseSegmentUrl(url);
    if (!this.seg) {
      this.readyState = 0;
      this.error = failure(`not a gateway segment: ${url}`);
      queueMicrotask(() => this.fire('error'));
      return;
    }
    this.duration = (this.seg.endUs - this.seg.startUs) / US;
    this.readyState = 1;
    const gen = this.sourceEpoch;
    queueMicrotask(() => {
      if (gen === this.sourceEpoch) this.fire('loadedmetadata');
    });
    this.request(this.seg.startUs);
  }

  get currentTime(): number {
    return this.pos;
  }
  set currentTime(t: number) {
    this.seekTo(t);
  }
  /** A scrub position (Review calls this while the timeline is dragged): just the one frame showing at `t`, from the
   *  gateway's `frame` request (the newest frame at or before it, or the keyframe for GOP video), and `seeked` once
   *  it's here. Any window still coming is given up. */
  fastSeek(t: number): void {
    const seg = this.seg;
    if (!seg) return;
    this.pos = Math.min(Math.max(0, Number.isFinite(t) ? t : 0), this.duration);
    this.ended = false;
    this.seeking = true;
    const epoch = ++this.seekEpoch;
    const us = this.abs();
    this.whenSettled(() => {
      if (epoch === this.seekEpoch && this.seg === seg) this.scrubTo(seg, us);
    });
  }

  private scrubTo(seg: Seg, us: number): void {
    this.resetForSeek();
    this.ensureWorker().postMessage({ type: 'playhead', us });
    this.framing = us;
    this.accepting = true;
    void this.connection().then(c => {
      if (c && this.framing === us && this.seg === seg)
        c.send({
          type: 'frame',
          stream: seg.stream,
          at_us: us,
          media_context: true,
          ...this.tag(),
        });
    });
  }

  play(): Promise<void> {
    if (!this.seg) return Promise.reject(new DOMException('no source', 'NotSupportedError'));
    if (!this.paused) return Promise.resolve();
    if (this.ended) {
      this.ended = false;
      this.seekTo(0);
    }
    this.paused = false;
    this.fire('play');
    if (!this.seeking) this.fire('playing');
    this.startClock();
    return Promise.resolve();
  }

  pause(): void {
    if (this.paused) return;
    this.paused = true;
    this.stopClock();
    this.fire('pause');
  }

  /** The segment grew to `duration` seconds (a stretch still being recorded): playback goes on into it. */
  extend(duration: number): void {
    const seg = this.seg;
    if (!seg || !(duration > this.duration)) return;
    seg.endUs = seg.startUs + Math.round(duration * US); // in place: requests in flight hold this segment
    this.duration = duration;
    this.grewAt = this.last || performance.now(); // on the animation clock the playhead moves by
    this.fire('durationchange');
  }

  /** With no src, lets go of the connection and the picture (what the engine does to a <video> it releases). */
  load(): void {
    if (this.srcUrl) return;
    this.generation++;
    this.connectionEpoch++;
    this.conn?.close();
    this.conn = null;
    this.opening = null;
    this.readyState = 0;
    this.duration = Number.NaN;
    this.fire('emptied');
  }

  getAttribute(name: string): string | null {
    if (name === 'src') return this.srcUrl || null;
    return this.canvas.getAttribute(name);
  }
  removeAttribute(name: string): void {
    if (name === 'src') this.src = '';
    else this.canvas.removeAttribute(name);
  }
  toggleAttribute(name: string, force?: boolean): boolean {
    const on = this.canvas.toggleAttribute(name, force);
    // The box layer draws over the camera's live-canvas that isn't off: the standby one is off.
    if (name === 'data-standby') this.canvas.classList.toggle('off', on);
    return on;
  }
  addEventListener(type: string, listener: () => void, options?: AddEventListenerOptions): void {
    this.events.addEventListener(type, listener, options);
  }
  removeEventListener(type: string, listener: () => void): void {
    this.events.removeEventListener(type, listener);
  }

  /** Stops everything for good: the clock, the connection, and the worker. */
  close(): void {
    this.closed = true;
    clearTimeout(this.retryTimer);
    clearTimeout(this.seekTimer);
    this.src = '';
    this.load();
    this.worker?.terminate();
    this.worker = null;
  }

  private fire(type: string): void {
    this.events.dispatchEvent(new Event(type));
  }

  /** The playhead in whole microseconds UTC: the gateway takes times as integers (a fraction made it refuse a
   *  request outright, so a scrub waited for an answer that never came). */
  private abs(): number {
    return Math.round((this.seg?.startUs ?? 0) + this.pos * US);
  }

  private seekTo(t: number): void {
    if (!this.seg) return;
    this.pos = Math.min(Math.max(0, Number.isFinite(t) ? t : 0), this.duration);
    this.ended = false;
    this.seeking = true;
    const epoch = ++this.seekEpoch;
    const us = this.abs();
    const l = this.loaded;
    if (l && us >= l.start && us < l.end && this.wanted === null) {
      clearTimeout(this.seekTimer); // a drag's waiting seek is superseded by this one, already loaded
      this.seekTimer = undefined;
      this.ensureWorker().postMessage({ type: 'playhead', us });
      queueMicrotask(() => {
        if (epoch === this.seekEpoch) this.settled();
      });
      return;
    }
    const seg = this.seg;
    this.whenSettled(() => {
      if (epoch !== this.seekEpoch || this.seg !== seg) return;
      this.resetForSeek();
      this.ensureWorker().postMessage({ type: 'playhead', us });
      this.request(Math.max(seg.startUs, us - SEEK_LEAD_US), SEEK_WINDOW_US);
    });
  }

  /** Runs a seek now, unless it comes mid-drag (DRAG_GAP_MS after the last) while a request is still out: then once
   *  the drag holds still for SEEK_SETTLE_MS, so a drag replaces the connection once, not once a step. */
  private whenSettled(run: () => void): void {
    const now = this.deps.now();
    const dragging = now - this.lastSeekAt < DRAG_GAP_MS;
    this.lastSeekAt = now;
    clearTimeout(this.seekTimer);
    this.seekTimer = undefined;
    if (dragging && this.canReq && this.conn !== null) {
      // One connection serves every position: send the newest at a steady cadence, cancelling the one before.
      const wait = SCRUB_MS - (now - this.lastSentAt);
      if (wait > 0) {
        this.seekTimer = setTimeout(() => {
          this.seekTimer = undefined;
          this.lastSentAt = this.deps.now();
          if (!this.closed) run();
        }, wait);
        return;
      }
      this.lastSentAt = now;
      run();
      return;
    }
    if (dragging && this.conn !== null && (this.wanted !== null || this.framing !== null)) {
      this.seekTimer = setTimeout(() => {
        this.seekTimer = undefined;
        if (!this.closed) run();
      }, SEEK_SETTLE_MS);
      return;
    }
    this.lastSentAt = now;
    run();
  }

  /** Binary replies have no request ID. An abandoned reply must stay on its obsolete connection, so it cannot
   *  become the new seek's decoded picture or completed-GOP frontier, even when their time windows overlap. */
  private resetForSeek(): void {
    if ((this.wanted !== null || this.framing !== null) && this.conn && this.outReq !== null) {
      // Its replies carry its id: cancel it and keep the connection (the gateway's `req`, #2384).
      this.conn.send({ type: 'cancel', req: this.outReq });
      this.outReq = null;
    } else if (this.wanted !== null || this.framing !== null) {
      this.connectionEpoch++;
      const obsolete = this.conn;
      this.conn = null;
      this.opening = null;
      obsolete?.close();
    }
    this.accepting = false;
    this.loaded = null;
    this.wanted = null;
    this.wantedEnd = null;
    this.wantedAfter = null;
    this.receivedMax = null;
    this.framing = null;
    this.packets.clear();
    clearTimeout(this.contextTimer);
    this.contextTimer = undefined;
    clearTimeout(this.retryTimer);
    this.generation++;
    this.sentContext = null;
    this.worker?.postMessage({ type: 'reset' });
  }

  /** The seek has the frames it needs. */
  /** A reply that arrives while a drag's seek waits (whenSettled) answers a request that seek superseded, so it
   *  doesn't complete it: `seeked` must mean the picture at the new position. */
  private settled(): void {
    if (!this.seeking || this.seekTimer !== undefined) return;
    this.seeking = false;
    this.readyState = 4;
    this.fire('seeked');
    if (!this.paused) this.fire('playing');
  }

  private request(startUs: number, windowUs: number = this.deps.windowUs, afterUs: number | null = null): void {
    const seg = this.seg;
    if (!seg || this.wanted === startUs) return;
    this.wanted = startUs;
    this.wantedAfter = afterUs;
    this.receivedMax = null;
    this.accepting = true;
    this.framing = null;
    const end = Math.min(startUs + windowUs, seg.endUs);
    this.wantedEnd = end;
    void this.connection().then(c => {
      if (c && this.wanted === startUs && this.seg === seg)
        c.send({
          type: 'range',
          stream: seg.stream,
          start_us: startUs,
          end_us: end,
          ...(afterUs === null ? {} : { after_us: afterUs }),
          media_context: true,
          ...this.tag(),
        });
    });
  }

  /** A new request id for the request about to go out, when the connection takes them. */
  private tag(): { req?: number } {
    if (!this.canReq) {
      this.outReq = null;
      return {};
    }
    this.outReq = ++this.reqSeq;
    return { req: this.outReq };
  }

  private connection(): Promise<Connection | null> {
    if (this.closed) return Promise.resolve(null);
    if (this.conn) return Promise.resolve(this.conn);
    if (this.opening) return this.opening;
    const gen = this.sourceEpoch;
    const epoch = ++this.connectionEpoch;
    this.canReq = false; // until this connection's hello says
    this.outReq = null;
    // Nothing is sent before the hello, which says whether requests can carry ids (every gateway says hello first).
    let hello: () => void = () => {};
    const said = new Promise<void>(resolve => {
      hello = resolve;
    });
    const handlers: Handlers = {
      onReply: r => {
        if (r.type === 'hello') hello();
        if (!this.closed && epoch === this.connectionEpoch) this.reply(r);
      },
      onBinary: buffer => {
        if (this.closed || epoch !== this.connectionEpoch) return;
        try {
          // A frame tagged with a request since cancelled or replaced is not this seek's.
          if (this.canReq && (decodeHeader(buffer).req ?? null) !== this.outReq) return;
          this.packets.packet(buffer);
          this.watchContexts();
        } catch {
          this.clearFrame();
          this.retry();
        }
      },
      onClose: () => {
        if (this.closed || epoch !== this.connectionEpoch) return;
        this.connectionEpoch++;
        this.conn = null;
        this.opening = null;
        this.retry(); // a window still wanted is asked for again on a new connection
      },
    };
    this.opening = this.deps.connect(this.deps.url, handlers).then(
      async c => {
        await said;
        if (this.closed || epoch !== this.connectionEpoch) {
          c.close();
          return null;
        }
        if (gen !== this.sourceEpoch) {
          // Invalidate before close, which can invoke callbacks immediately or much later.
          this.connectionEpoch++;
          c.close();
          // A replacement source may already be waiting on this same promise. Do not leave it
          // cached as resolved-null: open for the current source and let those requests continue.
          this.opening = null;
          return !this.closed && this.seg ? this.connection() : null;
        }
        this.conn = c;
        return c;
      },
      () => {
        if (epoch !== this.connectionEpoch) return null;
        this.connectionEpoch++;
        this.opening = null;
        this.retry();
        return null;
      },
    );
    return this.opening;
  }

  /** A packet whose media context never arrives is dropped, never shown with an older frame's context. */
  private watchContexts(): void {
    if (!this.packets.waiting) {
      clearTimeout(this.contextTimer);
      this.contextTimer = undefined;
      return;
    }
    if (this.contextTimer !== undefined) return;
    this.contextTimer = setTimeout(() => {
      this.contextTimer = undefined;
      if (this.closed || !this.seg || !this.packets.waiting) return;
      this.packets.clear();
      this.clearFrame();
      this.generation++;
      this.sentContext = null;
      this.worker?.postMessage({ type: 'reset' });
      this.connectionEpoch++;
      this.conn?.close();
      this.conn = null;
      this.opening = null;
      this.loaded = null;
      this.wanted = null;
      this.request(this.abs());
    }, 10_000);
  }

  private reply(r: Reply): void {
    if (r.type === 'hello') {
      this.canReq = r.features?.includes('req') ?? false;
      return;
    }
    // A reply to a request since cancelled or replaced (each echoes its `req`) is not this seek's.
    const req = (r as { req?: number }).req;
    if (this.canReq && req !== undefined && req !== this.outReq) return;
    if (r.type === 'media') {
      if (r.stream !== this.seg?.stream) return;
      try {
        this.packets.context(r);
        this.watchContexts();
      } catch {
        this.clearFrame();
        this.retry();
      }
    } else if (r.type === 'frame') {
      if (r.at_us !== this.framing) return; // a scrub position since passed
      this.framing = null;
      this.outReq = null;
      if (r.capture_us === null) {
        this.accepting = false;
        this.generation++;
        this.sentContext = null;
        this.worker?.postMessage({ type: 'reset' });
        this.packets.clear();
        clearTimeout(this.contextTimer);
        this.contextTimer = undefined;
        this.clearFrame();
      }
      if (this.readyState < 2) this.readyState = 2;
      this.settled();
    } else if (r.type === 'done') {
      if (r.start_us !== this.wanted) return; // a window no longer wanted (a seek went past it)
      const after = this.receivedMax === null ? this.wantedAfter : Math.max(this.wantedAfter ?? 0, this.receivedMax);
      this.loaded = { start: r.start_us, end: r.end_us, next: r.next_us, after };
      this.wanted = null;
      this.outReq = null;
      this.retryMs = RETRY_MS[0];
      this.codecs = r.codecs;
      this.judge();
      if (this.readyState < 2) this.readyState = 2;
      this.settled();
    } else if (r.type === 'error') {
      // The broker or a chunk was out of reach: ask again shortly rather than call the segment broken.
      this.retry();
    }
  }

  /** Asks for the wanted window again after a wait that doubles each time, up to RETRY_MS[1]. */
  private retry(): void {
    const want = this.wanted;
    const frame = this.framing;
    if ((want === null && frame === null) || !this.seg) return;
    clearTimeout(this.retryTimer);
    const seg = this.seg;
    this.retryTimer = setTimeout(() => {
      if (this.seg !== seg || this.wanted !== want || this.framing !== frame) return;
      if (frame !== null) this.fastSeek(this.pos);
      else if (want !== null) {
        this.wanted = null;
        this.request(want, this.deps.windowUs, this.wantedAfter);
      }
    }, this.retryMs);
    this.retryMs = Math.min(this.retryMs * 2, RETRY_MS[1]);
  }

  /** Whether the last window's frames can't be shown here: every codec in it is one the worker can't decode. */
  private judge(): void {
    const was = this.unplayable;
    this.unplayable = this.codecs.length > 0 && this.codecs.every(c => this.unsupported.has(c));
    this.unplayableCodec = this.unplayable ? (this.codecs[0] ?? null) : null;
    if (this.unplayable !== was) this.fire('unplayable');
  }

  private ensureWorker(): WorkerLike {
    if (this.worker) return this.worker;
    const w = this.deps.makeWorker();
    w.onmessage = e => {
      if (e.data.type === 'frame') {
        const bitmap = e.data.bitmap;
        try {
          if (
            !this.closed &&
            this.accepting &&
            this.worker === w &&
            (e.data.epoch === undefined || e.data.epoch === this.generation)
          ) {
            if (this.canvas.width !== bitmap.width) this.canvas.width = bitmap.width;
            if (this.canvas.height !== bitmap.height) this.canvas.height = bitmap.height;
            const context = this.canvas.getContext('2d');
            this.clearFrame();
            if (context) {
              context.drawImage(bitmap, 0, 0);
              if (Number.isSafeInteger(e.data.frameUs) && e.data.frameUs > 0)
                this.canvas.dataset.capturedAt = String(e.data.frameUs / US);
              else delete this.canvas.dataset.capturedAt;
              if (e.data.metadata) this.canvas.dataset.frameMetadata = JSON.stringify(e.data.metadata);
            }
          }
        } finally {
          bitmap.close();
        }
        return;
      }
      if (this.closed) return;
      if (e.data.type === 'codec') {
        // Capability depends on the actual parameter sets; a later profile can be supported in the same family.
        if (e.data.supported) this.unsupported.delete(e.data.codec);
        else this.unsupported.add(e.data.codec);
        this.judge();
      }
    };
    // The DOM canvas stays main-thread owned across ref cleanup, reconnects, and player remounts.
    // Each worker owns a fresh rendering surface and transfers finished bitmaps back for presentation.
    if (typeof OffscreenCanvas !== 'undefined') {
      const off = new OffscreenCanvas(this.canvas.width, this.canvas.height);
      w.postMessage({ type: 'canvas', canvas: off, returnBitmap: true }, [off]);
    }
    this.worker = w;
    return w;
  }

  private startClock(): void {
    if (this.clock) return;
    this.last = 0;
    const tick = (now: number) => {
      this.clock = 0;
      if (this.paused || !this.seg) return;
      if (this.last && !this.seeking) this.pos += ((now - this.last) / 1000) * this.playbackRate;
      this.last = now;
      if (this.pos >= this.duration && this.grewAt !== null && now - this.grewAt < GROW_WAIT_MS) {
        // At the live edge of a stretch still being recorded: hold there until it grows again.
        this.pos = this.duration;
        this.clock = this.deps.raf(tick);
        return;
      }
      if (this.pos >= this.duration) {
        this.pos = this.duration;
        this.paused = true;
        this.ended = true;
        this.fire('pause');
        this.fire('ended');
        return;
      }
      const us = this.abs();
      this.ensureWorker().postMessage({ type: 'playhead', us });
      // Ask for the next window a second before this one runs out.
      const l = this.loaded;
      const next = l ? Math.max(l.next ?? l.end, l.after === null ? 0 : l.after + 1) : 0;
      if (l && this.wanted === null && us > l.end - US && next < this.seg.endUs)
        this.request(next, this.deps.windowUs, l.after);
      else if (!l && this.wanted === null) this.request(us); // playing on from a scrub position: no window yet
      this.clock = this.deps.raf(tick);
    };
    this.clock = this.deps.raf(tick);
  }

  private stopClock(): void {
    if (this.clock) this.deps.cancelRaf(this.clock);
    this.clock = 0;
  }
}

export interface RecordedFrame {
  image: string;
  /** UTC seconds of the decoded frame; a GOP keyframe can precede the requested playhead. */
  captured_at: number;
  frame_metadata?: FrameMetadata;
}

interface RecordedFrameCaptureDeps extends Pick<GatewayVideoDeps, 'connect' | 'makeWorker' | 'url'> {
  makeCanvas?: () => HTMLCanvasElement;
  timeoutMs?: number;
}

const CAPTURE_TIMEOUT_MS = 10_000;
const MAX_IMAGE_CHARS = 16_000_000;
const MAX_IMAGE_PIXELS = 16_000_000;
const MAX_IMAGE_SIDE = 8192;
/** Match the gateway's bounded GOP range. Reject partial ranges instead of importing an older keyframe. */
const CAPTURE_MAX_BYTES = 64 * 1024 * 1024;
const CAPTURE_MAX_FRAMES = 600;
const CAPTURE_LOOKBACK_US = 15 * US;
const CAPTURE_REQUEST_ID = 1;

/** Capture native decoder pixels without displayed-canvas scaling or overlays. JPEG uses one frame; compressed
 *  video reads the complete bounded GOP interval through the playhead, then flushes delayed decoder output.
 *  Ordered WebSocket messages keep range completion behind its binary access units. The whole operation has one
 *  deadline, at most 600 range units/64 MiB, and at most 15 seconds of keyframe lookback. */
export function createRecordedFrameCapture(deps: RecordedFrameCaptureDeps = {}) {
  const capture = async (segmentUrl: string, atSeconds: number, signal?: AbortSignal): Promise<RecordedFrame> => {
    const seg = parseSegmentUrl(segmentUrl);
    const atUs = Math.round(atSeconds * US);
    if (
      !seg ||
      !Number.isSafeInteger(seg.stream) ||
      seg.stream > 0xffffffff ||
      !Number.isSafeInteger(seg.startUs) ||
      !Number.isSafeInteger(seg.endUs) ||
      !Number.isSafeInteger(atUs) ||
      atUs < seg.startUs ||
      atUs >= seg.endUs
    )
      throw new Error('The capture time is outside this recorded segment.');
    if (signal?.aborted) throw new DOMException('Recorded frame capture cancelled.', 'AbortError');
    if (typeof OffscreenCanvas === 'undefined')
      throw new DOMException('This browser cannot decode recorded frames.', 'NotSupportedError');
    const timeoutMs = deps.timeoutMs ?? CAPTURE_TIMEOUT_MS;
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Frame capture needs a positive timeout.');

    return new Promise<RecordedFrame>((resolve, reject) => {
      let finished = false;
      let connection: Connection | null = null;
      let worker: WorkerLike | null = null;
      let pending: ImageBitmap | null = null;
      let drawn: { bitmap: ImageBitmap; us: number; metadata?: FrameMetadata } | null = null;
      let expectedUs: number | undefined;
      let compressed = false;
      let rangeStart: number | undefined;
      const rangeEnd = atUs + 1;
      let rangeFrames = 0;
      let receivedBytes = 0;
      let captureComplete = false;
      let captureRequested = false;
      let sentContext: number | null = null;
      const packets = new MediaPackets((buffer, media) => {
        if (media && sentContext !== media.context_id) {
          worker?.postMessage({ type: 'media', media });
          sentContext = media.context_id;
        }
        worker?.postMessage({ type: 'message', buffer }, [buffer]);
      });
      const abort = () => finish(new DOMException('Recorded frame capture cancelled.', 'AbortError'));
      const timer = setTimeout(
        () => finish(new DOMException('The recorded frame did not arrive in time.', 'TimeoutError')),
        timeoutMs,
      );
      const finish = (error?: unknown, result?: RecordedFrame) => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        signal?.removeEventListener('abort', abort);
        pending?.close();
        pending = null;
        drawn?.bitmap.close();
        drawn = null;
        worker?.terminate();
        connection?.close();
        if (result) resolve(result);
        else reject(error);
      };
      signal?.addEventListener('abort', abort, { once: true });
      try {
        const canvas = (deps.makeCanvas ?? (() => document.createElement('canvas')))();
        const context = canvas.getContext('2d');
        if (!context) throw new Error('This browser cannot capture the recorded picture.');
        const capture = () => {
          if (finished || !drawn || drawn.us !== expectedUs || (compressed && !captureComplete)) return;
          try {
            const { bitmap, us } = drawn;
            const { width, height } = bitmap;
            if (
              !Number.isSafeInteger(width) ||
              !Number.isSafeInteger(height) ||
              width < 1 ||
              height < 1 ||
              width > MAX_IMAGE_SIDE ||
              height > MAX_IMAGE_SIDE ||
              width * height > MAX_IMAGE_PIXELS
            )
              throw new Error('The recorded frame exceeds the image size limit.');
            canvas.width = width;
            canvas.height = height;
            context.drawImage(bitmap, 0, 0);
            const image = canvas.toDataURL('image/jpeg', 0.9);
            if (
              !image.startsWith('data:image/jpeg;base64,') ||
              image.length <= 'data:image/jpeg;base64,'.length ||
              image.length > MAX_IMAGE_CHARS
            )
              throw new Error('The recorded frame could not be encoded within the image size limit.');
            finish(undefined, {
              image,
              captured_at: us / US,
              ...(drawn.metadata ? { frame_metadata: drawn.metadata } : {}),
            });
          } catch (error) {
            finish(error);
          }
        };
        worker = (deps.makeWorker ?? realWorker)();
        worker.onmessage = ({ data }) => {
          if (finished) {
            if (data.type === 'frame') data.bitmap.close();
            return;
          }
          if (data.type === 'frame') {
            pending?.close();
            drawn?.bitmap.close();
            drawn = null;
            pending = data.bitmap;
          } else if (data.type === 'drawn' && pending) {
            drawn = { bitmap: pending, us: data.frameUs, metadata: data.metadata };
            pending = null;
            if (data.overlay !== 'none') finish(new Error('The recorded frame contains an unexpected overlay.'));
            else capture();
          } else if (data.type === 'captured' && captureRequested && data.requestId === CAPTURE_REQUEST_ID) {
            if (
              rangeStart === undefined ||
              data.frameUs === null ||
              !Number.isSafeInteger(data.frameUs) ||
              data.frameUs < seg.startUs ||
              data.frameUs > atUs
            )
              finish(new Error('No decoded frame is available at the selected playhead.'));
            else {
              expectedUs = data.frameUs;
              captureComplete = true;
              capture();
            }
          } else if (data.type === 'error') {
            finish(new Error(data.message));
          } else if (data.type === 'codec' && !data.supported) {
            finish(new DOMException('This browser cannot decode the recorded frame.', 'NotSupportedError'));
          }
        };
        const surface = new OffscreenCanvas(1, 1);
        worker.postMessage({ type: 'canvas', canvas: surface, returnBitmap: true }, [surface]);
        worker.postMessage({ type: 'playhead', us: atUs });
        const startRange = () => {
          if (!compressed || expectedUs === undefined || rangeStart !== undefined || !connection) return;
          if (atUs - expectedUs > CAPTURE_LOOKBACK_US) {
            finish(new Error('The recorded keyframe is beyond the capture lookback limit.'));
            return;
          }
          rangeStart = expectedUs;
          try {
            connection.send({
              type: 'range',
              stream: seg.stream,
              start_us: rangeStart,
              end_us: rangeEnd,
              max_frames: CAPTURE_MAX_FRAMES,
              overlays: [],
              media_context: true,
            });
          } catch (error) {
            finish(error);
          }
        };
        const handlers: Handlers = {
          onReply: reply => {
            if (finished) return;
            if (reply.type === 'media') {
              try {
                packets.context(reply);
              } catch (error) {
                finish(error);
              }
            } else if (reply.type === 'error') finish(new Error(reply.message));
            else if (reply.type === 'frame' && reply.stream === seg.stream && reply.at_us === atUs) {
              if (reply.capture_us === null) finish(new Error('No recorded frame is available at this time.'));
              else if (
                !Number.isSafeInteger(reply.capture_us) ||
                reply.capture_us < seg.startUs ||
                reply.capture_us > atUs
              )
                finish(new Error('The gateway returned an invalid recorded frame time.'));
              else {
                expectedUs = reply.capture_us;
                if (compressed) startRange();
                else capture();
              }
            } else if (
              reply.type === 'done' &&
              compressed &&
              rangeStart !== undefined &&
              reply.stream === seg.stream &&
              reply.start_us === rangeStart &&
              reply.end_us === rangeEnd
            ) {
              if (
                reply.skipped !== 0 ||
                reply.frames !== rangeFrames ||
                rangeFrames === 0 ||
                (reply.next_us !== null && reply.next_us < rangeEnd)
              )
                finish(new Error('The recorded frame interval is incomplete or exceeds the capture limit.'));
              else {
                captureRequested = true;
                worker?.postMessage({ type: 'capture', requestId: CAPTURE_REQUEST_ID, us: atUs });
              }
            }
          },
          onBinary: buffer => {
            if (finished) return;
            try {
              receivedBytes += buffer.byteLength;
              if (receivedBytes > CAPTURE_MAX_BYTES)
                throw new Error('The recorded frame exceeds the capture byte limit.');
              const header = decodeHeader(buffer);
              if (header.streamId !== seg.stream || header.live) return;
              if (header.payloadType !== PayloadType.jpeg && header.payloadType !== PayloadType.video) return;
              if (rangeStart === undefined && header.payloadType === PayloadType.video) {
                // The probe's keyframe identifies where decoding must start, never the final captured picture.
                compressed = true;
                startRange();
                return;
              }
              if (rangeStart !== undefined) {
                rangeFrames++;
                if (rangeFrames > CAPTURE_MAX_FRAMES || header.captureUs > atUs + GOP_SLACK_US)
                  throw new Error('The recorded frame interval exceeds the capture limit.');
              } else if (header.captureUs > atUs) return;
              packets.packet(buffer);
            } catch (error) {
              finish(error);
            }
          },
          onClose: reason => finish(new Error(`Recorded frame connection closed: ${reason}`)),
        };
        const url =
          deps.url ?? defaultGatewayUrl(typeof location === 'undefined' ? { hostname: '', protocol: '' } : location);
        // Capture uses ordered WebSocket delivery, not independent WebTransport binary streams.
        const connect = deps.connect ?? (async (address: string, h: Handlers) => connectWebSocket(address, h));
        void connect(url, handlers).then(
          opened => {
            if (finished) opened.close();
            else if (opened.transport !== 'websocket') {
              opened.close();
              finish(new Error('Recorded frame capture requires an ordered WebSocket connection.'));
            } else {
              connection = opened;
              try {
                opened.send({
                  type: 'frame',
                  stream: seg.stream,
                  at_us: atUs,
                  media_context: true,
                });
              } catch (error) {
                finish(error);
              }
            }
          },
          error => finish(error),
        );
      } catch (error) {
        finish(error);
      }
    });
  };
  return capture;
}

/** Capture the native recorded frame at an absolute UTC playhead, without any Review overlays. */
export const captureRecordedFrame = createRecordedFrameCapture();
