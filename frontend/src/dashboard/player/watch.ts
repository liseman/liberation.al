// One gateway connection per page for every stream the page follows live: the gateway's `watch` sends each
// watched stream's frames on it, at the rate asked (its newest frame that decodes alone, at most max_fps a second),
// or every frame at 0. Consumers subscribe by stream name with the rate they need; the connection asks for each
// stream at the fastest any subscriber wants, and a new `watch` replaces the set whenever that changes.
//
// While the page is hidden the connection is closed outright, so a background tab costs nothing; it opens again when
// the page shows. A connection that drops is opened again with backoff while anything is subscribed. A gateway
// without the `watch` feature (an older one) gets no `watch`; `watchAvailable()` says so, and consumers keep their
// own ways.

import { decodeHeader, type FrameHeader } from './header';
import { defaultGatewayUrl, type MediaContext, type StreamSummary } from './protocol';
import { type Connection, connect as connectGateway, type Handlers, type Reply } from './transport';

/** A frame of a watched stream: its header, the whole message, and its media context when the gateway sent one. */
export interface WatchFrame {
  header: FrameHeader;
  buffer: ArrayBuffer;
  media: MediaContext | null;
}

type Connect = (url: string, handlers: Handlers) => Promise<Connection>;
type Listener = (frame: WatchFrame) => void;
/** What else a subscriber wants besides the stream's frames. */
export interface WatchOptions {
  /** Overlay groups (faces, objects, poses) to follow with the stream; only at every frame (fps 0) does the gateway
   *  send them. Their frames come to the same listener. */
  overlays?: readonly string[];
  /** The connection went: what was decoding its frames starts over with the next keyframe. */
  onReset?: () => void;
  /** This gateway doesn't answer `watch` (an older one): follow the stream another way. */
  onUnavailable?: () => void;
}
interface Subscriber extends WatchOptions {
  fps: number;
  fn: Listener;
}

/** The gateway caps a rate here; 0 is every frame. */
export const MAX_FPS = 10;
const RETRY_MS = [1000, 2000, 5000, 15000] as const;
/** How soon a stream the gateway didn't list is asked about again. */
const RELIST_MS = 10_000;

let connectImpl: Connect = connectGateway;
let urlOf = (): string =>
  defaultGatewayUrl(typeof location === 'undefined' ? { hostname: '', protocol: '' } : location);

const subscribers = new Map<string, Set<Subscriber>>();
const ids = new Map<string, number>(); // stream name -> id, from the gateway's `streams`
const names = new Map<number, string>();
const contexts = new Map<number, MediaContext>(); // stream id -> its newest media context
const owners = new Map<number, number>(); // overlay stream id -> the watched stream it comes with (`watching`)
let conn: Connection | null = null;
let opening: Promise<Connection | null> | null = null;
let generation = 0;
let ready = false; // hello and streams have come on this connection
let available: boolean | null = null; // the gateway answers `watch`; null until a hello says
let sent = ''; // the watch set last sent, so an unchanged one isn't sent again
let listedAt = 0;
/** The gateway runs a tagged `streams` beside the connection's frames (its hello says `streams_req`). */
let taggedStreams = false;
let nextReq = 1;
let retry = 0;
let retryTimer: ReturnType<typeof setTimeout> | undefined;
let syncQueued = false;
/** Holders keep the connection open with nothing watched (holdWatch), so a stream list can be asked on it. */
let holds = 0;
/** Callers of listStreams waiting for the gateway's next `streams`. */
let listWaiters: { resolve: (s: StreamSummary[]) => void; reject: (e: Error) => void }[] = [];

/** For tests: the connection to use, and the gateway's address. Forgets everything. */
export function setWatchForTests(connect: Connect, url = 'ws://gateway/ws'): void {
  connectImpl = connect;
  urlOf = () => url;
  close();
  subscribers.clear();
  holds = 0;
  for (const w of listWaiters) w.reject(new Error('reset for a test'));
  listWaiters = [];
  available = null;
  retry = 0;
  listedAt = 0;
  taggedStreams = false;
  nextReq = 1;
}

/** Whether the gateway answers `watch`: null until it has said hello. */
export function watchAvailable(): boolean | null {
  return available;
}

const hidden = (): boolean => typeof document !== 'undefined' && document.hidden;

/** The rate the gateway is asked for: 0 (every frame) when anyone wants that, else the fastest wanted. */
function rateOf(subs: Set<Subscriber>): number {
  let best = Number.NEGATIVE_INFINITY;
  for (const s of subs) {
    if (s.fps === 0) return 0;
    best = Math.max(best, s.fps);
  }
  return Math.min(best, MAX_FPS);
}

function close(): void {
  if (ready) for (const subs of subscribers.values()) for (const s of subs) s.onReset?.();
  generation++;
  clearTimeout(retryTimer);
  retryTimer = undefined;
  const old = conn;
  conn = null;
  opening = null;
  ready = false;
  sent = '';
  contexts.clear();
  owners.clear();
  old?.close();
}

/** Sends the watch set for what's subscribed now (after the current task, so a burst of changes sends one). */
function sync(): void {
  if (syncQueued) return;
  syncQueued = true;
  queueMicrotask(() => {
    syncQueued = false;
    apply();
  });
}

function apply(): void {
  if ((!subscribers.size && !holds) || hidden()) {
    close();
    return;
  }
  if (available === false) {
    failListers(new Error("this gateway doesn't answer watch"));
    return;
  }
  if (!conn) {
    open();
    return;
  }
  if (!ready) return;
  const streams: { id: number; max_fps: number; overlays?: string[] }[] = [];
  let missing = false;
  for (const [name, subs] of subscribers) {
    const id = ids.get(name);
    if (id === undefined) {
      missing = true;
      continue;
    }
    const max_fps = rateOf(subs);
    const overlays = [...new Set([...subs].flatMap(s => s.overlays ?? []))].sort();
    streams.push(max_fps === 0 && overlays.length ? { id, max_fps, overlays } : { id, max_fps });
  }
  if (missing && Date.now() - listedAt >= RELIST_MS) {
    listedAt = Date.now();
    askStreams(conn);
  }
  streams.sort((a, b) => a.id - b.id);
  const body = JSON.stringify(streams);
  if (body === sent || (!streams.length && sent === '')) return; // unchanged, or nothing watched to stop
  sent = body;
  conn.send(streams.length ? { type: 'watch', streams, media_context: true } : { type: 'unwatch' });
}

function open(): void {
  if (opening) return;
  const own = ++generation;
  opening = connectImpl(urlOf(), {
    onReply: r => {
      if (own === generation) reply(r);
    },
    onBinary: buffer => {
      if (own === generation) binary(buffer);
    },
    onClose: () => {
      if (own === generation) lost();
    },
  }).then(
    c => {
      if (own !== generation) {
        c.close();
        return null;
      }
      conn = c;
      return c;
    },
    () => {
      if (own === generation) lost();
      return null;
    },
  );
}

/** Ask the gateway for its streams on the watch connection. Tagged when it can run the listing beside the
 *  frames: untagged, the connection sent nothing until the listing was done, which took about 1.8 s when it re-read
 *  live chunks, long enough for every camera's queue to overflow and wait for a keyframe. */
function askStreams(c: Connection | null): void {
  c?.send(taggedStreams ? { type: 'streams', req: nextReq++ } : { type: 'streams' });
}

function failListers(error: Error): void {
  const waiting = listWaiters;
  listWaiters = [];
  for (const w of waiting) w.reject(error);
}

function lost(): void {
  close();
  failListers(new Error('the gateway connection went'));
  if ((!subscribers.size && !holds) || hidden() || available === false) return;
  const wait = RETRY_MS[Math.min(retry, RETRY_MS.length - 1)];
  retry++;
  retryTimer = setTimeout(() => {
    retryTimer = undefined;
    apply();
  }, wait);
}

function reply(r: Reply): void {
  if (r.type === 'hello') {
    available = r.features?.includes('watch') ?? false;
    taggedStreams = r.features?.includes('streams_req') ?? false;
    if (!available) {
      close();
      failListers(new Error("this gateway doesn't answer watch"));
      for (const subs of subscribers.values()) for (const s of subs) s.onUnavailable?.();
      return;
    }
    listedAt = Date.now();
    askStreams(conn);
  } else if (r.type === 'streams') {
    ids.clear();
    names.clear();
    for (const s of r.streams)
      if (s.name) {
        ids.set(s.name, s.id);
        names.set(s.id, s.name);
      }
    ready = true;
    retry = 0;
    const waiting = listWaiters;
    listWaiters = [];
    for (const w of waiting) w.resolve(r.streams);
    apply();
  } else if (r.type === 'unwatched' && r.reason) {
    lost(); // the gateway ended it (its broker connection went)
  } else if (r.type === 'watching') {
    owners.clear();
    for (const s of r.streams) for (const o of s.overlays) owners.set(o.id, s.id);
  } else if (r.type === 'media') contexts.set(r.stream, r);
}

function binary(buffer: ArrayBuffer): void {
  let header: FrameHeader;
  try {
    header = decodeHeader(buffer);
  } catch {
    return; // not a frame this page can read: skip it, as the player does
  }
  if (!header.watch) return; // a `live` feed or a request's frames: not the watch set's
  // An overlay comes on its own stream: hand it to the subscribers of the stream it was asked with.
  const name = names.get(owners.get(header.streamId) ?? header.streamId);
  const subs = name === undefined ? undefined : subscribers.get(name);
  if (!subs) return;
  const context = contexts.get(header.streamId);
  const frame: WatchFrame = {
    header,
    buffer,
    media: context && context.context_id === header.contextId ? context : null,
  };
  for (const s of subs) s.fn(frame);
}

/** Follows stream (a name, such as `kitchen.video`) at fps frames a second (0: every frame) until the function it
 *  returns is called. Every subscriber gets every frame of the stream that comes, at the fastest rate any asked. */
export function watchStream(stream: string, fps: number, fn: Listener, options: WatchOptions = {}): () => void {
  const mine: Subscriber = { ...options, fps: Math.max(0, fps), fn };
  if (available === false && options.onUnavailable) queueMicrotask(() => options.onUnavailable?.());
  let subs = subscribers.get(stream);
  if (!subs) {
    subs = new Set();
    subscribers.set(stream, subs);
  }
  subs.add(mine);
  sync();
  let done = false;
  return () => {
    if (done) return;
    done = true;
    const now = subscribers.get(stream);
    now?.delete(mine);
    if (now && !now.size) subscribers.delete(stream);
    sync();
  };
}

if (typeof document !== 'undefined') document.addEventListener('visibilitychange', sync);

/** Keeps the page's watch connection open, even with nothing watched, until the function it returns is called. */
export function holdWatch(): () => void {
  holds++;
  sync();
  let done = false;
  return () => {
    if (done) return;
    done = true;
    holds--;
    sync();
  };
}

/** The gateway's streams, asked on the page's watch connection (no connection of their own). Rejects when the
 *  gateway doesn't answer `watch`, the connection goes, or the page is hidden, so a caller can ask another way. */
export function listStreams(timeoutMs = 5000): Promise<StreamSummary[]> {
  if (available === false) return Promise.reject(new Error("this gateway doesn't answer watch"));
  if (hidden()) return Promise.reject(new Error('the page is hidden'));
  const release = holdWatch();
  return new Promise<StreamSummary[]>((resolve, reject) => {
    const timer = setTimeout(() => {
      listWaiters = listWaiters.filter(w => w !== waiter);
      reject(new Error('the gateway took too long to list its streams'));
    }, timeoutMs);
    const waiter = {
      resolve: (streams: StreamSummary[]) => {
        clearTimeout(timer);
        resolve(streams);
      },
      reject: (error: Error) => {
        clearTimeout(timer);
        reject(error);
      },
    };
    listWaiters.push(waiter);
    // On a ready connection, ask now; one still opening asks for the streams itself after its hello.
    if (conn && ready) askStreams(conn);
  }).finally(release);
}
