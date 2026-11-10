// The connection to the gateway: WebSockets by default, WebTransport when the gateway advertises it and the browser
// has it. Either way the page sees the same thing: control replies as JSON, binary messages as ArrayBuffers, and one
// send() for requests. Pure enough to test with a fake WebSocket; the WebTransport path is exercised in Chrome.

import type { MediaContext, StreamSummary } from './protocol';

/** A reply from the gateway (continuum-gateway's control.rs). */
export type Reply =
  | MediaContext
  | {
      type: 'hello';
      transport: string;
      webtransport: { url: string; cert_hash: string } | null;
      broker?: boolean;
      media_context?: boolean;
      /** The vision overlays a page can ask for by name (faces, bodies, poses, objects), with the broker. */
      overlay_groups?: string[];
      /** What this gateway answers beyond the basics: `watch` (many streams live) and `req` (request ids), #2380. */
      features?: string[];
    }
  | { type: 'streams'; streams: StreamSummary[] }
  | { type: 'live'; stream: number; overlays: { name: string; id: number }[] }
  | { type: 'watching'; streams: { id: number; max_fps: number; overlays: { name: string; id: number }[] }[] }
  | { type: 'unwatched'; frames: number; dropped: number; skipped: number; reason?: string }
  | { type: 'stopped'; frames: number; dropped: number; skipped: number; reason?: string }
  | {
      type: 'done';
      stream: number;
      start_us: number;
      end_us: number;
      frames: number;
      skipped: number;
      next_us: number | null;
      codecs: string[];
    }
  | {
      type: 'spans';
      stream: number;
      start_us: number;
      end_us: number;
      spans: [number, number][];
      /** Stretches whose video aged out past the history window (#1748). */
      expired?: [number, number][];
      /** The window held more spans than one reply lists: ask again over its halves. */
      partial?: boolean;
    }
  | { type: 'frame'; stream: number; at_us: number; capture_us: number | null }
  | { type: 'pong' }
  | { type: 'error'; message: string };

export interface Handlers {
  onReply: (reply: Reply) => void;
  onBinary: (buffer: ArrayBuffer) => void;
  onClose: (reason: string) => void;
}

export interface Connection {
  readonly transport: 'websocket' | 'webtransport';
  send(request: object): void;
  close(): void;
}

/** What a connection is built from, so tests can hand in fakes. */
export interface Environment {
  WebSocket: typeof WebSocket;
  WebTransport?: new (url: string, options?: object) => WebTransportLike;
  fetch: typeof fetch;
}

/** The slice of WebTransport the page uses (the lib.dom type is behind a flag in some TypeScript versions). */
export interface WebTransportLike {
  ready: Promise<unknown>;
  closed: Promise<unknown>;
  incomingUnidirectionalStreams: ReadableStream<ReadableStream<Uint8Array>>;
  createBidirectionalStream(): Promise<{ readable: ReadableStream<Uint8Array>; writable: WritableStream<Uint8Array> }>;
  close(): void;
}

/** The HTTP base (`http://host:port`) for a gateway WebSocket URL (`ws://host:port/ws`). */
export function httpBase(wsUrl: string): string {
  const u = new URL(wsUrl);
  u.protocol = u.protocol === 'wss:' ? 'https:' : 'http:';
  u.pathname = '';
  u.search = '';
  return u.toString().replace(/\/$/, '');
}

/** Hex to the bytes WebTransport wants in serverCertificateHashes. */
export function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

/** Opens a WebSocket connection to `wsUrl` (`ws://host:8766/ws`). */
export function connectWebSocket(
  wsUrl: string,
  handlers: Handlers,
  env: Environment = globalThis as unknown as Environment,
): Connection {
  const socket = new env.WebSocket(wsUrl);
  socket.binaryType = 'arraybuffer';
  socket.onmessage = (e: MessageEvent) => {
    if (typeof e.data === 'string') handlers.onReply(JSON.parse(e.data) as Reply);
    else if (e.data instanceof ArrayBuffer) handlers.onBinary(e.data);
  };
  let closed = false;
  const close = (reason: string) => {
    if (closed) return;
    closed = true;
    handlers.onClose(reason);
  };
  socket.onclose = (e: CloseEvent) => close(e.reason || `closed (${e.code})`);
  socket.onerror = () => close('socket error');
  return {
    transport: 'websocket',
    send: request => {
      if (socket.readyState === socket.OPEN) socket.send(JSON.stringify(request));
      else if (socket.readyState === socket.CONNECTING)
        socket.addEventListener('open', () => socket.send(JSON.stringify(request)), { once: true });
      // Closing or closed: the request is dropped; onClose already said so.
    },
    close: () => socket.close(),
  };
}

/** Opens a WebTransport session: control on one bidirectional stream (u32 LE length then JSON, both ways), each
 *  binary message its own incoming unidirectional stream. */
export async function connectWebTransport(
  info: { url: string; cert_hash: string },
  handlers: Handlers,
  env: Environment = globalThis as unknown as Environment,
): Promise<Connection> {
  if (!env.WebTransport) throw new Error('this browser has no WebTransport');
  const options = info.cert_hash
    ? { serverCertificateHashes: [{ algorithm: 'sha-256', value: hexToBytes(info.cert_hash) }] }
    : {};
  const wt = new env.WebTransport(info.url, options);
  await wt.ready;
  const control = await wt.createBidirectionalStream();
  const writer = control.writable.getWriter();
  void readControl(control.readable, handlers);
  void readUni(wt.incomingUnidirectionalStreams, handlers);
  void wt.closed.then(
    () => handlers.onClose('closed'),
    (e: unknown) => handlers.onClose(e instanceof Error ? e.message : String(e)),
  );
  return {
    transport: 'webtransport',
    send: request => {
      const body = new TextEncoder().encode(JSON.stringify(request));
      const out = new Uint8Array(4 + body.length);
      new DataView(out.buffer).setUint32(0, body.length, true);
      out.set(body, 4);
      void writer.write(out);
    },
    close: () => wt.close(),
  };
}

async function readControl(readable: ReadableStream<Uint8Array>, handlers: Handlers): Promise<void> {
  const reader = readable.getReader();
  let buf = new Uint8Array(0);
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    const joined = new Uint8Array(buf.length + value.length);
    joined.set(buf);
    joined.set(value, buf.length);
    buf = joined;
    for (;;) {
      if (buf.length < 4) break;
      const len = new DataView(buf.buffer, buf.byteOffset, 4).getUint32(0, true);
      if (buf.length < 4 + len) break;
      handlers.onReply(JSON.parse(new TextDecoder().decode(buf.subarray(4, 4 + len))) as Reply);
      buf = buf.slice(4 + len);
    }
  }
}

/** Each binary message is its own stream, read as they arrive but handed over in the order the gateway opened them:
 *  a long-GOP access unit must reach the decoder after its keyframe, and QUIC completes streams in any order. */
async function readUni(streams: ReadableStream<ReadableStream<Uint8Array>>, handlers: Handlers): Promise<void> {
  const reader = streams.getReader();
  let chain: Promise<void> = Promise.resolve();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    const bytes = collect(value);
    chain = chain.then(() => bytes).then(handlers.onBinary, () => undefined);
  }
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<ArrayBuffer> {
  const parts: Uint8Array[] = [];
  const reader = stream.getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parts.push(value);
  }
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out.buffer;
}

/** How long a gateway's answer to `GET /wt` is reused: every connection asked it, 20 times in one Review scrub
 *  (the audit), and it changes only when the gateway restarts. */
export const WT_INFO_MS = 60_000;
type WtInfo = { url: string; cert_hash: string } | null;
const wtInfo = new Map<string, { at: number; info: Promise<WtInfo> }>();

/** The gateway's WebTransport details (null when it's off), asked once a minute at most per gateway. */
function wtInfoOf(wsUrl: string, env: Environment): Promise<WtInfo> {
  const base = httpBase(wsUrl);
  const cached = wtInfo.get(base);
  if (cached && Date.now() - cached.at < WT_INFO_MS) return cached.info;
  // null when it's off (an older gateway answered 404)
  const info = env.fetch(`${base}/wt`).then(res => (res.ok ? (res.json() as Promise<WtInfo>) : null));
  wtInfo.set(base, { at: Date.now(), info });
  info.catch(() => {
    if (wtInfo.get(base)?.info === info) wtInfo.delete(base); // a failed ask is tried again next time
  });
  return info;
}

/** For tests: forget the gateways' WebTransport answers. */
export function forgetWtInfoForTests(): void {
  wtInfo.clear();
}

/** Connects the best way available: WebTransport when `GET /wt` on the gateway says it's on and the browser has it,
 *  else WebSockets. `prefer` forces one. */
export async function connect(
  wsUrl: string,
  handlers: Handlers,
  prefer: 'auto' | 'websocket' | 'webtransport' = 'auto',
  env: Environment = globalThis as unknown as Environment,
): Promise<Connection> {
  if (prefer !== 'websocket' && env.WebTransport) {
    try {
      const info = await wtInfoOf(wsUrl, env);
      if (info) return await connectWebTransport(info, handlers, env);
      if (prefer === 'webtransport') throw new Error('the gateway has WebTransport off');
    } catch (e) {
      if (prefer === 'webtransport') throw e;
    }
  } else if (prefer === 'webtransport') throw new Error('this browser has no WebTransport');
  return connectWebSocket(wsUrl, handlers, env);
}
