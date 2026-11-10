// Sparse overview pictures: one ordered read-only gateway connection and one decoder shared by all visible tiles.
// Every flight asks for an independent JPEG/keyframe, verifies its final reply, and flushes the existing player
// worker before publishing pixels. Unlike Live, a preview can be archived, so its media geometry stays intact.
import { decodeHeader, PayloadType } from '../player/header';
import { defaultGatewayUrl, type MediaContext } from '../player/protocol';
import { type Connection, connectWebSocket, type Handlers, type Reply } from '../player/transport';
import type { FromWorker, WorkerLike } from '../player/worker';
import { watchFrames } from './gateway';

export interface PreviewState {
  status: 'loading' | 'ready' | 'stale' | 'unavailable';
  capturedAt: number | null;
  error: string | null;
}
const REFRESH_MS = 3000;
const SPACING_MS = 250;
const BUDGET_MS = 5000;
const STALE_MS = 10000;
const MAX_PACKET_BYTES = 16 * 1024 * 1024;
type Listener = (state: PreviewState) => void;
interface Tile {
  cam: string;
  views: Map<HTMLCanvasElement, Listener>;
  state: PreviewState;
  due: number;
}
interface Flight {
  tile: Tile;
  at: number;
  id: number;
  stream: number | null;
  packet: ArrayBuffer | null;
  media: MediaContext | null;
  expected: number | null;
  bitmap: Extract<FromWorker, { type: 'frame' }> | null;
}
interface Dependencies {
  connect: (url: string, handlers: Handlers) => Connection;
  worker: () => WorkerLike;
  surface: () => OffscreenCanvas;
  url: () => string;
  /** WebCodecs is there: false on a plain-http page whose host isn't localhost (not a secure context). */
  canDecodeVideo: () => boolean;
  /** Live JPEG frames of a camera, as Live gets them (gateway.ts); returns the function that stops. */
  watch: (cam: string, fn: (jpeg: Uint8Array, capturedAt: number, flip: boolean) => void) => () => void;
  bitmap: (jpeg: Uint8Array) => Promise<ImageBitmap>;
}
const defaults: Dependencies = {
  connect: connectWebSocket,
  worker: () => {
    const own = new Worker(new URL('../player/worker.ts', import.meta.url), { type: 'module' });
    own.onerror = () => {
      if (worker === own) finish('The camera preview decoder failed; retrying.');
    };
    return own;
  },
  surface: () => new OffscreenCanvas(1, 1),
  url: () => defaultGatewayUrl(location),
  canDecodeVideo: () => typeof VideoDecoder !== 'undefined',
  watch: watchFrames,
  bitmap: jpeg => createImageBitmap(new Blob([jpeg as BlobPart], { type: 'image/jpeg' })),
};
let deps = defaults;
/** Browser boundaries only; tests retain the real scheduler, association checks, and canvas commits. */
export function setPreviewForTests(overrides: Partial<Dependencies>): void {
  deps = { ...defaults, ...overrides };
}

const tiles = new Map<string, Tile>();
const streams = new Map<string, number>();
let conn: Connection | null = null;
let worker: WorkerLike | null = null;
let flight: Flight | null = null;
let generation = 0;
let serial = 0;
let lastRequest = 0;
let gapUntil = 0;
let schedule: ReturnType<typeof setTimeout> | undefined;
let deadline: ReturnType<typeof setTimeout> | undefined;
let ticker: ReturnType<typeof setInterval> | undefined;
let liveStop: (() => void) | undefined; // the live JPEG watch a fallback flight holds

function publish(tile: Tile): void {
  if (tile.state.capturedAt !== null && Date.now() - tile.state.capturedAt * 1000 > STALE_MS)
    tile.state = { ...tile.state, status: 'stale' };
  for (const notify of tile.views.values()) notify({ ...tile.state });
}

function stopLive(): void {
  const stop = liveStop;
  liveStop = undefined;
  stop?.();
}

function closeTransport(): void {
  stopLive();
  generation++;
  const old = conn;
  conn = null;
  old?.close();
  worker?.terminate();
  worker = null;
  streams.clear();
}

function finish(error: string | null): void {
  const done = flight;
  if (!done) return;
  clearTimeout(deadline);
  deadline = undefined;
  stopLive();
  done.bitmap?.bitmap.close();
  flight = null;
  done.tile.due = Date.now() + REFRESH_MS;
  gapUntil = Date.now() + SPACING_MS;
  if (error) {
    done.tile.state = {
      ...done.tile.state,
      status: done.tile.state.capturedAt === null ? 'unavailable' : 'stale',
      error,
    };
    closeTransport(); // no untagged delayed packet/reply may become the next camera's frame
  }
  publish(done.tile);
  kick();
}

function capture(message: FromWorker, own: WorkerLike): void {
  if (worker !== own || !flight) {
    if (message.type === 'frame') message.bitmap.close();
    return;
  }
  const f = flight;
  if (message.type === 'frame') {
    if (message.epoch !== f.id) {
      message.bitmap.close();
      return;
    }
    f.bitmap?.bitmap.close();
    f.bitmap = message;
  } else if (message.type === 'captured' && message.requestId === f.id) {
    const pixels = f.bitmap;
    if (message.frameUs !== f.expected || !pixels || pixels.frameUs !== f.expected) {
      finish('The preview decoder did not return the requested picture.');
      return;
    }
    try {
      for (const canvas of f.tile.views.keys()) {
        const ctx = canvas.getContext('2d');
        if (!ctx) throw Error('This browser cannot draw camera previews.');
        canvas.width = pixels.bitmap.width;
        canvas.height = pixels.bitmap.height;
        delete canvas.dataset.capturedAt;
        delete canvas.dataset.frameMetadata;
        ctx.drawImage(pixels.bitmap, 0, 0);
        canvas.dataset.capturedAt = String(pixels.frameUs / 1e6);
        if (pixels.metadata) canvas.dataset.frameMetadata = JSON.stringify(pixels.metadata);
      }
      f.tile.state = { status: 'ready', capturedAt: pixels.frameUs / 1e6, error: null };
      finish(null);
    } catch (error) {
      // Resizing or a partial multi-view commit may have cleared/replaced pixels before drawing failed.
      // Invalidate every view together; a retained prior timestamp would describe pixels no longer on screen.
      for (const canvas of f.tile.views.keys()) {
        delete canvas.dataset.capturedAt;
        delete canvas.dataset.frameMetadata;
      }
      f.tile.state = { status: 'unavailable', capturedAt: null, error: null };
      finish(error instanceof Error ? error.message : 'Could not draw the camera preview.');
    }
  } else if (message.type === 'error') finish(message.message);
  else if (message.type === 'codec' && !message.supported)
    finish(`This browser cannot decode this camera's ${message.codec.toUpperCase()} preview.`);
}

function decode(f: Flight): void {
  if (!f.packet || f.expected === null) return;
  try {
    const header = decodeHeader(f.packet);
    if (header.captureUs !== f.expected || header.contextId !== f.media?.context_id)
      throw Error('The gateway preview metadata does not match its picture.');
    if (!worker) {
      const own = deps.worker();
      worker = own;
      own.onmessage = e => capture(e.data, own);
      const surface = deps.surface();
      own.postMessage({ type: 'canvas', canvas: surface, returnBitmap: true }, [surface]);
    }
    worker.postMessage({ type: 'reset' });
    worker.postMessage({ type: 'playhead', us: f.expected });
    if (f.media) worker.postMessage({ type: 'media', media: f.media });
    const copy = f.packet.slice(0);
    worker.postMessage({ type: 'message', buffer: copy, epoch: f.id }, [copy]);
    worker.postMessage({ type: 'capture', requestId: f.id, us: f.expected });
  } catch (error) {
    finish(error instanceof Error ? error.message : 'Could not decode the camera preview.');
  }
}

function request(): void {
  if (!flight || !conn) return;
  const id = streams.get(flight.tile.cam);
  if (id === undefined) {
    finish('This camera has no preview stream yet.');
    return;
  }
  flight.stream = id;
  try {
    conn.send({ type: 'frame', stream: id, at_us: flight.at, media_context: true });
  } catch {
    finish('Could not request the camera preview; retrying.');
  }
}

function reply(message: Reply): void {
  const f = flight;
  if (!f) return;
  if (message.type === 'hello') conn?.send({ type: 'streams' });
  else if (message.type === 'streams') {
    if (f.stream !== null) {
      finish('The gateway restarted the preview request.');
      return;
    }
    streams.clear();
    for (const stream of message.streams)
      if (stream.name?.endsWith('.video')) streams.set(stream.name.slice(0, -6), stream.id);
    request();
  } else if (message.type === 'media') {
    if (message.stream !== f.stream || f.media) {
      finish('The gateway returned unexpected preview metadata.');
      return;
    }
    f.media = message;
  } else if (message.type === 'frame') {
    if (message.stream !== f.stream || message.at_us !== f.at) {
      finish('The gateway returned an unrelated preview.');
      return;
    }
    if (message.capture_us === null) {
      finish('No recent preview picture is available.');
      return;
    }
    if (
      !Number.isSafeInteger(message.capture_us) ||
      message.capture_us <= 0 ||
      message.capture_us > f.at ||
      f.at - message.capture_us > 15_000_000 ||
      !f.packet
    ) {
      finish('The gateway returned an invalid preview capture time.');
      return;
    }
    f.expected = message.capture_us;
    decode(f);
  } else if (message.type === 'error') finish(message.message);
}

function binary(packet: ArrayBuffer): void {
  const f = flight;
  if (!f) return;
  try {
    const h = decodeHeader(packet);
    if (
      packet.byteLength > MAX_PACKET_BYTES ||
      f.packet ||
      h.streamId !== f.stream ||
      h.live ||
      (h.payloadType !== PayloadType.jpeg && h.payloadType !== PayloadType.video) ||
      (h.payloadType === PayloadType.video && !h.keyframe)
    )
      throw Error('The gateway returned an invalid preview packet.');
    f.packet = packet;
  } catch (error) {
    finish(error instanceof Error ? error.message : 'The preview packet could not be read.');
  }
}

// Without WebCodecs (a page on plain http under a name other than localhost, such as http://<host>.local:8765) the
// archived H.264 keyframe can't be decoded, and every tile said "Preview unavailable" while Live, whose raw cameras
// arrive as JPEG, still worked (2026-11-05). Then a tile takes one live JPEG frame, the way Live does.
const NO_DECODER = 'This page can’t decode camera video here; open the dashboard over https or on this computer.';

function startLive(tile: Tile): void {
  const mine: Flight = {
    tile,
    at: Date.now() * 1000,
    id: ++serial,
    stream: null,
    packet: null,
    media: null,
    expected: null,
    bitmap: null,
  };
  flight = mine;
  deadline = setTimeout(() => finish(NO_DECODER), BUDGET_MS);
  let taken = false;
  liveStop = deps.watch(tile.cam, (jpeg, capturedAt, flip) => {
    if (taken || flight !== mine) return;
    taken = true;
    stopLive();
    deps
      .bitmap(jpeg)
      .then(picture => {
        if (flight !== mine) {
          picture.close();
          return;
        }
        try {
          for (const canvas of tile.views.keys()) {
            const ctx = canvas.getContext('2d');
            if (!ctx) throw Error('This browser cannot draw camera previews.');
            canvas.width = picture.width;
            canvas.height = picture.height;
            delete canvas.dataset.frameMetadata;
            if (flip) {
              ctx.save();
              ctx.translate(picture.width, picture.height);
              ctx.scale(-1, -1);
              ctx.drawImage(picture, 0, 0);
              ctx.restore();
            } else ctx.drawImage(picture, 0, 0);
            canvas.dataset.capturedAt = String(capturedAt);
          }
          tile.state = { status: 'ready', capturedAt, error: null };
          finish(null);
        } catch (error) {
          finish(error instanceof Error ? error.message : 'Could not draw the camera preview.');
        } finally {
          picture.close();
        }
      })
      .catch(error => {
        if (flight === mine) finish(error instanceof Error ? error.message : 'Could not decode the camera preview.');
      });
  });
}

function start(tile: Tile): void {
  if (!deps.canDecodeVideo()) {
    startLive(tile);
    return;
  }
  lastRequest = Math.max(lastRequest + 1, Date.now() * 1000);
  flight = {
    tile,
    at: lastRequest,
    id: ++serial,
    stream: null,
    packet: null,
    media: null,
    expected: null,
    bitmap: null,
  };
  deadline = setTimeout(() => finish('Camera preview timed out; retrying.'), BUDGET_MS);
  if (conn) {
    request();
    return;
  }
  const own = ++generation;
  try {
    conn = deps.connect(deps.url(), {
      onReply: message => {
        if (generation === own) reply(message);
      },
      onBinary: packet => {
        if (generation === own) binary(packet);
      },
      onClose: () => {
        if (generation !== own) return;
        if (flight) finish('The preview connection closed; retrying.');
        else {
          closeTransport();
          kick();
        }
      },
    });
  } catch (error) {
    finish(error instanceof Error ? error.message : 'Could not open camera previews.');
  }
}

function kick(): void {
  clearTimeout(schedule);
  schedule = undefined;
  if (document.hidden || flight || !tiles.size) return;
  const next = [...tiles.values()].sort((a, b) => a.due - b.due)[0];
  if (!next) return;
  const wait = Math.max(0, next.due - Date.now(), gapUntil - Date.now());
  schedule = setTimeout(() => {
    schedule = undefined;
    if (!document.hidden && tiles.has(next.cam)) start(next);
  }, wait);
}

function visibility(): void {
  clearTimeout(schedule);
  schedule = undefined;
  if (document.hidden || !tiles.size) {
    clearInterval(ticker);
    ticker = undefined;
    clearTimeout(deadline);
    deadline = undefined;
    flight?.bitmap?.bitmap.close();
    flight = null;
    closeTransport();
  } else {
    ticker ??= setInterval(() => {
      for (const tile of tiles.values()) publish(tile);
    }, 1000);
    for (const tile of tiles.values()) {
      tile.due = Date.now();
      publish(tile);
    }
    kick();
  }
}

/** Attach a tile without a full Live subscription. Cleanup cancels queued work and invalidates in-flight work. */
export function attachPreview(cam: string, canvas: HTMLCanvasElement, notify: Listener): () => void {
  delete canvas.dataset.capturedAt;
  delete canvas.dataset.frameMetadata;
  let tile = tiles.get(cam);
  if (!tile) {
    tile = { cam, views: new Map(), state: { status: 'loading', capturedAt: null, error: null }, due: Date.now() };
    tiles.set(cam, tile);
  }
  const existing = [...tile.views.keys()].find(view => view.dataset.capturedAt !== undefined);
  if (existing) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      canvas.width = existing.width;
      canvas.height = existing.height;
      ctx.drawImage(existing, 0, 0);
      canvas.dataset.capturedAt = existing.dataset.capturedAt ?? '';
      if (existing.dataset.frameMetadata) canvas.dataset.frameMetadata = existing.dataset.frameMetadata;
    }
  }
  tile.views.set(canvas, notify);
  notify({ ...tile.state });
  if (!document.hidden)
    ticker ??= setInterval(() => {
      if (!document.hidden) for (const t of tiles.values()) publish(t);
    }, 1000);
  kick();
  let closed = false;
  return () => {
    if (closed) return;
    closed = true;
    tile.views.delete(canvas);
    if (tile.views.size) return;
    tiles.delete(cam);
    if (flight?.tile === tile) {
      clearTimeout(deadline);
      flight.bitmap?.bitmap.close();
      flight = null;
      gapUntil = Date.now() + SPACING_MS;
      closeTransport();
    }
    if (!tiles.size) visibility();
    else kick();
  };
}
if (typeof document !== 'undefined') document.addEventListener('visibilitychange', visibility);
