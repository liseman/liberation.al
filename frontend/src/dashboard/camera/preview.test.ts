import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { encodeHeader } from '../player/header';
import type { MediaContext } from '../player/protocol';
import type { Handlers } from '../player/transport';
import type { FromWorker, ToWorker } from '../player/worker';
import { controlVisibility, recordingCanvas } from './testFakes';

let preview: typeof import('./preview');
let wires: { handlers: Handlers; requests: Record<string, unknown>[]; close: ReturnType<typeof vi.fn> }[];
let workers: FakeWorker[];
let page: ReturnType<typeof controlVisibility>;
let offs: (() => void)[];
class FakeWorker {
  onmessage: ((event: MessageEvent<FromWorker>) => void) | null = null;
  messages: ToWorker[] = [];
  terminate = vi.fn();
  postMessage(message: ToWorker) {
    this.messages.push(message);
  }
  emit(data: FromWorker) {
    this.onmessage?.({ data } as MessageEvent<FromWorker>);
  }
}
const settle = (ms = 0) => vi.advanceTimersByTimeAsync(ms);
beforeEach(async () => {
  vi.useFakeTimers();
  vi.setSystemTime(100_000);
  vi.resetModules();
  page = controlVisibility();
  wires = [];
  workers = [];
  offs = [];
  preview = await import('./preview');
  preview.setPreviewForTests({
    url: () => 'ws://synthetic/ws',
    connect: (_url, handlers) => {
      const wire = { handlers, requests: [] as Record<string, unknown>[], close: vi.fn() };
      wires.push(wire);
      return {
        transport: 'websocket',
        send: request => wire.requests.push(request as Record<string, unknown>),
        close: wire.close,
      };
    },
    worker: () => {
      const worker = new FakeWorker();
      workers.push(worker);
      return worker;
    },
    surface: () => ({ width: 1, height: 1 }) as OffscreenCanvas,
    canDecodeVideo: () => true,
  });
});
afterEach(() => {
  for (const off of offs) off();
  page.restore();
  vi.useRealTimers();
});
function tile(cam: string) {
  const picture = recordingCanvas();
  const notify = vi.fn();
  const off = preview.attachPreview(cam, picture.canvas, notify);
  offs.push(off);
  return { ...picture, notify, off };
}
function wire() {
  const current = wires.at(-1);
  if (!current) throw Error('No connection');
  return current;
}
function boot() {
  wire().handlers.onReply({ type: 'hello', transport: 'websocket', webtransport: null });
  wire().handlers.onReply({
    type: 'streams',
    streams: ['camA', 'camB', 'camC'].map((cam, i) => ({
      id: i + 1,
      name: `${cam}.video`,
      format: 'h264',
      start_us: 0,
      end_us: 0,
      chunks: 0,
      frames: 0,
    })),
  });
}
function request() {
  const current = wire().requests.at(-1);
  if (current?.type !== 'frame') throw Error('No frame request');
  return { stream: Number(current.stream), at: Number(current.at_us) };
}
function packet(stream: number, at: number, video = true, contextId?: number) {
  const h = encodeHeader({
    streamId: stream,
    payloadType: video ? 1 : 4,
    keyframe: true,
    live: false,
    captureUs: at,
    contextId,
  });
  const bytes = new Uint8Array(h.length + 4);
  bytes.set(h);
  bytes.set([0, 0, 1, 0x65], h.length);
  return bytes.buffer;
}
function offered(media?: MediaContext) {
  const r = request(),
    at = r.at - 1_000_000;
  if (media) wire().handlers.onReply(media);
  wire().handlers.onBinary(packet(r.stream, at, true, media?.context_id));
  wire().handlers.onReply({ type: 'frame', stream: r.stream, at_us: r.at, capture_us: at });
  return at;
}
function displayed(at: number, metadata?: Extract<FromWorker, { type: 'frame' }>['metadata']) {
  const worker = workers.at(-1);
  if (!worker) throw Error('No decoder');
  const capture = worker.messages.at(-1),
    message = worker.messages.findLast(m => m.type === 'message');
  if (capture?.type !== 'capture' || message?.type !== 'message') throw Error('No flushed capture request');
  const bitmap = { width: 720, height: 1280, close: vi.fn() } as unknown as ImageBitmap;
  worker.emit({ type: 'frame', bitmap, frameUs: at, metadata, epoch: message.epoch });
  worker.emit({ type: 'captured', requestId: capture.requestId, frameUs: at });
  return bitmap;
}

it('serializes and spaces a fair camera rotation over one read-only connection and one decoder', async () => {
  const a = tile('camA'),
    b = tile('camB'),
    c = tile('camC');
  await settle();
  boot();
  expect(request().stream).toBe(1);
  expect(wire().requests).toHaveLength(2); // streams, then exactly one frame
  displayed(offered());
  await settle(249);
  expect(wire().requests).toHaveLength(2);
  await settle(1);
  expect(request().stream).toBe(2);
  displayed(offered());
  await settle(250);
  expect(request().stream).toBe(3);
  displayed(offered());
  expect(wires).toHaveLength(1);
  expect(workers).toHaveLength(1);
  expect([a.drawImage.mock.calls.length, b.drawImage.mock.calls.length, c.drawImage.mock.calls.length]).toEqual([
    1, 1, 1,
  ]);
  expect(wire().requests.map(r => r.type)).toEqual(['streams', 'frame', 'frame', 'frame']);
  expect(workers[0]?.messages.filter(m => m.type === 'reset')).toHaveLength(3);
  await settle(2499);
  expect(wire().requests).toHaveLength(4);
  await settle(1);
  expect(request().stream).toBe(1);
});

it('waits for the decoder flush and preserves portrait geometry, frame metadata, and actual pixel capture identity', async () => {
  const a = tile('camA');
  await settle();
  boot();
  const media: MediaContext = {
    type: 'media',
    context_id: 9,
    stream: 1,
    chunk_id: 'synthetic',
    revision: 2,
    profile_status: 'absent',
    native_geometry: { width: 720, height: 1280 },
  };
  const at = offered(media);
  expect(a.drawImage).not.toHaveBeenCalled();
  expect(a.canvas.dataset.capturedAt).toBeUndefined();
  expect(workers[0]?.messages.map(m => m.type)).toEqual(['canvas', 'reset', 'playhead', 'media', 'message', 'capture']);
  const metadata = {
    capturedAtUs: at,
    width: 720,
    height: 1280,
    flip: true,
    profileStatus: 'absent' as const,
    nativeGeometry: media.native_geometry,
  };
  const bitmap = displayed(at, metadata);
  expect([a.canvas.width, a.canvas.height]).toEqual([720, 1280]);
  expect(a.canvas.dataset.capturedAt).toBe(String(at / 1e6));
  expect(JSON.parse(a.canvas.dataset.frameMetadata ?? 'null')).toEqual(metadata);
  expect(bitmap.close).toHaveBeenCalledOnce();
});

it('keeps old pixels and their timestamp visibly stale when a later poll has no picture', async () => {
  const a = tile('camA');
  await settle();
  boot();
  const at = offered();
  displayed(at);
  await settle(3000);
  const r = request();
  wire().handlers.onReply({ type: 'frame', stream: r.stream, at_us: r.at, capture_us: null });
  expect(a.drawImage).toHaveBeenCalledOnce();
  expect(a.canvas.dataset.capturedAt).toBe(String(at / 1e6));
  expect(a.notify).toHaveBeenLastCalledWith({
    status: 'stale',
    capturedAt: at / 1e6,
    error: 'No recent preview picture is available.',
  });
  expect(wire().close).toHaveBeenCalledOnce();
});

it('invalidates the displayed identity when a replacement draw clears the canvas then fails', async () => {
  const a = tile('camA');
  await settle();
  boot();
  const at = offered();
  displayed(at, {
    capturedAtUs: at,
    width: 720,
    height: 1280,
    flip: false,
    profileStatus: 'absent',
    nativeGeometry: { width: 720, height: 1280 },
  });
  expect(a.canvas.dataset.capturedAt).toBe(String(at / 1e6));
  expect(a.canvas.dataset.frameMetadata).toBeDefined();

  await settle(3000);
  a.drawImage.mockImplementationOnce(() => {
    throw Error('synthetic canvas draw failure');
  });
  displayed(offered());

  expect(a.canvas.dataset.capturedAt).toBeUndefined();
  expect(a.canvas.dataset.frameMetadata).toBeUndefined();
  expect(a.notify.mock.calls.at(-1)?.[0]).toMatchObject({ status: 'unavailable', capturedAt: null });
});

it('invalidates every shared view when a later canvas fails after an earlier one already committed new pixels', async () => {
  const a = tile('camA');
  const b = tile('camA');
  await settle();
  boot();
  const at = offered();
  displayed(at, {
    capturedAtUs: at,
    width: 720,
    height: 1280,
    flip: false,
    profileStatus: 'absent',
    nativeGeometry: { width: 720, height: 1280 },
  });
  expect(a.canvas.dataset.capturedAt).toBe(String(at / 1e6));
  expect(b.canvas.dataset.capturedAt).toBe(String(at / 1e6));

  await settle(3000);
  b.drawImage.mockImplementationOnce(() => {
    throw Error('synthetic second-view draw failure');
  });
  displayed(offered());

  expect(a.drawImage).toHaveBeenCalledTimes(2);
  expect(b.drawImage).toHaveBeenCalledTimes(2);
  for (const view of [a, b]) {
    expect(view.canvas.dataset.capturedAt).toBeUndefined();
    expect(view.canvas.dataset.frameMetadata).toBeUndefined();
    expect(view.notify.mock.calls.at(-1)?.[0]).toMatchObject({ status: 'unavailable', capturedAt: null });
  }
});

it('counts handshake time in the total budget and moves to the next camera without a catch-up burst', async () => {
  const a = tile('camA');
  tile('camB');
  await settle();
  const old = wire();
  await settle(5000);
  expect(old.close).toHaveBeenCalledOnce();
  expect(a.notify.mock.calls.at(-1)?.[0]).toMatchObject({
    status: 'unavailable',
    error: expect.stringMatching(/timed out/),
  });
  await settle(249);
  expect(wires).toHaveLength(1);
  await settle(1);
  expect(wires).toHaveLength(2);
  boot();
  expect(request().stream).toBe(2);
  old.handlers.onBinary(packet(1, 99_000_000));
  expect(workers).toHaveLength(0);
});

it('pauses while hidden, closes active work and ignores late worker pixels after cancellation', async () => {
  const a = tile('camA');
  await settle();
  boot();
  const at = offered();
  const old = workers[0];
  if (!old) throw Error('No worker');
  const capture = old.messages.at(-1);
  if (capture?.type !== 'capture') throw Error('No capture');
  page.set(true);
  expect(old.terminate).toHaveBeenCalledOnce();
  const bitmap = { width: 1, height: 1, close: vi.fn() } as unknown as ImageBitmap;
  old.emit({ type: 'frame', bitmap, frameUs: at, epoch: capture.requestId });
  expect(bitmap.close).toHaveBeenCalledOnce();
  expect(a.drawImage).not.toHaveBeenCalled();
  await settle(30_000);
  expect(wires).toHaveLength(1);
  page.set(false);
  await settle();
  expect(wires).toHaveLength(2);
});

it('detaching the in-flight camera closes its generation before another tile can request', async () => {
  const a = tile('camA');
  tile('camB');
  await settle();
  boot();
  const old = wire();
  a.off();
  await settle(250);
  boot();
  expect(request().stream).toBe(2);
  old.handlers.onReply({ type: 'error', message: 'late old error' });
  expect(wire().close).not.toHaveBeenCalled();
});

it.each(['stream', 'capture', 'context', 'duplicate'])(
  'rejects a mismatched %s packet without painting it',
  async kind => {
    const a = tile('camA');
    await settle();
    boot();
    const r = request(),
      at = r.at - 1_000_000;
    const bytes = packet(kind === 'stream' ? 2 : 1, at, true, kind === 'context' ? 7 : undefined);
    wire().handlers.onBinary(bytes);
    if (kind === 'duplicate') wire().handlers.onBinary(bytes);
    wire().handlers.onReply({ type: 'frame', stream: 1, at_us: r.at, capture_us: kind === 'capture' ? at - 1 : at });
    expect(a.drawImage).not.toHaveBeenCalled();
    expect(wire().close).toHaveBeenCalledOnce();
    expect(a.notify.mock.calls.at(-1)?.[0]).toMatchObject({ status: 'unavailable' });
  },
);

it('reopens an idle closed connection instead of sending on its dead socket', async () => {
  tile('camA');
  await settle();
  boot();
  displayed(offered());
  const old = wire();
  old.handlers.onClose('closed at idle');
  await settle(3000);
  expect(wires).toHaveLength(2);
  boot();
  expect(request().stream).toBe(1);
});

it('makes cleanup idempotent when a camera is detached then reattached', async () => {
  const old = tile('camA');
  await settle();
  boot();
  old.off();
  const current = tile('camA');
  old.off();
  await settle(250);
  boot();
  displayed(offered());
  expect(current.drawImage).toHaveBeenCalledOnce();
});

it('discards a delayed bitmap epoch from the previous camera on a reused decoder', async () => {
  tile('camA');
  const b = tile('camB');
  await settle();
  boot();
  const at = offered();
  displayed(at);
  const w = workers[0];
  if (!w) throw Error('No worker');
  const old = w.messages.find(m => m.type === 'message');
  if (old?.type !== 'message') throw Error('No packet');
  await settle(250);
  const next = offered();
  const late = { width: 10, height: 10, close: vi.fn() } as unknown as ImageBitmap;
  w.emit({ type: 'frame', bitmap: late, frameUs: at, epoch: old.epoch });
  expect(late.close).toHaveBeenCalledOnce();
  expect(b.drawImage).not.toHaveBeenCalled();
  displayed(next);
  expect(b.drawImage).toHaveBeenCalledOnce();
});
