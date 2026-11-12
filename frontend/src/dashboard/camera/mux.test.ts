import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { encodeHeader, HEADER_LEN } from '../player/header';
import type { FromWorker, ToWorker } from '../player/worker';
import { controlVisibility, fakeBitmaps, fakeGateway, recordingCanvas } from './testFakes';

// mux.ts and gateway.ts keep their state at module level, so each test loads fresh copies.
let mux: typeof import('./mux');
let net: ReturnType<typeof fakeGateway>;
const settle = (ms = 0) => vi.advanceTimersByTimeAsync(ms);
const jpeg = (n: number) => new Uint8Array(n).fill(0xff);

beforeEach(async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
  vi.resetModules();
  const gw = await import('./gateway');
  net = fakeGateway();
  gw.setGatewayForTests(net.connect);
  mux = await import('./mux');
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('sessions', () => {
  it('watches each camera on screen once, however many canvases show it, and lets it go with its last canvas', async () => {
    const offs = [
      mux.attach('cam0', document.createElement('canvas')),
      mux.attach('cam1', document.createElement('canvas')),
      mux.attach('cam0', document.createElement('canvas')),
    ];
    await settle();
    expect(net.conns).toHaveLength(2);
    offs[0]?.();
    expect(net.open()).toHaveLength(2);
    offs[2]?.();
    expect(net.open()).toHaveLength(1);
    offs[1]?.();
    expect(net.open()).toHaveLength(0);
    offs[1]?.(); // a second cleanup is harmless
    await settle(30_000);
    expect(net.conns).toHaveLength(2); // nothing reconnects once no canvas is left
  });

  it("paints the camera's kept picture as soon as it's decoded, dimmed, until its live frames come", async () => {
    const thumbs = await import('./thumbs');
    const watch = await import('../player/watch');
    let deliver: ((f: import('../player/watch').WatchFrame) => void) | null = null;
    vi.spyOn(watch, 'watchStream').mockImplementation((_s, _fps, fn) => {
      deliver = fn;
      return () => {};
    });
    thumbs.setThumbDecodersForTests({
      jpeg: async () => ({ width: 4, height: 3, close: () => {} }) as unknown as ImageBitmap,
      video: async () => ({ width: 4, height: 3, close: () => {} }) as unknown as ImageBitmap,
    });
    const keep = thumbs.keepThumb('cam0');
    const head = encodeHeader({ streamId: 1, payloadType: 4, keyframe: true, live: true, watch: true, captureUs: 7e6 });
    (deliver as unknown as (f: import('../player/watch').WatchFrame) => void)({
      header: { streamId: 1, payloadType: 4, keyframe: true, live: true, watch: true, flip: false, captureUs: 7e6 },
      buffer: head.buffer as ArrayBuffer,
      media: null,
    });
    const view = recordingCanvas();
    const off = mux.attach('cam0', view.canvas);
    await vi.waitFor(() => expect(view.drawImage).toHaveBeenCalledOnce());
    expect([view.canvas.width, view.canvas.height, view.canvas.dataset.capturedAt]).toEqual([4, 3, '7']);
    expect(mux.useLiveNow.getState().cam0).toBe(false); // shown, dimmed: not yet live
    off();
    keep();
  });

  it('attaching nothing does nothing', async () => {
    mux.attach('cam0', null)();
    await settle();
    expect(net.conns).toHaveLength(0);
  });

  it('lets the cameras go while the page is hidden and watches them again when shown', async () => {
    const page = controlVisibility();
    try {
      const off = mux.attach('cam0', document.createElement('canvas'));
      await settle();
      page.set(true);
      expect(net.open()).toHaveLength(0);
      page.set(false);
      await settle();
      expect(net.open()).toHaveLength(1);
      expect(net.conns).toHaveLength(2);
      off();
    } finally {
      page.restore();
    }
  });
});

describe('frames', () => {
  it("draws each camera's frames on every canvas showing it, sized to the picture", async () => {
    const bitmaps = fakeBitmaps();
    const a = recordingCanvas();
    const b = recordingCanvas();
    const other = recordingCanvas();
    const offs = [mux.attach('cam0', a.canvas), mux.attach('cam0', b.canvas), mux.attach('cam1', other.canvas)];
    await settle();
    net.conns[0]?.hello();
    net.conns[0]?.streams({ 'cam0.video': 1 });
    net.conns[0]?.jpeg(1_000_000, jpeg(5));
    await settle();
    expect(bitmaps.sizes).toEqual([5]);
    expect(a.drawImage).toHaveBeenCalledTimes(1);
    expect(b.drawImage).toHaveBeenCalledTimes(1);
    expect(other.drawImage).not.toHaveBeenCalled();
    expect([a.canvas.width, a.canvas.height]).toEqual([640, 360]);
    for (const off of offs) off();
  });

  it('turns a flipped camera’s frames the right way up and leaves upright frames alone (2026-11-03)', async () => {
    fakeBitmaps();
    const { canvas, drawImage, translate, scale } = recordingCanvas();
    const off = mux.attach('cam0', canvas);
    await settle();
    const c = net.ready('cam0', 1);
    c.jpeg(1_000_000, jpeg(5), true);
    await settle();
    expect(drawImage).toHaveBeenCalledTimes(1);
    expect(translate).toHaveBeenCalledWith(640, 360);
    expect(scale).toHaveBeenCalledWith(-1, -1);
    c.jpeg(2_000_000, jpeg(5));
    await settle();
    expect(drawImage).toHaveBeenCalledTimes(2);
    expect(translate).toHaveBeenCalledTimes(1);
    expect(scale).toHaveBeenCalledTimes(1);
    off();
  });

  it('keeps only the newest frame while one is decoding', async () => {
    const bitmaps = fakeBitmaps({ hold: true });
    const { canvas, drawImage } = recordingCanvas();
    const off = mux.attach('cam0', canvas);
    await settle();
    const c = net.ready('cam0', 1);
    c.jpeg(1_000_000, jpeg(1));
    c.jpeg(2_000_000, jpeg(2));
    c.jpeg(3_000_000, jpeg(3));
    await settle();
    expect(bitmaps.sizes).toEqual([1]);
    expect(canvas.dataset.capturedAt).toBeUndefined();
    expect(drawImage).not.toHaveBeenCalled();
    bitmaps.release();
    await settle();
    expect(bitmaps.sizes).toEqual([1, 3]); // the frame of 2 bytes was skipped
    expect(drawImage).toHaveBeenCalledTimes(1);
    expect(canvas.dataset.capturedAt).toBe('1'); // timestamp belongs to the displayed pixels, not pending frame 3
    bitmaps.release();
    await settle();
    expect(bitmaps.sizes).toEqual([1, 3]);
    expect(canvas.dataset.capturedAt).toBe('3');
    off();
  });

  it('does not mark a failed canvas draw as a captured frame', async () => {
    fakeBitmaps();
    const { canvas, drawImage } = recordingCanvas();
    drawImage.mockImplementation(() => {
      throw new Error('canvas draw failed');
    });
    const off = mux.attach('cam0', canvas);
    await settle();
    net.ready('cam0', 1).jpeg(5_000_000, jpeg(1));
    await settle();
    expect(drawImage).toHaveBeenCalledTimes(1);
    expect(canvas.dataset.capturedAt).toBeUndefined();
    off();
  });

  it.each(['cam0', '__proto__', 'constructor', 'toString'])('tracks %s live only while frames arrive', async cam => {
    fakeBitmaps();
    const off = mux.attach(cam, document.createElement('canvas'));
    await settle();
    expect(mux.useLiveNow.getState()[cam]).toBeUndefined();
    net.ready(cam, 1).jpeg(1, jpeg(4));
    await settle();
    expect(mux.useLiveNow.getState()[cam]).toBe(true);
    await settle(3900);
    expect(mux.useLiveNow.getState()[cam]).toBe(true);
    await settle(1100);
    expect(mux.useLiveNow.getState()[cam]).toBe(false);
    off();
  });
});

function fakeVideoWorkers() {
  class DecodeWorker {
    static all: DecodeWorker[] = [];
    onmessage: ((event: MessageEvent<FromWorker>) => void) | null = null;
    messages: { message: ToWorker; transfer?: Transferable[] }[] = [];
    terminate = vi.fn();
    constructor() {
      DecodeWorker.all.push(this);
    }
    postMessage(message: ToWorker, transfer?: Transferable[]) {
      this.messages.push({ message, transfer });
    }
    emit(data: FromWorker) {
      this.onmessage?.({ data } as MessageEvent<FromWorker>);
    }
  }
  vi.stubGlobal('Worker', DecodeWorker);
  vi.stubGlobal(
    'OffscreenCanvas',
    class {
      width: number;
      height: number;
      constructor(width: number, height: number) {
        this.width = width;
        this.height = height;
      }
    },
  );
  return DecodeWorker.all;
}

function encoded(captureUs: number, flip = false): ArrayBuffer {
  const bytes = new Uint8Array(HEADER_LEN + 5);
  bytes.set(encodeHeader({ streamId: 1, payloadType: 1, keyframe: true, live: true, flip, captureUs }));
  bytes.set([0, 0, 1, 0x67, 0x42], HEADER_LEN);
  return bytes.buffer;
}

it('decodes RTSP access units once per camera and shares upright portrait pixels with their capture identity', async () => {
  const workers = fakeVideoWorkers();
  const a = recordingCanvas(),
    b = recordingCanvas();
  const offs = [mux.attach('cam0', a.canvas), mux.attach('cam0', b.canvas)];
  await settle();
  const c = net.ready('cam0', 1);
  const packet = encoded(5_000_000, true);
  c.handlers.onBinary(packet);
  expect(workers).toHaveLength(1);
  const worker = workers[0];
  if (!worker) throw Error('No worker');
  const sent = worker.messages.find(x => x.message.type === 'message');
  expect(sent?.message.type).toBe('message');
  if (sent?.message.type !== 'message') throw Error('No encoded packet reached the decoder');
  expect(sent.message.buffer).not.toBe(packet);
  expect(new Uint8Array(sent.message.buffer)).toEqual(new Uint8Array(packet));
  expect(sent.transfer).toEqual([sent.message.buffer]);
  const bitmap = { width: 720, height: 1280, close: vi.fn() } as unknown as ImageBitmap;
  const metadata = {
    capturedAtUs: 5_000_000,
    width: 720,
    height: 1280,
    flip: true,
    profileStatus: 'unsupported' as const,
  };
  worker.emit({ type: 'frame', bitmap, frameUs: 5_000_000, metadata });
  expect([a.canvas.width, a.canvas.height]).toEqual([720, 1280]);
  expect(a.canvas.dataset.capturedAt).toBe('5');
  expect(JSON.parse(a.canvas.dataset.frameMetadata ?? 'null')).toEqual({
    ...metadata,
    profileStatus: 'absent',
    nativeGeometry: { width: 720, height: 1280 },
  });
  expect(b.drawImage).toHaveBeenCalledTimes(1);
  expect(bitmap.close).toHaveBeenCalledTimes(1);
  offs[0]?.();
  expect(worker.terminate).not.toHaveBeenCalled();
  offs[1]?.();
  expect(worker.terminate).toHaveBeenCalledTimes(1);
});

it('asks the video decoder for small copies only while something taps the camera, and hands each one on', async () => {
  const workers = fakeVideoWorkers();
  const { canvas } = recordingCanvas();
  const off = mux.attach('cam0', canvas);
  await settle();
  const c = net.ready('cam0', 1);
  c.handlers.onBinary(encoded(1_000_000));
  const worker = workers[0];
  if (!worker) throw Error('No worker');
  const asked = () => worker.messages.flatMap(x => (x.message.type === 'small' ? [x.message.on] : []));
  // Nothing taps the camera: the worker is told not to make them
  expect(asked()).toEqual([false]);
  const picture = () => ({ width: 1920, height: 1080, close: vi.fn() }) as unknown as ImageBitmap;
  const seen: (ImageBitmap | undefined)[] = [];
  const stop = mux.tapFrames('cam0', (_bitmap, _at, _flip, small) => seen.push(small));
  expect(asked()).toEqual([false, true]);
  const bitmap = picture();
  const small = picture();
  worker.emit({ type: 'frame', bitmap, small, frameUs: 1_000_000 });
  expect(seen).toEqual([small]);
  // Both are closed once the frame has been drawn and handed on
  expect(bitmap.close).toHaveBeenCalledTimes(1);
  expect(small.close).toHaveBeenCalledTimes(1);
  stop();
  expect(asked()).toEqual([false, true, false]);
  off();
});

it('closes encoded decoders on hide and reconnect, discards late bitmaps and never lets an old connection kill its replacement', async () => {
  const workers = fakeVideoWorkers();
  const page = controlVisibility();
  const { canvas, drawImage } = recordingCanvas();
  const off = mux.attach('cam0', canvas);
  try {
    await settle();
    const first = net.ready('cam0', 1);
    first.handlers.onBinary(encoded(1_000_000));
    const w = workers[0];
    if (!w) throw Error('No initial decoder');
    page.set(true);
    expect(w.terminate).toHaveBeenCalledTimes(1);
    const late = { width: 640, height: 360, close: vi.fn() } as unknown as ImageBitmap;
    w.emit({ type: 'frame', bitmap: late, frameUs: 1_000_000 });
    expect(late.close).toHaveBeenCalledOnce();
    expect(drawImage).not.toHaveBeenCalled();
    page.set(false);
    await settle();
    const current = net.ready('cam0', 1);
    current.handlers.onBinary(encoded(2_000_000));
    expect(workers).toHaveLength(2);
    first.fail();
    expect(current.closed).toBe(false);
    current.fail();
    expect(workers[1]?.terminate).toHaveBeenCalledOnce();
    await settle(1000);
    net.ready('cam0', 1).handlers.onBinary(encoded(3_000_000));
    expect(workers).toHaveLength(3);
  } finally {
    off();
    page.restore();
  }
});

it('makes codec errors explicit and clears them only after a successful displayed frame', async () => {
  const workers = fakeVideoWorkers();
  const { canvas } = recordingCanvas();
  const off = mux.attach('cam0', canvas);
  await settle();
  net.ready('cam0', 1).handlers.onBinary(encoded(1_000_000));
  const w = workers[0];
  if (!w) throw Error('No decoder');
  w.emit({ type: 'codec', codec: 'avc', supported: false });
  expect(mux.useLiveError.getState().cam0).toMatch(/cannot decode.*AVC/);
  expect(canvas.dataset.capturedAt).toBeUndefined();
  const bitmap = { width: 640, height: 360, close: vi.fn() } as unknown as ImageBitmap;
  w.emit({ type: 'frame', bitmap, frameUs: 2_000_000 });
  expect(mux.useLiveError.getState().cam0).toBeUndefined();
  expect(canvas.dataset.capturedAt).toBe('2');
  off();
});

it('does not draw a JPEG decode from a detached attachment onto a replacement canvas', async () => {
  const bitmaps = fakeBitmaps({ hold: true });
  const old = recordingCanvas();
  const off = mux.attach('cam0', old.canvas);
  await settle();
  net.ready('cam0', 1).jpeg(1_000_000, jpeg(3));
  off();
  const fresh = recordingCanvas();
  const stop = mux.attach('cam0', fresh.canvas);
  await settle();
  bitmaps.release();
  await settle();
  expect(fresh.drawImage).not.toHaveBeenCalled();
  expect(fresh.canvas.dataset.capturedAt).toBeUndefined();
  stop();
});

it('discards JPEG pixels from a lost connection even when its delayed decode finishes after reconnect', async () => {
  const bitmaps = fakeBitmaps({ hold: true });
  const { canvas, drawImage } = recordingCanvas();
  const off = mux.attach('cam0', canvas);
  await settle();
  const old = net.ready('cam0', 1);
  old.jpeg(1_000_000, jpeg(1));
  old.fail();
  await settle(1000);
  net.ready('cam0', 1).jpeg(2_000_000, jpeg(2));
  bitmaps.release();
  await settle();
  expect(drawImage).not.toHaveBeenCalled();
  bitmaps.release();
  await settle();
  expect(canvas.dataset.capturedAt).toBe('2');
  off();
});
