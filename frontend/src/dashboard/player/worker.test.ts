import * as flatbuffers from 'flatbuffers';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Box, Overlay as FbOverlay, Joint, Skeleton } from './generated/continuum/overlay.js';
import { encodeHeader, HEADER_LEN, PayloadType } from './header';
import type { Overlay } from './overlays';
import { draw, type FromWorker, Pipeline, sniffCodec } from './worker';

/** SVT-AV1's first temporal unit for a 720p camera: delimiter, sequence header (level 3.1), then a frame. */
const AV1_KEY = new Uint8Array([
  0x12, 0x00, 0x0a, 0x0f, 0x02, 0x00, 0x00, 0x2d, 0x6a, 0x67, 0xfd, 0x9e, 0x35, 0x7c, 0xc8, 0x08, 0x08, 0x0c, 0x20,
  0x32, 0x02, 0x10, 0x00,
]);

/** jsdom has no ImageBitmap or ImageDecoder: a picture is a closable box, and the decoder returns one per call. */
class FakeBitmap {
  closed = false;
  width = 4;
  height = 3;
  close() {
    this.closed = true;
  }
}
class FakeImageDecoder {
  static made: FakeBitmap[] = [];
  async decode() {
    const image = new FakeBitmap();
    FakeImageDecoder.made.push(image);
    return { image };
  }
  close() {}
}

function jpegMessage(captureUs: number, flip = false, contextId?: number): ArrayBuffer {
  const header = encodeHeader({
    streamId: 1,
    payloadType: PayloadType.jpeg,
    keyframe: true,
    live: false,
    captureUs,
    flip,
    contextId,
  });
  const out = new Uint8Array(header.length + 2);
  out.set(header);
  out.set([0xff, 0xd8], header.length);
  return out.buffer;
}

function fakeContext() {
  const calls: string[] = [];
  const ctx = new Proxy({} as Record<string, unknown>, {
    get: (_t, name: string) => {
      if (name === 'lineWidth' || name === 'font' || name === 'strokeStyle' || name === 'fillStyle') return 1;
      return (...args: unknown[]) => {
        calls.push(`${name}(${args.map(a => (typeof a === 'number' ? Math.round(a) : String(a))).join(',')})`);
      };
    },
    set: () => true,
  });
  return { ctx: ctx as unknown as OffscreenCanvasRenderingContext2D, calls };
}

class MarkedBitmap extends FakeBitmap {
  override width = 2;
  override height = 2;
  pixels = [1, 2, 3, 4]; // distinct corners: top-left, top-right, bottom-left, bottom-right
}

/** A 2x2 raster implementing the Canvas transform operations, so expected corner pixels are hand-checked. */
function markedCanvas() {
  type Matrix = [number, number, number, number, number, number];
  let matrix: Matrix = [1, 0, 0, 1, 0, 0];
  const stack: Matrix[] = [];
  let pixels = [0, 0, 0, 0];
  const order: string[] = [];
  const strokes: { bounds: number[]; matrix: number[] }[] = [];
  const ctx = {
    save() {
      stack.push([...matrix]);
    },
    restore() {
      matrix = stack.pop() ?? matrix;
      order.push('restore');
    },
    translate(x: number, y: number) {
      matrix[4] += matrix[0] * x + matrix[2] * y;
      matrix[5] += matrix[1] * x + matrix[3] * y;
    },
    scale(x: number, y: number) {
      matrix[0] *= x;
      matrix[1] *= x;
      matrix[2] *= y;
      matrix[3] *= y;
    },
    rotate(angle: number) {
      const [a, b, c, d, e, f] = matrix;
      const cos = Math.cos(angle),
        sin = Math.sin(angle);
      matrix = [a * cos + c * sin, b * cos + d * sin, -a * sin + c * cos, -b * sin + d * cos, e, f];
    },
    drawImage(picture: { pixels: number[] }) {
      const next = [0, 0, 0, 0];
      for (let y = 0; y < 2; y++)
        for (let x = 0; x < 2; x++) {
          const dx = Math.floor(matrix[0] * (x + 0.5) + matrix[2] * (y + 0.5) + matrix[4]);
          const dy = Math.floor(matrix[1] * (x + 0.5) + matrix[3] * (y + 0.5) + matrix[5]);
          next[dy * 2 + dx] = picture.pixels[y * 2 + x] ?? 0;
        }
      pixels = next;
      order.push('pixels');
    },
    strokeRect(...bounds: number[]) {
      strokes.push({ bounds, matrix: [...matrix] });
      order.push('overlay');
    },
  };
  const canvas = {
    width: 2,
    height: 2,
    getContext: () => ctx,
    transferToImageBitmap() {
      const bitmap = new MarkedBitmap();
      bitmap.pixels = [...pixels];
      return bitmap;
    },
  } as unknown as OffscreenCanvas;
  return { canvas, order, strokes, pixels: () => pixels };
}

function boxMessage(captureUs: number): ArrayBuffer {
  const builder = new flatbuffers.Builder(128);
  const box = Box.createBox(builder, 0, 0, 0.5, 0.5, 1, 0, 120, 0, 0);
  const boxes = FbOverlay.createBoxesVector(builder, [box]);
  FbOverlay.startOverlay(builder);
  FbOverlay.addCaptureUs(builder, BigInt(captureUs));
  FbOverlay.addBoxes(builder, boxes);
  builder.finish(FbOverlay.endOverlay(builder));
  const payload = builder.asUint8Array();
  const buffer = new Uint8Array(HEADER_LEN + payload.length);
  buffer.set(encodeHeader({ streamId: 1, payloadType: PayloadType.overlay, keyframe: false, live: false, captureUs }));
  buffer.set(payload, HEADER_LEN);
  return buffer.buffer;
}

function poseMessage(captureUs: number, people: number): ArrayBuffer {
  const builder = new flatbuffers.Builder(512);
  const layout = builder.createString('line');
  const skeletons = Array.from({ length: people }, (_, i) => {
    const x = (i + 1) / 4;
    const box = Box.createBox(builder, x, 0.1, x + 0.1, 0.9, 1, 0, 0, 0, 0);
    Skeleton.startJointsVector(builder, 2);
    Joint.createJoint(builder, x, 0.6, 0.9);
    Joint.createJoint(builder, x, 0.2, 0.9);
    const joints = builder.endVector();
    return Skeleton.createSkeleton(builder, box, layout, joints);
  });
  const vector = FbOverlay.createSkeletonsVector(builder, skeletons);
  FbOverlay.startOverlay(builder);
  FbOverlay.addCaptureUs(builder, BigInt(captureUs));
  FbOverlay.addSkeletons(builder, vector);
  builder.finish(FbOverlay.endOverlay(builder));
  const payload = builder.asUint8Array();
  const out = new Uint8Array(HEADER_LEN + payload.length);
  out.set(encodeHeader({ streamId: 1, payloadType: PayloadType.overlay, keyframe: false, live: true, captureUs }));
  out.set(payload, HEADER_LEN);
  return out.buffer;
}

/** Parameter headers are sufficient for the mocked decoder; these fixtures do not claim full SPS decoding. */
function avcKey(profile: number, compatibility: number, level: number, tail = 0x80): Uint8Array {
  return new Uint8Array([
    0,
    0,
    1,
    0x09,
    0xf0,
    0,
    0,
    0,
    1,
    0x67,
    profile,
    compatibility,
    level,
    tail,
    0,
    0,
    1,
    0x68,
    0xce,
    0,
    0,
    1,
    0x65,
    0x88,
  ]);
}
function videoBytes(captureUs: number, payload: Uint8Array, keyframe = true): ArrayBuffer {
  const buffer = new Uint8Array(HEADER_LEN + payload.length);
  buffer.set(encodeHeader({ streamId: 1, payloadType: PayloadType.video, keyframe, live: false, captureUs }));
  buffer.set(payload, HEADER_LEN);
  return buffer.buffer;
}

describe('sniffCodec', () => {
  it('tells the codec from the parameter sets', () => {
    expect(sniffCodec(new Uint8Array([0, 0, 0, 1, 0x67, 1, 0, 0, 1, 0x65]))).toBe('avc');
    expect(sniffCodec(new Uint8Array([0, 0, 1, 0x40, 0x01, 0, 0, 1, 0x42, 0x01]))).toBe('hevc');
    expect(sniffCodec(new Uint8Array([0, 0, 1, 0x41, 5]))).toBeNull();
    // An archived AV1 keyframe (#1479): temporal delimiter, then the sequence header; a delta unit has none.
    expect(sniffCodec(AV1_KEY)).toBe('av1');
    expect(sniffCodec(new Uint8Array([0x12, 0, 0x32, 1, 0]))).toBeNull();
  });
});

describe('draw', () => {
  it('draws boxes, the halpe26 bones only between seen joints, and captions', () => {
    const { ctx, calls } = fakeContext();
    const joints: [number, number, number][] = Array.from(
      { length: 26 },
      () => [-1, -1, 0] as [number, number, number],
    );
    joints[5] = [0.2, 0.2, 0.9]; // left shoulder
    joints[7] = [0.3, 0.4, 0.9]; // left elbow
    joints[9] = [0.1, 0.1, 0.1]; // left wrist, below the confidence floor
    const o: Overlay = {
      captureUs: 1,
      boxes: [{ xyxy: [0.1, 0.1, 0.5, 0.5], score: 1, label: 'cat', hue: 30 }],
      skeletons: [{ box: { xyxy: [0, 0, 1, 1], score: 1, label: '', hue: null }, layout: 'halpe26', joints }],
      captions: [{ text: 'hi', speaker: 'Joe' }],
    };
    draw(ctx, o, 100, 50);
    expect(calls).toContain('strokeRect(10,5,40,20)');
    expect(calls).toContain('fillText(cat,14,1)');
    expect(calls.filter(c => c.startsWith('lineTo'))).toEqual(['lineTo(30,20)']); // shoulder to elbow only
    expect(calls.filter(c => c.startsWith('arc('))).toHaveLength(2);
    expect(calls.at(-1)).toMatch(/fillText\(Joe: hi/);
  });
});

describe('Pipeline', () => {
  it('retains one newest picture in live mode even when decoded capture times arrive out of order', async () => {
    const posted: FromWorker[] = [];
    const p = new Pipeline(m => posted.push(m));
    const { ctx } = fakeContext();
    p.setCanvas({ width: 4, height: 3, getContext: () => ctx } as unknown as OffscreenCanvas);
    p.show('live');
    await p.take(jpegMessage(1_000_000));
    await p.take(jpegMessage(3_000_000));
    await p.take(jpegMessage(2_000_000));
    expect(posted.filter(m => m.type === 'stats').at(-1)).toMatchObject({ buffered: 1 });
    expect(posted.filter(m => m.type === 'drawn').at(-1)).toMatchObject({ frameUs: 3_000_000 });
    expect(FakeImageDecoder.made.map(b => b.closed)).toEqual([true, false, true]);
    p.reset();
  });

  it.each([2, 3])('draws all %i people when the rest of their poses arrive after the frame', async people => {
    const p = new Pipeline(() => {});
    const { ctx, calls } = fakeContext();
    p.setCanvas({ getContext: () => ctx, width: 4, height: 3 } as unknown as OffscreenCanvas);
    await p.take(jpegMessage(1_000_000));
    await p.take(poseMessage(1_000_000, 1));
    p.show('live');
    calls.length = 0;
    await p.take(poseMessage(1_000_000, people));
    expect(calls.filter(c => c.startsWith('lineTo'))).toHaveLength(people);
    expect(calls.filter(c => c.startsWith('arc'))).toHaveLength(people * 2);
  });
  it('redraws a displayed frame when its late inference overlay arrives', async () => {
    const posted: FromWorker[] = [];
    const p = new Pipeline(m => posted.push(m));
    const { ctx, calls } = fakeContext();
    p.setCanvas({ getContext: () => ctx, width: 4, height: 3 } as unknown as OffscreenCanvas);
    await p.take(jpegMessage(1_000_000));
    p.show('live');
    await p.take(boxMessage(1_000_000));
    expect(calls.filter(c => c.startsWith('strokeRect'))).toHaveLength(1);
    expect(posted.filter(m => m.type === 'drawn').at(-1)).toMatchObject({
      frameUs: 1_000_000,
      overlay: 'exact',
    });
  });
  const posted: FromWorker[] = [];
  beforeEach(() => {
    posted.length = 0;
    FakeImageDecoder.made = [];
    vi.stubGlobal('ImageDecoder', FakeImageDecoder);
    vi.stubGlobal('ImageBitmap', FakeBitmap);
  });
  afterEach(() => vi.unstubAllGlobals());

  it('decodes messages one after another and keeps what is ahead of the playhead', async () => {
    const p = new Pipeline(m => posted.push(m));
    const { ctx, calls } = fakeContext();
    const canvas = { width: 0, height: 0, getContext: () => ctx } as unknown as OffscreenCanvas;
    p.setCanvas(canvas);
    // A burst of 700 frames at 33 ms: more than the buffer, all ahead of a playhead that hasn't started.
    for (let i = 0; i < 700; i++) void p.take(jpegMessage(1_000_000 + i * 33_000));
    await p.take(jpegMessage(1_000_000 + 700 * 33_000));
    const stats = posted.filter(m => m.type === 'stats').at(-1);
    expect(stats).toMatchObject({ type: 'stats', decoded: 701, buffered: 600 });
    // The oldest frames went; the newest are there.
    const closed = FakeImageDecoder.made.filter(b => b.closed).length;
    expect(closed).toBe(101);
    expect(FakeImageDecoder.made[0]?.closed).toBe(true);
    expect(FakeImageDecoder.made[700]?.closed).toBe(false);

    // Show the frame at the playhead and report the overlay pick.
    p.show(1_000_000 + 400 * 33_000 + 10);
    expect(calls.some(c => c.startsWith('drawImage'))).toBe(true);
    const drawn = posted.filter(m => m.type === 'drawn').at(-1);
    expect(drawn).toMatchObject({ type: 'drawn', frameUs: 1_000_000 + 400 * 33_000, overlay: 'none', matched: false });
    p.show(1_000_000 + 400 * 33_000 + 10);
    expect(posted.filter(m => m.type === 'drawn')).toHaveLength(1); // the same frame is not drawn twice

    // Frames behind the playhead beyond the backlog are evicted as more arrive.
    await p.take(jpegMessage(1_000_000 + 701 * 33_000));
    const after = posted.filter(m => m.type === 'stats').at(-1);
    // 601 buffered plus one new, minus the 271 behind the playhead (the shown frame among them) beyond the
    // 30-frame backlog.
    expect(after).toMatchObject({ type: 'stats', buffered: 331 });
    p.reset();
    expect(FakeImageDecoder.made.every(b => b.closed)).toBe(true);
  });

  it('draws the new window after a seek back past the picture on screen (#1997)', async () => {
    const p = new Pipeline(m => posted.push(m));
    const { ctx } = fakeContext();
    p.setCanvas({ width: 0, height: 0, getContext: () => ctx } as unknown as OffscreenCanvas);
    // Playing at 60 s: its frame is on screen.
    p.show(60_000_000);
    await p.take(jpegMessage(59_990_000));
    // Seek back to 10 s; its window arrives (frames from 9.9 s, 33 ms apart, ending past the playhead).
    p.show(10_000_000);
    for (let i = 0; i < 99; i++) void p.take(jpegMessage(9_900_000 + i * 33_000));
    await p.take(jpegMessage(9_900_000 + 99 * 33_000));
    const drawn = posted.filter(m => m.type === 'drawn').map(m => m.type === 'drawn' && m.frameUs);
    expect(drawn.at(-1)).toBe(9_999_000); // the newest at or before 10 s, not the old 60 s picture
    p.show(11_000_000); // and it plays on through the window
    expect(posted.filter(m => m.type === 'drawn').at(-1)).toMatchObject({ frameUs: 9_900_000 + 33 * 33_000 });
  });

  it('keeps the window at the playhead when stale frames ahead fill the buffer (#1997)', async () => {
    const p = new Pipeline(m => posted.push(m));
    const { ctx } = fakeContext();
    p.setCanvas({ width: 0, height: 0, getContext: () => ctx } as unknown as OffscreenCanvas);
    // Earlier positions left 590 frames from 100 s on, all ahead of the playhead once it seeks back to 10 s.
    p.show(99_000_000);
    for (let i = 0; i < 590; i++) void p.take(jpegMessage(100_000_000 + i * 33_000));
    p.show(10_000_000);
    for (let i = 0; i < 99; i++) void p.take(jpegMessage(9_900_000 + i * 33_000));
    await p.take(jpegMessage(9_900_000 + 99 * 33_000));
    expect(posted.filter(m => m.type === 'stats').at(-1)).toMatchObject({ buffered: 600 });
    expect(posted.filter(m => m.type === 'drawn').at(-1)).toMatchObject({ frameUs: 9_999_000 });
    p.show(11_000_000); // the whole new window stayed
    expect(posted.filter(m => m.type === 'drawn').at(-1)).toMatchObject({ frameUs: 9_900_000 + 33 * 33_000 });
  });

  it('draws a frame that lands after the playhead was set, as a paused seek needs', async () => {
    const p = new Pipeline(m => posted.push(m));
    const { ctx, calls } = fakeContext();
    p.setCanvas({ width: 0, height: 0, getContext: () => ctx } as unknown as OffscreenCanvas);
    p.show(2_000_000); // the seek: nothing there yet
    expect(calls.some(c => c.startsWith('drawImage'))).toBe(false);
    await p.take(jpegMessage(1_990_000)); // the window arrives
    expect(posted.filter(m => m.type === 'drawn').map(m => m.type === 'drawn' && m.frameUs)).toEqual([1_990_000]);
    await p.take(jpegMessage(2_500_000)); // after the playhead: not drawn yet
    expect(posted.filter(m => m.type === 'drawn')).toHaveLength(1);
  });

  it('returns a rendered bitmap when the DOM canvas stays owned by the page', async () => {
    const p = new Pipeline(m => posted.push(m));
    const { ctx } = fakeContext();
    const bitmap = new FakeBitmap();
    const transfer = vi.fn(() => bitmap);
    p.setCanvas(
      { width: 0, height: 0, getContext: () => ctx, transferToImageBitmap: transfer } as unknown as OffscreenCanvas,
      true,
    );
    p.show(1_000_000);
    await p.take(jpegMessage(1_000_000));
    expect(transfer).toHaveBeenCalledOnce();
    expect(posted.find(m => m.type === 'frame')).toMatchObject({
      type: 'frame',
      bitmap,
      frameUs: 1_000_000,
      metadata: { capturedAtUs: 1_000_000, profileStatus: 'unsupported', width: 4, height: 3 },
    });
    p.reset();
  });

  function bufferedDecoder(marked = false) {
    const decoded: number[] = [];
    const instances: VideoDecoderInit[] = [];
    const configs: VideoDecoderConfig[] = [];
    const emit = (instance: number, timestamp: number) => {
      const frame = {
        timestamp,
        displayWidth: marked ? 2 : 1920,
        displayHeight: marked ? 2 : 1080,
        pixels: [1, 2, 3, 4],
        close: vi.fn(),
      };
      instances[instance]?.output(frame as unknown as VideoFrame);
      return frame;
    };
    const flush = vi.fn(async () => {
      decoded.forEach(timestamp => {
        emit(instances.length - 1, timestamp);
      });
    });
    vi.stubGlobal(
      'VideoDecoder',
      class {
        static async isConfigSupported(config: VideoDecoderConfig) {
          return { supported: true, config };
        }
        state = 'configured';
        constructor(init: VideoDecoderInit) {
          instances.push(init);
        }
        configure(config: VideoDecoderConfig) {
          configs.push(config);
        }
        decode(chunk: EncodedVideoChunk) {
          decoded.push(chunk.timestamp);
        }
        flush = flush;
        close() {
          this.state = 'closed';
        }
      },
    );
    vi.stubGlobal(
      'EncodedVideoChunk',
      class {
        timestamp: number;
        constructor(init: EncodedVideoChunkInit) {
          this.timestamp = init.timestamp;
        }
      },
    );
    const video = (captureUs: number, keyframe = false, flip = false) => {
      const payload = keyframe ? avcKey(100, 0, 40) : new Uint8Array([0, 0, 1, 0x41, 1, 0]);
      const buffer = new Uint8Array(videoBytes(captureUs, payload, keyframe));
      buffer.set(encodeHeader({ streamId: 1, payloadType: PayloadType.video, keyframe, live: false, captureUs, flip }));
      return buffer.buffer;
    };
    return {
      decoded,
      configs,
      flush,
      video,
      emit,
      fail: (instance: number) => instances[instance]?.error(new DOMException('decode failed')),
    };
  }

  it.each([
    [66, 0xe0, 31, 'avc1.42e01f'],
    [77, 0x40, 40, 'avc1.4d4028'],
    [100, 0, 51, 'avc1.640033'],
  ])(
    'configures AVC from the actual SPS profile %s compatibility %s level %s',
    async (profile, compatibility, level, codec) => {
      const decoder = bufferedDecoder();
      const p = new Pipeline(message => posted.push(message));
      await p.take(videoBytes(1_000_000, avcKey(Number(profile), Number(compatibility), Number(level))));
      expect(decoder.configs).toEqual([expect.objectContaining({ codec, avc: { format: 'annexb' } })]);
      expect(decoder.decoded).toEqual([1_000_000]);
      p.reset();
    },
  );

  it('configures AV1 from the sequence header, without Annex B, and switches codecs at a keyframe', async () => {
    const decoder = bufferedDecoder();
    const p = new Pipeline(message => posted.push(message));
    await p.take(videoBytes(1_000_000, avcKey(66, 0xe0, 31)));
    await p.take(videoBytes(2_000_000, AV1_KEY)); // the archived chunk after an H.264 one
    await p.take(videoBytes(3_000_000, new Uint8Array([0x12, 0, 0x32, 1, 0]), false));
    expect(decoder.configs).toEqual([
      expect.objectContaining({ codec: 'avc1.42e01f', avc: { format: 'annexb' } }),
      { codec: 'av01.0.05M.08', optimizeForLatency: true },
    ]);
    expect(decoder.flush).toHaveBeenCalledOnce();
    expect(decoder.decoded).toEqual([1_000_000, 2_000_000, 3_000_000]);
    p.reset();
  });

  it('flushes and reconfigures changed AVC parameter sets even when the profile is unchanged', async () => {
    const decoder = bufferedDecoder();
    const p = new Pipeline(message => posted.push(message));
    await p.take(videoBytes(1_000_000, avcKey(100, 0, 40, 0x80)));
    await p.take(videoBytes(2_000_000, avcKey(100, 0, 40, 0x81)));
    expect(decoder.flush).toHaveBeenCalledOnce();
    expect(decoder.configs).toHaveLength(2);
    expect(decoder.decoded).toEqual([1_000_000, 2_000_000]);
    p.reset();
  });

  it.each([
    [0, 0, 1, 0x65, 0x88],
    [0, 0, 1, 0x67, 66, 0xe0],
    [0, 0, 1, 0x67, 66, 0xe0, 31, 0x80],
  ])('refuses an initial keyframe with missing or incomplete parameter sets (%j)', async (...bytes) => {
    const decoder = bufferedDecoder();
    const p = new Pipeline(message => posted.push(message));
    await p.take(videoBytes(1_000_000, new Uint8Array(bytes)));
    expect(decoder.configs).toEqual([]);
    expect(decoder.decoded).toEqual([]);
    expect(posted.some(message => message.type === 'error')).toBe(true);
    await p.take(videoBytes(2_000_000, avcKey(66, 0xe0, 31)));
    expect(decoder.configs).toEqual([expect.objectContaining({ codec: 'avc1.42e01f' })]);
    p.reset();
  });

  it('reuses complete AVC configuration for a bare IDR and resets it before a new source', async () => {
    const decoder = bufferedDecoder();
    const p = new Pipeline(message => posted.push(message));
    await p.take(videoBytes(1_000_000, avcKey(66, 0xe0, 31)));
    await p.take(videoBytes(2_000_000, new Uint8Array([0, 0, 1, 0x65, 0x88])));
    expect(decoder.configs).toHaveLength(1);
    expect(decoder.decoded).toEqual([1_000_000, 2_000_000]);
    p.reset();
    await p.take(videoBytes(3_000_000, new Uint8Array([0, 0, 1, 0x65, 0x88])));
    expect(decoder.configs).toHaveLength(1);
    expect(decoder.decoded).toEqual([1_000_000, 2_000_000]);
  });

  it('refuses malformed changed parameter sets then recovers at a complete keyframe', async () => {
    const decoder = bufferedDecoder();
    const p = new Pipeline(message => posted.push(message));
    await p.take(videoBytes(1_000_000, avcKey(66, 0xe0, 31)));
    await p.take(videoBytes(2_000_000, new Uint8Array([0, 0, 1, 0x67, 77])));
    await p.take(decoder.video(2_500_000));
    expect(decoder.decoded).toEqual([1_000_000]);
    await p.take(videoBytes(3_000_000, avcKey(77, 0x40, 40)));
    expect(decoder.configs.map(config => config.codec)).toEqual(['avc1.42e01f', 'avc1.4d4028']);
    expect(decoder.decoded).toEqual([1_000_000, 3_000_000]);
    p.reset();
  });

  it('queues capture behind access units, flushes delayed B frames, and redraws the newest frame before the playhead', async () => {
    const decoder = bufferedDecoder();
    const p = new Pipeline(m => posted.push(m));
    const { ctx } = fakeContext();
    p.setCanvas(
      {
        width: 0,
        height: 0,
        getContext: () => ctx,
        transferToImageBitmap: () => new FakeBitmap(),
      } as unknown as OffscreenCanvas,
      true,
    );
    p.show(10_000_000);
    void p.take(decoder.video(9_000_000, true));
    void p.take(decoder.video(10_020_000)); // future reference frame, decoded ahead of its preceding B frame
    void p.take(decoder.video(9_990_000));
    await p.capture(17, 10_000_000);
    expect(decoder.decoded).toEqual([9_000_000, 10_020_000, 9_990_000]);
    expect(decoder.flush).toHaveBeenCalledOnce();
    expect(posted.at(-1)).toEqual({ type: 'captured', requestId: 17, frameUs: 9_990_000 });
    expect(posted.at(-2)).toMatchObject({ type: 'drawn', frameUs: 9_990_000, overlay: 'none' });
    expect(posted.at(-3)).toMatchObject({ type: 'frame' });
    p.reset();
  });

  it('reports a capture flush failure instead of confirming an old picture', async () => {
    const decoder = bufferedDecoder();
    decoder.flush.mockRejectedValueOnce(new Error('incomplete GOP'));
    const p = new Pipeline(m => posted.push(m));
    void p.take(decoder.video(9_000_000, true));
    await p.capture(18, 10_000_000);
    expect(posted.at(-1)).toEqual({ type: 'error', message: 'incomplete GOP' });
    expect(posted.some(m => m.type === 'captured')).toBe(false);
    p.reset();
  });

  it('reports no frame when a completed capture has no picture before its playhead', async () => {
    const p = new Pipeline(m => posted.push(m));
    const { ctx } = fakeContext();
    p.setCanvas({ width: 0, height: 0, getContext: () => ctx } as unknown as OffscreenCanvas);
    void p.take(jpegMessage(11_000_000));
    await p.capture(19, 10_000_000);
    expect(posted.at(-1)).toEqual({ type: 'captured', requestId: 19, frameUs: null });
    p.reset();
  });

  it('rotates native JPEG corners 180 degrees before drawing upright overlay coordinates', async () => {
    vi.stubGlobal(
      'ImageDecoder',
      class {
        async decode() {
          return { image: new MarkedBitmap() };
        }
        close() {}
      },
    );
    const raster = markedCanvas();
    const p = new Pipeline(m => posted.push(m));
    p.setCanvas(raster.canvas, true);
    p.show(1_000_000);
    await p.take(boxMessage(1_000_000));
    await p.take(jpegMessage(1_000_000, true));
    expect(raster.pixels()).toEqual([4, 3, 2, 1]); // 180 degrees, not either single-axis mirror
    expect(raster.order).toEqual(['pixels', 'restore', 'overlay']);
    expect(raster.strokes).toEqual([{ bounds: [0, 0, 1, 1], matrix: [1, 0, 0, 1, 0, 0] }]);
    p.show('live');
    await p.take(jpegMessage(2_000_000));
    expect(raster.pixels()).toEqual([1, 2, 3, 4]);
    p.reset();
    await p.take(jpegMessage(1_000_000));
    p.show('live');
    expect(raster.pixels()).toEqual([1, 2, 3, 4]);
    p.reset();
  });

  it('rotates JPEG bitmap fallback pixels when ImageDecoder is unavailable', async () => {
    Reflect.deleteProperty(globalThis, 'ImageDecoder');
    vi.stubGlobal(
      'createImageBitmap',
      vi.fn(async () => new MarkedBitmap()),
    );
    const raster = markedCanvas();
    const p = new Pipeline(m => posted.push(m));
    p.setCanvas(raster.canvas);
    p.show(1_000_000);
    await p.take(jpegMessage(1_000_000, true));
    expect(raster.pixels()).toEqual([4, 3, 2, 1]);
    p.reset();
  });

  it('keeps orientation with reordered video timestamps through the final capture flush', async () => {
    const decoder = bufferedDecoder(true);
    const raster = markedCanvas();
    const p = new Pipeline(m => posted.push(m));
    p.setCanvas(raster.canvas, true);
    p.show(10_000_000);
    void p.take(decoder.video(9_000_000, true));
    void p.take(decoder.video(10_020_000));
    void p.take(decoder.video(9_990_000, false, true));
    void p.take(decoder.video(10_040_000));
    decoder.flush.mockImplementationOnce(async () => {
      decoder.emit(0, 10_020_000);
      decoder.emit(0, 9_990_000);
      decoder.emit(0, 9_000_000);
      decoder.emit(0, 10_040_000);
    });
    await p.capture(20, 10_000_000);
    expect(posted.at(-1)).toEqual({ type: 'captured', requestId: 20, frameUs: 9_990_000 });
    expect(posted.findLast(m => m.type === 'frame')).toMatchObject({ bitmap: { pixels: [4, 3, 2, 1] } });
    p.show('live');
    expect(raster.pixels()).toEqual([1, 2, 3, 4]);
    p.reset();
  });

  it('keeps orientation attached to retained JPEG pictures while old cache entries are evicted', async () => {
    const made: MarkedBitmap[] = [];
    vi.stubGlobal(
      'ImageDecoder',
      class {
        async decode() {
          const image = new MarkedBitmap();
          made.push(image);
          return { image };
        }
        close() {}
      },
    );
    const raster = markedCanvas();
    const p = new Pipeline(m => posted.push(m));
    p.setCanvas(raster.canvas);
    for (let i = 0; i < 601; i++) await p.take(jpegMessage(1_000_000 + i * 33_000, i === 500));
    expect(made[0]?.closed).toBe(true);
    p.show(17_500_000); // sample 500, still retained after sample 0 was evicted
    expect(raster.pixels()).toEqual([4, 3, 2, 1]);
    p.reset();
    expect(made.every(bitmap => bitmap.closed)).toBe(true);
  });

  it('projects gateway overlays onto confirmed native geometry and hides unknown geometry', async () => {
    vi.stubGlobal(
      'ImageDecoder',
      class {
        async decode() {
          const image = new FakeBitmap();
          image.width = 640;
          image.height = 360;
          return { image };
        }
        close() {}
      },
    );
    const { ctx, calls } = fakeContext();
    const p = new Pipeline(m => posted.push(m));
    p.setCanvas({ getContext: () => ctx, width: 640, height: 360 } as unknown as OffscreenCanvas);
    const media = {
      type: 'media' as const,
      context_id: 1,
      stream: 1,
      chunk_id: 'chunk',
      revision: 1,
      profile_status: 'absent' as const,
      native_geometry: { width: 640, height: 360 },
    };
    await p.take(boxMessage(1_000_000));
    await p.take(jpegMessage(1_000_000, true), media);
    p.show(1_000_000);
    expect(calls).toContain('strokeRect(0,0,320,180)');
    calls.length = 0;
    await p.take(jpegMessage(1_000_000, true), { ...media, profile_status: 'unavailable' });
    expect(calls.some(call => call.startsWith('strokeRect'))).toBe(false);
    p.reset();
  });

  it('shares one chunk context across queued pictures and fails closed after reset discards it', async () => {
    const p = new Pipeline(m => posted.push(m));
    const { ctx } = fakeContext();
    p.setCanvas(
      {
        getContext: () => ctx,
        width: 0,
        height: 0,
        transferToImageBitmap: () => new FakeBitmap(),
      } as unknown as OffscreenCanvas,
      true,
    );
    p.context({
      type: 'media',
      context_id: 1,
      stream: 1,
      chunk_id: 'chunk',
      revision: 1,
      profile_status: 'absent',
      native_geometry: { width: 4, height: 3 },
    });
    await p.take(jpegMessage(1_000_000, false, 1));
    await p.take(jpegMessage(2_000_000, false, 1));
    p.show(2_000_000);
    expect(posted.findLast(m => m.type === 'frame')).toMatchObject({
      metadata: { profileStatus: 'absent', nativeGeometry: { width: 4, height: 3 } },
    });
    p.reset();
    await p.take(jpegMessage(2_000_000, false, 1));
    expect(posted.at(-1)).toMatchObject({ type: 'error', message: 'Recorded frame context is unavailable.' });
  });

  it('flushes and switches codec when HEVC and H264 share a playback range', async () => {
    const decoder = bufferedDecoder();
    const p = new Pipeline(m => posted.push(m));
    const { ctx } = fakeContext();
    p.setCanvas({ getContext: () => ctx, width: 0, height: 0 } as unknown as OffscreenCanvas);
    const hevc = videoBytes(1_000_000, new Uint8Array([0, 0, 1, 0x40, 1, 0]));
    await p.take(hevc);
    expect(p.codecName).toBe('hevc');
    await p.take(decoder.video(2_000_000, true));
    expect(decoder.flush).toHaveBeenCalledOnce();
    expect(p.codecName).toBe('avc');
    p.show(1_000_000);
    expect(posted.findLast(m => m.type === 'drawn')).toMatchObject({ frameUs: 1_000_000 });
    p.reset();
  });

  it('keeps equal presentation timestamps attached to their own chunk context through reordered output', async () => {
    const decoder = bufferedDecoder(true);
    const raster = markedCanvas();
    const p = new Pipeline(m => posted.push(m));
    p.setCanvas(raster.canvas, true);
    const context = (id: number) => ({
      type: 'media' as const,
      context_id: id,
      stream: 1,
      chunk_id: String(id),
      revision: 1,
      profile_status: id === 1 ? ('absent' as const) : ('unavailable' as const),
      native_geometry: null,
    });
    await p.take(decoder.video(1_000_000, true, true), context(1), 4);
    await p.take(decoder.video(1_000_000, true), context(2), 4);
    // The second unit has an opaque negative decoder tag because the real presentation timestamp is in flight.
    decoder.emit(0, -1);
    p.show(1_000_000);
    expect(posted.findLast(m => m.type === 'frame')).toMatchObject({
      frameUs: 1_000_000,
      epoch: 4,
      metadata: { profileStatus: 'unavailable', flip: false },
    });
    decoder.emit(0, 1_000_000);
    expect(posted.findLast(m => m.type === 'frame')).toMatchObject({
      frameUs: 1_000_000,
      metadata: { profileStatus: 'absent', flip: true },
    });
    p.reset();
  });

  it('bounds pending video orientation and closes outputs whose metadata was evicted', async () => {
    const decoder = bufferedDecoder(true);
    const raster = markedCanvas();
    const p = new Pipeline(m => posted.push(m));
    p.setCanvas(raster.canvas);
    await p.take(decoder.video(1_000_000, true, true));
    for (let i = 1; i < 601; i++) await p.take(decoder.video(1_000_000 + i * 33_000));
    const evicted = decoder.emit(0, 1_000_000);
    expect(evicted.close).toHaveBeenCalledOnce();
    decoder.emit(0, 20_800_000); // the last submitted timestamp retains its own upright orientation
    p.show('live');
    expect(raster.pixels()).toEqual([1, 2, 3, 4]);
    p.reset();
  });

  it('rejects old decoder outputs after an error without stealing a replacement timestamp orientation', async () => {
    const decoder = bufferedDecoder(true);
    const raster = markedCanvas();
    const p = new Pipeline(m => posted.push(m));
    p.setCanvas(raster.canvas);
    await p.take(decoder.video(1_000_000, true, true));
    decoder.fail(0);
    await p.take(decoder.video(1_000_000, true));
    const stale = decoder.emit(0, 1_000_000);
    expect(stale.close).toHaveBeenCalledOnce();
    decoder.emit(1, 1_000_000);
    p.show(1_000_000);
    expect(raster.pixels()).toEqual([1, 2, 3, 4]);
    p.reset();
  });

  it('closes a delayed JPEG from before reset instead of retaining its old orientation', async () => {
    let finish!: (value: { image: MarkedBitmap }) => void;
    vi.stubGlobal(
      'ImageDecoder',
      class {
        decode() {
          return new Promise<{ image: MarkedBitmap }>(resolve => {
            finish = resolve;
          });
        }
        close() {}
      },
    );
    const p = new Pipeline(m => posted.push(m));
    const raster = markedCanvas();
    p.setCanvas(raster.canvas);
    const pending = p.take(jpegMessage(1_000_000, true));
    await Promise.resolve();
    p.reset();
    const obsolete = new MarkedBitmap();
    finish({ image: obsolete });
    await pending;
    p.show(1_000_000);
    expect(obsolete.closed).toBe(true);
    expect(raster.order).toEqual([]);
  });

  it('reports a bad message instead of throwing', async () => {
    const p = new Pipeline(m => posted.push(m));
    await p.take(new ArrayBuffer(3));
    expect(posted[0]).toMatchObject({ type: 'error' });
  });
});
