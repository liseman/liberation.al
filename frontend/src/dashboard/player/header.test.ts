import { describe, expect, it } from 'vitest';
import { decodeHeader, encodeHeader, type FrameHeader, HEADER_LEN, PayloadType, splitMessage } from './header';

/** The same vectors as continuum-gateway's header.rs, so both ends agree byte for byte. */
const VECTORS: [FrameHeader, number[]][] = [
  [
    {
      streamId: 7,
      payloadType: PayloadType.video,
      keyframe: true,
      live: false,
      flip: false,
      captureUs: 1_700_000_000_123_456,
    },
    [7, 0, 0, 0, 0x01, 0x01, 0, 0, 0x40, 0x22, 0x20, 0x18, 0x24, 0x0a, 0x06, 0x00],
  ],
  [
    { streamId: 0x01020304, payloadType: PayloadType.overlay, keyframe: false, live: false, flip: false, captureUs: 1 },
    [4, 3, 2, 1, 0x02, 0x00, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  ],
  [
    { streamId: 0, payloadType: PayloadType.arrow, keyframe: false, live: false, flip: false, captureUs: 0 },
    [0, 0, 0, 0, 0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  [
    {
      streamId: 0xffffffff,
      payloadType: PayloadType.jpeg,
      keyframe: true,
      live: false,
      flip: false,
      captureUs: Number.MAX_SAFE_INTEGER,
    },
    [0xff, 0xff, 0xff, 0xff, 0x04, 0x01, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x1f, 0x00],
  ],
];

const LIVE: [FrameHeader, number[]] = [
  { streamId: 9, payloadType: PayloadType.jpeg, keyframe: true, live: true, flip: false, captureUs: 2 },
  [9, 0, 0, 0, 0x04, 0x03, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
];

describe('the frame header', () => {
  it('keeps the live flag in bit 1 of the flags byte, as the gateway writes it', () => {
    expect(Array.from(encodeHeader(LIVE[0]))).toEqual(LIVE[1]);
    expect(decodeHeader(new Uint8Array(LIVE[1]).buffer)).toEqual(LIVE[0]);
  });

  it('uses bit 2 for 180-degree rotation without changing keyframe, live, or reserved bytes', () => {
    const header = { streamId: 9, payloadType: PayloadType.jpeg, keyframe: true, live: true, flip: true, captureUs: 2 };
    const bytes = [9, 0, 0, 0, 0x04, 0x07, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
    expect(Array.from(encodeHeader(header))).toEqual(bytes);
    expect(decodeHeader(new Uint8Array(bytes).buffer)).toEqual(header);
  });

  it.each([
    { keyframe: false, live: false, flags: 4 },
    { keyframe: true, live: false, flags: 5 },
    { keyframe: false, live: true, flags: 6 },
    { keyframe: true, live: true, flags: 7 },
  ])('keeps rotation independent of keyframe=$keyframe/live=$live', ({ keyframe, live, flags }) => {
    const bytes = encodeHeader({
      streamId: 9,
      payloadType: PayloadType.video,
      captureUs: 2,
      flip: true,
      keyframe,
      live,
    });
    expect(Array.from(bytes)).toEqual([9, 0, 0, 0, 1, flags, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
    expect(decodeHeader(bytes.buffer as ArrayBuffer)).toMatchObject({ flip: true, keyframe, live });
  });

  it('defaults old messages without the rotation bit to upright pixels', () => {
    const header = { streamId: 9, payloadType: PayloadType.jpeg, keyframe: true, live: true, captureUs: 2 };
    expect(Array.from(encodeHeader(header))).toEqual(LIVE[1]);
    expect(decodeHeader(new Uint8Array(LIVE[1]).buffer).flip).toBe(false);
  });

  it('encodes every payload type to its vector and decodes it back', () => {
    for (const [header, bytes] of VECTORS) {
      expect(Array.from(encodeHeader(header))).toEqual(bytes);
      expect(decodeHeader(new Uint8Array(bytes).buffer)).toEqual(header);
    }
    expect(VECTORS.map(([h]) => h.payloadType)).toEqual(Object.values(PayloadType));
  });

  it('refuses short, unknown, reserved, and too-large headers', () => {
    expect(() => decodeHeader(new ArrayBuffer(15))).toThrow(/16 byte header/);
    const bad = new Uint8Array(VECTORS[0]?.[1] ?? []);
    bad[4] = 9;
    expect(() => decodeHeader(bad.buffer)).toThrow(/unknown payload type 0x09/);
    const reserved = new Uint8Array(VECTORS[0]?.[1] ?? []);
    reserved[7] = 1;
    expect(() => decodeHeader(reserved.buffer)).toThrow(/reserved/);
    const huge = new Uint8Array(16);
    huge[4] = 1;
    huge.fill(0xff, 8);
    expect(() => decodeHeader(huge.buffer)).toThrow(/exactly/);
  });

  it("reads a request id after the header and any media context id, as the gateway's with_request writes it", () => {
    // The vectors of continuum-gateway's header.rs request_tests: JPEG on stream 4 at 5000 us, keyframe.
    const base = [4, 0, 0, 0, 0x04, 0x01, 0, 0, 0x88, 0x13, 0, 0, 0, 0, 0, 0];
    const plain = new Uint8Array([...base, 7, 0, 0, 0, 0x6a, 0x70, 0x65, 0x67]);
    plain[5] = 0x01 | 0x20;
    const one = splitMessage(plain.buffer);
    expect([one.header.req, one.header.contextId, Array.from(one.payload)]).toEqual([
      7,
      undefined,
      [0x6a, 0x70, 0x65, 0x67],
    ]);
    const both = new Uint8Array([...base, 3, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0x6a, 0x70, 0x65, 0x67]);
    both[5] = 0x01 | 0x08 | 0x20;
    const two = splitMessage(both.buffer);
    expect([two.header.contextId, two.header.req, Array.from(two.payload)]).toEqual([
      3,
      0xffffffff,
      [0x6a, 0x70, 0x65, 0x67],
    ]);
    for (let n = 16; n < 24; n++) expect(() => decodeHeader(both.buffer.slice(0, n))).toThrow(/truncated/);
    expect(decodeHeader(new Uint8Array(base).buffer).req).toBeUndefined();
  });

  it("marks a frame of the connection's watch set, keeping its other flags", () => {
    const watched = new Uint8Array(16);
    watched.set([4, 0, 0, 0, 0x01, 0x01 | 0x02 | 0x04 | 0x10]);
    const h = decodeHeader(watched.buffer);
    expect([h.watch, h.live, h.keyframe, h.flip]).toEqual([true, true, true, true]);
    expect(
      decodeHeader(new Uint8Array([4, 0, 0, 0, 1, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).buffer).watch,
    ).toBeUndefined();
  });

  it('encodes the watch bit and both extensions in the order the gateway reads them, and back', () => {
    const h = {
      streamId: 9,
      payloadType: PayloadType.video,
      keyframe: true,
      live: true,
      flip: false,
      captureUs: 1234,
      contextId: 5,
      req: 6,
      watch: true,
    };
    const bytes = encodeHeader(h);
    expect(bytes.length).toBe(24);
    expect(bytes[5]).toBe(0x01 | 0x02 | 0x08 | 0x10 | 0x20);
    expect(Array.from(bytes.slice(16))).toEqual([5, 0, 0, 0, 6, 0, 0, 0]);
    expect(decodeHeader(bytes.buffer as ArrayBuffer)).toEqual(h);
  });

  it('splits a message into its header and payload without copying', () => {
    const header = encodeHeader({
      streamId: 3,
      payloadType: PayloadType.jpeg,
      keyframe: true,
      live: false,
      flip: false,
      captureUs: 42,
    });
    const message = new Uint8Array(HEADER_LEN + 3);
    message.set(header);
    message.set([1, 2, 3], HEADER_LEN);
    const { header: h, payload } = splitMessage(message.buffer);
    expect(h.captureUs).toBe(42);
    expect(Array.from(payload)).toEqual([1, 2, 3]);
    expect(payload.buffer).toBe(message.buffer);
  });
});
