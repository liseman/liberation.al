// The 16 byte header at the start of every binary message from the Continuum gateway (docs/CONTINUUM.md, "Browser
// gateway"; the Rust side is continuum/crates/continuum-gateway/src/header.rs, with the same test vectors).
// Little-endian: stream_id u32 at 0, payload_type u8 at 4, flags u8 at 5 (bit 0 keyframe, bit 1 live,
// bit 2 rotate 180 degrees, bit 3 a media context id follows, bit 4 from the connection's `watch` set, bit 5 a
// request id follows), two zero bytes, and capture_us u64 at 8, read with getBigUint64 and turned into a number
// (exact until the year 2255). After the 16 bytes come the u32 extensions the flags announce, the media context id
// first and then the request id (#2384); the payload follows them.

export const HEADER_LEN = 16;

/** What follows the header. */
export const PayloadType = {
  /** One H.264 or HEVC access unit, Annex B. */
  video: 0x01,
  /** A FlatBuffer Overlay (schemas/overlay.fbs). */
  overlay: 0x02,
  /** An Arrow IPC table. */
  arrow: 0x03,
  /** One JPEG frame. */
  jpeg: 0x04,
} as const;
export type PayloadType = (typeof PayloadType)[keyof typeof PayloadType];

export interface FrameHeader {
  streamId: number;
  payloadType: PayloadType;
  keyframe: boolean;
  /** From a live subscription, not a range (so a page tells the two apart when they interleave). */
  live: boolean;
  /** Rotate raw camera pixels 180 degrees before upright overlays; never a single-axis mirror. */
  flip: boolean;
  /** Capture time, microseconds UTC. */
  captureUs: number;
  /** Present only for opted-in recorded media. */
  contextId?: number;
  /** From the connection's `watch` set (the live bit is set too), not its single `live` feed. */
  watch?: boolean;
  /** The `req` of the request that sent this frame, when it carried one. */
  req?: number;
}

const FLAG_CONTEXT = 0x08;
const FLAG_WATCH = 0x10;
const FLAG_REQUEST = 0x20;

/** Where a message's payload starts: after the header and the u32 extensions its flags announce. */
export function payloadOffset(header: Pick<FrameHeader, 'contextId' | 'req'>): number {
  return HEADER_LEN + (header.contextId === undefined ? 0 : 4) + (header.req === undefined ? 0 : 4);
}

const TYPES = new Set<number>(Object.values(PayloadType));

/** Reads the header at the start of a message. Throws for a short message, an unknown type, or reserved bytes. */
export function decodeHeader(buffer: ArrayBuffer, offset = 0): FrameHeader {
  if (buffer.byteLength - offset < HEADER_LEN)
    throw new Error(`a frame needs a 16 byte header, got ${buffer.byteLength - offset} bytes`);
  const view = new DataView(buffer, offset, HEADER_LEN);
  if (view.getUint8(6) !== 0 || view.getUint8(7) !== 0) throw new Error('reserved header bytes are not zero');
  const type = view.getUint8(4);
  if (!TYPES.has(type)) throw new Error(`unknown payload type 0x${type.toString(16).padStart(2, '0')}`);
  const us = view.getBigUint64(8, true);
  if (us > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('capture_us is past what a number holds exactly');
  const flags = view.getUint8(5);
  const contextual = (flags & FLAG_CONTEXT) !== 0;
  const tagged = (flags & FLAG_REQUEST) !== 0;
  const end = HEADER_LEN + (contextual ? 4 : 0) + (tagged ? 4 : 0);
  if (buffer.byteLength - offset < end) throw new Error('the header extensions are truncated');
  const u32 = (at: number) => new DataView(buffer, offset + at, 4).getUint32(0, true);
  return {
    ...(contextual ? { contextId: u32(HEADER_LEN) } : {}),
    ...(tagged ? { req: u32(HEADER_LEN + (contextual ? 4 : 0)) } : {}),
    ...((flags & FLAG_WATCH) !== 0 ? { watch: true } : {}),
    streamId: view.getUint32(0, true),
    payloadType: type as PayloadType,
    keyframe: (view.getUint8(5) & 0x01) !== 0,
    live: (view.getUint8(5) & 0x02) !== 0,
    flip: (view.getUint8(5) & 0x04) !== 0,
    captureUs: Number(us),
  };
}

/** The 16 bytes for a header (the Rust gateway writes these; the page writes them only in tests). */
export function encodeHeader(h: Omit<FrameHeader, 'flip'> & { flip?: boolean }): Uint8Array {
  const out = new Uint8Array(payloadOffset(h));
  const view = new DataView(out.buffer);
  view.setUint32(0, h.streamId, true);
  view.setUint8(4, h.payloadType);
  view.setUint8(
    5,
    (h.keyframe ? 1 : 0) |
      (h.live ? 2 : 0) |
      (h.flip ? 4 : 0) |
      (h.contextId === undefined ? 0 : FLAG_CONTEXT) |
      (h.watch ? FLAG_WATCH : 0) |
      (h.req === undefined ? 0 : FLAG_REQUEST),
  );
  view.setBigUint64(8, BigInt(h.captureUs), true);
  if (h.contextId !== undefined) view.setUint32(HEADER_LEN, h.contextId, true);
  if (h.req !== undefined) view.setUint32(HEADER_LEN + (h.contextId === undefined ? 0 : 4), h.req, true);
  return out;
}

/** Splits a message into its header and its payload (a view, no copy). */
export function splitMessage(buffer: ArrayBuffer): { header: FrameHeader; payload: Uint8Array } {
  const header = decodeHeader(buffer);
  return { header, payload: new Uint8Array(buffer, payloadOffset(header)) };
}
