// Stand-ins for the browser pieces jsdom lacks, for this folder's tests only: EventSource, animation frames,
// createImageBitmap, and Continuum's gateway (a fake connection a test drives).
import { act } from '@testing-library/react';
import * as flatbuffers from 'flatbuffers';
import { vi } from 'vitest';
import {
  Box as FbBox,
  Joint as FbJoint,
  Overlay as FbOverlay,
  Skeleton as FbSkeleton,
} from '../player/generated/continuum/overlay.js';
import { encodeHeader, HEADER_LEN } from '../player/header';
import type { Connection, Handlers } from '../player/transport';

/** The browser's EventSource. */
export class FakeEventSource {
  static readonly CLOSED = 2;
  static all: FakeEventSource[] = [];
  readyState = 1;
  readonly url: string;
  onmessage: ((e: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(url: string) {
    this.url = url;
    FakeEventSource.all.push(this);
  }
  get closed() {
    return this.readyState === FakeEventSource.CLOSED;
  }
  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
  send(data: unknown) {
    act(() => this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) })));
  }
  /** The server went away for good: the browser gives up and reports it. */
  fail() {
    this.readyState = FakeEventSource.CLOSED;
    this.onerror?.();
  }
}

/** Animation frames that run only when a test steps them, at a time it picks. */
export function fakeFrames() {
  const frames = new Map<number, FrameRequestCallback>();
  let nextId = 0;
  vi.stubGlobal('requestAnimationFrame', (f: FrameRequestCallback) => {
    frames.set(++nextId, f);
    return nextId;
  });
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id));
  return {
    get pending() {
      return frames.size;
    },
    step(now: number) {
      act(() => {
        const due = [...frames.values()];
        frames.clear();
        for (const f of due) f(now);
      });
    },
  };
}

/** createImageBitmap, decoding any blob as a 640x360 picture; with hold, each decode waits for release(). */
export function fakeBitmaps({ hold = false } = {}) {
  const sizes: number[] = [];
  const waiting: (() => void)[] = [];
  const decode = vi.fn((blob: Blob) => {
    sizes.push(blob.size);
    const bmp = { width: 640, height: 360, close: vi.fn() } as unknown as ImageBitmap;
    if (!hold) return Promise.resolve(bmp);
    return new Promise<ImageBitmap>(resolve => waiting.push(() => resolve(bmp)));
  });
  vi.stubGlobal('createImageBitmap', decode);
  return {
    decode,
    /** The byte size of each JPEG decoded, in order. */
    sizes,
    release() {
      waiting.shift()?.();
    },
  };
}

/** A canvas whose 2D context records drawImage calls, and the transforms around them. */
export function recordingCanvas() {
  const canvas = document.createElement('canvas');
  const drawImage = vi.fn();
  const translate = vi.fn();
  const scale = vi.fn();
  const context = { drawImage, translate, scale, save: vi.fn(), restore: vi.fn() };
  canvas.getContext = (() => context) as unknown as HTMLCanvasElement['getContext'];
  return { canvas, drawImage, translate, scale };
}

/** Let document.hidden be set by the test, firing visibilitychange. restore() puts jsdom's own back. */
export function controlVisibility() {
  let hidden = false;
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
  return {
    set(value: boolean) {
      hidden = value;
      document.dispatchEvent(new Event('visibilitychange'));
    },
    restore() {
      Reflect.deleteProperty(document, 'hidden');
    },
  };
}

/** A box in a fake gateway overlay. */
export interface FakeBox {
  xyxy: [number, number, number, number];
  label?: string;
  kind?: 'face' | 'body' | 'object';
  id?: string;
}

/** A fake gateway overlay's skeleton: its box and 26 [x, y, confidence] joints. */
export interface FakeSkeleton {
  id?: string;
  hue?: number;
  box: [number, number, number, number, number];
  joints: number[][];
}

/** The gateway's 16 byte header (player/header.ts) then the payload. */
function gatewayMessage(payloadType: number, captureUs: number, payload: Uint8Array, flip = false): ArrayBuffer {
  const out = new Uint8Array(HEADER_LEN + payload.length);
  out.set(encodeHeader({ streamId: 1, payloadType: payloadType as 1, keyframe: true, live: true, flip, captureUs }));
  out.set(payload, HEADER_LEN);
  return out.buffer;
}

/** A message of the connection's watch set (#2387): stream `streamId`, with the watch and live bits, and a media
 *  context id when given. */
export function watchedMessage(
  streamId: number,
  payloadType: number,
  captureUs: number,
  payload: Uint8Array,
  opts: { keyframe?: boolean; contextId?: number } = {},
): ArrayBuffer {
  const head = encodeHeader({
    streamId,
    payloadType: payloadType as 1,
    keyframe: opts.keyframe ?? true,
    live: true,
    watch: true,
    captureUs,
    contextId: opts.contextId,
  });
  const out = new Uint8Array(head.length + payload.length);
  out.set(head);
  out.set(payload, head.length);
  return out.buffer;
}

/** An overlay FlatBuffer, built the way the gateway builds one. */
export function overlayBytes(captureUs: number, boxes: FakeBox[], skeletons: FakeSkeleton[]): Uint8Array {
  const b = new flatbuffers.Builder(256);
  const box = (x: FakeBox) => {
    const label = b.createString(x.label ?? '');
    const kind = x.kind ? b.createString(x.kind) : 0;
    const id = x.id ? b.createString(x.id) : 0;
    return FbBox.createBox(b, x.xyxy[0], x.xyxy[1], x.xyxy[2], x.xyxy[3], 1, label, -1, kind, id);
  };
  const boxOffsets = boxes.map(box);
  const skeletonOffsets = skeletons.map(s => {
    const sb = FbBox.createBox(
      b,
      s.box[0],
      s.box[1],
      s.box[2],
      s.box[3],
      s.box[4],
      b.createString(''),
      s.hue ?? -1,
      0,
      s.id ? b.createString(s.id) : 0,
    );
    const layout = b.createString('halpe26');
    FbSkeleton.startJointsVector(b, s.joints.length);
    for (const j of [...s.joints].reverse()) FbJoint.createJoint(b, j[0] ?? 0, j[1] ?? 0, j[2] ?? 0);
    const joints = b.endVector();
    return FbSkeleton.createSkeleton(b, sb, layout, joints);
  });
  const boxVec = FbOverlay.createBoxesVector(b, boxOffsets);
  const skelVec = FbOverlay.createSkeletonsVector(b, skeletonOffsets);
  FbOverlay.startOverlay(b);
  FbOverlay.addCaptureUs(b, BigInt(captureUs));
  FbOverlay.addBoxes(b, boxVec);
  FbOverlay.addSkeletons(b, skelVec);
  b.finish(FbOverlay.endOverlay(b));
  return b.asUint8Array();
}

/** One fake gateway connection: what the page sent, and the gateway's side to drive. */
export class FakeGatewayConn {
  sent: Record<string, unknown>[] = [];
  closed = false;
  readonly handlers: Handlers;
  constructor(handlers: Handlers) {
    this.handlers = handlers;
  }
  /** Says hello; the page then asks for the streams. */
  hello() {
    this.handlers.onReply({ type: 'hello', transport: 'websocket', webtransport: null, broker: true });
  }
  /** Lists the streams: each name with its id. */
  streams(list: Record<string, number>) {
    const streams = Object.entries(list).map(([name, id]) => ({
      id,
      name,
      live: true,
      format: 'passthrough',
      start_us: 0,
      end_us: 0,
      chunks: 0,
      frames: 0,
    }));
    this.handlers.onReply({ type: 'streams', streams });
  }
  /** Says the page is following stream `stream` live. */
  live(stream: number) {
    this.handlers.onReply({ type: 'live', stream, overlays: [] });
  }
  /** Sends a JPEG frame; flip marks it from a camera mounted upside down. */
  jpeg(captureUs: number, bytes = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]), flip = false) {
    this.handlers.onBinary(gatewayMessage(4, captureUs, bytes, flip));
  }
  /** Sends an overlay. */
  overlay(captureUs: number, boxes: FakeBox[], skeletons: FakeSkeleton[] = []) {
    this.handlers.onBinary(gatewayMessage(2, captureUs, overlayBytes(captureUs, boxes, skeletons)));
  }
  /** The connection drops. */
  fail() {
    this.handlers.onClose('gone');
  }
  /** The last request the page sent. */
  get last(): Record<string, unknown> | undefined {
    return this.sent.at(-1);
  }
}

/** A fake gateway: every connection the page opens, in order. */
export function fakeGateway() {
  const conns: FakeGatewayConn[] = [];
  const connect = async (_url: string, h: Handlers): Promise<Connection> => {
    const c = new FakeGatewayConn(h);
    conns.push(c);
    return {
      transport: 'websocket',
      send: r => c.sent.push(r as Record<string, unknown>),
      close: () => {
        c.closed = true;
      },
    };
  };
  /** The open connections. */
  const open = () => conns.filter(c => !c.closed);
  /** Takes the newest open connection through hello and streams: camera `cam` is stream `id`. */
  const ready = (cam: string, id: number): FakeGatewayConn => {
    const c = open().at(-1);
    if (!c) throw new Error('no open gateway connection');
    c.hello();
    c.streams({ [`${cam}.video`]: id });
    c.live(id);
    return c;
  };
  return { connect, conns, open, ready };
}
