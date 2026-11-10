import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { controlVisibility } from '../camera/testFakes';
import { encodeHeader, PayloadType } from './header';
import type { Connection, Handlers } from './transport';

// watch.ts keeps its state at module level, so each test loads a fresh copy.
let watch: typeof import('./watch');
interface Wire {
  handlers: Handlers;
  sent: Record<string, unknown>[];
  closed: boolean;
}
let wires: Wire[];
const settle = (ms = 0) => vi.advanceTimersByTimeAsync(ms);
const STREAMS = ['kitchen.video', 'patio.video', 'cam0.video'];

beforeEach(async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
  vi.resetModules();
  watch = await import('./watch');
  wires = [];
  watch.setWatchForTests(async (_url, handlers): Promise<Connection> => {
    const wire: Wire = { handlers, sent: [], closed: false };
    wires.push(wire);
    return {
      transport: 'websocket',
      send: r => wire.sent.push(r as Record<string, unknown>),
      close: () => {
        wire.closed = true;
      },
    };
  });
});
afterEach(() => {
  // Forget this copy's subscribers: its visibility listener outlives the test.
  watch.setWatchForTests(() => new Promise(() => {}));
  vi.useRealTimers();
});

const wire = () => wires.at(-1) as Wire;
/** The gateway says hello (with the features it has) and lists its streams. */
async function greet(features = ['watch', 'req']) {
  await settle();
  wire().handlers.onReply({ type: 'hello', transport: 'websocket', webtransport: null, features });
  wire().handlers.onReply({
    type: 'streams',
    streams: STREAMS.map((name, i) => ({
      id: i + 1,
      name,
      format: 'h264_au',
      start_us: 0,
      end_us: 0,
      chunks: 0,
      frames: 0,
    })),
  });
  await settle();
}
const watches = () => wire().sent.filter(r => r.type === 'watch' || r.type === 'unwatch');
function frame(stream: number, opts: { watch?: boolean; contextId?: number } = {}) {
  return encodeHeader({
    streamId: stream,
    payloadType: PayloadType.video,
    keyframe: true,
    live: true,
    captureUs: 1000,
    watch: opts.watch ?? true,
    contextId: opts.contextId,
  }).buffer as ArrayBuffer;
}

it('follows every subscribed stream on one connection, each at the fastest rate any subscriber wants', async () => {
  watch.watchStream('kitchen.video', 1, () => {});
  watch.watchStream('patio.video', 1, () => {});
  watch.watchStream('kitchen.video', 4, () => {});
  await greet();
  expect(wires).toHaveLength(1);
  expect(watches()).toEqual([
    {
      type: 'watch',
      streams: [
        { id: 1, max_fps: 4 },
        { id: 2, max_fps: 1 },
      ],
      media_context: true,
    },
  ]);
  const every = watch.watchStream('patio.video', 0, () => {}); // every frame wins over any rate
  await settle();
  expect(watches().at(-1)?.streams).toEqual([
    { id: 1, max_fps: 4 },
    { id: 2, max_fps: 0 },
  ]);
  every();
  await settle();
  expect(watches().at(-1)?.streams).toEqual([
    { id: 1, max_fps: 4 },
    { id: 2, max_fps: 1 },
  ]);
  expect(wires).toHaveLength(1);
});

it("hands each subscriber its stream's watched frames, with their media context, and nothing else", async () => {
  const kitchen = vi.fn();
  const patio = vi.fn();
  watch.watchStream('kitchen.video', 1, kitchen);
  watch.watchStream('patio.video', 1, patio);
  await greet();
  const media = {
    type: 'media' as const,
    stream: 1,
    context_id: 7,
    capture_epoch: 1,
    geometry: null,
    color: null,
    profile: null,
  };
  wire().handlers.onReply(media as never);
  wire().handlers.onBinary(frame(1, { contextId: 7 }));
  wire().handlers.onBinary(frame(1, { watch: false })); // a `live` feed's or a request's frame
  wire().handlers.onBinary(frame(3)); // a stream nobody here follows
  wire().handlers.onBinary(new ArrayBuffer(3)); // unreadable
  expect(kitchen).toHaveBeenCalledOnce();
  expect(kitchen.mock.calls[0]?.[0].media).toBe(media);
  expect(kitchen.mock.calls[0]?.[0].header.streamId).toBe(1);
  expect(patio).not.toHaveBeenCalled();
});

it('closes the connection while the page is hidden and follows the same streams again when it shows', async () => {
  const page = controlVisibility();
  try {
    watch.watchStream('kitchen.video', 1, () => {});
    await greet();
    page.set(true);
    await settle();
    expect(wire().closed).toBe(true);
    await settle(60_000);
    expect(wires).toHaveLength(1); // nothing reconnects while hidden
    page.set(false);
    await greet();
    expect(wires).toHaveLength(2);
    expect(watches()).toEqual([{ type: 'watch', streams: [{ id: 1, max_fps: 1 }], media_context: true }]);
  } finally {
    page.restore();
  }
});

it('lets the connection go with the last subscriber, and reconnects with backoff while any remain', async () => {
  const off = watch.watchStream('kitchen.video', 1, () => {});
  await greet();
  wire().handlers.onClose('gone');
  await settle(999);
  expect(wires).toHaveLength(1);
  await settle(1);
  expect(wires).toHaveLength(2);
  await greet();
  off();
  await settle();
  expect(wire().closed).toBe(true);
  await settle(60_000);
  expect(wires).toHaveLength(2);
});

it("sends no watch to a gateway that doesn't answer it, and says so", async () => {
  watch.watchStream('kitchen.video', 1, () => {});
  expect(watch.watchAvailable()).toBeNull();
  await settle();
  wire().handlers.onReply({ type: 'hello', transport: 'websocket', webtransport: null });
  await settle(60_000);
  expect(watch.watchAvailable()).toBe(false);
  expect(wire().sent).toEqual([]);
  expect([wires.length, wire().closed]).toEqual([1, true]);
});

it('asks for the stream list again when a subscribed stream was not on it', async () => {
  watch.watchStream('bedroom.video', 1, () => {});
  await greet();
  expect(watches()).toEqual([]);
  await settle(10_000);
  watch.watchStream('kitchen.video', 1, () => {});
  await settle();
  expect(wire().sent.filter(r => r.type === 'streams')).toHaveLength(2);
});

it('asks for overlays with a stream followed at every frame, and hands its overlay frames to that stream', async () => {
  const kitchen = vi.fn();
  watch.watchStream('kitchen.video', 0, kitchen, { overlays: ['objects', 'faces'] });
  watch.watchStream('kitchen.video', 1, () => {}, { overlays: ['poses'] });
  watch.watchStream('patio.video', 1, () => {}, { overlays: ['faces'] }); // at a rate: no overlays come
  await greet();
  expect(watches()).toEqual([
    {
      type: 'watch',
      streams: [
        { id: 1, max_fps: 0, overlays: ['faces', 'objects', 'poses'] },
        { id: 2, max_fps: 1 },
      ],
      media_context: true,
    },
  ]);
  wire().handlers.onReply({
    type: 'watching',
    streams: [
      { id: 1, max_fps: 0, overlays: [{ name: 'face.match', id: 40 }] },
      { id: 2, max_fps: 1, overlays: [] },
    ],
  });
  wire().handlers.onBinary(frame(40));
  expect(kitchen.mock.calls.map(c => c[0].header.streamId)).toEqual([40]);
});

it('tells subscribers when the connection goes, so their decoders start over', async () => {
  const reset = vi.fn();
  watch.watchStream('kitchen.video', 0, () => {}, { onReset: reset });
  await greet();
  wire().handlers.onClose('gone');
  expect(reset).toHaveBeenCalledOnce();
});

it("tells subscribers, now and later, when the gateway doesn't answer watch", async () => {
  const early = vi.fn();
  watch.watchStream('kitchen.video', 0, () => {}, { onUnavailable: early });
  await settle();
  wire().handlers.onReply({ type: 'hello', transport: 'websocket', webtransport: null });
  expect(early).toHaveBeenCalledOnce();
  const late = vi.fn();
  watch.watchStream('patio.video', 0, () => {}, { onUnavailable: late });
  await settle();
  expect(late).toHaveBeenCalledOnce();
});

it('lists the streams on the one connection, which a hold keeps open with nothing watched', async () => {
  const release = watch.holdWatch();
  await settle();
  const first = watch.listStreams();
  await greet(); // the connection's own hello and streams answer the first ask
  expect((await first).map(s => s.name)).toEqual(STREAMS);
  const second = watch.listStreams();
  await settle();
  expect(wire().sent.filter(r => r.type === 'streams')).toHaveLength(2); // asked again on the same connection
  wire().handlers.onReply({ type: 'streams', streams: [] });
  expect(await second).toEqual([]);
  expect([wires.length, wire().closed]).toEqual([1, false]);
  expect(watches()).toEqual([]); // held, nothing watched: no watch, no unwatch
  release();
  await settle();
  expect(wire().closed).toBe(true);
});

// #2772: an untagged `streams` held the connection's frames until it was answered (about 1.8 s when the gateway re-read
// live chunks), so every camera stalled until its next keyframe. A gateway that says `streams_req` runs a tagged one
// beside the frames; an older one gets them untagged, as before.
it.each([
  [
    ['watch', 'req', 'streams_req'],
    [
      { type: 'streams', req: 1 },
      { type: 'streams', req: 2 },
    ],
  ],
  [
    ['watch', 'req'],
    [{ type: 'streams' }, { type: 'streams' }],
  ],
])('tags its stream lists when the gateway runs them beside the frames (%j)', async (features, asks) => {
  const release = watch.holdWatch();
  await greet(features);
  const listed = watch.listStreams();
  await settle();
  expect(wire().sent.filter(r => r.type === 'streams')).toEqual(asks);
  wire().handlers.onReply({ type: 'streams', streams: [], ...(features.includes('streams_req') ? { req: 2 } : {}) });
  expect(await listed).toEqual([]); // a tagged answer counts as the list asked for
  release();
});

it("rejects a stream list on a gateway that doesn't answer watch, so the caller can ask another way", async () => {
  const asked = watch.listStreams();
  await settle();
  wire().handlers.onReply({ type: 'hello', transport: 'websocket', webtransport: null });
  await expect(asked).rejects.toThrow(/watch/);
  await expect(watch.listStreams()).rejects.toThrow(/watch/);
});
