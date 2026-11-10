import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { fakeGateway, overlayBytes, watchedMessage } from './testFakes';

// gateway.ts and player/watch.ts keep their state at module level, so each test loads fresh copies of both.
let gw: typeof import('./gateway');
let watch: typeof import('../player/watch');
let net: ReturnType<typeof fakeGateway>;
const settle = (ms = 0) => vi.advanceTimersByTimeAsync(ms);

beforeEach(async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
  vi.resetModules();
  net = fakeGateway();
  watch = await import('../player/watch');
  watch.setWatchForTests(net.connect);
  gw = await import('./gateway');
  gw.setGatewayForTests(net.connect, 'ws://gateway/ws', true);
});
afterEach(() => {
  watch.setWatchForTests(() => new Promise(() => {}));
  vi.useRealTimers();
});

/** The newest connection's gateway says hello (with these features) and lists the cameras. */
async function greet(features: string[] = ['watch', 'req']) {
  await settle();
  const c = net.open().at(-1);
  if (!c) throw new Error('no connection');
  c.handlers.onReply({ type: 'hello', transport: 'websocket', webtransport: null, features });
  c.streams({ 'cam0.video': 1, 'kitchen.video': 2 });
  await settle();
  return c;
}

it("follows every camera Live shows on the page's one connection, at every frame with their overlays", async () => {
  const jpegs: number[] = [];
  const boxes: string[] = [];
  gw.watchFrames('cam0', (_jpeg, at) => jpegs.push(at));
  gw.watchBoxes('cam0', d => boxes.push(...(d.faces ?? []).map(f => f.name)));
  gw.watchVideo('kitchen', () => {});
  const c = await greet();
  expect(net.conns).toHaveLength(1);
  expect(c.sent.filter(r => r.type === 'watch')).toEqual([
    {
      type: 'watch',
      streams: [
        { id: 1, max_fps: 0, overlays: ['faces', 'objects', 'poses'] },
        { id: 2, max_fps: 0, overlays: ['faces', 'objects', 'poses'] },
      ],
      media_context: true,
    },
  ]);
  expect(c.sent.some(r => r.type === 'live')).toBe(false);
  c.handlers.onReply({
    type: 'watching',
    streams: [
      { id: 1, max_fps: 0, overlays: [{ name: 'face.match', id: 30 }] },
      { id: 2, max_fps: 0, overlays: [] },
    ],
  });
  c.handlers.onBinary(watchedMessage(1, 4, 5_000_000, new Uint8Array([0xff, 0xd8, 0xff, 0xd9])));
  c.handlers.onBinary(
    watchedMessage(
      30,
      2,
      5_000_000,
      overlayBytes(5_000_000, [{ xyxy: [0, 0, 1, 1], kind: 'face', label: 'Ana', id: 't1' }], []),
    ),
  );
  expect(jpegs).toEqual([5]);
  expect(boxes).toContain('Ana');
});

it("hands a camera's video its media context, and resets its decoder when the connection goes", async () => {
  const got: (string | null)[] = [];
  gw.watchVideo('kitchen', (buffer, media) => got.push(buffer === null ? null : `ctx ${media?.context_id ?? '-'}`));
  const c = await greet();
  const media = {
    type: 'media',
    stream: 2,
    context_id: 9,
    capture_epoch: 1,
    geometry: null,
    color: null,
    profile: null,
  };
  c.handlers.onReply(media as never);
  c.handlers.onBinary(watchedMessage(2, 1, 6_000_000, new Uint8Array([0, 0, 0, 1, 0x65]), { contextId: 9 }));
  c.fail();
  expect(got).toEqual(['ctx 9', null]);
});

it("gives each camera a connection of its own on a gateway that doesn't answer watch", async () => {
  gw.watchFrames('cam0', () => {});
  await greet([]); // an older gateway: no features
  await settle();
  const own = net.open().at(-1);
  expect(net.conns).toHaveLength(2);
  own?.hello();
  own?.streams({ 'cam0.video': 1 });
  expect(own?.last).toEqual({ type: 'live', stream: 1, overlays: ['faces', 'objects', 'poses'] });
});

it('lets the camera go from the watch set when nothing on the page shows it', async () => {
  const off = gw.watchFrames('cam0', () => {});
  const c = await greet();
  off();
  await settle();
  expect(c.closed).toBe(true); // its last stream gone, the page's watch connection closes
});

it("polls Live's camera list on the page's one connection instead of a connection each time", async () => {
  const { renderHook } = await import('@testing-library/react');
  const { result } = renderHook(() => gw.useGatewayLiveCams(60));
  const c = await greet();
  await settle(gw.CAMS_POLL_MS * 3);
  expect(net.conns).toHaveLength(1);
  expect(c.sent.filter(r => r.type === 'streams').length).toBeGreaterThanOrEqual(3);
  expect(result.current).toEqual([]); // the fake's streams have no frames: none live
});
