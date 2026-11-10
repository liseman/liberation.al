import { describe, expect, it, vi } from 'vitest';
import type { Connection, Handlers, Reply } from '../player/transport';
import {
  cameraOf,
  daySegments,
  IDLE_CLOSE_MS,
  MAX_DAY_SEGMENTS,
  MAX_SPANS_PARTS,
  normalizeSegments,
  SPANS_TIMEOUT_MS,
} from './gatewaySegments';
import type { RawSegment } from './types';

/** A fake gateway that answers `streams` with `streams`, and each `spans` with spans[id]. */
function gateway(
  streams: { id: number; name: string }[],
  spans: Record<number, [number, number][]>,
  expired: Record<number, [number, number][]> = {},
) {
  const sent: Record<string, unknown>[] = [];
  let closed = false;
  let opened = 0;
  const connect = async (_url: string, h: Handlers): Promise<Connection> => {
    opened += 1;
    queueMicrotask(() => h.onReply({ type: 'hello', transport: 'websocket', webtransport: null }));
    return {
      transport: 'websocket',
      send: r => {
        const req = r as { type: string; stream?: number; start_us?: number; end_us?: number };
        sent.push(req);
        if (req.type === 'streams')
          queueMicrotask(() =>
            h.onReply({
              type: 'streams',
              streams: streams.map(s => ({ ...s, format: 'jpeg', start_us: 0, end_us: 0 })),
            }),
          );
        else if (req.type === 'spans' && req.stream !== undefined)
          queueMicrotask(() =>
            h.onReply({
              type: 'spans',
              stream: req.stream ?? 0,
              start_us: req.start_us ?? 0,
              end_us: req.end_us ?? 0,
              spans: spans[req.stream ?? 0] ?? [],
              ...(expired[req.stream ?? 0] ? { expired: expired[req.stream ?? 0] } : {}),
            }),
          );
      },
      close: () => {
        closed = true;
      },
    };
  };
  return { connect, sent, closed: () => closed, opened: () => opened };
}

describe('cameraOf', () => {
  it("names the camera of a camera's video stream, live or v1's, and nothing else", () => {
    expect(cameraOf('cam0.video')).toBe('cam0');
    expect(cameraOf('front-door.video_v1')).toBe('front-door');
    expect(cameraOf('face.detection')).toBeNull();
  });
});

describe('daySegments', () => {
  it("lists every camera stream's stretches over the day, as segments the gateway's player plays", async () => {
    const gw = gateway(
      [
        { id: 1, name: 'cam0.video' },
        { id: 2, name: 'cam0.video_v1' },
        { id: 3, name: 'face.detection' },
      ],
      { 1: [[1_000_000, 4_000_000]], 2: [[100_000, 200_000]] },
    );
    const segs = await daySegments(0, 10, { connect: gw.connect, url: 'ws://x/ws' });
    expect(segs).toEqual([
      { cam: 'cam0', start: 1, end: 4, url: 'gateway:1:1000000:4000000' },
      { cam: 'cam0', start: 0.1, end: 0.2, url: 'gateway:2:100000:200000' },
    ]);
    expect(gw.sent.filter(r => r.type === 'spans')).toEqual([
      { type: 'spans', stream: 1, start_us: 0, end_us: 10_000_000, gap_us: 10_000_000, media_context: true },
      { type: 'spans', stream: 2, start_us: 0, end_us: 10_000_000, gap_us: 10_000_000, media_context: true },
    ]);
  });

  it("asks only about the named cameras' streams, live and v1's, when given cameras", async () => {
    const gw = gateway(
      [
        { id: 1, name: 'cam0.video' },
        { id: 2, name: 'cam1.video' },
        { id: 3, name: 'cam0.video_v1' },
        { id: 4, name: 'cam2.video' },
      ],
      { 1: [[1_000_000, 4_000_000]], 2: [[1_000_000, 4_000_000]], 3: [[100_000, 200_000]] },
    );
    const segs = await daySegments(0, 10, { connect: gw.connect, url: 'ws://only/ws', cams: ['cam0'] });
    expect(gw.sent.filter(r => r.type === 'spans').map(r => r.stream)).toEqual([1, 3]);
    expect(new Set(segs.map(s => s.cam))).toEqual(new Set(['cam0']));
  });

  it('lists again on the same connection, and hangs up once listings stop', async () => {
    vi.useFakeTimers();
    try {
      const gw = gateway([{ id: 1, name: 'cam0.video' }], { 1: [[1_000_000, 4_000_000]] });
      await daySegments(0, 10, { connect: gw.connect, url: 'ws://reuse/ws' });
      const again = await daySegments(0, 20, { connect: gw.connect, url: 'ws://reuse/ws' });
      expect(again).toEqual([{ cam: 'cam0', start: 1, end: 4, url: 'gateway:1:1000000:4000000' }]);
      expect([gw.opened(), gw.closed()]).toEqual([1, false]);
      await vi.advanceTimersByTimeAsync(IDLE_CLOSE_MS);
      expect(gw.closed()).toBe(true);
      await daySegments(0, 10, { connect: gw.connect, url: 'ws://reuse/ws' }); // a new one once it's gone
      expect(gw.opened()).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it('lists video that aged out past the history window as expired, apart from what still plays', async () => {
    const gw = gateway(
      [{ id: 1, name: 'cam0.video' }],
      { 1: [[5_000_000, 8_000_000]] },
      { 1: [[1_000_000, 4_000_000]] },
    );
    const segs = await daySegments(0, 10, { connect: gw.connect, url: 'ws://x/ws' });
    expect(segs).toEqual([
      { cam: 'cam0', start: 1, end: 4, url: 'gateway:1:1000000:4000000', expired: true },
      { cam: 'cam0', start: 5, end: 8, url: 'gateway:1:5000000:8000000' },
    ]);
  });

  it('is empty when no camera has a stream', async () => {
    const gw = gateway([{ id: 3, name: 'face.detection' }], {});
    expect(await daySegments(0, 10, { connect: gw.connect, url: 'ws://x/ws' })).toEqual([]);
  });

  it('fails when the gateway is unreachable, says an error, or takes too long', async () => {
    await expect(
      daySegments(0, 10, { connect: () => Promise.reject(new Error('refused')), url: 'ws://x/ws' }),
    ).rejects.toThrow('refused');
    const err = async (_u: string, h: Handlers): Promise<Connection> => {
      queueMicrotask(() => h.onReply({ type: 'error', message: 'spans: the broker: down' }));
      return { transport: 'websocket', send: () => {}, close: () => {} };
    };
    await expect(daySegments(0, 10, { connect: err, url: 'ws://x/ws' })).rejects.toThrow('the broker: down');
    vi.useFakeTimers();
    try {
      const silent = async (): Promise<Connection> => ({ transport: 'websocket', send: () => {}, close: () => {} });
      const waiting = daySegments(0, 10, { connect: silent, url: 'ws://x/ws' });
      const caught = expect(waiting).rejects.toThrow('took too long');
      await vi.advanceTimersByTimeAsync(SPANS_TIMEOUT_MS);
      await caught;
    } finally {
      vi.useRealTimers();
    }
  });
});

type SpansReply = Extract<Reply, { type: 'spans' }>;
function windowGateway(answer: (a: number, b: number) => Omit<SpansReply, 'type' | 'stream' | 'start_us' | 'end_us'>) {
  const sent: { start: number; end: number }[] = [];
  let closes = 0;
  const connect = async (_url: string, h: Handlers): Promise<Connection> => {
    queueMicrotask(() => h.onReply({ type: 'hello', transport: 'websocket', webtransport: null }));
    return {
      transport: 'websocket',
      close: () => {
        closes++;
      },
      send: request => {
        const r = request as { type: string; start_us: number; end_us: number };
        if (r.type === 'streams')
          queueMicrotask(() =>
            h.onReply({
              type: 'streams',
              streams: [{ id: 1, name: 'cam0.video', format: 'jpeg', start_us: 0, end_us: 0 }],
            }),
          );
        else if (r.type === 'spans') {
          sent.push({ start: r.start_us, end: r.end_us });
          queueMicrotask(() =>
            h.onReply({
              type: 'spans',
              stream: 1,
              start_us: r.start_us,
              end_us: r.end_us,
              ...answer(r.start_us, r.end_us),
            }),
          );
        }
      },
    };
  };
  return { connect, sent, closes: () => closes };
}

describe('bounded listings', () => {
  it('splits a partial window in half and asks again, rejoining a span the split cut in two', async () => {
    const gateway = windowGateway((a, b) =>
      b - a > 2_000_000 ? { spans: [[a, b]], partial: true } : { spans: [[a, b]] },
    );
    const segments = await daySegments(0, 4, { connect: gateway.connect, url: 'ws://split/ws' });
    expect(gateway.sent).toEqual([
      { start: 0, end: 4_000_000 },
      { start: 0, end: 2_000_000 },
      { start: 2_000_000, end: 4_000_000 },
    ]);
    expect(segments).toEqual([{ cam: 'cam0', start: 0, end: 4, url: 'gateway:1:0:4000000' }]);
  });

  it('takes a reply without partial as the whole window', async () => {
    const gateway = windowGateway(() => ({
      spans: [
        [0, 1_000_000],
        [3_000_000, 4_000_000],
      ],
    }));
    const segments = await daySegments(0, 4, { connect: gateway.connect, url: 'ws://whole/ws' });
    expect(gateway.sent).toEqual([{ start: 0, end: 4_000_000 }]);
    expect(segments.map(s => [s.start, s.end])).toEqual([
      [0, 1],
      [3, 4],
    ]);
  });

  it('keeps expired video apart from the playable span it touches', async () => {
    const gateway = windowGateway(() => ({ spans: [[2_000_000, 4_000_000]], expired: [[0, 2_000_000]] }));
    const segments = await daySegments(0, 4, { connect: gateway.connect, url: 'ws://touching/ws' });
    expect(segments).toEqual([
      { cam: 'cam0', start: 0, end: 2, url: 'gateway:1:0:2000000', expired: true },
      { cam: 'cam0', start: 2, end: 4, url: 'gateway:1:2000000:4000000' },
    ]);
  });

  it('bounds subdivision work and never returns a successful partial prefix', async () => {
    const gateway = windowGateway((a, b) => ({ spans: [[a, b]], partial: b - a > 2_000_000 }));
    await expect(daySegments(0, 1024, { connect: gateway.connect, url: 'ws://bounded/ws' })).rejects.toThrow(
      'safe window bound',
    );
    expect(gateway.sent).toHaveLength(MAX_SPANS_PARTS);
  });

  it('rejects a partial minimum window, then accepts a clean retry', async () => {
    let bad = true;
    const gateway = windowGateway((a, b) => ({ spans: [[a, b]], ...(bad ? { partial: true } : {}) }));
    await expect(daySegments(0, 1, { connect: gateway.connect, url: 'ws://small/ws' })).rejects.toThrow(
      'smallest safe window',
    );
    expect(gateway.closes()).toBe(0); // the listing failed on its data; the connection is fine for the next
    bad = false;
    expect(await daySegments(0, 1, { connect: gateway.connect, url: 'ws://small/ws' })).toEqual([
      { cam: 'cam0', start: 0, end: 1, url: 'gateway:1:0:1000000' },
    ]);
  });
});

describe('a fragmented day', () => {
  // Bursts of a second, 5 s apart: MAX_DAY_SEGMENTS + 808 of them, as a camera recording in short bursts lists.
  const n = MAX_DAY_SEGMENTS + 808;
  const bursts = Array.from({ length: n }, (_, i) => ({
    cam: 'cam0',
    start: i * 5,
    end: i * 5 + 1,
    url: `gateway:1:${i * 5_000_000}:${(i * 5 + 1) * 1_000_000}`,
  }));

  it('joins wider gaps until the day fits, instead of failing it', () => {
    expect(normalizeSegments(bursts)).toEqual([
      { cam: 'cam0', start: 0, end: (n - 1) * 5 + 1, url: `gateway:1:0:${((n - 1) * 5 + 1) * 1_000_000}` },
    ]);
  });

  it('keeps gaps when the day fits, and video aged out apart from what plays', () => {
    const few: RawSegment[] = [
      ...bursts.slice(0, 2),
      { cam: 'cam0', start: 20, end: 30, url: 'gateway:1:20000000:30000000', expired: true },
    ];
    expect(normalizeSegments(few).map(s => [s.start, s.end, s.expired ?? false])).toEqual([
      [0, 1, false],
      [5, 6, false],
      [20, 30, true],
    ]);
  });

  it('joins expired slivers with each other, never into playable video, until the day fits', () => {
    const gone: RawSegment[] = bursts.map(s => ({ ...s, expired: true }));
    const after = n * 5 + 10;
    const playing = {
      cam: 'cam0',
      start: after,
      end: after + 60,
      url: `gateway:1:${after * 1_000_000}:${(after + 60) * 1_000_000}`,
    };
    expect(normalizeSegments([...gone, playing]).map(s => [s.start, s.end, s.expired ?? false])).toEqual([
      [0, (n - 1) * 5 + 1, true],
      [after, after + 60, false],
    ]);
  });

  it('still fails a listing that no joining brings under the bound', () => {
    const apart = Array.from({ length: n }, (_, i) => ({
      cam: 'cam0',
      start: i * 7200,
      end: i * 7200 + 1,
      url: `gateway:1:${i * 7_200_000_000}:${(i * 7200 + 1) * 1_000_000}`,
    }));
    expect(() => normalizeSegments(apart)).toThrow('safe bound');
  });
});

it('bounds each camera independently', async () => {
  const requested: number[] = [];
  const connect = async (_url: string, handlers: Handlers): Promise<Connection> => {
    const reply = (value: Reply) => queueMicrotask(() => handlers.onReply(value));
    queueMicrotask(() => reply({ type: 'hello', transport: 'websocket', webtransport: null }));
    return {
      transport: 'websocket',
      close() {},
      send(request: object) {
        const message = request as { type: string; stream: number; start_us: number; end_us: number };
        if (message.type === 'streams')
          reply({ type: 'streams', streams: [1, 2, 3].map(id => ({ id, name: `cam${id}.video` })) } as Reply);
        if (message.type !== 'spans') return;
        requested.push(message.stream);
        const a = message.start_us,
          b = message.end_us;
        const partial = b - a > 2_000_000;
        reply({
          type: 'spans',
          stream: message.stream,
          start_us: a,
          end_us: b,
          spans: partial ? [] : [[a, b]],
          ...(partial ? { partial } : {}),
        });
      },
    };
  };
  const segments = await daySegments(0, 128, { connect, url: 'ws://three-cameras/ws' });
  expect(requested).toHaveLength(381);
  expect([1, 2, 3].map(stream => requested.filter(value => value === stream).length)).toEqual([127, 127, 127]);
  expect(segments.map(segment => [segment.cam, segment.start, segment.end])).toEqual([
    ['cam1', 0, 128],
    ['cam2', 0, 128],
    ['cam3', 0, 128],
  ]);
});
