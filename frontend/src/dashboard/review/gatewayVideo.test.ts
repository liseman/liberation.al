import { afterEach, describe, expect, it, vi } from 'vitest';
import { encodeHeader, HEADER_LEN, PayloadType } from '../player/header';
import type { Connection, Handlers } from '../player/transport';
import type { FromWorker, ToWorker, WorkerLike } from '../player/worker';
import { createRecordedFrameCapture, GatewayVideo, gatewaySegmentUrl, parseSegmentUrl } from './gatewayVideo';

/** A fake gateway: each connection's requests, its handlers to answer with, and whether it was closed. */
/** A fake gateway: each connection says hello first (listing `features`), then records what's sent on it. */
function fakeGateway(features: string[] = []) {
  const conns: { sent: Record<string, unknown>[]; handlers: Handlers; closed: boolean; last?: unknown }[] = [];
  const connect = async (_url: string, handlers: Handlers): Promise<Connection> => {
    queueMicrotask(() => handlers.onReply({ type: 'hello', transport: 'websocket', webtransport: null, features }));
    const c = {
      sent: [] as Record<string, unknown>[],
      handlers,
      closed: false,
      get last() {
        return this.sent.at(-1);
      },
    };
    conns.push(c);
    return {
      transport: 'websocket',
      send: r => c.sent.push(r as Record<string, unknown>),
      close: () => {
        c.closed = true;
      },
    };
  };
  return { connect, conns };
}

class FakeWorker implements WorkerLike {
  posted: ToWorker[] = [];
  onmessage: WorkerLike['onmessage'] = null;
  postMessage(m: ToWorker) {
    this.posted.push(m);
  }
  terminated = false;
  terminate() {
    this.terminated = true;
  }
  /** The playheads posted, in microseconds. */
  get heads(): number[] {
    return this.posted.flatMap(m => (m.type === 'playhead' && typeof m.us === 'number' ? [m.us] : []));
  }
}

/** A manual animation clock: step(ms) runs the frame callback once, ms later. */
function clock() {
  let cb: ((now: number) => void) | null = null;
  let now = 1000;
  return {
    raf: (f: (now: number) => void) => {
      cb = f;
      return 1;
    },
    cancelRaf: () => {
      cb = null;
    },
    step(ms: number) {
      now += ms;
      const f = cb;
      cb = null;
      f?.(now);
    },
  };
}

const flush = () => new Promise(r => setTimeout(r, 0));

/** `wall`: the wall clock the player tells drags by; by default a second passes between any two reads, so no two
 *  seeks count as a drag. */
function rig(wall?: () => number, features: string[] = []) {
  const net = fakeGateway(features);
  const worker = new FakeWorker();
  const time = clock();
  let t = 0;
  const v = new GatewayVideo(document.createElement('canvas'), {
    connect: net.connect,
    makeWorker: () => worker,
    windowUs: 2_000_000,
    raf: time.raf,
    cancelRaf: time.cancelRaf,
    now: wall ?? (() => (t += 1000)),
  });
  const events: string[] = [];
  for (const e of ['loadedmetadata', 'seeked', 'play', 'playing', 'pause', 'ended', 'error', 'emptied'])
    v.addEventListener(e, () => events.push(e));
  /** The gateway answers the last range: done, continuing at next. */
  const done = (next: number | null = null) => {
    const r = net.conns.at(-1)?.last as { start_us: number; end_us: number };
    net.conns.at(-1)?.handlers.onReply({
      type: 'done',
      stream: 7,
      start_us: r.start_us,
      end_us: r.end_us,
      frames: 1,
      skipped: 0,
      next_us: next,
      codecs: ['jpeg'],
    });
  };
  return { net, worker, time, v, events, done };
}

describe('segment URLs', () => {
  it('name a stream and the stretch it covers, and nothing else parses', () => {
    expect(parseSegmentUrl(gatewaySegmentUrl(7, 10, 20))).toEqual({ stream: 7, startUs: 10, endUs: 20 });
    expect(parseSegmentUrl('cam0/x.mp4')).toBeNull();
    expect(parseSegmentUrl('gateway:7:20:10')).toBeNull();
  });
});

describe('GatewayVideo', () => {
  it('opens a fresh connection when the source changes during connection setup', async () => {
    const net = fakeGateway();
    let release!: () => void;
    const gate = new Promise<void>(resolve => {
      release = resolve;
    });
    let calls = 0;
    const v = new GatewayVideo(document.createElement('canvas'), {
      makeWorker: () => new FakeWorker(),
      connect: async (url, handlers) => {
        calls++;
        if (calls === 1) await gate;
        return net.connect(url, handlers);
      },
    });
    v.src = gatewaySegmentUrl(7, 0, 10_000_000);
    v.src = gatewaySegmentUrl(8, 20_000_000, 30_000_000);
    release();
    await flush();
    await flush();
    expect(net.conns.some(c => !c.closed && c.sent.some(r => r.type === 'range' && r.stream === 8))).toBe(true);
    // The obsolete socket reports its close asynchronously, after the new socket is already usable.
    net.conns[0]?.handlers.onClose('old source closed');
    const range = net.conns[1]?.last as { start_us: number; end_us: number };
    net.conns[1]?.handlers.onReply({
      type: 'done',
      stream: 8,
      start_us: range.start_us,
      end_us: range.end_us,
      frames: 0,
      skipped: 0,
      next_us: null,
      codecs: [],
    });
    v.fastSeek(1);
    await flush();
    expect(calls).toBe(2);
    expect(net.conns[1]?.last).toEqual({ type: 'frame', media_context: true, stream: 8, at_us: 21_000_000 });
    net.conns[0]?.handlers.onReply({ type: 'frame', stream: 8, at_us: 21_000_000, capture_us: 21_000_000 });
    expect(v.seeking).toBe(true);
    net.conns[1]?.handlers.onReply({ type: 'frame', stream: 8, at_us: 21_000_000, capture_us: 21_000_000 });
    expect(v.seeking).toBe(false);
    v.close();
  });

  it.each(['close', 'error'])('retries a pending scrub frame after gateway %s', async kind => {
    vi.useFakeTimers();
    try {
      const { net, v, events } = rig();
      v.src = gatewaySegmentUrl(7, 0, 60_000_000);
      await vi.advanceTimersByTimeAsync(0);
      v.fastSeek(20);
      await vi.advanceTimersByTimeAsync(0);
      const connections = net.conns.length;
      if (kind === 'close') net.conns.at(-1)?.handlers.onClose('gone');
      else net.conns.at(-1)?.handlers.onReply({ type: 'error', message: 'temporarily unavailable' });
      await vi.advanceTimersByTimeAsync(500);
      const connection = net.conns.at(-1);
      expect(net.conns).toHaveLength(connections + 1);
      expect(connection?.sent.filter(r => r.type === 'frame')).toHaveLength(1);
      expect(connection?.last).toEqual({ type: 'frame', media_context: true, stream: 7, at_us: 20_000_000 });
      connection?.handlers.onReply({ type: 'frame', stream: 7, at_us: 20_000_000, capture_us: 19_900_000 });
      expect(v.seeking).toBe(false);
      expect(events).toContain('seeked');
      v.close();
    } finally {
      vi.useRealTimers();
    }
  });

  it('takes a segment: its length at once, then the first window from the gateway', async () => {
    const { net, v, events } = rig();
    v.src = gatewaySegmentUrl(7, 5_000_000, 15_000_000);
    expect(v.duration).toBe(10);
    expect(v.readyState).toBe(1);
    await flush();
    expect(events).toEqual(['loadedmetadata']);
    expect(net.conns.at(-1)?.last).toEqual({
      type: 'range',
      media_context: true,
      stream: 7,
      start_us: 5_000_000,
      end_us: 7_000_000,
    });
    expect(v.getAttribute('src')).toBe('gateway:7:5000000:15000000');
  });

  it('seeks by asking for the window at the new time, and says seeked once it has come', async () => {
    const { net, worker, v, events, done } = rig();
    v.src = gatewaySegmentUrl(7, 5_000_000, 15_000_000);
    await flush();
    done();
    v.currentTime = 6;
    expect(v.seeking).toBe(true);
    await flush();
    // half a second around the point (the frame showing there is a little before it), not a whole playing window
    expect(net.conns.at(-1)?.last).toEqual({
      type: 'range',
      media_context: true,
      stream: 7,
      start_us: 10_750_000,
      end_us: 11_250_000,
    });
    expect(worker.heads.at(-1)).toBe(11_000_000);
    done();
    expect(v.seeking).toBe(false);
    expect(events.filter(e => e === 'seeked')).toHaveLength(1); // a segment's first window isn't a seek
    // A seek inside the window already loaded needs nothing new.
    const sent = net.conns.at(-1)?.sent.length;
    const resets = worker.posted.filter(m => m.type === 'reset').length;
    const connections = net.conns.length;
    v.currentTime = 6.1;
    await flush();
    expect(net.conns.at(-1)?.sent.length).toBe(sent);
    expect(net.conns).toHaveLength(connections);
    expect(worker.posted.filter(m => m.type === 'reset')).toHaveLength(resets);
    expect(events.at(-1)).toBe('seeked');
  });

  it('scrubs by asking for the one frame showing at a time, and says seeked when it comes', async () => {
    const { net, worker, v, events } = rig();
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    v.fastSeek(20);
    await flush();
    expect(net.conns.at(-1)?.last).toEqual({ type: 'frame', media_context: true, stream: 7, at_us: 20_000_000 });
    expect(worker.heads.at(-1)).toBe(20_000_000);
    const frame = (us: number) => {
      const b = new Uint8Array(HEADER_LEN + 2);
      b.set(encodeHeader({ streamId: 7, payloadType: PayloadType.jpeg, keyframe: true, live: false, captureUs: us }));
      return b.buffer;
    };
    const before = worker.posted.filter(m => m.type === 'message').length;
    net.conns.at(-1)?.handlers.onBinary(frame(1_500_000)); // the first window's, late: not wanted
    net.conns.at(-1)?.handlers.onBinary(frame(19_980_000)); // the scrub position's
    expect(worker.posted.filter(m => m.type === 'message').length - before).toBe(1);
    net.conns.at(-1)?.handlers.onReply({ type: 'frame', stream: 7, at_us: 20_000_000, capture_us: 19_980_000 });
    expect(v.seeking).toBe(false);
    expect(events.at(-1)).toBe('seeked');
    // An answer for a position since passed changes nothing.
    v.fastSeek(30);
    net.conns.at(-1)?.handlers.onReply({ type: 'frame', stream: 7, at_us: 20_000_000, capture_us: 19_980_000 });
    expect(v.seeking).toBe(true);
  });

  it('asks for whole microseconds, as the gateway takes them, however the position falls', async () => {
    const { net, v } = rig();
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    v.fastSeek(20.0000017); // a timeline position in seconds, a fraction of a microsecond off
    await flush();
    expect(net.conns.at(-1)?.last).toEqual({ type: 'frame', media_context: true, stream: 7, at_us: 20_000_002 });
    v.currentTime = 30.3333333;
    await flush();
    const r = net.conns.at(-1)?.last as { start_us: number; end_us: number };
    expect([Number.isInteger(r.start_us), Number.isInteger(r.end_us)]).toEqual([true, true]);
  });

  it('a timeline drag replaces the connection once, at its end, not once a step', async () => {
    let wall = 0;
    const { net, v } = rig(() => wall);
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    v.fastSeek(10); // the drag's first step goes out at once on the open connection
    await flush();
    const before = net.conns.length;
    vi.useFakeTimers();
    try {
      for (let i = 1; i <= 100; i++) {
        wall += 10; // a step every 10 ms, its frame still out
        v.fastSeek(10 + i * 0.1);
      }
      expect(net.conns.length).toBe(before); // nothing opened mid-drag
      wall += 100;
      await vi.advanceTimersByTimeAsync(100); // the drag holds still: its last position is asked for
    } finally {
      vi.useRealTimers();
    }
    await flush();
    expect(net.conns.length - before).toBe(1);
    expect(net.conns.at(-1)?.last).toEqual({ type: 'frame', media_context: true, stream: 7, at_us: 20_000_000 });
  });

  it("a reply to a request a waiting drag seek superseded doesn't complete that seek", async () => {
    let wall = 0;
    const { net, v, events } = rig(() => wall);
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    v.fastSeek(10);
    await flush();
    const scrub = net.conns.at(-1);
    vi.useFakeTimers();
    try {
      wall += 10;
      v.currentTime = 30; // mid-drag: waits for the drag to settle while the 10 s frame is still out
      const seekedBefore = events.filter(e => e === 'seeked').length;
      scrub?.handlers.onReply({ type: 'frame', stream: 7, at_us: 10_000_000, capture_us: 9_980_000 });
      expect(v.seeking).toBe(true);
      expect(events.filter(e => e === 'seeked')).toHaveLength(seekedBefore);
      wall += 100;
      await vi.advanceTimersByTimeAsync(100);
    } finally {
      vi.useRealTimers();
    }
    await flush();
    const r = net.conns.at(-1)?.last as { type: string; start_us: number };
    expect([r.type, r.start_us]).toEqual(['range', 29_750_000]); // the waiting seek went out, and is still seeking
    expect(v.seeking).toBe(true);
  });

  describe('on a gateway with request ids (#2384)', () => {
    const REQ = ['watch', 'req'];
    const tagged = (stream: number, us: number, req: number) =>
      encodeHeader({ streamId: stream, payloadType: PayloadType.jpeg, keyframe: true, live: false, captureUs: us, req })
        .buffer as ArrayBuffer;

    it('a 100-step drag keeps one connection and shows the newest position about 8 times a second', async () => {
      let wall = 0;
      const { net, v } = rig(() => wall, REQ);
      v.src = gatewaySegmentUrl(7, 0, 60_000_000);
      await flush();
      const conn = net.conns.at(-1);
      vi.useFakeTimers();
      try {
        for (let i = 0; i < 100; i++) {
          v.fastSeek(10 + i * 0.1); // a step every 10 ms for a second
          await vi.advanceTimersByTimeAsync(10);
          wall += 10;
        }
        await vi.advanceTimersByTimeAsync(200);
      } finally {
        vi.useRealTimers();
      }
      await flush();
      expect(net.conns).toHaveLength(1);
      expect(conn?.closed).toBe(false);
      const frames = (conn?.sent ?? []).filter(r => r.type === 'frame');
      // 1000 ms of drag at one position per 120 ms: 8 or 9, then the last position once the drag stops.
      expect(frames.length).toBeGreaterThanOrEqual(8);
      expect(frames.length).toBeLessThanOrEqual(10);
      expect(frames.at(-1)?.at_us).toBe(19_900_000);
      const reqs = frames.map(r => r.req as number);
      expect(new Set(reqs).size).toBe(reqs.length); // each its own id
      // The segment's first window, then every position but the last, was cancelled when the next went out, on the
      // same connection.
      const window = (conn?.sent ?? []).find(r => r.type === 'range')?.req;
      expect((conn?.sent ?? []).filter(r => r.type === 'cancel').map(r => r.req)).toEqual([
        window,
        ...reqs.slice(0, -1),
      ]);
    });

    it("ignores a cancelled request's frames and replies, and settles on the newest one's", async () => {
      const { net, worker, v, events } = rig(undefined, REQ);
      v.src = gatewaySegmentUrl(7, 0, 60_000_000);
      await flush();
      v.fastSeek(10);
      await flush();
      const first = net.conns.at(-1)?.last as { req: number };
      v.fastSeek(20); // a lone seek a second later: the first is cancelled, not the connection
      await flush();
      const second = net.conns.at(-1)?.last as { req: number; type: string };
      expect([second.type, second.req > first.req, net.conns.length]).toEqual(['frame', true, 1]);
      const before = worker.posted.filter(m => m.type === 'message').length;
      const seeked = events.filter(e => e === 'seeked').length;
      net.conns.at(-1)?.handlers.onBinary(tagged(7, 9_990_000, first.req));
      net.conns.at(-1)?.handlers.onReply({
        type: 'frame',
        stream: 7,
        at_us: 10_000_000,
        capture_us: 9_990_000,
        req: first.req,
      } as never);
      expect(worker.posted.filter(m => m.type === 'message').length).toBe(before);
      expect(events.filter(e => e === 'seeked')).toHaveLength(seeked);
      net.conns.at(-1)?.handlers.onBinary(tagged(7, 19_990_000, second.req));
      net.conns.at(-1)?.handlers.onReply({
        type: 'frame',
        stream: 7,
        at_us: 20_000_000,
        capture_us: 19_990_000,
        req: second.req,
      } as never);
      expect(worker.posted.filter(m => m.type === 'message').length).toBe(before + 1);
      expect(events.at(-1)).toBe('seeked');
    });
  });

  it('a lone seek after a pause goes out at once', async () => {
    let wall = 0;
    const { net, v } = rig(() => wall);
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    v.fastSeek(10);
    await flush();
    wall += 1000;
    v.fastSeek(30);
    await flush();
    expect(net.conns.at(-1)?.last).toEqual({ type: 'frame', media_context: true, stream: 7, at_us: 30_000_000 });
  });

  it("doesn't decode the frames of a window a later seek gave up on", async () => {
    const { net, worker, v, done } = rig();
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    done();
    v.currentTime = 5; // asks for 4.75 to 5.25 s
    await flush();
    v.currentTime = 40; // scrubbed on before it came: asks for 39.75 to 40.25 s
    await flush();
    const frame = (us: number) => {
      const b = new Uint8Array(HEADER_LEN + 2);
      b.set(encodeHeader({ streamId: 7, payloadType: PayloadType.jpeg, keyframe: true, live: false, captureUs: us }));
      return b.buffer;
    };
    const before = worker.posted.filter(m => m.type === 'message').length;
    net.conns.at(-1)?.handlers.onBinary(frame(5_000_000)); // the first seek's frame, arriving late
    net.conns.at(-1)?.handlers.onBinary(frame(39_900_000)); // the wanted one
    expect(worker.posted.filter(m => m.type === 'message').length - before).toBe(1);
  });

  it('decodes the tail of a GOP that runs past its window, the next window starting after it (#917)', async () => {
    const { net, worker, v } = rig();
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    v.currentTime = 5; // asks for 4.75 to 5.25 s
    await flush();
    const frame = (us: number) => {
      const b = new Uint8Array(HEADER_LEN + 2);
      b.set(encodeHeader({ streamId: 7, payloadType: PayloadType.jpeg, keyframe: true, live: false, captureUs: us }));
      return b.buffer;
    };
    const before = worker.posted.filter(m => m.type === 'message').length;
    net.conns.at(-1)?.handlers.onBinary(frame(5_000_000)); // in the window
    net.conns.at(-1)?.handlers.onBinary(frame(6_000_000)); // the same GOP, past the window's end
    expect(worker.posted.filter(m => m.type === 'message').length - before).toBe(2);
  });

  it('continues after the complete GOP tail using its maximum PTS, while seeks request their prefix', async () => {
    const { net, time, v, done } = rig();
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    for (const us of [1_900_000, 3_500_000, 3_100_000]) {
      net.conns.at(-1)?.handlers.onBinary(
        encodeHeader({
          streamId: 7,
          payloadType: PayloadType.video,
          keyframe: us === 1_900_000,
          live: false,
          captureUs: us,
        }).slice().buffer,
      );
    }
    done();
    await v.play();
    time.step(0);
    time.step(1100);
    await flush();
    expect(net.conns.at(-1)?.last).toMatchObject({
      type: 'range',
      start_us: 3_500_001,
      end_us: 5_500_001,
      after_us: 3_500_000,
    });
    done();
    v.currentTime = 40;
    await flush();
    expect(net.conns.at(-1)?.last).toEqual({
      type: 'range',
      stream: 7,
      start_us: 39_750_000,
      end_us: 40_250_000,
      media_context: true,
    });
    v.close();
  });

  it('does not commit abandoned 8-second packets into a backward 1-second seek', async () => {
    const { net, worker, v, time, done } = rig();
    const packet = (us: number) => {
      const b = new Uint8Array(HEADER_LEN + 2);
      b.set(encodeHeader({ streamId: 7, payloadType: PayloadType.video, keyframe: true, live: false, captureUs: us }));
      return b.buffer;
    };
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    done();
    v.currentTime = 8;
    await flush();
    const abandoned = net.conns.at(-1);
    if (!abandoned) throw new Error('missing first seek connection');
    const oldRequest = abandoned.last as { start_us: number; end_us: number };
    abandoned.handlers.onBinary(packet(8_000_000));
    const oldMessage = worker.posted.findLast(m => m.type === 'message');
    v.currentTime = 1;
    await flush();
    const active = net.conns.at(-1);
    if (!active) throw new Error('missing active seek connection');
    expect(abandoned.closed).toBe(true);
    expect(active).not.toBe(abandoned);
    const bitmap = { width: 640, height: 360, close: vi.fn() } as unknown as ImageBitmap;
    const width = v.canvas.width;
    worker.onmessage?.({
      data: { type: 'frame', bitmap, epoch: oldMessage?.epoch, frameUs: 8_000_000 },
    } as MessageEvent);
    expect(v.canvas.width).toBe(width);
    expect(bitmap.close).toHaveBeenCalledOnce();
    const before = worker.posted.filter(m => m.type === 'message').length;
    abandoned.handlers.onBinary(packet(8_000_000));
    abandoned.handlers.onReply({
      type: 'done',
      stream: 7,
      start_us: oldRequest.start_us,
      end_us: oldRequest.end_us,
      frames: 1,
      skipped: 0,
      next_us: null,
      codecs: ['avc'],
    });
    expect(worker.posted.filter(m => m.type === 'message')).toHaveLength(before);
    active.handlers.onBinary(packet(1_000_000));
    const request = active.last as { start_us: number; end_us: number };
    active.handlers.onReply({
      type: 'done',
      stream: 7,
      start_us: request.start_us,
      end_us: request.end_us,
      frames: 1,
      skipped: 0,
      next_us: null,
      codecs: ['avc'],
    });
    const accepted = worker.posted.filter(m => m.type === 'message').length;
    abandoned.handlers.onClose('late close');
    abandoned.handlers.onBinary(packet(8_000_000));
    expect(worker.posted.filter(m => m.type === 'message')).toHaveLength(accepted);
    await v.play();
    time.step(0);
    await flush();
    expect(active.last).toMatchObject({ type: 'range', start_us: 1_250_000, after_us: 1_000_000 });
    v.close();
  });

  it('invalidates connections still opening during rapid backward seeks', async () => {
    const { net, v } = rig();
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    v.currentTime = 8;
    v.currentTime = 1;
    await flush();
    const active = net.conns.at(-1);
    if (!active) throw new Error('missing active seek connection');
    expect(net.conns).toHaveLength(3);
    expect(net.conns.slice(0, -1).every(c => c.closed && c.sent.length === 0)).toBe(true);
    expect(active.sent).toEqual([
      { type: 'range', stream: 7, start_us: 750_000, end_us: 1_250_000, media_context: true },
    ]);
    v.close();
  });

  it('retries with the committed frontier rather than a partially received next GOP', async () => {
    const { net, time, v, done } = rig();
    const packet = (us: number) =>
      encodeHeader({
        streamId: 7,
        payloadType: PayloadType.video,
        keyframe: true,
        live: false,
        captureUs: us,
      }).slice().buffer;
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    net.conns.at(-1)?.handlers.onBinary(packet(3_500_000));
    done();
    await v.play();
    time.step(0);
    time.step(1100);
    await flush();
    net.conns.at(-1)?.handlers.onBinary(packet(5_000_000)); // no done: this GOP is not committed
    vi.useFakeTimers();
    try {
      net.conns.at(-1)?.handlers.onReply({ type: 'error', message: 'temporarily unavailable' });
      await vi.advanceTimersByTimeAsync(500);
      expect(net.conns.at(-1)?.last).toMatchObject({ start_us: 3_500_001, after_us: 3_500_000 });
      done(4_000_000); // a capped page with no new frames still advances
      time.step(4000);
      await vi.advanceTimersByTimeAsync(0);
      expect(net.conns.at(-1)?.last).toMatchObject({ start_us: 4_000_000, after_us: 3_500_000 });
    } finally {
      v.close();
      vi.useRealTimers();
    }
  });

  it('does not continue beyond the segment when a GOP tail already covers its end', async () => {
    const { net, time, v, done } = rig();
    v.src = gatewaySegmentUrl(7, 0, 3_000_000);
    await flush();
    net.conns.at(-1)?.handlers.onBinary(
      encodeHeader({
        streamId: 7,
        payloadType: PayloadType.video,
        keyframe: true,
        live: false,
        captureUs: 3_100_000,
      }).slice().buffer,
    );
    done();
    await v.play();
    time.step(0);
    time.step(1100);
    await flush();
    expect(net.conns.at(-1)?.sent).toHaveLength(1);
    v.close();
  });

  it('plays at its rate, asks for the next window before this one runs out, and ends at the segment end', async () => {
    const { net, worker, time, v, events, done } = rig();
    v.src = gatewaySegmentUrl(7, 0, 3_000_000);
    await flush();
    done(2_000_000);
    v.playbackRate = 2;
    await v.play();
    expect(events.slice(-2)).toEqual(['play', 'playing']);
    time.step(0);
    time.step(250); // 0.5 s at double speed
    expect(v.currentTime).toBeCloseTo(0.5);
    expect(worker.heads.at(-1)).toBe(500_000);
    time.step(300); // 1.1 s: within a second of the window's end
    await flush();
    expect(net.conns.at(-1)?.last).toEqual({
      type: 'range',
      media_context: true,
      stream: 7,
      start_us: 2_000_000,
      end_us: 3_000_000,
    });
    done();
    time.step(1000); // past the end
    expect(v.currentTime).toBe(3);
    expect(v.ended && v.paused).toBe(true);
    expect(events.slice(-2)).toEqual(['pause', 'ended']);
  });

  it('pauses, and lets go of its connection when its src is removed', async () => {
    const { net, v, events } = rig();
    v.src = gatewaySegmentUrl(7, 0, 3_000_000);
    await flush();
    await v.play();
    v.pause();
    expect(events.at(-1)).toBe('pause');
    v.removeAttribute('src');
    v.load();
    expect(net.conns.at(-1)?.closed).toBe(true);
    expect(events.at(-1)).toBe('emptied');
    expect(v.getAttribute('src')).toBeNull();
  });

  it('asks again after a gateway error or a dropped connection, never calling the segment broken', async () => {
    vi.useFakeTimers();
    try {
      const { net, v, events } = rig();
      v.src = gatewaySegmentUrl(7, 0, 3_000_000);
      await vi.advanceTimersByTimeAsync(0);
      net.conns.at(-1)?.handlers.onReply({ type: 'error', message: 'range: the broker: unreachable' });
      await vi.advanceTimersByTimeAsync(500);
      expect(net.conns.at(-1)?.sent.filter(r => r.type === 'range')).toHaveLength(2);
      net.conns.at(-1)?.handlers.onClose('the gateway restarted');
      await vi.advanceTimersByTimeAsync(1000); // the wait doubled
      expect(net.conns[1]?.last).toEqual({
        type: 'range',
        media_context: true,
        stream: 7,
        start_us: 0,
        end_us: 2_000_000,
      });
      expect(events).not.toContain('error');
      expect(v.error).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('reports a URL that is not a segment as an error', async () => {
    const { v, events } = rig();
    v.src = 'x.mp4';
    await flush();
    expect(events).toContain('error');
  });

  it("says the picture is unplayable while a window holds only codecs the browser can't decode", async () => {
    const { net, worker, v, done } = rig();
    v.src = gatewaySegmentUrl(7, 0, 3_000_000);
    await flush();
    const answer = (codecs: string[]) => {
      const r = net.conns.at(-1)?.last as { start_us: number; end_us: number };
      net.conns.at(-1)?.handlers.onReply({
        type: 'done',
        stream: 7,
        start_us: r.start_us,
        end_us: r.end_us,
        frames: 1,
        skipped: 0,
        next_us: null,
        codecs,
      });
    };
    v.currentTime = 0.1; // the worker exists once a seek has posted a playhead
    await flush();
    worker.onmessage?.({ data: { type: 'codec', codec: 'av1', supported: false } } as MessageEvent);
    await flush();
    answer(['av1']); // archived AV1 (#1479) in a browser without an AV1 decoder
    expect([v.unplayable, v.unplayableCodec]).toEqual([true, 'av1']);
    v.currentTime = 2.5; // a window of JPEG frames
    await flush();
    answer(['jpeg']);
    expect([v.unplayable, v.unplayableCodec]).toEqual([false, null]);
    void done;
  });

  it('clears an unsupported AVC profile when the latest AVC configuration is supported', async () => {
    const { net, worker, v } = rig();
    const changes: boolean[] = [];
    v.addEventListener('unplayable', () => changes.push(v.unplayable));
    v.src = gatewaySegmentUrl(7, 0, 3_000_000);
    v.currentTime = 0.1;
    await flush();
    worker.onmessage?.({ data: { type: 'codec', codec: 'avc', supported: false } } as MessageEvent);
    const request = net.conns.at(-1)?.last as { start_us: number; end_us: number };
    net.conns.at(-1)?.handlers.onReply({
      type: 'done',
      stream: 7,
      start_us: request.start_us,
      end_us: request.end_us,
      frames: 1,
      skipped: 0,
      next_us: null,
      codecs: ['avc'],
    });
    expect(v.unplayable).toBe(true);
    worker.onmessage?.({ data: { type: 'codec', codec: 'avc', supported: true } } as MessageEvent);
    expect(v.unplayable).toBe(false);
    expect(changes).toEqual([true, false]);
    v.close();
    worker.onmessage?.({ data: { type: 'codec', codec: 'avc', supported: false } } as MessageEvent);
    expect(v.unplayable).toBe(false);
  });

  it('keeps the standby attribute on its canvas, where the page hides it', () => {
    const { v } = rig();
    v.toggleAttribute('data-standby', true);
    expect(v.canvas.hasAttribute('data-standby')).toBe(true);
  });
});

describe('a stretch still being recorded (#806)', () => {
  it('grows when told, plays on into the new part, and waits at the live edge instead of ending', async () => {
    const { net, time, v, events, done } = rig();
    v.src = gatewaySegmentUrl(7, 0, 2_000_000);
    await flush();
    done(null);
    await v.play();
    time.step(0);
    v.extend(3); // listed again: a second longer
    expect(v.duration).toBe(3);
    time.step(2500); // past the old end
    expect(v.currentTime).toBeCloseTo(2.5);
    expect(v.ended).toBe(false);
    await flush();
    expect(net.conns.at(-1)?.last).toEqual({
      type: 'range',
      media_context: true,
      stream: 7,
      start_us: 2_000_000,
      end_us: 3_000_000,
    });
    done();
    time.step(1000); // at the live edge: holds there
    expect(v.currentTime).toBe(3);
    expect(v.ended).toBe(false);
    expect(events).not.toContain('ended');
    time.step(20_000); // nothing grew for long: it ends after all
    time.step(16);
    expect(v.ended).toBe(true);
  });
});

it('reuses a DOM canvas across player disposal without transferring it twice', () => {
  const canvas = document.createElement('canvas');
  let transferred = false;
  Object.defineProperty(canvas, 'transferControlToOffscreen', {
    value: () => {
      if (transferred) throw new DOMException('canvas already transferred', 'InvalidStateError');
      transferred = true;
      return {} as OffscreenCanvas;
    },
  });
  const net = fakeGateway();
  const deps = { connect: net.connect, makeWorker: () => new FakeWorker() };
  const first = new GatewayVideo(canvas, deps);
  first.src = gatewaySegmentUrl(7, 0, 1_000_000);
  first.fastSeek(0);
  first.close();
  const second = new GatewayVideo(canvas, deps);
  second.src = gatewaySegmentUrl(7, 0, 1_000_000);
  expect(() => second.fastSeek(0)).not.toThrow();
  second.close();
});

it('ignores late video frames after close instead of recreating a worker on the old canvas', async () => {
  const net = fakeGateway();
  const makeWorker = vi.fn(() => new FakeWorker());
  const video = new GatewayVideo(document.createElement('canvas'), { connect: net.connect, makeWorker });
  video.src = gatewaySegmentUrl(7, 0, 1_000_000);
  video.fastSeek(0);
  await flush();
  video.close();
  net.conns.at(-1)?.handlers.onBinary(new ArrayBuffer(0));
  expect(makeWorker).toHaveBeenCalledTimes(1);
});

it('presents returned bitmaps and closes frames that arrive after disposal', () => {
  const canvas = document.createElement('canvas');
  const paint = vi.fn();
  vi.spyOn(canvas, 'getContext').mockReturnValue({ drawImage: paint } as unknown as CanvasRenderingContext2D);
  const worker = new FakeWorker();
  const video = new GatewayVideo(canvas, { makeWorker: () => worker, connect: fakeGateway().connect });
  video.src = gatewaySegmentUrl(7, 0, 1_000_000);
  video.fastSeek(0);
  const bitmap = { width: 64, height: 48, close: vi.fn() } as unknown as ImageBitmap;
  worker.onmessage?.({ data: { type: 'frame', bitmap, frameUs: 500_000 } } as MessageEvent);
  expect(paint).toHaveBeenCalledWith(bitmap, 0, 0);
  expect(bitmap.close).toHaveBeenCalledOnce();
  expect([canvas.width, canvas.height]).toEqual([64, 48]);
  expect(canvas.dataset.capturedAt).toBe('0.5');
  video.close();
  const stale = { width: 64, height: 48, close: vi.fn() } as unknown as ImageBitmap;
  worker.onmessage?.({ data: { type: 'frame', bitmap: stale } } as MessageEvent);
  expect(paint).toHaveBeenCalledOnce();
  expect(stale.close).toHaveBeenCalledOnce();
});

describe('native recorded frame capture', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  function captureRig() {
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
    const net = fakeGateway();
    const worker = new FakeWorker();
    const canvas = document.createElement('canvas');
    canvas.style.width = '160px';
    const draw = vi.fn();
    vi.spyOn(canvas, 'getContext').mockReturnValue({ drawImage: draw } as unknown as CanvasRenderingContext2D);
    const encode = vi.spyOn(canvas, 'toDataURL').mockReturnValue('data:image/jpeg;base64,Y2FwdHVyZQ==');
    const capture = createRecordedFrameCapture({
      connect: net.connect,
      makeWorker: () => worker,
      makeCanvas: () => canvas,
      timeoutMs: 1000,
    });
    const emit = (data: FromWorker) => worker.onmessage?.({ data } as MessageEvent<FromWorker>);
    const frame = (us: number, width = 4096, height = 2160) => {
      const close = vi.fn();
      const bitmap = { width, height, close } as unknown as ImageBitmap;
      emit({ type: 'frame', bitmap, frameUs: us });
      emit({ type: 'drawn', frameUs: us, overlay: 'none', overlayUs: null, matched: false });
      return { bitmap, close };
    };
    const ack = (captureUs: number | null) =>
      net.conns[0]?.handlers.onReply({
        type: 'frame',
        stream: 7,
        at_us: 10_000_000,
        capture_us: captureUs,
      });
    return { net, worker, canvas, draw, encode, capture, emit, frame, ack };
  }

  it('requests only the exact frame, encodes its native pixels, and returns the actual decoded timestamp', async () => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await flush();
    expect(r.net.conns[0]?.sent).toEqual([{ type: 'frame', media_context: true, stream: 7, at_us: 10_000_000 }]);
    const settled = vi.fn();
    void promise.then(settled);
    const stale = r.frame(5_000_000);
    r.ack(9_980_000);
    await Promise.resolve();
    expect(settled).not.toHaveBeenCalled();
    expect(r.encode).not.toHaveBeenCalled();
    const actual = r.frame(9_980_000);
    await expect(promise).resolves.toEqual({ image: 'data:image/jpeg;base64,Y2FwdHVyZQ==', captured_at: 9.98 });
    expect([r.canvas.width, r.canvas.height]).toEqual([4096, 2160]);
    expect(r.draw).toHaveBeenCalledExactlyOnceWith(actual.bitmap, 0, 0);
    expect(r.encode).toHaveBeenCalledExactlyOnceWith('image/jpeg', 0.9);
    expect(stale.close).toHaveBeenCalledOnce();
    expect(actual.close).toHaveBeenCalledOnce();
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
  });

  it('captures the non-keyframe at the playhead only after the complete GOP has decoded', async () => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    const settled = vi.fn();
    void promise.then(settled);
    await flush();
    const video = (captureUs: number, keyframe = false) =>
      encodeHeader({
        streamId: 7,
        payloadType: PayloadType.video,
        captureUs,
        keyframe,
        live: false,
      }).buffer as ArrayBuffer;
    r.net.conns[0]?.handlers.onBinary(video(9_000_000, true));
    r.frame(9_000_000);
    r.ack(9_000_000);
    await Promise.resolve();
    expect(settled).not.toHaveBeenCalled();
    expect(r.encode).not.toHaveBeenCalled();
    expect(r.net.conns[0]?.last).toEqual({
      type: 'range',
      media_context: true,
      stream: 7,
      start_us: 9_000_000,
      end_us: 10_000_001,
      max_frames: 600,
      overlays: [],
    });
    r.net.conns[0]?.handlers.onBinary(video(9_000_000, true));
    r.net.conns[0]?.handlers.onBinary(video(9_990_000));
    r.net.conns[0]?.handlers.onBinary(video(10_020_000)); // future reference needed for reordered B frames
    expect(r.worker.posted.filter(m => m.type === 'message')).toHaveLength(3);
    r.net.conns[0]?.handlers.onReply({
      type: 'done',
      stream: 7,
      start_us: 9_000_000,
      end_us: 10_000_001,
      frames: 3,
      skipped: 0,
      next_us: null,
      codecs: ['avc'],
    });
    expect(r.worker.posted.at(-1)).toEqual({ type: 'capture', requestId: 1, us: 10_000_000 });
    const selected = r.frame(9_990_000);
    expect(r.encode).not.toHaveBeenCalled();
    r.emit({ type: 'captured', requestId: 99, frameUs: 9_990_000 });
    expect(r.encode).not.toHaveBeenCalled();
    r.emit({ type: 'captured', requestId: 1, frameUs: 9_990_000 });
    await expect(promise).resolves.toEqual({ image: 'data:image/jpeg;base64,Y2FwdHVyZQ==', captured_at: 9.99 });
    expect(r.draw).toHaveBeenCalledExactlyOnceWith(selected.bitmap, 0, 0);
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
  });

  it.each([
    { next_us: 9_500_000, skipped: 0, frames: 1 },
    { next_us: null, skipped: 1, frames: 1 },
    { next_us: null, skipped: 0, frames: 2 },
  ])('rejects incomplete compressed ranges: $next_us/$skipped/$frames', async status => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await flush();
    const video = encodeHeader({
      streamId: 7,
      payloadType: PayloadType.video,
      captureUs: 9_000_000,
      keyframe: true,
      live: false,
    }).buffer as ArrayBuffer;
    r.net.conns[0]?.handlers.onBinary(video);
    r.ack(9_000_000);
    r.net.conns[0]?.handlers.onBinary(video);
    r.net.conns[0]?.handlers.onReply({
      type: 'done',
      stream: 7,
      start_us: 9_000_000,
      end_us: 10_000_001,
      codecs: ['avc'],
      ...status,
    });
    await expect(promise).rejects.toThrow('incomplete');
    expect(r.worker.posted.some(m => m.type === 'capture')).toBe(false);
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
  });

  it('bounds all compressed bytes before handing them to the decoder', async () => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await flush();
    r.net.conns[0]?.handlers.onBinary(new ArrayBuffer(64 * 1024 * 1024 + 1));
    await expect(promise).rejects.toThrow('byte limit');
    expect(r.worker.posted.filter(m => m.type === 'message')).toHaveLength(0);
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
  });

  it('rejects a keyframe beyond the bounded decode lookback', async () => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 0, 30_000_000), 20);
    await flush();
    r.net.conns[0]?.handlers.onBinary(
      encodeHeader({
        streamId: 7,
        payloadType: PayloadType.video,
        captureUs: 4_000_000,
        keyframe: true,
        live: false,
      }).buffer as ArrayBuffer,
    );
    r.net.conns[0]?.handlers.onReply({ type: 'frame', stream: 7, at_us: 20_000_000, capture_us: 4_000_000 });
    await expect(promise).rejects.toThrow('lookback limit');
    expect(r.net.conns[0]?.sent).toHaveLength(1);
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
  });

  it('keeps the original deadline while compressed decoding waits for flush output', async () => {
    vi.useFakeTimers();
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    const outcome = expect(promise).rejects.toMatchObject({ name: 'TimeoutError' });
    await Promise.resolve();
    const video = encodeHeader({
      streamId: 7,
      payloadType: PayloadType.video,
      captureUs: 9_000_000,
      keyframe: true,
      live: false,
    }).buffer as ArrayBuffer;
    r.net.conns[0]?.handlers.onBinary(video);
    r.ack(9_000_000);
    r.net.conns[0]?.handlers.onBinary(video);
    r.net.conns[0]?.handlers.onReply({
      type: 'done',
      stream: 7,
      start_us: 9_000_000,
      end_us: 10_000_001,
      frames: 1,
      skipped: 0,
      next_us: null,
      codecs: ['avc'],
    });
    expect(r.worker.posted.at(-1)).toEqual({ type: 'capture', requestId: 1, us: 10_000_000 });
    await vi.advanceTimersByTimeAsync(1000);
    await outcome;
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('waits for the matching gateway acknowledgment when decoding finishes first', async () => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await flush();
    r.frame(9_980_000);
    expect(r.encode).not.toHaveBeenCalled();
    r.net.conns[0]?.handlers.onReply({ type: 'frame', stream: 7, at_us: 6_000_000, capture_us: 9_980_000 });
    expect(r.encode).not.toHaveBeenCalled();
    r.ack(9_980_000);
    expect((await promise).captured_at).toBe(9.98);
  });

  it('drops overlays and unrelated media before the clean decoder sees them', async () => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await flush();
    const binary = (payloadType: (typeof PayloadType)[keyof typeof PayloadType], streamId = 7) =>
      encodeHeader({
        streamId,
        payloadType,
        captureUs: 9_980_000,
        keyframe: true,
        live: false,
      }).buffer as ArrayBuffer;
    r.net.conns[0]?.handlers.onBinary(binary(PayloadType.overlay));
    r.net.conns[0]?.handlers.onBinary(binary(PayloadType.jpeg, 9));
    expect(r.worker.posted.filter(m => m.type === 'message')).toHaveLength(0);
    const media = binary(PayloadType.jpeg);
    r.net.conns[0]?.handlers.onBinary(media);
    expect(r.worker.posted.filter(m => m.type === 'message')).toEqual([{ type: 'message', buffer: media }]);
    r.ack(9_980_000);
    r.frame(9_980_000);
    await promise;
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, 4.9, 15])('rejects a playhead outside the segment: %s', async at => {
    const r = captureRig();
    await expect(r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), at)).rejects.toThrow('outside');
    expect(r.net.conns).toHaveLength(0);
    expect(r.worker.posted).toHaveLength(0);
  });

  it.each([
    [8193, 100],
    [8192, 8192],
    [0, 100],
  ])('rejects invalid native dimensions %s x %s', async (width, height) => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await flush();
    r.ack(9_980_000);
    const picture = r.frame(9_980_000, width, height);
    await expect(promise).rejects.toThrow('size limit');
    expect(r.draw).not.toHaveBeenCalled();
    expect(picture.close).toHaveBeenCalledOnce();
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
  });

  it.each(['data:,', `data:image/jpeg;base64,${'A'.repeat(16_000_000)}`])(
    'rejects an empty or oversized encoding',
    async image => {
      const r = captureRig();
      r.encode.mockReturnValue(image);
      const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
      await flush();
      r.ack(9_980_000);
      r.frame(9_980_000);
      await expect(promise).rejects.toThrow('size limit');
      expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
    },
  );

  it('rejects no-frame replies and releases the connection and decoder', async () => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await flush();
    r.ack(null);
    await expect(promise).rejects.toThrow('No recorded frame');
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
  });

  it.each(['gateway', 'decoder', 'unsupported', 'closed'])('cleans up a %s failure', async source => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await flush();
    if (source === 'gateway') r.net.conns[0]?.handlers.onReply({ type: 'error', message: 'unavailable' });
    else if (source === 'decoder') r.emit({ type: 'error', message: 'decode failed' });
    else if (source === 'unsupported') r.emit({ type: 'codec', codec: 'avc', supported: false });
    else r.net.conns[0]?.handlers.onClose('disconnected');
    await expect(promise).rejects.toMatchObject({ message: expect.any(String) });
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
  });

  it('bounds a missing decoded frame and clears its timer', async () => {
    vi.useFakeTimers();
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    const outcome = expect(promise).rejects.toMatchObject({ name: 'TimeoutError' });
    await vi.advanceTimersByTimeAsync(1000);
    await outcome;
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('aborts immediately and also closes a connection that opens after cancellation', async () => {
    const r = captureRig();
    let open!: (connection: Connection) => void;
    const close = vi.fn();
    const capture = createRecordedFrameCapture({
      makeWorker: () => r.worker,
      makeCanvas: () => r.canvas,
      connect: () =>
        new Promise(resolve => {
          open = resolve;
        }),
    });
    const controller = new AbortController();
    const promise = capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10, controller.signal);
    controller.abort();
    await expect(promise).rejects.toMatchObject({ name: 'AbortError' });
    expect(r.worker.terminated).toBe(true);
    open({ transport: 'websocket', send: vi.fn(), close });
    await Promise.resolve();
    expect(close).toHaveBeenCalledOnce();
  });

  it('rejects a gateway lookback that falls outside the selected recorded segment', async () => {
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await flush();
    const stale = r.frame(4_980_000);
    r.ack(4_980_000);
    await expect(promise).rejects.toThrow('invalid recorded frame time');
    expect(r.encode).not.toHaveBeenCalled();
    expect(stale.close).toHaveBeenCalledOnce();
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
  });

  it('releases a decoded bitmap and timer when cancelled while awaiting the acknowledgment', async () => {
    vi.useFakeTimers();
    const r = captureRig();
    const controller = new AbortController();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10, controller.signal);
    await Promise.resolve();
    const picture = r.frame(9_980_000);
    controller.abort();
    await expect(promise).rejects.toMatchObject({ name: 'AbortError' });
    expect(picture.close).toHaveBeenCalledOnce();
    expect(r.worker.terminated && r.net.conns[0]?.closed).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
    const lateClose = vi.fn();
    r.emit({ type: 'frame', bitmap: { close: lateClose } as unknown as ImageBitmap, frameUs: 1_000_000 });
    expect(lateClose).toHaveBeenCalledOnce();
  });

  it('cleans up its decoder and timer if the transport fails to open', async () => {
    vi.useFakeTimers();
    const r = captureRig();
    const capture = createRecordedFrameCapture({
      makeWorker: () => r.worker,
      makeCanvas: () => r.canvas,
      connect: async () => {
        throw new Error('gateway offline');
      },
    });
    await expect(capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10)).rejects.toThrow('gateway offline');
    expect(r.worker.terminated).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('releases its timer after successful capture', async () => {
    vi.useFakeTimers();
    const r = captureRig();
    const promise = r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10);
    await Promise.resolve();
    r.ack(9_980_000);
    r.frame(9_980_000);
    await promise;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('creates no decoder or connection when already cancelled', async () => {
    const r = captureRig();
    const controller = new AbortController();
    controller.abort();
    await expect(r.capture(gatewaySegmentUrl(7, 5_000_000, 15_000_000), 10, controller.signal)).rejects.toMatchObject({
      name: 'AbortError',
    });
    expect(r.net.conns).toHaveLength(0);
    expect(r.worker.posted).toHaveLength(0);
  });
});

describe('recorded frame contexts', () => {
  const context = (id: number) => ({
    type: 'media' as const,
    context_id: id,
    stream: 7,
    chunk_id: 'chunk',
    revision: 1,
    profile_status: 'absent' as const,
    native_geometry: { width: 640, height: 360 },
  });
  const packet = (id?: number) => {
    const h = encodeHeader({
      streamId: 7,
      payloadType: PayloadType.jpeg,
      keyframe: true,
      live: false,
      captureUs: 1_000_000,
      contextId: id,
    });
    const b = new Uint8Array(h.length + 2);
    b.set(h);
    return b.buffer;
  };
  it('holds a packet until its media context arrives, then sends that context once', async () => {
    const { net, worker, v } = rig();
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    v.fastSeek(1);
    await flush();
    const before = worker.posted.filter(m => m.type === 'message').length;
    net.conns.at(-1)?.handlers.onBinary(packet(9)); // binary can precede WT control
    expect(worker.posted.filter(m => m.type === 'message')).toHaveLength(before);
    net.conns.at(-1)?.handlers.onReply(context(9));
    expect(worker.posted.filter(m => m.type === 'message')).toHaveLength(before + 1);
    expect(worker.posted.findLast(m => m.type === 'media')).toMatchObject({ media: context(9) });
    net.conns.at(-1)?.handlers.onBinary(packet(9));
    expect(worker.posted.filter(m => m.type === 'media')).toHaveLength(1);
    expect(worker.posted.filter(m => m.type === 'message')).toHaveLength(before + 2);
    v.close();
  });
  it('clears confirmed missing pixels and refuses delayed output from the old seek', async () => {
    const { net, worker, v } = rig();
    const drawImage = vi.fn(),
      clearRect = vi.fn();
    vi.spyOn(v.canvas, 'getContext').mockReturnValue({ drawImage, clearRect } as unknown as CanvasRenderingContext2D);
    v.src = gatewaySegmentUrl(7, 0, 60_000_000);
    await flush();
    v.fastSeek(1);
    const bitmap = { width: 640, height: 360, close: vi.fn() } as unknown as ImageBitmap;
    worker.onmessage?.({ data: { type: 'frame', bitmap, frameUs: 1_000_000 } } as MessageEvent);
    expect(v.canvas.dataset.capturedAt).toBe('1');
    v.fastSeek(50);
    await flush();
    net.conns.at(-1)?.handlers.onReply({ type: 'frame', stream: 7, at_us: 50_000_000, capture_us: null });
    expect(clearRect).toHaveBeenLastCalledWith(0, 0, 640, 360);
    expect(v.canvas.dataset.capturedAt).toBeUndefined();
    worker.onmessage?.({ data: { type: 'frame', bitmap, frameUs: 1_000_000 } } as MessageEvent);
    expect(drawImage).toHaveBeenCalledOnce();
    expect(v.canvas.dataset.capturedAt).toBeUndefined();
    expect(bitmap.close).toHaveBeenCalledTimes(2);
    v.close();
  });

  it('reconnects if a media context never arrives instead of reusing a previous one', async () => {
    vi.useFakeTimers();
    try {
      const { net, worker, v } = rig();
      v.src = gatewaySegmentUrl(7, 0, 60_000_000);
      await vi.advanceTimersByTimeAsync(0);
      net.conns[0]?.handlers.onBinary(packet(90));
      await vi.advanceTimersByTimeAsync(10_000);
      expect(net.conns[0]?.closed).toBe(true);
      expect(net.conns).toHaveLength(2);
      expect(worker.posted.some(m => m.type === 'message')).toBe(false);
      expect(v.canvas.dataset.frameMetadata).toBeUndefined();
      v.close();
    } finally {
      vi.useRealTimers();
    }
  });
});
