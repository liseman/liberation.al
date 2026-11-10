import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LiveFrame } from './gateway';
import { fakeGateway } from './testFakes';

// gateway.ts keeps its sessions at module level, so each test loads a fresh copy.
let gw: typeof import('./gateway');
let net: ReturnType<typeof fakeGateway>;
const flush = () => vi.advanceTimersByTimeAsync(0); // a connection opens on a resolved promise

beforeEach(async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
  vi.resetModules();
  gw = await import('./gateway');
  net = fakeGateway();
  gw.setGatewayForTests(net.connect);
  wall = 10; // the wall clock holds are measured by, seconds: just after the overlays' capture times below
  gw.setClockForTests(() => wall);
});
let wall = 0;
afterEach(() => vi.useRealTimers());

const person = (x: number) => Array.from({ length: 26 }, (_, i) => (i === 5 ? [x, 0.3, 0.9] : [-1, -1, 0]));

it('a successful selected-face claim updates current subscribers before any next gateway frame', async () => {
  const got: LiveFrame[] = [];
  const off = gw.watchBoxes('cam0', frame => got.push(frame));
  await flush();
  net
    .ready('cam0', 7)
    .overlay(2_000_000, [{ xyxy: [0.1, 0.1, 0.2, 0.2], label: 'Face 5', kind: 'face', id: 'run:cam0:7' }]);
  const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({ name: 'Dana' }));
  try {
    const { claimLiveFace } = await import('../people/liveFace');
    await claimLiveFace(
      { source: { camera: 'cam0', at: 2, track: 'run:cam0:7', group: 5 }, image: 'selected crop' },
      'dana',
    );
    expect(got.at(-1)?.faces?.map(face => face.name)).toEqual(['Dana']);
    expect(got).toHaveLength(2);
  } finally {
    fetch.mockRestore();
    off();
  }
});

describe('confirmed face names', () => {
  const source = { camera: 'cam0', at: 2, track: 'run:cam0:7', group: 5 };
  const face = (track = source.track, label = 'Face 5') => ({
    xyxy: [0.1, 0.1, 0.2, 0.2] as [number, number, number, number],
    label,
    kind: 'face' as const,
    id: track,
  });

  it('reads existing held boxes without opening sessions, and never returns expired faces', async () => {
    expect(gw.currentBoxes('cam0')).toBeNull();
    expect(net.conns).toHaveLength(0);
    const off = gw.watchBoxes('cam0', () => {});
    await flush();
    net.ready('cam0', 7).overlay(2_000_000, [face()]);
    expect(gw.currentBoxes('cam0')?.faces?.[0]?.name).toBe('Face 5');
    wall += gw.HOLD_S.faces + 0.01;
    expect(gw.currentBoxes('cam0')?.faces).toEqual([]);
    expect(net.conns).toHaveLength(1);
    off();
    expect(gw.currentBoxes('cam0')).toBeNull();
  });

  it('leaves labels untouched after a failed claim', async () => {
    const got: LiveFrame[] = [];
    const off = gw.watchBoxes('cam0', frame => got.push(frame));
    await flush();
    net.ready('cam0', 7).overlay(2_000_000, [face()]);
    const fetch = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(Response.json({ error: 'Select the face again' }, { status: 409 }));
    try {
      const { claimLiveFace } = await import('../people/liveFace');
      await expect(claimLiveFace({ source, image: 'crop' }, 'Dana')).rejects.toThrow('Select the face again');
      expect(got).toHaveLength(1);
      expect(gw.currentBoxes('cam0')?.faces?.[0]?.name).toBe('Face 5');
    } finally {
      fetch.mockRestore();
      off();
    }
  });

  it('scopes names to the exact camera, stable track and unnamed group', async () => {
    const off = gw.watchBoxes('cam0', () => {});
    await flush();
    const camera = net.ready('cam0', 7);
    camera.overlay(2_000_000, [face(), face('other-track'), face('different-group', 'Face 6')]);
    const other = gw.watchBoxes('cam1', () => {});
    await flush();
    net.ready('cam1', 8).overlay(2_000_000, [face()]);
    gw.nameLiveFace(source, 'Dana');
    expect(gw.currentBoxes('cam0')?.faces?.map(f => f.name)).toEqual(['Dana', 'Face 5', 'Face 6']);
    expect(gw.currentBoxes('cam1')?.faces?.[0]?.name).toBe('Face 5');
    camera.overlay(2_100_000, [face(source.track, 'Face 6')]);
    expect(gw.currentBoxes('cam0')?.faces?.[0]?.name).toBe('Face 6');
    off();
    other();
  });

  it('does not extend stale holds, but retains the claim through later unnamed backend frames', async () => {
    const got: LiveFrame[] = [];
    const off = gw.watchBoxes('cam0', frame => got.push(frame));
    await flush();
    const connection = net.ready('cam0', 7);
    connection.overlay(2_000_000, [face()]);
    wall += gw.HOLD_S.faces + 0.01;
    gw.nameLiveFace(source, 'Dana');
    expect(got.at(-1)?.faces).toEqual([]);
    connection.overlay(2_100_000, [face()]);
    expect(got.at(-1)?.faces?.[0]?.name).toBe('Dana');
    wall += gw.HOLD_S.faces + 0.01;
    expect(gw.currentBoxes('cam0')?.faces).toEqual([]);
    off();
  });

  it('retires overrides at a named backend frame and ignores older overlays or late claim responses', async () => {
    const off = gw.watchBoxes('cam0', () => {});
    await flush();
    const connection = net.ready('cam0', 7);
    connection.overlay(2_000_000, [face()]);
    gw.nameLiveFace(source, 'Dana');
    connection.overlay(3_000_000, [face(source.track, 'Alex')]);
    connection.overlay(2_500_000, [face()]);
    gw.nameLiveFace(source, 'Dana');
    expect(gw.currentBoxes('cam0')?.faces?.[0]?.name).toBe('Alex');
    connection.overlay(4_000_000, [face()]);
    expect(gw.currentBoxes('cam0')?.faces?.[0]?.name).toBe('Face 5');
    off();
  });

  it('bounds claims per session and discards them when the camera is released', async () => {
    const off = gw.watchBoxes('cam0', () => {});
    await flush();
    for (let i = 0; i < 129; i++) gw.nameLiveFace({ ...source, track: String(i) }, `Person ${i}`);
    net.ready('cam0', 7).overlay(2_000_000, [face('0'), face('1'), face('128')]);
    expect(gw.currentBoxes('cam0')?.faces?.map(f => f.name)).toEqual(['Face 5', 'Person 1', 'Person 128']);
    off();
    const stop = gw.watchBoxes('cam0', () => {});
    await flush();
    net.ready('cam0', 7).overlay(2_000_000, [face('128')]);
    expect(gw.currentBoxes('cam0')?.faces?.[0]?.name).toBe('Face 5');
    stop();
  });
});

describe('sessions', () => {
  it("opens one per camera, shared by its frames and boxes, follows the camera's video with faces and objects", async () => {
    const offFrames = gw.watchFrames('cam0', () => {});
    const offBoxes = gw.watchBoxes('cam0', () => {});
    const offOther = gw.watchFrames('cam1', () => {});
    await flush();
    expect(net.conns).toHaveLength(2);
    const c = net.conns[0];
    c?.hello();
    expect(c?.last).toEqual({ type: 'streams' });
    c?.streams({ 'cam0.video': 7, 'cam1.video': 8 });
    expect(c?.last).toEqual({ type: 'live', stream: 7, overlays: ['faces', 'objects', 'poses'] });
    offFrames();
    expect(c?.closed).toBe(false); // the boxes still watch it
    offBoxes();
    expect(c?.closed).toBe(true);
    offOther();
    expect(net.open()).toHaveLength(0);
  });

  it('keeps the same feed while local views change whether they draw poses', async () => {
    const off = gw.watchBoxes('cam0', () => {});
    await flush();
    const c = net.ready('cam0', 7);
    const stop = gw.wantPoses('cam0');
    expect(c.last).toEqual({ type: 'live', stream: 7, overlays: ['faces', 'objects', 'poses'] });
    const stop2 = gw.wantPoses('cam0');
    expect(c.sent.filter(r => r.type === 'live')).toHaveLength(1); // drawing preferences do not replace the feed
    stop();
    stop2();
    expect(c.last).toEqual({ type: 'live', stream: 7, overlays: ['faces', 'objects', 'poses'] });
    off();
  });

  it("tries again later, backing off, when the camera isn't registered or the connection drops; never once released", async () => {
    const off = gw.watchFrames('cam0', () => {});
    await flush();
    net.conns[0]?.hello();
    net.conns[0]?.streams({ 'other.video': 3 });
    expect(net.conns[0]?.closed).toBe(true);
    await vi.advanceTimersByTimeAsync(999);
    expect(net.conns).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(net.conns).toHaveLength(2);
    net.ready('cam0', 7).fail();
    await vi.advanceTimersByTimeAsync(1000); // the live reply reset the backoff
    expect(net.conns).toHaveLength(3);
    off();
    net.conns[2]?.fail();
    await vi.advanceTimersByTimeAsync(20_000);
    expect(net.conns).toHaveLength(3);
  });
});

describe('frames and boxes', () => {
  it('hands JPEG frames to every frame watcher of the camera', async () => {
    const got: number[] = [];
    const times: number[] = [];
    const off = gw.watchFrames('cam0', j => got.push(j.length));
    const flips: boolean[] = [];
    const stopTimes = gw.watchFrames('cam0', (_j, capturedAt, flip) => {
      times.push(capturedAt);
      flips.push(flip);
    });
    await flush();
    const c = net.ready('cam0', 7);
    c.jpeg(1_234_567, new Uint8Array([0xff, 0xd8, 1, 2, 0xff, 0xd9]));
    c.jpeg(2_000_000, new Uint8Array([0xff, 0xd8, 0xff, 0xd9]), true);
    expect(got).toEqual([6, 4]);
    expect(times).toEqual([1.234567, 2]);
    expect(flips).toEqual([false, true]);
    expect(net.conns).toHaveLength(1);
    off();
    stopTimes();
  });

  it('turns an overlay into the LiveFrame the box renderer takes: faces by track, unknown when unnamed, objects', async () => {
    const got: LiveFrame[] = [];
    const off = gw.watchBoxes('cam0', d => got.push(d));
    await flush();
    net.ready('cam0', 7).overlay(2_500_000, [
      { xyxy: [0.1, 0.1, 0.2, 0.2], label: 'Sam', kind: 'face', id: 'r:cam0:1' },
      { xyxy: [0.3, 0.1, 0.4, 0.2], label: '', kind: 'face', id: 'r:cam0:2' },
      { xyxy: [0.5, 0.5, 0.6, 0.6], label: 'mug', kind: 'object' },
      { xyxy: [0, 0, 1, 1], kind: 'body' },
    ]);
    expect(got).toHaveLength(1);
    const d = got[0];
    expect(d?.cam).toBe('cam0');
    expect(d?.ts).toBe(2.5);
    expect(d?.faces?.map(f => [f.track, f.name])).toEqual([
      ['r:cam0:1', 'Sam'],
      ['r:cam0:2', 'unknown'],
    ]);
    expect(d?.objects?.map(o => o.label)).toEqual(['mug']);
    expect(d?.poses).toBeUndefined();
    off();
  });

  it('carries the latest skeletons into later updates while they are fresh, then lets them go', async () => {
    const got: LiveFrame[] = [];
    const off = gw.watchBoxes('cam0', d => got.push(d));
    const stop = gw.wantPoses('cam0');
    await flush();
    const c = net.ready('cam0', 7);
    c.overlay(1_000_000, [], [{ box: [0.2, 0.1, 0.6, 0.9, 0.8], joints: person(0.4) }]);
    wall += 0.4;
    c.overlay(1_500_000, [{ xyxy: [0.1, 0.1, 0.2, 0.2], label: 'Sam', kind: 'face', id: 't' }]);
    wall += 0.2; // 0.6 s after the skeletons came
    c.overlay(1_600_000, [{ xyxy: [0.1, 0.1, 0.2, 0.2], label: 'Sam', kind: 'face', id: 't' }]);
    expect(got.map(d => d.pose_ts)).toEqual([1, 1, undefined]);
    expect(got[1]?.poses?.[0]?.keypoints[5]?.map(v => Math.round(v * 10))).toEqual([4, 3, 9]);
    expect(got[1]?.poses?.[0]?.box.map(v => Math.round(v * 10))).toEqual([2, 1, 6, 9, 8]);
    stop();
    off();
  });

  it('keeps each group in later updates, so faces and objects never wipe each other, and with every frame', async () => {
    const got: LiveFrame[] = [];
    const off = gw.watchBoxes('cam0', d => got.push(d));
    await flush();
    const c = net.ready('cam0', 7);
    c.overlay(2_000_000, [{ xyxy: [0.5, 0.5, 0.6, 0.6], label: 'mug', kind: 'object' }]);
    c.overlay(2_100_000, [{ xyxy: [0.1, 0.1, 0.2, 0.2], label: 'Sam', kind: 'face', id: 't' }]);
    c.jpeg(2_200_000, new Uint8Array([0xff, 0xd8, 0xff, 0xd9]));
    const last = got.at(-1);
    expect(got).toHaveLength(3);
    expect(last?.faces?.map(f => f.name)).toEqual(['Sam']);
    expect(last?.objects?.map(o => o.label)).toEqual(['mug']);
    expect(last?.ts).toBe(2.1); // the newest record's capture time, not the frame's
    off();
  });

  it('lets faces go half a second after they last came, and objects half a minute after', async () => {
    const got: LiveFrame[] = [];
    const off = gw.watchBoxes('cam0', d => got.push(d));
    await flush();
    const c = net.ready('cam0', 7);
    c.overlay(2_000_000, [
      { xyxy: [0.5, 0.5, 0.6, 0.6], label: 'mug', kind: 'object' },
      { xyxy: [0.1, 0.1, 0.2, 0.2], label: 'Sam', kind: 'face', id: 't' },
    ]);
    wall += 0.6;
    c.jpeg(2_700_000, new Uint8Array([0xff, 0xd8, 0xff, 0xd9]));
    expect(got.at(-1)?.faces).toEqual([]);
    expect(got.at(-1)?.objects?.map(o => o.label)).toEqual(['mug']);
    wall += gw.HOLD_S.objects;
    c.jpeg(3_000_000, new Uint8Array([0xff, 0xd8, 0xff, 0xd9]));
    expect(got.at(-1)?.objects).toEqual([]);
    off();
  });

  it('keeps the newer objects when an older grounding comes after', async () => {
    const got: LiveFrame[] = [];
    const off = gw.watchBoxes('cam0', d => got.push(d));
    await flush();
    const c = net.ready('cam0', 7);
    c.overlay(5_000_000, [{ xyxy: [0.5, 0.5, 0.6, 0.6], label: 'mug', kind: 'object' }]);
    c.overlay(4_000_000, [{ xyxy: [0.5, 0.5, 0.6, 0.6], label: 'kettle', kind: 'object' }]);
    expect(got.at(-1)?.objects?.map(o => o.label)).toEqual(['mug']);
    off();
  });
});

describe('which cameras are live at the gateway', () => {
  const stream = (name: string, ageS: number | null) => ({
    id: 1,
    name,
    live: true,
    format: 'passthrough',
    start_us: 0,
    end_us: ageS === null ? 0 : (1000 - ageS) * 1e6,
  });

  it("takes a camera's live video with a recent frame, and nothing else", () => {
    const streams = [
      stream('cam1.video', 5),
      stream('cam0.video', 0.1),
      stream('cam2.video', 31), // too old
      stream('cam3.video', null), // holds nothing yet
      stream('cam4.video_v1', 1), // v1's recordings, not live
      stream('face.detection', 1),
    ];
    expect(gw.liveCamsOf(streams, 1000, 30)).toEqual(['cam0', 'cam1']);
  });

  it('asks the gateway for its streams on a connection of its own, and hangs up', async () => {
    const asked = gw.gatewayStreams();
    await vi.waitFor(() => expect(net.open()).toHaveLength(1));
    const c = net.open()[0];
    c?.hello();
    expect(c?.sent).toEqual([{ type: 'streams' }]);
    c?.streams({ 'cam0.video': 3 });
    expect((await asked).map(s => s.name)).toEqual(['cam0.video']);
    expect(c?.closed).toBe(true);
  });
});

it('keeps one live feed delivering frames through repeated skeleton toggles', async () => {
  const frames: number[] = [];
  const stopFrames = gw.watchFrames('cam0', frame => frames.push(frame.length));
  const stopBoxes = gw.watchBoxes('cam0', () => {});
  await flush();
  const connection = net.ready('cam0', 7);
  for (let n = 0; n < 3; n++) {
    const stopPoses = gw.wantPoses('cam0');
    connection.jpeg(1_000_000 + n, new Uint8Array([0xff, 0xd8, n, 0xff, 0xd9]));
    stopPoses();
  }
  expect(frames).toHaveLength(3);
  expect(connection.sent.filter(request => request.type === 'live')).toHaveLength(1);
  expect(net.conns).toHaveLength(1);
  stopFrames();
  stopBoxes();
});

it('retains skeleton track and color through the live SVG subscription', async () => {
  const got: LiveFrame[] = [];
  const off = gw.watchBoxes('cam0', d => got.push(d));
  const stop = gw.wantPoses('cam0');
  await flush();
  net
    .ready('cam0', 7)
    .overlay(1_000_000, [], [{ box: [0.2, 0.1, 0.6, 0.9, 0.8], joints: person(0.4), id: 'run:cam0:2', hue: 121 }]);
  expect(got.at(-1)?.poses?.[0]).toMatchObject({ track_id: 'run:cam0:2', hue: 121 });
  stop();
  off();
});
