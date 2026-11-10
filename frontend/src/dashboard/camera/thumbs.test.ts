import { beforeEach, expect, it, vi } from 'vitest';
import { encodeHeader, PayloadType } from '../player/header';
import type { WatchFrame } from '../player/watch';
import { thumbCams } from './ThumbFeed';
import { recordingCanvas } from './testFakes';

// player/watch.ts is the boundary: the page's one watch connection, delivering each followed stream's frames.
const followed: { stream: string; fps: number; deliver: (f: WatchFrame) => void; off: ReturnType<typeof vi.fn> }[] = [];
vi.mock('../player/watch', () => ({
  watchStream: (stream: string, fps: number, deliver: (f: WatchFrame) => void) => {
    const off = vi.fn();
    followed.push({ stream, fps, deliver, off });
    return off;
  },
}));

let thumbs: typeof import('./thumbs');
const jpeg = vi.fn();
const video = vi.fn();
beforeEach(async () => {
  followed.length = 0;
  vi.resetModules();
  thumbs = await import('./thumbs');
  jpeg.mockReset();
  video.mockReset();
  thumbs.setThumbDecodersForTests({ jpeg, video });
});

const bitmap = (width = 896, height = 512) => ({ width, height, close: vi.fn() }) as unknown as ImageBitmap;
function frame(type: number, captureUs: number, opts: { flip?: boolean; payload?: number[] } = {}): WatchFrame {
  const h = {
    streamId: 5,
    payloadType: type as 1 | 4,
    keyframe: true,
    live: true,
    watch: true,
    captureUs,
    flip: opts.flip,
  };
  const head = encodeHeader(h);
  const out = new Uint8Array(head.length + (opts.payload ?? [1, 2]).length);
  out.set(head);
  out.set(opts.payload ?? [1, 2], head.length);
  return { header: { ...h, flip: !!opts.flip }, buffer: out.buffer, media: null };
}

it('follows each kept camera at 1 frame a second and decodes nothing until a canvas asks', () => {
  thumbs.keepThumb('kitchen');
  expect(followed.map(f => [f.stream, f.fps])).toEqual([['kitchen.video', 1]]);
  followed[0]?.deliver(frame(PayloadType.video, 1_000_000));
  followed[0]?.deliver(frame(PayloadType.video, 2_000_000));
  expect([jpeg.mock.calls.length, video.mock.calls.length]).toEqual([0, 0]);
});

it("decodes only the newest kept frame onto a canvas, with its capture time, then says it's painted", async () => {
  thumbs.keepThumb('kitchen');
  const older = frame(PayloadType.video, 1_000_000);
  const newest = frame(PayloadType.video, 2_000_000);
  followed[0]?.deliver(older);
  followed[0]?.deliver(newest); // overwritten in place: one frame per camera, never a queue
  const picture = bitmap();
  video.mockResolvedValue(picture);
  const target = recordingCanvas();
  const painted = vi.fn();
  thumbs.paintThumb('kitchen', target.canvas, painted);
  await vi.waitFor(() => expect(painted).toHaveBeenCalledOnce());
  expect(video).toHaveBeenCalledExactlyOnceWith(newest);
  expect(target.drawImage).toHaveBeenCalledExactlyOnceWith(picture, 0, 0);
  expect([target.canvas.width, target.canvas.height, target.canvas.dataset.capturedAt]).toEqual([896, 512, '2']);
  expect(picture.close).toHaveBeenCalledOnce();
});

it("turns an upside-down camera's JPEG half way round, as Live draws it", async () => {
  thumbs.keepThumb('cam0');
  followed[0]?.deliver(frame(PayloadType.jpeg, 3_000_000, { flip: true, payload: [0xff, 0xd8] }));
  jpeg.mockResolvedValue(bitmap(4, 3));
  const target = recordingCanvas();
  const painted = vi.fn();
  thumbs.paintThumb('cam0', target.canvas, painted);
  await vi.waitFor(() => expect(painted).toHaveBeenCalledOnce());
  expect(Array.from(jpeg.mock.calls[0]?.[0] as Uint8Array)).toEqual([0xff, 0xd8]); // the payload, not the header
  expect(target.translate).toHaveBeenCalledWith(4, 3);
  expect(target.scale).toHaveBeenCalledWith(-1, -1);
});

it('never paints over a newer live frame that got to the canvas first, nor after the tile went', async () => {
  thumbs.keepThumb('kitchen');
  followed[0]?.deliver(frame(PayloadType.video, 1_000_000));
  video.mockResolvedValue(bitmap());
  const live = recordingCanvas();
  live.canvas.dataset.capturedAt = '5'; // the live session drew first
  const paintedLive = vi.fn();
  thumbs.paintThumb('kitchen', live.canvas, paintedLive);
  const gone = recordingCanvas();
  const paintedGone = vi.fn();
  thumbs.paintThumb('kitchen', gone.canvas, paintedGone)();
  await vi.waitFor(() => expect(video).toHaveBeenCalledTimes(2));
  await Promise.resolve();
  expect([live.drawImage.mock.calls.length, gone.drawImage.mock.calls.length]).toEqual([0, 0]);
  expect([paintedLive.mock.calls.length, paintedGone.mock.calls.length]).toEqual([0, 0]);
});

it('keeps nothing for a camera dropped from the feed (paused for privacy), so nothing from before shows', () => {
  const off = thumbs.keepThumb('kitchen');
  followed[0]?.deliver(frame(PayloadType.video, 1_000_000));
  off();
  expect(followed[0]?.off).toHaveBeenCalledOnce();
  thumbs.paintThumb('kitchen', recordingCanvas().canvas, () => {});
  expect(video).not.toHaveBeenCalled();
});

it('feeds only active cameras that vision says are on and that are not paused for privacy', () => {
  const states = { kitchen: { state: 'on' }, patio: { state: 'on' }, driveway: { state: 'retrying' }, cam0: {} };
  expect(thumbCams(['kitchen', 'patio', 'driveway', 'cam0', 'bedroom'], new Set(['patio']), states)).toEqual([
    'kitchen',
  ]);
  expect(thumbCams(['kitchen'], new Set(), undefined)).toEqual([]);
});
