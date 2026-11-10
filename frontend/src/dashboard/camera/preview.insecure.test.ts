// Without WebCodecs (a plain-http page that isn't localhost), tiles take one live JPEG frame instead.
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { Connection, Handlers } from '../player/transport';
import { controlVisibility, recordingCanvas } from './testFakes';

let preview: typeof import('./preview');
let page: ReturnType<typeof controlVisibility>;
let watchers: {
  cam: string;
  fn: (jpeg: Uint8Array, capturedAt: number, flip: boolean) => void;
  stop: ReturnType<typeof vi.fn>;
}[];
let connect: ReturnType<typeof vi.fn<(url: string, handlers: Handlers) => Connection>>;
const offs: (() => void)[] = [];
const settle = (ms = 0) => vi.advanceTimersByTimeAsync(ms);

beforeEach(async () => {
  vi.useFakeTimers();
  vi.setSystemTime(100_000);
  vi.resetModules();
  page = controlVisibility();
  watchers = [];
  connect = vi.fn<(url: string, handlers: Handlers) => Connection>();
  preview = await import('./preview');
  preview.setPreviewForTests({
    canDecodeVideo: () => false,
    connect,
    watch: (cam, fn) => {
      const stop = vi.fn();
      watchers.push({ cam, fn, stop });
      return stop;
    },
    bitmap: async () => ({ width: 640, height: 360, close: vi.fn() }) as unknown as ImageBitmap,
  });
});
afterEach(() => {
  for (const off of offs.splice(0)) off();
  page.restore();
  vi.useRealTimers();
});

it('draws a live JPEG frame, turned the right way up, without the archive or a decoder', async () => {
  const { canvas, drawImage, translate, scale } = recordingCanvas();
  canvas.getContext = (() => ({ drawImage, translate, scale, save: vi.fn(), restore: vi.fn() })) as never;
  const notify = vi.fn();
  offs.push(preview.attachPreview('cam0', canvas, notify));
  await settle();
  expect(watchers.map(w => w.cam)).toEqual(['cam0']);
  watchers[0]?.fn(new Uint8Array([0xff, 0xd8, 0xff, 0xd9]), 99.5, true);
  await settle();
  expect(drawImage).toHaveBeenCalledTimes(1);
  expect(translate).toHaveBeenCalledWith(640, 360);
  expect(scale).toHaveBeenCalledWith(-1, -1);
  expect(watchers[0]?.stop).toHaveBeenCalled();
  expect(notify).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'ready', capturedAt: 99.5 }));
  expect(connect).not.toHaveBeenCalled();
});

it('says why when no live frame comes, and lets go of the live watch', async () => {
  const { canvas } = recordingCanvas();
  const notify = vi.fn();
  offs.push(preview.attachPreview('cam1', canvas, notify));
  await settle(6000);
  expect(notify).toHaveBeenLastCalledWith(
    expect.objectContaining({ status: 'unavailable', error: expect.stringMatching(/https/) }),
  );
  expect(watchers[0]?.stop).toHaveBeenCalled();
});
