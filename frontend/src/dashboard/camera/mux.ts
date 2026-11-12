// Live camera frames onto canvases, through Continuum's gateway.
//
// Each camera on screen has one gateway session (gateway.ts), shared by every canvas showing it and by its boxes
// layer; its JPEG frames are decoded once and drawn on each of those canvases. A camera's session is released when
// its last canvas goes (leaving the Live tab) or the page is hidden, and taken again when they come back.
import { create } from 'zustand';
import { watchFrames, watchVideo } from './gateway';
import type { FrameMetadata } from './geometry';
import { paintThumb } from './thumbs';
import { LiveVideoDecoder } from './videoDecoder';

/** cam -> true while its frames arrive (false after 4 s without one), for snapshot fallbacks. */
export const useLiveNow = create<Record<string, boolean>>(() => Object.create(null));

export const useLiveError = create<Record<string, string | undefined>>(() => Object.create(null));

export const STALE_MS = 4000; // no frame for this long and a camera is no longer live

const canvases = new Map<string, Set<HTMLCanvasElement>>();
const lastAt = new Map<string, number>(); // cam -> ms of its last drawn frame
const watching = new Map<string, () => void>(); // cam -> stops its frames
let ticker: ReturnType<typeof setInterval> | undefined;

/** Something that also wants each decoded frame of a camera (Live's Enhance view): handed the picture before the
 *  canvases' copy is closed, never kept. capturedAt is in seconds; flip as for draw. */
export type FrameTap = (bitmap: ImageBitmap, capturedAt: number, flip: boolean) => void;
const taps = new Map<string, Set<FrameTap>>();

/** Hand cam's frames to tap as they're drawn; returns the function that stops. It never starts a camera on its own:
 *  a canvas showing the camera (attach) does that. */
export function tapFrames(cam: string, tap: FrameTap): () => void {
  let set = taps.get(cam);
  if (!set) {
    set = new Set();
    taps.set(cam, set);
  }
  const mine = set;
  mine.add(tap);
  return () => {
    mine.delete(tap);
    if (!mine.size && taps.get(cam) === mine) taps.delete(cam);
  };
}

/** Hand a frame to cam's taps. The demo's still cameras (LiveCam.tsx) call it too. */
export function feedTaps(cam: string, bitmap: ImageBitmap, capturedAt: number, flip = false): void {
  for (const tap of taps.get(cam) ?? []) {
    try {
      tap(bitmap, capturedAt, flip);
    } catch {
      // a tap's failure is its own: the plain picture is drawn regardless
    }
  }
}

/** Draw a frame on each of cam's canvases. flip: the camera is mounted upside down (its video.flip) and this picture is
 *  as captured, so it is turned half way round here, as the Review player turns its frames (live JPEG; live video
 *  arrives already turned by the player worker). Overlay boxes come in upright coordinates either way. */
function draw(cam: string, bmp: ImageBitmap, capturedAt: number, metadata?: FrameMetadata, flip = false): void {
  try {
    for (const c of canvases.get(cam) ?? []) {
      if (c.width !== bmp.width || c.height !== bmp.height) {
        delete c.dataset.capturedAt; // resizing clears pixels, so the previous frame's identity no longer applies
        c.width = bmp.width;
        c.height = bmp.height;
      }
      const context = c.getContext('2d');
      if (!context) continue;
      if (flip) {
        context.save();
        context.translate(bmp.width, bmp.height);
        context.scale(-1, -1);
        context.drawImage(bmp, 0, 0);
        context.restore();
      } else context.drawImage(bmp, 0, 0);
      // Freeze readers must see the identity of these pixels, never the newest received or pending JPEG.
      if (Number.isFinite(capturedAt) && capturedAt > 0) c.dataset.capturedAt = String(capturedAt);
      else delete c.dataset.capturedAt;
      if (metadata) c.dataset.frameMetadata = JSON.stringify(metadata);
      else delete c.dataset.frameMetadata;
    }
    feedTaps(cam, bmp, capturedAt, flip);
  } finally {
    bmp.close();
  }
  useLiveError.setState(Object.assign(Object.create(null), useLiveError.getState(), { [cam]: undefined }), true);
  lastAt.set(cam, Date.now());
  if (!useLiveNow.getState()[cam])
    useLiveNow.setState(Object.assign(Object.create(null), useLiveNow.getState(), { [cam]: true }), true);
  startTicker();
}

/** A watcher owns its decode generation, so a late JPEG or worker frame cannot paint a replacement canvas. */
function watchCamera(cam: string): () => void {
  let active = true;
  let jpegState: { decoding: boolean; pending?: { jpeg: Uint8Array; capturedAt: number; flip: boolean } } = {
    decoding: false,
  };
  let video: LiveVideoDecoder | undefined;
  const fail = (error: string) => {
    if (active)
      useLiveError.setState(Object.assign(Object.create(null), useLiveError.getState(), { [cam]: error }), true);
  };
  const decode = (jpeg: Uint8Array, capturedAt: number, flip: boolean): void => {
    const state = jpegState;
    if (state.decoding) {
      state.pending = { jpeg, capturedAt, flip };
      return;
    }
    state.decoding = true;
    createImageBitmap(new Blob([jpeg as Uint8Array<ArrayBuffer>], { type: 'image/jpeg' }))
      .then(b => {
        if (active && state === jpegState) draw(cam, b, capturedAt, undefined, flip);
        else b.close();
      })
      .catch(() => {})
      .finally(() => {
        state.decoding = false;
        const next = state.pending;
        state.pending = undefined;
        if (active && state === jpegState && next) decode(next.jpeg, next.capturedAt, next.flip);
      });
  };
  const offJpeg = watchFrames(cam, (jpeg, at, flip) => decode(jpeg, at, flip));
  const offVideo = watchVideo(cam, (buffer, media) => {
    if (buffer === null) {
      jpegState = { decoding: false };
      video?.close();
      video = undefined;
      if (useLiveNow.getState()[cam])
        useLiveNow.setState(Object.assign(Object.create(null), useLiveNow.getState(), { [cam]: false }), true);
      return;
    }
    try {
      video ??= new LiveVideoDecoder((bitmap, at, metadata) => {
        if (active) draw(cam, bitmap, at, metadata);
        else bitmap.close();
      }, fail);
      video.take(buffer, media);
    } catch (error) {
      fail(error instanceof Error ? error.message : 'Could not decode live camera video.');
    }
  });
  return () => {
    active = false;
    jpegState.pending = undefined;
    video?.close();
    offJpeg();
    offVideo();
  };
}

/** Watch the cameras with canvases (while the page is visible), and stop watching the rest. */
function sync(): void {
  const want = document.hidden ? [] : [...canvases.keys()].filter(c => canvases.get(c)?.size);
  for (const [cam, stop] of watching)
    if (!want.includes(cam)) {
      stop();
      watching.delete(cam);
    }
  for (const cam of want) if (!watching.has(cam)) watching.set(cam, watchCamera(cam));
}

/** Once a second: mark cameras with no frame for 4 s as not live. It runs while any canvas is attached or any camera
 *  is still marked live. */
function tick(): void {
  const t = Date.now();
  const v: Record<string, boolean> = Object.assign(Object.create(null), useLiveNow.getState());
  let changed = false;
  for (const c of Object.keys(v))
    if (v[c] && t - (lastAt.get(c) ?? 0) > STALE_MS) {
      v[c] = false;
      changed = true;
    }
  if (changed) useLiveNow.setState(v, true);
  if (!canvases.size && !Object.values(v).some(Boolean)) {
    clearInterval(ticker);
    ticker = undefined;
  }
}

function startTicker(): void {
  ticker ??= setInterval(tick, 1000);
}

/** Show camera cam on canvas; returns the function that stops (use as an effect cleanup). */
export function attach(cam: string, canvas: HTMLCanvasElement | null): () => void {
  if (!canvas) return () => {};
  delete canvas.dataset.frameMetadata;
  delete canvas.dataset.capturedAt; // a reused canvas cannot retain another attachment's capture identity
  let set = canvases.get(cam);
  if (!set) {
    set = new Set();
    canvases.set(cam, set);
  }
  const mine = set;
  mine.add(canvas);
  // The camera's kept picture (thumbs.ts) shows at once, dimmed as a last frame is, unless live frames came first.
  const unthumb = paintThumb(cam, canvas, () => {
    if (useLiveNow.getState()[cam] === undefined)
      useLiveNow.setState(Object.assign(Object.create(null), useLiveNow.getState(), { [cam]: false }), true);
  });
  startTicker();
  sync();
  return () => {
    unthumb();
    mine.delete(canvas);
    if (!mine.size && canvases.get(cam) === mine) canvases.delete(cam);
    sync();
  };
}

if (typeof document !== 'undefined') document.addEventListener('visibilitychange', sync);
