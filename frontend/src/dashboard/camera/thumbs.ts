// The latest picture of every live camera, kept from app start on whatever page, so Live's tiles paint at once
// instead of waiting a couple of seconds for their first frame.
//
// Each camera is followed at 1 frame a second on the page's one watch connection (player/watch.ts), which the
// gateway answers with the camera's newest frame that decodes alone (a JPEG, or a video keyframe). The slot keeps
// only that newest frame, encoded, overwritten as each one comes: no history and no queue, and nothing decoded in the
// background. When a Live tile attaches (mux.ts), the kept frame is decoded once and drawn on it, dimmed as a last
// frame is, unless the live session's own frames got there first. The watch connection closes while the page is
// hidden; a camera paused for privacy (or no longer on) is let go, so no picture from before a pause is shown.
import { PayloadType, splitMessage } from '../player/header';
import { type WatchFrame, watchStream } from '../player/watch';
import { LiveVideoDecoder } from './videoDecoder';

/** How many frames a second each kept camera is followed at. */
export const THUMB_FPS = 1;

/** What turns a kept frame into pixels; tests hand in their own. */
export interface ThumbDecoders {
  jpeg: (bytes: Uint8Array) => Promise<ImageBitmap>;
  video: (frame: WatchFrame) => Promise<ImageBitmap>;
}

/** One H.264/HEVC keyframe through the player worker (as Live decodes its video), then the worker goes. */
function decodeVideo(frame: WatchFrame): Promise<ImageBitmap> {
  return new Promise((resolve, reject) => {
    let decoder: LiveVideoDecoder | null = null;
    const timer = setTimeout(() => {
      decoder?.close();
      reject(new Error('the camera picture took too long to decode'));
    }, 5000);
    try {
      decoder = new LiveVideoDecoder(
        bitmap => {
          clearTimeout(timer);
          decoder?.close();
          resolve(bitmap);
        },
        error => {
          clearTimeout(timer);
          decoder?.close();
          reject(new Error(error));
        },
      );
      decoder.take(frame.buffer, frame.media ?? undefined);
    } catch (error) {
      clearTimeout(timer);
      reject(error);
    }
  });
}

let decoders: ThumbDecoders = {
  jpeg: bytes => createImageBitmap(new Blob([bytes as BlobPart], { type: 'image/jpeg' })),
  video: decodeVideo,
};
/** For tests: how kept frames are decoded. */
export function setThumbDecodersForTests(d: ThumbDecoders): void {
  decoders = d;
}

interface Slot {
  latest: WatchFrame | null;
}
const slots = new Map<string, Slot>();

/** Keeps cam's newest frame until the function it returns is called. */
export function keepThumb(cam: string): () => void {
  const slot: Slot = { latest: null };
  slots.set(cam, slot);
  const off = watchStream(`${cam}.video`, THUMB_FPS, frame => {
    // Only a frame that decodes alone: while Live follows the camera at every frame, its delta frames and overlays
    // come here too.
    const { payloadType, keyframe } = frame.header;
    if (payloadType === PayloadType.jpeg || (payloadType === PayloadType.video && keyframe)) slot.latest = frame;
  });
  return () => {
    off();
    if (slots.get(cam) === slot) slots.delete(cam);
  };
}

const seconds = (c: HTMLCanvasElement): number => Number(c.dataset.capturedAt ?? Number.NaN);

/** Draws cam's kept frame on canvas once it's decoded, then calls painted, unless the canvas has a newer picture by
 *  then (the live session's) or the returned function was called first. Nothing when the camera isn't kept. */
export function paintThumb(cam: string, canvas: HTMLCanvasElement, painted: () => void): () => void {
  const frame = slots.get(cam)?.latest;
  if (!frame) return () => {};
  let cancelled = false;
  const { header, payload } = splitMessage(frame.buffer);
  const at = header.captureUs / 1e6;
  // A JPEG is as captured, so a camera mounted upside down is turned here (as mux.ts draws live JPEG); video comes
  // out of the player worker already upright.
  const turn = header.payloadType === 0x04 && header.flip;
  const pixels = header.payloadType === 0x04 ? decoders.jpeg(payload) : decoders.video(frame);
  pixels.then(
    bitmap => {
      try {
        const mine = seconds(canvas);
        if (cancelled || (Number.isFinite(mine) && mine >= at)) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        if (turn) {
          ctx.save();
          ctx.translate(bitmap.width, bitmap.height);
          ctx.scale(-1, -1);
          ctx.drawImage(bitmap, 0, 0);
          ctx.restore();
        } else ctx.drawImage(bitmap, 0, 0);
        canvas.dataset.capturedAt = String(at);
        delete canvas.dataset.frameMetadata;
        painted();
      } finally {
        bitmap.close();
      }
    },
    () => {}, // no picture yet is what the tile shows anyway
  );
  return () => {
    cancelled = true;
  };
}
