// Live's encoded access units use the existing player worker's WebCodecs decoder and upright pixel handling.

import type { SmallCopy } from '../enhance/small';
import type { MediaContext } from '../player/protocol';
import type { WorkerLike } from '../player/worker';
import type { FrameMetadata } from './geometry';

export class LiveVideoDecoder {
  private readonly worker: WorkerLike;
  private closed = false;

  constructor(
    onFrame: (bitmap: ImageBitmap, at: number, metadata?: FrameMetadata, small?: SmallCopy) => void,
    onError: (error: string) => void,
  ) {
    if (typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined')
      throw new Error('This browser cannot decode live network-camera video. Try a current browser.');
    const worker = new Worker(new URL('../player/worker.ts', import.meta.url), { type: 'module' });
    this.worker = worker;
    worker.onerror = () => {
      if (!this.closed) onError('The live camera decoder failed. Reload to try again.');
    };
    this.worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'frame') {
        if (this.closed) {
          m.bitmap.close();
          m.small?.picture.close();
          return;
        }
        // Live feeds are native pixels, so the worker's absent media context is expected here. Preserve orientation
        // and the displayed capture time.
        const metadata = m.metadata && {
          ...m.metadata,
          profileStatus: 'absent' as const,
          nativeGeometry: { width: m.bitmap.width, height: m.bitmap.height },
        };
        try {
          onFrame(m.bitmap, m.frameUs / 1e6, metadata, m.small);
        } catch (error) {
          onError(error instanceof Error ? error.message : 'Could not draw live camera video.');
        }
      } else if (!this.closed && m.type === 'codec' && !m.supported)
        onError(`This browser cannot decode this camera's ${m.codec.toUpperCase()} live video.`);
      else if (!this.closed && m.type === 'error') onError(m.message);
    };
    try {
      const canvas = new OffscreenCanvas(1, 1);
      this.worker.postMessage({ type: 'canvas', canvas, returnBitmap: true }, [canvas]);
      this.worker.postMessage({ type: 'playhead', us: 'live' });
    } catch (error) {
      this.close();
      throw error;
    }
  }

  /** media: the frame's media context, when the gateway tagged it with one (a watched stream's frames can be). */
  take(buffer: ArrayBuffer, media?: MediaContext): void {
    if (this.closed) return;
    const copy = buffer.slice(0);
    this.worker.postMessage({ type: 'message', buffer: copy, ...(media ? { media } : {}) }, [copy]);
  }

  /** Whether each frame should come with its small copy (the player worker makes it, off this thread). */
  wantSmall(on: boolean): void {
    if (!this.closed) this.worker.postMessage({ type: 'small', on });
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.worker.terminate();
  }
}
