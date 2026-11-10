import type { FrameMetadata } from '../dashboard/camera/geometry';
import type { Box } from './api';
// A displayed crop or recorded frame stays in this tab until Objects consumes it. Navigation saves nothing.
export interface LiveObjectDraft {
  kind?: 'crop';
  camera: string;
  name: string;
  image: string;
  captured_at: number;
  /** The clicked detection inside `image`, normalized [x, y, w, h]: Objects outlines exactly this box with EdgeTAM. */
  box?: Box;
}
export interface ReviewFrameDraft {
  kind: 'full_frame';
  frame_metadata?: FrameMetadata;
  camera: string;
  image: string;
  captured_at: number;
  name?: never;
}
export type ObjectDraft = LiveObjectDraft | ReviewFrameDraft;
let pending: { draft: ObjectDraft; createdAt: number } | null = null;

/** Preserve the gateway's complete JPEG and recorded timestamp without a crop or a suggested name. */
export function handoffReviewFrame(
  image: string,
  camera: string,
  captured_at: number,
  frame_metadata?: FrameMetadata,
): void {
  if (!camera || !Number.isFinite(captured_at) || captured_at < 0) {
    throw new Error('No recorded frame is available.');
  }
  if (!image.startsWith('data:image/jpeg;base64,') || image.length > 16_000_000) {
    throw new Error('Expected a bounded JPEG recorded frame.');
  }
  pending = {
    draft: {
      kind: 'full_frame',
      camera,
      image,
      captured_at,
      ...(frame_metadata ? { frame_metadata: JSON.parse(JSON.stringify(frame_metadata)) as FrameMetadata } : {}),
    },
    createdAt: Date.now(),
  };
}

export function cropLiveObject(canvas: HTMLCanvasElement, box: readonly number[], camera: string, name: string): void {
  // A gateway canvas carries the capture identity of its actual pixels. Legacy callers without one keep their
  // existing wall-clock fallback; an explicitly corrupt identity must never become a newly observed object.
  const rawCapturedAt = canvas.dataset.capturedAt;
  const capturedAt = rawCapturedAt === undefined ? Date.now() / 1000 : Number(rawCapturedAt);
  if (!Number.isFinite(capturedAt) || capturedAt < 0 || rawCapturedAt?.trim() === '')
    throw new Error('No displayed capture time is available.');
  if (box.length !== 4 || !box.every(Number.isFinite) || canvas.width <= 0 || canvas.height <= 0) {
    throw new Error('No displayed object picture is available.');
  }
  const [x1, y1, x2, y2] = box.map(v => Math.max(0, Math.min(1, v))) as [number, number, number, number];
  const rawWidth = (x2 - x1) * canvas.width;
  const rawHeight = (y2 - y1) * canvas.height;
  if (rawWidth < 2 || rawHeight < 2) throw new Error('The detected object is too small to crop.');
  // A little surrounding context lets SAM distinguish a rectangular object's edge from the crop boundary.
  const x = Math.max(0, Math.floor(x1 * canvas.width - rawWidth * 0.05));
  const y = Math.max(0, Math.floor(y1 * canvas.height - rawHeight * 0.05));
  const width = Math.min(canvas.width, Math.ceil(x2 * canvas.width + rawWidth * 0.05)) - x;
  const height = Math.min(canvas.height, Math.ceil(y2 * canvas.height + rawHeight * 0.05)) - y;
  if (width < 2 || height < 2) throw new Error('The detected object is too small to crop.');
  const crop = document.createElement('canvas');
  const scale = Math.min(1, 1280 / Math.max(width, height));
  crop.width = Math.max(1, Math.round(width * scale));
  crop.height = Math.max(1, Math.round(height * scale));
  const context = crop.getContext('2d');
  if (!context) throw new Error('Could not prepare the object picture.');
  context.drawImage(canvas, x, y, width, height, 0, 0, crop.width, crop.height);
  const image = crop.toDataURL('image/jpeg', 0.85);
  if (!image.startsWith('data:image/jpeg;base64,') || image.length > 2_700_000) {
    throw new Error('The object picture is too large.');
  }
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const left = clamp((x1 * canvas.width - x) / width);
  const top = clamp((y1 * canvas.height - y) / height);
  const objectBox: Box = [
    left,
    top,
    clamp((x2 * canvas.width - x) / width) - left,
    clamp((y2 * canvas.height - y) / height) - top,
  ];
  pending = {
    draft: {
      kind: 'crop',
      camera,
      name: name.trim().slice(0, 80),
      image,
      captured_at: capturedAt,
      box: objectBox,
    },
    createdAt: Date.now(),
  };
}

export function peekObjectDraft(): ObjectDraft | null {
  if (pending && Date.now() - pending.createdAt > 60_000) pending = null;
  return pending?.draft ?? null;
}

export function takeObjectDraft(): ObjectDraft | null {
  const draft = peekObjectDraft();
  pending = null;
  return draft;
}

// Existing Live callers share the same one-shot tab-local handoff.
export const peekLiveObject = peekObjectDraft;
export const takeLiveObject = takeObjectDraft;
