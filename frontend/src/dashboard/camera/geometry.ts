// Coordinates belong to the actual decoded capture, independently of its CSS placement.
export type Fit = 'fill' | 'contain' | 'cover' | 'none' | 'scale-down';

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Where an image's picture is drawn inside its element (CSS object-fit), in element pixels. */
export function contentRect(elW: number, elH: number, natW: number, natH: number, fit: string = 'fill'): Rect {
  if (!natW || !natH || fit === 'fill' || !elW || !elH) return { x: 0, y: 0, w: elW, h: elH };
  let s = fit === 'cover' ? Math.max(elW / natW, elH / natH) : Math.min(elW / natW, elH / natH);
  if (fit === 'none') s = 1;
  if (fit === 'scale-down') s = Math.min(1, Math.min(elW / natW, elH / natH));
  const w = natW * s;
  const h = natH * s;
  return { x: (elW - w) / 2, y: (elH - h) / 2, w, h };
}

export interface FrameMetadata {
  capturedAtUs: number;
  width: number;
  height: number;
  flip: boolean;
  profileStatus: 'absent' | 'unsupported' | 'unavailable';
  nativeGeometry?: { width: number; height: number } | null;
}
export type NormBox = [number, number, number, number];
const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const dimension = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value > 0 && value <= 8192;

/** The source image's rectangle within the upright decoded picture; null means geometry is unknown. */
export function sourceContentRect(meta: FrameMetadata): Rect | null {
  if (!dimension(meta.width) || !dimension(meta.height) || typeof meta.flip !== 'boolean') return null;
  return meta.profileStatus === 'absent' &&
    meta.nativeGeometry?.width === meta.width &&
    meta.nativeGeometry?.height === meta.height
    ? { x: 0, y: 0, w: meta.width, h: meta.height }
    : null;
}

function normalized(box: readonly number[]): NormBox | null {
  if (box.length !== 4 || !box.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) return null;
  const [x1, y1, x2, y2] = box as NormBox;
  return x2 > x1 && y2 > y1 ? [x1, y1, x2, y2] : null;
}

/** Source boxes are already upright. Only the encoded content rectangle follows the video rotation. */
export function sourceBoxToDecoded(box: readonly number[], meta: FrameMetadata): NormBox | null {
  const b = normalized(box),
    r = sourceContentRect(meta);
  if (!b || !r) return null;
  return [
    (r.x + b[0] * r.w) / meta.width,
    (r.y + b[1] * r.h) / meta.height,
    (r.x + b[2] * r.w) / meta.width,
    (r.y + b[3] * r.h) / meta.height,
  ];
}

/** Manual selection coordinates belong to decoded pixels; reject letterbox padding at enrollment. */
export function decodedBoxToSource(box: readonly number[], meta: FrameMetadata): NormBox | null {
  const b = normalized(box),
    r = sourceContentRect(meta);
  if (!b || !r) return null;
  const result: NormBox = [
    (b[0] * meta.width - r.x) / r.w,
    (b[1] * meta.height - r.y) / r.h,
    (b[2] * meta.width - r.x) / r.w,
    (b[3] * meta.height - r.y) / r.h,
  ];
  if (result.some(v => v < -1e-9 || v > 1 + 1e-9)) return null;
  return result.map(v => Math.max(0, Math.min(1, v))) as NormBox;
}

/** Metadata must name the actual canvas pixels, never the seek target or the stream's newest profile. */
export function canvasFrameMetadata(canvas: HTMLCanvasElement | null): FrameMetadata | null {
  if (!canvas?.dataset.frameMetadata) return null;
  try {
    const meta: unknown = JSON.parse(canvas.dataset.frameMetadata);
    const at = Number(canvas.dataset.capturedAt);
    if (
      !record(meta) ||
      !Number.isSafeInteger(meta.capturedAtUs) ||
      !Number.isFinite(at) ||
      Math.abs(Number(meta.capturedAtUs) - at * 1e6) > 1 ||
      meta.width !== canvas.width ||
      meta.height !== canvas.height
    )
      return null;
    const typed = meta as unknown as FrameMetadata;
    return sourceContentRect(typed) ? typed : null;
  } catch {
    return null;
  }
}
