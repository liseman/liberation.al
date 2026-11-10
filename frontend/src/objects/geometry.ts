// Box selection on a frozen camera view, in the image's own pixels.
import type { Box } from './api';

export const STRIP = 50; // the preview's bottom caption strip can never be part of a selection
export const MIN_BOX = 16; // the smallest selection, in image pixels, on each side

export interface Size {
  width: number;
  height: number;
}

/** Where a pointer at (clientX, clientY) lands in an image of size `image` drawn into `rect`, clamped to the image
 *  and kept above the caption strip. */
export function imagePoint(clientX: number, clientY: number, rect: DOMRect, image: Size): [number, number] {
  const x = ((clientX - rect.left) * image.width) / rect.width;
  const y = ((clientY - rect.top) * image.height) / rect.height;
  return [Math.max(0, Math.min(image.width, x)), Math.max(0, Math.min(image.height - STRIP - 1, y))];
}

/** The box with corners a and b, whichever way it was dragged. */
export function boxBetween(a: [number, number], b: [number, number]): Box {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])];
}

export const bigEnough = (box: Box | null): box is Box => !!box && box[2] >= MIN_BOX && box[3] >= MIN_BOX;

/** A box in image pixels as fractions of the image, which is what the server stores. */
export function normalized(box: Box, image: Size): Box {
  return [box[0] / image.width, box[1] / image.height, box[2] / image.width, box[3] / image.height];
}

/** Coordinates and crop boxes below use the full frozen frame, with no preview caption strip. */
export type Point = [x: number, y: number];
export interface ImageView {
  scale: number;
  x: number;
  y: number;
}
export type CropHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

export function fitImageView(image: Size, viewport: Size): ImageView {
  const scale = Math.min(viewport.width / image.width, viewport.height / image.height);
  return { scale, x: (viewport.width - image.width * scale) / 2, y: (viewport.height - image.height * scale) / 2 };
}

export const frameToView = ([x, y]: Point, view: ImageView): Point => [
  x * view.scale + view.x,
  y * view.scale + view.y,
];
export const viewToFrame = ([x, y]: Point, view: ImageView): Point => [
  (x - view.x) / view.scale,
  (y - view.y) / view.scale,
];
export const clampFramePoint = ([x, y]: Point, image: Size): Point => [
  clamp(x, 0, image.width),
  clamp(y, 0, image.height),
];

/** Keep the whole image visible at fit size, or keep the viewport covered when zoomed in. */
export function clampImageView(view: ImageView, image: Size, viewport: Size): ImageView {
  const fit = fitImageView(image, viewport).scale;
  const scale = clamp(view.scale, fit, fit * 16);
  const width = image.width * scale,
    height = image.height * scale;
  return {
    scale,
    x: width <= viewport.width ? (viewport.width - width) / 2 : clamp(view.x, viewport.width - width, 0),
    y: height <= viewport.height ? (viewport.height - height) / 2 : clamp(view.y, viewport.height - height, 0),
  };
}

/** Keep the frame point under `anchor` stationary, except where the image's edges constrain panning. */
export function zoomImageView(view: ImageView, anchor: Point, scale: number, image: Size, viewport: Size): ImageView {
  const point = viewToFrame(anchor, view);
  const fit = fitImageView(image, viewport).scale;
  const next = clamp(scale, fit, fit * 16);
  return clampImageView(
    { scale: next, x: anchor[0] - point[0] * next, y: anchor[1] - point[1] * next },
    image,
    viewport,
  );
}

export function panImageView(view: ImageView, delta: Point, image: Size, viewport: Size): ImageView {
  return clampImageView({ ...view, x: view.x + delta[0], y: view.y + delta[1] }, image, viewport);
}

/** An image-pixel crop constrained to the full frame, with at least 16 pixels on each available axis. */
export function clampFrameCrop(box: Box, image: Size): Box {
  const left = clamp(Math.round(box[0]), 0, image.width);
  const top = clamp(Math.round(box[1]), 0, image.height);
  const right = clamp(Math.round(box[0] + box[2]), 0, image.width);
  const bottom = clamp(Math.round(box[1] + box[3]), 0, image.height);
  const width = clamp(right - left, Math.min(MIN_BOX, image.width), image.width);
  const height = clamp(bottom - top, Math.min(MIN_BOX, image.height), image.height);
  return [clamp(left, 0, image.width - width), clamp(top, 0, image.height - height), width, height];
}

export function cropFromPoints(a: Point, b: Point, image: Size): Box {
  return clampFrameCrop(boxBetween(clampFramePoint(a, image), clampFramePoint(b, image)), image);
}

export function moveFrameCrop(box: Box, delta: Point, image: Size): Box {
  const [x, y, width, height] = clampFrameCrop(box, image);
  return [
    clamp(Math.round(x + delta[0]), 0, image.width - width),
    clamp(Math.round(y + delta[1]), 0, image.height - height),
    width,
    height,
  ];
}

/** Resize from the original crop; the opposite edge stays fixed even if the pointer crosses it. */
export function resizeFrameCrop(box: Box, handle: CropHandle, delta: Point, image: Size): Box {
  let [left, top, width, height] = clampFrameCrop(box, image);
  let right = left + width,
    bottom = top + height;
  width = Math.min(MIN_BOX, image.width);
  height = Math.min(MIN_BOX, image.height);
  if (handle.includes('w')) left = clamp(left + delta[0], 0, right - width);
  if (handle.includes('e')) right = clamp(right + delta[0], left + width, image.width);
  if (handle.includes('n')) top = clamp(top + delta[1], 0, bottom - height);
  if (handle.includes('s')) bottom = clamp(bottom + delta[1], top + height, image.height);
  return clampFrameCrop([left, top, right - left, bottom - top], image);
}

/** A two-pointer gesture preserves its original frame anchor while the midpoint translates and distance changes. */
export function pinchImageView(
  view: ImageView,
  startCenter: Point,
  center: Point,
  ratio: number,
  image: Size,
  viewport: Size,
): ImageView {
  const anchor = viewToFrame(startCenter, view);
  const fit = fitImageView(image, viewport).scale;
  const scale = clamp(view.scale * ratio, fit, fit * 16);
  return clampImageView({ scale, x: center[0] - anchor[0] * scale, y: center[1] - anchor[1] * scale }, image, viewport);
}
