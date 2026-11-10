// Loading evidence images and drawing them, with a box, onto a canvas.
import type { Box } from './api';
import { STRIP } from './geometry';

export const BOX_COLOR = '#6fd3c3';

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Image unavailable; capture a new view.'));
    img.src = url;
  });
}

/** Draw img at its own size onto canvas, optionally shading the caption strip and outlining a box (in pixels). */
export function draw(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  box: Box | null,
  shadeStrip = false,
  mask?: HTMLImageElement,
): void {
  if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  if (mask) ctx.drawImage(mask, 0, 0, canvas.width, canvas.height);
  if (shadeStrip) {
    ctx.fillStyle = 'rgba(0,0,0,.65)';
    ctx.fillRect(0, canvas.height - STRIP, canvas.width, STRIP);
  }
  if (box) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = BOX_COLOR;
    ctx.strokeRect(...box);
  }
}
