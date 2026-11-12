// The small copy of a frame that Live's Enhance view analyses, and the luma copy its motion estimate reads, made
// with 2D canvases. One implementation for two places: the player worker makes them for live video, off the page's
// thread, and hands them over with each frame; the renderer makes them itself for frames that arrive without (JPEG
// frames, the demo's stills).
//
// Why the worker: reading a canvas's pixels waits for the graphics card wherever canvases live on it (Chromium), and
// halving a 1920 by 1080 frame costs 3 ms on the processor where they don't (Firefox). Either would be spent on the
// page's thread once a frame, and the page's thread is what draws the video.

import { LUMA, type Luma, lumaSize, smallSizes } from './maths';

type Canvas = HTMLCanvasElement | OffscreenCanvas;
type Context = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

interface Step {
  canvas: Canvas;
  context: Context;
}

/** A frame's small copy as its source hands it over: the picture (maths.ts smallSizes says what size), upright, and
 *  its luma (lumaSize), row by row, 0 to 1. */
export interface SmallCopy {
  picture: ImageBitmap;
  luma: Float32Array;
  width: number;
  height: number;
}

export class SmallCopier {
  private readonly make: (width: number, height: number) => Canvas;
  private steps: Step[] = [];
  private half: Step | null = null;
  private lumaStep: Step | null = null;
  private luma: Luma = { data: new Float32Array(0), width: 0, height: 0 };
  private made = '';

  /** make: a canvas of that size (the page's or a worker's kind). */
  constructor(make: (width: number, height: number) => Canvas) {
    this.make = make;
  }

  private step(width: number, height: number, read = false): Step {
    const canvas = this.make(width, height);
    const context = canvas.getContext('2d', read ? { alpha: false, willReadFrequently: true } : { alpha: false });
    if (!context) throw new Error('This browser has no 2D canvas.');
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'low'; // plain bilinear: at exactly half size, the 2 by 2 average
    return { canvas, context: context as Context };
  }

  /** The size of the small copy of a width by height frame. */
  static size(width: number, height: number): { width: number; height: number } {
    const sizes = smallSizes(width, height);
    return sizes[sizes.length - 1] ?? { width, height };
  }

  /** Make source's small copy and read its luma. source is width by height, upside down when flip. The canvas
   *  returned holds the small copy until the next call. The luma is this copier's own array, overwritten by the
   *  next call, unless fresh asks for a new one (to hand to another thread). */
  copy(
    source: CanvasImageSource,
    width: number,
    height: number,
    flip = false,
    fresh = false,
  ): { canvas: Canvas; luma: Luma } {
    const key = `${width}x${height}${flip ? ' flipped' : ''}`;
    if (key !== this.made) {
      this.steps = smallSizes(width, height).map(s => this.step(s.width, s.height));
      const first = this.steps[0];
      // An upside-down picture is turned as the first copy is made
      if (flip && first) first.context.setTransform(-1, 0, 0, -1, first.canvas.width, first.canvas.height);
      const last = this.steps[this.steps.length - 1]?.canvas;
      this.half = last ? this.step(Math.ceil(last.width / 2), Math.ceil(last.height / 2), true) : null;
      const dims = lumaSize(width, height);
      this.lumaStep = this.step(dims.width, dims.height, true);
      this.luma = { data: new Float32Array(dims.width * dims.height), ...dims };
      this.made = key;
    }
    const small = this.steps[this.steps.length - 1];
    const { half, lumaStep } = this;
    if (!small || !half || !lumaStep) throw new Error('This browser has no 2D canvas.');
    let from: CanvasImageSource = source;
    for (const s of this.steps) {
      s.context.drawImage(from, 0, 0, s.canvas.width, s.canvas.height);
      from = s.canvas;
    }
    // The luma copy comes from half the small copy: blurred to about its own size, so fine detail doesn't unsettle
    // the motion estimate
    const { width: lw, height: lh } = this.luma;
    half.context.drawImage(small.canvas, 0, 0, half.canvas.width, half.canvas.height);
    lumaStep.context.drawImage(half.canvas, 0, 0, lw, lh);
    const pixels = lumaStep.context.getImageData(0, 0, lw, lh).data;
    const data = fresh ? new Float32Array(lw * lh) : this.luma.data;
    const [r, g, b] = LUMA;
    for (let i = 0, p = 0; i < data.length; i++, p += 4)
      data[i] = (r * (pixels[p] ?? 0) + g * (pixels[p + 1] ?? 0) + b * (pixels[p + 2] ?? 0)) / 255;
    return { canvas: small.canvas, luma: { data, width: lw, height: lh } };
  }
}

/** A frame's source may spend this share of the time between frames on the frame and its small copy together. */
const ROOM = 0.6;
/** Without room, the copy is tried again this often (sooner the first few times, while its cost is still being
 *  learnt: the first copy also pays for its canvases). */
const RETRY_MS = 10_000;
const RETRY_FIRST_MS = 1000;

/** Decides, frame by frame, whether a frame's source has room to make the small copy as well as the frame. The
 *  video comes first: where drawing a frame and handing it over already takes most of the time between frames (a
 *  1920 by 1080 stream at 60 a second in a browser whose canvases live on the processor), the copy would make the
 *  source miss frames, so it is left to the page's thread, which has the room there. */
export class CopyRoom {
  private base = Number.NaN; // ms a frame takes without the copy
  private copy = Number.NaN; // ms the copy takes
  private gap = Number.NaN; // ms between frames
  private lastUs = Number.NaN;
  private retryAt = 0;
  private trials = 0;

  /** Whether to make the copy with the frame being drawn now (now in ms). */
  wanted(now: number): boolean {
    // Until its cost and the frame rate are known, make it: that is how they are learnt
    if (Number.isNaN(this.copy) || Number.isNaN(this.gap) || Number.isNaN(this.base)) return true;
    if (this.base + this.copy <= ROOM * this.gap) return true;
    if (now < this.retryAt) return false;
    // One try, to see whether there is room now
    this.trials += 1;
    this.retryAt = now + (this.trials <= 3 ? RETRY_FIRST_MS : RETRY_MS);
    return true;
  }

  /** After a frame: its capture time in microseconds, how long it took without the copy, and how long the copy
   *  took when one was made (both in ms). */
  took(capturedUs: number, baseMs: number, copyMs?: number): void {
    const gap = (capturedUs - this.lastUs) / 1000;
    this.lastUs = capturedUs;
    if (gap > 0 && gap < 1000) this.gap = Number.isNaN(this.gap) ? gap : this.gap + 0.1 * (gap - this.gap);
    this.base = Number.isNaN(this.base) ? baseMs : this.base + 0.1 * (baseMs - this.base);
    if (copyMs !== undefined) this.copy = Number.isNaN(this.copy) ? copyMs : this.copy + 0.5 * (copyMs - this.copy);
  }
}
