// The maths behind Live's Enhance view, kept free of WebGL so it can be tested on made-up frames.
//
// Enhance stabilises first, then highlights change: Eulerian filters look at fixed pixel positions over time, so an
// unremoved camera wobble of d pixels turns every edge into a change of about |grad I| * d. The order is
//
//   1. estimate where the current frame sits against a reference frame (a translation, from a small luma copy:
//      estimateShift), and warp it back by that much (Stabiliser decides when to trust, reset, or fade);
//   2. run a causal temporal band-pass per pixel on the stabilised frames: two running averages with different time
//      constants, fast minus slow (BandPass here, the same sums in the shader);
//   3. divide the band-passed energy by local contrast, so what a sub-pixel residual leaves along edges stays dark
//      while change on flat ground shows (changeStrength).
//
// The colour preset is linear Eulerian video magnification (Wu et al., "Eulerian Video Magnification for Revealing
// Subtle Changes in the World", SIGGRAPH 2012): the band-passed colour, amplified and added back. MIT holds patents on
// Eulerian video magnification; they permit non-commercial use, which is how this project uses it. A product sold on
// this code would need a licence from MIT.

/** Luma weights (Rec. 601), the same in the shader. */
export const LUMA = [0.299, 0.587, 0.114] as const;

/** The weight a first-order running average gives a new sample after dt seconds, for time constant tau: frame-rate
 *  independent, so a 10 fps camera and a 30 fps one get the same band. */
export function emaAlpha(dt: number, tau: number): number {
  if (!(dt > 0)) return 0;
  if (!(tau > 0)) return 1;
  return 1 - Math.exp(-dt / tau);
}

/** Time constants for a band from low to high hertz: the slow average passes below low, the fast one below high. */
export function bandTaus(lowHz: number, highHz: number): { fast: number; slow: number } {
  return { fast: 1 / (2 * Math.PI * highHz), slow: 1 / (2 * Math.PI * lowHz) };
}

/** A causal temporal band-pass per sample: fast running average minus slow one. The shader runs the same update on
 *  every pixel; this copy is the reference the tests hold it to. */
export class BandPass {
  private fast: Float32Array | null = null;
  private slow: Float32Array | null = null;

  readonly fastTau: number;
  readonly slowTau: number;

  constructor(fastTau: number, slowTau: number) {
    this.fastTau = fastTau;
    this.slowTau = slowTau;
  }

  /** Take one frame's samples, dt seconds after the last, and return the band-passed values. The first frame (or a
   *  frame of a different size) starts both averages at itself, so it passes nothing. */
  step(samples: Float32Array, dt: number): Float32Array {
    if (!this.fast || !this.slow || this.fast.length !== samples.length) {
      this.fast = Float32Array.from(samples);
      this.slow = Float32Array.from(samples);
      return new Float32Array(samples.length);
    }
    const af = emaAlpha(dt, this.fastTau);
    const as = emaAlpha(dt, this.slowTau);
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const x = samples[i] ?? 0;
      const f = (this.fast[i] ?? 0) + af * (x - (this.fast[i] ?? 0));
      const s = (this.slow[i] ?? 0) + as * (x - (this.slow[i] ?? 0));
      this.fast[i] = f;
      this.slow[i] = s;
      out[i] = f - s;
    }
    return out;
  }

  reset(): void {
    this.fast = null;
    this.slow = null;
  }
}

/** Below this much band-passed luma (on a 0 to 1 scale) is compression noise, not change. */
export const NOISE_FLOOR = 0.012;
/** How much local contrast (luma step per sample) raises the bar, so a residual shimmer along edges stays dark. Along
 *  an edge, band-passed luma is about gradient times displacement, so this sets the smallest movement shown there:
 *  HIGHLIGHT_LOW * EDGE_WEIGHT, about 0.3 of a filter sample (a pixel or two of the full frame). A stabiliser's
 *  leftover error is a fraction of a full-frame pixel, well under it. */
export const EDGE_WEIGHT = 0.25;
/** Normalised change below LOW shows nothing, above HIGH shows fully; smooth between. */
export const HIGHLIGHT_LOW = 1.2;
export const HIGHLIGHT_HIGH = 3.0;

/** How strongly one pixel's change shows, 0 to 1: its band-passed luma over the local contrast, soft-thresholded.
 *  contrast is half the range of the slow average over the 3 by 3 around the pixel: a step edge's gradient per
 *  sample, and a thin line's too (a central difference reads a thin bright line between two dark ones as flat, and
 *  the line then lights up at every wobble). The shader computes the same. */
export function changeStrength(bandLuma: number, contrast: number): number {
  const n = Math.abs(bandLuma) / (NOISE_FLOOR + EDGE_WEIGHT * Math.abs(contrast));
  const t = Math.min(1, Math.max(0, (n - HIGHLIGHT_LOW) / (HIGHLIGHT_HIGH - HIGHLIGHT_LOW)));
  return t * t * (3 - 2 * t);
}

/** Near the frame's edge a wobble brings in pixels from beyond it (the edge row, repeated), which is not change: the
 *  highlight is zero within EDGE_MARGIN of each edge (plus the shift there) and ramps up over EDGE_RAMP. */
export const EDGE_MARGIN = 0.03;
export const EDGE_RAMP = 0.04;

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** How much highlight an image position (x, y, fractions of the frame) may show, 0 to 1, with the frame shifted by
 *  (shiftX, shiftY) fractions. The shaders compute the same. */
export function edgeMask(x: number, y: number, shiftX = 0, shiftY = 0): number {
  const mx = Math.abs(shiftX) + EDGE_MARGIN;
  const my = Math.abs(shiftY) + EDGE_MARGIN;
  return (
    smoothstep(mx, mx + EDGE_RAMP, x) *
    smoothstep(mx, mx + EDGE_RAMP, 1 - x) *
    smoothstep(my, my + EDGE_RAMP, y) *
    smoothstep(my, my + EDGE_RAMP, 1 - y)
  );
}

// Motion: a translation from a small luma copy (long side LUMA_SIZE), by inverse-compositional Lucas-Kanade on a grid
// of blocks, coarse to fine over three levels, then the median of the blocks that agree. A person walking through
// moves a few blocks and the median ignores them; a block with no texture or with a large leftover error is dropped.

/** The long side of the luma copy motion is estimated on. A multiple of 4: the shader packs four lumas per texel. */
export const LUMA_SIZE = 256;
const LEVELS = 3;
const GRID_X = 4;
const GRID_Y = 3;
const EDGE = 0.08; // blocks keep this fraction of each side clear, so a shifted sample stays inside the picture
const MIN_TEXTURE = 2e-4; // a block's smallest Hessian eigenvalue per sample, below which it can't place itself
const MAX_RESIDUAL = 0.05; // mean absolute luma error left after aligning, above which a block saw real change
const AGREE = 0.75; // luma-copy pixels: blocks within this of the median agree with it

/** A luma image: samples row by row, 0 to 1. */
export interface Luma {
  data: Float32Array;
  width: number;
  height: number;
}

/** The luma copy's size for a frame: long side LUMA_SIZE, the short side to match, both multiples of 4. */
export function lumaSize(width: number, height: number): { width: number; height: number } {
  const round4 = (v: number) => Math.max(4, Math.round(v / 4) * 4);
  return width >= height
    ? { width: LUMA_SIZE, height: round4((LUMA_SIZE * height) / Math.max(1, width)) }
    : { width: round4((LUMA_SIZE * width) / Math.max(1, height)), height: LUMA_SIZE };
}

/** Half the size, by averaging each 2 by 2. */
function halve(l: Luma): Luma {
  const width = Math.max(1, l.width >> 1);
  const height = Math.max(1, l.height >> 1);
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = 2 * y * l.width + 2 * x;
      data[y * width + x] =
        ((l.data[i] ?? 0) + (l.data[i + 1] ?? 0) + (l.data[i + l.width] ?? 0) + (l.data[i + l.width + 1] ?? 0)) / 4;
    }
  return { data, width, height };
}

function pyramid(l: Luma): Luma[] {
  const levels = [l];
  for (let i = 1; i < LEVELS; i++) levels.push(halve(levels[i - 1] as Luma));
  return levels;
}

/** Bilinear sample with the edges clamped. */
function sample(l: Luma, x: number, y: number): number {
  const cx = Math.min(l.width - 1, Math.max(0, x));
  const cy = Math.min(l.height - 1, Math.max(0, y));
  const x0 = Math.min(l.width - 2, Math.floor(cx));
  const y0 = Math.min(l.height - 2, Math.floor(cy));
  const fx = cx - x0;
  const fy = cy - y0;
  const i = y0 * l.width + x0;
  const a = l.data[i] ?? 0;
  const b = l.data[i + 1] ?? 0;
  const c = l.data[i + l.width] ?? 0;
  const d = l.data[i + l.width + 1] ?? 0;
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

/** One block of the reference at one level: its sample positions, values, gradients, and inverse Hessian. */
interface Template {
  xs: Int32Array;
  ys: Int32Array;
  values: Float32Array;
  gx: Float32Array;
  gy: Float32Array;
  inv: [number, number, number]; // inverse of [[a, b], [b, c]] as (a, b, c)
  textured: boolean;
}

function template(l: Luma, bx: number, by: number, step: number): Template {
  const x0 = Math.round(l.width * (EDGE + ((1 - 2 * EDGE) * bx) / GRID_X));
  const x1 = Math.round(l.width * (EDGE + ((1 - 2 * EDGE) * (bx + 1)) / GRID_X));
  const y0 = Math.round(l.height * (EDGE + ((1 - 2 * EDGE) * by) / GRID_Y));
  const y1 = Math.round(l.height * (EDGE + ((1 - 2 * EDGE) * (by + 1)) / GRID_Y));
  const xs: number[] = [];
  const ys: number[] = [];
  for (let y = Math.max(1, y0); y < Math.min(l.height - 1, y1); y += step)
    for (let x = Math.max(1, x0); x < Math.min(l.width - 1, x1); x += step) {
      xs.push(x);
      ys.push(y);
    }
  const n = xs.length;
  const values = new Float32Array(n);
  const gx = new Float32Array(n);
  const gy = new Float32Array(n);
  let a = 0;
  let b = 0;
  let c = 0;
  for (let k = 0; k < n; k++) {
    const i = (ys[k] ?? 0) * l.width + (xs[k] ?? 0);
    values[k] = l.data[i] ?? 0;
    const dx = ((l.data[i + 1] ?? 0) - (l.data[i - 1] ?? 0)) / 2;
    const dy = ((l.data[i + l.width] ?? 0) - (l.data[i - l.width] ?? 0)) / 2;
    gx[k] = dx;
    gy[k] = dy;
    a += dx * dx;
    b += dx * dy;
    c += dy * dy;
  }
  const det = a * c - b * b;
  // The smaller eigenvalue: a block of plain wall, or of one straight edge, can't say where it moved
  const minEig = (a + c) / 2 - Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const textured = n > 0 && det > 0 && minEig / n > MIN_TEXTURE;
  return {
    xs: Int32Array.from(xs),
    ys: Int32Array.from(ys),
    values,
    gx,
    gy,
    inv: textured ? [c / det, -b / det, a / det] : [0, 0, 0],
    textured,
  };
}

/** A reference frame prepared for estimateShift: its pyramid's block templates, coarsest level first. */
export interface Reference {
  width: number;
  height: number;
  levels: Template[][];
}

export function prepareReference(l: Luma): Reference {
  const levels = pyramid(l)
    .map((level, i) => {
      const blocks: Template[] = [];
      for (let by = 0; by < GRID_Y; by++)
        for (let bx = 0; bx < GRID_X; bx++) blocks.push(template(level, bx, by, i === 0 ? 2 : 1));
      return blocks;
    })
    .reverse();
  return { width: l.width, height: l.height, levels };
}

/** Where the current frame sits against the reference, in luma-copy pixels: current(x + dx, y + dy) matches
 *  reference(x, y). null when too few blocks place themselves, or they don't agree (a pan too large, a light switched
 *  on, someone filling the picture). */
export function estimateShift(ref: Reference, current: Luma): { dx: number; dy: number } | null {
  if (current.width !== ref.width || current.height !== ref.height) return null;
  const levels = pyramid(current).reverse();
  const blocks = ref.levels[0]?.length ?? 0;
  const found: { dx: number; dy: number }[] = [];
  for (let k = 0; k < blocks; k++) {
    let dx = 0;
    let dy = 0;
    let ok = true;
    let residual = 0;
    for (let li = 0; li < levels.length; li++) {
      const t = ref.levels[li]?.[k];
      const img = levels[li];
      if (!t || !img) break;
      if (li > 0) {
        dx *= 2;
        dy *= 2;
      }
      if (!t.textured) {
        ok = false;
        break;
      }
      const iterations = li === 0 ? 5 : 3;
      for (let it = 0; it < iterations + 1; it++) {
        // Mean difference first: an exposure drift moves every sample, not the picture
        let mean = 0;
        const n = t.values.length;
        const cur = new Float32Array(n);
        for (let s = 0; s < n; s++) {
          const v = sample(img, (t.xs[s] ?? 0) + dx, (t.ys[s] ?? 0) + dy);
          cur[s] = v;
          mean += v - (t.values[s] ?? 0);
        }
        mean /= Math.max(1, n);
        let bxSum = 0;
        let bySum = 0;
        residual = 0;
        for (let s = 0; s < n; s++) {
          const e = (cur[s] ?? 0) - (t.values[s] ?? 0) - mean;
          residual += Math.abs(e);
          bxSum += (t.gx[s] ?? 0) * e;
          bySum += (t.gy[s] ?? 0) * e;
        }
        residual /= Math.max(1, n);
        if (it === iterations) break; // the last pass only measures the residual at the final shift
        const [a, b, c] = t.inv;
        const ddx = a * bxSum + b * bySum;
        const ddy = b * bxSum + c * bySum;
        dx -= ddx;
        dy -= ddy;
        if (!Number.isFinite(dx) || !Number.isFinite(dy) || Math.abs(dx) > img.width / 4) {
          ok = false;
          break;
        }
        if (Math.abs(ddx) < 0.005 && Math.abs(ddy) < 0.005) {
          it = iterations - 1; // converged: measure once more and stop
        }
      }
      if (!ok) break;
    }
    if (ok && residual <= MAX_RESIDUAL) found.push({ dx, dy });
  }
  if (found.length < 3) return null;
  const median = (values: number[]) => {
    const s = [...values].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? (s[m] ?? 0) : ((s[m - 1] ?? 0) + (s[m] ?? 0)) / 2;
  };
  const mx = median(found.map(f => f.dx));
  const my = median(found.map(f => f.dy));
  const agree = found.filter(f => Math.hypot(f.dx - mx, f.dy - my) <= AGREE);
  if (agree.length < Math.max(3, Math.ceil(found.length / 2))) return null;
  return {
    dx: agree.reduce((s, f) => s + f.dx, 0) / agree.length,
    dy: agree.reduce((s, f) => s + f.dy, 0) / agree.length,
  };
}

/** Shifts smaller than this (luma-copy pixels, about one full-size pixel at 1080p) are estimate noise on a still
 *  camera: they're shrunk away, so a still picture never swims. */
export const DEADBAND = 0.12;
/** The largest shift compensated, as a fraction of the luma copy's width. Beyond it the camera moved, not wobbled. */
export const MAX_SHIFT = 0.05;
/** Seconds for the highlight to come back after a reset. */
export const FADE_IN_S = 1.0;
/** A still camera's reference is renewed this often, so a slowly changing room keeps its blocks matching. */
export const REFRESH_S = 10;
/** A gap longer than this between frames starts over. */
export const GAP_S = 1.0;
/** A camera within this of one place (luma-copy pixels) is at rest there. */
export const REST_SLACK = 0.5;
/** At rest at least this far from its reference (luma-copy pixels) for SETTLE_S, a camera has been nudged, not
 *  wobbled: where it now sits becomes the reference. Without this the picture stayed pulled back to the old place for
 *  good, with the frame's edge smeared across the gap. */
export const REST_AWAY = 1.0;
export const SETTLE_S = 1.0;
/** A camera that went beyond MAX_SHIFT is on the move. It counts as stopped after this long at rest. */
export const ARRIVE_S = 0.5;

export interface Stabilised {
  /** Where to sample the current frame for each output position, as a fraction of the frame (upright). */
  dx: number;
  dy: number;
  /** The filter's averages must start again: the frames before this one no longer line up with it. */
  reset: boolean;
  /** How much of the highlight to show, 0 to 1: it fades out on a reset and back in over FADE_IN_S. */
  gain: number;
}

/** Holds a reference frame and says, per frame, how far to shift the frame back onto it. A camera too far from its
 *  reference, or one that can't be placed, resets onto the current frame and fades the highlight, rather than letting
 *  the whole picture light up. A camera on the move (a pan, a camera being carried) is shown as it is until it stops:
 *  holding each stretch of a pan still and then letting go turned smooth video into a run of jumps. */
export class Stabiliser {
  private ref: Reference | null = null;
  private refAge = 0;
  private gain = 0;
  /** Where the camera last sat still (against the reference, luma-copy pixels), and for how long it has. */
  private rest = { dx: 0, dy: 0 };
  private restFor = 0;
  private moving = false;

  update(luma: Luma, dt: number): Stabilised {
    const restart = (): Stabilised => {
      this.ref = prepareReference(luma);
      this.refAge = 0;
      this.gain = 0;
      this.rest = { dx: 0, dy: 0 };
      this.restFor = 0;
      return { dx: 0, dy: 0, reset: true, gain: 0 };
    };
    // A gap, or frame times running backwards (a stream that restarted): the frames before no longer line up
    if (!this.ref || this.ref.width !== luma.width || this.ref.height !== luma.height || !(dt >= 0 && dt < GAP_S)) {
      this.moving = false;
      return restart();
    }
    const shift = estimateShift(this.ref, luma);
    if (!shift) return restart();
    if (Math.hypot(shift.dx, shift.dy) > MAX_SHIFT * luma.width) {
      this.moving = true;
      return restart();
    }
    if (Math.hypot(shift.dx - this.rest.dx, shift.dy - this.rest.dy) <= REST_SLACK) this.restFor += dt;
    else {
      this.rest = shift;
      this.restFor = 0;
    }
    if (this.moving) {
      if (this.restFor < ARRIVE_S) return { dx: 0, dy: 0, reset: true, gain: 0 };
      this.moving = false;
      return restart();
    }
    if (this.restFor >= SETTLE_S && Math.hypot(this.rest.dx, this.rest.dy) >= REST_AWAY) return restart();
    const shrink = (v: number) => (Math.abs(v) <= DEADBAND ? 0 : Math.sign(v) * (Math.abs(v) - DEADBAND));
    const dx = shrink(shift.dx);
    const dy = shrink(shift.dy);
    this.refAge += dt;
    this.gain = Math.min(1, this.gain + dt / FADE_IN_S);
    if (dx === 0 && dy === 0 && this.refAge > REFRESH_S) {
      // Nothing to correct either way, so swapping the reference moves nothing on screen
      this.ref = prepareReference(luma);
      this.refAge = 0;
    }
    return { dx: dx / luma.width, dy: dy / luma.height, reset: false, gain: this.gain };
  }

  reset(): void {
    this.ref = null;
    this.moving = false;
  }
}
