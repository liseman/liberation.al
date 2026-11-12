// Live's Enhance view: a transparent WebGL2 canvas over the large camera's picture that draws only the highlight.
// The picture itself stays the plain one the page already draws (camera/mux.ts), steadied by a CSS translate that
// the layer sets from `shift` (Enhance.tsx). Each frame the page decodes is handed over before it is closed:
//
//   1. a small copy of the frame is made with 2D canvases, each half the size of the one before (a 2 by 2 average),
//      until its long side is 640 or less, and a luma copy is read from it (small.ts). For live video the player
//      worker has already made both, off the page's thread, and hands them over with the frame;
//   2. Stabiliser (maths.ts) estimates the shift against its reference frame on the CPU, from the luma copy;
//   3. the small copy is uploaded, and a state pass, at a reduced size (which is also the spatial low-pass), samples
//      it shifted back and updates two running averages per pixel into half-float targets (ping-pong, two outputs);
//   4. a strength pass, at the same size, divides the band-pass (fast minus slow) by the local contrast of the slow
//      average (half its range over 3 by 3; maths.ts changeStrength, the same sums), with mipmaps for the glow;
//   5. an overlay pass draws the look as premultiplied colour and alpha: the page shows picture * (1 - alpha) + colour.
//
// Why the full frame never goes through WebGL: measured at 1920 by 1080 in Firefox, uploading it cost 18 ms of a
// 24 ms frame (there is no path that stays on the graphics card there), and reading a luma copy back for the motion
// estimate stalled the page behind whatever else the card was doing. Uploading the small copy costs about a
// millisecond, and nothing here waits for the graphics card: if it has not finished the frame before, this frame's
// highlight is skipped and the last one stays over the new picture.
//
// The colour look is linear Eulerian video magnification (Wu et al., SIGGRAPH 2012); MIT's patents on it permit
// non-commercial use, which is how this project uses it (maths.ts says more).

import {
  bandTaus,
  EDGE_MARGIN,
  EDGE_RAMP,
  EDGE_WEIGHT,
  emaAlpha,
  HIGHLIGHT_HIGH,
  HIGHLIGHT_LOW,
  LUMA,
  type Luma,
  NOISE_FLOOR,
  Stabiliser,
} from './maths';
import { SmallCopier, type SmallCopy } from './small';

export type Look = 'subtle' | 'contrast' | 'colour';

/** Counters a test or a measurement reads off the canvas element (never written to the DOM). */
export interface EnhanceStats {
  /** Frames worked on. */
  frames: number;
  /** Milliseconds the page's thread spent in draw, summed. */
  ms: number;
  resets: number;
  /** Frames whose highlight was not redrawn because the graphics card had not finished the one before. */
  late: number;
  /** Frames passed over whole because the work was taking too much of each frame's time. */
  skipped: number;
  /** One frame in this many is worked on (1: every frame). */
  every: number;
}

export type EnhanceCanvas = HTMLCanvasElement & { enhanceStats?: EnhanceStats };

/** The band each look highlights, and the state's long side (its spatial blur). */
const BANDS: Record<Look, { low: number; high: number; size: number }> = {
  // Change: from a slow drift (a quarter hertz) to a quick movement (2.5 Hz)
  subtle: { low: 0.25, high: 2.5, size: 320 },
  contrast: { low: 0.25, high: 2.5, size: 320 },
  // Colour: around a resting pulse (0.8 to 2 Hz), heavily blurred, as Eulerian magnification needs
  colour: { low: 0.8, high: 2.0, size: 64 },
};
/** With reduced motion, the fast average is slower, so the highlight eases rather than flickers. */
const CALM_FAST_TAU = 0.25;
/** How much the colour look amplifies the band-passed colour. */
const COLOUR_GAIN = 40;
/** The overlay's long side is at most this. The highlight is a blurred field, so the page scaling it up shows
 *  nothing. */
const OVERLAY_MAX = 960;
/** The Contrast look greys the picture by this much with a CSS filter on the plain canvas (Enhance.tsx), since an
 *  overlay can dim a picture but not take its colour out. */
export const CONTRAST_SATURATION = 0.5;
/** Enhance may hold the page's thread for this share of the time between frames. Above it, frames are passed over. */
const BUDGET = 0.5;
/** The most frames passed over between two worked on: with BUDGET, a worked frame then takes no longer than the
 *  time between two frames, so the frame after it is never held up. If that is still too slow, Enhance stops and
 *  says so. */
const EVERY_MOST = 2;
/** Frames between two looks at the load. */
const REVIEW = 30;

const VERTEX = `#version 300 es
in vec2 a;
void main() { gl_Position = vec4(a, 0.0, 1.0); }`;

// Positions are image positions p, (0, 0) at the top left: render targets and the canvas have y up, the small copy
// has its top row first, so p.y = 1 - fragCoord.y / height everywhere, and a render target is sampled at
// (p.x, 1 - p.y).
const COMMON = `#version 300 es
precision highp float;
const vec3 LUMA = vec3(${LUMA.join(', ')});
uniform sampler2D small;
uniform vec2 shift;
uniform vec2 size;
uniform float lod; // the small copy's mip level that matches the state's size
// maths.ts edgeMask: no highlight where a wobble brings in pixels from beyond the frame's edge
float edgeMask(vec2 p) {
  vec2 m = abs(shift) + ${EDGE_MARGIN.toFixed(3)};
  vec2 a = smoothstep(m, m + ${EDGE_RAMP.toFixed(3)}, p) * smoothstep(m, m + ${EDGE_RAMP.toFixed(3)}, 1.0 - p);
  return a.x * a.y;
}
`;

const STATE_PASS = `${COMMON}
uniform sampler2D fastIn;
uniform sampler2D slowIn;
uniform float af;
uniform float as;
layout(location = 0) out vec4 fastOut;
layout(location = 1) out vec4 slowOut;
void main() {
  vec2 p = vec2(gl_FragCoord.x, size.y - gl_FragCoord.y) / size;
  vec3 c = textureLod(small, p + shift, lod).rgb;
  ivec2 t = ivec2(gl_FragCoord.xy);
  fastOut = vec4(mix(texelFetch(fastIn, t, 0).rgb, c, af), 1.0);
  slowOut = vec4(mix(texelFetch(slowIn, t, 0).rgb, c, as), 1.0);
}`;

const STRENGTH_PASS = `${COMMON}
uniform sampler2D fastIn;
uniform sampler2D slowIn;
out vec4 o;
float slowLuma(ivec2 t) { return dot(texelFetch(slowIn, t, 0).rgb, LUMA); }
// How strongly the change at each state sample shows (maths.ts changeStrength, the same sums)
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  ivec2 top = textureSize(slowIn, 0) - 1;
  vec3 band = texelFetch(fastIn, t, 0).rgb - texelFetch(slowIn, t, 0).rgb;
  // Local contrast: half the range of the slow average over the 3 by 3 around (a step edge's gradient, and a thin
  // line's too, which a central difference reads as flat)
  float lo = 1.0;
  float hi = 0.0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
      float l = slowLuma(clamp(t + ivec2(dx, dy), ivec2(0), top));
      lo = min(lo, l);
      hi = max(hi, l);
    }
  float n = abs(dot(band, LUMA)) / (${NOISE_FLOOR.toFixed(4)} + ${EDGE_WEIGHT.toFixed(4)} * 0.5 * (hi - lo));
  // Masked before the mipmaps, so the glow's blur can't carry the edge back in
  vec2 p = vec2(gl_FragCoord.x, size.y - gl_FragCoord.y) / size;
  o = vec4(smoothstep(${HIGHLIGHT_LOW.toFixed(3)}, ${HIGHLIGHT_HIGH.toFixed(3)}, n) * edgeMask(p), 0.0, 0.0, 1.0);
}`;

// The output is premultiplied: the page shows picture * (1 - o.a) + o.rgb, so o.rgb above o.a adds light.
const OVERLAY_PASS = `${COMMON}
uniform sampler2D fastIn;
uniform sampler2D slowIn;
uniform sampler2D strengthIn;
uniform int look;
uniform float gain;
out vec4 o;
const vec3 GLOW = vec3(1.0, 0.72, 0.22);
void main() {
  vec2 p = vec2(gl_FragCoord.x, size.y - gl_FragCoord.y) / size;
  vec2 t = vec2(p.x, 1.0 - p.y);
  float inside = edgeMask(p) * gain;
  if (look == 2) {
    // Linear Eulerian colour: amplify the band-passed colour, mostly its chroma, and add it to the picture. What
    // it takes away is taken as a share of the picture (alpha), reckoned on the small copy.
    vec3 band = texture(fastIn, t).rgb - texture(slowIn, t).rgb;
    float bl = dot(band, LUMA);
    vec3 d = inside * ${COLOUR_GAIN.toFixed(1)} * (band - bl + 0.2 * bl);
    vec3 v = max(textureLod(small, p + shift, lod).rgb, 1.0 / 255.0);
    vec3 share = -d / v;
    float a = clamp(max(max(share.r, share.g), share.b), 0.0, 1.0);
    o = vec4(max(d + a * v, 0.0), a);
    return;
  }
  // A moving edge changes a thin line of samples: blurred copies (mip levels down) widen it into a glow
  float s0 = texture(strengthIn, t).r;
  float s2 = textureLod(strengthIn, t, 2.0).r;
  if (look == 0) {
    // Subtle: the picture as it is, with a soft amber glow, wide enough to find, that shows at least at about a third
    // once anything passes the threshold, and lifts the picture under it so it reads on warm and bright ground too
    float h = clamp(max(max(s0, 3.0 * s2), 4.0 * textureLod(strengthIn, t, 3.0).r), 0.0, 1.0) * inside;
    float shown = smoothstep(0.02, 0.12, h);
    float a = clamp(0.35 * shown + 0.5 * h, 0.0, 0.85);
    o = vec4(GLOW * a, 1.0 - (1.0 + 0.15 * shown) * (1.0 - a));
  } else {
    // Contrast: everything still is dimmed and greyed; what changes is lit. The greying is the CSS filter on the
    // plain picture (CONTRAST_SATURATION); where it is lit, the colour the filter took is put back from the small
    // copy.
    float h = clamp(max(s0, 3.0 * s2), 0.0, 1.0) * inside;
    vec3 v = textureLod(small, p + shift, lod).rgb;
    vec3 colour = (1.0 - ${CONTRAST_SATURATION.toFixed(2)}) * (v - dot(v, LUMA));
    vec3 lit = 1.25 * (0.35 * GLOW + 0.65 * colour);
    o = vec4(max(mix(vec3(0.02), lit, h), 0.0), 1.0 - mix(0.35, 1.25 * 0.65, h));
  }
}`;

function compile(gl: WebGL2RenderingContext, fragment: string): WebGLProgram {
  const program = gl.createProgram();
  for (const [type, source] of [
    [gl.VERTEX_SHADER, VERTEX],
    [gl.FRAGMENT_SHADER, fragment],
  ] as const) {
    const shader = gl.createShader(type);
    if (!shader) throw new Error('WebGL2 could not create a shader.');
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
      throw new Error(`Enhance shader: ${gl.getShaderInfoLog(shader) ?? 'did not compile'}`);
    gl.attachShader(program, shader);
    gl.deleteShader(shader); // freed with its program
  }
  gl.bindAttribLocation(program, 0, 'a');
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    throw new Error(`Enhance shader: ${gl.getProgramInfoLog(program) ?? 'did not link'}`);
  return program;
}

const UNIFORMS = [
  'small',
  'fastIn',
  'slowIn',
  'strengthIn',
  'shift',
  'size',
  'lod',
  'af',
  'as',
  'look',
  'gain',
] as const;
type Uniforms = Record<(typeof UNIFORMS)[number], WebGLUniformLocation | null>;

/** A pass: its program, its uniforms (looked up once), and its samplers pointed at their texture units. */
interface Pass {
  program: WebGLProgram;
  at: Uniforms;
}

interface Target {
  fast: WebGLTexture;
  slow: WebGLTexture;
  fbo: WebGLFramebuffer;
}

/** Thrown by draw when the work takes too long even on every other frame: the plain picture is better. */
export class TooSlow extends Error {
  constructor() {
    super("this computer can't keep up with this camera's video");
  }
}

/** Draws the highlight for a camera's frames onto one canvas, and says how far to shift the plain picture. Throws
 *  from the constructor where WebGL2 or half-float targets are missing. */
export class Enhancer {
  private readonly gl: WebGL2RenderingContext;
  private readonly passes: { state: Pass; strength: Pass; overlay: Pass };
  private readonly stabiliser = new Stabiliser();
  private readonly corners: WebGLBuffer;
  private disposed = false;
  private frameSize = { width: 0, height: 0 };
  /** Makes the small copy of a frame that arrives without one. */
  private readonly copier = new SmallCopier((width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  });
  private smallSize = { width: 0, height: 0 };
  private smallTexture: WebGLTexture | null = null;
  private smallLod = 0;
  private targets: [Target, Target] | null = null;
  /** Each state sample's highlight strength, with four mip levels for the glow. */
  private strength: { texture: WebGLTexture; fbo: WebGLFramebuffer } | null = null;
  private stateSize = { width: 0, height: 0 };
  private current = 0;
  /** The capture time of the last frame the stabiliser saw, and of the last one the averages took. */
  private lastAt = Number.NaN;
  private stateAt = Number.NaN;
  private look: Look = 'subtle';
  private calm = false;
  private needsReset = true;
  /** Set once the graphics card has been sent a frame, until it reports the frame done. */
  private pending: WebGLSync | null = null;
  private gain = 0;
  private count = 0;
  private workMs = 0;
  private worked = 0;
  private gapS = 0;
  private gaps = 0;
  private over = 0;
  readonly stats: EnhanceStats = { frames: 0, ms: 0, resets: 0, late: 0, skipped: 0, every: 1 };
  /** Where to sample the current frame for each position shown, as a fraction of the frame: the plain picture is
   *  moved by minus this. */
  readonly shift = { x: 0, y: 0 };
  /** Whether to work on fewer frames when they take too long. A measurement that waits inside draw turns it off. */
  adapt = true;
  /** Called at the end of each stage of draw with its name, by a measurement that wants the split. */
  mark?: (name: string) => void;

  private readonly canvas: EnhanceCanvas;

  constructor(canvas: EnhanceCanvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { alpha: true, antialias: false, depth: false, premultipliedAlpha: true });
    if (!gl) throw new Error('This browser has no WebGL2.');
    if (!gl.getExtension('EXT_color_buffer_float') && !gl.getExtension('EXT_color_buffer_half_float'))
      throw new Error('This browser cannot render to half-float textures.');
    this.gl = gl;
    this.passes = {
      state: this.pass(STATE_PASS),
      strength: this.pass(STRENGTH_PASS),
      overlay: this.pass(OVERLAY_PASS),
    };
    this.corners = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.corners);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    canvas.enhanceStats = this.stats;
  }

  /** Pick the look; a different band starts the averages over. */
  setLook(look: Look, calm: boolean): void {
    if (look === this.look && calm === this.calm) return;
    if (BANDS[look].size !== BANDS[this.look].size || BANDS[look].low !== BANDS[this.look].low) this.needsReset = true;
    this.look = look;
    this.calm = calm;
  }

  /** Take one frame. at: its capture time in seconds; flip: the picture arrives upside down (its camera's
   *  video.flip); copy: the frame's small copy and luma, upright, when its source made them (live video's player
   *  worker does). Afterwards `shift` says where the plain picture belongs. Throws TooSlow when this computer cannot
   *  keep up even at the lowest rate. */
  draw(
    frame: CanvasImageSource & { width: number; height: number },
    at: number,
    flip = false,
    copy?: SmallCopy | null,
  ): void {
    const gl = this.gl;
    if (this.disposed || gl.isContextLost()) return;
    const { width, height } = frame;
    if (!width || !height) return;
    // Under load, one frame in `every` is worked on; the others keep the last shift and highlight
    this.count += 1;
    if (this.stats.every > 1 && this.count % this.stats.every !== 0) {
      this.stats.skipped += 1;
      return;
    }
    const started = performance.now();
    this.prepare(width, height);
    if (!this.targets) return;

    // 1. The small copy and its luma: the frame's source made them, or they are made here (2D canvases)
    let picture: TexImageSource;
    let luma: Luma;
    const { smallSize } = this;
    if (
      copy &&
      copy.picture.width === smallSize.width &&
      copy.picture.height === smallSize.height &&
      copy.luma.length === copy.width * copy.height
    ) {
      picture = copy.picture;
      luma = { data: copy.luma, width: copy.width, height: copy.height };
    } else {
      const made = this.copier.copy(frame, width, height, flip);
      picture = made.canvas;
      luma = made.luma;
    }
    this.mark?.('small copy');

    // 2. Stabilise
    const dt = Number.isFinite(this.lastAt) ? at - this.lastAt : Number.POSITIVE_INFINITY;
    this.lastAt = at;
    const stable = this.stabiliser.update(luma, dt);
    if (stable.reset) this.needsReset = true;
    this.shift.x = stable.dx;
    this.shift.y = stable.dy;
    this.gain = stable.gain;
    this.mark?.('motion estimate');

    // 3 to 5. The highlight, unless the graphics card is still on the frame before: then the last highlight stays
    if (this.pending && gl.clientWaitSync(this.pending, 0, 0) === gl.TIMEOUT_EXPIRED) this.stats.late += 1;
    else {
      if (this.pending) gl.deleteSync(this.pending);
      this.pending = null;
      this.highlight(picture, at);
    }

    const ms = performance.now() - started;
    this.stats.frames += 1;
    this.stats.ms += ms;
    this.review(ms, dt);
  }

  /** Upload the small copy, update the averages, and draw the look. */
  private highlight(small: TexImageSource, at: number): void {
    const gl = this.gl;
    const targets = this.targets;
    if (!targets) return;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.smallTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, small);
    gl.generateMipmap(gl.TEXTURE_2D);
    this.mark?.('upload');

    const reset = this.needsReset;
    if (reset) this.stats.resets += 1;
    this.needsReset = false;
    const dt = Number.isFinite(this.stateAt) ? at - this.stateAt : Number.POSITIVE_INFINITY;
    this.stateAt = at;
    const band = BANDS[this.look];
    const taus = bandTaus(band.low, band.high);
    const fastTau = this.calm && this.look !== 'colour' ? CALM_FAST_TAU : taus.fast;
    const from = targets[this.current] as Target;
    const to = targets[1 - this.current] as Target;
    const { x, y } = this.shift;
    const { width, height } = this.stateSize;

    // 3. The running averages
    gl.bindFramebuffer(gl.FRAMEBUFFER, to.fbo);
    gl.viewport(0, 0, width, height);
    const state = this.select(this.passes.state);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, from.fast);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, from.slow);
    gl.uniform2f(state.shift, x, y);
    gl.uniform2f(state.size, width, height);
    gl.uniform1f(state.lod, this.smallLod);
    gl.uniform1f(state.af, reset ? 1 : emaAlpha(dt, fastTau));
    gl.uniform1f(state.as, reset ? 1 : emaAlpha(dt, taus.slow));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.current = 1 - this.current;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, to.fast);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, to.slow);
    this.mark?.('filter');

    // 4. How strongly each sample's change shows, with mipmaps for the glow (the colour look doesn't use it)
    const strength = this.strength;
    if (this.look !== 'colour' && strength) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, strength.fbo);
      const at4 = this.select(this.passes.strength);
      gl.uniform2f(at4.shift, x, y);
      gl.uniform2f(at4.size, width, height);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, strength.texture);
      gl.generateMipmap(gl.TEXTURE_2D);
    }
    this.mark?.('strength');

    // 5. The look
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    const overlay = this.select(this.passes.overlay);
    gl.uniform2f(overlay.shift, x, y);
    gl.uniform2f(overlay.size, this.canvas.width, this.canvas.height);
    gl.uniform1f(overlay.lod, this.smallLod);
    gl.uniform1i(overlay.look, this.look === 'subtle' ? 0 : this.look === 'contrast' ? 1 : 2);
    gl.uniform1f(overlay.gain, this.gain);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.pending = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    this.mark?.('composite');
  }

  /** Every REVIEW frames worked on: if they are taking more than BUDGET of the time between frames, work on every
   *  other one. If that is not enough either, the picture is better off plain, and the caller is told. */
  private review(ms: number, dt: number): void {
    this.workMs += ms;
    this.worked += 1;
    if (dt > 0 && dt < 1) {
      this.gapS += dt;
      this.gaps += 1;
    }
    if (this.worked < REVIEW || !this.adapt) return;
    const each = this.workMs / this.worked;
    // dt spans `every` frames while frames are being passed over
    const between = this.gaps ? ((this.gapS / this.gaps) * 1000) / this.stats.every : 0;
    this.workMs = 0;
    this.worked = 0;
    this.gapS = 0;
    this.gaps = 0;
    if (!(between > 0)) return;
    const wanted = Math.ceil(each / (BUDGET * between));
    // Too slow twice running, already at the lowest rate: not a passing hiccup
    this.over = wanted > EVERY_MOST && this.stats.every === EVERY_MOST ? this.over + 1 : 0;
    if (this.over >= 2) throw new TooSlow();
    this.stats.every = Math.min(EVERY_MOST, Math.max(1, wanted));
  }

  /** Size the canvases and textures for a frame of width by height, when it or the look's size changes. */
  private prepare(width: number, height: number): void {
    const gl = this.gl;
    const band = BANDS[this.look];
    const scale = band.size / Math.max(width, height);
    const state = { width: Math.max(2, Math.round(width * scale)), height: Math.max(2, Math.round(height * scale)) };
    const sameFrame = width === this.frameSize.width && height === this.frameSize.height;
    if (sameFrame && state.width === this.stateSize.width && state.height === this.stateSize.height) return;

    if (!sameFrame) {
      this.frameSize = { width, height };
      this.smallSize = SmallCopier.size(width, height);
      const out = Math.min(1, OVERLAY_MAX / Math.max(width, height));
      this.canvas.width = Math.round(width * out);
      this.canvas.height = Math.round(height * out);
      this.stabiliser.reset();
    }

    // The small copy on the graphics card, with mipmaps: the state samples it at its own size, between two levels,
    // so what it averages is already blurred to that size. Sampled sharp, fine detail flickers with every sub-pixel
    // wobble and lights (measured: up to 3.5 % of the picture, against 0.15 %).
    const small = this.smallSize;
    if (this.smallTexture) gl.deleteTexture(this.smallTexture);
    this.smallTexture = gl.createTexture();
    this.smallLod = Math.max(0, Math.log2(Math.max(small.width, small.height) / band.size));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.smallTexture);
    const levels = Math.floor(Math.log2(Math.max(small.width, small.height))) + 1;
    gl.texStorage2D(gl.TEXTURE_2D, levels, gl.RGBA8, small.width, small.height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    for (const t of this.targets ?? []) {
      gl.deleteTexture(t.fast);
      gl.deleteTexture(t.slow);
      gl.deleteFramebuffer(t.fbo);
    }
    const target = (): Target => {
      const texture = () => {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, state.width, state.height);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return t;
      };
      const fast = texture();
      const slow = texture();
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fast, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, slow, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('This browser cannot render the running averages.');
      return { fast, slow, fbo };
    };
    this.targets = [target(), target()];
    if (this.strength) {
      gl.deleteTexture(this.strength.texture);
      gl.deleteFramebuffer(this.strength.fbo);
    }
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 4, gl.R8, state.width, state.height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    // The overlay samples it before the first strength pass has run only in the colour look, which doesn't
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    this.strength = { texture, fbo };
    this.stateSize = state;
    this.current = 0;
    this.needsReset = true;
  }

  /** A program with its uniforms looked up and its samplers set, once. */
  private pass(fragment: string): Pass {
    const gl = this.gl;
    const program = compile(gl, fragment);
    const at = {} as Uniforms;
    for (const name of UNIFORMS) at[name] = gl.getUniformLocation(program, name);
    const pass = { program, at };
    this.select(pass);
    gl.uniform1i(at.small, 0);
    gl.uniform1i(at.fastIn, 1);
    gl.uniform1i(at.slowIn, 2);
    gl.uniform1i(at.strengthIn, 3);
    return pass;
  }

  /** gl.useProgram, in one place: Biome takes any use* call for a React hook. */
  private select(pass: Pass): Uniforms {
    // biome-ignore lint/correctness/useHookAtTopLevel: WebGL's useProgram, not a React hook
    this.gl.useProgram(pass.program);
    return pass.at;
  }

  /** Free the GPU memory now rather than when the canvas is collected. Everything made here is deleted. The context
   *  itself is given up only once the canvas has left the page: a lost context can never be used again, and a canvas
   *  still on the page may be handed to a new Enhancer (React's development double-mount does exactly that, and
   *  Enhance then reported itself unavailable). */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    for (const pass of Object.values(this.passes)) gl.deleteProgram(pass.program);
    gl.deleteBuffer(this.corners);
    if (this.pending) gl.deleteSync(this.pending);
    if (this.smallTexture) gl.deleteTexture(this.smallTexture);
    for (const t of this.targets ?? []) {
      gl.deleteTexture(t.fast);
      gl.deleteTexture(t.slow);
      gl.deleteFramebuffer(t.fbo);
    }
    if (this.strength) {
      gl.deleteTexture(this.strength.texture);
      gl.deleteFramebuffer(this.strength.fbo);
    }
    this.pending = null;
    this.smallTexture = null;
    this.targets = null;
    this.strength = null;
    if (!this.canvas.isConnected) gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
