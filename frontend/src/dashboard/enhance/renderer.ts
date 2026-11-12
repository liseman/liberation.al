// Live's Enhance view on the GPU: one WebGL2 canvas over the large camera's picture, fed the frames the page already
// decodes (camera/mux.ts hands each one over before drawing it). Per frame:
//
//   1. upload the frame and its mipmaps, and read back a 256-wide luma copy (four lumas packed per texel);
//   2. Stabiliser (maths.ts) estimates the shift against its reference frame on the CPU;
//   3. a state pass, at a reduced size (which is also the spatial low-pass), samples the frame shifted back and
//      updates two running averages per pixel into half-float targets (ping-pong, two outputs at once);
//   4. a strength pass, at the same size, divides the band-pass (fast minus slow) by the local contrast of the slow
//      average (half its range over 3 by 3; maths.ts changeStrength, the same sums), with mipmaps so the display can
//      widen it into a glow;
//   5. a display pass draws the shifted frame in the chosen look.
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
  lumaSize,
  NOISE_FLOOR,
  Stabiliser,
} from './maths';

export type Look = 'subtle' | 'contrast' | 'colour';

/** Counters a test or a measurement reads off the canvas element (never written to the DOM). */
export interface EnhanceStats {
  frames: number;
  /** Milliseconds spent in draw, summed: upload, readback, motion estimate, and both passes. */
  ms: number;
  resets: number;
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
/** The largest frame drawn: a 4K camera is shown at 1080p's long side. */
const MAX_OUT = 1920;

/** Uniforms set as integers: the samplers' texture units and the look. */
const INTEGERS = new Set(['video', 'fastIn', 'slowIn', 'strengthIn', 'look']);

const VERTEX = `#version 300 es
in vec2 a;
void main() { gl_Position = vec4(a, 0.0, 1.0); }`;

// Positions are image positions p, (0, 0) at the top left: render targets and the canvas have y up, the uploaded frame
// has its top row first, so p.y = 1 - fragCoord.y / height everywhere, and a render target is sampled at (p.x, 1 - p.y).
const COMMON = `#version 300 es
precision highp float;
const vec3 LUMA = vec3(${LUMA.join(', ')});
uniform sampler2D video;
uniform bool flip;
uniform vec2 shift;
vec2 frameAt(vec2 p) { vec2 s = p + shift; return flip ? 1.0 - s : s; }
// maths.ts edgeMask: no highlight where a wobble brings in pixels from beyond the frame's edge
float edgeMask(vec2 p) {
  vec2 m = abs(shift) + ${EDGE_MARGIN.toFixed(3)};
  vec2 a = smoothstep(m, m + ${EDGE_RAMP.toFixed(3)}, p) * smoothstep(m, m + ${EDGE_RAMP.toFixed(3)}, 1.0 - p);
  return a.x * a.y;
}
`;

const LUMA_PASS = `${COMMON}
uniform vec2 size; // the luma copy, in lumas
uniform float lod;
out vec4 o;
float l(float x, float y) {
  vec2 p = vec2((x + 0.5) / size.x, y / size.y);
  return dot(textureLod(video, flip ? 1.0 - p : p, lod).rgb, LUMA);
}
void main() {
  float x0 = floor(gl_FragCoord.x) * 4.0;
  float y = size.y - gl_FragCoord.y;
  o = vec4(l(x0, y), l(x0 + 1.0, y), l(x0 + 2.0, y), l(x0 + 3.0, y));
}`;

const STATE_PASS = `${COMMON}
uniform sampler2D fastIn;
uniform sampler2D slowIn;
uniform vec2 size;
uniform float lod;
uniform float af;
uniform float as;
layout(location = 0) out vec4 fastOut;
layout(location = 1) out vec4 slowOut;
void main() {
  vec2 p = vec2(gl_FragCoord.x, size.y - gl_FragCoord.y) / size;
  vec3 c = textureLod(video, frameAt(p), lod).rgb;
  ivec2 t = ivec2(gl_FragCoord.xy);
  fastOut = vec4(mix(texelFetch(fastIn, t, 0).rgb, c, af), 1.0);
  slowOut = vec4(mix(texelFetch(slowIn, t, 0).rgb, c, as), 1.0);
}`;

const STRENGTH_PASS = `${COMMON}
uniform sampler2D fastIn;
uniform sampler2D slowIn;
uniform vec2 size;
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

const DISPLAY_PASS = `${COMMON}
uniform sampler2D fastIn;
uniform sampler2D slowIn;
uniform sampler2D strengthIn;
uniform vec2 size;
uniform vec2 texel; // one state sample
uniform int look;
uniform float gain;
uniform float colourGain;
out vec4 o;
const vec3 GLOW = vec3(1.0, 0.72, 0.22);
void main() {
  vec2 p = vec2(gl_FragCoord.x, size.y - gl_FragCoord.y) / size;
  vec3 v = texture(video, frameAt(p)).rgb;
  vec2 t = vec2(p.x, 1.0 - p.y);
  float inside = edgeMask(p) * gain;
  if (look == 2) {
    // Linear Eulerian colour: amplify the band-passed colour, mostly its chroma, and add it back
    vec3 band = texture(fastIn, t).rgb - texture(slowIn, t).rgb;
    float bl = dot(band, LUMA);
    o = vec4(clamp(v + inside * colourGain * (band - bl + 0.2 * bl), 0.0, 1.0), 1.0);
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
    vec3 lifted = min(v * (1.0 + 0.15 * shown), 1.0);
    o = vec4(mix(lifted, GLOW, a), 1.0);
  } else {
    // Contrast: everything still is dimmed and greyed; what changes is lit
    float h = clamp(max(s0, 3.0 * s2), 0.0, 1.0) * inside;
    float y = dot(v, LUMA);
    vec3 ground = mix(vec3(y), v, 0.5) * 0.35 + 0.02;
    vec3 lit = min(mix(v, GLOW, 0.35) * 1.25, 1.0);
    o = vec4(mix(ground, lit, h), 1.0);
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
  }
  gl.bindAttribLocation(program, 0, 'a');
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    throw new Error(`Enhance shader: ${gl.getProgramInfoLog(program) ?? 'did not link'}`);
  return program;
}

/** Whether this browser can run Enhance: WebGL2, with half-float render targets for the running averages. */
export function enhanceSupported(): boolean {
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return false;
    const ok = !!(gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float'));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return ok;
  } catch {
    return false;
  }
}

interface Target {
  fast: WebGLTexture;
  slow: WebGLTexture;
  fbo: WebGLFramebuffer;
}

/** Draws enhanced frames onto one canvas. Throws from the constructor where WebGL2 or half-float targets are missing. */
export class Enhancer {
  private readonly gl: WebGL2RenderingContext;
  private readonly programs: { luma: WebGLProgram; state: WebGLProgram; strength: WebGLProgram; display: WebGLProgram };
  private readonly stabiliser = new Stabiliser();
  private readonly lumaFbo: WebGLFramebuffer;
  private readonly lumaTex: WebGLTexture;
  private video: WebGLTexture | null = null;
  private videoSize = { width: 0, height: 0 };
  private lumaDims = { width: 0, height: 0 };
  private lumaBytes = new Uint8Array(0);
  private lumaFloats = new Float32Array(0);
  private targets: [Target, Target] | null = null;
  /** Each state sample's highlight strength, with four mip levels for the glow. */
  private strength: { texture: WebGLTexture; fbo: WebGLFramebuffer } | null = null;
  private stateSize = { width: 0, height: 0 };
  private current = 0;
  private lastAt = Number.NaN;
  private look: Look = 'subtle';
  private calm = false;
  private needsReset = true;
  private readonly locations = new Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>();
  readonly stats: EnhanceStats = { frames: 0, ms: 0, resets: 0 };

  private readonly canvas: EnhanceCanvas;

  constructor(canvas: EnhanceCanvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false, premultipliedAlpha: false });
    if (!gl) throw new Error('This browser has no WebGL2.');
    if (!gl.getExtension('EXT_color_buffer_float') && !gl.getExtension('EXT_color_buffer_half_float'))
      throw new Error('This browser cannot render to half-float textures.');
    this.gl = gl;
    this.programs = {
      luma: compile(gl, LUMA_PASS),
      state: compile(gl, STATE_PASS),
      strength: compile(gl, STRENGTH_PASS),
      display: compile(gl, DISPLAY_PASS),
    };
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.lumaTex = gl.createTexture();
    this.lumaFbo = gl.createFramebuffer();
    canvas.enhanceStats = this.stats;
  }

  /** Pick the look; a different band starts the averages over. */
  setLook(look: Look, calm: boolean): void {
    if (look === this.look && calm === this.calm) return;
    if (BANDS[look].size !== BANDS[this.look].size || BANDS[look].low !== BANDS[this.look].low) this.needsReset = true;
    this.look = look;
    this.calm = calm;
  }

  /** Draw one frame. at: its capture time in seconds; flip: the picture arrives upside down (its camera's video.flip). */
  draw(frame: TexImageSource & { width: number; height: number }, at: number, flip = false): void {
    const started = performance.now();
    const gl = this.gl;
    if (gl.isContextLost()) return;
    const { width, height } = frame;
    if (!width || !height) return;
    this.prepare(width, height);
    const video = this.video;
    if (!video || !this.targets) return;

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, video);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, frame);
    gl.generateMipmap(gl.TEXTURE_2D);

    // 1. The luma copy, read back for the motion estimate
    const lw = this.lumaDims.width;
    const lh = this.lumaDims.height;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.lumaFbo);
    gl.viewport(0, 0, lw / 4, lh);
    this.select(this.programs.luma);
    this.uniforms(this.programs.luma, { flip, size: [lw, lh], lod: Math.log2(width / lw), video: 0 });
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.readPixels(0, 0, lw / 4, lh, gl.RGBA, gl.UNSIGNED_BYTE, this.lumaBytes);
    // Rows come back bottom first
    for (let r = 0; r < lh; r++) {
      const from = r * lw;
      const to = (lh - 1 - r) * lw;
      for (let x = 0; x < lw; x++) this.lumaFloats[to + x] = (this.lumaBytes[from + x] ?? 0) / 255;
    }

    // 2. Stabilise
    const dt = Number.isFinite(this.lastAt) ? at - this.lastAt : Number.POSITIVE_INFINITY;
    this.lastAt = at;
    const stable = this.stabiliser.update({ data: this.lumaFloats, width: lw, height: lh }, dt);
    const reset = stable.reset || this.needsReset;
    if (reset) this.stats.resets += 1;
    this.needsReset = false;

    // 3. The running averages
    const band = BANDS[this.look];
    const taus = bandTaus(band.low, band.high);
    const fastTau = this.calm && this.look !== 'colour' ? CALM_FAST_TAU : taus.fast;
    const from = this.targets[this.current] as Target;
    const to = this.targets[1 - this.current] as Target;
    gl.bindFramebuffer(gl.FRAMEBUFFER, to.fbo);
    gl.viewport(0, 0, this.stateSize.width, this.stateSize.height);
    this.select(this.programs.state);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, from.fast);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, from.slow);
    this.uniforms(this.programs.state, {
      video: 0,
      fastIn: 1,
      slowIn: 2,
      flip,
      shift: [stable.dx, stable.dy],
      size: [this.stateSize.width, this.stateSize.height],
      lod: Math.log2(width / this.stateSize.width),
      af: reset ? 1 : emaAlpha(dt, fastTau),
      as: reset ? 1 : emaAlpha(dt, taus.slow),
    });
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.current = 1 - this.current;

    // 4. How strongly each sample's change shows, with mipmaps for the glow (the colour look doesn't use it)
    const strength = this.strength;
    if (this.look !== 'colour' && strength) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, strength.fbo);
      this.select(this.programs.strength);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, to.fast);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, to.slow);
      this.uniforms(this.programs.strength, {
        fastIn: 1,
        slowIn: 2,
        shift: [stable.dx, stable.dy],
        size: [this.stateSize.width, this.stateSize.height],
      });
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, strength.texture);
      gl.generateMipmap(gl.TEXTURE_2D);
    }

    // 5. The picture
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.select(this.programs.display);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, to.fast);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, to.slow);
    this.uniforms(this.programs.display, {
      video: 0,
      fastIn: 1,
      slowIn: 2,
      strengthIn: 3,
      flip,
      shift: [stable.dx, stable.dy],
      size: [this.canvas.width, this.canvas.height],
      texel: [1 / this.stateSize.width, 1 / this.stateSize.height],
      look: this.look === 'subtle' ? 0 : this.look === 'contrast' ? 1 : 2,
      gain: stable.gain,
      colourGain: COLOUR_GAIN,
    });
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.stats.frames += 1;
    this.stats.ms += performance.now() - started;
  }

  /** Size the textures for a frame of width by height, when it changes (and the first time). */
  private prepare(width: number, height: number): void {
    const gl = this.gl;
    const band = BANDS[this.look];
    const scale = band.size / Math.max(width, height);
    const state = { width: Math.max(2, Math.round(width * scale)), height: Math.max(2, Math.round(height * scale)) };
    const sameVideo = width === this.videoSize.width && height === this.videoSize.height;
    if (sameVideo && state.width === this.stateSize.width && state.height === this.stateSize.height) return;

    if (!sameVideo) {
      if (this.video) gl.deleteTexture(this.video);
      this.video = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.video);
      const levels = Math.floor(Math.log2(Math.max(width, height))) + 1;
      gl.texStorage2D(gl.TEXTURE_2D, levels, gl.RGBA8, width, height);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.videoSize = { width, height };

      this.lumaDims = lumaSize(width, height);
      gl.bindTexture(gl.TEXTURE_2D, this.lumaTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA8,
        this.lumaDims.width / 4,
        this.lumaDims.height,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
      );
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.lumaFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.lumaTex, 0);
      this.lumaBytes = new Uint8Array(this.lumaDims.width * this.lumaDims.height);
      this.lumaFloats = new Float32Array(this.lumaDims.width * this.lumaDims.height);

      const out = Math.min(1, MAX_OUT / Math.max(width, height));
      this.canvas.width = Math.round(width * out);
      this.canvas.height = Math.round(height * out);
      this.stabiliser.reset();
    }

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
    gl.texStorage2D(gl.TEXTURE_2D, 4, gl.RGBA8, state.width, state.height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    this.strength = { texture, fbo };
    this.stateSize = state;
    this.current = 0;
    this.needsReset = true;
  }

  /** gl.useProgram, in one place: Biome takes any use* call for a React hook. */
  private select(program: WebGLProgram): void {
    // biome-ignore lint/correctness/useHookAtTopLevel: WebGL's useProgram, not a React hook
    this.gl.useProgram(program);
  }

  private uniforms(program: WebGLProgram, values: Record<string, number | boolean | [number, number]>): void {
    const gl = this.gl;
    let locations = this.locations.get(program);
    if (!locations) {
      locations = new Map();
      this.locations.set(program, locations);
    }
    for (const [name, value] of Object.entries(values)) {
      let at = locations.get(name);
      if (at === undefined) {
        at = gl.getUniformLocation(program, name);
        locations.set(name, at);
      }
      if (at === null) continue;
      if (Array.isArray(value)) gl.uniform2f(at, value[0], value[1]);
      else if (typeof value === 'boolean') gl.uniform1i(at, value ? 1 : 0);
      else if (INTEGERS.has(name)) gl.uniform1i(at, value);
      else gl.uniform1f(at, value);
    }
  }

  /** Free the GPU memory now rather than when the canvas is collected. */
  dispose(): void {
    this.gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
