// Times Live's Enhance view (frontend/src/dashboard/enhance/renderer.ts) on made-up frames, in a headless browser on
// this machine, and checks what it shows. No camera, no dashboard, and no running service is involved: a worker
// draws a made-up grey scene the way the player worker hands frames over (an OffscreenCanvas turned into an
// ImageBitmap and posted), and the real renderer draws them.
//
//   node tools/enhance_bench.mjs frontend --browser chromium-gpu --preset timing
//   node tools/enhance_bench.mjs frontend --browser firefox --preset paced --sizes 1920x1080@60
//   node tools/enhance_bench.mjs frontend --browser chromium-software --preset behaviour
//   node tools/enhance_bench.mjs frontend --renderer /path/to/another/renderer.ts ...   (to compare two versions)
//   node tools/enhance_bench.mjs frontend --copy page ...   (the renderer makes the small copy, as for JPEG frames)
//   node tools/enhance_bench.mjs frontend --preset behaviour --detail plain   (a scene without hairlines)
//   node tools/enhance_bench.mjs frontend --preset behaviour --pick 3,11   (only those runs, counted from 0)
//
// Browsers: chromium-gpu (Playwright's headless Chromium on the graphics card through ANGLE and Vulkan),
// chromium-software (the same on SwiftShader, a CPU rasteriser: how a machine without a usable graphics card does),
// and firefox (the installed Firefox, headless, in a throwaway profile; it takes its graphics context from the
// desktop session, so it needs WAYLAND_DISPLAY or DISPLAY, and it opens no window).
//
// Presets:
//   timing     frames back to back, waiting for the graphics card after each: milliseconds a frame, then the same
//              again with a wait after each stage, for the split (upload, analysis, filter, composite).
//   paced      frames at the stream's own rate for a few seconds, as Live delivers them: how long the page's thread
//              is held per frame, how late frames arrive, and how many frames' graphics work finished in its slot.
//   behaviour  what the view shows at 960x540 (or the one size given): a still picture, wobble alone, wobble with a
//              small real motion, a knock, and a pan: how much of the picture is lit, in how many frames the moving
//              patch's ground is lit, and how many pixels the steadied picture sits from where the scene is.
//
// Each run prints one JSON line. The graphics cards' utilisation during each run is sampled with nvidia-smi when it
// is installed, because a card that is also running models answers more slowly.
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';

const args = process.argv.slice(2);
const frontend = args[0] && !args[0].startsWith('--') ? args[0] : '';
const option = (name, fallback) => {
  const at = args.indexOf(`--${name}`);
  return at >= 0 && args[at + 1] ? args[at + 1] : fallback;
};
if (!frontend) {
  console.error('usage: node tools/enhance_bench.mjs <frontend directory> [--browser b] [--preset p] [--renderer file]');
  process.exit(2);
}
const need = createRequire(path.resolve(frontend) + path.sep);
const browserName = option('browser', 'chromium-gpu');
const preset = option('preset', 'timing');
const renderer = path.resolve(option('renderer', path.join(frontend, 'src/dashboard/enhance/renderer.ts')));
const sizes = option('sizes', '960x540@30,1280x720@30,1920x1080@30,1920x1080@60')
  .split(',')
  .map(s => /^(\d+)x(\d+)@(\d+)$/.exec(s))
  .filter(Boolean)
  .map(m => ({ width: Number(m[1]), height: Number(m[2]), fps: Number(m[3]) }));
const looks = option('looks', 'subtle,contrast,colour').split(',');
const seconds = Number(option('seconds', '6'));
const firefoxBinary = option('firefox', '/usr/lib/firefox/firefox');
// 'worker' (the frame's source hands over its small copy and luma, as live video does) or 'page' (the renderer
// makes them)
const workerCopy = option('copy', 'worker') === 'worker';
// 'fine' (the scene has hairlines and fine stripes, which is hard on any stabiliser) or 'plain' (blocks and lines)
const fine = option('detail', 'fine') === 'fine';

/** The runs a preset asks for. */
function runsFor() {
  const runs = [];
  if (preset === 'timing')
    for (const size of sizes)
      for (const look of looks) {
        runs.push({ mode: 'sync', ...size, look, frames: 200, scene: 'busy' });
        runs.push({ mode: 'sync', ...size, look, frames: 120, scene: 'busy', stages: true });
      }
  else if (preset === 'paced')
    for (const size of sizes)
      for (const look of looks) runs.push({ mode: 'paced', ...size, look, frames: size.fps * seconds, scene: 'busy' });
  else if (preset === 'behaviour') {
    const at = { mode: 'behaviour', ...(sizes.length === 1 ? sizes[0] : { width: 960, height: 540, fps: 30 }), look: 'subtle' };
    runs.push({ ...at, scene: 'still', frames: 240 });
    for (const amplitude of [0.5, 2, 4, 8, 16]) runs.push({ ...at, scene: 'wobble', amplitude, frames: 300 });
    for (const travel of [1, 2, 3])
      runs.push({ ...at, scene: 'wobble', amplitude: 4, travel, frames: 300 });
    runs.push({ ...at, scene: 'still', travel: 3, frames: 300 });
    runs.push({ ...at, scene: 'knock', frames: 300 });
    runs.push({ ...at, scene: 'pan', frames: 300 });
    runs.push({ ...at, scene: 'wobble', amplitude: 4, travel: 3, frames: 300, look: 'contrast' });
  } else {
    console.error(`unknown preset ${preset}`);
    process.exit(2);
  }
  // --pick 3,4: only those runs of the preset, counted from 0
  const pick = option('pick', '');
  const picked = pick ? pick.split(',').map(i => runs[Number(i)]).filter(Boolean) : runs;
  return picked.map(run => ({ ...run, workerCopy, fine }));
}

// ---- The page: everything below until the server runs in the browser ----

/** The frame source, in a worker: a made-up grey scene, shifted and with a small moving patch, handed over the way
 *  the player worker hands frames over. */
function frameWorker() {
  const MARGIN = 256;
  let base;
  let patch;
  let canvas;
  let context;
  let run;
  let copier = null;
  try {
    importScripts('/small.js');
  } catch {
    // no small.ts beside this renderer
  }
  let seed = 7;
  const random = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const grey = v => `rgb(${v},${v},${v})`;
  /** Where the patch's flat ground sits, in frame pixels. */
  const ground = () => {
    const side = Math.round(run.height * 0.3);
    return { x: Math.round(run.width * 0.58), y: Math.round(run.height * 0.35), side };
  };

  function prepare() {
    const w = run.width + 2 * MARGIN;
    const h = run.height + 2 * MARGIN;
    const scale = run.width / 960;
    base = new OffscreenCanvas(w, h);
    const c = base.getContext('2d');
    const gradient = c.createLinearGradient(0, 0, w, h);
    gradient.addColorStop(0, grey(70));
    gradient.addColorStop(1, grey(150));
    c.fillStyle = gradient;
    c.fillRect(0, 0, w, h);
    for (let i = 0; i < 700; i++) {
      c.fillStyle = grey(Math.round(30 + 200 * random()));
      c.fillRect(random() * w, random() * h, (8 + 90 * random()) * scale, (8 + 70 * random()) * scale);
    }
    c.lineWidth = 2 * scale;
    for (let i = 0; i < 120; i++) {
      c.strokeStyle = grey(Math.round(20 + 220 * random()));
      c.beginPath();
      c.moveTo(random() * w, random() * h);
      c.lineTo(random() * w, random() * h);
      c.stroke();
    }
    // Fine detail, which a careless small copy turns into flicker when the camera wobbles: hairlines and stripes
    c.lineWidth = 1;
    for (let i = 0; run.fine && i < 300; i++) {
      c.strokeStyle = grey(Math.round(20 + 220 * random()));
      c.beginPath();
      c.moveTo(random() * w, random() * h);
      c.lineTo(random() * w, random() * h);
      c.stroke();
    }
    for (let x = 0; run.fine && x < Math.round(w * 0.12); x += 2) {
      c.fillStyle = grey(x % 4 ? 60 : 190);
      c.fillRect(Math.round(w * 0.3) + x, Math.round(h * 0.55), 1, Math.round(h * 0.15));
    }
    const g = ground();
    c.fillStyle = grey(110);
    c.fillRect(MARGIN + g.x, MARGIN + g.y, g.side, g.side);
    const side = Math.round(g.side * 0.4);
    patch = new OffscreenCanvas(side, side);
    const p = patch.getContext('2d');
    const cell = Math.max(2, Math.round(side / 6));
    for (let y = 0; y < side; y += cell)
      for (let x = 0; x < side; x += cell) {
        p.fillStyle = grey(Math.round(40 + 180 * random()));
        p.fillRect(x, y, cell, cell);
      }
    canvas = new OffscreenCanvas(run.width, run.height);
    context = canvas.getContext('2d');
    // A renderer without a small.ts beside it (an older one) takes the frame alone
    const maker = run.workerCopy && self.EnhanceSmall?.SmallCopier;
    copier = maker ? new maker((cw, ch) => new OffscreenCanvas(cw, ch)) : null;
  }

  /** How far the camera has moved at t seconds, in frame pixels. */
  function offset(t) {
    const scale = run.width / 960;
    const a = (run.amplitude ?? 0) * scale;
    if (run.scene === 'wobble' || run.scene === 'busy')
      return [a * Math.sin(2 * Math.PI * 1.7 * t), 0.6 * a * Math.sin(2 * Math.PI * 2.3 * t + 1)];
    if (run.scene === 'knock') return [t >= 3 ? 24 * scale : 0, 0];
    if (run.scene === 'pan') return [192 * scale * Math.min(1, Math.max(0, t - 3)), 0];
    return [0, 0];
  }

  function frame(i) {
    const t = i / run.fps;
    const [ox, oy] = offset(t);
    context.drawImage(base, -MARGIN - ox, -MARGIN - oy);
    if (run.travel) {
      const g = ground();
      const scale = run.width / 960;
      const dx = (run.travel * scale * Math.sin(2 * Math.PI * t)) / 2;
      context.drawImage(patch, g.x + (g.side - patch.width) / 2 + dx - ox, g.y + (g.side - patch.height) / 2 - oy);
    }
    // The small copy and its luma, as the player worker makes them for live video (enhance/small.ts), before the
    // canvas's picture leaves with the frame
    let small;
    if (copier) {
      const made = copier.copy(canvas, run.width, run.height, false, true);
      const picture = made.canvas.transferToImageBitmap();
      small = { picture, luma: made.luma.data, width: made.luma.width, height: made.luma.height };
    }
    const bitmap = canvas.transferToImageBitmap();
    const sent = performance.timeOrigin + performance.now();
    const moved = small ? [bitmap, small.picture, small.luma.buffer] : [bitmap];
    self.postMessage({ type: 'frame', bitmap, small, i, t, ox, oy, sent }, moved);
  }

  self.onmessage = e => {
    const m = e.data;
    if (m.type === 'run') {
      run = m.run;
      if (run.scene === 'busy') {
        run.amplitude = 2;
        run.travel = 3;
      }
      prepare();
      context.drawImage(base, -MARGIN, -MARGIN);
      const still = canvas.transferToImageBitmap();
      self.postMessage({ type: 'ready', still, ground: ground() }, [still]);
    } else if (m.type === 'frame') frame(m.i);
    else if (m.type === 'pace') {
      const started = performance.now();
      let i = 0;
      const tick = () => {
        if (i >= run.frames) return;
        frame(i);
        i += 1;
        setTimeout(tick, Math.max(0, started + (i * 1000) / run.fps - performance.now()));
      };
      tick();
    }
  };
}

/** The page: runs each configured run through the renderer and posts what it measured. */
async function benchPage() {
  const config = await (await fetch('/config')).json();
  const { Enhancer } = window.EnhanceBundle;
  const median = values => {
    if (!values.length) return null;
    const s = [...values].sort((a, b) => a - b);
    return s[s.length >> 1];
  };
  const centile = (values, q) => {
    if (!values.length) return null;
    const s = [...values].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(s.length * q))];
  };
  const round = v => (v === null || v === undefined ? v : Math.round(v * 100) / 100);
  const pixel = new Uint8Array(4);
  /** Wait for the graphics card to finish everything sent so far. */
  const wait = gl => {
    const bound = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    gl.bindFramebuffer(gl.FRAMEBUFFER, bound);
  };

  const info = (() => {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return { renderer: 'no WebGL2' };
    const e = gl.getExtension('WEBGL_debug_renderer_info');
    return {
      renderer: e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      floatTargets: !!gl.getExtension('EXT_color_buffer_float'),
      webgpu: 'gpu' in navigator,
      agent: navigator.userAgent,
    };
  })();
  if ('gpu' in navigator) info.webgpuAdapter = !!(await navigator.gpu.requestAdapter().catch(() => null));
  await fetch('/result', { method: 'POST', body: JSON.stringify({ info }) });

  async function one(run) {
    const worker = new Worker('/worker.js');
    const plain = document.createElement('canvas');
    const canvas = document.createElement('canvas');
    plain.width = run.width;
    plain.height = run.height;
    for (const c of [plain, canvas]) {
      c.style.cssText = 'position:absolute;left:0;top:0;width:960px;height:540px';
      document.body.append(c);
    }
    const plain2d = plain.getContext('2d');
    const enhancer = new Enhancer(canvas);
    enhancer.setLook(run.look, false);
    const gl = canvas.getContext('webgl2');
    const ready = await new Promise(resolve => {
      worker.onmessage = e => resolve(e.data);
      worker.postMessage({ type: 'run', run });
    });
    const started = Date.now();
    const out = { ...run };

    // What is lit is read off a copy of the highlight no more than 960 wide: it is a blurred field
    let probe = null;
    const probeScale = Math.min(1, 960 / run.width);
    const pw = Math.round(run.width * probeScale);
    const ph = Math.round(run.height * probeScale);
    if (run.mode === 'behaviour') {
      const c = document.createElement('canvas');
      c.width = pw;
      c.height = ph;
      probe = c.getContext('2d', { willReadFrequently: true });
    }
    ready.still.close();

    const stages = {};
    let waits = 0;
    let last = 0;
    if (run.stages) enhancer.adapt = false; // the waits would read as load
    if (run.stages)
      enhancer.mark = name => {
        wait(gl);
        const now = performance.now();
        stages[name] = (stages[name] ?? 0) + now - last;
        // What a wait costs with nothing to wait for: each stage's time includes one
        wait(gl);
        last = performance.now();
        stages.wait = (stages.wait ?? 0) + last - now;
        waits += 1;
      };

    const total = [];
    const held = [];
    const plainMs = [];
    const late = [];
    const lit = [];
    const litGround = [];
    const off = [];
    const fences = [];
    const gpuMs = [];
    let frames = 0;
    let firstArrival = 0;
    let lastArrival = 0;
    let animationFrames = 0;
    let worstGap = 0;
    let lastAnimation = 0;
    let watching = true;
    const poll = () => {
      const now = performance.now();
      while (fences.length) {
        const f = fences[0];
        const status = gl.clientWaitSync(f.sync, 0, 0);
        if (status === gl.TIMEOUT_EXPIRED) break;
        gl.deleteSync(f.sync);
        gpuMs.push(now - f.at);
        fences.shift();
      }
    };
    const animate = now => {
      if (!watching) return;
      if (lastAnimation) worstGap = Math.max(worstGap, now - lastAnimation);
      lastAnimation = now;
      animationFrames += 1;
      poll();
      requestAnimationFrame(animate);
    };
    let timer = 0;
    if (run.mode === 'paced') {
      requestAnimationFrame(animate);
      timer = setInterval(poll, 2);
    }

    await new Promise(resolve => {
      worker.onmessage = e => {
        const m = e.data;
        const arrived = performance.now();
        if (!frames) firstArrival = arrived;
        lastArrival = arrived;
        if (run.mode === 'paced') {
          late.push(performance.timeOrigin + arrived - m.sent);
          poll();
        }
        plain2d.drawImage(m.bitmap, 0, 0);
        const t1 = performance.now();
        last = t1;
        enhancer.draw(m.bitmap, m.t, false, m.small);
        const t2 = performance.now();
        if (run.mode === 'paced') fences.push({ sync: gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0), at: t2 });
        else wait(gl);
        const t3 = performance.now();
        m.bitmap.close();
        m.small?.picture.close();
        frames += 1;
        if (frames > 20) {
          plainMs.push(t1 - arrived);
          held.push(t2 - t1);
          total.push(t3 - t1);
        }
        // How far the steadied picture sits from where the scene is, in frame pixels: the camera's true offset
        // plus the shift the renderer applied
        const applied = enhancer.shift ?? enhancer.lastShift;
        if (applied) off.push([m.ox + applied.x * run.width, m.oy + applied.y * run.height]);
        if (run.mode === 'behaviour' && probe) {
          // The highlight alone, over black: a renderer that draws the picture too draws a grey scene, and a pixel
          // with colour in it is the highlight either way
          probe.fillStyle = '#000';
          probe.fillRect(0, 0, pw, ph);
          probe.drawImage(canvas, 0, 0, pw, ph);
          const data = probe.getImageData(0, 0, pw, ph).data;
          const g = ready.ground;
          const gx0 = g.x * probeScale;
          const gx1 = (g.x + g.side) * probeScale;
          const gy0 = g.y * probeScale;
          const gy1 = (g.y + g.side) * probeScale;
          let count = 0;
          let inGround = 0;
          for (let y = 0; y < ph; y++)
            for (let x = 0; x < pw; x++) {
              const i = 4 * (y * pw + x);
              if (data[i] - data[i + 2] > 40) {
                count += 1;
                if (x >= gx0 && x < gx1 && y >= gy0 && y < gy1) inGround += 1;
              }
            }
          lit.push(count / (pw * ph));
          litGround.push(inGround / ((gx1 - gx0) * (gy1 - gy0)));
        }
        if (frames >= run.frames) resolve();
        else if (run.mode !== 'paced') worker.postMessage({ type: 'frame', i: frames });
      };
      worker.postMessage(run.mode === 'paced' ? { type: 'pace' } : { type: 'frame', i: 0 });
    });

    if (run.mode === 'paced') {
      // Let the last frames' graphics work finish
      await new Promise(r => setTimeout(r, 300));
      poll();
      clearInterval(timer);
      watching = false;
      const slot = 1000 / run.fps;
      // Frames the page took in each second: the stream's own rate when the page keeps up
      out.framesASecond = round(((frames - 1) * 1000) / Math.max(1, lastArrival - firstArrival));
      out.lateMs = { median: round(median(late)), p95: round(centile(late, 0.95)), max: round(Math.max(...late)) };
      out.graphicsMs = { median: round(median(gpuMs)), p95: round(centile(gpuMs, 0.95)) };
      out.inSlot = round(gpuMs.filter(v => v <= slot).length / Math.max(1, frames));
      out.unfinished = fences.length;
      out.animationFrames = animationFrames;
      out.worstAnimationGapMs = round(worstGap);
    }
    out.frames = frames;
    out.heldMs = { median: round(median(held)), p95: round(centile(held, 0.95)) };
    out.plainMs = { median: round(median(plainMs)), p95: round(centile(plainMs, 0.95)) };
    if (run.mode !== 'paced') {
      out.frameMs = { median: round(median(total)), p95: round(centile(total, 0.95)) };
      out.framesASecond = round(1000 / (median(total) + median(plainMs)));
    }
    if (run.stages) {
      // Each stage net of the bare wait; "wait" is one bare wait
      const bare = (stages.wait ?? 0) / Math.max(1, waits);
      const worked = enhancer.stats.frames || frames;
      out.stages = Object.fromEntries(
        Object.entries(stages).map(([k, v]) => [k, round(k === 'wait' ? bare : Math.max(0, v / worked - bare))]),
      );
    }
    if (run.mode === 'behaviour') {
      const from = Math.round(2 * run.fps); // once the highlight has faded in
      const after = values => values.slice(from);
      out.litMost = round(100 * Math.max(...after(lit)));
      out.litMean = round((100 * after(lit).reduce((s, v) => s + v, 0)) / Math.max(1, after(lit).length));
      out.groundLitFrames = round(after(litGround).filter(v => v >= 0.002).length / Math.max(1, after(litGround).length));
      // A steadied picture sits at one place (where the camera was when the reference was taken) however the
      // camera wobbles: how far it strays from that place, in frame pixels
      const rest = [median(after(off).map(v => v[0])), median(after(off).map(v => v[1]))];
      const stray = after(off).map(v => Math.hypot(v[0] - rest[0], v[1] - rest[1]));
      out.strayMean = round(stray.reduce((s, v) => s + v, 0) / Math.max(1, stray.length));
      out.strayMost = round(Math.max(0, ...stray));
      // A knock or a pan: where the picture sits against the scene each half second
      out.placeEachHalfSecond = [];
      for (let i = 0; i < off.length; i += run.fps / 2)
        out.placeEachHalfSecond.push(round(median(off.slice(i, i + run.fps / 2).map(v => Math.hypot(v[0], v[1])))));
      out.litEachHalfSecond = [];
      for (let i = 0; i < lit.length; i += run.fps / 2)
        out.litEachHalfSecond.push(round(100 * Math.max(...lit.slice(i, i + run.fps / 2))));
    }
    out.stats = { ...enhancer.stats };
    out.wall = [started, Date.now()];
    enhancer.dispose();
    worker.terminate();
    plain.remove();
    canvas.remove();
    return out;
  }

  for (const run of config.runs) {
    let result;
    try {
      result = await one(run);
    } catch (error) {
      result = { ...run, error: String(error?.stack ?? error) };
    }
    await fetch('/result', { method: 'POST', body: JSON.stringify(result) });
  }
  await fetch('/done', { method: 'POST' });
}

// ---- The driver ----

// Vite is the dashboard's own bundler: the renderer and what it imports, as one script that sets EnhanceBundle
const built = await need('vite').build({
  configFile: false,
  logLevel: 'silent',
  build: { write: false, minify: false, lib: { entry: renderer, formats: ['iife'], name: 'EnhanceBundle' } },
});
const bundle = (Array.isArray(built) ? built[0] : built).output[0].code;
// The small-copy maker beside the renderer, for the frame worker, when this renderer has one
const smallSource = path.join(path.dirname(renderer), 'small.ts');
let smallBundle = '';
if (existsSync(smallSource)) {
  const made = await need('vite').build({
    configFile: false,
    logLevel: 'silent',
    build: { write: false, minify: false, lib: { entry: smallSource, formats: ['iife'], name: 'EnhanceSmall' } },
  });
  smallBundle = (Array.isArray(made) ? made[0] : made).output[0].code;
}
const runs = runsFor();

/** Utilisation of each graphics card, sampled four times a second: [ms, [percent, ...]]. */
const samples = [];
let line = [];
const sampler = spawn(
  'nvidia-smi',
  ['--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits', '-lms', '250'],
  { stdio: ['ignore', 'pipe', 'ignore'] },
);
sampler.on('error', () => {});
sampler.stdout.on('data', chunk => {
  for (const row of String(chunk).split('\n')) {
    const m = /^(\d+),\s*(\d+)/.exec(row);
    if (!m) continue;
    if (m[1] === '0' && line.length) {
      samples.push([Date.now(), line]);
      line = [];
    }
    line.push(Number(m[2]));
  }
});
const busy = wall => {
  const mine = samples.filter(([at]) => at >= wall[0] && at <= wall[1]).map(([, cards]) => cards);
  if (!mine.length) return undefined;
  return mine[0].map((_, card) => Math.round(mine.reduce((s, cards) => s + (cards[card] ?? 0), 0) / mine.length));
};

let close = () => {};
const finished = new Promise(resolve => {
  const server = createServer((req, res) => {
    const send = (type, body) => {
      res.setHeader('content-type', type);
      res.end(body);
    };
    if (req.method === 'POST') {
      let body = '';
      req.on('data', chunk => {
        body += chunk;
      });
      req.on('end', () => {
        res.end('ok');
        if (req.url === '/done') resolve();
        else {
          const result = JSON.parse(body);
          if (result.wall) result.cardsBusy = busy(result.wall);
          delete result.wall;
          console.log(JSON.stringify({ browser: browserName, ...result }));
        }
      });
    } else if (req.url === '/config') send('application/json', JSON.stringify({ runs }));
    else if (req.url === '/bundle.js') send('text/javascript', bundle);
    else if (req.url === '/small.js') send('text/javascript', smallBundle);
    else if (req.url === '/worker.js') send('text/javascript', `(${frameWorker.toString()})()`);
    else if (req.url === '/page.js') send('text/javascript', `(${benchPage.toString()})()`);
    else
      send(
        'text/html',
        '<!doctype html><body style="margin:0"><script src="/bundle.js"></script><script src="/page.js"></script>',
      );
  });
  server.listen(0, '127.0.0.1', async () => {
    const address = `http://127.0.0.1:${server.address().port}/`;
    if (browserName === 'firefox') {
      const profile = mkdtempSync(path.join(tmpdir(), 'enhance-bench-'));
      const prefs = {
        'webgl.sanitize-unmasked-renderer': false, // so the run can say which graphics card it drew on
        'privacy.reduceTimerPrecision': false, // times to better than a millisecond
        'browser.shell.checkDefaultBrowser': false,
        'browser.aboutwelcome.enabled': false,
        'datareporting.policy.dataSubmissionEnabled': false,
        'toolkit.telemetry.reportingpolicy.firstRun': false,
      };
      writeFileSync(
        path.join(profile, 'user.js'),
        Object.entries(prefs)
          .map(([k, v]) => `user_pref(${JSON.stringify(k)}, ${JSON.stringify(v)});`)
          .join('\n'),
      );
      // Its own profile and --no-remote, so it never reaches a Firefox that is already open
      const firefox = spawn(
        firefoxBinary,
        ['--headless', '--no-remote', '--new-instance', '--profile', profile, address],
        { stdio: 'ignore' },
      );
      close = () => {
        firefox.kill('SIGTERM');
        setTimeout(() => rmSync(profile, { recursive: true, force: true }), 1500);
        server.close();
      };
    } else {
      const flags =
        browserName === 'chromium-software'
          ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
          : ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan'];
      const browser = await need('playwright').chromium.launch({ args: flags });
      const page = await browser.newPage({ viewport: { width: 960, height: 540 } });
      page.on('pageerror', error => console.error(`page: ${error.message}`));
      await page.goto(address);
      close = () => {
        browser.close();
        server.close();
      };
    }
  });
});
const limit = setTimeout(() => {
  console.error('The browser did not finish in time.');
  close();
  sampler.kill();
  setTimeout(() => process.exit(1), 2500);
}, Number(option('timeout', '600')) * 1000);
await finished;
clearTimeout(limit);
close();
sampler.kill();
setTimeout(() => process.exit(0), 2500);
