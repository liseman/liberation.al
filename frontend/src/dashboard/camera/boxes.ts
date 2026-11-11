// Boxes drawn by the page over a camera picture: faces with names, pets, and notable objects. The picture itself is plain
// (the gateway sends them as data in its overlay frames, decoded by player/overlays.ts; Review reads the same
// detections back from the records see/ writes), so they can be smoothed, faded, and mirrored with readable labels.
//
// Each update can carry the wall time its frame was captured; boxes are drawn on every animation frame a little
// behind the newest update, interpolated between the updates on either side by capture time (not arrival time, which
// jitters with the network), after a One Euro filter per edge that holds a still box steady and still follows fast
// movement. Boxes fade in and out, and a changed name crossfades. The DOM is built here, not by React: it changes on
// every frame while boxes move. Once every box has settled (faded in or out, crossfaded, past its newest sample), the
// loop drops to a heartbeat every MEASURE_S until the next update, and a frame writes a style only when its value
// changed, so still boxes cost no style, layout, or paint work. Its class names are global, in camera.css.

import { type Pose, wireframe } from '../pose/skeleton';
import { type LiveFrame, wantPoses, watchBoxes } from './gateway';

/** A box over a picture. box is [x1, y1, x2, y2] as fractions 0 to 1 of the picture. */
export interface Box {
  id: string | number;
  box: readonly number[];
  label: string;
  kind: 'face' | 'object' | 'pet';
  tone?: 'known' | 'unknown';
  faceSource?: { camera: string; at: number; track: string; group: number };
}

/** What onPick receives: the clicked box, and where its element sits in the layer's parent, in pixels. */
export interface BoxPick extends Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

export type { Fit, Rect } from './geometry';
export { contentRect } from './geometry';

import { contentRect, type Rect } from './geometry';

/** The One Euro filter (Casiez et al. 2012): a low-pass whose cutoff rises with speed. Values on a 0 to 1000 scale. */
export class OneEuro {
  minCutoff: number;
  beta: number;
  dCutoff: number;
  x: number | null = null;
  dx = 0;
  t: number | null = null;

  constructor(minCutoff = 0.5, beta = 0.01, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(x: number, t: number): number {
    if (this.x === null || this.t === null || t <= this.t) {
      if (this.x === null) {
        this.x = x;
        this.t = t;
      }
      return this.x;
    }
    const dt = Math.min(1, t - this.t);
    const dx = (x - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (dx - this.dx);
    this.x += OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (x - this.x);
    this.t = t;
    return this.x;
  }
}

export const DELAY_S = 0.05; // draw this far behind the newest frame, so there's an update on either side to interpolate
export const STALE_S = 1.0; // no update for this long (by arrival) and the boxes fade away
export const FADE_S = 0.15; // fade time constant
export const SWAP_S = 0.25; // a name change crossfades over this long
export const MEASURE_S = 0.5; // the box layer's placement is read again at least this often

type Sample = [t: number, box: number[]];

/** The styles last written to a box's elements, so a frame rewrites only what changed. */
interface Written {
  className: string;
  transform: string;
  width: string;
  height: string;
  opacity: string;
  below: boolean;
  nu: string;
  old: string;
}

interface Track {
  f: OneEuro[];
  written?: Written;
  samples: Sample[];
  opacity: number;
  el: HTMLDivElement | null;
  label: string;
  old: string | null;
  swap: number;
  kind: Box['kind'];
  tone: Box['tone'];
  here: boolean;
  raw: Box;
}

export interface BoxOptions {
  mirror?: boolean;
  onPick?: ((pick: BoxPick) => void) | null;
  /** Which faces onPick takes: unknown ones (to name them), or every face (Live's large view links each). */
  pickFaces?: 'unknown' | 'all';
  pickObjects?: boolean;
  /** Pets can be clicked too (Review's "Not <name>" on a pet's box). */
  pickPets?: boolean;
  /** Told when the pointer comes onto a box (with it) and leaves it (null): Review reveals a box's actions on hover. */
  onHover?: ((pick: BoxPick | null) => void) | null;
  /** No update for this long (by arrival) and the boxes fade away. Review keeps them while paused (Infinity). */
  staleS?: number;
}

/** Smooths and draws boxes into a layer element on every animation frame. */
export class BoxRenderer {
  layer: HTMLElement;
  mirror: boolean;
  onPick: ((pick: BoxPick) => void) | null;
  onHover: ((pick: BoxPick | null) => void) | null;
  tracks = new Map<string | number, Track>();
  offset: number | null = null; // arrival minus capture time, the smallest seen lately (clock skew plus the least delay)
  heard = 0;
  rect: Rect = { x: 0, y: 0, w: 0, h: 0 };
  skel: SVGSVGElement | null = null; // the skeletons drawn over the picture (skeletons())
  // The skeletons fade like the boxes: their opacity eases toward 1 while a pose is fresh and toward 0 once
  // an update brings none or updates stop, so a missed pose turn dims them briefly instead of blinking them out.
  skelOpacity = 0;
  skelHere = false;
  skelHeard = 0;
  last: number;
  raf = 0; // the requested animation frame, 0 while the loop idles
  idle: ReturnType<typeof setTimeout> | null = null; // the heartbeat while settled
  // measure() reads layout, which forces one every frame. It reads again only when the picture, its natural size,
  // or a watched size changes, or after MEASURE_S (a safety net for moves no size change shows).
  pic: Element | null = null;
  nat = '';
  css = '';
  dirty = true;
  measuredAt = 0;
  watch: ResizeObserver | null =
    typeof ResizeObserver === 'function'
      ? new ResizeObserver(() => {
          this.dirty = true;
          this.measure();
          this.wake();
        })
      : null;

  pickFaces: 'unknown' | 'all';
  pickObjects: boolean;
  pickPets: boolean;

  staleS: number;

  constructor(
    layer: HTMLElement,
    {
      mirror = false,
      onPick = null,
      onHover = null,
      pickFaces = 'unknown',
      pickObjects = false,
      pickPets = false,
      staleS = STALE_S,
    }: BoxOptions = {},
  ) {
    this.layer = layer;
    this.staleS = staleS;
    this.mirror = mirror;
    this.onPick = onPick;
    this.onHover = onHover;
    this.pickFaces = pickFaces;
    this.pickObjects = pickObjects;
    this.pickPets = pickPets;
    this.last = performance.now();
    this.wake();
  }

  /** Run the frame loop now, if it's idle: something changed (an update, skeletons, a resize, an option). */
  wake(): void {
    if (this.idle !== null) {
      clearTimeout(this.idle);
      this.idle = null;
    }
    if (!this.raf) {
      this.last = performance.now(); // an idle stretch isn't one long frame
      this.raf = requestAnimationFrame(t => this.frame(t));
    }
  }

  /** A new set of boxes, from the frame captured at ts (wall seconds; arrival time when omitted). */
  push(list: readonly Box[], ts?: number | null): void {
    const arrive = Date.now() / 1000;
    const at = ts ?? arrive;
    const off = arrive - at;
    this.offset = this.offset === null || off < this.offset ? off : this.offset + 0.002; // creeps back up after a spike
    this.heard = arrive;
    const seen = new Set<string | number>();
    for (const b of list) {
      seen.add(b.id);
      let s = this.tracks.get(b.id);
      if (!s) {
        s = {
          f: [0, 1, 2, 3].map(() => new OneEuro()),
          samples: [],
          opacity: 0,
          el: null,
          label: b.label,
          old: null,
          swap: 1,
          kind: b.kind,
          tone: b.tone,
          here: true,
          raw: b,
        };
        this.tracks.set(b.id, s);
      }
      if (b.label !== s.label) {
        s.old = s.label;
        s.label = b.label;
        s.swap = 0;
      }
      s.kind = b.kind;
      s.tone = b.tone;
      s.here = true;
      s.raw = b;
      // Time went back (Review seeking back): smoothing and interpolation start over, or the box would hold where it
      // was until time caught up with its last sample.
      const lastAt = s.samples.at(-1)?.[0];
      if (lastAt != null && at < lastAt - 0.01) {
        s.f = [0, 1, 2, 3].map(() => new OneEuro());
        s.samples = [];
      }
      const track = s;
      const v = b.box.slice(0, 4).map((x, i) => (track.f[i]?.filter(x * 1000, at) ?? x * 1000) / 1000);
      s.samples.push([at, v]);
      if (s.samples.length > 6) s.samples.shift();
    }
    for (const [id, s] of this.tracks) if (!seen.has(id)) s.here = false;
    this.wake();
  }

  /** Where a box is at capture time t: between the samples either side, else the nearest one. */
  static at(s: { samples: readonly Sample[] }, t: number): number[] | null {
    const S = s.samples;
    const first = S[0];
    if (!first) return null;
    if (t <= first[0]) return first[1];
    for (let i = S.length - 1; i > 0; i--) {
      const [t0, a] = S[i - 1] as Sample;
      const [t1, b] = S[i] as Sample;
      if (t >= t0 && t <= t1) {
        const k = t1 > t0 ? (t - t0) / (t1 - t0) : 1;
        return a.map((v, j) => v + ((b[j] ?? v) - v) * k);
      }
    }
    return (S.at(-1) as Sample)[1];
  }

  /** Place the box layer over the picture. at is the frame's time (performance.now() by default). */
  measure(at = performance.now()): void {
    // Nothing to read while the picture is the same one, still in the page, and nothing said it changed: no query.
    if (!this.dirty && this.watch && this.pic?.isConnected && at - this.measuredAt < MEASURE_S * 1000) return;
    // A live picture is a canvas fed by mux.ts (one connection for every camera); otherwise an <img> or a <video>.
    const p = this.layer.parentElement;
    const cv = p?.querySelector<HTMLCanvasElement>('canvas.live-canvas:not(.off)');
    const pic = cv ?? p?.querySelector<HTMLImageElement | HTMLVideoElement>('img, video');
    if (!pic) {
      this.layer.style.display = 'none'; // no picture to draw over (a camera with no snapshot shows a note)
      this.watch?.disconnect();
      this.pic = null; // so a picture coming back is measured and shown again
      this.css = '';
      return;
    }
    // A snapshot fallback has its boxes drawn in, so boxes are not drawn over it.
    const shown = !!cv || !pic.classList.contains('snap');
    let natW = 0;
    let natH = 0;
    if (pic instanceof HTMLCanvasElement) [natW, natH] = [pic.width, pic.height];
    else if (pic instanceof HTMLVideoElement) [natW, natH] = [pic.videoWidth, pic.videoHeight];
    else [natW, natH] = [pic.naturalWidth, pic.naturalHeight];
    const nat = `${natW}x${natH}${shown ? '' : ' hidden'}`;
    if (pic !== this.pic) {
      this.watch?.disconnect();
      this.watch?.observe(pic);
      if (p) this.watch?.observe(p);
      this.pic = pic;
      this.dirty = true;
    }
    if (nat !== this.nat || at - this.measuredAt >= MEASURE_S * 1000) this.dirty = true;
    if (!this.dirty && this.watch) return;
    this.dirty = false;
    this.nat = nat;
    this.measuredAt = at;
    const r = contentRect(pic.clientWidth, pic.clientHeight, natW, natH, getComputedStyle(pic).objectFit);
    this.rect = { x: pic.offsetLeft + r.x, y: pic.offsetTop + r.y, w: r.w, h: r.h };
    const css = `left:${this.rect.x}px;top:${this.rect.y}px;width:${this.rect.w}px;height:${this.rect.h}px;${shown ? '' : 'display:none'}`;
    if (css !== this.css) {
      this.layer.style.cssText = css;
      this.css = css;
    }
  }

  frame(nowMs: number): void {
    this.raf = 0;
    const dt = Math.min(0.1, (nowMs - this.last) / 1000);
    this.last = nowMs;
    this.measure(nowMs);
    const wall = Date.now() / 1000;
    const stale = wall - this.heard > this.staleS;
    const t = wall - (this.offset ?? 0) - DELAY_S;
    const k = 1 - Math.exp(-dt / FADE_S);
    let moving = false; // whether the next frame would draw anything different
    for (const [id, s] of this.tracks) {
      const target = s.here && !stale ? 1 : 0;
      s.opacity += (target - s.opacity) * k;
      s.swap = Math.min(1, s.swap + dt / SWAP_S);
      if (!target && s.opacity < 0.02) {
        s.el?.remove();
        this.tracks.delete(id);
        continue;
      }
      if (target && s.opacity > 0.995) s.opacity = 1; // settled: an exact value, so its style stops changing
      if (Math.abs(target - s.opacity) > 0 || s.swap < 1 || t < (s.samples.at(-1)?.[0] ?? -Infinity)) moving = true;
      const v = BoxRenderer.at(s, t);
      if (!v) continue;
      this.draw(s, v);
    }
    if (this.skel) {
      const target = this.skelHere && wall - this.skelHeard <= this.staleS ? 1 : 0;
      this.skelOpacity += (target - this.skelOpacity) * k;
      if (!target && this.skelOpacity < 0.02) this.clearSkeletons();
      else {
        if (target && this.skelOpacity > 0.995) this.skelOpacity = 1;
        if (this.skelOpacity !== target) moving = true;
        const o = this.skelOpacity.toFixed(3);
        if (this.skel.style.opacity !== o) this.skel.style.opacity = o;
      }
    }
    if (moving) {
      this.raf = requestAnimationFrame(n => this.frame(n));
      return;
    }
    // Settled: a heartbeat keeps measure()'s safety net and notices boxes going stale, sooner if they're due to.
    const visible = this.tracks.size > 0 || !!this.skel;
    const staleIn = visible && Number.isFinite(this.staleS) ? (this.heard + this.staleS - wall) * 1000 + 1 : Infinity;
    this.idle = setTimeout(
      () => {
        this.idle = null;
        this.wake();
      },
      Math.max(0, Math.min(MEASURE_S * 1000, staleIn)),
    );
  }

  draw(s: Track, box: readonly number[]): void {
    let [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
    if (!s.el) {
      const el = document.createElement('div');
      const label = document.createElement('span');
      label.className = 'bx-label';
      for (const name of ['bx-new', 'bx-old']) {
        const part = document.createElement('span');
        part.className = name;
        label.append(part);
      }
      el.append(label);
      // The box as onPick and onHover give it: what it is, and where its element sits in the layer's parent. A box is
      // placed by a transform, which offsetLeft and offsetTop don't see (they were always the layer's corner), so this
      // measures where it is drawn.
      const at = (): BoxPick => {
        const box = el.getBoundingClientRect();
        const parent = this.layer.parentElement?.getBoundingClientRect();
        return {
          ...s.raw,
          left: box.left - (parent?.left ?? 0),
          top: box.top - (parent?.top ?? 0),
          width: box.width,
          height: box.height,
        };
      };
      const pick = (e: Event) => {
        if (!this.onPick || !el.classList.contains('pickable')) return;
        e.stopPropagation();
        this.onPick(at());
      };
      el.addEventListener('click', pick);
      el.addEventListener('pointerenter', e => {
        if (e.pointerType === 'mouse') this.onHover?.(at());
      });
      el.addEventListener('pointerleave', e => {
        if (e.pointerType === 'mouse') this.onHover?.(null);
      });
      el.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') pick(e);
      });
      this.layer.append(el);
      s.el = el;
    }
    if (this.mirror) [x1, x2] = [1 - x2, 1 - x1];
    const { w, h } = this.rect;
    const el = s.el;
    const pickable =
      !!this.onPick &&
      ((s.kind === 'object' && this.pickObjects) ||
        (s.kind === 'pet' && this.pickPets) ||
        (s.kind === 'face' && (s.tone === 'unknown' || this.pickFaces === 'all')));
    const className = `bx bx-${s.kind}${s.tone ? ` bx-${s.tone}` : ''}${pickable ? ' pickable' : ''}`;
    if (pickable !== el.hasAttribute('tabindex')) {
      // A box that does something is a keyboard stop too, named by its label.
      if (pickable) {
        el.tabIndex = 0;
        el.setAttribute('role', 'button');
      } else {
        el.removeAttribute('tabindex');
        el.removeAttribute('role');
      }
    }
    if (pickable && el.getAttribute('aria-label') !== s.label) el.setAttribute('aria-label', s.label);
    const next: Written = {
      className,
      transform: `translate(${(x1 * w).toFixed(1)}px,${(y1 * h).toFixed(1)}px)`,
      width: `${Math.max(2, (x2 - x1) * w).toFixed(1)}px`,
      height: `${Math.max(2, (y2 - y1) * h).toFixed(1)}px`,
      opacity: s.opacity.toFixed(3),
      below: y1 * h < 18, // no room above at the top edge: label inside the box
      nu: s.swap.toFixed(2),
      old: (1 - s.swap).toFixed(2),
    };
    // Only what changed is written: a still box costs no style recalculation, layout, or paint.
    const was = s.written;
    if (was?.className !== next.className) {
      el.className = next.className;
      el.classList.toggle('bx-below', next.below);
    } else if (was.below !== next.below) el.classList.toggle('bx-below', next.below);
    if (was?.transform !== next.transform) el.style.transform = next.transform;
    if (was?.width !== next.width) el.style.width = next.width;
    if (was?.height !== next.height) el.style.height = next.height;
    if (was?.opacity !== next.opacity) el.style.opacity = next.opacity;
    const [nu, old] = [...(el.firstElementChild?.children ?? [])] as HTMLElement[];
    if (nu && old) {
      if (nu.textContent !== s.label) nu.textContent = s.label;
      if (old.textContent !== (s.old ?? '')) old.textContent = s.old ?? '';
      if (was?.nu !== next.nu) nu.style.opacity = next.nu;
      if (was?.old !== next.old) old.style.opacity = next.old;
    }
    s.written = next;
  }

  /** Draw these people's skeletons over the picture (vision's halpe26 poses, 0 to 1 of the frame), replacing the
   *  last ones. null, [], or poses with no joint placed keep the last skeletons and fade them out (frame()), so a
   *  missed pose turn doesn't blink them. They're drawn in the layer's own 0 to 1 space, so they sit on the picture
   *  wherever measure() put the layer, and they mirror with the boxes. */
  skeletons(people: readonly Pose[] | null): void {
    const svgNs = 'http://www.w3.org/2000/svg';
    const drawn = (people ?? []).map(p => ({ ...wireframe(p), pose: p })).filter(w => w.joints.length > 0);
    if (!drawn.length) {
      this.skelHere = false;
      this.wake(); // they fade out
      return;
    }
    this.skelHere = true;
    this.skelHeard = Date.now() / 1000;
    if (!this.skel) {
      this.skel = document.createElementNS(svgNs, 'svg');
      this.skel.setAttribute('class', 'bx-skeletons');
      this.skel.setAttribute('viewBox', '0 0 1 1');
      this.skel.setAttribute('preserveAspectRatio', 'none');
      this.skel.setAttribute('aria-hidden', 'true');
      this.skel.style.opacity = '0'; // frame() fades it in, so it never shows at full strength first
      this.layer.prepend(this.skel); // under the boxes, so a face box stays clickable
    }
    const g = document.createElementNS(svgNs, 'g');
    if (this.mirror) g.setAttribute('transform', 'matrix(-1 0 0 1 1 0)');
    for (const { bones, joints, pose } of drawn) {
      const person = document.createElementNS(svgNs, 'g');
      if (pose.track_id) person.dataset.trackId = pose.track_id;
      if (pose.hue !== null && pose.hue !== undefined && Number.isFinite(pose.hue))
        person.style.setProperty('--skeleton-color', `hsl(${pose.hue} 90% 60%)`);
      for (const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] of bones) {
        const line = document.createElementNS(svgNs, 'line');
        line.setAttribute('x1', String(x1));
        line.setAttribute('y1', String(y1));
        line.setAttribute('x2', String(x2));
        line.setAttribute('y2', String(y2));
        person.append(line);
      }
      for (const [x = 0, y = 0] of joints) {
        // A zero-length round-capped line draws a dot that stays round in the stretched 0 to 1 space.
        const dot = document.createElementNS(svgNs, 'line');
        dot.setAttribute('class', 'bx-joint');
        for (const [k, v] of [
          ['x1', x],
          ['y1', y],
          ['x2', x],
          ['y2', y],
        ] as const)
          dot.setAttribute(k, String(v));
        person.append(dot);
      }
      g.append(person);
    }
    this.skel.replaceChildren(g);
    this.wake();
  }

  /** Take the skeletons off at once, with no fade (the viewer switched them off). */
  clearSkeletons(): void {
    this.skel?.remove();
    this.skel = null;
    this.skelOpacity = 0;
    this.skelHere = false;
  }

  close(): void {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    if (this.idle !== null) clearTimeout(this.idle);
    this.idle = null;
    this.watch?.disconnect();
    this.layer.replaceChildren();
    this.skel = null;
  }
}

/** One camera frame's boxes, from its gateway session (gateway.ts): the shape v1's /video/live/boxes sent. */
export type { LiveFrame };

const UNKNOWN_FACE = /^(Face \d+|unknown)$/;

/** Vision's boxes for one frame of a camera, in BoxRenderer's format. */
export const liveBoxes = (d: LiveFrame): Box[] => [
  ...(d.objects ?? []).map(o => ({ id: `o${o.id}`, box: o.box, label: o.label, kind: 'object' as const })),
  ...(d.faces ?? []).map(f => ({
    id: `f${f.track}`,
    ...(/^Face \d+$/.test(f.name) && d.face_ts != null
      ? { faceSource: { camera: d.cam, at: d.face_ts, track: String(f.track), group: Number(f.name.slice(5)) } }
      : {}),
    box: f.box,
    label: f.name,
    kind: 'face' as const,
    tone: UNKNOWN_FACE.test(f.name) ? ('unknown' as const) : ('known' as const),
  })),
];

/** Ask for camera cam's skeletons while this page shows it; returns the function that stops asking. */
export const wantSkeletons = (cam: string): (() => void) => wantPoses(cam);

/** Call fn with each of camera cam's live box frames, from its gateway session; returns the function that stops. */
export const subscribe = (cam: string, fn: (d: LiveFrame) => void): (() => void) => watchBoxes(cam, fn);
