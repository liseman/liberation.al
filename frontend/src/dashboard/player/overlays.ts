// Overlays by capture time, and how one is picked for a frame: the exact match when there is one, else the two
// nearest interpolated (high-rate data like IMU or SLAM pose lands between frames), else nothing when the nearest is
// too far (the stream has a gap, or the overlay for this frame hasn't arrived). Pure, so it's testable without a
// canvas; the worker draws what this returns.

/** A box in frame fractions (0 to 1), with its label and an optional hue so a subject keeps its colour. */
export interface OverlayBox {
  xyxy: [number, number, number, number];
  score: number;
  label: string;
  hue: number | null;
  /** What it is (vision's `face`, `body`, or `object`), when the overlay says. */
  kind?: string;
  /** What it follows across frames (a face's track), when the overlay says. */
  id?: string;
}

/** One joint: place and confidence. */
export type Joint = [number, number, number];

export interface OverlaySkeleton {
  box: OverlayBox;
  layout: string;
  joints: Joint[];
}

export interface OverlayCaption {
  text: string;
  speaker: string;
}

/** What to draw over one frame (the decoded form of schemas/overlay.fbs). */
export interface Overlay {
  captureUs: number;
  boxes: OverlayBox[];
  skeletons: OverlaySkeleton[];
  captions: OverlayCaption[];
}

/** How an overlay was chosen for a frame, for the sync log. */
export type Pick =
  | { kind: 'exact'; overlay: Overlay }
  | { kind: 'interpolated'; overlay: Overlay; between: [number, number] }
  | { kind: 'nearest'; overlay: Overlay; offsetUs: number }
  | { kind: 'none' };

/** Overlays farther than this from a frame aren't drawn on it. */
export const MAX_GAP_US = 100_000;
/** Keep at most this many overlays per stream; older ones go when more arrive. */
export const KEEP = 600;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpBox(a: OverlayBox, b: OverlayBox, t: number): OverlayBox {
  return {
    xyxy: [
      lerp(a.xyxy[0], b.xyxy[0], t),
      lerp(a.xyxy[1], b.xyxy[1], t),
      lerp(a.xyxy[2], b.xyxy[2], t),
      lerp(a.xyxy[3], b.xyxy[3], t),
    ],
    score: lerp(a.score, b.score, t),
    label: t < 0.5 ? a.label : b.label,
    hue: a.hue,
    ...(a.kind ? { kind: a.kind } : {}),
    ...(a.id ? { id: a.id } : {}),
  };
}

/** Identified subjects are paired by identity, never by changing detector order. An unmatched subject comes from
 *  the nearer snapshot; completely legacy sets retain their positional behavior. */
function blendTracked<T>(
  a: T[],
  b: T[],
  t: number,
  identity: (value: T) => string | undefined,
  blend: (a: T, b: T) => T,
): T[] {
  const near = t < 0.5 ? a : b;
  const before = new Map(a.filter(value => identity(value)).map(value => [identity(value), value]));
  const after = new Map(b.filter(value => identity(value)).map(value => [identity(value), value]));
  return near.map(value => {
    const id = identity(value);
    const first = id ? before.get(id) : undefined;
    const second = id ? after.get(id) : undefined;
    return first && second ? blend(first, second) : value;
  });
}

function lerpSkeleton(a: OverlaySkeleton, b: OverlaySkeleton, t: number, tracked = true): OverlaySkeleton {
  if (a.joints.length !== b.joints.length || (tracked && a.layout !== b.layout)) return tracked && t >= 0.5 ? b : a;
  return {
    box: lerpBox(a.box, b.box, t),
    layout: a.layout,
    joints: a.joints.map((joint, index) => {
      const other = b.joints[index] ?? joint;
      if (joint[2] <= 0 || other[2] <= 0) return joint;
      return [lerp(joint[0], other[0], t), lerp(joint[1], other[1], t), Math.min(joint[2], other[2])] as Joint;
    }),
  };
}

/** Blends two overlays at `t` (0 is `a`, 1 is `b`). People entering, leaving, or changing order keep their identity. */
export function interpolate(a: Overlay, b: Overlay, t: number): Overlay {
  const boxes = [...a.boxes, ...b.boxes].some(box => box.id)
    ? blendTracked(
        a.boxes,
        b.boxes,
        t,
        box => box.id,
        (first, second) => lerpBox(first, second, t),
      )
    : a.boxes.map((box, i) => (b.boxes[i] ? lerpBox(box, b.boxes[i], t) : box));
  const skeletons = [...a.skeletons, ...b.skeletons].some(s => s.box.id)
    ? blendTracked(
        a.skeletons,
        b.skeletons,
        t,
        s => s.box.id,
        (first, second) => lerpSkeleton(first, second, t),
      )
    : a.skeletons.length !== b.skeletons.length
      ? t < 0.5
        ? a.skeletons
        : b.skeletons
      : a.skeletons.map((s, i) => (b.skeletons[i] ? lerpSkeleton(s, b.skeletons[i], t, false) : s));
  return {
    captureUs: Math.round(lerp(a.captureUs, b.captureUs, t)),
    boxes,
    skeletons,
    captions: t < 0.5 ? a.captions : b.captions,
  };
}

/** The overlays of one stream, by capture time. */
export class OverlayMap {
  private readonly byTime = new Map<number, Overlay>();
  private times: number[] = [];
  private sorted = true;

  get size(): number {
    return this.byTime.size;
  }

  add(overlay: Overlay): void {
    if (!this.byTime.has(overlay.captureUs)) {
      this.times.push(overlay.captureUs);
      this.sorted = false;
    }
    this.byTime.set(overlay.captureUs, overlay);
    if (this.byTime.size > KEEP) {
      this.sort();
      const drop = this.times.splice(0, this.byTime.size - KEEP);
      for (const t of drop) this.byTime.delete(t);
    }
  }

  clear(): void {
    this.byTime.clear();
    this.times = [];
  }

  private sort(): void {
    if (!this.sorted) {
      this.times.sort((a, b) => a - b);
      this.sorted = true;
    }
  }

  /** The overlay for a frame captured at `us`. */
  pick(us: number, maxGapUs = MAX_GAP_US): Pick {
    const exact = this.byTime.get(us);
    if (exact) return { kind: 'exact', overlay: this.withNearbyPoses(exact, us, maxGapUs) };
    if (this.times.length === 0) return { kind: 'none' };
    this.sort();
    // The first time past us, by binary search.
    let lo = 0;
    let hi = this.times.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((this.times[mid] ?? Number.POSITIVE_INFINITY) < us) lo = mid + 1;
      else hi = mid;
    }
    const after = this.times[lo];
    const before = this.times[lo - 1];
    if (before !== undefined && after !== undefined && after - before <= 2 * maxGapUs) {
      const a = this.byTime.get(before);
      const b = this.byTime.get(after);
      if (a && b)
        return {
          kind: 'interpolated',
          overlay: this.withNearbyPoses(interpolate(a, b, (us - before) / (after - before)), us, maxGapUs),
          between: [before, after],
        };
    }
    const candidates = [before, after].filter((t): t is number => t !== undefined);
    let best: number | undefined;
    for (const t of candidates) if (best === undefined || Math.abs(t - us) < Math.abs(best - us)) best = t;
    if (best === undefined || Math.abs(best - us) > maxGapUs) return { kind: 'none' };
    const overlay = this.byTime.get(best);
    return overlay
      ? { kind: 'nearest', overlay: this.withNearbyPoses(overlay, us, maxGapUs), offsetUs: best - us }
      : { kind: 'none' };
  }

  /** Face and body inference sample different camera frames. A face-only snapshot does not clear a nearby pose.
   *  Keep the existing freshness bound so stopped pose output cannot leave a skeleton behind indefinitely. */
  private withNearbyPoses(overlay: Overlay, us: number, maxGapUs: number): Overlay {
    if (overlay.skeletons.length) return overlay;
    let nearest: Overlay | undefined;
    let distance = maxGapUs + 1;
    for (const candidate of this.byTime.values()) {
      const gap = Math.abs(candidate.captureUs - us);
      if (candidate.skeletons.length && gap <= maxGapUs && gap < distance) {
        nearest = candidate;
        distance = gap;
      }
    }
    return nearest ? { ...overlay, skeletons: nearest.skeletons } : overlay;
  }
}
