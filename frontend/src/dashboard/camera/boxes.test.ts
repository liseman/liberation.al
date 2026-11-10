import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Pose } from '../pose/skeleton';
import { type Box, BoxRenderer, contentRect, liveBoxes, MEASURE_S, OneEuro } from './boxes';
import { fakeFrames } from './testFakes';

/** A box's translate as numbers: jsdom writes the transform back in its own format ("translate(40px, 20px)"), so the
 *  tests read the position, not the string. */
function translated(el: HTMLElement | null | undefined): [number, number] | null {
  const m = el?.style.transform.match(/translate\(\s*(-?[\d.]+)px\s*,\s*(-?[\d.]+)px\s*\)/);
  return m ? [Number(m[1]), Number(m[2])] : null;
}

describe('contentRect', () => {
  // A 16:9 picture (1280x720) in a 400x400 element.
  it('fills the element when the fit is fill', () => {
    expect(contentRect(400, 400, 1280, 720, 'fill')).toEqual({ x: 0, y: 0, w: 400, h: 400 });
  });
  it('letterboxes when contained: 400 wide is 225 high, centred', () => {
    expect(contentRect(400, 400, 1280, 720, 'contain')).toEqual({ x: 0, y: 87.5, w: 400, h: 225 });
  });
  it('overflows both sides equally when covering: 400 high is 711.1 wide', () => {
    const r = contentRect(400, 400, 1280, 720, 'cover');
    expect(r.h).toBe(400);
    expect(r.w).toBeCloseTo(711.111, 3);
    expect(r.x).toBeCloseTo(-155.556, 3);
    expect(r.y).toBe(0);
  });
  it('keeps the natural size with none, centred', () => {
    expect(contentRect(400, 400, 200, 100, 'none')).toEqual({ x: 100, y: 150, w: 200, h: 100 });
  });
  it('never scales up with scale-down', () => {
    expect(contentRect(400, 400, 200, 100, 'scale-down')).toEqual({ x: 100, y: 150, w: 200, h: 100 });
    expect(contentRect(400, 400, 1280, 720, 'scale-down')).toEqual({ x: 0, y: 87.5, w: 400, h: 225 });
  });
  it('has no bars when the shapes match', () => {
    expect(contentRect(800, 450, 1280, 720, 'contain')).toEqual({ x: 0, y: 0, w: 800, h: 450 });
  });
  it('is the whole element before the picture has a size', () => {
    expect(contentRect(300, 200, 0, 0, 'contain')).toEqual({ x: 0, y: 0, w: 300, h: 200 });
  });
});

describe('OneEuro', () => {
  it('passes the first value through, and holds for a sample at or before the last time', () => {
    const f = new OneEuro();
    expect(f.filter(100, 1)).toBe(100);
    expect(f.filter(500, 1)).toBe(100);
    expect(f.filter(500, 0.5)).toBe(100);
  });

  it('moves part way to a new value, more when it moves fast', () => {
    // 100 then 300 a second later. The speed estimate: alpha(1 Hz, 1 s) = 1 / (1 + 1/(2 pi)) = 0.862697, so
    // dx = 0.862697 * 200 = 172.539. The cutoff is 0.5 + 0.01 * 172.539 = 2.22539 Hz, so
    // alpha = 1 / (1 + 1/(2 pi * 2.22539)) = 0.933256, and x = 100 + 0.933255 * 200 = 286.651.
    const f = new OneEuro();
    f.filter(100, 0);
    expect(f.filter(300, 1)).toBeCloseTo(286.651, 2);
    expect(f.dx).toBeCloseTo(172.539, 2);
  });

  it('caps the time step at 1 s, so a long gap moves it no further than a 1 s one', () => {
    const f = new OneEuro();
    f.filter(100, 0);
    expect(f.filter(300, 60)).toBeCloseTo(286.651, 2);
  });

  it('smooths a slow step heavily: 10 units in 1/30 s moves about 1', () => {
    // dt = 1/30: alpha(1 Hz) = 1 / (1 + 30/(2 pi)) = 0.173171, dx = 0.173171 * 300 = 51.951;
    // cutoff 1.01951 Hz, alpha = 1 / (1 + 30/(2 pi * 1.01951)) = 0.175955, so x = 0 + 1.75955.
    const f = new OneEuro();
    f.filter(0, 0);
    expect(f.filter(10, 1 / 30)).toBeCloseTo(1.75955, 4);
  });
});

describe('BoxRenderer.at', () => {
  const s = {
    samples: [
      [10, [0, 0, 0, 0]],
      [11, [1, 2, 3, 4]],
    ] as [number, number[]][],
  };
  it('interpolates between the samples either side by capture time', () => {
    expect(BoxRenderer.at(s, 10.25)).toEqual([0.25, 0.5, 0.75, 1]);
  });
  it('holds the nearest sample outside them', () => {
    expect(BoxRenderer.at(s, 9)).toEqual([0, 0, 0, 0]);
    expect(BoxRenderer.at(s, 12)).toEqual([1, 2, 3, 4]);
  });
  it('has nothing without samples', () => {
    expect(BoxRenderer.at({ samples: [] }, 1)).toBeNull();
  });
});

describe('liveBoxes', () => {
  it("turns vision's objects and faces into boxes, unknown faces marked", () => {
    const d = {
      cam: 'cam0',
      objects: [{ id: 7, box: [0.1, 0.1, 0.2, 0.2], label: 'cup' }],
      faces: [
        { track: 1, box: [0, 0, 1, 1], name: 'Alice' },
        { track: 2, box: [0, 0, 1, 1], name: 'Face 3' },
        { track: 3, box: [0, 0, 1, 1], name: 'unknown' },
        { track: 4, box: [0, 0, 1, 1], name: 'Face 3b' },
      ],
    };
    expect(liveBoxes(d).map(b => [b.id, b.kind, b.label, b.tone])).toEqual([
      ['o7', 'object', 'cup', undefined],
      ['f1', 'face', 'Alice', 'known'],
      ['f2', 'face', 'Face 3', 'unknown'],
      ['f3', 'face', 'unknown', 'unknown'],
      ['f4', 'face', 'Face 3b', 'known'],
    ]);
  });
  it('copes with a frame missing a list', () => {
    expect(liveBoxes({ cam: 'cam0' })).toEqual([]);
  });
});

describe('BoxRenderer', () => {
  const T = 1_800_000_000; // wall seconds
  const FADE_IN = 1 - Math.exp(-0.1 / 0.15); // opacity after one 100 ms frame from 0: 0.48658
  let frames: ReturnType<typeof fakeFrames>;
  let layer: HTMLDivElement;
  let now: number; // animation frame time, ms

  beforeEach(() => {
    // The settled loop's heartbeat too, and performance.now(): the renderer stamps its frame clock with it when it
    // wakes, so a real clock moving on under load would drift from these frames' times (a flaky test, not a bug).
    vi.useFakeTimers({ toFake: ['Date', 'setTimeout', 'clearTimeout', 'performance'] });
    vi.setSystemTime(T * 1000);
    frames = fakeFrames();
    // A 400x200 live canvas showing a 400x200 picture, with the layer beside it.
    const parent = document.createElement('div');
    const canvas = document.createElement('canvas');
    canvas.className = 'live-canvas';
    canvas.width = 400;
    canvas.height = 200;
    Object.defineProperty(canvas, 'clientWidth', { value: 400 });
    Object.defineProperty(canvas, 'clientHeight', { value: 200 });
    layer = document.createElement('div');
    parent.append(canvas, layer);
    document.body.append(parent);
  });
  afterEach(() => {
    document.body.replaceChildren();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  const make = (opts?: ConstructorParameters<typeof BoxRenderer>[1]) => {
    const r = new BoxRenderer(layer, opts);
    now = r.last;
    return r;
  };
  /** Run one animation frame 100 ms after the last, the clock moving with it. */
  const frame = () => {
    now += 100;
    vi.advanceTimersByTime(100); // the clock, and a settled loop's heartbeat, which asks for the next frame
    frames.step(now);
  };
  /** Frames until a settled renderer has measured again: a picture change is noticed within MEASURE_S. */
  const untilMeasured = () => {
    for (let i = 0; i <= MEASURE_S * 10; i++) frame();
  };
  const face: Box = { id: 'f1', box: [0.1, 0.2, 0.3, 0.6], label: 'Alice', kind: 'face', tone: 'known' };
  const boxes = () => [...layer.querySelectorAll<HTMLElement>('.bx')];
  const labels = (el: HTMLElement) => [...el.querySelectorAll<HTMLElement>('.bx-label > span')];

  it('adds a box where it is in the picture, fading in', () => {
    const r = make();
    r.push([face], T);
    frame();
    const [el] = boxes();
    expect(el?.className).toBe('bx bx-face bx-known');
    expect(translated(el)).toEqual([40, 40]);
    expect(el?.style.width).toBe('80px');
    expect(el?.style.height).toBe('80px');
    expect(Number(el?.style.opacity)).toBeCloseTo(FADE_IN, 3);
    expect(labels(el as HTMLElement).map(s => s.textContent)).toEqual(['Alice', '']);
    frame();
    expect(Number(el?.style.opacity)).toBeCloseTo(1 - (1 - FADE_IN) ** 2, 3);
    expect(layer.style.width).toBe('400px');
    r.close();
  });

  it('reads the layout only when a size or the picture changes, or after MEASURE_S, not every frame (#486)', () => {
    let resized = () => {};
    let watched: Element[] = [];
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(cb: () => void) {
          resized = cb;
        }
        observe(el: Element) {
          watched.push(el);
        }
        disconnect() {
          watched = [];
        }
      },
    );
    const reads = vi.spyOn(window, 'getComputedStyle');
    const canvas = document.querySelector('canvas') as HTMLCanvasElement;
    const r = make();
    frame();
    expect(reads).toHaveBeenCalledTimes(1);
    expect(layer.style.width).toBe('400px');
    expect(watched).toEqual([canvas, canvas.parentElement]); // the picture and the box it sits in
    frame();
    frame();
    expect(reads).toHaveBeenCalledTimes(1); // nothing changed
    resized(); // the picture or its parent changed size: measured there and then
    expect(reads).toHaveBeenCalledTimes(2);
    canvas.width = 200; // a new picture size: measured within MEASURE_S
    untilMeasured();
    expect(reads).toHaveBeenCalledTimes(3);
    untilMeasured(); // and again after a while, in case it moved without changing size
    expect(reads).toHaveBeenCalledTimes(4);
    r.close();
    expect(watched).toEqual([]);
    reads.mockRestore();
  });

  it('shows the layer again when the picture comes back after a moment without one', () => {
    const r = make();
    frame();
    const canvas = document.querySelector('canvas') as HTMLCanvasElement;
    const parent = canvas.parentElement as HTMLElement;
    canvas.remove();
    untilMeasured();
    expect(layer.style.display).toBe('none');
    parent.prepend(canvas);
    untilMeasured();
    expect(layer.style.display).toBe('');
    expect(layer.style.width).toBe('400px');
    r.close();
  });

  it('follows an update, and crossfades a changed name', () => {
    const r = make();
    r.push([face], T);
    frame();
    // The same face a second later, moved right, now named: drawn at its filtered place, 286.651 per 1000 (OneEuro).
    vi.setSystemTime((T + 1) * 1000);
    r.push([{ ...face, box: [0.3, 0.2, 0.5, 0.6], label: 'Alicia' }], T + 1);
    frame();
    const [el] = boxes();
    expect(boxes()).toHaveLength(1);
    expect(translated(el)).toEqual([114.7, 40]);
    const [nu, old] = labels(el as HTMLElement);
    expect([nu?.textContent, old?.textContent]).toEqual(['Alicia', 'Alice']);
    expect([nu?.style.opacity, old?.style.opacity]).toEqual(['0.4', '0.6']); // 100 ms of a 250 ms crossfade
    frame();
    frame();
    expect([nu?.style.opacity, old?.style.opacity]).toEqual(['1', '0']);
    r.close();
  });

  it('fades out a box the next update leaves out, then removes it', () => {
    const r = make();
    r.push([face], T);
    frame();
    r.push([], T + 0.1);
    // From 0.48658, each 100 ms keeps exp(-0.1 / 0.15) = 0.51342 of it: under 0.02 after the fifth frame.
    for (let i = 1; i <= 4; i++) {
      frame();
      expect(Number(boxes()[0]?.style.opacity)).toBeCloseTo(FADE_IN * (1 - FADE_IN) ** i, 3);
    }
    frame();
    expect(boxes()).toHaveLength(0);
    expect(r.tracks.size).toBe(0);
    r.close();
  });

  it('fades every box when updates stop for over a second', () => {
    const r = make();
    r.push([face], T);
    frame();
    for (let i = 0; i < 9; i++) frame(); // 1 s after the update: still fresh
    expect(Number(boxes()[0]?.style.opacity)).toBeGreaterThan(0.99);
    frame();
    expect(Number(boxes()[0]?.style.opacity)).toBeLessThan(0.52);
    r.close();
  });

  describe('skeletons (#592)', () => {
    // One person with every joint placed and sure; a pose with no joint placed counts as none.
    const person = (): Pose => ({
      box: [0.2, 0.2, 0.6, 0.9],
      keypoints: Array.from({ length: 26 }, (_, i) => [0.3 + i * 0.01, 0.3 + i * 0.01, 0.9]),
    });
    const unplaced = (): Pose => ({ box: [0, 0, 1, 1], keypoints: Array.from({ length: 26 }, () => [-1, -1, 0.9]) });
    const svg = () => layer.querySelector<SVGSVGElement>('svg.bx-skeletons');
    const opacity = () => Number(svg()?.style.opacity);

    it('keeps each tracked skeleton color when pose order changes', () => {
      const r = make();
      const a = { ...person(), track_id: 'run:cam0:1', hue: 30 };
      const b = { ...person(), track_id: 'run:cam0:2', hue: 190 };
      r.skeletons([a, b]);
      const colors = () =>
        [...layer.querySelectorAll<SVGGElement>('[data-track-id]')].map(g => [
          g.dataset.trackId,
          g.style.getPropertyValue('--skeleton-color'),
        ]);
      expect(colors()).toEqual([
        ['run:cam0:1', 'hsl(30 90% 60%)'],
        ['run:cam0:2', 'hsl(190 90% 60%)'],
      ]);
      r.skeletons([b, a]);
      expect(colors()).toEqual([
        ['run:cam0:2', 'hsl(190 90% 60%)'],
        ['run:cam0:1', 'hsl(30 90% 60%)'],
      ]);
      r.close();
    });

    it('fade in when drawn', () => {
      const r = make();
      r.skeletons([person()]);
      frame();
      expect(opacity()).toBeCloseTo(FADE_IN, 3);
      r.close();
    });

    it('dim through a missed pose turn instead of blinking out, and come back', () => {
      const r = make();
      r.skeletons([person()]);
      for (let i = 0; i < 5; i++) frame();
      expect(opacity()).toBeGreaterThan(0.95);
      r.skeletons([]); // a pose turn that found nobody
      frame();
      expect(svg()).not.toBeNull();
      expect(opacity()).toBeLessThan(0.95);
      expect(opacity()).toBeGreaterThan(0.4);
      r.skeletons([person()]); // the next turn finds them again
      const dimmed = opacity();
      frame();
      expect(opacity()).toBeGreaterThan(dimmed);
      r.close();
    });

    it('are taken away once faded out', () => {
      const r = make();
      r.skeletons([person()]);
      for (let i = 0; i < 5; i++) frame();
      r.skeletons(null);
      for (let i = 0; i < 10; i++) frame();
      expect(svg()).toBeNull();
      r.close();
    });

    it('fade when updates stop for over a second, like the boxes', () => {
      const r = make();
      r.skeletons([person()]);
      for (let i = 0; i < 10; i++) frame(); // 1 s after the pose: still fresh
      expect(opacity()).toBeGreaterThan(0.99);
      frame();
      expect(opacity()).toBeLessThan(0.52);
      r.close();
    });

    it('treat poses with no joint placed as no skeleton', () => {
      const r = make();
      r.skeletons([person()]);
      for (let i = 0; i < 5; i++) frame();
      const before = svg()?.innerHTML;
      r.skeletons([unplaced()]);
      frame();
      expect(svg()?.innerHTML).toBe(before); // the last skeleton stays, fading
      expect(opacity()).toBeLessThan(0.95);
      r.close();
    });

    it('go at once when switched off', () => {
      const r = make();
      r.skeletons([person()]);
      for (let i = 0; i < 5; i++) frame();
      r.clearSkeletons();
      expect(svg()).toBeNull();
      frame();
      expect(svg()).toBeNull();
      r.close();
    });
  });

  it('starts a box afresh when time goes back (Review seeking back), rather than holding it where it was', () => {
    const r = make({ staleS: Number.POSITIVE_INFINITY });
    r.push([face], T);
    for (let i = 0; i < 5; i++) frame();
    const before = boxes()[0]?.style.transform;
    r.push([{ ...face, box: [0.5, 0.5, 0.7, 0.9] }], T - 10); // ten seconds earlier, elsewhere
    for (let i = 0; i < 5; i++) frame();
    expect(boxes()[0]?.style.transform).not.toBe(before);
    r.close();
  });

  it('keeps the boxes with no updates when told to (Review, paused: #374)', () => {
    const r = make({ staleS: Number.POSITIVE_INFINITY });
    r.push([face], T);
    for (let i = 0; i < 30; i++) frame(); // 3 s without an update
    expect(Number(boxes()[0]?.style.opacity)).toBeGreaterThan(0.99);
    r.close();
  });

  it('mirrors the box but not its label', () => {
    const r = make({ mirror: true });
    r.push([face], T);
    frame();
    const [el] = boxes();
    expect(translated(el)).toEqual([280, 40]); // x 0.1 to 0.3 becomes 0.7 to 0.9
    expect(el?.style.width).toBe('80px');
    expect(labels(el as HTMLElement)[0]?.textContent).toBe('Alice');
    r.close();
  });

  it('puts the label inside a box at the top edge', () => {
    const r = make();
    r.push([{ ...face, box: [0.1, 0.05, 0.3, 0.6] }], T); // 10 px from the top: no room above
    frame();
    expect(boxes()[0]?.classList.contains('bx-below')).toBe(true);
    r.close();
  });

  it('lets an unknown face be picked, with where it is', () => {
    const onPick = vi.fn();
    const r = make({ onPick });
    const stranger: Box = { id: 'f2', box: [0.5, 0.5, 0.7, 0.9], label: 'Face 3', kind: 'face', tone: 'unknown' };
    r.push([face, stranger], T);
    frame();
    const [known, unknown] = boxes();
    expect(known?.classList.contains('pickable')).toBe(false);
    expect(unknown?.classList.contains('pickable')).toBe(true);
    known?.click();
    expect(onPick).not.toHaveBeenCalled();
    unknown?.click();
    expect(onPick).toHaveBeenCalledWith({ ...stranger, left: 0, top: 0, width: 0, height: 0 }); // jsdom has no layout
    r.close();
  });

  it("with pickFaces 'all', lets every face be picked, by keyboard too (#352)", () => {
    const onPick = vi.fn();
    const r = make({ onPick, pickFaces: 'all' });
    r.push([face], T);
    frame();
    const [known] = boxes();
    expect(known?.classList.contains('pickable')).toBe(true);
    expect(known?.getAttribute('role')).toBe('button');
    expect(known?.tabIndex).toBe(0);
    expect(known?.getAttribute('aria-label')).toBe('Alice');
    known?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(onPick).toHaveBeenCalledWith({ ...face, left: 0, top: 0, width: 0, height: 0 });
    r.close();
  });

  it('makes objects keyboard-pickable only when explicitly enabled', () => {
    const onPick = vi.fn();
    const r = make({ onPick, pickObjects: true });
    const object: Box = { id: 'object', kind: 'object', label: 'mug', box: [0.1, 0.1, 0.4, 0.4] };
    r.push([object], 1);
    r.frame(1000);
    const button = layer.querySelector<HTMLElement>('.bx-object');
    expect(button?.getAttribute('role')).toBe('button');
    button?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(onPick).toHaveBeenCalledWith(expect.objectContaining({ kind: 'object', label: 'mug' }));
    r.close();
  });

  it('renders pets without offering face or object picking, including when a tracked marker changes kind', () => {
    const onPick = vi.fn();
    const r = make({ onPick, pickFaces: 'all', pickObjects: true });
    r.push([face], T);
    frame();
    const marker = boxes()[0] as HTMLElement;
    expect(marker.getAttribute('role')).toBe('button');

    const pet: Box = { ...face, kind: 'pet', label: 'Tabby · cat', tone: 'unknown' };
    r.push([pet], T + 0.1);
    frame();
    expect(boxes()).toEqual([marker]);
    expect(marker.classList.contains('bx-pet')).toBe(true);
    expect(marker.classList.contains('bx-face')).toBe(false);
    expect(labels(marker)[0]?.textContent).toBe('Tabby · cat');
    expect(marker.classList.contains('pickable')).toBe(false);
    expect(marker.hasAttribute('tabindex')).toBe(false);
    expect(marker.hasAttribute('role')).toBe(false);
    marker.click();
    marker.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    marker.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
    expect(onPick).not.toHaveBeenCalled();
    r.close();
  });

  it('offers no picking without onPick', () => {
    const r = make();
    r.push([{ ...face, tone: 'unknown' }], T);
    frame();
    expect(boxes()[0]?.classList.contains('pickable')).toBe(false);
    r.close();
  });

  it('hides the boxes over a snapshot fallback, which has its own drawn in', () => {
    const canvas = layer.parentElement?.querySelector('canvas');
    canvas?.classList.add('off');
    const snap = document.createElement('img');
    snap.className = 'snap';
    layer.before(snap);
    const r = make();
    frame();
    expect(layer.style.display).toBe('none');
    canvas?.classList.remove('off');
    untilMeasured();
    expect(layer.style.display).toBe('');
    r.close();
  });

  it('hides the boxes when there is no picture at all', () => {
    const r = make();
    frame();
    expect(layer.style.display).toBe('');
    layer.parentElement?.querySelector('canvas')?.remove();
    untilMeasured();
    expect(layer.style.display).toBe('none');
    r.close();
  });

  describe('settled (#2407)', () => {
    /** A box faded fully in and past its update: nothing about it changes any more. */
    const settled = () => {
      const r = make({ staleS: Number.POSITIVE_INFINITY });
      r.push([face], T);
      for (let i = 0; i < 12; i++) frame();
      return r;
    };

    it('stops asking for animation frames once its boxes settle, and an update starts them again', () => {
      const r = settled();
      expect(Number(boxes()[0]?.style.opacity)).toBe(1);
      expect(frames.pending).toBe(0); // idle: only the heartbeat timer is left
      r.push([{ ...face, box: [0.2, 0.2, 0.4, 0.6] }], T + 2);
      expect(frames.pending).toBe(1);
      r.close();
    });

    it('writes no style for a box that does not change', () => {
      const r = settled();
      // Count every write to a box's transform or opacity, even one to the value it already has.
      const writes: string[] = [];
      const proto = Object.getPrototypeOf(boxes()[0]?.style) as object;
      const restore: (() => void)[] = [];
      for (const name of ['transform', 'opacity', 'width', 'height']) {
        const d = Object.getOwnPropertyDescriptor(proto, name);
        if (!d?.set) continue;
        const set = d.set;
        Object.defineProperty(proto, name, {
          ...d,
          set(this: CSSStyleDeclaration, v: string) {
            writes.push(name);
            set.call(this, v);
          },
        });
        restore.push(() => Object.defineProperty(proto, name, d));
      }
      try {
        untilMeasured();
        untilMeasured();
      } finally {
        for (const undo of restore) undo();
      }
      expect(restore.length).toBeGreaterThan(0); // the setters were found and watched
      expect(writes).toEqual([]);
      r.close();
    });

    it('still fades a box out when updates stop, while idle', () => {
      const r = make(); // Live's: stale after a second without an update
      r.push([face], T);
      for (let i = 0; i < 9; i++) frame();
      expect(Number(boxes()[0]?.style.opacity)).toBe(1);
      for (let i = 0; i < 4; i++) frame();
      expect(Number(boxes()[0]?.style.opacity)).toBeLessThan(0.6);
      r.close();
    });
  });

  it('stops drawing and empties the layer when closed', () => {
    const r = make();
    r.push([face], T);
    frame();
    r.close();
    expect(layer.childElementCount).toBe(0);
    expect(frames.pending).toBe(0);
  });
});
