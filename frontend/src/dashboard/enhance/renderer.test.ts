import { afterEach, describe, expect, it, vi } from 'vitest';
import { SMALL_MAX, smallSizes } from './maths';
import { Enhancer, TooSlow } from './renderer';

/** A stand-in WebGL2 context that records every call, counts what is made and freed, and behaves as a lost context
 *  does once lost: nothing can be made on it and it has no extensions. jsdom has no WebGL. */
function fakeGl() {
  const made: Record<string, number> = {};
  const freed: Record<string, number> = {};
  const state = {
    lost: false,
    made,
    freed,
    /** Every call, by name. */
    calls: [] as string[],
    /** What was uploaded as a texture: the long side of each source. */
    uploads: [] as number[],
    /** Whether the graphics card is still on the frame before. */
    busy: false,
  };
  const count = (name: string) => state.calls.filter(c => c === name).length;
  const gl = new Proxy(
    {},
    {
      get(_, name) {
        if (typeof name !== 'string') return undefined;
        if (name === name.toUpperCase())
          return name === 'FRAMEBUFFER_COMPLETE' ? 1 : name === 'TIMEOUT_EXPIRED' ? 3 : 2;
        const create = /^create(\w+)$/.exec(name);
        const remove = /^delete(\w+)$/.exec(name);
        return (...args: unknown[]) => {
          state.calls.push(name);
          if (create || name === 'fenceSync') {
            if (state.lost) return null;
            const kind = create?.[1] ?? 'Sync';
            made[kind] = (made[kind] ?? 0) + 1;
            return {};
          }
          if (remove) {
            if (args[0]) freed[remove[1] ?? ''] = (freed[remove[1] ?? ''] ?? 0) + 1;
            return undefined;
          }
          if (name === 'getExtension') {
            if (state.lost) return null;
            if (args[0] !== 'WEBGL_lose_context') return {};
            return {
              loseContext: () => {
                state.lost = true;
              },
            };
          }
          if (name === 'texSubImage2D') {
            const source = args.at(-1) as { width: number; height: number };
            state.uploads.push(Math.max(source.width, source.height));
          }
          if (name === 'clientWaitSync') return state.busy ? 3 : 2;
          if (name === 'isContextLost') return state.lost;
          if (name === 'getShaderParameter' || name === 'getProgramParameter') return !state.lost;
          if (name === 'checkFramebufferStatus') return 1;
          if (name === 'getUniformLocation') return {};
          return undefined;
        };
      },
    },
  );
  return { gl, state, count };
}

/** Canvases whose WebGL2 context is gl and whose 2D context does nothing (a black picture), with the sizes each 2D
 *  canvas was drawn at. jsdom has neither. */
function canvases(gl: unknown) {
  const copies: number[] = [];
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
    this: HTMLCanvasElement,
    kind: string,
  ) {
    if (kind === 'webgl2') return gl as RenderingContext;
    return {
      drawImage: () => {
        copies.push(Math.max(this.width, this.height));
      },
      getImageData: (_x: number, _y: number, w: number, h: number) => ({ data: new Uint8ClampedArray(4 * w * h) }),
      setTransform: () => {},
    } as unknown as RenderingContext;
  } as typeof HTMLCanvasElement.prototype.getContext);
  return { canvas: document.createElement('canvas'), copies };
}

const FRAME = { width: 64, height: 36 } as unknown as ImageBitmap;
const FULL = { width: 1920, height: 1080 } as unknown as ImageBitmap;

afterEach(() => {
  document.body.replaceChildren();
  vi.restoreAllMocks();
});

describe('Enhancer.dispose', () => {
  it('leaves the context usable while its canvas is still on the page, so the view can start again on it', () => {
    const { gl, state } = fakeGl();
    const { canvas } = canvases(gl);
    document.body.append(canvas);
    new Enhancer(canvas).dispose();
    expect(state.lost).toBe(false);
    // What React's development double-mount does, and any remount that keeps the element
    const again = new Enhancer(canvas);
    again.draw(FRAME, 1);
    expect(again.stats.frames).toBe(1);
  });

  it('frees everything it made', () => {
    const { gl, state } = fakeGl();
    const { canvas } = canvases(gl);
    document.body.append(canvas);
    const enhancer = new Enhancer(canvas);
    enhancer.draw(FRAME, 1);
    enhancer.draw(FRAME, 1.03);
    enhancer.dispose();
    expect(state.made.Texture).toBeGreaterThan(0);
    expect(state.made.Sync).toBeGreaterThan(0);
    expect(state.freed).toEqual(state.made);
    // And draws nothing more
    enhancer.draw(FRAME, 1.06);
    expect(enhancer.stats.frames).toBe(2);
  });

  it('gives its context up at once when the canvas has left the page', () => {
    const { gl, state } = fakeGl();
    new Enhancer(canvases(gl).canvas).dispose();
    expect(state.lost).toBe(true);
  });
});

describe('the small copy', () => {
  it('halves the frame until its long side fits, and copies a small frame once at its own size', () => {
    expect(smallSizes(1920, 1080)).toEqual([
      { width: 960, height: 540 },
      { width: 480, height: 270 },
    ]);
    expect(smallSizes(1280, 720)).toEqual([{ width: 640, height: 360 }]);
    expect(smallSizes(640, 360)).toEqual([{ width: 640, height: 360 }]);
    expect(smallSizes(1080, 1920).at(-1)).toEqual({ width: 270, height: 480 });
    expect(smallSizes(3840, 2160)).toHaveLength(3);
  });
});

describe('Enhancer.draw', () => {
  it('never waits for the graphics card: no pixels, buffers, or finished work are read back from it', () => {
    const { gl, state } = fakeGl();
    const enhancer = new Enhancer(canvases(gl).canvas);
    for (let i = 0; i < 40; i++) enhancer.draw(FULL, 1 + i / 60);
    expect(enhancer.stats.frames).toBe(40);
    const waits = ['readPixels', 'getBufferSubData', 'finish', 'getError', 'getParameter', 'getSyncParameter'];
    expect(state.calls.filter(c => waits.includes(c))).toEqual([]);
    // The one question it asks the card (is the last frame done?) is asked without waiting: zero timeout
    expect(state.calls).toContain('clientWaitSync');
  });

  it('never sends the full frame to the graphics card, only the small copy', () => {
    const { gl, state } = fakeGl();
    const { canvas, copies } = canvases(gl);
    const enhancer = new Enhancer(canvas);
    for (let i = 0; i < 3; i++) enhancer.draw(FULL, 1 + i / 60);
    expect(state.uploads).toHaveLength(3);
    expect(Math.max(...state.uploads)).toBeLessThanOrEqual(SMALL_MAX);
    // Each frame: the halvings (960, 480), then the luma copy the motion estimate reads (256)
    expect(copies.slice(0, 3)).toEqual([960, 480, 256]);
    // The highlight's own canvas is no larger than it needs to be
    expect(Math.max(canvas.width, canvas.height)).toBeLessThanOrEqual(960);
  });

  it('keeps the last highlight over the new picture when the graphics card has not finished the one before', () => {
    const { gl, state, count } = fakeGl();
    const enhancer = new Enhancer(canvases(gl).canvas);
    enhancer.draw(FRAME, 1);
    const passes = count('drawArrays');
    expect(passes).toBeGreaterThan(0);
    state.busy = true;
    enhancer.draw(FRAME, 1.02);
    enhancer.draw(FRAME, 1.04);
    // Nothing more was sent to the card, but the frames were still steadied
    expect(count('drawArrays')).toBe(passes);
    expect(count('texSubImage2D')).toBe(1);
    expect(enhancer.stats).toMatchObject({ frames: 3, late: 2 });
    state.busy = false;
    enhancer.draw(FRAME, 1.06);
    expect(count('drawArrays')).toBe(2 * passes);
  });

  /** Feed frames at 60 a second with each draw taking ms of the page's thread. */
  function paced(ms: number, frames: number): Enhancer {
    const { gl } = fakeGl();
    const enhancer = new Enhancer(canvases(gl).canvas);
    let now = 0;
    let ending = false;
    vi.spyOn(performance, 'now').mockImplementation(() => {
      // Read at the start and at the end of a draw
      if (ending) now += ms;
      ending = !ending;
      return now;
    });
    for (let i = 0; i < frames; i++) enhancer.draw(FRAME, 1 + i / 60);
    return enhancer;
  }

  it('works on every frame while that takes little of the time between frames', () => {
    expect(paced(3, 200).stats).toMatchObject({ frames: 200, skipped: 0, every: 1 });
  });

  it('works on fewer frames when each takes too much of the time between them', () => {
    // 10 ms of a 16.7 ms frame is over the budget; every other frame is within it
    const { stats } = paced(10, 300);
    expect(stats.every).toBe(2);
    expect(stats.skipped).toBeGreaterThan(100);
    expect(stats.frames + stats.skipped).toBe(300);
  });

  it('stops, saying this computer cannot keep up, when even one frame in four takes too long', () => {
    expect(() => paced(40, 400)).toThrow(TooSlow);
  });
});
