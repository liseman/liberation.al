import { afterEach, describe, expect, it, vi } from 'vitest';
import { Enhancer } from './renderer';

/** A stand-in WebGL2 context that counts what is made and freed, and behaves as a lost context does once lost:
 *  nothing can be made on it and it has no extensions. jsdom has no WebGL. */
function fakeGl() {
  const made: Record<string, number> = {};
  const freed: Record<string, number> = {};
  const state = { lost: false, made, freed };
  const gl = new Proxy(
    {},
    {
      get(_, name) {
        if (typeof name !== 'string') return undefined;
        if (name === name.toUpperCase()) return name === 'FRAMEBUFFER_COMPLETE' ? 1 : 2; // a constant
        const create = /^create(\w+)$/.exec(name);
        if (create)
          return () => {
            if (state.lost) return null;
            made[create[1] ?? ''] = (made[create[1] ?? ''] ?? 0) + 1;
            return {};
          };
        const remove = /^delete(\w+)$/.exec(name);
        if (remove)
          return (thing: unknown) => {
            if (thing) freed[remove[1] ?? ''] = (freed[remove[1] ?? ''] ?? 0) + 1;
          };
        if (name === 'getExtension')
          return (extension: string) => {
            if (state.lost) return null;
            if (extension !== 'WEBGL_lose_context') return {};
            return {
              loseContext: () => {
                state.lost = true;
              },
            };
          };
        if (name === 'isContextLost') return () => state.lost;
        if (name === 'getShaderParameter' || name === 'getProgramParameter') return () => !state.lost;
        if (name === 'checkFramebufferStatus') return () => 1;
        if (name === 'getUniformLocation') return () => ({});
        return () => undefined;
      },
    },
  );
  return { gl, state };
}

function canvasWith(gl: unknown): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  vi.spyOn(canvas, 'getContext').mockReturnValue(gl as RenderingContext);
  return canvas;
}

const FRAME = { width: 64, height: 36 } as unknown as ImageBitmap;

afterEach(() => {
  document.body.replaceChildren();
  vi.restoreAllMocks();
});

describe('Enhancer.dispose', () => {
  it('leaves the context usable while its canvas is still on the page, so the view can start again on it', () => {
    const { gl, state } = fakeGl();
    const canvas = canvasWith(gl);
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
    const canvas = canvasWith(gl);
    document.body.append(canvas);
    const enhancer = new Enhancer(canvas);
    enhancer.draw(FRAME, 1);
    enhancer.draw(FRAME, 1.03);
    enhancer.dispose();
    expect(state.made.Texture).toBeGreaterThan(0);
    expect(state.freed).toEqual(state.made);
    // And draws nothing more
    enhancer.draw(FRAME, 1.06);
    expect(enhancer.stats.frames).toBe(2);
  });

  it('gives its context up at once when the canvas has left the page', () => {
    const { gl, state } = fakeGl();
    new Enhancer(canvasWith(gl)).dispose();
    expect(state.lost).toBe(true);
  });
});
