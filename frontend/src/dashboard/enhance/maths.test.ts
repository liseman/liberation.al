import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
  BandPass,
  bandTaus,
  changeStrength,
  DEADBAND,
  EDGE_MARGIN,
  edgeMask,
  emaAlpha,
  estimateShift,
  type Luma,
  lumaSize,
  prepareReference,
  Stabiliser,
} from './maths';

const W = 256;
const H = 144;

/** A made-up textured scene: a sum of soft ripples at fixed seeded angles, defined everywhere, so a shift of any
 *  fraction of a pixel can be drawn exactly. */
function scene(seed = 1): (x: number, y: number) => number {
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  const waves = Array.from({ length: 24 }, () => ({
    kx: (rand() - 0.5) * 0.6,
    ky: (rand() - 0.5) * 0.6,
    phase: rand() * Math.PI * 2,
    amp: 0.02 + rand() * 0.03,
  }));
  return (x, y) => 0.5 + waves.reduce((sum, w) => sum + w.amp * Math.sin(w.kx * x + w.ky * y + w.phase), 0);
}

/** The scene seen by a camera moved by (dx, dy): frame(x, y) = scene(x - dx, y - dy), so frame(x + dx) = scene(x). */
function frame(
  look: (x: number, y: number) => number,
  dx = 0,
  dy = 0,
  extra?: (x: number, y: number) => number | null,
): Luma {
  const data = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data[y * W + x] = extra?.(x, y) ?? look(x - dx, y - dy);
  return { data, width: W, height: H };
}

/** Deterministic noise of about the given size, as H.264 leaves on a still picture. */
function noisy(l: Luma, size: number, seed: number): Luma {
  let s = seed;
  const data = l.data.map(v => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return v + (s / 4294967296 - 0.5) * 2 * size;
  });
  return { ...l, data };
}

describe('emaAlpha', () => {
  it('weighs a new sample the same per second at any frame rate', () => {
    // Ten steps at 1/30 s leave the same share of the old value as one step of 1/3 s
    const tenSmall = (1 - emaAlpha(1 / 30, 0.5)) ** 10;
    const oneLarge = 1 - emaAlpha(1 / 3, 0.5);
    expect(tenSmall).toBeCloseTo(oneLarge, 10);
  });

  it('takes nothing from no time and everything with no memory', () => {
    expect(emaAlpha(0, 0.5)).toBe(0);
    expect(emaAlpha(0.1, 0)).toBe(1);
  });
});

describe('BandPass', () => {
  it('passes nothing on a still picture', () => {
    const band = new BandPass(0.06, 0.6);
    const still = Float32Array.of(0.2, 0.5, 0.9);
    for (let i = 0; i < 50; i++) expect([...band.step(still, 1 / 30)]).toEqual([0, 0, 0]);
  });

  it('lights up on a change and fades once it holds', () => {
    const band = new BandPass(0.06, 0.6);
    band.step(Float32Array.of(0.2), 1 / 30);
    const first = band.step(Float32Array.of(0.6), 1 / 30)[0] ?? 0;
    expect(first).toBeGreaterThan(0.1);
    let last = first;
    for (let i = 0; i < 150; i++) last = band.step(Float32Array.of(0.6), 1 / 30)[0] ?? 0;
    expect(Math.abs(last)).toBeLessThan(0.01);
  });

  it('passes a flicker inside its band more than ones far below or above it', () => {
    const { fast, slow } = bandTaus(0.25, 2.5);
    const swing = (hz: number) => {
      const band = new BandPass(fast, slow);
      let peak = 0;
      for (let i = 0; i < 30 * 20; i++) {
        const out = band.step(Float32Array.of(0.5 + 0.1 * Math.sin(2 * Math.PI * hz * (i / 30))), 1 / 30)[0] ?? 0;
        if (i > 30 * 10) peak = Math.max(peak, Math.abs(out)); // after it settles
      }
      return peak;
    };
    expect(swing(0.8)).toBeGreaterThan(2 * swing(0.02));
    expect(swing(0.8)).toBeGreaterThan(2 * swing(14));
  });
});

describe('changeStrength', () => {
  it('shows nothing at the compression noise floor and all of a clear change on flat ground', () => {
    expect(changeStrength(0.005, 0)).toBe(0);
    expect(changeStrength(0.08, 0)).toBe(1);
  });

  it('shows the same change less on an edge than on flat ground', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0.02, max: 0.2, noNaN: true }),
        fc.double({ min: 0.05, max: 1, noNaN: true }),
        (change, edge) => {
          expect(changeStrength(change, edge)).toBeLessThanOrEqual(changeStrength(change, 0));
        },
      ),
    );
    // A sub-pixel residual along a strong edge stays dark
    expect(changeStrength(0.04, 0.4)).toBe(0);
  });

  it('is symmetric in the sign of the change', () => {
    expect(changeStrength(-0.05, 0.01)).toBe(changeStrength(0.05, 0.01));
  });
});

describe('edgeMask', () => {
  it('shows nothing within 3 % of any edge, more with the frame shifted, and everything in the middle', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: EDGE_MARGIN, noNaN: true }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        fc.double({ min: -0.05, max: 0.05, noNaN: true }),
        fc.double({ min: -0.05, max: 0.05, noNaN: true }),
        (edge, along, sx, sy) => {
          for (const [x, y] of [
            [edge, along],
            [1 - edge, along],
            [along, edge],
            [along, 1 - edge],
          ] as const)
            expect(edgeMask(x, y, sx, sy)).toBe(0);
        },
      ),
    );
    expect(edgeMask(0.05, 0.5, 0.03, 0)).toBe(0); // a shift of 3 % widens the margin by as much
    expect(edgeMask(0.5, 0.5, 0.05, 0.05)).toBe(1);
  });
});

describe('estimateShift', () => {
  const look = scene(7);
  const ref = prepareReference(frame(look));

  it('finds a camera wobble to a few hundredths of a pixel, in any direction up to a few pixels', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -4, max: 4, noNaN: true }),
        fc.double({ min: -4, max: 4, noNaN: true }),
        (dx, dy) => {
          const found = estimateShift(ref, frame(look, dx, dy));
          expect(found).not.toBeNull();
          expect(found?.dx ?? Number.NaN).toBeCloseTo(dx, 1);
          expect(found?.dy ?? Number.NaN).toBeCloseTo(dy, 1);
        },
      ),
      { numRuns: 25 },
    );
  });

  it('ignores someone walking through a corner of the picture', () => {
    const person = (x: number, y: number) => (x > 20 && x < 110 && y > 10 && y < 90 ? 0.9 - 0.002 * x : null);
    const found = estimateShift(ref, frame(look, 1.5, -0.5, person));
    expect(found?.dx ?? Number.NaN).toBeCloseTo(1.5, 1);
    expect(found?.dy ?? Number.NaN).toBeCloseTo(-0.5, 1);
  });

  it('gives up on a picture with nothing to hold on to', () => {
    const flat = { data: new Float32Array(W * H).fill(0.5), width: W, height: H };
    expect(estimateShift(prepareReference(flat), flat)).toBeNull();
  });

  it('gives up on a different scene rather than guessing', () => {
    expect(estimateShift(ref, frame(scene(99)))).toBeNull();
  });
});

describe('Stabiliser', () => {
  const look = scene(3);

  it('holds a still camera exactly still through compression noise', () => {
    const stab = new Stabiliser();
    stab.update(noisy(frame(look), 0.01, 1), 1 / 30);
    for (let i = 2; i < 120; i++) {
      const out = stab.update(noisy(frame(look), 0.01, i), 1 / 30);
      expect(out.reset).toBe(false);
      expect(out.dx).toBe(0);
      expect(out.dy).toBe(0);
    }
  });

  it('pulls a wobbling camera back onto its first frame and fades the highlight in', () => {
    const stab = new Stabiliser();
    const first = stab.update(frame(look), 1 / 30);
    expect(first).toMatchObject({ reset: true, gain: 0 });
    let out = first;
    for (let i = 1; i <= 30; i++) {
      const wobble = 2 * Math.sin(i);
      out = stab.update(frame(look, wobble, -wobble / 2), 1 / 30);
      expect(out.reset).toBe(false);
      // Sampling the frame at x + dx puts it back on the reference, short of the dead band
      expect(Math.abs(out.dx * W - wobble)).toBeLessThanOrEqual(DEADBAND + 0.05);
    }
    expect(out.gain).toBeCloseTo(1, 6);
  });

  it('starts over and hides the highlight when the camera moves too far to compensate', () => {
    const stab = new Stabiliser();
    for (let i = 0; i < 40; i++) stab.update(frame(look), 1 / 30);
    const moved = stab.update(frame(scene(11)), 1 / 30);
    expect(moved).toMatchObject({ reset: true, gain: 0, dx: 0, dy: 0 });
    const next = stab.update(frame(scene(11)), 1 / 30);
    expect(next.reset).toBe(false);
    expect(next.gain).toBeLessThan(0.1);
  });

  it('starts over after a gap in the stream', () => {
    const stab = new Stabiliser();
    stab.update(frame(look), 1 / 30);
    expect(stab.update(frame(look), 2).reset).toBe(true);
  });
});

describe('lumaSize', () => {
  it('keeps the frame shape with a long side of 256 and sides the shader can pack four to a texel', () => {
    expect(lumaSize(1920, 1080)).toEqual({ width: 256, height: 144 });
    expect(lumaSize(1080, 1920)).toEqual({ width: 144, height: 256 });
    fc.assert(
      fc.property(fc.integer({ min: 16, max: 8000 }), fc.integer({ min: 16, max: 8000 }), (w, h) => {
        const s = lumaSize(w, h);
        expect(s.width % 4).toBe(0);
        expect(s.height % 4).toBe(0);
        expect(Math.max(s.width, s.height)).toBe(256);
      }),
    );
  });
});
