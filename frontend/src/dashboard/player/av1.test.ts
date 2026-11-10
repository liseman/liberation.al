import { describe, expect, it } from 'vitest';
import { av1Sequence, looksLikeAv1 } from './av1';

const tu = (...bytes: number[]) => new Uint8Array(bytes);
// SVT-AV1 preset 8 output (the archive pass's), from a 1280x720 30 fps camera: a temporal delimiter, then the
// sequence header OBU (type 1, with its size), then the frame.
const KEY_720P = tu(
  0x12,
  0x00,
  0x0a,
  0x0f,
  0x02,
  0x00,
  0x00,
  0x2d,
  0x6a,
  0x67,
  0xfd,
  0x9e,
  0x35,
  0x7c,
  0xc8,
  0x08,
  0x08,
  0x0c,
  0x20,
  0x32,
  0x02,
  0x10,
  0x00,
);
// The same encoder at 320x240: level 2.0, without the initial display delay flag.
const KEY_240P = tu(
  0x12,
  0x00,
  0x0a,
  0x0e,
  0x00,
  0x00,
  0x00,
  0x04,
  0x3c,
  0xff,
  0xbc,
  0x4a,
  0xf9,
  0x10,
  0x60,
  0x60,
  0x60,
  0x40,
);

describe('av1Sequence', () => {
  it.each([
    ['720p, level 3.1', KEY_720P, 'av01.0.05M.08'],
    ['240p, level 2.0', KEY_240P, 'av01.0.00M.08'],
    // profile 0, no timing info, one operating point, level index 12 (5.0) in the high tier
    ['a high-tier level', tu(0x12, 0x00, 0x0a, 0x04, 0x00, 0x00, 0x00, 0x64), 'av01.0.12H.08'],
  ])('names the codec from %s', (_, bytes, codec) => {
    expect(av1Sequence(bytes)).toEqual({ kind: 'ready', codec, signature: expect.any(String) });
  });

  it('tells sequence headers apart, so a new one reconfigures the decoder', () => {
    const a = av1Sequence(KEY_720P);
    const b = av1Sequence(KEY_240P);
    expect(a.kind === 'ready' && b.kind === 'ready' && a.signature !== b.signature).toBe(true);
  });

  it('finds no sequence header in a delta unit', () => {
    expect(av1Sequence(tu(0x12, 0x00, 0x32, 0x01, 0x00))).toEqual({ kind: 'none' });
  });

  it.each([
    ['a size past the end', tu(0x12, 0x00, 0x0a, 0x0f, 0x02)],
    ['a forbidden bit', tu(0x92, 0x00)],
    ['a header too short to name a level', tu(0x0a, 0x01, 0x00)],
  ])('rejects %s', (_, bytes) => {
    expect(av1Sequence(bytes).kind).toBe('invalid');
  });
});

describe('looksLikeAv1', () => {
  it('knows a temporal delimiter from an Annex B start code', () => {
    expect(looksLikeAv1(KEY_720P)).toBe(true);
    expect(looksLikeAv1(tu(0, 0, 0, 1, 0x67))).toBe(false);
    expect(looksLikeAv1(tu())).toBe(false);
  });
});
