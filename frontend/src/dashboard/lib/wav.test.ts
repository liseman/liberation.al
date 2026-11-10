import { describe, expect, it } from 'vitest';
import { encodeWav } from './wav';

const text = (wav: Uint8Array, o: number, n: number) => String.fromCharCode(...wav.slice(o, o + n));

describe('encodeWav', () => {
  it('writes a 16-bit mono PCM header', () => {
    const wav = encodeWav([new Float32Array([0, 0.5, -0.5]), new Float32Array([2, -2])], 48000);
    const v = new DataView(wav.buffer);
    expect(wav.length).toBe(44 + 5 * 2);
    expect([text(wav, 0, 4), text(wav, 8, 4), text(wav, 12, 4), text(wav, 36, 4)]).toEqual([
      'RIFF',
      'WAVE',
      'fmt ',
      'data',
    ]);
    expect(v.getUint32(4, true)).toBe(36 + 10);
    expect([v.getUint16(20, true), v.getUint16(22, true), v.getUint16(34, true)]).toEqual([1, 1, 16]); // PCM, mono
    expect([v.getUint32(24, true), v.getUint32(28, true), v.getUint32(40, true)]).toEqual([48000, 96000, 10]);
  });

  it('scales samples and clamps the ones out of range', () => {
    const v = new DataView(encodeWav([new Float32Array([0, 0.5, -0.5, 2, -2])], 16000).buffer);
    expect([0, 1, 2, 3, 4].map(i => v.getInt16(44 + i * 2, true))).toEqual([0, 16383, -16384, 32767, -32768]);
  });

  it('is a header alone with no samples', () => {
    expect(encodeWav([], 16000).length).toBe(44);
  });
});
