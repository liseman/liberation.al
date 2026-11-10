import { describe, expect, it } from 'vitest';
import { avcParameterSets } from './avc';

const sps = [0x67, 66, 0xe0, 31, 0x80];
const pps = [0x68, 0xce];
const packet = (...units: number[][]) => new Uint8Array(units.flatMap(unit => [0, 0, 0, 1, ...unit]));
describe('AVC in-band configuration', () => {
  it('finds an SPS after AUD and SEI, preserving profile compatibility bits', () => {
    expect(avcParameterSets(packet([9, 0xf0], [6, 5, 0x80], sps, pps, [0x65, 0x88]))).toMatchObject({
      kind: 'ready',
      codec: 'avc1.42e01f',
    });
  });
  it('treats ordering, duplicate sets, and trailing Annex-B zero bytes as the same configuration', () => {
    const original = avcParameterSets(packet(sps, pps));
    expect(avcParameterSets(new Uint8Array([...packet(pps, sps, sps), 0, 0]))).toEqual(original);
  });
  it('detects SPS and PPS body changes even with the same codec string', () => {
    const original = avcParameterSets(packet(sps, pps));
    expect(avcParameterSets(packet([...sps, 0x81], pps))).not.toEqual(original);
    expect(avcParameterSets(packet(sps, [...pps, 0x81]))).not.toEqual(original);
  });
  it.each([
    [[0x67, 66, 0xe0]],
    [[0x67, 66, 0xe0, 31, 0x80]],
    [[0x68, 0xce]],
    [
      [0xe7, 66, 0xe0, 31, 0x80],
      [0x68, 0xce],
    ],
    [
      [0x07, 66, 0xe0, 31, 0x80],
      [0x68, 0xce],
    ],
    [
      [0x67, 66, 0xe3, 31, 0x80],
      [0x68, 0xce],
    ],
    [
      [0x67, 0, 0, 31, 0x80],
      [0x68, 0xce],
    ],
    [
      [0x67, 66, 0, 0, 0x80],
      [0x68, 0xce],
    ],
    [[0x67, 66, 0xe0, 31, 0x80], [0x68]],
    [
      [0x67, 66, 0xe0, 31, 0x80],
      [0x67, 100, 0, 40, 0x80],
      [0x68, 0xce],
    ],
  ])('rejects incomplete, forbidden, reserved, or conflicting parameter headers (%j)', (...units) => {
    expect(avcParameterSets(packet(...units))).toMatchObject({ kind: 'invalid' });
  });
  it('distinguishes an ordinary reference unit from incomplete parameter sets', () => {
    expect(avcParameterSets(packet([0x41, 0x80]))).toEqual({ kind: 'none' });
    expect(avcParameterSets(new Uint8Array())).toEqual({ kind: 'none' });
  });
  it('bounds parameter bytes and parameter count', () => {
    expect(avcParameterSets(packet([...sps, ...Array.from({ length: 65536 }, () => 0x55)], pps))).toMatchObject({
      kind: 'invalid',
    });
    expect(avcParameterSets(packet(...Array.from({ length: 257 }, () => sps), pps))).toMatchObject({ kind: 'invalid' });
  });
});
