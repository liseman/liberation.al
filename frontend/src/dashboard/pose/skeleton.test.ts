import { describe, expect, it } from 'vitest';
import { BONES, HALPE26, MIN_CONF, wireframe } from './skeleton';

const unplaced = [-1, -1, 0];
const pose = (joints: Record<number, number[]>) => ({
  box: [0.2, 0.1, 0.6, 0.9, 0.9],
  keypoints: HALPE26.map((_, i) => joints[i] ?? unplaced),
});

describe('wireframe', () => {
  it('draws only the joints, and the bones between them, the model is sure of', () => {
    const p = pose({
      5: [0.4, 0.3, 0.9], // left shoulder
      7: [0.35, 0.45, 0.8], // left elbow
      9: [0.3, 0.6, 0.4], // left wrist: a guess, under see-actions' sure joint (0.5)
      6: [0.55, 0.3, MIN_CONF], // right shoulder: exactly at the line, kept
    });
    const { bones, joints } = wireframe(p);
    expect(joints).toEqual([
      [0.4, 0.3],
      [0.55, 0.3],
      [0.35, 0.45],
    ]);
    expect(bones).toEqual([[0.4, 0.3, 0.35, 0.45]]); // shoulder to elbow; elbow to wrist is dropped with the wrist
  });

  it('leaves out a joint the model did not place, whatever its confidence says', () => {
    expect(wireframe(pose({ 0: [-1, -1, 0.9] })).joints).toEqual([]);
  });

  it('joins only real joints', () => {
    expect(BONES.every(([a, b]) => a >= 0 && b >= 0 && a < HALPE26.length && b < HALPE26.length && a !== b)).toBe(true);
  });

  it('never joins an ear to a shoulder (the neck carries the shoulders)', () => {
    const ears = [HALPE26.indexOf('left ear'), HALPE26.indexOf('right ear')];
    const shoulders = [HALPE26.indexOf('left shoulder'), HALPE26.indexOf('right shoulder')];
    expect([...ears, ...shoulders].every(i => i >= 0)).toBe(true);
    const joinsEarToShoulder = ([a, b]: readonly [number, number]) =>
      (ears.includes(a) && shoulders.includes(b)) || (ears.includes(b) && shoulders.includes(a));
    expect(BONES.some(joinsEarToShoulder)).toBe(false);
  });
});
