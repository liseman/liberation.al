import { describe, expect, it } from 'vitest';
import { type Frame, fitView, frameAt } from './playback';

const at = (ts: number): Frame => ({ ts, box: [0, 0, 1, 1, 1], keypoints: [] });
const frames = [at(10), at(10.1), at(12)];

describe('frameAt', () => {
  it('shows the last frame at or before the moment', () => {
    expect(frameAt(frames, 10)?.ts).toBe(10);
    expect(frameAt(frames, 10.05)?.ts).toBe(10);
    expect(frameAt(frames, 10.3)?.ts).toBe(10.1);
  });
  it('shows nothing before the first frame or once the last one is stale', () => {
    expect(frameAt(frames, 9.9)).toBeNull();
    expect(frameAt(frames, 11)).toBeNull(); // 0.9 s after 10.1: the track wasn't seen
    expect(frameAt(frames, 12.4)?.ts).toBe(12);
    expect(frameAt([], 10)).toBeNull();
  });
});

describe('fitView', () => {
  const posed = (...joints: number[][]): Frame => ({ ts: 0, box: [0, 0, 1, 1, 1], keypoints: joints });
  it('centres a square on the sure joints, padded, and leaves out unsure and unplaced ones', () => {
    const frames = [posed([0.4, 0.5, 0.9], [0.5, 0.9, 0.9], [0.95, 0.05, 0.1]), posed([0.6, 0.8, 0.9], [-1, -1, 0])];
    const [x, y, side] = fitView(frames);
    expect(side).toBeCloseTo(0.4 * 1.3); // the taller side, 0.5 to 0.9, padded by 15% each way
    expect(x + side / 2).toBeCloseTo(0.5);
    expect(y + side / 2).toBeCloseTo(0.7);
  });
  it('shows the whole frame with no sure joints, and never more than it', () => {
    expect(fitView([])).toEqual([0, 0, 1]);
    expect(fitView([posed([0.5, 0.5, 0.1])])).toEqual([0, 0, 1]);
    expect(fitView([posed([0, 0, 1], [1, 1, 1])])[2]).toBe(1);
  });
});
