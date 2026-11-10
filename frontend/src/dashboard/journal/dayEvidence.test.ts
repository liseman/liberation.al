import { describe, expect, it } from 'vitest';
import { type EvidenceSource, evidenceAt, NEAR_S } from './dayEvidence';

const T = 1_700_000_000;
const src = (over: Partial<EvidenceSource>): EvidenceSource => ({
  label: 'h1',
  stream: 'camera.caption',
  ts_start: T,
  ts_end: T,
  stage: 'clip',
  caption: 'a kitchen',
  camera: 'cam0',
  t: T,
  url: '/journal/evidence/2023-11-14/cam0-1700000000000.mp4',
  ...over,
});

describe('evidenceAt', () => {
  it("picks the shown camera's source with the moment nearest the playhead", () => {
    const near = src({ label: 'h2', t: T + 2, ts_start: T + 2, ts_end: T + 2 });
    const sources = [src({ camera: 'cam1' }), src({ t: T - 4, ts_start: T - 4, ts_end: T - 4 }), near];
    expect(evidenceAt(sources, 'cam0', T + 1)).toBe(near);
  });

  it.each([
    ['within NEAR_S of the moment', T + NEAR_S, true],
    ['just past it', T + NEAR_S + 0.5, false],
  ])('matches a moment %s', (_, t, matches) => {
    expect(evidenceAt([src({})], 'cam0', t) !== null).toBe(matches);
  });

  it("never shows a long source's clip far from its moment, and never a speech source", () => {
    // An action over a minute: its clip is the 6 s around its moment, not anything 40 s on (review of #2846).
    const span = src({ ts_start: T, ts_end: T + 60, t: T + 2 });
    expect(evidenceAt([span], 'cam0', T + 40)).toBeNull();
    expect(evidenceAt([span], 'cam0', T + 4)).toBe(span);
    expect(evidenceAt([src({ camera: null, stage: 'audio' })], 'cam0', T)).toBeNull();
  });
});
