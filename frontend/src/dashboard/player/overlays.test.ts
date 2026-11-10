import { describe, expect, it } from 'vitest';
import { interpolate, KEEP, type Overlay, OverlayMap } from './overlays';

function overlay(captureUs: number, x = 0.1): Overlay {
  return {
    captureUs,
    boxes: [{ xyxy: [x, 0.2, x + 0.1, 0.4], score: 0.9, label: `at ${captureUs}`, hue: 120 }],
    skeletons: [
      {
        box: { xyxy: [x, 0, x + 0.5, 1], score: 1, label: '', hue: null },
        layout: 'line',
        joints: [
          [x, 0.5, 0.9],
          [-1, -1, 0],
        ],
      },
    ],
    captions: [{ text: `t=${captureUs}`, speaker: '' }],
  };
}

describe('OverlayMap.pick', () => {
  it('keeps nearby complete poses when a newer face-only snapshot arrives', () => {
    const m = new OverlayMap();
    const poses = overlay(1_000_000);
    poses.skeletons.push({
      box: { xyxy: [0.6, 0, 0.9, 1], score: 1, label: '', hue: null },
      layout: 'line',
      joints: [[0.6, 0.5, 0.9]],
    });
    m.add(poses);
    m.add({ captureUs: 1_033_333, boxes: [], skeletons: [], captions: [] });
    const chosen = m.pick(1_033_333);
    expect(chosen.kind).toBe('exact');
    if (chosen.kind !== 'none') expect(chosen.overlay.skeletons).toEqual(poses.skeletons);
    m.add({ captureUs: 1_100_001, boxes: [], skeletons: [], captions: [] });
    const stale = m.pick(1_100_001);
    if (stale.kind !== 'none') expect(stale.overlay.skeletons).toEqual([]);
  });
  it('keeps every person from the nearer snapshot when the number of skeletons changes', () => {
    const before = overlay(1000);
    const after = overlay(3000);
    after.skeletons.push({
      box: { xyxy: [0.6, 0, 0.9, 1], score: 1, label: '', hue: null },
      layout: 'line',
      joints: [[0.6, 0.5, 0.9]],
    });
    expect(interpolate(before, after, 0.75).skeletons).toEqual(after.skeletons);
    expect(interpolate(after, before, 0.75).skeletons).toEqual(before.skeletons);
  });
  it('returns the exact overlay for a frame time', () => {
    const m = new OverlayMap();
    m.add(overlay(1000));
    m.add(overlay(2000));
    const p = m.pick(2000);
    expect(p.kind).toBe('exact');
    if (p.kind === 'exact') expect(p.overlay.captureUs).toBe(2000);
  });

  it('interpolates between the two nearest when the frame falls between them', () => {
    const m = new OverlayMap();
    m.add(overlay(1000, 0.0));
    m.add(overlay(3000, 0.4));
    const p = m.pick(2000);
    expect(p.kind).toBe('interpolated');
    if (p.kind !== 'interpolated') return;
    expect(p.between).toEqual([1000, 3000]);
    expect(p.overlay.boxes[0]?.xyxy[0]).toBeCloseTo(0.2);
    expect(p.overlay.skeletons[0]?.joints[0]?.[0]).toBeCloseTo(0.2);
    expect(p.overlay.skeletons[0]?.joints[1]).toEqual([-1, -1, 0]); // an unplaced joint stays unplaced
    expect(p.overlay.captureUs).toBe(2000);
  });

  it('takes the nearest within the gap at the edges, and nothing beyond it', () => {
    const m = new OverlayMap();
    m.add(overlay(1_000_000));
    expect(m.pick(1_050_000)).toMatchObject({ kind: 'nearest', offsetUs: -50_000 });
    expect(m.pick(1_200_000)).toEqual({ kind: 'none' });
    expect(m.pick(900_000)).toMatchObject({ kind: 'nearest', offsetUs: 100_000 });
    expect(new OverlayMap().pick(5)).toEqual({ kind: 'none' });
  });

  it('does not interpolate across a long gap', () => {
    const m = new OverlayMap();
    m.add(overlay(0));
    m.add(overlay(10_000_000));
    expect(m.pick(50_000)).toMatchObject({ kind: 'nearest' });
    expect(m.pick(5_000_000)).toEqual({ kind: 'none' });
  });

  it('keeps only the newest KEEP overlays', () => {
    const m = new OverlayMap();
    for (let i = 0; i <= KEEP + 10; i++) m.add(overlay(i * 33_000));
    expect(m.size).toBe(KEEP);
    expect(m.pick(0)).toEqual({ kind: 'none' });
    expect(m.pick((KEEP + 10) * 33_000).kind).toBe('exact');
    m.clear();
    expect(m.size).toBe(0);
  });

  it('preserves legacy poses when their joint counts cannot be interpolated', () => {
    const a = overlay(0);
    const b = overlay(100);
    b.skeletons[0]?.joints.pop();
    expect(interpolate(a, b, 0.75).skeletons).toEqual(a.skeletons);
  });

  it('interpolate pairs boxes by index and keeps the unmatched ones', () => {
    const a = overlay(0);
    const b = { ...overlay(100), boxes: [] };
    const mid = interpolate(a, b, 0.5);
    expect(mid.boxes).toHaveLength(1);
    expect(mid.captions[0]?.text).toBe('t=100');
  });
});

describe('tracked interpolation', () => {
  function tracked(at: number, tracks: [string, number][]): Overlay {
    const rows = tracks.map(([id, x]) => {
      const row = overlay(at, x);
      const originalBox = row.boxes[0];
      const originalSkeleton = row.skeletons[0];
      if (!originalBox || !originalSkeleton) throw new Error('test overlay missing person');
      const box = { ...originalBox, id, kind: 'body' };
      const skeleton = { ...originalSkeleton, box: { ...originalSkeleton.box, id, kind: 'body' } };
      return { box, skeleton };
    });
    return { captureUs: at, boxes: rows.map(r => r.box), skeletons: rows.map(r => r.skeleton), captions: [] };
  }
  it('matches boxes and joints by track when detection order swaps', () => {
    const a = tracked(0, [
      ['a', 0],
      ['b', 0.8],
    ]);
    const b = tracked(100, [
      ['b', 0.6],
      ['a', 0.2],
    ]);
    const mid = interpolate(a, b, 0.25);
    expect(mid.boxes.map(box => [box.id, box.xyxy[0]])).toEqual([
      ['a', 0.05],
      ['b', 0.75],
    ]);
    expect(mid.skeletons.map(s => [s.box.id, s.joints[0]?.[0]])).toEqual([
      ['a', 0.05],
      ['b', 0.75],
    ]);
  });
  it('does not blend a replacement track into the person who left', () => {
    const a = tracked(0, [
      ['a', 0],
      ['b', 0.8],
    ]);
    const b = tracked(100, [
      ['a', 0.2],
      ['c', 0.6],
    ]);
    const nearA = interpolate(a, b, 0.25);
    expect(nearA.skeletons[1]).toEqual(a.skeletons[1]);
    expect(nearA.boxes[1]).toEqual(a.boxes[1]);
    const nearB = interpolate(a, b, 0.75);
    expect(nearB.skeletons[1]).toEqual(b.skeletons[1]);
    expect(nearB.boxes[1]).toEqual(b.boxes[1]);
  });
});
