import { renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { RawSegment } from '../review/types';
import { clipSegments, useClip } from './clip';

describe('clipSegments', () => {
  const seg = (cam: string, start: number, end: number): RawSegment =>
    ({ cam, start, end, url: `/v/${cam}/${start}` }) as RawSegment;

  it("keeps the clip camera's segments over the clip, in order, the last cut off at the clip's end", () => {
    const got = clipSegments(
      [seg('cam0', 300, 600), seg('cam1', 0, 300), seg('cam0', 0, 300), seg('cam0', 900, 1200)],
      'cam0',
      290,
      310,
    );
    expect(got.map(s => [s.start, s.end])).toEqual([
      [0, 300],
      [300, 310],
    ]);
  });

  it('is empty when nothing was kept for the clip', () => {
    expect(clipSegments([seg('cam1', 0, 300)], 'cam0', 10, 20)).toEqual([]);
  });
});

describe('useClip', () => {
  it("opens the camera's clip cut at its end, so a People card's video stops there too", async () => {
    const segments: RawSegment[] = [
      { cam: 'cam0', start: 100, end: 400, url: 'gateway:1:0:300000000' },
      { cam: 'cam1', start: 100, end: 400, url: 'gateway:2:0:300000000' },
    ];
    const { result, unmount } = renderHook(() =>
      useClip({ cam: 'cam0', start: 150, end: 160, loadSegments: async () => segments }),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.clip).toEqual({ at: 150, segments: [{ ...segments[0], end: 160 }] });
    unmount();
  });

  it('opens a moment-long clip (a 0 s sighting) with enough video to show a frame', async () => {
    const segments: RawSegment[] = [{ cam: 'cam0', start: 100, end: 400, url: 'gateway:1:0:300000000' }];
    const load = vi.fn(async (_start: number, _end: number, _signal: AbortSignal) => segments);
    const { result, unmount } = renderHook(() => useClip({ cam: 'cam0', start: 150, end: 150, loadSegments: load }));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(load).toHaveBeenCalledWith(150, 155, expect.any(AbortSignal));
    expect(result.current.clip?.segments[0]?.end).toBe(155);
    unmount();
  });

  it('says the recording is gone when nothing was kept', async () => {
    const { result, unmount } = renderHook(() =>
      useClip({ cam: 'cam0', start: 150, end: 160, loadSegments: async () => [] }),
    );
    await waitFor(() => expect(result.current.error).toBe('This recording is no longer kept.'));
    unmount();
  });
});
