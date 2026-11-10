import { afterEach, describe, expect, it, vi } from 'vitest';
import { cropLiveObject, handoffReviewFrame, peekObjectDraft, takeLiveObject, takeObjectDraft } from './handoff';

afterEach(() => {
  takeLiveObject();
  vi.restoreAllMocks();
});

describe('Live object handoff', () => {
  it('crops normalized image coordinates and consumes the draft once', () => {
    const source = document.createElement('canvas');
    source.width = 1000;
    source.height = 500;
    const drawImage = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      drawImage,
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,AAAA');
    cropLiveObject(source, [0.1, 0.2, 0.6, 0.8], 'cam2', ' mug ');
    expect(drawImage).toHaveBeenCalledWith(source, 75, 85, 550, 330, 0, 0, 550, 330);
    const draft = takeLiveObject();
    expect(draft).toMatchObject({ camera: 'cam2', name: 'mug', image: 'data:image/jpeg;base64,AAAA' });
    // The detection itself (x 100 to 600, y 100 to 400) inside the 75,85 550x330 crop: what EdgeTAM is asked to outline.
    const box = draft && 'box' in draft ? draft.box : undefined;
    expect(box?.map(v => Number(v.toFixed(4)))).toEqual([0.0455, 0.0455, 0.9091, 0.9091]);
    expect(takeLiveObject()).toBeNull();
  });
  it('keeps the displayed capture time and portrait pixel coordinates for a network camera', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1_800_000_010_000);
    const source = document.createElement('canvas');
    source.width = 720;
    source.height = 1280;
    source.dataset.capturedAt = '1800000000.125';
    const drawImage = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      drawImage,
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,AAAA');
    cropLiveObject(source, [0.25, 0.25, 0.75, 0.75], 'entry-portrait', 'cup');
    expect(drawImage).toHaveBeenCalledWith(source, 162, 288, 396, 704, 0, 0, 396, 704);
    expect(takeObjectDraft()).toMatchObject({ camera: 'entry-portrait', name: 'cup', captured_at: 1_800_000_000.125 });
  });

  it('rejects missing and degenerate picture regions', () => {
    const source = document.createElement('canvas');
    source.width = 100;
    source.height = 100;
    expect(() => cropLiveObject(source, [0.5, 0.2, 0.5, 0.8], 'cam0', 'mug')).toThrow('too small');
    expect(takeLiveObject()).toBeNull();
  });
});

describe('Review frame handoff', () => {
  it('keeps the complete historical frame, its camera, and exact timestamp without a name', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1_800_000_000_000);
    const image = 'data:image/jpeg;base64,wholeFrame';
    handoffReviewFrame(image, 'cam2', 1_700_000_000.125);
    expect(peekObjectDraft()).toEqual({ kind: 'full_frame', camera: 'cam2', image, captured_at: 1_700_000_000.125 });
    expect(takeObjectDraft()).toEqual({ kind: 'full_frame', camera: 'cam2', image, captured_at: 1_700_000_000.125 });
    expect(takeObjectDraft()).toBeNull();
  });

  it('expires a historical frame one minute after handoff creation', () => {
    const now = vi.spyOn(Date, 'now').mockReturnValue(1_800_000_000_000);
    handoffReviewFrame('data:image/jpeg;base64,wholeFrame', 'cam1', 1_700_000_000);
    now.mockReturnValue(1_800_000_059_999);
    expect(peekObjectDraft()).not.toBeNull();
    now.mockReturnValue(1_800_000_060_001);
    expect(takeObjectDraft()).toBeNull();
  });

  it.each([NaN, Infinity, -1])('rejects an invalid recorded timestamp %s', capturedAt => {
    expect(() => handoffReviewFrame('data:image/jpeg;base64,wholeFrame', 'cam1', capturedAt)).toThrow(
      'No recorded frame',
    );
    expect(takeObjectDraft()).toBeNull();
  });
});

it('carries actual recorded geometry with the complete JPEG and unchanged capture timestamp', () => {
  const metadata = {
    capturedAtUs: 1_700_000_000_125_000,
    width: 640,
    height: 360,
    flip: true,
    profileStatus: 'absent' as const,
    nativeGeometry: { width: 640, height: 360 },
  };
  handoffReviewFrame('data:image/jpeg;base64,wholeFrame', 'cam2', 1_700_000_000.125, metadata);
  expect(takeObjectDraft()).toEqual({
    kind: 'full_frame',
    camera: 'cam2',
    captured_at: 1_700_000_000.125,
    image: 'data:image/jpeg;base64,wholeFrame',
    frame_metadata: metadata,
  });
});
