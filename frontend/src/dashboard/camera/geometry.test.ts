import { describe, expect, it } from 'vitest';
import {
  canvasFrameMetadata,
  contentRect,
  decodedBoxToSource,
  type FrameMetadata,
  sourceBoxToDecoded,
  sourceContentRect,
} from './geometry';

const metadata: FrameMetadata = {
  capturedAtUs: 1_700_000_000_000_000,
  width: 640,
  height: 360,
  flip: false,
  profileStatus: 'absent',
  nativeGeometry: { width: 640, height: 360 },
};
describe('actual capture geometry', () => {
  it('maps boxes one to one on a confirmed native capture', () => {
    expect(sourceContentRect(metadata)).toEqual({ x: 0, y: 0, w: 640, h: 360 });
    expect(sourceBoxToDecoded([0.25, 0, 0.5, 1], metadata)).toEqual([0.25, 0, 0.5, 1]);
    expect(decodedBoxToSource([0.25, 0, 0.5, 1], metadata)).toEqual([0.25, 0, 0.5, 1]);
    expect(sourceBoxToDecoded([0.25, 0, 0.5, 1], { ...metadata, flip: true })).toEqual([0.25, 0, 0.5, 1]);
    expect(contentRect(400, 400, 640, 360, 'contain')).toEqual({ x: 0, y: 87.5, w: 400, h: 225 });
  });
  it('rejects unknown geometry: another profile status, mismatched native size, or malformed boxes', () => {
    expect(sourceContentRect({ ...metadata, profileStatus: 'unavailable' })).toBeNull();
    expect(sourceContentRect({ ...metadata, profileStatus: 'unsupported' })).toBeNull();
    expect(sourceContentRect({ ...metadata, nativeGeometry: null })).toBeNull();
    expect(sourceContentRect({ ...metadata, width: 1280 })).toBeNull();
    expect(sourceBoxToDecoded([0, 0, 1, 1], { ...metadata, nativeGeometry: { width: 1280, height: 720 } })).toBeNull();
    expect(sourceBoxToDecoded([0.5, 0, 0.25, 1], metadata)).toBeNull();
  });
  it('requires actual canvas timestamp and dimensions', () => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    canvas.dataset.capturedAt = '1700000000';
    canvas.dataset.frameMetadata = JSON.stringify(metadata);
    expect(canvasFrameMetadata(canvas)).toEqual(metadata);
    canvas.dataset.capturedAt = '1700000001';
    expect(canvasFrameMetadata(canvas)).toBeNull();
  });
});
