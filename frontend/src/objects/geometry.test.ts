import { describe, expect, it } from 'vitest';
import {
  bigEnough,
  boxBetween,
  clampFrameCrop,
  cropFromPoints,
  fitImageView,
  frameToView,
  imagePoint,
  MIN_BOX,
  moveFrameCrop,
  normalized,
  panImageView,
  pinchImageView,
  resizeFrameCrop,
  STRIP,
  viewToFrame,
  zoomImageView,
} from './geometry';

const image = { width: 640, height: 360 };
// The image drawn at half size, 100 px from the left edge of the viewport.
const rect = { left: 100, top: 20, width: 320, height: 180 } as DOMRect;

describe('imagePoint', () => {
  it('maps a pointer on the scaled canvas to image pixels', () => {
    expect(imagePoint(100 + 80, 20 + 45, rect, image)).toEqual([160, 90]);
  });

  it('clamps a pointer outside the canvas to the image edges', () => {
    expect(imagePoint(0, 0, rect, image)).toEqual([0, 0]);
    expect(imagePoint(10_000, 0, rect, image)[0]).toBe(640);
  });

  it('keeps the point above the caption strip', () => {
    const [, y] = imagePoint(150, 20 + 180, rect, image); // the very bottom of the canvas
    expect(y).toBe(360 - STRIP - 1);
  });
});

describe('boxBetween', () => {
  it('gives the same box whichever corner the drag started from', () => {
    const box = [10, 20, 30, 40];
    expect(boxBetween([10, 20], [40, 60])).toEqual(box);
    expect(boxBetween([40, 60], [10, 20])).toEqual(box);
    expect(boxBetween([40, 20], [10, 60])).toEqual(box);
  });
});

describe('bigEnough', () => {
  it.each([
    [[0, 0, MIN_BOX, MIN_BOX], true],
    [[0, 0, MIN_BOX - 1, 100], false],
    [[0, 0, 100, MIN_BOX - 1], false],
  ] as const)('%j is %s', (box, expected) => {
    expect(bigEnough([...box])).toBe(expected);
  });

  it('rejects no selection', () => {
    expect(bigEnough(null)).toBe(false);
  });
});

describe('normalized', () => {
  it('divides by the image size', () => {
    expect(normalized([64, 36, 320, 180], image)).toEqual([0.1, 0.1, 0.5, 0.5]);
  });
});

describe('full-frame editor geometry', () => {
  const viewport = { width: 320, height: 180 };
  it('fits the image without changing pixel coordinates or trimming a caption strip', () => {
    expect(fitImageView(image, viewport)).toEqual({ scale: 0.5, x: 0, y: 0 });
    expect(cropFromPoints([0, 0], [640, 360], image)).toEqual([0, 0, 640, 360]);
    expect(fitImageView({ width: 640, height: 640 }, viewport)).toEqual({ scale: 0.28125, x: 70, y: 0 });
  });
  it.each([
    { point: [0, 0] as [number, number], view: { scale: 0.5, x: 0, y: 0 } },
    { point: [640, 360] as [number, number], view: { scale: 2, x: -200, y: -90 } },
    { point: [123.25, 45.5] as [number, number], view: { scale: 1.25, x: -33.5, y: 12 } },
  ])('round trips a frame point through a zoomed and panned viewport: $point', ({ point, view }) => {
    const actual = viewToFrame(frameToView(point, view), view);
    expect(actual[0]).toBeCloseTo(point[0], 10);
    expect(actual[1]).toBeCloseTo(point[1], 10);
  });
  it('zooms around the pointer rather than the center of the viewport', () => {
    const fit = fitImageView(image, viewport);
    const zoomed = zoomImageView(fit, [80, 45], 1, image, viewport);
    expect(zoomed).toEqual({ scale: 1, x: -80, y: -45 });
    expect(viewToFrame([80, 45], zoomed)).toEqual([160, 90]);
  });
  it('combines pinch midpoint translation and focal zoom from the original gesture', () => {
    const pinched = pinchImageView(fitImageView(image, viewport), [160, 90], [180, 100], 2, image, viewport);
    expect(pinched).toEqual({ scale: 1, x: -140, y: -80 });
    expect(viewToFrame([180, 100], pinched)).toEqual([320, 180]);
  });
  it('bounds zoom and pan so the image cannot disappear', () => {
    expect(zoomImageView(fitImageView(image, viewport), [160, 90], 0.01, image, viewport)).toEqual({
      scale: 0.5,
      x: 0,
      y: 0,
    });
    expect(panImageView({ scale: 1, x: -100, y: -50 }, [1000, -1000], image, viewport)).toEqual({
      scale: 1,
      x: 0,
      y: -180,
    });
    expect(zoomImageView(fitImageView(image, viewport), [160, 90], 1000, image, viewport).scale).toBe(8);
  });
  it.each([
    { box: [-50, -30, 1000, 900] as [number, number, number, number], expected: [0, 0, 640, 360] },
    { box: [635, 355, 2, 2] as [number, number, number, number], expected: [624, 344, 16, 16] },
  ])('clamps crop edges and minimum size: $box', ({ box, expected }) => {
    expect(clampFrameCrop(box, image)).toEqual(expected);
    expect(clampFrameCrop(clampFrameCrop(box, image), image)).toEqual(expected);
  });
  it('quantizes full-resolution crop edges after fractional zoom and resizing', () => {
    const view = { scale: 0.625, x: -40, y: -22.5 };
    const crop = cropFromPoints(viewToFrame([30, 20], view), viewToFrame([130, 70], view), image);
    expect(crop).toEqual([112, 68, 160, 80]);
    expect(moveFrameCrop(crop, [0.4, 0.7], image)).toEqual([112, 69, 160, 80]);
    expect(resizeFrameCrop(crop, 'se', [0.6, 0.3], image)).toEqual([112, 68, 161, 80]);
    expect(resizeFrameCrop(crop, 'nw', [999.2, 999.2], image)).toEqual([256, 132, 16, 16]);
  });

  it('keeps opposite resize edges fixed, including when a handle crosses them', () => {
    expect(resizeFrameCrop([100, 80, 200, 120], 'nw', [999, 999], image)).toEqual([284, 184, 16, 16]);
    expect(resizeFrameCrop([100, 80, 200, 120], 'se', [999, 999], image)).toEqual([100, 80, 540, 280]);
    expect(resizeFrameCrop([100, 80, 200, 120], 'w', [-999, 999], image)).toEqual([0, 80, 300, 120]);
    expect(moveFrameCrop([100, 80, 200, 120], [-999, 999], image)).toEqual([0, 240, 200, 120]);
  });
});
