// Each camera's picture shape, width over height (#2299): learnt from a still or a played canvas, and kept per
// camera so its other clips open at the right shape. A module of its own, apart from clip.ts, so a poster can size
// itself without loading Review's engine (#3127).
const ratios = new Map<string, number>();
export const WIDE = 16 / 9;

/** Width over height, or null for a size not known yet. */
export const ratioOf = (width: number, height: number): number | null =>
  width > 1 && height > 1 ? width / height : null;
export const cameraRatio = (cam: string): number | null => ratios.get(cam) ?? null;
export const learnRatio = (cam: string, ratio: number): void => {
  ratios.set(cam, ratio);
};
