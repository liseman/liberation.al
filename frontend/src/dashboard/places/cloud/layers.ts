// A scene of several cameras' clouds (#445) draws each camera's points as its own layer: shown or hidden, left in
// its own colours or tinted with the camera's colour so it is plain whose a point is, and with the points outside the
// plan left out or kept. What is drawn is worked out here, from the points' own tags (load.ts), so it is tested
// without a canvas.

/** A colour a camera, in the order of the scene's cameras: Okabe and Ito's palette, which stays apart for people
 *  who see colour differently, on a light ground and a dark one. Red, green, blue, 0 to 255. */
export const TINTS: readonly (readonly [number, number, number])[] = [
  [0, 114, 178],
  [230, 159, 0],
  [0, 158, 115],
  [204, 121, 167],
  [213, 94, 0],
  [86, 180, 233],
  [240, 228, 66],
];

export const tintOf = (index: number): readonly [number, number, number] =>
  TINTS[index % TINTS.length] as readonly [number, number, number];

/** A camera's tint as a CSS colour, for its swatch beside its name. */
export const tintCss = (index: number): string => `rgb(${tintOf(index).join(' ')})`;

/** How much of a tinted point is the camera's colour; the rest is the point's own. */
export const TINT_SHARE = 0.45;

/** Every point's colour with its camera's tint mixed in (red, green, blue bytes each). A point with no colour of
 *  its own is the tint alone. */
export function tinted(count: number, colours: Uint8Array | null, views: Uint16Array): Uint8Array {
  const out = new Uint8Array(count * 3);
  for (let i = 0; i < count; i++) {
    const tint = tintOf(views[i] ?? 0);
    for (let k = 0; k < 3; k++) {
      const own = colours ? (colours[i * 3 + k] ?? 0) : (tint[k] ?? 0);
      out[i * 3 + k] = Math.round(own * (1 - TINT_SHARE) + (tint[k] ?? 0) * TINT_SHARE);
    }
  }
  return out;
}

/** The squares (two triangles, four corners a point) of the points to draw: those of the cameras not in `hidden`,
 *  and, with `inside`, not outside the plan. Points with no tags are all drawn. */
export function pointIndices(
  count: number,
  views: Uint16Array | null | undefined,
  outside: Uint8Array | null | undefined,
  hidden: ReadonlySet<number>,
  inside: boolean,
): Uint32Array {
  const drawn = (i: number) => !(views && hidden.has(views[i] ?? -1)) && !(inside && outside?.[i]);
  let n = 0;
  for (let i = 0; i < count; i++) if (drawn(i)) n++;
  const out = new Uint32Array(n * 6);
  for (let i = 0, at = 0; i < count; i++)
    if (drawn(i)) {
      out.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], at);
      at += 6;
    }
  return out;
}

/** How many of each camera's points lie outside the plan, by the camera's place in the scene. */
export function outsideCounts(views: Uint16Array | null | undefined, outside: Uint8Array | null | undefined): number[] {
  const out: number[] = [];
  if (!views || !outside) return out;
  for (let i = 0; i < views.length; i++) if (outside[i]) out[views[i] ?? 0] = (out[views[i] ?? 0] ?? 0) + 1;
  return out;
}
