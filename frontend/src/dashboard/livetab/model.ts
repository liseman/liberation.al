// The Live tab's pure logic: reading faces. Turns, and searching them, are transcript/model.ts's.
// ---- faces in the scene ----

export interface SceneFace {
  name?: string;
  bbox?: [number, number, number, number]; // x1, y1, x2, y2 in the caption tile's pixels
  bearing_deg?: number | null;
  [field: string]: unknown;
}

export const UNKNOWN_FACE = /^Face (\d+)$/; // vision groups unnamed faces as "Face N" (vision/faces.py group_unknown)
export const MIN_FACE_PX = 24; // smaller faces (preview pixels) are usually photos on a wall, not someone in the room

/** Where a face is, as a person would say it: bearing 0 is straight ahead, clockwise. */
export function where(deg: number | null | undefined): string {
  if (deg == null) return '';
  const h = Math.round((((deg % 360) + 360) % 360) / 30) % 12;
  return h === 0 ? 'straight ahead' : `at ${h} o'clock`;
}

/** The known names among faces, once each. */
export const knownIn = (faces: SceneFace[] | undefined): string[] => [
  ...new Set(
    (faces ?? []).map(f => f.name).filter((n): n is string => !!n && n !== 'unknown' && !UNKNOWN_FACE.test(n)),
  ),
];

/** The unknown face groups worth asking about: big enough to be someone in the room, once per group. */
export function unknownFaces<F extends SceneFace>(faces: F[]): F[] {
  const seen = new Set<string>();
  return faces.filter(f => {
    const [x1, y1, x2, y2] = f.bbox ?? [0, 0, 0, 0];
    if (!f.name || !UNKNOWN_FACE.test(f.name) || x2 - x1 < MIN_FACE_PX || y2 - y1 < MIN_FACE_PX || seen.has(f.name))
      return false;
    seen.add(f.name);
    return true;
  });
}

/** The face group's number from its "Face N" name, or null. */
export function faceGroup(name: string | undefined): number | null {
  const m = UNKNOWN_FACE.exec(name ?? '');
  return m?.[1] ? Number(m[1]) : null;
}
