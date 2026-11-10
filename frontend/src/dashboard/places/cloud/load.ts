// Reading what the 3D view draws, by format, so one viewer grows into whatever the maps become. A place map is
// a point cloud in PLY (spatial/ply.py: x, y, z, red, green, blue, and its own counts, which this skips). A
// revision can also have a 3D Gaussian splat (spatial/splats.py), served as SOG (spatial/sog.py), which
// PlayCanvas reads itself (scene.ts). Nothing here touches the GPU, so it's tested without one.

/** The formats the view can draw now. */
export const FORMATS = ['ply'] as const;
export type CloudFormat = (typeof FORMATS)[number];

export const canDraw = (format: string | null | undefined): format is CloudFormat =>
  (FORMATS as readonly string[]).includes(format ?? '');

/** A map's points: x, y, z each, and red, green, blue bytes each when the file has colours. */
export interface Cloud {
  count: number;
  positions: Float32Array;
  colours: Uint8Array | null;
}

/** Bytes a PLY scalar takes, by its type's name (both spellings the format allows). */
const SIZES: Record<string, number> = {
  char: 1,
  uchar: 1,
  int8: 1,
  uint8: 1,
  short: 2,
  ushort: 2,
  int16: 2,
  uint16: 2,
  int: 4,
  uint: 4,
  int32: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
};

/** A binary little-endian PLY's vertices as points: float x, y, z, and uchar red, green, blue when all three are
 *  there; every other property is stepped over by its size. */
function parsePly(bytes: ArrayBuffer): Cloud {
  const head = new TextDecoder('ascii').decode(new Uint8Array(bytes, 0, Math.min(bytes.byteLength, 4096)));
  const end = head.indexOf('end_header\n');
  if (!head.startsWith('ply') || end < 0) throw new Error("The map isn't a PLY file.");
  if (!head.includes('format binary_little_endian')) throw new Error('The map is a PLY this view cannot read.');
  let count = 0;
  let stride = 0;
  let inVertex = false;
  const at: Record<string, { offset: number; type: string }> = {};
  for (const line of head.slice(0, end).split('\n')) {
    const words = line.trim().split(/\s+/);
    if (words[0] === 'element') {
      inVertex = words[1] === 'vertex';
      if (inVertex) count = Number(words[2]);
    } else if (words[0] === 'property' && inVertex) {
      const [, type = '', name = ''] = words;
      const size = SIZES[type];
      if (!size) throw new Error(`The map has a property this view cannot read (${type}).`);
      at[name] = { offset: stride, type };
      stride += size;
    }
  }
  const floats = ['x', 'y', 'z'].every(k => at[k]?.type === 'float' || at[k]?.type === 'float32');
  const start = end + 'end_header\n'.length;
  if (!floats || start + count * stride > bytes.byteLength) throw new Error('The map is a PLY this view cannot read.');
  const view = new DataView(bytes, start);
  const positions = new Float32Array(count * 3);
  const rgb = ['red', 'green', 'blue'].every(k => SIZES[at[k]?.type ?? ''] === 1);
  const colours = rgb ? new Uint8Array(count * 3) : null;
  const [x, y, z] = [at.x?.offset ?? 0, at.y?.offset ?? 0, at.z?.offset ?? 0];
  const [r, g, b] = [at.red?.offset ?? 0, at.green?.offset ?? 0, at.blue?.offset ?? 0];
  for (let i = 0; i < count; i++) {
    const o = i * stride;
    positions[i * 3] = view.getFloat32(o + x, true);
    positions[i * 3 + 1] = view.getFloat32(o + y, true);
    positions[i * 3 + 2] = view.getFloat32(o + z, true);
    if (colours) {
      colours[i * 3] = view.getUint8(o + r);
      colours[i * 3 + 1] = view.getUint8(o + g);
      colours[i * 3 + 2] = view.getUint8(o + b);
    }
  }
  return { count, positions, colours };
}

/** A map file's bytes as points: positions, and colours when it has them. */
export function parseCloud(bytes: ArrayBuffer, format: CloudFormat): Cloud {
  switch (format) {
    case 'ply':
      return parsePly(bytes);
  }
}

/** The splat formats the view can draw: what PlayCanvas is given (scene.ts). */
export const SPLAT_FORMATS = ['sog'] as const;

export const canDrawSplat = (format: string | null | undefined): boolean =>
  (SPLAT_FORMATS as readonly string[]).includes(format ?? '');

/** The camera a splat opens from (#44): one whose frames it trained on, since anywhere else it's only smear. That is
 *  the first registered camera; null with none. */
export const openingCamera = (names: readonly string[]): string | null => names[0] ?? null;
