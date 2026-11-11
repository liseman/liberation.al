// Reading what the 3D view draws, by format, so one viewer grows into whatever the maps become. A place map is
// a point cloud in PLY (spatial/ply.py: x, y, z, red, green, blue, the view each point came from, and its own
// counts, which this skips). A
// revision can also have a 3D Gaussian splat (spatial/splats.py), served as SOG (spatial/sog.py), which
// PlayCanvas reads itself (scene.ts). Nothing here touches the GPU, so it's tested without one.

/** The formats the view can draw now. */
export const FORMATS = ['ply'] as const;
export type CloudFormat = (typeof FORMATS)[number];

export const canDraw = (format: string | null | undefined): format is CloudFormat =>
  (FORMATS as readonly string[]).includes(format ?? '');

/** A map's points: x, y, z each, red, green, blue bytes each when the file has colours, and the view each came
 *  from (an index into the scene's `sources.views`) when the file says. */
export interface Cloud {
  count: number;
  positions: Float32Array;
  colours: Uint8Array | null;
  views: Uint16Array | null;
}

/** The view of a point or Gaussian no view is known for (memory/places.py `UNTAGGED`): always drawn. */
export const UNTAGGED = 0xffff;

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

/** A binary little-endian PLY's vertices as points: float x, y, z, uchar red, green, blue when all three are
 *  there, and a ushort `view` when there is one; every other property is stepped over by its size. */
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
  const tagged = at.view?.type === 'ushort' || at.view?.type === 'uint16';
  const views = tagged ? new Uint16Array(count) : null;
  const v = at.view?.offset ?? 0;
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
    if (views) views[i] = view.getUint16(o + v, true);
  }
  return { count, positions, colours, views };
}

/** A map file's bytes as points: positions, and colours when it has them. */
export function parseCloud(bytes: ArrayBuffer, format: CloudFormat): Cloud {
  switch (format) {
    case 'ply':
      return parsePly(bytes);
  }
}

/** A splat's views file (GET /places/splat?format=views): each Gaussian's view, a little-endian uint16 each, in
 *  the SOG's own order. Null for bytes that can't be one (an odd length), and for one that isn't `count` long when
 *  that's known: views of another splat would hide the wrong Gaussians. */
export function parseViews(bytes: ArrayBuffer, count?: number | null): Uint16Array | null {
  if (bytes.byteLength % 2) return null;
  const n = bytes.byteLength / 2;
  if (count != null && count !== n) return null;
  const out = new Uint16Array(n);
  const view = new DataView(bytes);
  for (let i = 0; i < n; i++) out[i] = view.getUint16(i * 2, true);
  return out;
}

/** The splat formats the view can draw: what PlayCanvas is given (scene.ts). */
export const SPLAT_FORMATS = ['sog'] as const;

export const canDrawSplat = (format: string | null | undefined): boolean =>
  (SPLAT_FORMATS as readonly string[]).includes(format ?? '');

/** Why `bytes` can't be the SOG the viewer is given, or null when they look like one. A SOG is a zip, which starts
 *  with a local file header and ends with a directory record; checked here so a file that's cut short, or kept by
 *  the browser from an earlier encoding of the same splat, is said on the page instead of failing in the engine. */
export function splatFault(bytes: ArrayBuffer): string | null {
  const b = new Uint8Array(bytes);
  if (!b.length) return 'the file is empty';
  const zip = b[0] === 0x50 && b[1] === 0x4b && b[2] === 0x03 && b[3] === 0x04;
  if (!zip) return "the file isn't a splat this view can draw (it may be a copy the browser kept from an older format)";
  // the directory's end record is in the last 22 bytes, or a comment's length (at most 65,535) before them
  for (let i = b.length - 22, stop = Math.max(0, b.length - 22 - 0xffff); i >= stop; i--)
    if (b[i] === 0x50 && b[i + 1] === 0x4b && b[i + 2] === 0x05 && b[i + 3] === 0x06) return null;
  return 'the file is cut short';
}

/** A response's bytes, telling `progress` how many have come and of how many (the response's own length, else
 *  `expected`, else 0 when nobody knows). */
export async function readBytes(
  response: Response,
  expected: number | null | undefined,
  progress: (received: number, total: number) => void,
): Promise<ArrayBuffer> {
  const total = Number(response.headers.get('content-length')) || expected || 0;
  const reader = response.body?.getReader();
  if (!reader) {
    const whole = await response.arrayBuffer();
    progress(whole.byteLength, total || whole.byteLength);
    return whole;
  }
  const parts: Uint8Array[] = [];
  let received = 0;
  progress(0, total);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    received += value.byteLength;
    progress(received, Math.max(total, received));
  }
  const out = new Uint8Array(received);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.byteLength;
  }
  return out.buffer;
}

/** The camera a splat opens from (#44): one whose frames it trained on, since anywhere else it's only smear. That is
 *  the first registered camera; null with none. */
export const openingCamera = (names: readonly string[]): string | null => names[0] ?? null;
