// What Walk mode stands on and is stopped by: a mesh of the place's floor and its plan's walls, handed to
// SuperSplat's viewer as its collision. A splat has no surfaces of its own, and the viewer's walk camera keeps
// its eye above whatever its collision mesh has under it and slides along what the mesh puts in its way.
//
// The viewer takes collision as a URL and reads a mesh only from one whose path ends in `.glb`, so the mesh is
// written as a GLB and given as a `data:` URL whose last bytes are ".glb": a few unused bytes at the end of the
// GLB's buffer, which a GLB reader never looks at.
import type { V3 } from './geometry';

/** How high a wall stands above the floor (metres): over the walk camera's head. */
export const WALL_HEIGHT = 2.4;
/** How far the floor reaches past everything drawn (metres), so the walk never steps off its edge. */
export const FLOOR_MARGIN = 3;

/** Triangles: three numbers a corner, three corners a triangle. */
export interface Triangles {
  positions: Float32Array;
  indices: Uint32Array;
}

/** A floor (its four corners, in order round it) and walls (each a line on the floor, from and to) as triangles,
 *  the walls standing `height` along `up`. A wall is a sheet with both faces, so it stops a walk from either
 *  side whichever way the collision reads a triangle's facing. */
export function collisionMesh(
  floor: readonly V3[],
  walls: readonly [V3, V3][],
  up: V3,
  height = WALL_HEIGHT,
): Triangles {
  const positions: number[] = [];
  const indices: number[] = [];
  const quad = (a: V3, b: V3, c: V3, d: V3, bothFaces: boolean) => {
    const at = positions.length / 3;
    positions.push(...a, ...b, ...c, ...d);
    indices.push(at, at + 1, at + 2, at, at + 2, at + 3);
    if (bothFaces) indices.push(at, at + 2, at + 1, at, at + 3, at + 2);
  };
  if (floor.length === 4) quad(floor[0] as V3, floor[1] as V3, floor[2] as V3, floor[3] as V3, true);
  const lift = (p: V3): V3 => [p[0] + up[0] * height, p[1] + up[1] * height, p[2] + up[2] * height];
  for (const [from, to] of walls) quad(from, to, lift(to), lift(from), true);
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

const TAIL = '.glb';

/** Triangles as a binary glTF: one mesh, positions and 32-bit indices, and four unused bytes spelling ".glb" at
 *  the end of its buffer (see the top of this file). */
export function glb(mesh: Triangles): Uint8Array {
  const { positions, indices } = mesh;
  const min = [0, 1, 2].map(k => positions.reduce((m, v, i) => (i % 3 === k ? Math.min(m, v) : m), Infinity));
  const max = [0, 1, 2].map(k => positions.reduce((m, v, i) => (i % 3 === k ? Math.max(m, v) : m), -Infinity));
  const bin = positions.byteLength + indices.byteLength + TAIL.length; // both arrays are whole multiples of 4 bytes
  const json = JSON.stringify({
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, mode: 4 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3', min, max },
      { bufferView: 1, componentType: 5125, count: indices.length, type: 'SCALAR' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 },
      { buffer: 0, byteOffset: positions.byteLength, byteLength: indices.byteLength, target: 34963 },
    ],
    buffers: [{ byteLength: bin }],
  });
  const text = new TextEncoder().encode(json.padEnd(Math.ceil(json.length / 4) * 4, ' '));
  const out = new Uint8Array(12 + 8 + text.length + 8 + bin);
  const view = new DataView(out.buffer);
  view.setUint32(0, 0x46546c67, true); // "glTF"
  view.setUint32(4, 2, true);
  view.setUint32(8, out.length, true);
  view.setUint32(12, text.length, true);
  view.setUint32(16, 0x4e4f534a, true); // "JSON"
  out.set(text, 20);
  const at = 20 + text.length;
  view.setUint32(at, bin, true);
  view.setUint32(at + 4, 0x004e4942, true); // "BIN"
  out.set(new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength), at + 8);
  out.set(new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength), at + 8 + positions.byteLength);
  out.set(new TextEncoder().encode(TAIL), out.length - TAIL.length);
  return out;
}

/** A GLB's bytes as a `data:` URL that ends in ".glb": every byte percent-encoded but that tail. */
export function glbUrl(bytes: Uint8Array): string {
  let body = '';
  for (const b of bytes.subarray(0, bytes.length - TAIL.length)) body += `%${b.toString(16).padStart(2, '0')}`;
  return `data:model/gltf-binary,${body}${TAIL}`;
}
