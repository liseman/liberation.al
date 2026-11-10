// Loading what the 3D view draws, by format, so one viewer grows into whatever the maps become. A place map is
// a point cloud in PLY (spatial/ply.py: x, y, z, red, green, blue, and its own counts, which the loader skips). A
// revision of fixed cameras can also have a 3D Gaussian splat (spatial/splats.py), served as SPZ; Spark (MIT)
// draws it as a SplatMesh, a three.js object in the same scene, so there's never a second viewer.
import type { BufferGeometry, WebGLRenderer } from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

/** The formats the view can draw now. */
export const FORMATS = ['ply'] as const;
export type CloudFormat = (typeof FORMATS)[number];

export const canDraw = (format: string | null | undefined): format is CloudFormat =>
  (FORMATS as readonly string[]).includes(format ?? '');

/** A map file's bytes as points: positions, and colours when it has them (0 to 1). */
export function parseCloud(bytes: ArrayBuffer, format: CloudFormat): BufferGeometry {
  switch (format) {
    case 'ply':
      return new PLYLoader().parse(bytes);
  }
}

/** The splat formats the view can draw. */
export const SPLAT_FORMATS = ['spz'] as const;

export const canDrawSplat = (format: string | null | undefined): boolean =>
  (SPLAT_FORMATS as readonly string[]).includes(format ?? '');

/** A splat in the scene: Spark's renderer (it sorts and draws every splat mesh in the scene) and the mesh. */
export interface LoadedSplat {
  spark: import('@sparkjsdev/spark').SparkRenderer;
  mesh: import('@sparkjsdev/spark').SplatMesh;
}

/** An SPZ file's bytes as a splat mesh, with the Spark renderer that draws it. Spark is imported here, on the first
 *  splat, so a view that never shows one never loads it. `onDirty` is Spark asking for a frame (its sort or decode
 *  finished); the view draws on demand, so it schedules one.
 *
 *  An SPZ is stored in SPZ's right-up-back frame: spatial/splats.py negates y and z of the map frame, as Niantic's
 *  converter does from a PLY. Spark reads it as stored, so the mesh is turned 180 degrees about x, which is exactly
 *  that negation undone (positions, rotations, and view-dependent colour alike), putting it back on the map. */
export async function loadSplat(
  bytes: ArrayBuffer,
  renderer: WebGLRenderer,
  onDirty: () => void,
): Promise<LoadedSplat> {
  const { SparkRenderer, SplatFileType, SplatMesh } = await import('@sparkjsdev/spark');
  const spark = new SparkRenderer({ renderer, onDirty });
  const mesh = new SplatMesh({ fileBytes: bytes, fileType: SplatFileType.SPZ });
  mesh.quaternion.set(1, 0, 0, 0);
  await mesh.initialized;
  return { spark, mesh };
}
