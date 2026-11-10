// Each camera's lane label. The cameras are angles of one rig, named like "front camera" and "rear camera" (vision.camera_names,
// which the page shell carries as views); one with no name goes by its id.
import { capitalize } from '../lib/format';
import { shell } from '../shell';

/** A camera's lane label: its name with a capital ("Front camera"), else its id. */
export function viewName(cam: string, views: Readonly<Record<string, string>> = shell().views): string {
  const value = Object.hasOwn(views, cam) ? views[cam] : undefined;
  const name = typeof value === 'string' ? value.trim() : '';
  return name ? capitalize(name) : cam;
}

// A generic numbered id, like cam0 or camera-2: one that says nothing about the view. An id someone chose in
// the device list (patio, desk-left) already reads as a name.
const GENERIC = /^cam(?:era)?[-_]?\d+$/i;

/** The cameras still going by a generic id like cam0 because they have no name, which pages should ask to name. */
export function unnamed(cams: readonly string[], views: Readonly<Record<string, string>> = shell().views): string[] {
  return cams.filter(cam => GENERIC.test(cam) && viewName(cam, views) === cam);
}
