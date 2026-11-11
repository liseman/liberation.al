// What the 3D view says about a place's splat, in plain words, from the state GET /places/scene names
// (`splat_state`): there is none yet, or the one shown was made for an earlier map, or the newest one doesn't line up
// with this map and isn't shown. What it says while there is nothing to draw (`learningNote`): that no camera has a
// cloud yet, and what would make one. What it says about one camera's own cloud (`viewNote`, `viewWhy`): how it was
// sized, when it was made, and why a camera has none or no newer one. And what it says about the owner's plan: why it
// isn't drawn on a map that isn't lined up with it. Nothing here draws, so it's tested as text.
import { dayWords, hm, isoDay } from '../../lib/format';
import type { CameraView, Scene, ViewWhy } from '../api';

const when = (at: number) =>
  new Date(at * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

/** What to say under the view about the place's splat, or null when the splat shown is this map's own. */
export function splatNote(scene: Scene): string | null {
  if (scene.view) return null; // a camera's own cloud is points alone: no splat goes with it
  switch (scene.splat_state) {
    case 'current':
      return null;
    case 'earlier':
      return `This splat was made for an earlier map of this place${scene.splat_at ? ` (${when(scene.splat_at)})` : ''}; it lines up with the current one. A splat of the current map isn't made yet.`;
    case 'stale':
      return `The splat from ${scene.splat_stale ? when(scene.splat_stale.at) : 'before'} was made for an earlier map that doesn't line up with this one, so it isn't shown. A new one isn't made yet.`;
    case 'none':
      return `No splat of this place yet.`;
    default:
      // a scene that doesn't name the state: say only what's certain
      return scene.splat ? null : 'No splat of this place yet.';
  }
}

/** Where a line names the Live page, for the view to put a link there (`CloudView`'s `Words`). */
export const LIVE = '{Live}';

const TURN_ON = `Turn on Learn surroundings in ${LIVE}, and each camera chosen there gets a cloud of what it sees.`;

/** What to say while there is nothing to draw and no camera to pick: that no camera has a point cloud yet, and what
 *  would make one. Null with something drawn, or when the scene doesn't say. */
export function learningNote(scene: Scene): string | null {
  const learning = scene.learning;
  if (!learning || scene.revision) return null;
  const none = 'No camera has a point cloud of this place yet.';
  if (!learning.on) return `${none} ${TURN_ON}`;
  if (!learning.cameras)
    return `${none} No camera is chosen to learn from. Choose one beside Learn surroundings in ${LIVE}.`;
  return `${none} Each chosen camera gets one once it has had about 20 seconds with nobody in view.`;
}

const WHY: Record<Exclude<ViewWhy, 'off'>, string> = {
  not_chosen: `This camera isn't chosen to learn from. Choose it beside Learn surroundings in ${LIVE}.`,
  waiting: 'Waiting for about 20 seconds with nobody in view.',
  people: 'Someone was in view each time it looked. It needs about 20 seconds with nobody in view.',
  dark: 'The view is too dark, or the camera is in night mode.',
  no_runtime: "The model that makes point clouds isn't set up on this machine (spatial.pointmap_runtime in pack.toml).",
  gpu: "There wasn't enough free GPU memory. It tries again in a couple of minutes.",
  failed: 'The last try failed. It tries again in a couple of minutes.',
  unsure: "The model wasn't sure of enough of the picture (glare, glass, or a very plain view).",
  flat: 'The picture had no depth to speak of.',
  lens: "This camera's lens calibration in the device list can't be used.",
};

/** What keeps a camera from a cloud, or from a newer one, in plain words; null when nothing does. With learning
 *  off it says what that means here: a cloud that is showing won't update, and a camera without one gets none. */
export function viewWhy(view: Pick<CameraView, 'why' | 'cloud'> | undefined): string | null {
  if (!view?.why) return null;
  if (view.why === 'off')
    return view.cloud ? `Learn surroundings is off, so this picture won't update. Turn it on in ${LIVE}.` : TURN_ON;
  return WHY[view.why] ?? null;
}

/** When a cloud was made, as people say it: "today 01:40", "yesterday 18:40", "Thu 5 Nov 13:31". */
const made = (at: number, now: Date) => `${dayWords(isoDay(new Date(at * 1000)), now)} ${hm(at)}`;

/** What to say about the camera's cloud that is drawn: whose view it is, first, since it is that camera's part of the
 *  place and not the place; then how many points, when it was made, and how it was sized. A size that is only
 *  approximate says what would make it true: the camera's mounting height, or, when no floor is in its view, that a
 *  height wouldn't help. `name` is the camera as the page names it. Null for a scene that isn't one camera's cloud. */
export function viewNote(scene: Scene, name?: string, now = new Date()): string | null {
  const view = scene.view;
  if (!view || !scene.revision) return null;
  const facts = [view.points ? `${view.points.toLocaleString()} points` : null, view.at ? made(view.at, now) : null];
  let size: string;
  if (view.size === 'height') size = "Sized from the camera's mounting height.";
  else if (view.floor_found === false)
    size = "Approximate size: no floor is in this camera's view to measure its height from.";
  else
    size =
      'Approximate size. Give this camera its mounting height (mount.height_m on its entry in config/devices.toml) for a true one.';
  const counted = facts.filter(Boolean).join(', ');
  return `What ${name ?? view.camera} sees on its own. ${counted ? `${counted}. ` : ''}${size}`;
}

/** What to say about the owner's plan on this map: nothing while it's drawn where it belongs (or the place has no
 *  plan of the owner's), and that it isn't drawn on a map it isn't lined up with. The plan itself is never replaced:
 *  the Plan view shows it as drawn. */
export function planNote(scene: Scene): string | null {
  if (scene.plan_source !== 'owner' || !scene.revision) return null;
  if (!scene.plan_aligned)
    return "Your plan isn't drawn on this map: the map isn't lined up with it. The Plan view shows your plan as you drew it.";
  return null;
}

/** A splat's loading, as text: how much of it has come, then that it's being drawn. */
export function loadingText(received: number, total: number, drawing: boolean): string {
  if (drawing) return 'Drawing the splat…';
  const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;
  if (!total) return received ? `Loading the splat: ${mb(received)}…` : 'Loading the splat…';
  return `Loading the splat: ${Math.min(100, Math.floor((received / total) * 100))}% of ${mb(total)}`;
}
