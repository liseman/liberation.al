// What the 3D view says about a place's splat, in plain words, from the state GET /places/scene names
// (`splat_state`): there is none yet, or the one shown was made for an earlier map, or the newest one doesn't line up
// with this map and isn't shown. What it says while there is nothing to draw (`learningNote`): that no camera has a
// cloud yet, and what would make one. What it says about one camera's own cloud (`viewNote`, `viewWhy`): how it was
// sized, when it was made, and why a camera has none or no newer one. And what it says about the owner's plan: why it
// isn't drawn on a map that isn't lined up with it. Nothing here draws, so it's tested as text.
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

/** What to say while there is nothing to draw and no camera to pick: that no camera has a point cloud yet, and what
 *  would make one. Null with something drawn, or when the scene doesn't say. */
export function learningNote(scene: Scene): string | null {
  const learning = scene.learning;
  if (!learning || scene.revision) return null;
  const none = 'No camera has a point cloud of this place yet.';
  if (!learning.on)
    return `${none} Learn surroundings is off, so none is made. Turn it on on the Live tab, and each camera chosen there gets a cloud of what it sees.`;
  if (!learning.cameras)
    return `${none} No camera is chosen to learn from. Choose one beside Learn surroundings on the Live tab.`;
  return `${none} Each chosen camera gets one once it has had about 20 seconds with nobody in view.`;
}

const WHY: Record<ViewWhy, string> = {
  off: 'Learn surroundings is off, so nothing is made or refreshed. Turn it on on the Live tab.',
  not_chosen: "This camera isn't chosen to learn from. Choose it beside Learn surroundings on the Live tab.",
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

/** Why a camera has no cloud, or no newer one, in plain words; null when nothing stands in its way. */
export function viewWhy(view: Pick<CameraView, 'why'> | undefined): string | null {
  return view?.why ? (WHY[view.why] ?? null) : null;
}

/** What to say about the camera's cloud that is drawn: how many points, when it was made, and how it was sized. Null
 *  for a scene that isn't one camera's own cloud. */
export function viewNote(scene: Scene): string | null {
  const view = scene.view;
  if (!view || !scene.revision) return null;
  const points = view.points ? `${view.points.toLocaleString()} points` : 'Points';
  const made = view.at ? ` from ${when(view.at)}` : '';
  const size =
    view.size === 'height'
      ? "Sized from the camera's mounting height."
      : "Approximate size: the model's own metres, which can be a fifth off. A mounting height for this camera in the device list pins it.";
  return `${points}${made}, as this camera alone sees it. ${size}`;
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
