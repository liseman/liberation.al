// What the 3D view says about a place's splat, in plain words, from the state GET /places/scene names
// (`splat_state`): there is none yet, or the one shown was made for an earlier map, or the newest one doesn't line up
// with this map and isn't shown. What it says while there is nothing to draw (`learningNote`): that no camera has a
// cloud yet, and what would make one. What it says about one camera's own cloud (`viewNote`, `viewWhy`): how it was
// sized, when it was made, and why a camera has none or no newer one. And what it says about the owner's plan: why it
// isn't drawn on a map that isn't lined up with it. Nothing here draws, so it's tested as text.
import { dayWords, hm, isoDay } from '../../lib/format';
import type { CameraView, Scene, ScenePair, ScenePlaced, ViewWhy } from '../api';

const when = (at: number) =>
  new Date(at * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

/** What to say under the view about the place's splat, or null when the splat shown is this map's own. */
export function splatNote(scene: Scene): string | null {
  if (scene.view || scene.combined) return null; // cameras' clouds are points alone: no splat goes with them
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

/** How a cloud's size is said: it is the model's own estimate of distance, not a measurement. */
export const APPROXIMATE = "Approximate size: the model's own estimate of distances, which can be a tenth off or more.";

/** What to say about the camera's cloud that is drawn: whose view it is, first, since it is that camera's part of the
 *  place and not the place; then how many points, when it was made, and that its size is approximate. `name` is the
 *  camera as the page names it. Null for a scene that isn't one camera's cloud. */
export function viewNote(scene: Scene, name?: string, now = new Date()): string | null {
  const view = scene.view;
  if (!view || !scene.revision) return null;
  const facts = [view.points ? `${view.points.toLocaleString()} points` : null, view.at ? made(view.at, now) : null];
  const counted = facts.filter(Boolean).join(', ');
  return `What ${name ?? view.camera} sees on its own. ${counted ? `${counted}. ` : ''}${APPROXIMATE}`;
}

// ---- the scene of all the cameras' clouds (#445)

const metres = (m: number) => (m < 0.995 ? `${Math.round(m * 100)} cm` : `${m.toFixed(1)} m`);
const degrees = (d: number) => `${Math.round(d)}°`;
const listed = (names: readonly string[]) =>
  names.length < 2 ? (names[0] ?? '') : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;

/** How one camera is placed in the scene, or why it isn't in it, in plain words. `name` names a camera for the page. */
export function placedLine(camera: ScenePlaced, name: (camera: string) => string = c => c): string {
  switch (camera.how) {
    case 'spot':
      return 'Placed from its spot on your plan.';
    case 'adjusted':
      return `Placed from its spot, adjusted against ${listed(camera.against.map(name))} by what their pictures share: ${metres(camera.shift_m)} and ${degrees(camera.turn_deg)} from the spot.`;
    case 'moved':
      return `This camera was moved: by its pictures, ${metres(camera.shift_m)} and ${degrees(camera.turn_deg)} from its spot. Its cloud followed.`;
    case 'moved_against':
      return `This camera was moved. It is placed from what its picture shares with ${listed(camera.against.map(name))}'s.`;
    case 'no_spot':
      return 'Not in the scene: it has no spot on your plan. Give it one in the plan editor.';
    case 'no_facing':
      return 'Not in the scene: its spot has no facing. Set which way it looks in the plan editor.';
    case 'moved_unplaced':
      return "This camera was moved and isn't placed: its picture matches neither its old one nor another camera's. Set its spot again in the plan editor.";
  }
}

/** What two cameras' pictures said of each other, for the pairs worth a line; null for the rest (different space, a
 *  pair one of which has moved, a measurement from another spot). */
export function pairLine(pair: ScenePair, name: (camera: string) => string = c => c, now = new Date()): string | null {
  const both = `${name(pair.a)} and ${name(pair.b)}`;
  switch (pair.state) {
    case 'tightened': {
      const when = pair.measured ? ` (measured ${made(pair.measured, now)})` : '';
      const kept = pair.latest
        ? " A later look didn't match well enough (pictures at night often don't), so that measurement is kept."
        : '';
      return `${both} see the same space and are adjusted to what their pictures share${when}: the spots were ${metres(pair.off_m)} and ${degrees(pair.off_deg)} from it.${kept}`;
    }
    case 'unsure':
      return `${both} see some of the same space, but their pictures don't agree well enough to adjust them. They are placed from their spots only.`;
    case 'disagrees':
      return `${both}: their pictures put them ${metres(pair.off_m)} and ${degrees(pair.off_deg)} from where their spots do. One of the two spots is off. Nothing was moved.`;
    default:
      return null;
  }
}

/** What the scene is, and isn't, said once under it. */
export const SCENE_LIMITS =
  "Each camera's cloud is a relief of what it alone sees, standing where its camera is: not a whole room. Where two overlap they meet with a seam, and cameras that share no view are only as right as their spots.";

/** What to say when the scene has nothing to draw. */
export function sceneEmpty(cameras: readonly ScenePlaced[]): string {
  if (!cameras.length) return 'No camera has a point cloud to place yet.';
  return 'No camera can be placed yet: each needs a spot with a facing on your plan.';
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
