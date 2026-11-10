// What the 3D view says about a place's splat, in plain words, from the state GET /places/scene names
// (`splat_state`, `splat_wait`): there is none yet and why, the one shown is still improving or was made for an
// earlier map, or the newest one doesn't line up with this map and isn't shown. And what it says about learning
// (`learning`, #259): whether the map is being learned from the cameras, how many views it was built from, and
// when it last improved. And what it says about the owner's plan (`plan_lineup`): why it isn't drawn on a map that
// isn't lined up with it yet. Nothing here draws, so it's tested as text.
import type { Learning, Scene, SplatWait } from '../api';

const when = (at: number) =>
  new Date(at * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

/** " It tries again at 01:00.", or nothing while no pass is scheduled. */
const again = (wait: SplatWait) => (wait.pass_at ? ` It tries again at ${wait.pass_at}.` : '');

/** Why the map has no splat of its own, and when one is due. */
function waiting(wait: SplatWait): string {
  const last = wait.tried ? ` (${when(wait.tried)})` : '';
  switch (wait.reason) {
    case 'few_frames':
      return `It's waiting for daylight: a splat is made from the fixed cameras' lit daytime frames, and there were too few when it was last tried${last}.${again(wait)}`;
    case 'no_cameras':
      return 'This map has no cameras registered yet to make one from.';
    case 'no_runtime':
      return "The splat trainer isn't installed on this machine, so none is made.";
    case 'failed':
      return `Making it failed when it was last tried${last}.${again(wait)}`;
    case 'learned':
      return "A learned map's splat is written right after the map, so it appears here once it's trained.";
    case 'pass':
      return wait.pass_at
        ? `It's made from the fixed cameras' daytime frames, by a pass that runs at ${wait.pass_at}.`
        : 'None has been made for this map, and nothing makes splats on a schedule yet.';
  }
}

/** What to say under the view about the place's splat, or null when the splat shown is this map's own. */
export function splatNote(scene: Scene): string | null {
  const wait = scene.splat_wait ? ` ${waiting(scene.splat_wait)}` : '';
  switch (scene.splat_state) {
    case 'current':
    case 'improving': // the line about learning says it's still improving
      return null;
    case 'earlier':
      return `This splat was made for an earlier map of this place${scene.splat_at ? ` (${when(scene.splat_at)})` : ''}; it lines up with the current one. A splat of the current map isn't made yet.${wait}`;
    case 'stale':
      return `The splat from ${scene.splat_stale ? when(scene.splat_stale.at) : 'before'} was made for an earlier map that doesn't line up with this one, so it isn't shown. A new one isn't made yet.${wait}`;
    case 'none':
      return `No splat of this place yet.${wait}`;
    default:
      // a scene that doesn't name the state: say only what's certain
      return scene.splat ? null : 'No splat of this place yet.';
  }
}

const LIVE = 'Turn on Learn surroundings on the Live tab';

/** "Built from 42 views, last improved Nov 10, 3:12 PM.", as much of it as is known. */
function builtFrom(learning: Learning): string {
  const views = learning.views;
  const parts = [
    views == null ? null : `built from ${views} ${views === 1 ? 'view' : 'views'}`,
    learning.improved == null ? null : `last improved ${when(learning.improved)}`,
  ].filter(Boolean);
  const said = parts.join(', ');
  return said ? ` ${said.charAt(0).toUpperCase()}${said.slice(1)}.` : '';
}

/** What to say about learning: for a place with no map, that it has none and what would start one; for a place
 *  with one, whether it's still improving, how many views it was built from, and when it last improved. Null when
 *  the scene doesn't say (a server from before it did). */
export function learningNote(scene: Scene): string | null {
  const learning = scene.learning;
  if (!learning) return null;
  if (!scene.revision) {
    const none = 'No 3D map of this place yet.';
    if (!learning.on) return `${none} Learning is off. ${LIVE}, and a map is built from what the cameras see.`;
    if (!learning.cameras)
      return `${none} Learning is on, but no camera is chosen to learn from. Choose one on the Live tab.`;
    return learning.improving
      ? `${none} Learning is on, and the cameras' first views are being put together. A map appears here once there's one to show.`
      : `${none} Learning is on. A map appears here once the cameras have seen enough to start one.`;
  }
  const built = builtFrom(learning);
  if (!learning.on) return `Learning is off, so this map stays as it is.${built} ${LIVE} to keep improving it.`;
  if (learning.improving)
    return `Learning is on, and this map is still improving: new views are being added to it.${built}`;
  return `Learning is on. This map improves when the cameras see something it doesn't have yet.${built}`;
}

/** Why the owner's plan isn't drawn on a map that isn't lined up with it yet. */
function unaligned(reason: NonNullable<Scene['plan_lineup']>['reason']): string {
  switch (reason) {
    case 'no_floor':
      return "the map has no floor yet to lay it on. It's lined up once the map has one.";
    case 'too_little':
      return "the map covers too little of your plan so far to tell where it goes. It's lined up once the cameras have seen more of the place.";
    case 'ambiguous':
      return "what the map has so far fits your plan more than one way. It's lined up once the cameras have seen enough to tell.";
    case 'poor_fit':
      return "the map doesn't sit on your plan's walls well enough yet. It's tried again as the map grows.";
    case 'plan_unreadable':
      return 'your plan has no room to line the map up with.';
    default:
      return "the map hasn't been lined up with it yet. That's tried each time the map grows.";
  }
}

/** What to say about the owner's plan on this map: nothing while it's drawn where it belongs (or the place has no
 *  plan of the owner's); why it isn't drawn while the map isn't lined up with it; and that the map has moved off it
 *  when a line-up no longer holds. The plan itself is never replaced: the Plan view shows it as drawn. */
export function planNote(scene: Scene): string | null {
  if (scene.plan_source !== 'owner' || !scene.revision) return null;
  if (!scene.plan_aligned)
    return `Your plan isn't drawn on this map yet: ${unaligned(scene.plan_lineup?.reason ?? null)} The Plan view shows your plan as you drew it.`;
  if (scene.plan_lineup?.reason === 'drifted')
    return 'Your plan is drawn where the map was lined up with it, but the map has moved off it since, so the outline may be off.';
  return null;
}

/** A splat's loading, as text: how much of it has come, then that it's being drawn. */
export function loadingText(received: number, total: number, drawing: boolean): string {
  if (drawing) return 'Drawing the splat…';
  const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;
  if (!total) return received ? `Loading the splat: ${mb(received)}…` : 'Loading the splat…';
  return `Loading the splat: ${Math.min(100, Math.floor((received / total) * 100))}% of ${mb(total)}`;
}
