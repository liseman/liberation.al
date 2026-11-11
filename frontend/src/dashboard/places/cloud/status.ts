// What the 3D view says about a place's splat, in plain words, from the state GET /places/scene names
// (`splat_state`): there is none yet, or the one shown was made for an earlier map, or the newest one doesn't line up
// with this map and isn't shown. And what it says about the map (`learning`): with none, that the 3D map is being
// rebuilt with a new method and whether pictures are being collected for it. And what it says about the owner's plan:
// why it isn't drawn on a map that isn't lined up with it. Nothing here draws, so it's tested as text.
import type { Scene } from '../api';

const when = (at: number) =>
  new Date(at * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

/** What to say under the view about the place's splat, or null when the splat shown is this map's own. */
export function splatNote(scene: Scene): string | null {
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

const LIVE = 'Turn on Learn surroundings on the Live tab';

/** What to say about the map: for a place with none, that the 3D map is being rebuilt with a new method and whether
 *  pictures are being collected for it. Null with a map, or when the scene doesn't say. */
export function learningNote(scene: Scene): string | null {
  const learning = scene.learning;
  if (!learning || scene.revision) return null;
  const none = 'No 3D map of this place yet: the 3D map is being rebuilt with a new method.';
  if (!learning.on) return `${none} ${LIVE} to collect pictures for it.`;
  if (!learning.cameras) return `${none} No camera is chosen to collect pictures from. Choose one on the Live tab.`;
  return `${none} Pictures are being collected for it.`;
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
