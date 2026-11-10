// What the 3D view says about a place's splat, in plain words, from the state GET /places/scene names
// (`splat_state`, `splat_wait`): there is none yet and why, the one shown was made for an earlier map, or the newest
// one doesn't line up with this map and isn't shown. Nothing here draws, so it's tested as text.
import type { Scene, SplatWait } from '../api';

const when = (at: number) =>
  new Date(at * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

/** "at 01:00", or nothing when the pass has no time of day. */
const passAt = (wait: SplatWait) => (wait.pass_at ? ` at ${wait.pass_at}` : '');

/** Why the map has no splat of its own, and when one is due. */
function waiting(wait: SplatWait): string {
  const last = wait.tried ? ` (${when(wait.tried)})` : '';
  switch (wait.reason) {
    case 'few_frames':
      return `It's waiting for daylight: the nightly pass makes it from the fixed cameras' lit daytime frames, and had too few when it last ran${last}. It tries again${passAt(wait)}.`;
    case 'no_cameras':
      return 'This map has no fixed cameras for the nightly pass to make one from; a walkthrough brings its own.';
    case 'no_runtime':
      return "The splat trainer isn't installed on this machine, so none is made.";
    case 'off':
      return 'Splats are turned off in the settings (spatial.splats_enabled), so none is made.';
    case 'failed':
      return `Making it failed when the nightly pass last ran${last}. It tries again${passAt(wait)}.`;
    case 'pass':
      return `The nightly pass makes it from the fixed cameras' daytime frames; it runs${passAt(wait) || ' each night'}.`;
  }
}

/** What to say under the view about the place's splat, or null when the splat shown is this map's own. */
export function splatNote(scene: Scene): string | null {
  const wait = scene.splat_wait ? ` ${waiting(scene.splat_wait)}` : '';
  switch (scene.splat_state) {
    case 'current':
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

/** A splat's loading, as text: how much of it has come, then that it's being drawn. */
export function loadingText(received: number, total: number, drawing: boolean): string {
  if (drawing) return 'Drawing the splat…';
  const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;
  if (!total) return received ? `Loading the splat: ${mb(received)}…` : 'Loading the splat…';
  return `Loading the splat: ${Math.min(100, Math.floor((received / total) * 100))}% of ${mb(total)}`;
}
