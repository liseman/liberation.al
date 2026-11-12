// What the 3D view says about a place's splat, in plain words, from the state GET /places/scene names
// (`splat_state`): there is none yet, or the one shown was made for an earlier map, or the newest one doesn't line up
// with this map and isn't shown. What it says while there is nothing to draw (`learningNote`): that no camera has a
// cloud yet, and what would make one. What it says about one camera's own cloud (`viewNote`, `viewWhy`): how it was
// sized, when it was made, and why a camera has none or no newer one. And what it says about the owner's plan: why it
// isn't drawn on a map that isn't lined up with it. Nothing here draws, so it's tested as text.
import { ago, capitalize, dayWords, hm, isoDay, plural } from '../../lib/format';
import type { CameraView, CloudLevel, Scene, SceneFit, ScenePair, ScenePlaced, ViewWhy } from '../api';

const when = (at: number) =>
  new Date(at * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

/** What to say under the view about the place's splat, or null when the splat shown is this map's own. No producer
 *  writes a splat today (`memory/places.SHOWN_PRODUCERS` is empty), so no scene carries a `splat_state` and this is
 *  reached once one does. Its `default` branch takes a state it doesn't name ("improving", or none). */
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
  const level = levelLine(view.level, view.tilt_deg, view.roll_deg);
  const fit = view.fit ? fitLine(view.fit) : null;
  const sized = view.fit?.state === 'fitted' ? '' : APPROXIMATE;
  return `What ${name ?? view.camera} sees on its own. ${[counted ? `${counted}.` : '', level, fit, sized].filter(Boolean).join(' ')}`;
}

/** How a cloud was stood upright, and the camera's tilt and roll by that; null when the scene doesn't say. */
export function levelLine(level?: CloudLevel, tilt = 0, roll = 0): string | null {
  const rolled = Math.round(Math.abs(roll)) >= 1 ? ` and is rolled ${Math.round(Math.abs(roll))}°` : '';
  const camera = `the camera looks ${Math.round(tilt)}° down${rolled}`;
  switch (level) {
    case 'floor':
      return `Levelled from its own floor: ${camera}.`;
    case 'walls':
      return `No floor in view, so levelled from its walls: ${camera}.`;
    case 'none':
      return 'No floor or wall in view to level it by: it stands as if the camera were upright, so it may lean.';
    default:
      return null;
  }
}

// ---- the scene of all the cameras' clouds (#445)

const metres = (m: number) => (m < 0.995 ? `${Math.round(m * 100)} cm` : `${m.toFixed(1)} m`);
const degrees = (d: number) => `${Math.round(d)}°`;
const listed = (names: readonly string[]) =>
  names.length < 2 ? (names[0] ?? '') : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;

/** What placed a followed camera, as a sentence. */
const PLACED_BY: Record<string, string> = {
  tracking: 'Placed where it was tracked to as it moved.',
  pictures: "Placed by what its new picture shares with an earlier one or another camera's.",
  turned: 'Placed where it stood, by the walls it now sees.',
  walls: 'Placed by the walls it now sees, where they fit your plan.',
};

/** Why a changed view can't be placed, to follow "its view changed, and". */
const LOST: Record<string, string> = {
  no_walls: 'there is no wall in view to place it by',
  no_plan: 'your plan has no walls to place it by',
  no_match: "the walls it sees don't fit your plan anywhere well enough",
  ambiguous: 'the walls it sees fit your plan in more than one place',
};

/** How long ago a camera was placed: "a moment ago" for the first minute and a half, then `ago`'s words. */
const since = (at: number, now: number) => (now - at < 90 ? 'a moment ago' : ago(at, now));

/** How one camera is placed in the scene, or why it isn't in it, in plain words. `name` names a camera for the page;
 *  `now` is the time to count a followed camera's age from (epoch seconds). */
export function placedLine(
  camera: ScenePlaced,
  name: (camera: string) => string = c => c,
  now = Date.now() / 1000,
): string {
  switch (camera.how) {
    case 'spot':
      return camera.follow?.pinned
        ? 'Placed from its spot on your plan, as you asked. It is followed again when its view next changes.'
        : 'Placed from its spot on your plan.';
    case 'adjusted':
      return `Placed from its spot, adjusted against ${listed(camera.against.map(name))} by what their pictures share: ${metres(camera.shift_m)} and ${degrees(camera.turn_deg)} from the spot.`;
    case 'following': {
      const moves = [
        camera.turn_deg >= 0.5 ? `turned ${degrees(camera.turn_deg)}` : null,
        camera.shift_m >= 0.01 ? `moved ${metres(camera.shift_m)}` : null,
      ].filter(Boolean);
      const what = moves.length
        ? `${moves.join(', ')} from your plan's spot`
        : "it stands where your plan's spot has it";
      const when = camera.follow?.since ? `, ${since(camera.follow.since, now)}` : '';
      const by = PLACED_BY[camera.follow?.by ?? ''];
      // While it is being tracked, where its cloud stands is background to where the camera is now.
      const lead = camera.track ? `Before it began to move: ${what}` : capitalize(what);
      return `${lead}${when}.${by ? ` ${by}` : ''} Your plan is unchanged.`;
    }
    case 'no_spot':
      return 'Not in the scene: it has no spot on your plan. Give it one in the plan editor.';
    case 'no_facing':
      return 'Not in the scene: its spot has no facing. Set which way it looks in the plan editor.';
    case 'lost': {
      const why = LOST[camera.follow?.lost ?? ''];
      if (!why) return 'Its view changed. It is being placed again.';
      return `Its view changed, and ${why}. It isn't drawn, keeps where it last stood, and is tried again with each new picture.`;
    }
  }
}

/** What a moving camera's track says, in plain words: where the camera is now and how far that can be trusted, or,
 *  once it has come to rest, what it waits for. Null for a camera that isn't tracked. `now` is epoch seconds. */
export function trackLine(camera: ScenePlaced, now = Date.now() / 1000): string | null {
  const track = camera.track;
  if (!track) return null;
  const moves = [
    track.from_m >= 0.01 ? `${metres(track.from_m)} away` : null,
    track.from_deg >= 0.5 ? `turned ${degrees(track.from_deg)}` : null,
  ].filter(Boolean);
  const where = moves.length ? `${moves.join(' and ')} from where its points were made` : 'where its points were made';
  if (track.state === 'ended')
    return `It came to rest ${since(track.at, now)}, ${where}. It's placed for good after about 20 seconds with nobody in view.`;
  if (!track.pose) return 'It began to move. Its first frames are being placed.';
  const unheld = track.held === null ? null : Math.round(track.at - track.held);
  const held =
    unheld === null
      ? 'No wall has held it to your plan yet, so it may be off.'
      : unheld < 5
        ? "Held to your plan's walls."
        : `Last held to your plan's walls ${unheld} s before, so it may have drifted since.`;
  return `It is moving, now ${where}, ${since(track.at, now)}. ${held}`;
}

/** What the place's map took of a camera's latest picture, in plain words; null before it was offered one. */
export function mapLine(camera: ScenePlaced): string | null {
  const map = camera.map;
  if (!map) return null;
  const count = plural(map.points, 'point').replace(String(map.points), map.points.toLocaleString());
  const holds = `The map holds ${count} from this camera`;
  if (map.state === 'refused')
    return map.why === 'full'
      ? `Its latest picture isn't in the map: the map is full. ${holds}.`
      : `Its latest picture isn't in the map or drawn: it stands somewhere else than the pictures the map is made of, so it would be a second copy. ${holds}.`;
  const moves = [
    map.moved_m >= 0.01 ? `moved ${metres(map.moved_m)}` : null,
    map.turned_deg >= 0.5 ? `turned ${degrees(map.turned_deg)}` : null,
  ].filter(Boolean);
  const lined = moves.length
    ? ` It was ${moves.join(' and ')} to line up with what the map holds there.`
    : map.off_m > 0.15
      ? ` It couldn't be lined up with what the map holds there (${metres(map.off_m)} out).`
      : '';
  const out = map.left_out
    ? ` ${plural(map.left_out, 'point').replace(String(map.left_out), map.left_out.toLocaleString())} of it ${map.left_out === 1 ? 'is' : 'are'} left out: better-placed pictures saw nothing there.`
    : '';
  const latest = map.added
    ? `${map.added.toLocaleString()} from its latest picture`
    : 'its latest picture showed nothing new';
  return `${holds}: ${latest}.${lined}${out}`;
}

/** The one way back a followed or unplaced camera's row offers, or null: put it on the plan's spot, or, once that
 *  was done, follow it again. Nothing is offered in the moment before a changed view is placed. */
export function wayBack(camera: ScenePlaced): { use: 'spot' | 'cloud'; label: string } | null {
  if (camera.how === 'spot' && camera.follow?.pinned) return { use: 'cloud', label: 'Follow it again' };
  if (camera.how === 'following' || (camera.how === 'lost' && camera.follow?.lost !== 'pending'))
    return { use: 'spot', label: "Use the plan's spot" };
  return null;
}

/** How a camera's cloud was squared to the plan's walls, or why it wasn't; null when nothing was tried (no spot, or
 *  a view that isn't placed). */
export function squareLine(camera: ScenePlaced): string | null {
  const square = camera.square;
  if (!square) return null;
  switch (square.state) {
    case 'squared': {
      if (!camera.squared) return null;
      const slid = camera.fitted ? 0 : Math.hypot(square.right_m, square.ahead_m); // a fit slides it instead
      const turned = Math.abs(square.turn_deg) >= 0.5;
      const moves = [
        turned ? `turned ${Math.abs(square.turn_deg).toFixed(1)}°` : null,
        slid >= 0.01 ? `moved ${metres(slid)}` : null,
      ].filter(Boolean);
      const what = moves.length
        ? `Squared to your plan by its walls: ${moves.join(' and ')}.`
        : "Its walls already run along your plan's.";
      const held =
        square.slide === 'beyond' && !camera.fitted
          ? " Not slid onto the plan's walls: they are farther off than a slide goes."
          : '';
      return `${what}${held}`;
    }
    case 'beyond':
      return `Not squared: its walls line up with your plan only ${Math.round(Math.abs(square.turn_deg))}° from the facing you set, farther than a cloud is turned. Check which way its spot faces.`;
    case 'no_walls':
      return 'No wall in view to square it to your plan by: it faces the way its spot says.';
    case 'no_plan':
      return 'No wall of your plan near it to square it to: it faces the way its spot says.';
  }
}

const percent = (scale: number) => {
  const by = Math.round(Math.abs(scale - 1) * 100);
  return by ? `resized by ${by} percent (${scale < 1 ? 'smaller' : 'larger'})` : 'its size kept';
};

/** How a cloud was fitted to the rooms of the plan, or in plain words why it wasn't. A fitted line keeps how far its
 *  walls stood from the plan's before, when that was more than a few centimetres or degrees. */
export function fitLine(fit: SceneFit): string {
  const slid = Math.hypot(fit.dx, fit.dy);
  switch (fit.state) {
    case 'fitted': {
      const walls = fit.walls.length === 1 ? '1 wall' : `${fit.walls.length} walls`;
      const moved = slid >= 0.01 ? `, moved ${metres(slid)}` : '';
      const one = fit.walls.length === 1 ? ' One wall gives its size, not where it stands across it.' : '';
      const off = Math.max(0, ...fit.walls.map(wall => Math.abs(wall.before_m)));
      const ran = Math.max(0, ...fit.walls.map(wall => wall.off_deg));
      const before = [
        off >= 0.05 ? `stood up to ${metres(off)} from your plan's` : null,
        ran >= 3 ? `ran up to ${degrees(ran)} off them` : null,
      ].filter(Boolean);
      const was = before.length ? ` Before, its walls ${before.join(' and ')}.` : '';
      return `Fitted to the room: ${percent(fit.scale)}${moved}, matched to ${walls}.${one}${was}`;
    }
    case 'no_walls':
      return 'Not fitted to the room: no wall in view.';
    case 'no_match':
      return "Not fitted to the room: the wall it sees isn't where a wall of your plan is. It may be an island, a half wall, or furniture that the plan doesn't draw.";
    case 'beyond':
      return `Not fitted to the room: its walls ask for it to be ${percent(fit.scale)} and moved ${metres(slid)}, more than a fit may do. Check where its spot is.`;
  }
}

/** What a flat wall is, said once by the option that draws them. */
export const FLAT_NOTE = "A flat wall is your plan's wall with what the camera saw painted on it.";

/** The honest part, where it applies: walls of one cloud that are out with each other by `BEND_DEG` or more. */
export function bendLine(camera: ScenePlaced): string | null {
  const bend = camera.square?.bend_deg ?? 0;
  return bend >= BEND_DEG
    ? `Walls in this cloud bend by about ${Math.round(bend)}°: that is the depth estimate, not the placement.`
    : null;
}
const BEND_DEG = 2;

/** When a camera's cloud's picture was taken. */
export function pictureLine(camera: ScenePlaced, now = new Date()): string | null {
  return camera.at ? `From a picture taken ${made(camera.at, now)}.` : null;
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
    case 'checked':
      return `${both} see the same space. Both are squared to your plan, which their pictures don't move: the pictures put the two ${metres(pair.off_m)} and ${degrees(pair.off_deg)} from where they stand.`;
    case 'unsure':
      return `${both} see some of the same space, but their pictures don't agree well enough to adjust them. They are placed from their spots only.`;
    case 'disagrees':
      return `${both}: their pictures put them ${metres(pair.off_m)} and ${degrees(pair.off_deg)} from where their spots do. One of the two spots is off. Nothing was moved.`;
    default:
      return null;
  }
}

/** What the scene is, and isn't, said once over its cameras. */
/** How much of the plan's floor the map covers so far, as a line; null while the place has no map or no floor. */
export function coveredLine(covered: { floor_m2: number; plan_m2: number } | null | undefined): string | null {
  if (!covered || !(covered.plan_m2 > 0)) return null;
  const share = Math.min(100, Math.round((100 * covered.floor_m2) / covered.plan_m2));
  return `It covers about ${share} percent of your plan's floor so far (${Math.round(covered.floor_m2)} of ${Math.round(covered.plan_m2)} m²).`;
}

export const SCENE_LIMITS =
  'One map of what the cameras have seen, each surface once. It fills in as they look around, so expect gaps.';

/** A camera's state in the scene, in a word or two, and whether it wants the owner ("warn"). A camera that is
 *  tracked as it moves (`track`) says that before how its cloud is placed. */
export function placedState(
  how: ScenePlaced['how'],
  track?: 'tracking' | 'ended' | null,
  leftOut = false,
): { label: string; tone: 'plain' | 'warn' } {
  if (track) return { label: track === 'tracking' ? 'Tracking' : 'Moved', tone: 'plain' };
  // Its latest picture disagrees with the place's map and isn't drawn: that wants the owner more than how it stands.
  if (leftOut && how !== 'lost') return { label: 'Left out', tone: 'warn' };
  switch (how) {
    case 'spot':
      return { label: 'Placed', tone: 'plain' };
    case 'adjusted':
      return { label: 'Adjusted', tone: 'plain' };
    case 'following':
      return { label: 'Following', tone: 'plain' };
    case 'lost':
      return { label: 'Not placed', tone: 'warn' };
    case 'no_spot':
    case 'no_facing':
      return { label: 'No spot', tone: 'plain' };
  }
}

/** What to say when the scene has nothing to draw. */
export function sceneEmpty(cameras: readonly ScenePlaced[]): string {
  if (!cameras.length) return 'No camera has a point cloud to place yet.';
  return 'No camera can be placed yet: each needs a spot with a facing on your plan.';
}

/** What to say about the owner's plan on this map: nothing while it's drawn where it belongs (or the place has no
 *  plan of the owner's), and that it isn't drawn on a map it isn't lined up with. The plan itself is never replaced:
 *  the Plan view shows it as drawn. No reachable scene sets an unaligned owner plan (the combined scene always says
 *  it is aligned), so this says nothing today. */
export function planNote(scene: Scene): string | null {
  if (scene.plan_source !== 'owner' || !scene.revision) return null;
  if (!scene.plan_aligned)
    return "Your plan isn't drawn on this map: the map isn't lined up with it. The Plan view shows your plan as you drew it.";
  return null;
}

/** A splat's loading, as text: how much of it has come, then that it's being drawn. No producer writes a splat today
 *  (`memory/places.SHOWN_PRODUCERS` is empty), so this waits with the rest of the splat path (`splatNote`). */
export function loadingText(received: number, total: number, drawing: boolean): string {
  if (drawing) return 'Drawing the splat…';
  const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;
  if (!total) return received ? `Loading the splat: ${mb(received)}…` : 'Loading the splat…';
  return `Loading the splat: ${Math.min(100, Math.floor((received / total) * 100))}% of ${mb(total)}`;
}
