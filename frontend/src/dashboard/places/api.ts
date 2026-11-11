// Known › Places' reads: the places the house has (GET /places, memory/places.py) and where people and pets were in
// one of them over a window of time (GET /places/positions; the position layer is spatial/positions.py).
import { keepPreviousData, queryOptions } from '@tanstack/react-query';
import type { Scene as PlanGraph } from '../../plan-editor/protocol';
import { request } from '../../shared/http';

/** A place, as GET /places lists it. */
export interface PlaceSummary {
  id: string;
  /** The owner's name for it, or its placeholder ("Place 1"). */
  name: string;
  named: boolean;
  /** Its plan as stored: a Pascal scene's JSON (the owner's if there is one, else the machine's), or null. */
  plan: string | null;
  plan_source: 'owner' | 'machine' | null;
  /** A machine plan was made after the owner's correction that holds. */
  machine_newer: boolean;
  /** The newest machine revision: when it was made, and by what. */
  revision: { id: string; at: number; producer: string } | null;
  /** The revision whose frame the plan is drawn in (positions come back in it). */
  plan_revision: string | null;
  /** The newest machine plan, while the owner's correction holds over it (to compare), else null. */
  machine_plan: string | null;
  /** When the house was last there (epoch seconds), or null. */
  last_visit: number | null;
  /** When anyone was last seen there by its fixed cameras, apart from the owner's visits. */
  last_activity?: number | null;
}

export const placesQuery = () =>
  queryOptions({
    queryKey: ['places'],
    queryFn: ({ signal }) => request<{ places: PlaceSummary[] }>('/places', undefined, signal),
    staleTime: 60_000,
  });

/** Who a path belongs to. `id` is "person:<id>", "pet:<id>", "rig", or a camera track for someone not known yet. */
export interface Entity {
  id: string;
  name: string | null;
  kind: 'person' | 'pet' | 'rig' | 'unknown';
}

/** One position: when, where on the plan (metres, the plan's own frame), in which room, and how sure. */
export interface Point {
  t: number;
  x: number;
  y: number;
  room: string | null;
  confidence: number;
  source: string;
}

/** A stretch of time someone was known to be in a room, but not where in it (a zone visit). */
export interface RoomSpan {
  entity: string;
  /** The plan's room id, when the room is one of the plan's; null for a zone that isn't. */
  room: string | null;
  /** The room's (or zone's) name. */
  name: string;
  start: number;
  end: number;
  confidence: number;
  source: string;
}

export interface Positions {
  place: string;
  start: number;
  end: number;
  entities: Entity[];
  /** One path per camera track (`track`), so one entity can have several: two cameras' views of one person aren't
   *  joined. */
  tracks: { entity: string; track?: string; points: Point[] }[];
  rooms: RoomSpan[];
  /** How many cameras can place people here: posed from a spot on the owner's plan, or in the place's map. With
   *  none, nobody can be placed there. */
  cameras?: number;
  /** Each camera of the place: how it's posed, and what a spot still lacks for a pose ("height", "facing", "lens",
   *  "tilt", or "check"; empty when it places). */
  camera_notes?: { camera: string; by: 'spot' | 'map'; missing: string[] }[];
  /** Whether the place has a map. With none, and no camera spot on its plan, the page says "no map yet". */
  map?: boolean;
}

export const positionsQuery = (place: string, start: number, end: number) =>
  queryOptions({
    queryKey: ['places', 'positions', place, start, end],
    queryFn: ({ signal }) =>
      request<Positions>(
        `/places/positions?${new URLSearchParams({ place, start: String(start), end: String(end) })}`,
        undefined,
        signal,
      ),
    staleTime: 30_000,
    // A read of a busy place is about a second of server work, and the server stops one the page gave up on. A re-read
    // on every return to the window is still work for nothing: while the window follows now, PlacePaths moves it (a
    // new read) every FOLLOW_NOW_MS anyway.
    refetchOnWindowFocus: false,
  });

/** Save the owner's corrected plan for a place (taught state over the machine's), or go back to the machine's with "".
 *  `base` is the revision whose frame the edited text is in: the plan's plan_revision when the editor
 *  opened it. */
export const savePlan = (place: string, plan: string, base?: string | null) =>
  request<{ ok: true; plan: string | null; plan_source: 'owner' | 'machine' | null }>('/places/plan', {
    place,
    plan,
    ...(base ? { base } : {}),
  });

/** One saved version of a place's owner plan (GET /places/plan/history): when it was saved (epoch seconds), how many
 *  rooms it has, whether it was a withdrawal (an empty plan, so the machine's held), whether it is the one that holds, and what changed from the version before it. */
export interface PlanVersion {
  id: string;
  at: number;
  rooms: number;
  source: string;
  withdrawn: boolean;
  current: boolean;
  summary: string;
}

/** A place's saved plan versions, oldest first. */
export interface PlanHistory {
  place: string;
  versions: PlanVersion[];
}

export const planHistoryQuery = (place: string) =>
  queryOptions({
    queryKey: ['places', 'plan-history', place],
    queryFn: ({ signal }) =>
      request<PlanHistory>(`/places/plan/history?${new URLSearchParams({ place })}`, undefined, signal),
    staleTime: 60_000,
  });

/** A place's plan as a Pascal scene (GET /places/plan/scene), for the plan editor: the plan that holds, or with
 *  `record` an earlier version of it. `base` is the revision it is drawn in, which a save sends back. A place with
 *  no plan answers 404, and a plan that can't be read 422. */
export interface PlanScene {
  place: string;
  scene: PlanGraph;
  base: string | null;
}

export const planSceneQuery = (place: string, record?: string) =>
  queryOptions({
    queryKey: ['places', 'plan-scene', place, record ?? null],
    queryFn: ({ signal }) =>
      request<PlanScene>(
        `/places/plan/scene?${new URLSearchParams({ place, ...(record ? { record } : {}) })}`,
        undefined,
        signal,
      ),
    staleTime: 0, // the editor opens on the plan as it is now
    gcTime: 0,
    retry: false, // no plan yet is an answer, not a failure to try again
  });

/** Make an earlier version of the owner's plan hold again: written as a new version on top, so the one it replaces
 *  stays in the history and can be put back the same way. */
export const restorePlan = (place: string, record: string) =>
  request<{ ok: true; plan: string | null; plan_source: 'owner' | 'machine' | null }>('/places/plan/restore', {
    place,
    record,
  });

/** Whether pictures are collected from the cameras (the Live tab's "Learn surroundings"), which is also what keeps
 *  each camera's own cloud up to date. */
export interface Learning {
  /** The switch. */
  on: boolean;
  /** How many cameras it samples now; none while it's off. */
  cameras: number;
}

/** Why a camera has no cloud, or no newer one: what the cloud maker noted of its last try, or that learning is off,
 *  the camera isn't chosen, or it is waiting for a picture with nobody in it. */
export type ViewWhy =
  | 'off'
  | 'not_chosen'
  | 'waiting'
  | 'no_runtime'
  | 'dark'
  | 'people'
  | 'lens'
  | 'gpu'
  | 'failed'
  | 'unsure'
  | 'flat';

/** One camera the 3D view offers: whether it has a cloud of its own view, and why not (or why no newer one). */
export interface CameraView {
  camera: string;
  cloud: boolean;
  /** When its cloud was made (epoch seconds), how many points it has, and how it was sized; null without one. */
  at: number | null;
  points: number | null;
  size: 'approximate' | null;
  why: ViewWhy | null;
}

/** One camera in a place's scene of all its cameras' clouds: how its cloud is placed, or why it isn't in the scene. */
export interface ScenePlaced {
  camera: string;
  points: number;
  placed: boolean;
  /** "spot": from its spot on the plan alone; "adjusted": from its spot, corrected against the cameras in `against`
   *  by what their pictures share; "moved": it moved, and its cloud followed by what its pictures measured;
   *  "moved_against": it moved, and is placed from what it shares with the camera in `against`. Not in the scene:
   *  "no_spot", "no_facing", and "moved_unplaced" (it moved, and its picture matches nothing). */
  how: 'spot' | 'adjusted' | 'moved' | 'moved_against' | 'no_spot' | 'no_facing' | 'moved_unplaced';
  size: 'approximate';
  against: string[];
  /** How far from its spot it is placed: metres, and degrees of facing. */
  shift_m: number;
  turn_deg: number;
  /** The spot offered for a camera that moved: plan x and y, and facing in degrees. Saved only when taken. */
  suggest: [number, number, number] | null;
  /** When its cloud's picture was taken (epoch seconds). */
  at?: number;
  /** What its cloud was levelled by: its own floor, its walls (no floor in view), or nothing (it stands as if the
   *  camera were upright). With the camera's tilt (degrees it looks down) and roll by that. */
  level?: CloudLevel;
  tilt_deg?: number;
  roll_deg?: number;
  /** Its cloud's squaring to the plan's walls, when one was made from this cloud, spot, and plan; `squared` when it
   *  is what placed the camera (a camera that moved is placed by its pictures instead). */
  square?: SceneSquare | null;
  squared?: boolean;
  /** Its cloud's fit to the rooms of the plan, when one was made from this cloud, spot, and plan; `fitted` when it is
   *  what sized and placed the cloud, whose floor and matched walls can then be drawn flat. */
  fit?: SceneFit | null;
  fitted?: boolean;
}

/** How a cloud was fitted to the plan's rooms: resized about its camera and slid so its walls land on the plan's. */
export interface SceneFit {
  /** "fitted"; "no_walls": no wall in the cloud; "no_match": no wall of it is where a plan wall is; "beyond": the fit
   *  asks for more than it may do (`scale`, `dx`, `dy` say what it would have taken). */
  state: 'fitted' | 'no_walls' | 'no_match' | 'beyond';
  /** Its size across the floor against the model's own metres, and upwards. */
  scale: number;
  up: number;
  /** The slide of the camera from its spot (m). */
  dx: number;
  dy: number;
  /** The walls matched to plan walls: how far each ran off its plan wall (degrees), and stood from it before and
   *  after the fit (m). */
  walls: { off_deg: number; before_m: number; after_m: number }[];
  /** The walls in the cloud. */
  seen: number;
}

export type CloudLevel = 'floor' | 'walls' | 'none';

/** How a cloud was squared to the plan: turned about the vertical so its walls run along the plan's, and slid onto
 *  them where they have clear counterparts. A rigid move, never a stretch. */
export interface SceneSquare {
  /** "squared"; "no_walls": no wall in the cloud; "no_plan": no plan wall near the camera; "beyond": its walls line
   *  up only farther round than a cloud is turned, so it was left alone (`turn_deg` is what it would have taken). */
  state: 'squared' | 'no_walls' | 'no_plan' | 'beyond';
  turn_deg: number;
  right_m: number;
  ahead_m: number;
  /** Why it wasn't slid, when it wasn't: no wall with one clear counterpart, or a slide past the bound. */
  slide: '' | 'no_counterpart' | 'beyond';
  /** How far the cloud's walls are out with each other (degrees), which no rigid move takes out. */
  bend_deg: number;
}

/** What two cameras' pictures said of each other. */
export interface ScenePair {
  a: string;
  b: string;
  /** "tightened": they match, and adjust the pair; "apart": they show different space; "unsure": they share some,
   *  but don't agree well enough; "disagrees": they match, but farther from the spots than a correction may go;
   *  "moved": one of the two has moved since; "stale": measured at another spot; "checked": both cameras are squared
   *  to the plan, which the measurement doesn't move, so it only says how far off the two are. */
  state: 'tightened' | 'checked' | 'apart' | 'unsure' | 'disagrees' | 'moved' | 'stale';
  /** How far what they measured is from what the spots say: metres, and degrees. */
  off_m: number;
  off_deg: number;
  /** When the measurement in use was made (epoch seconds). */
  measured: number | null;
  /** Why a newer look wasn't used in its place, when one wasn't (a night picture that didn't match, say). */
  latest: string | null;
}

/** What the 3D view draws for a place (GET /places/scene), all in its map's frame, or `revision: null` while it
 *  has nothing to draw. With `view`, the scene is one camera's own cloud, not the place's map. */
export interface Scene {
  place: string;
  revision: string | null;
  at?: number;
  /** The map file's URL (served with byte ranges and cached for good), its format, and its size. */
  map?: string;
  map_format?: string | null;
  map_bytes?: number;
  /** The revision's 3D Gaussian splat: its URL (cached for good like the map), format, and size; null while
   *  it has none. */
  splat?: string | null;
  splat_format?: string | null;
  splat_bytes?: number | null;
  /** Which revision the splat was made for: this one, or an earlier one in the same map frame. */
  splat_revision?: string | null;
  /** The newest splat when it was made in another map frame, so it would sit out of place: named, not drawn. */
  splat_stale?: { revision: string; at: number } | null;
  /** What the splat was trained from, as its trainer named it (the fixed cameras' frames are "cameras"). */
  splat_from?: string | null;
  /** Which state the splat is in, so the page doesn't guess: made for this map ("current"), made for an earlier map
   *  in the same frame, so it still lines up ("earlier"), made in another frame and so not offered ("stale"), or
   *  there is none. */
  splat_state?: 'current' | 'earlier' | 'stale' | 'none';
  /** When the map the offered splat was made for was built (epoch seconds). */
  splat_at?: number | null;
  /** The floor plane: a unit normal and its offset (a, b, c, d), in the map's frame. */
  floor?: number[] | null;
  /** Map to plan, 3x4 row by row. */
  plan_transform?: number[] | null;
  cameras?: Record<string, { pose: number[]; focal: number; width: number; height: number; at: number }>;
  /** The place's plan (a Pascal scene's JSON), and whose it is. A place with an owner's plan always gives that one. It's in this
   *  revision's frame only with a `plan_transform`: an owner's plan a learned map isn't lined up with yet has none,
   *  and isn't drawn on the map. */
  plan?: string | null;
  plan_source?: 'owner' | 'machine';
  plan_aligned?: boolean;
  /** Whether pictures are collected for the map; also given while the place has no map. */
  learning?: Learning;
  /** The cameras whose own clouds the view offers. */
  views?: CameraView[];
  /** The scene of all the place's cameras' clouds, each placed where its camera is (asked for with `all`): every
   *  camera with how it is placed or why it isn't, in the order the points name them, and each pair's reading. */
  combined?: { place: string; id: string; cameras: ScenePlaced[]; pairs: ScenePair[] };
  /** The camera whose own cloud this scene is (its points, in a level frame of its own, with the camera in it), with
   *  how it was sized: always the model's own approximate metres (`cloud.size`, spatial/camera_clouds.py); a measured
   *  mounting height only checks which plane is the floor. Only `camera` while that camera has no cloud yet. */
  view?: {
    camera: string;
    size?: 'approximate';
    floor_found?: boolean;
    points?: number;
    at?: number;
    /** What the cloud was levelled by, and the camera's tilt and roll by that. */
    level?: CloudLevel;
    tilt_deg?: number;
    roll_deg?: number;
    /** Its fit to the room, for a camera with a spot, and whether this scene is drawn with flat walls and floor. */
    fit?: SceneFit | null;
    flat?: boolean;
  };
}

/** Take the spot the scene suggests for a camera that moved: its spot on the owner's plan is moved and turned to
 *  it, saved as a new version of the plan. */
export const acceptSpot = (place: string, camera: string) =>
  request<{ ok: true; spot: { at: [number, number]; yaw: number }; previous: string | null }>('/places/scene/spot', {
    place,
    camera,
  });

/** Take that back: the plan version it replaced (`record`) holds again, and the camera is a moved one once more. */
export const undoSpot = (place: string, camera: string, record: string) =>
  request<{ ok: true }>('/places/scene/spot/undo', { place, camera, record });

/** How often the 3D view asks for the scene again while it's open: a learned map gets a newer splat every so
 *  often while learning is on, and the view draws it without a reload. With learning off it asks seldom, to notice
 *  that it was turned on. */
export const SCENE_LEARNING_MS = 10_000;
export const SCENE_IDLE_MS = 60_000;

/** A place's scene; with `camera`, that camera's own cloud; with `all`, every camera's cloud placed in the place.
 *  `flat` (the default) draws the floor and matched walls of clouds fitted to their rooms flat; without it every
 *  point is where its cloud has it. Changing the camera keeps the scene before it on screen until the new one
 *  answers. */
export const sceneQuery = (place: string, camera?: string | null, all = false, flat = true) =>
  queryOptions({
    queryKey: ['places', 'scene', place, all ? '@all' : (camera ?? null), flat],
    queryFn: ({ signal }) =>
      request<Scene>(
        `/places/scene?${new URLSearchParams({
          place,
          ...(all ? { all: '1' } : camera ? { camera } : {}),
          ...(flat ? {} : { flat: '0' }),
        })}`,
        undefined,
        signal,
      ),
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
    refetchInterval: q => (q.state.data?.learning?.on ? SCENE_LEARNING_MS : SCENE_IDLE_MS),
  });
