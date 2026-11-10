// Known › Places' reads: the places the house has (GET /places, memory/places.py) and where people and pets were in
// one of them over a window of time (GET /places/positions; the position layer is spatial/positions.py).
import { queryOptions } from '@tanstack/react-query';
import { request, upload } from '../../shared/http';

/** A place, as GET /places lists it. */
export interface PlaceSummary {
  id: string;
  /** The owner's name for it, or its placeholder ("Place 1"). */
  name: string;
  named: boolean;
  /** Its floorplan in the line DSL (the owner's correction if there is one, else the machine's), or null. */
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
  /** How many walkthroughs went into the place (#3). */
  walks?: number;
  /** The plan's rooms no LiDAR walk covered, in plan order (every room while it has never been walked); null when it
   *  can't tell (no plan, or no fused map in the plan's frame). */
  unwalked_rooms?: { id: string; name: string }[] | null;
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

/** One owner plan in a place's history (GET /places/plan/history): when it was saved (epoch seconds), how many rooms
 *  it has, and whether it was a withdrawal (an empty plan, so the machine's held). */
export interface PlanVersion {
  id: string;
  at: number;
  rooms: number;
  source: string;
  withdrawn: boolean;
}

/** A place's owner plans, oldest first, and `previous`: the one Revert plan puts back (null with none). */
export interface PlanHistory {
  place: string;
  versions: PlanVersion[];
  previous: PlanVersion | null;
}

export const planHistoryQuery = (place: string) =>
  queryOptions({
    queryKey: ['places', 'plan-history', place],
    queryFn: ({ signal }) =>
      request<PlanHistory>(`/places/plan/history?${new URLSearchParams({ place })}`, undefined, signal),
    staleTime: 60_000,
  });

/** Make an earlier owner plan hold again (Revert plan): written as a new record, so the one it replaces stays in the
 *  history and can be put back the same way. */
export const restorePlan = (place: string, record: string) =>
  request<{ ok: true; plan: string | null; plan_source: 'owner' | 'machine' | null }>('/places/plan/restore', {
    place,
    record,
  });

/** What the 3D view draws for a place (GET /places/scene), all in its map's frame, or `revision: null` while it
 *  has no map. */
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
  /** What the splat was trained from (#44): a walk's own frames, or the fixed cameras'. */
  splat_from?: 'walk' | 'cameras' | null;
  /** The floor plane: a unit normal and its offset (a, b, c, d), in the map's frame. */
  floor?: number[] | null;
  /** Map to plan, 3x4 row by row. */
  plan_transform?: number[] | null;
  cameras?: Record<string, { pose: number[]; focal: number; width: number; height: number; at: number }>;
  /** The plan in this revision's frame (DSL), and whose it is. */
  plan?: string | null;
  plan_source?: 'owner' | 'machine';
}

export const sceneQuery = (place: string) =>
  queryOptions({
    queryKey: ['places', 'scene', place],
    queryFn: ({ signal }) => request<Scene>(`/places/scene?${new URLSearchParams({ place })}`, undefined, signal),
    staleTime: 5 * 60_000,
  });

/** A walk import (dashboard/walk_import.py): a walkthrough dropped on a place, uploaded, then imported into it in the
 *  background. `status` goes uploading, extracting, importing, then done or failed. */
export interface WalkJob {
  id: string;
  /** The place it was dropped on. */
  place: string;
  /** The dropped file's name. */
  name: string;
  kind: 'stray' | 'video';
  status: 'uploading' | 'extracting' | 'importing' | 'done' | 'failed';
  /** The importer's newest log line, while it runs. */
  detail: string;
  error: string;
  /** What the import printed: the place and revision it wrote, and `meant_for` when the walk didn't register in the
   *  place it was dropped on and was stored as a place of its own. */
  result: { place?: string; revision?: string; registered?: string; meant_for?: string } & Record<string, string>;
  /** A failed import kept its files, so it can run again without a second upload. */
  retry: boolean;
  started: number;
  finished: number | null;
}

/** GET /places/walk: the newest import, the largest upload taken (MB), and whether a plain video is taken here. */
export interface WalkImports {
  job: WalkJob | null;
  max_mb: number;
  video: boolean;
}

export const walkActive = (job: WalkJob | null | undefined): boolean =>
  !!job && job.status !== 'done' && job.status !== 'failed';

export const walkQuery = () =>
  queryOptions({
    queryKey: ['places', 'walk'],
    queryFn: ({ signal }) => request<WalkImports>('/places/walk', undefined, signal),
    // While an import runs the page follows it; otherwise there is nothing to ask about.
    refetchInterval: q => (walkActive(q.state.data?.job) ? 2000 : false),
    staleTime: 0,
  });

/** Upload a walk (the zip Stray Scanner exports, or a video) for a place. The body is the file itself, sent as
 *  application/octet-stream: the type the route takes, and one no cross-site page can send unasked. */
export const uploadWalk = (place: string, file: File, onProgress?: (share: number) => void) =>
  upload<{ job: WalkJob }>(
    `/places/walk/upload?${new URLSearchParams({ place, name: file.name })}`,
    file.slice(0, file.size, 'application/octet-stream'),
    onProgress,
  );

export const retryWalk = () => request<{ job: WalkJob }>('/places/walk/retry', {});
export const dismissWalk = () => request<{ job: null }>('/places/walk/dismiss', {});
