// The Objects page's API (dashboard/objects.py), served by the dashboard under /objects on Continuum.

import type { FrameMetadata } from '../dashboard/camera/geometry';
import { request } from '../shared/http';
import type { ObjectDraft } from './handoff';

export type Box = [x: number, y: number, w: number, h: number];

export interface Sighting {
  camera: string;
  captured_at: number;
  kind: 'manual_label' | 'visual_match';
  evidence: string;
  bbox: Box; // normalized to the image, 0 to 1
  stale: boolean;
}

export interface MemoryObject {
  id: string;
  name: string;
  aliases: string[];
  note: string;
  template: string;
  mode: 'visual' | 'manual_only';
  last_seen: Sighting | null;
}

export interface CameraStatus {
  camera: string;
  /** Its name from Settings (vision.camera_names), or its id when it has none. */
  name?: string;
  available: boolean;
  stale?: boolean;
  source?: string;
  error?: string;
}

export interface Status {
  objects: number;
  cameras: CameraStatus[];
  observer: { finished_at: number } | null;
  observer_error: string | null;
}

export interface QueryResult {
  status: 'last_seen' | 'ambiguous' | 'unknown';
  answer: string;
  evidence?: Sighting;
  history?: Sighting[];
}

export interface Snapshot {
  frame_metadata?: FrameMetadata;
  token: string;
  evidence: string;
  camera: string;
  captured_at: number;
  stale: boolean;
  source: string;
}

export interface Draft {
  snapshot: Snapshot;
  name: string;
  saved_at: number;
}

export interface Segmentation {
  selection: string;
  snapshot: string;
  bbox: Box;
  mask: string;
  score: number;
  model: string;
}

/** One object Find objects found: a selection like `segment`'s, with its outline (normalized [x, y] polygons). */
export interface FoundObject extends Segmentation {
  outline: [number, number][][];
}

export interface EnrollRequest {
  name: string;
  camera: string;
  snapshot: string;
  selection?: string;
  full_crop?: boolean;
  crop?: Box; // full-resolution frame pixels [x,y,w,h], unlike normalized bbox
  direct_crop?: boolean;
  bbox: Box;
  aliases: string;
  note: string;
}

// Cleanup can run after navigation has already changed location.pathname. Keep every job request on its owner.
const base = '/objects';

export const evidenceUrl = (token: string): string => `${base}/evidence/${encodeURIComponent(token)}`;

export const getStatus = () => request<Status>(`${base}/api/status`);
export const getObjects = () => request<{ objects: MemoryObject[] }>(`${base}/api/objects`);
export const getDraft = () => request<{ draft: Draft | null }>(`${base}/api/draft`);
export const askWhere = (q: string) => request<QueryResult>(`${base}/api/query?q=${encodeURIComponent(q)}`);
export const capture = (camera: string) => request<Snapshot>(`${base}/api/capture`, { camera });
/** EdgeTAM once for the box or point the owner chose (normalized to the snapshot). `job` lets a new view cancel it. */
export const segment = (
  snapshot: string,
  prompt: { bbox: Box } | { point: [number, number] },
  job: string,
  signal?: AbortSignal,
) => request<Segmentation>(`${base}/api/segment`, { snapshot, ...prompt, job }, signal);
/** EdgeTAM's automatic masks over `crop` (frame pixels: the region in view), or the whole snapshot. */
export const findObjects = (snapshot: string, crop: Box | null, job: string, signal?: AbortSignal) =>
  request<{ snapshot: string; objects: FoundObject[]; ms: number }>(
    `${base}/api/find`,
    { snapshot, ...(crop ? { crop } : {}), job },
    signal,
  );
export const cancelSegment = (job: string) =>
  request<{ ok: boolean; cancelled: boolean }>(`${base}/api/cancel`, { job });
export const enroll = (body: EnrollRequest) => request<{ object: MemoryObject }>(`${base}/api/enroll`, body);

/** The picture only: a Live draft's clicked box stays in the browser for the one EdgeTAM request after import. */
export const importView = (draft: ObjectDraft) => {
  const { box: _box, ...picture } = draft.kind === 'full_frame' ? { ...draft, box: undefined } : draft;
  return request<Snapshot>(`${base}/api/import`, picture);
};
