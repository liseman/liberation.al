// Requests for the Ingest tab and Settings' sources (dashboard/ingest.py): the items and sources, uploads, links,
// source polls, Google Photos, and Wyze's settings (saved through /config, as Settings saves them).
import { request, upload } from '../../shared/http';

export const INGEST_KEY = ['ingest'] as const;

export type ItemStatus = 'queued' | 'converting' | 'registering' | 'done' | 'failed';

export interface IngestItem {
  id: number;
  sha256: string;
  source: string;
  source_key: string | null;
  name: string;
  kind: 'photo' | 'video' | 'audio' | null;
  original_ts: number | null;
  time_from: 'exif' | 'ffprobe' | 'source' | 'upload' | null;
  added_ts: number;
  status: ItemStatus;
  detail: string;
  stream: string;
  outputs: { kind: string; stream: string; chunk_id: string }[];
  /** What reading it found (#2719), from the analysis ledger; null while it's still being brought in. Absent from a
   *  server before #2737. */
  analysis?: Analysis | null;
}

export type Analysis =
  | { state: 'waiting' }
  | { state: 'retrying' }
  | { state: 'failed'; error: string | null }
  | { state: 'done'; frames: number; captions: number; lines: number };

export interface SourceInfo {
  name: string;
  label: string;
  ongoing: boolean;
  blocked?: boolean; // setting it up can't go on from here (a package missing on this machine): its sign-in waits
  later?: boolean; // no way to bring it in yet (a stub): Settings lists it under Coming later (#2728)
  configured: boolean;
  message: string;
  seen: number;
}

export interface Job {
  id: number;
  what: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  error: string;
  item: number | null;
}

export interface WyzeSettings {
  cameras: string[];
  rtsp_url: string;
  segment_s: number;
  email: boolean;
  password: boolean;
  api_id: boolean;
  api_key: boolean;
}

export interface IngestDeletion {
  id: number;
  name: string;
  detail: string;
  blocked?: boolean;
}

export interface IngestView {
  items: IngestItem[];
  deletions?: IngestDeletion[];
  sources: SourceInfo[];
  jobs: Job[];
  control: boolean;
  wyze: WyzeSettings;
}

export interface Album {
  id: string;
  title: string;
}

export interface RemotePhoto {
  key: string;
  name: string;
  taken_at: number | null;
  url: string;
  kind: 'photo' | 'video';
  thumb: string;
  label?: string;
}

export const getIngest = (signal?: AbortSignal) => request<IngestView>('/ingest/api/items', undefined, signal);

/** One file, its bytes the request's body, with how much of it is sent so far (#2718). */
export const uploadFile = (file: File, onProgress?: (share: number) => void) =>
  upload<{ item: IngestItem; new: boolean }>(
    `/ingest/api/upload?name=${encodeURIComponent(file.name)}`,
    file,
    onProgress,
  );

export const addLink = (url: string) => request<{ job: Job }>('/ingest/api/link', { url });

export const retryItem = (id: number) => request<{ job: Job }>('/ingest/api/retry', { id });

export const deleteItem = (id: number, { poll = false, signal }: { poll?: boolean; signal?: AbortSignal } = {}) =>
  request<{ ok: true; deleted: boolean; pending: boolean }>(
    '/ingest/api/delete',
    { id, ...(poll ? { poll: true } : {}) },
    signal,
  );

export const pollSource = (name: string) =>
  request<{ job: Job }>(`/ingest/api/sources/${encodeURIComponent(name)}/poll`, {});

export const googleLogin = () => request<{ ok: boolean }>('/ingest/api/google_photos/login', {});

export const googleAlbums = (signal?: AbortSignal) =>
  request<{ albums: Album[] }>('/ingest/api/google_photos/albums', undefined, signal);

export const googlePhotos = (album: string | null, signal?: AbortSignal) =>
  request<{ items: RemotePhoto[] }>(
    `/ingest/api/google_photos/photos${album ? `?album=${encodeURIComponent(album)}` : ''}`,
    undefined,
    signal,
  );

export const googleImport = (items: RemotePhoto[]) =>
  request<{ job: Job }>('/ingest/api/google_photos/import', { items });

/** The Wyze form's values as settings: a credential left blank keeps the one saved (config.save does that). */
export function wyzeValues(form: {
  cameras: string;
  rtsp_url: string;
  email: string;
  password: string;
  api_id: string;
  api_key: string;
}): Record<string, unknown> {
  const values: Record<string, unknown> = {
    'ingest.wyze_cameras': form.cameras
      .split(/[,\n]/)
      .map(c => c.trim())
      .filter(Boolean),
    'ingest.wyze_rtsp_url': form.rtsp_url.trim(),
  };
  for (const k of ['email', 'password', 'api_id', 'api_key'] as const)
    if (form[k].trim()) values[`ingest.wyze_${k}`] = form[k].trim();
  return values;
}

export const saveWyze = (values: Record<string, unknown>) =>
  request<{ ok: boolean; restarted?: string[] }>('/config', { values });
