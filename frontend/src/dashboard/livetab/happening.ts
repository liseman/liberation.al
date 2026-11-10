// "What's happening" (#354, #356): a plain-language log of the room, newest at the bottom. The server writes it; Live
// polls for what is new since the last reply while it shows (a hidden tab polls nothing, #330).
import { type QueryClient, useQuery, useQueryClient } from '@tanstack/react-query';
import { request } from '../../shared/http';

export interface HappeningItem {
  id: string; // stable ("p:<episode id>" or "t:<start ts>"): rows key on it, and a poll never doubles one
  removed?: boolean; // a revised anonymous episode was identified on the same physical track
  ts: number; // when it happened
  kind: 'arrival' | 'departure' | 'activity' | 'talk';
  text: string; // "Kara walked in and sat on the couch": third person, no trailing period
  people: string[]; // the known people in it
  cam: string | null;
  camera_name: string | null; // the camera's name (vision.camera_names)
}

export interface Happening {
  items: HappeningItem[]; // oldest first
  updated: number | null; // where the newest item was written (a stream position): the next poll asks for what is newer
}

export const HAPPENING_KEY = ['live', 'happening'] as const;
export const HAPPENING_MS = 10_000;
export const KEEP = 50; // rows the rail shows

/** prev with next's items added: each id once, oldest first, the newest KEEP. */
export function merge(prev: Happening | undefined, next: Partial<Happening> | null): Happening {
  const byId = new Map((prev?.items ?? []).map(i => [i.id, i]));
  for (const i of Array.isArray(next?.items) ? next.items : []) {
    if (!i?.id) continue;
    if (i.removed) byId.delete(i.id);
    else byId.set(i.id, i);
  }
  const items = [...byId.values()].sort((a, b) => a.ts - b.ts).slice(-KEEP);
  return { items, updated: next?.updated ?? prev?.updated ?? null };
}

async function poll(client: QueryClient, signal?: AbortSignal): Promise<Happening> {
  const prev = client.getQueryData<Happening>(HAPPENING_KEY);
  const since = prev?.updated;
  const r = await request<Partial<Happening> | null>(
    `/live/happening?limit=${KEEP}${since != null ? `&since=${encodeURIComponent(since)}` : ''}`,
    undefined,
    signal,
  );
  return merge(prev, r);
}

export function useHappening() {
  const client = useQueryClient();
  return useQuery({
    queryKey: HAPPENING_KEY,
    queryFn: ({ signal }) => poll(client, signal),
    refetchInterval: HAPPENING_MS,
    staleTime: 0, // each poll asks only for what is new, so asking again on show costs nothing
  });
}
