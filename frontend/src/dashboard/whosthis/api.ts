// "Who's this?" data: the unnamed face groups (#310, vision/face_guess.py) and unnamed voices (#347,
// people/voice_guess.py) waiting for a name, each with a best guess, and putting one aside. Live and People share
// these queries and their cache.
import { type QueryClient, queryOptions } from '@tanstack/react-query';
import { HttpError, messageOf, request } from '../../shared/http';

export type Band = 'probable' | 'maybe' | 'none';

/** The best guess at who an unnamed face or voice is: a 0 to 100 score and its band. */
export interface Guess {
  id?: string; // what naming sends back (#2366): the person's name for faces and voices
  name: string;
  score: number;
  band: Band;
  close: string | null; // the runner-up, when being near them is what held the score down
}

/** Who was heard while a face was in view, or seen while a voice spoke (#225, people/together.py): a suggestion,
 *  with how many separate moments, the newest, and the newest few moments as its evidence. */
export interface Together {
  name: string;
  moments: number;
  last: number;
  times: number[];
  others: Record<string, number>; // anyone else seen or heard with it, and how many moments
}

interface Waiting {
  id: number;
  label: string;
  count: number; // times seen (a face) or lines heard (a voice)
  last_seen: number | null;
  guess: Guess | null; // null when there's no good guess
  score: number;
  band: Band;
  reason: string;
  not?: string[]; // the people it was said not to be (#2366)
  together?: Together | null; // a suggestion from who was seen or heard with it (#225)
}

/** An unnamed face group waiting for a name. */
export interface UnknownGroup extends Waiting {
  shots: { id: number; ts: number; url: string }[]; // its newest recorded pictures, newest first
  shot_count?: number; // saved recent pictures (#2008); absent for a face just seen
  shots_pending?: number;
  shots_unavailable?: number;
}

/** An unnamed voice waiting for a name. */
export interface UnknownVoice extends Waiting {
  // its most useful clips to play, best first; line is the clip's line key (#247), what requests about it send
  clips: { ts: number; line?: string; text: string; dur: number; url: string }[];
}

export const UNKNOWN_GROUPS_KEY = ['faces', 'unknown', 'groups'] as const;
export const UNKNOWN_VOICES_KEY = ['voices', 'unknown', 'groups'] as const;

interface Listed<T> {
  groups: T[];
  count: number | null;
  skipped: number[];
  pending: boolean;
  refreshing?: boolean; // the server's last answer, patched, while it works out the full one (#2332)
  shots_pending?: number;
  shots_unavailable?: number;
  groups_without_shots?: number;
}

const listed = async <T>(url: string, signal: AbortSignal): Promise<Listed<T>> => {
  const r = await request<{
    groups?: T[];
    count?: number | null;
    skipped?: number[];
    pending?: boolean;
    refreshing?: boolean;
    shots_pending?: number;
    shots_unavailable?: number;
    groups_without_shots?: number;
  }>(url, undefined, signal, 'page');
  const groups = r.groups ?? [];
  const pending = r.pending === true;
  return {
    groups,
    count: pending ? null : (r.count ?? groups.length),
    skipped: r.skipped ?? [],
    pending,
    refreshing: r.refreshing === true,
    shots_pending: r.shots_pending,
    shots_unavailable: r.shots_unavailable,
    groups_without_shots: r.groups_without_shots,
  };
};

/** How many faces and voices wait in Who's this? in all, as one number: People's line, Who's this?'s own count, and
 *  Known's People badge all show it (#2451). Null until both lists have been read: the voices alone, while the faces
 *  still load, would be a smaller number that then jumps. A list whose last read failed keeps the count it last had
 *  (its rows stay on screen), and one that never loaded counts as none, so one outage doesn't hide the other's count. */
export function waitingTotal(faces: WaitingRead, voices: WaitingRead): number | null {
  const n = (r: WaitingRead) => r.data?.count ?? (r.isError ? 0 : null);
  const f = n(faces);
  const v = n(voices);
  return f == null || v == null ? null : f + v;
}

/** What waitingTotal needs of one list's query: its count once read, and whether the read failed. */
export interface WaitingRead {
  data?: { count: number | null } | undefined;
  isError?: boolean;
}

/** The face groups waiting for a name, best guess first, how many, and the ones put aside. */
export const unknownGroupsQuery = () =>
  queryOptions({
    queryKey: UNKNOWN_GROUPS_KEY,
    queryFn: ({ signal }) => listed<UnknownGroup>('/faces/unknown/groups', signal),
    refetchInterval: query =>
      query.state.status !== 'error' && query.state.data?.pending
        ? 1000
        : query.state.status !== 'error' && (query.state.data?.shots_pending || query.state.data?.refreshing)
          ? 2000
          : 30_000,
    retry: false,
  });

/** The unnamed voices waiting for a name, best guess first, how many, and the ones put aside. */
export const unknownVoicesQuery = () =>
  queryOptions({
    queryKey: UNKNOWN_VOICES_KEY,
    queryFn: ({ signal }) => listed<UnknownVoice>('/voices/unknown/groups', signal),
    refetchInterval: query =>
      query.state.status !== 'error' && query.state.data?.pending
        ? 1000
        : query.state.status !== 'error' && query.state.data?.refreshing
          ? 2000
          : 30_000,
    retry: false,
  });

/** A face group or voice was named or put aside: take it, and one from the count, off the cached list at once, so
 *  every count of what waits (People's list, Live) drops before the list is read again. That read can take many
 *  seconds (a name makes the server work the face groups out afresh), and naming several in a row restarts it, so
 *  the counts stayed put until a pause (#2330). An in-flight read is cancelled first, since it would bring the old
 *  list back. One that isn't listed changes nothing; the next read has it right. */
export function dropWaiting(client: QueryClient, kind: 'face' | 'voice', id: number): void {
  const key = kind === 'face' ? UNKNOWN_GROUPS_KEY : UNKNOWN_VOICES_KEY;
  void client.cancelQueries({ queryKey: key });
  client.setQueryData<Listed<Waiting>>(key, old =>
    old?.groups.some(g => g.id === id)
      ? {
          ...old,
          groups: old.groups.filter(g => g.id !== id),
          count: old.count == null ? old.count : Math.max(0, old.count - 1),
        }
      : old,
  );
}

/** These face groups are not this person, or (`not` false) take that back (#2366): the guess drops them. A face
 *  group stays waiting, guessed afresh. {ok, error} rather than a throw. */
export async function markNotFace(client: QueryClient, name: string, groups: number[], not = true) {
  let r: { ok: boolean; error?: string | null };
  try {
    r = await request<{ ok: boolean; error?: string | null }>('/faces/not', { name, groups, shots: [], not });
  } catch (e) {
    r = { ok: false, error: (e instanceof HttpError ? e.reason : messageOf(e)) || 'That did not work' };
  }
  void client.invalidateQueries({ queryKey: UNKNOWN_GROUPS_KEY });
  return r;
}

/** One clip of a waiting voice was answered on its own ("Multiple speakers", or "Nobody"): take it off the cached list
 *  at once, and the voice too when it was its last clip, as the next read will (the patched answer drops it as well).
 *  Before, the clip stayed until the full list was worked out again, which can take a while. */
export function dropClip(client: QueryClient, voice: number, clip: { ts: number; line?: string }): void {
  // By its line's key; a clip from before keys (#247) by its time.
  const same = (c: { ts: number; line?: string }) => (clip.line ? c.line === clip.line : c.ts === clip.ts);
  void client.cancelQueries({ queryKey: UNKNOWN_VOICES_KEY });
  client.setQueryData<Listed<UnknownVoice>>(UNKNOWN_VOICES_KEY, old => {
    const g = old?.groups.find(v => v.id === voice);
    if (!old || !g?.clips.some(same)) return old;
    const clips = g.clips.filter(c => !same(c));
    if (clips.length) return { ...old, groups: old.groups.map(v => (v.id === voice ? { ...v, clips } : v)) };
    return {
      ...old,
      groups: old.groups.filter(v => v.id !== voice),
      count: old.count == null ? old.count : Math.max(0, old.count - 1),
    };
  });
}

/** Put a face group or a voice aside: it leaves "Who's this?" for good. {ok, error} rather than a throw. */
export async function ignore(client: QueryClient, kind: 'face' | 'voice', id: number) {
  let r: { ok: boolean; error?: string | null };
  try {
    r = await request<{ ok: boolean; error?: string | null }>('/inbox/skip', { kind, id });
  } catch (e) {
    r = { ok: false, error: (e instanceof HttpError ? e.reason : messageOf(e)) || 'Could not put this aside' };
  }
  if (r.ok) dropWaiting(client, kind, id);
  void client.invalidateQueries({ queryKey: kind === 'face' ? UNKNOWN_GROUPS_KEY : UNKNOWN_VOICES_KEY });
  return r;
}

export const ignoreFace = (client: QueryClient, id: number) => ignore(client, 'face', id);

/** One recorded picture of a face group. */
export interface Shot {
  id: string;
  ts: number;
  url: string;
}

/** A page of a face group's pictures, newest first, and how many it has in all (#2008). */
export const groupShots = (id: number, offset: number, limit: number, signal?: AbortSignal) =>
  request<{ sightings: Shot[]; total: number; shots_pending?: number; shots_unavailable?: number }>(
    `/faces/unknown/${id}/sightings?offset=${offset}&limit=${limit}`,
    undefined,
    signal,
  );

/** "Not this person": the pictures leave the group for good, and its face is rebuilt from the rest. A group left with
 *  none is put aside. */
export const rejectShots = (group: number, ids: string[]) =>
  request<{ ok: boolean; removed: number; left: number; ignored: boolean; error?: string }>('/faces/unknown/reject', {
    group,
    ids,
  });
