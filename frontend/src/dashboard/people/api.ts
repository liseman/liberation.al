// The People tab's server data (dashboard/server.py), ported from static/dashboard/tab-people.js, inbox.js, and
// voice-tools.js. A failed request throws the server's own message (shared/http.ts), which the tab shows as is.
import type { QueryClient } from '@tanstack/react-query';
import { HttpError, request } from '../../shared/http';
import { type LineRef, oneLine, someLines } from '../live/types';

type Opts = { signal?: AbortSignal };
const enc = encodeURIComponent;

/** "multiple speakers" is a line label, not a person: the server can list it, and the tab never shows it. */
export const MULTI = 'multiple speakers';

// ---- GET /people ----

/** One person on the roster. */
export interface Person {
  name: string;
  owner?: boolean;
  photo?: string | null;
  role?: string | null;
  organization?: string | null;
  lines?: number;
  room?: number; // of lines, those heard in the room
  calls?: number; // and those on calls
  aliases?: string[];
  enrolled?: boolean; // a voice learned from a recording
  voices: number[];
  faces?: number; // face samples
  last_seen?: number | null;
  last_seen_pending?: boolean;
  last_seen_error?: string | null;
  number?: string | null;
  last_heard?: number | null;
  last_seen_cam?: string | null; // the camera of last_seen
  last_seen_place?: string | null; // that camera's place name (#313), when the server sends one
  here_now?: { cam: string | null; place?: string | null; since?: number | null } | null; // their face in the live scene in the last 30 s, and since when
}

export interface UnknownFace {
  id: number;
  label: string;
  photo?: string | null;
  count?: number;
}

export interface UnknownVoice {
  id: number;
  label: string;
  last_seen?: number | null;
}

export interface VoiceClip {
  id: number;
  label: string;
  ts: number;
  text?: string;
  dur?: number | null;
}

export interface PeopleView {
  people: Person[];
  unknown: { voices: UnknownVoice[]; faces: UnknownFace[]; voice_clips?: VoiceClip[]; voice_clip_total?: number };
}

const EMPTY_VIEW: PeopleView = { people: [], unknown: { voices: [], faces: [], voice_clips: [], voice_clip_total: 0 } };

/** A /people reply (or a mutation reply that carries the view) in one shape. An old server sent a bare list. */
export function shape(data: unknown): PeopleView {
  if (Array.isArray(data)) return { ...EMPTY_VIEW, people: data as Person[] };
  const d = (data ?? {}) as Partial<PeopleView>;
  return {
    people: (d.people ?? []).filter(p => p.name !== MULTI),
    unknown: { ...EMPTY_VIEW.unknown, ...(d.unknown ?? {}) },
  };
}

export const PEOPLE_KEY = ['people'] as const;
export const getPeople = ({ signal }: Opts = {}) => request<unknown>('/people', undefined, signal).then(shape);

// ---- one person ----

export interface Profile {
  name: string;
  owner?: boolean;
  role?: string | null;
  organization?: string | null;
  voices: number[];
  enrolled?: boolean;
  faces?: number;
  last_seen?: number | null;
  last_seen_pending?: boolean;
  last_seen_error?: string | null;
  number?: string | null;
  aliases?: string[];
  lines?: number;
  room_lines?: number;
  call_lines?: number;
  last_heard?: number | null;
  memory?: string[];
  note?: string | null;
}

/** A profile from their row of the people list, to show at once while GET /people/<name> adds what only it has
 *  (memory notes): the way a pet's page shows from the Pets list (#2397). */
export function profileFromRow(p: Person): Profile {
  return {
    name: p.name,
    owner: p.owner,
    role: p.role,
    organization: p.organization,
    voices: p.voices,
    enrolled: p.enrolled,
    faces: p.faces,
    last_seen: p.last_seen,
    last_seen_pending: p.last_seen_pending,
    last_seen_error: p.last_seen_error,
    number: p.number,
    aliases: p.aliases,
    lines: p.lines,
    room_lines: p.room,
    call_lines: p.calls,
    last_heard: p.last_heard,
  };
}

/** After a change the server answered with its people view: show that list, and mark every other people query (the
 *  profiles and cards) out of date, since a rename, merge, or claim changes them too. A tab shown again reuses data
 *  under 30 s old (#330), so nothing else would read them again. */
export function tookView(client: QueryClient, reply: unknown): void {
  client.setQueryData(PEOPLE_KEY, shape(reply));
  void client.invalidateQueries({ queryKey: PEOPLE_KEY, predicate: q => q.queryKey.length > PEOPLE_KEY.length });
}

export const profileKey = (name: string) => ['people', 'profile', name] as const;
export const getProfile = (name: string, { signal }: Opts = {}) =>
  request<Profile>(`/people/${enc(name)}`, undefined, signal);

// ---- GET /people/<name>/card (#307): the contact card's photos, recent video, and recent voice ----

/** A face shot. id, cam, and score are null only for someone tagged live with no recorded sightings yet (their
 * newest saved crop stands in). score orders shots; it is never shown. */
export interface CardShot {
  id: string | null; // the detection's payload id (#549)
  url: string;
  ts: number;
  cam: string | null;
  score: number | null;
}
/** A recent stretch of video they're in. */
export interface CardVideo {
  id?: string; // "cam0@<start>", for corrections (#309)
  cam: string;
  place?: string | null; // the camera's place name (#313), when the server sends one
  start: number;
  end: number;
  duration_s: number;
  thumb: string | null; // a full-scene recorded frame, independent of face-photo quality
  what: string | null; // the journal entry for that time, else the camera caption
  review?: string;
}
/** A recent line they said. line is the transcript line's key (#306); treat it as opaque. */
export interface CardVoice {
  line: string;
  ts_start: number;
  text: string;
  mic: string | null;
}
export interface Card {
  name: string;
  photo: CardShot | null; // the main photo: the best-scoring shot
  shots: CardShot[]; // up to 6, best first, spread over days
  shots_pending?: number;
  shots_unavailable?: number;
  videos: CardVideo[]; // up to 4, newest first
  videos_pending?: number;
  videos_unavailable?: number;
  voices: CardVoice[]; // up to 4, newest first
  last_seen: { ts: number; cam: string | null; place?: string | null } | null; // the newest recorded sighting
  here_now: { cam: string | null; place?: string | null; since?: number | null } | null; // their face in the live scene in the last 30 s, and since when
  refreshing?: boolean; // an older card, answered as is while the server reads it again: ask again shortly (#2415)
}
export const cardKey = (name: string) => ['people', 'card', name] as const;
export const getCard = (name: string, { signal }: Opts = {}) =>
  request<Card>(`/people/${enc(name)}/card`, undefined, signal);

// ---- what nobody has named yet, and giving it to someone (#309) ----

/** An unknown face group: up to 4 photos, best first, and how often and when it was seen. */
export interface UnnamedFace {
  id: number;
  photos: string[];
  shots_pending?: number;
  shots_unavailable?: number;
  count: number;
  last_seen: number | null;
  cam: string | null;
  place?: string | null; // that camera's place name (#313)
}
/** An unnamed voice cluster, with one of its lines to play. */
export interface UnnamedVoice {
  id: number;
  label: string; // "Unnamed voice A"
  lines: number;
  last_heard: number | null;
  sample: { line: string; ts_start: number; text: string } | null;
}
/** A stretch of video on one camera where only unnamed faces were seen. Giving it to someone names those faces. */
export interface UnnamedVideo {
  id: string; // "cam0@<start>"
  cam: string;
  place?: string | null; // the camera's place name (#313)
  start: number;
  end: number;
  duration_s: number;
  thumb: string | null;
  what: string | null;
}
export interface Unnamed {
  shots_pending?: number;
  shots_unavailable?: number;
  videos_pending?: number;
  videos_unavailable?: number;
  incomplete?: string; // Available picks remain usable when face history could not be read.
  faces: UnnamedFace[];
  voices: UnnamedVoice[];
  videos: UnnamedVideo[];
}
export const UNNAMED_KEY = ['people', 'unnamed'] as const;
export const getUnnamed = ({ signal }: Opts = {}) => request<Unnamed>('/inbox/unnamed', undefined, signal, 'page');

export interface Picked {
  faces: number[];
  voices: number[];
  videos: string[];
}
/** What didn't stick, one entry per item. */
export interface Failed {
  kind: 'face' | 'voice' | 'video';
  id: number | string;
  error: string;
}
export type AssignReply = ViewReply & {
  assigned?: { faces: number; voices: number; videos: number };
  failed?: Failed[];
  undo?: string | null;
};
/** Several assignments at once, each to its own person, with one Undo for them all (Who's this?'s Yes to all). */
export type AssignManyReply = ViewReply & {
  assigned?: { name: string; faces: number; voices: number; videos: number }[];
  failed?: (Partial<Failed> & { name?: string; error: string })[];
  undo?: string | null;
};
export const assignMany = (items: ({ name: string } & Picked)[]) =>
  request<AssignManyReply>('/people/assign-many', { items });
/** Give what was picked to name (a new name makes the person). A 400 when nothing stuck. */
export const assign = (name: string, picked: Picked) => request<AssignReply>('/people/assign', { name, ...picked });

export type CorrectKind = 'shot' | 'video' | 'voice' | 'track'; // track: a person's box in Review, "<cam>:<track>@<ts>" (#2607)
export type Correction = { name: string; kind: CorrectKind; id: number | string } & (
  | { action: 'move'; to: string }
  | { action: 'remove' | 'main' | 'hide' }
);
export interface CorrectReply {
  ok: boolean;
  error?: string;
  undo?: string | null;
  to?: string; // a move's target, as the server spells it
  card?: Card;
}
/** A correction from the card's ⋯ menu: move to someone else, remove from this person, use as main photo, or hide. */
export const correct = (c: Correction) => request<CorrectReply>('/people/correct', c);

/** ok is false, with errors, when part of it couldn't be put back; people is the roster list. */
export type UndoReply = ViewReply & { card?: Card | null; errors?: string[] };
export const undoChange = (token: string) => request<UndoReply>('/people/undo', { undo: token });

export interface Lookup {
  error?: string;
  result?: string;
  note?: string;
  looked_up?: string;
}
export const lookUp = (name: string) => request<Lookup>(`/people/${enc(name)}/lookup`, {});

// ---- people writes ----

/** A reply that may carry the fresh people view. */
export type ViewReply = Partial<PeopleView> & { ok?: boolean; error?: string; name?: string; notes?: string[] };

export const addPerson = (name: string, number: string) => request<{ name: string }>('/people/add', { name, number });
export const claim = (kind: 'face' | 'voice', id: number, name: string) =>
  request<ViewReply>('/people/claim', { kind, id, name });
export const rename = (from: string, to: string) =>
  request<ViewReply & { merged?: boolean; restarted?: boolean }>('/people/rename', { from, to });
export type OwnerMergeReply = ViewReply & { confirmation?: boolean; voices?: number[]; merged?: boolean };
export async function mergeIntoOwner(name: string, voices?: number[]): Promise<OwnerMergeReply> {
  try {
    return await request<OwnerMergeReply>('/people/me', { name, ...(voices === undefined ? {} : { voices }) });
  } catch (e) {
    if (
      e instanceof HttpError &&
      e.status === 409 &&
      e.data &&
      typeof e.data === 'object' &&
      'confirmation' in e.data &&
      e.data.confirmation === true
    )
      return e.data as OwnerMergeReply;
    throw e;
  }
}

export const describe = (name: string, role: string, organization: string) =>
  request<{ ok: boolean; error?: string; name: string; role: string; organization: string }>('/people/describe', {
    name,
    role,
    organization,
  });
export const deletePerson = (name: string) =>
  request<ViewReply & { removed?: { voices?: number; faces?: number; photos?: number; sightings?: number } }>(
    '/people/delete',
    { name },
  );
export const unlabelLine = (line: LineRef) => request<{ ok: boolean }>('/transcript/unlabel', oneLine(line));
export const tagLines = (lines: readonly LineRef[], name: string) =>
  request<{ ok: boolean; queued?: number; note?: string | null }>('/speakers/tag', { ...someLines(lines), name });

// ---- Who's this? (../whosthis): labels, their log, and skips ----

export interface LoggedLabel {
  ts: number;
  ts_start: number;
  line?: string; // the line's key (#247); missing on labels given before keys
  text?: string;
  new?: { speaker?: string };
  old?: { speaker?: string };
}
export const REFINE_LOG_KEY = ['people', 'refine', 'log'] as const;
export const getRefineLog = ({ signal }: Opts = {}) =>
  request<{ labels: LoggedLabel[]; total: number }>('/speakers/refine/log', undefined, signal);

export const SKIPS_KEY = ['people', 'skips'] as const;
export const getSkips = ({ signal }: Opts = {}) =>
  request<{ face?: number; voice?: number; line?: number }>('/inbox/skips', undefined, signal);

export const refine = (
  body: { action: 'assign' | 'multi'; name?: string } & ({ line: string } | { ts_start: number }),
) => request<{ ok: boolean; error?: string; entry?: { speaker?: string } }>('/speakers/refine', body);
export const undoLabels = (body: { lines: string[] } | { ts_starts: number[] } | { since: number }) =>
  request<{ undone?: { ts_start: number; line?: string }[]; errors?: string[]; error?: string }>(
    '/speakers/refine/undo',
    body,
  );
export const skipItem = (body: { kind: 'face' | 'voice' | 'line'; id: number | string } | { kind: 'clear' }) =>
  request<{ ok: boolean; error?: string; skipped?: number }>('/inbox/skip', body);
export const faceSightings = (id: number, { signal }: Opts = {}) =>
  request<{ sightings: { id: string; ts: number; url: string }[] }>(
    `/faces/unknown/${id}/sightings`,
    undefined,
    signal,
  );

// ---- calls ----

export interface Call {
  day: string;
  dir: string;
  from?: string | null;
  started?: string;
  durationSeconds?: number;
  turns?: number;
  speakers?: Record<string, unknown>;
  // Saved per call by the bridge (#2684): which way it went ("unknown" for calls from before), who a placed call
  // was to (a person's name, or null for a typed number), the owner's instructions, and the assistant's last line on a
  // call that had them. `from` is the other party's number either way.
  direction?: 'in' | 'out' | 'unknown' | null;
  to?: { name?: string | null; number?: string | null } | null;
  instructions?: string | null;
  outcome?: string | null;
}
export const CALLS_KEY = ['calls'] as const;
export const getCalls = ({ signal }: Opts = {}) =>
  request<Call[]>('/calls', undefined, signal).then(d => (Array.isArray(d) ? d : []));
export const callDetail = (c: Call, { signal }: Opts = {}) =>
  request<{ transcript: { who?: string; t: number; text: string }[] }>(
    `/calls/detail?day=${enc(c.day)}&dir=${enc(c.dir)}`,
    undefined,
    signal,
  );
/** The call on now (GET /calls/live, #2697): which way, the other party's number, a name when the bridge knows one
 *  (the contact called, or the owner calling in), when it began, and its folder keys (as /calls and /calls/detail). */
export interface LiveCall {
  direction: 'in' | 'out' | 'unknown';
  number: string | null;
  name: string | null;
  since: number;
  day: string;
  dir: string;
}
export const LIVE_CALL_KEY = [...CALLS_KEY, 'live'] as const;
export const getLiveCall = ({ signal }: Opts = {}) =>
  request<{ live: LiveCall | null; count: number }>('/calls/live', undefined, signal);
/** End only the selected live record; the server resolves and verifies its provider identity. */
export const hangupCall = ({ day, dir }: Pick<LiveCall, 'day' | 'dir'>) =>
  request<{ ok: boolean; detail: string }>('/calls/hangup', { day, dir, confirmed: true });
export const callAudio = (c: Call) => `/calls/audio?day=${enc(c.day)}&dir=${enc(c.dir)}&which=mixed`;
export const placeCall = (name: string, message: string, number: string) =>
  request<{ ok: boolean; error?: string; detail?: string }>('/calls/place', { name, message, number, confirmed: true });

// ---- voice clones ----

export interface Clone {
  id: number;
  label: string;
  clean_s?: number; // heard: every line of theirs that could go into a reference
  usable_s?: number; // usable for cloning: lines the owner named, plus matched lines that passed the voice check
  confirmed_s?: number; // of that, the lines the owner named
  unchecked_s?: number; // matched lines not checked yet (building checks them)
  rejected_s?: number; // matched lines that didn't pass the voice check at the last build
  needs_s?: number; // the least a reference takes
  ready: boolean; // enough heard to try a build
  built: boolean;
}
export const CLONES_KEY = ['clones'] as const;
export const getClones = ({ signal }: Opts = {}) =>
  request<{ clones: Clone[]; error?: string | null }>('/clones', undefined, signal);
export const buildClone = (id: number) =>
  request<{ ok: boolean; utterances?: number; seconds?: number }>('/clones/build', { id });
export const sayClone = (id: number, text: string) =>
  request<{ ok: boolean; error?: string; url: string }>('/clones/say', { id, text });
export const removeClone = (id: number) => request<{ ok: boolean; removed?: string[] }>('/clones/remove', { id });
export const cloneAudio = (id: number) => `/clones/audio?id=${id}&t=${Date.now()}`;

/** One of a person's recent lines that could go into their clone's reference (#312). */
export interface PickableLine {
  line: string; // the line's key (#247)
  ts_start: number;
  text: string;
  dur_s: number;
  confirmed: boolean; // the owner already said this line is theirs
}
export const getPickable = (id: number, { signal }: Opts = {}) =>
  request<{ lines: PickableLine[] }>(`/clones/${id}/lines`, undefined, signal);
/** Make a voice's clone reference from the lines picked, in the order given (POST /clones/pick). */
export const pickClips = (id: number, lines: string[]) =>
  request<{ ok: boolean; seconds?: number; clips?: number; ref_text?: string }>('/clones/pick', { id, lines });

// ---- clips ----

/** A line's audio: by its key (#247), or by its ts_start as before keys. */
export const lineClip = (line: LineRef) =>
  typeof line === 'string' ? `/clip?line=${encodeURIComponent(line)}` : `/clip?ts_start=${line}`;
export const voiceClip = (id: number) => `/speaker/${id}/latest-clip?t=${Date.now()}`;
export const latestOf = (id: number, { signal }: Opts = {}) =>
  request<{ ts_start?: number }>(`/speaker/${id}/latest`, undefined, signal);

// ---- Why this name (#288) ----

/** Why a line or a recorded face has its name (GET /identity/why, dashboard/why_name.py). */
export interface Why {
  name: string;
  subject: 'line' | 'shot';
  // line or shot: something was said about this very one; voice: the line follows its voice's name; face: the face
  // matched when it was seen; heard: nothing stands, so it reads as it was heard
  via: 'line' | 'shot' | 'voice' | 'face' | 'detached' | 'hand' | 'heard' | 'none'; // the rule that named it
  basis: string | null; // the deciding assertion's basis (domain/identity.py Basis), when one decided
  asserted_at: number | null;
  source: string | null;
  confidence: number | null;
  similarity: number | null; // how closely the voice or face matched when heard or seen (cosine)
  threshold: number; // what a match needs
  voice: { id: number; label: string | null } | null;
  mark: Record<string, unknown>; // what POST /measurement/mark needs, less shown and right (#224)
}
export type WhyTarget = { line: LineRef } | { shot: string };
export const whyKey = (t: WhyTarget) => ['identity', 'why', 'line' in t ? String(t.line) : `shot:${t.shot}`] as const;
export const getWhy = (t: WhyTarget, { signal }: Opts = {}) =>
  request<Why>(
    `/identity/why?${'line' in t ? new URLSearchParams(Object.entries(oneLine(t.line)).map(([k, v]) => [k, String(v)])) : `shot=${encodeURIComponent(t.shot)}`}`,
    undefined,
    signal,
  );
/** Mark the name shown on a line or a face right or wrong, for the measurement set (#224). It changes no name. */
export const markName = (why: Why, right: boolean) =>
  request<{ ok: boolean; error?: string }>('/measurement/mark', { ...why.mark, shown: why.name, right });
