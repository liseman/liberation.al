// The Pets tab's data (#1332, #2019): named pets come from /pets (../petLinks), their sightings, the groups still
// waiting for a name, and each species' recognition check from /pets/sightings (people/pet_recognition.py).

import { queryOptions } from '@tanstack/react-query';
import { request } from '../../../shared/http';
import { capitalize, hm } from '../../lib/format';
import type { Pet, PetsView } from '../petLinks';

export type { Pet, PetsView };

export const OTHER = 'other'; // the label for an animal that is none of the named pets
export const IGNORED = 'ignored'; // put aside: off the list, and never an example for recognition (#2035)

export interface Guess {
  label: string;
  similarity: number;
  clear?: boolean; // a suggestion that is a clear match, while recognition is off for its species (#2291)
}
/** The guess every naming list shares (faces', voices', and pets', #2366): `id` is the label to send back. */
export interface NamingGuess {
  id: string;
  name: string;
  score: number; // 0 to 100
  band: 'probable' | 'maybe' | 'none';
  close: string | null;
}
export interface Sighting {
  track_id: string;
  species: string;
  cam: string;
  start: number;
  end: number;
  samples: number;
  label: string | null;
  photo_ready?: boolean;
  thumb_ready?: boolean;
  recognized?: Guess;
  suggestion?: Guess;
  not?: string[]; // the pets the owner said it isn't (#2364): never guessed as them again
  guess?: NamingGuess | null;
}
/** A named pet's sighting: one the owner labeled, or one recognition named. */
export interface Seen {
  track_id: string;
  cam: string;
  start: number;
  end: number;
  how: 'labeled' | 'recognized';
  photo_ready?: boolean;
  thumb_ready?: boolean;
  video_status?: 'ready' | 'pending' | 'unavailable';
  video_thumb?: string | null;
}
export interface Check {
  species: string;
  labeled: number;
  right: number;
  wrong: number;
  unknown: number;
  on: boolean;
  reason: string | null;
}
export interface SightingGroup {
  id: string;
  species: string;
  sightings: Sighting[];
  guess?: NamingGuess | null; // what its sightings name most (#2366)
  not?: string[]; // the pets every sighting in it was said not to be
  photos_pending?: number;
  photos_unavailable?: number;
}
export interface SightingsView {
  photos_pending?: number;
  videos_pending?: number;
  photos_unavailable?: number;
  seen_pending?: Record<string, number>;
  seen_unavailable?: Record<string, number>;
  label_days?: number;
  seen?: Record<string, Seen[]>; // each named pet's newest sightings, newest first
  groups: SightingGroup[];
  count?: number; // the groups waiting for a name
  recognition: Check[];
  names: Record<string, string>;
}

/** The longest a card of one pet's sightings spans, first to last (Known design pass): a short window, so a pet seen
 *  all day is still a card an hour, not one card for the whole day. */
export const PET_WINDOW_S = 60 * 60;

const BAND_RANK = { probable: 2, maybe: 1, none: 0 } as const;

/** One card per pet per short window (Known design pass): groups of one species that the server guessed as the same
 *  pet, all within PET_WINDOW_S first to last, become one group, so "Probably Bess, 1 sighting" doesn't come six times in a
 *  morning, or twice in one minute from two cameras. A group with no guess stays as it is: it could be any animal. The
 *  merged guess is the least sure of its parts (lower score, weaker band), so merging never makes a card look surer.
 *  Groups keep the server's order (newest first); a merged one takes its newest part's place and id. */
export function byWindow(groups: readonly SightingGroup[]): SightingGroup[] {
  const out: SightingGroup[] = [];
  const open = new Map<string, { group: SightingGroup; first: number; last: number }>(); // per species and pet
  for (const g of groups) {
    const starts = g.sightings.map(s => s.start);
    const ends = g.sightings.map(s => s.end);
    const first = Math.min(...starts);
    const last = Math.max(...ends);
    const key = g.guess && g.guess.band !== 'none' ? `${g.species}\u0000${g.guess.id}` : null;
    const joined = key ? open.get(key) : undefined;
    const span = joined ? Math.max(joined.last, last) - Math.min(joined.first, first) : Number.POSITIVE_INFINITY;
    if (key && joined && span <= PET_WINDOW_S && joined.group.guess && g.guess) {
      const into = joined.group;
      const a = into.guess as NamingGuess;
      const b = g.guess;
      into.sightings = [...into.sightings, ...g.sightings].sort((x, y) => y.start - x.start);
      into.guess = {
        ...a,
        score: Math.min(a.score, b.score),
        band: BAND_RANK[a.band] <= BAND_RANK[b.band] ? a.band : b.band,
      };
      into.not = (into.not ?? []).filter(n => (g.not ?? []).includes(n));
      into.photos_pending = (into.photos_pending ?? 0) + (g.photos_pending ?? 0) || undefined;
      into.photos_unavailable = (into.photos_unavailable ?? 0) + (g.photos_unavailable ?? 0) || undefined;
      joined.first = Math.min(joined.first, first);
      joined.last = Math.max(joined.last, last);
      continue;
    }
    const copy = { ...g, sightings: [...g.sightings] };
    out.push(copy);
    if (key) open.set(key, { group: copy, first, last });
  }
  return out;
}

/** How many cards wait for a name: the merged groups when the server listed all of them, else its own count. */
export const waitingGroups = (view: SightingsView | undefined): number | undefined =>
  !view ? undefined : view.count != null && view.count > view.groups.length ? view.count : byWindow(view.groups).length;

export const SIGHTINGS_KEY = ['pets', 'sightings'] as const;
export const getSightings = (signal?: AbortSignal) => request<SightingsView>('/pets/sightings', undefined, signal);
/** The sightings waiting for a name, read again every 2 s while their pictures are still being made. Pets lists them
 *  and Known's Pets badge counts them (#2451). */
export const sightingsQuery = () =>
  queryOptions({
    queryKey: SIGHTINGS_KEY,
    queryFn: ({ signal }) => getSightings(signal),
    refetchInterval: query =>
      query.state.status !== 'error' && (query.state.data?.photos_pending || query.state.data?.videos_pending)
        ? 2000
        : false,
  });
export const labelSightings = (tracks: string[], label: string | null) =>
  request<{ labeled: number }>('/pets/sightings/label', { track_ids: tracks, label });
export const savePet = (pet: { id: string | null; name: string; species: string; sex: string | null }) =>
  request<{ pets: Pet[] }>('/pets/save', pet);
export const removePet = (id: string) => request<{ pets: Pet[] }>('/pets/remove', { id });
/** These sightings are not this pet, or (`not` false) take that back (#2364). One labeled as the pet loses the label. */
export const markNot = (tracks: string[], pet: string, not = true) =>
  request<{ marked: number }>('/pets/sightings/not', { track_ids: tracks, pet, not });
/** A pet card's photo choices (#2363): use a sighting as its photo, go back to its newest, or hide or show one. */
export const choosePhoto = (id: string, action: 'main' | 'newest' | 'hide' | 'show', track?: string) =>
  request<{ pets: Pet[] }>('/pets/photo', { id, action, track: track ?? null });

/** The sighting a pet's photo shows (#2363): the one chosen on its card, else its newest whose picture is ready and
 *  that isn't hidden from the card; undefined while there's none. `large`: the card's full-size photo, which needs the
 *  saved crop, where an avatar needs only its thumbnail. */
export function petPhoto(pet: Pet, seen: Seen[] | undefined, large = false): string | undefined {
  if (pet.photo) return pet.photo;
  const hidden = new Set(pet.hidden ?? []);
  const ready = (s: Seen) => (large ? s.photo_ready : (s.thumb_ready ?? s.photo_ready)) !== false;
  return seen?.find(s => ready(s) && !hidden.has(s.track_id))?.track_id;
}

export const photoUrl = (track: string, grid = false) =>
  `/pets/sightings/photo?track=${encodeURIComponent(track)}${grid ? '&size=80' : ''}`;

export const CLIP_MIN_S = 5; // a sighting of one look is played as this long a clip, so there's something to see

/** Saved clips among a pet's newest sightings; unprepared thumbnails never enter the card. */
export function petVideos(seen: Seen[], max = 4) {
  return seen.slice(0, max).flatMap(s => {
    if (s.video_status !== 'ready' || !s.video_thumb) return [];
    const end = Math.max(s.end, s.start + CLIP_MIN_S);
    return [
      {
        cam: s.cam,
        start: s.start,
        end,
        duration_s: Math.round((end - s.start) * 10) / 10,
        thumb: s.video_thumb,
        what: null,
      },
    ];
  });
}

/** "Nov 6, 11:09": the day, and the time in 24 hours as the rest of the app says it (lib/format). */
const when = (seconds: number) =>
  `${new Date(seconds * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}, ${hm(seconds)}`;
const clock = hm;

/** A span in few words: one time when it is within a minute, the day once when it is one day. */
export function span(first: number, last: number) {
  if (when(first) === when(last)) return when(first);
  if (new Date(first * 1000).toDateString() === new Date(last * 1000).toDateString()) {
    return `${when(first)} to ${clock(last)}`;
  }
  return `${when(first)} to ${when(last)}`;
}

export const species = capitalize;

/** Who a label names, in a sentence: a pet's name, "another animal", or "a removed pet". */
export function nameOf(label: string, names: Record<string, string>) {
  return label === OTHER ? 'another animal' : (names[label] ?? 'a removed pet');
}
