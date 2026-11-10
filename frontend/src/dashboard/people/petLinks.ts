// Where an animal word goes (#1986): a taught pet's name or a species opens Known's Pets tab, at that pet or for that
// species, never Objects. Pets.tsx and Live's large view share it, and the Pets query is here so both read one cache.
import { request } from '../../shared/http';

export interface Pet {
  id: string;
  name: string;
  species: string;
  sex?: 'female' | 'male' | null;
  photo?: string | null; // the sighting (its track) chosen as its photo on its card (#2363); none: its newest
  hidden?: string[]; // the sightings hidden from its card (#2363)
}

/** A named pet a camera sees now (#2606): where, since when, and its newest track there (a photo of it now). */
export interface PetHere {
  cam: string;
  camera_name: string; // the camera's name (vision.camera_names), else its id
  since: number;
  track: string;
}

export interface PetsView {
  pets: Pet[];
  species: string[];
  here?: Record<string, PetHere>; // by pet id
}

export const PETS_KEY = ['pets'] as const;

export const getPets = (signal?: AbortSignal) => request<PetsView>('/pets', undefined, signal);

/** The animal classes the detector reports (people/pets.py SPECIES), for a label read before /pets answers. */
const DETECTED = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'];

/** A word as it compares: lowercased and trimmed, without a plural s ("dogs" is "dog"). */
const word = (s: string): string => {
  const w = s.trim().toLowerCase();
  return w.length > 3 && w.endsWith('s') ? w.slice(0, -1) : w;
};

/** The Pets link for a label, or null when it names no pet and no species: `/pets#pet=<name>` for a taught pet (its
 *  name, whatever the case), else `/pets#species=<species>` for an animal's species. */
export function petHref(label: string, view: PetsView | undefined): string | null {
  const pet = view?.pets.find(p => p.name.trim().toLowerCase() === label.trim().toLowerCase());
  if (pet) return `/pets#pet=${encodeURIComponent(pet.name)}`;
  const species = new Set([...DETECTED, ...(view?.species ?? [])]);
  const found = word(label);
  return species.has(found) ? `/pets#species=${encodeURIComponent(found)}` : null;
}

/** A link to Pets' sightings waiting for a name (the command palette's "Sightings to name"). */
export const SIGHTINGS_HASH = '#sightings';

/** What a Pets link asks for: one pet by name, or one species; nothing for any other hash. */
export function petInHash(hash: string): { pet?: string; species?: string } {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const pet = params.get('pet')?.trim();
  if (pet) return { pet };
  const species = params.get('species')?.trim().toLowerCase();
  return species ? { species } : {};
}
