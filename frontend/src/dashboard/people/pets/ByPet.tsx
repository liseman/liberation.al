// Pets' "By pet": the naming inbox's bulk view (../../naming/Bulk) over every waiting sighting recognition or
// a suggestion names as a pet. The checked ones are labeled as the pet; the unchecked ones are marked not that pet,
// so they stop being guessed as it and keep waiting.
import { Bulk, type BulkTarget } from '../../naming/Bulk';
import { cameraName } from '../profile/role';
import { type Pet, photoUrl, type Sighting, type SightingGroup, span } from './api';

export const LABEL_MAX = 500; // the most sightings /pets/sightings/label takes in one call

/** The pet a sighting's guess names, with how alike it looks (0 to 1): the server's shared guess, else a
 *  recognition match, else a suggestion. */
function guessed(s: Sighting): { label: string; similarity: number } | undefined {
  if (s.guess !== undefined)
    return s.guess && s.guess.band !== 'none' ? { label: s.guess.id, similarity: s.guess.score / 100 } : undefined;
  return s.recognized ?? s.suggestion;
}

/** The waiting sightings guessed as each pet, best match first. */
export function byPet(groups: SightingGroup[]): Map<string, Sighting[]> {
  const out = new Map<string, Sighting[]>();
  for (const s of groups.flatMap(g => g.sightings)) {
    const g = guessed(s);
    if (g) out.set(g.label, [...(out.get(g.label) ?? []), s]);
  }
  for (const list of out.values()) list.sort((a, b) => (guessed(b)?.similarity ?? 0) - (guessed(a)?.similarity ?? 0));
  return out;
}

export function ByPet({
  groups,
  pets,
  onSave,
  thumb = false,
}: {
  groups: SightingGroup[];
  pets: Pet[];
  onSave: (pet: string, checked: string[], unchecked: string[]) => void;
  thumb?: boolean; // the small picture size: load the thumbnail
}) {
  const found = byPet(groups);
  const targets: BulkTarget[] = pets.map(p => ({
    id: p.id,
    name: p.name,
    tiles: (found.get(p.id) ?? []).map(s => {
      const score = Math.round((guessed(s)?.similarity ?? 0) * 100);
      return {
        key: s.track_id,
        src: photoUrl(s.track_id, thumb),
        label: `${s.species} on ${cameraName(s.cam)}, ${span(s.start, s.end)}, match ${score}%`,
        caption: `${cameraName(s.cam)} · ${span(s.start, s.end)} · ${score}%`,
      };
    }),
  }));
  return (
    <Bulk
      label="By pet"
      which="Which pet"
      noun="sightings"
      storageKey="pets.byPet.pet"
      targets={targets}
      max={LABEL_MAX}
      onSave={(target, checked, unchecked) => onSave(target.id, checked, unchecked)}
    />
  );
}
