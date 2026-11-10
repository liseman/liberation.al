// Walk coverage on Known › Places (#3): a place registered only from fixed cameras has a sparse cloud and a rough
// plan, and a LiDAR walkthrough gives it metric scale and full walls. GET /places says how many walks went into each
// place (`walks`) and which of its plan's rooms no LiDAR walk covered (`unwalked_rooms`, null when it can't tell), so
// the list marks a place that needs a walk, and the place's panel says which rooms. A walk is dropped at the top of the page to import it (WalkImport.tsx).
import { Badge, Text } from '@mantine/core';
import { plural } from '../lib/format';
import type { Place } from './examples';

/** The how-to for recording a walk (docs/WALKTHROUGH.md). */
export const WALKTHROUGH_DOC = 'https://github.com/liseman/liberation.al/blob/main/docs/WALKTHROUGH.md';

/** What a place still needs walked: the whole place (never walked), some rooms, or nothing (null). The examples and
 *  a walked place whose rooms can't be told need nothing shown. */
export type WalkNeed = { kind: 'place' } | { kind: 'rooms'; rooms: string[] } | null;

export function walkNeed(place: Place): WalkNeed {
  if (place.example) return null;
  if (!place.walks) return { kind: 'place' };
  const rooms = place.unwalkedRooms ?? [];
  return rooms.length ? { kind: 'rooms', rooms: rooms.map(r => r.name) } : null;
}

/** Names as a phrase: "Den", "Den and Hall", "Den, Hall, and Study". */
export function namesWords(names: readonly string[]): string {
  if (names.length <= 2) return names.join(' and ');
  return `${names.slice(0, -1).join(', ')}, and ${names.at(-1)}`;
}

/** The list row's mark: "Not walked", or how many rooms aren't. */
export function WalkBadge({ place }: { place: Place }) {
  const need = walkNeed(place);
  if (!need) return null;
  return (
    <Badge size="xs" variant="light" color="yellow" tt="none" fw={500}>
      {need.kind === 'place' ? 'Not walked' : `${plural(need.rooms.length, 'room', 'rooms')} not walked`}
    </Badge>
  );
}

/** The place panel's note: what isn't walked yet. The import itself, and the link to how to record a walk, are
 *  WalkImport's, at the top of the page. */
export function WalkNote({ place }: { place: Place }) {
  const need = walkNeed(place);
  if (!need) return null;
  return (
    <div role="note" aria-label="Walkthrough" style={{ marginBottom: 8 }}>
      <Text size="sm">
        {need.kind === 'place'
          ? 'This place has never been walked. A LiDAR walkthrough gives it metric scale and full walls.'
          : `Not walked yet: ${namesWords(need.rooms)}. A LiDAR walkthrough through ${need.rooms.length === 1 ? 'it' : 'them'} fills in the walls the cameras can't see.`}
      </Text>
    </div>
  );
}
