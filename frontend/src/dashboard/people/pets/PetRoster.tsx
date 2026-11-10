// The Pets list, left of a pet's card, as People's roster is: a count, search, "New pet", how many sighting
// groups wait for a name (Pets' "Who's this?"), then every named pet with its photo and where it was last seen.
import { Button, Group, Text } from '@mantine/core';
import { useState } from 'react';
import { plural } from '../../lib/format';
import { EntityAvatar } from '../entity';
import { whereabouts } from '../profile/role';
import { RosterColumn, RosterRow } from '../RosterColumn';
import { type Pet, petPhoto, photoUrl, type Seen, species as Species } from './api';

/** A pet's avatar photo: the one chosen on its card, else its newest ready one (petPhoto), as a thumbnail. */
const petAvatar = (pet: Pet, seen: Seen[] | undefined): string | null => {
  const track = petPhoto(pet, seen);
  return track ? photoUrl(track, true) : null;
};

/** "Cat, last seen Nov 3, 8:50 PM by cam0", or "Cat, not seen yet". */
export function petMeta(pet: Pet, seen: Seen[] | undefined): string {
  const newest = seen?.[0];
  const where = newest ? whereabouts({ last_seen: newest.end, last_seen_cam: newest.cam }) : '';
  const what = [Species(pet.species), pet.sex].filter(Boolean).join(' · ');
  return `${what}, ${where ? where.charAt(0).toLowerCase() + where.slice(1) : 'not seen yet'}`;
}

export function PetRoster({
  pets,
  seen,
  waiting,
  loaded,
  error,
  errorDetail = null,
  retrying,
  onRetry,
  only,
  onShowAll,
  sel,
  onOpen,
  onNew,
}: {
  pets: Pet[];
  seen: Record<string, Seen[]>;
  waiting: number | null; // sighting groups waiting for a name; null while unknown
  loaded: boolean;
  error: string | null;
  errorDetail?: string | null; // the server's reason, behind Show details
  retrying: boolean;
  onRetry: () => void;
  only?: string; // one species only (a link from an animal's box)
  onShowAll: () => void;
  sel: string | null; // a pet's id, '?' for the sightings waiting, or null
  onOpen: (sel: string) => void; // a click, Enter, or I on a row
  onNew: () => void;
}) {
  const [query, setQuery] = useState('');
  const q = query.trim().toLowerCase();
  const shown = pets.filter(p => (!only || p.species === only) && (!q || p.name.toLowerCase().includes(q)));
  const label =
    waiting == null
      ? 'Sightings to name'
      : waiting
        ? `${plural(waiting, 'animal', 'animals')} ${waiting === 1 ? 'needs' : 'need'} a name`
        : 'Sightings to name';
  return (
    <RosterColumn
      title="Pets"
      titleId="pets-h"
      noun="pets"
      count={pets.length}
      query={query}
      onQuery={setQuery}
      newLabel="New pet"
      onNew={onNew}
      waiting={label}
      waitingCurrent={sel == null || sel === '?'}
      onWaiting={() => onOpen('?')}
      notice={
        only && (
          <Group gap="xs" mt="xs">
            <Text size="sm" role="status">
              Showing {only} only.
            </Text>
            <Button size="compact-sm" variant="subtle" onClick={onShowAll}>
              Show all
            </Button>
          </Group>
        )
      }
      loaded={loaded}
      error={error}
      errorDetail={errorDetail}
      retrying={retrying}
      onRetry={onRetry}
      empty={{
        title: only ? `No named ${only} yet` : 'No named pets yet',
        text: 'Add one with New pet, or name one in Sightings to name.',
      }}
      noMatch={{ title: `No pet matches “${query.trim()}”`, text: 'Check the spelling, or add one with New pet.' }}
      groups={[
        {
          title: 'Everyone',
          list: 'Named pets',
          rows: shown.map(p => (
            <RosterRow
              key={p.id}
              avatar={<EntityAvatar name={p.name} photo={petAvatar(p, seen[p.id])} />}
              label={p.name}
              description={petMeta(p, seen[p.id])}
              current={sel === p.id}
              onOpen={() => onOpen(p.id)}
            />
          )),
        },
      ]}
    />
  );
}
