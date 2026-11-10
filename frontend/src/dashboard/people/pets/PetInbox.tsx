// Pets' "Sightings to name": every group of animal sightings nobody has named yet, in the shared naming
// inbox (../../naming): sorted and sized there, one NamingRow each, with its newest picture large, when and
// where it was seen, the best guess, and the answers: "Yes, Mia" for a guess (filled for a Probably, outlined for a
// Maybe), "Not Mia", which tells recognition that group isn't her, and the name picker,
// a Cascader of a species (the detected one first), then that species' pets and Other… for a new pet (naming a pet of
// another species corrects the sighting's). "Show all" opens every picture of the group; selecting some and "Not this
// pet" leaves them out, so naming the group names only the rest (they wait again on their own). "By pet"
// labels one pet's guessed sightings in bulk.
import { Button, Cascader, Group, Text } from '@mantine/core';
import { IconPhotoMinus, IconPhotoX, IconUserMinus } from '@tabler/icons-react';
import { useState } from 'react';
import { ProcessingImages } from '../../../shared/ProcessingImages';
import { SafeImage } from '../../../shared/SafeImage';
import { capitalize, plural } from '../../lib/format';
import { LoadError, PageLoading } from '../../lib/page';
import { SelectableGrid } from '../../lib/SelectableGrid';
import { type InboxGuess, NamingInbox, useLeaving } from '../../naming/NamingInbox';
import { NamingRow } from '../../naming/NamingRow';
import '../../whosthis/whosthis.css';
import who from '../../whosthis/whosthis.module.css';
import { cameraName } from '../profile/role';
import { type Guess, IGNORED, nameOf, OTHER, type Pet, photoUrl, type Sighting, type SightingGroup, span } from './api';
import { ByPet } from './ByPet';

const title = capitalize;

/** The group's best guess: the pet that clear matches (recognition's, or a clear suggestion while recognition is off
 *  for the species) name most often, else the one suggestions name most often. Its score is the mean
 *  similarity of the sightings that name it, as a percentage: how alike they look, not how many agree. */
export function guessOf(sightings: Sighting[]): { label: string; sure: boolean; score: number } | null {
  const tally = (pick: (s: Sighting) => Guess | undefined) => {
    const found = new Map<string, number[]>();
    for (const s of sightings) {
      const g = pick(s);
      if (g) found.set(g.label, [...(found.get(g.label) ?? []), g.similarity]);
    }
    const [label, sims] = [...found].sort((a, b) => b[1].length - a[1].length)[0] ?? [];
    return label && sims ? { label, score: Math.round((100 * sims.reduce((x, y) => x + y, 0)) / sims.length) } : null;
  };
  const sure = tally(s => s.recognized ?? (s.suggestion?.clear ? s.suggestion : undefined));
  if (sure) return { ...sure, sure: true };
  const maybe = tally(s => s.suggestion);
  return maybe ? { ...maybe, sure: false } : null;
}

/** A group's guess as the inbox shows it: the server's for the whole group, else worked out here from the
 *  sightings left in it (after "Not this pet", or when the response omits a group guess). Only a named pet is a
 *  guess. */
export function groupGuess(
  group: SightingGroup,
  sightings: Sighting[],
  names: Record<string, string>,
): InboxGuess | null {
  const whole = sightings.length === group.sightings.length && group.guess !== undefined;
  const guess = whole
    ? group.guess && group.guess.band !== 'none'
      ? { id: group.guess.id, score: group.guess.score, band: group.guess.band }
      : null
    : (() => {
        const g = guessOf(sightings);
        return g ? { id: g.label, score: g.score, band: g.sure ? ('probable' as const) : ('maybe' as const) } : null;
      })();
  if (!guess || guess.id === OTHER || !names[guess.id]) return null;
  return { ...guess, name: nameOf(guess.id, names) };
}

const OTHER_PREFIX = 'other:'; // Other… under a species: a new pet of it

/** "Who's this?", Mantine's Cascader: the species in the first column (the detected one on top), each
 *  opening its named pets and Other… for a new one in the next. Both columns show at once, so a group the detector
 *  got wrong is a different species away. Up and Down move, Right opens a species, Left goes back, Enter picks, and
 *  Esc closes, as the Cascader does. */
export function NamePicker({
  detected,
  speciesList,
  pets,
  guess,
  onPet,
  onOther,
}: {
  detected: string;
  speciesList: string[];
  pets: Pet[];
  guess?: { id: string; score: number };
  onPet: (pet: Pet) => void;
  onOther: (species: string) => void;
}) {
  // The detected species first when a pet can be one; a sighting of "some animal" (a zebra) gets the list.
  const first = speciesList.includes(detected) ? [detected] : [];
  const data = [...first, ...speciesList.filter(s => s !== detected)].map(s => ({
    value: s,
    label: `${title(s)}${s === detected ? ' (detected)' : ''}`,
    children: [
      ...pets
        .filter(p => p.species === s)
        .sort((a, b) => (a.id === guess?.id ? -1 : b.id === guess?.id ? 1 : 0))
        .map(p => ({ value: p.id, label: p.id === guess?.id ? `${p.name} (guess, ${guess.score}%)` : p.name })),
      { value: `${OTHER_PREFIX}${s}`, label: 'Other…' },
    ],
  }));
  return (
    <Cascader
      size="xs"
      w="100%"
      aria-label="Who's this? Which pet is this?"
      placeholder="Who's this?"
      data={data}
      value={null}
      columnWidth={150}
      styles={{ columnOptionLabel: { whiteSpace: 'normal', overflowWrap: 'anywhere' } }}
      comboboxProps={{ width: 'max-content', position: 'bottom-start' }}
      onChange={path => {
        const pick = path?.at(-1);
        if (!pick) return;
        if (pick.startsWith(OTHER_PREFIX)) onOther(pick.slice(OTHER_PREFIX.length));
        else {
          const pet = pets.find(p => p.id === pick);
          if (pet) onPet(pet);
        }
      }}
    />
  );
}

function GroupRow({
  group,
  pets,
  names,
  speciesList,
  onLabel,
  onNot,
  onNewPet,
  thumb,
}: {
  group: SightingGroup;
  pets: Pet[];
  names: Record<string, string>;
  speciesList: string[];
  onLabel: (tracks: string[], label: string) => void;
  onNot?: (tracks: string[], pet: string) => void;
  onNewPet: (species: string, then: (pet: Pet) => void) => void;
  thumb: boolean; // the small size: load the thumbnail
}) {
  const leaving = useLeaving();
  const [all, setAll] = useState(false);
  const [left, setLeft] = useState<ReadonlySet<string>>(() => new Set()); // "Not this pet": out of this naming
  const [picked, setPicked] = useState<ReadonlySet<string>>(() => new Set());
  const kept = group.sightings.filter(s => !left.has(s.track_id));
  const photos = kept.filter(s => (s.thumb_ready ?? s.photo_ready) !== false);
  const lead = kept.find(s => s.photo_ready !== false); // the large picture is the saved crop, not the thumbnail
  const pending = group.photos_pending ?? kept.filter(s => (s.thumb_ready ?? s.photo_ready) === false).length;
  const unavailable = group.photos_unavailable ?? 0;
  const guessed = groupGuess(group, kept, names);
  const label = plural(kept.length, `${group.species} sighting`, `${group.species} sightings`);
  if (!kept.length) return null;
  const first = Math.min(...kept.map(s => s.start));
  const last = Math.max(...kept.map(s => s.end));
  const cams = [...new Set(kept.map(s => cameraName(s.cam)))].join(', ');
  const tracks = kept.map(s => s.track_id);
  const name = (value: string) => onLabel(tracks, value);
  const toggle = (track: string) =>
    setPicked(have => {
      const next = new Set(have);
      if (!next.delete(track)) next.add(track);
      return next;
    });
  const score = guessed?.score ?? 0;
  return (
    <NamingRow
      rowKey={group.id}
      label={label}
      lead={lead ? { src: photoUrl(lead.track_id, thumb), alt: `${group.species} on ${cameraName(lead.cam)}` } : null}
      guess={guessed}
      pictureAnswers={[
        ...(lead && guessed && onNot
          ? [
              {
                label: `Not ${guessed.name}: just this picture`,
                caption: 'Not pic',
                hint: `Just the large picture isn't ${guessed.name}. It's guessed again without ${guessed.name}; the rest of the group keeps its guess. Undo takes it back.`,
                icon: <IconPhotoMinus size={16} aria-hidden />,
                shortcut: 'P',
                onClick: () => onNot([lead.track_id], guessed.id),
              },
            ]
          : []),
        ...(lead
          ? [
              {
                label: 'Ignore this picture',
                caption: 'Drop pic',
                hint: 'Ignore just the large picture: it leaves the list, and recognition never learns from it. The rest of the group stays. Undo brings it back.',
                icon: <IconPhotoX size={16} aria-hidden />,
                shortcut: 'G',
                onClick: () => onLabel([lead.track_id], IGNORED),
              },
            ]
          : []),
      ]}
      hotkeys={photos.length || pending > 0 ? { a: () => setAll(true) } : undefined}
      meta={
        <>
          {!all && <ProcessingImages count={pending} unavailable={unavailable} />}
          <Text size="xs" c="dimmed">
            {label}, {span(first, last)}
            {cams ? ` · ${cams}` : ''}
            {!all && (photos.length > 1 || pending > 0) && (
              <>
                {' · '}
                <Button variant="subtle" size="compact-xs" aria-keyshortcuts="A" onClick={() => setAll(true)}>
                  Show all ({photos.length})
                </Button>
              </>
            )}
          </Text>
        </>
      }
      onYes={g => name(g.id)}
      onNot={onNot ? g => onNot(tracks, g.id) : undefined}
      notHint={who =>
        `None of these sightings is ${who}. They stay in the list, guessed again without ${who}. Undo takes it back.`
      }
      onIgnore={() => name(IGNORED)}
      ignoreHint="Not a pet to name: every sighting here leaves the list, and recognition never learns from them. Undo brings them back."
      onSpace={photos.length || pending > 0 ? () => setAll(true) : undefined}
      picker={
        <NamePicker
          detected={group.species}
          speciesList={speciesList}
          pets={pets}
          guess={guessed ? { id: guessed.id, score } : undefined}
          onPet={pet => {
            leaving(group.id);
            name(pet.id);
          }}
          onOther={species =>
            onNewPet(species, pet => {
              leaving(group.id);
              name(pet.id);
            })
          }
        />
      }
    >
      {all && (
        <section className={who.picker} aria-label={`Every picture of ${label}`}>
          <Group justify="space-between" gap="xs">
            <Text size="xs" c="dimmed">
              {picked.size
                ? `${picked.size} of ${photos.length} selected`
                : `${plural(photos.length, 'picture', 'pictures')}. Select the ones that aren't this pet.`}
            </Text>
            <Group gap={6}>
              <Button
                leftSection={<IconUserMinus size={14} aria-hidden />}
                size="compact-sm"
                variant="subtle"
                disabled={!picked.size}
                onClick={() => {
                  setLeft(have => new Set([...have, ...picked]));
                  setPicked(new Set());
                }}
              >
                Not this pet
              </Button>
              <Button
                size="compact-sm"
                variant="subtle"
                disabled={!picked.size}
                onClick={() => {
                  onLabel([...picked], IGNORED);
                  setPicked(new Set());
                }}
              >
                Ignore these
              </Button>
              <Button size="compact-sm" variant="subtle" onClick={() => setAll(false)}>
                Hide
              </Button>
            </Group>
          </Group>
          <SelectableGrid
            label="Pick the pictures that aren't this pet"
            tiles={photos.map(s => ({
              key: s.track_id,
              label: `${group.species} on ${cameraName(s.cam)}, ${span(s.start, s.end)}`,
              content: <SafeImage className={who.square} src={photoUrl(s.track_id, true)} alt="" loading="lazy" />,
            }))}
            isOn={key => picked.has(key)}
            onToggle={toggle}
            tone="remove"
          />
          <ProcessingImages count={pending} unavailable={unavailable} />
        </section>
      )}
    </NamingRow>
  );
}

export function PetInbox({
  groups,
  pets,
  names,
  speciesList,
  loading,
  error,
  onRetry,
  retrying = false,
  onBack,
  onLabel,
  onNot,
  onBulk,
  onNewPet,
}: {
  groups: SightingGroup[];
  pets: Pet[];
  names: Record<string, string>;
  speciesList: string[]; // every species a pet can be, for naming a group the detector got wrong
  loading: boolean;
  error: string | null; // the server's reason, behind Show details
  onRetry: () => void;
  retrying?: boolean;
  onBack: () => void;
  onLabel: (tracks: string[], label: string) => void;
  onNot?: (tracks: string[], pet: string) => void; // "Not Mia"
  onBulk: (pet: string, checked: string[], unchecked: string[]) => void; // By pet's save
  onNewPet: (species: string, then: (pet: Pet) => void) => void;
}) {
  return (
    <NamingInbox
      prefs="pet"
      // M until this browser picks a size, as Who's this? does: at L a row was a 300px picture beside two short lines,
      // three to a screen (Known's design pass, P3).
      defaultSize="M"
      title="Sightings to name"
      headingId="pet-sightings-h"
      onBack={onBack}
      backLabel="Back to pets"
      status={
        loading ? (
          <PageLoading label="sightings" shape="rows" lines={3} />
        ) : error ? (
          <LoadError message="Couldn't load sightings" detail={error} onRetry={onRetry} busy={retrying} />
        ) : null
      }
      empty="No sightings are waiting for a name"
      bulkLabel={pets.length > 0 ? 'By pet' : undefined}
      bulk={
        pets.length > 0
          ? ({ thumb }) =>
              loading || error ? null : <ByPet groups={groups} pets={pets} onSave={onBulk} thumb={thumb} />
          : undefined
      }
      items={groups.map(g => ({
        key: g.id,
        guess: groupGuess(g, g.sightings, names),
        render: ({ thumb }) => (
          <GroupRow
            group={g}
            pets={pets}
            names={names}
            speciesList={speciesList}
            onLabel={onLabel}
            onNot={onNot}
            onNewPet={onNewPet}
            thumb={thumb}
          />
        ),
      }))}
    />
  );
}
