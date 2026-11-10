// Pets' "Sightings to name": every group of animal sightings nobody has named yet, in the shared naming
// inbox (../../naming): sorted and sized there, one NamingRow each, with its newest picture large, when and
// where it was seen, and the best guess. A card has two buttons, by its guess. With one: "<Name>", which labels its
// sightings as that pet, and "Not <Name>", which tells recognition they aren't (never guessed as that pet again) and
// leaves them waiting. With none: the name picker, a Cascader of a species (the detected one first), then that
// species' pets and Other… for a new pet (naming a pet of another species corrects the sighting's), plus "Ignore"
// (these sightings, from now on) and "Block" (sightings very like them are no longer listed).
//
// A card with several pictures has no buttons until it is selected, which opens every picture, all selected: clicking
// one deselects it, and that leaves it out of the answer. Deselected sightings stay unnamed and are grouped again on
// the next read. "By pet" labels one pet's guessed sightings in bulk, and "Show ignored and blocked" lists what was
// hidden, each with a way back.
import { Cascader, Text } from '@mantine/core';
import { useState } from 'react';
import { ProcessingImages } from '../../../shared/ProcessingImages';
import { capitalize, plural } from '../../lib/format';
import { LoadError, PageLoading } from '../../lib/page';
import { type HiddenItem, HiddenItems } from '../../naming/HiddenItems';
import { type InboxGuess, NamingInbox, useLeaving } from '../../naming/NamingInbox';
import { type CardChoice, NamingRow } from '../../naming/NamingRow';
import { Pictures, toggled } from '../../naming/Pictures';
import '../../whosthis/whosthis.css';
import { cameraName } from '../profile/role';
import {
  type Guess,
  HIDDEN_SIGHTINGS_KEY,
  type HiddenSighting,
  hiddenSightings,
  IGNORED,
  nameOf,
  OTHER,
  type Pet,
  photoUrl,
  type Sighting,
  type SightingGroup,
  span,
} from './api';
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
 *  sightings left in it (some were answered, or the response omits a group guess). Only a named pet is a guess. */
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
  onPet,
  onOther,
}: {
  detected: string;
  speciesList: string[];
  pets: Pet[];
  onPet: (pet: Pet) => void;
  onOther: (species: string) => void;
}) {
  // The detected species first when a pet can be one; a sighting of "some animal" (a zebra) gets the list.
  const first = speciesList.includes(detected) ? [detected] : [];
  const data = [...first, ...speciesList.filter(s => s !== detected)].map(s => ({
    value: s,
    label: `${title(s)}${s === detected ? ' (detected)' : ''}`,
    children: [
      ...pets.filter(p => p.species === s).map(p => ({ value: p.id, label: p.name })),
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
  onBlock,
  onNewPet,
  thumb,
}: {
  group: SightingGroup;
  pets: Pet[];
  names: Record<string, string>;
  speciesList: string[];
  onLabel: (tracks: string[], label: string) => void;
  onNot?: (tracks: string[], pet: string) => void;
  onBlock?: (tracks: string[]) => void;
  onNewPet: (species: string, then: (pet: Pet) => void) => void;
  thumb: boolean; // the small size: load the thumbnail
}) {
  const leaving = useLeaving();
  // The sightings deselected on the open card: left out of its answer. They stay unnamed and are grouped again.
  const [out, setOut] = useState<ReadonlySet<string>>(() => new Set());
  const kept = group.sightings;
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
  // What the buttons act on: every sighting but the deselected ones (one whose picture isn't made yet included).
  const tracks = kept.map(s => s.track_id).filter(t => !out.has(t));
  const whole = tracks.length === kept.length;
  const answered = (run: () => void) => () => {
    if (!tracks.length) return;
    if (whole) leaving(group.id);
    setOut(new Set()); // what was left out is what the card has next, all selected
    run();
  };
  const name = (value: string) => answered(() => onLabel(tracks, value))();
  const these = tracks.length === 1 ? 'This sighting is' : 'These sightings are';
  const choices: CardChoice[] = guessed
    ? [
        {
          label: guessed.name,
          hint: `${these} ${guessed.name}. Recognition learns from them. Undo takes it back.`,
          shortcut: 'Y',
          onClick: () => name(guessed.id),
        },
        {
          label: `Not ${guessed.name}`,
          hint: `${these} not ${guessed.name}. ${guessed.name} is never the guess for them again, and they wait for another name. Undo takes it back.`,
          shortcut: 'N',
          onClick: () => {
            if (!tracks.length) return;
            setOut(new Set());
            onNot?.(tracks, guessed.id);
          },
        },
      ]
    : [
        {
          label: 'Ignore',
          hint: 'Ignore the selected sightings from now on: they leave the list, and recognition never learns from them. Undo brings them back.',
          shortcut: 'I',
          onClick: () => name(IGNORED),
        },
        {
          label: 'Block',
          hint: 'Stop listing this animal: the selected sightings go, and so does any sighting very like them from now on. Undo brings them back.',
          shortcut: 'B',
          onClick: answered(() => onBlock?.(tracks)),
        },
      ];
  return (
    <NamingRow
      rowKey={group.id}
      label={label}
      lead={lead ? { src: photoUrl(lead.track_id, thumb), alt: `${group.species} on ${cameraName(lead.cam)}` } : null}
      guess={guessed}
      choices={choices}
      choicesDisabled={!tracks.length}
      foldable={kept.length > 1}
      meta={
        <>
          <ProcessingImages count={pending} unavailable={unavailable} />
          <Text size="xs" c="dimmed">
            {label}, {span(first, last)}
            {cams ? ` · ${cams}` : ''}
          </Text>
        </>
      }
      picker={
        !guessed && (
          <NamePicker
            detected={group.species}
            speciesList={speciesList}
            pets={pets}
            onPet={pet => name(pet.id)}
            onOther={species => onNewPet(species, pet => name(pet.id))}
          />
        )
      }
    >
      {expanded =>
        expanded &&
        kept.length > 1 && (
          <Pictures
            label={label}
            pictures={photos.map(s => ({
              key: s.track_id,
              label: `${group.species} on ${cameraName(s.cam)}, ${span(s.start, s.end)}`,
              src: photoUrl(s.track_id, true),
            }))}
            total={kept.length}
            out={out}
            onToggle={track => setOut(have => toggled(have, track))}
          />
        )
      }
    </NamingRow>
  );
}

/** A hidden sighting as "Show ignored and blocked" lists it. */
const hiddenItem = (s: HiddenSighting): HiddenItem => ({
  key: s.track_id,
  label: `${title(s.species || 'animal')}${s.cam ? ` on ${cameraName(s.cam)}` : ''}${
    s.start != null ? `, ${span(s.start, s.end ?? s.start)}` : ''
  }`,
  pictures: [photoUrl(s.track_id, true)],
});

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
  onBlock,
  onRestore,
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
  onBlock?: (tracks: string[]) => void; // "Block": these and any sighting very like them
  // "Show ignored and blocked": un-ignore or unblock one sighting; rejects with why it couldn't
  onRestore?: (kind: 'ignored' | 'blocked', track: string) => Promise<void>;
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
            onBlock={onBlock}
            onNewPet={onNewPet}
            thumb={thumb}
          />
        ),
      }))}
      footer={
        onRestore && (
          <HiddenItems
            queryKey={HIDDEN_SIGHTINGS_KEY}
            load={async signal => {
              const found = await hiddenSightings(signal);
              return {
                ignored: found.ignored.map(hiddenItem),
                ignoredCount: found.ignored_count,
                blocked: found.blocked.map(hiddenItem),
                blockedCount: found.blocked_count,
              };
            }}
            onRestore={(kind, item) => onRestore(kind, item.key)}
          />
        )
      }
    />
  );
}
