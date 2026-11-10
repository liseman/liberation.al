// Known's Pets tab, in People's form: the named pets on the left, as People's roster, and on the right either
// a pet's card (as a contact card) or "Sightings to name", the animal groups nobody has named yet (as Who's this?).
// A link in: #pet=<name> opens that pet's card, #species=<species> lists only that species. Labeling a group
// takes it off at once, with Undo in the toast; only sightings still waiting for a label are listed.
import { VisuallyHidden } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { messageOf } from '../../shared/http';
import { confirm } from '../lib/confirm';
import type { Message } from '../lib/message';
import { Toast } from '../lib/Toast';
import { knownTabOf } from '../router';
import classes from './people.module.css';
import { getPets, PETS_KEY, petInHash, SIGHTINGS_HASH } from './petLinks';
import {
  byWindow,
  choosePhoto,
  IGNORED,
  labelSightings,
  markNot,
  nameOf,
  type Pet,
  type PetsView,
  removePet,
  type Seen,
  SIGHTINGS_KEY,
  sightingsQuery,
} from './pets/api';
import { PetCard } from './pets/PetCard';
import { PetForm } from './pets/PetForm';
import { PetInbox } from './pets/PetInbox';
import { PetRoster } from './pets/PetRoster';
import { oneColumn } from './RosterColumn';

type Said = Message<() => Promise<void>>;
type Editing = { pet: Pet | null; species?: string; then?: (pet: Pet) => void } | null;
const NO_SEEN: Seen[] = [];

export function Pets() {
  const narrow = useMediaQuery('(max-width: 820px)') ?? false; // people.module.css's one column
  const client = useQueryClient();
  const query = useQuery({ queryKey: PETS_KEY, queryFn: ({ signal }) => getPets(signal) });
  const sightings = useQuery(sightingsQuery());
  const [focus, setFocus] = useState(() => petInHash(location.hash));
  // Links in, on every hashchange and each time the tab shows (it stays mounted, its effects paused, while hidden):
  // a pet or species link wins over the last pet picked, and #sightings opens what waits for a name (the command
  // palette's jumps, the review).
  useEffect(() => {
    const follow = () => {
      if (knownTabOf(location.pathname) !== 'pets') return;
      if (location.hash === SIGHTINGS_HASH) {
        setSel('?');
        history.replaceState(history.state, '', location.pathname + location.search);
        return;
      }
      const link = petInHash(location.hash);
      setFocus(link);
      if (link.pet || link.species) {
        setSel(null);
      }
    };
    follow();
    window.addEventListener('hashchange', follow);
    return () => window.removeEventListener('hashchange', follow);
  }, []);
  const pets = query.data?.pets ?? [];
  const options = query.data?.species ?? [];
  // A pet's id, '?' for the sightings waiting for a name, or null (that list, on wide windows).
  // On a phone, the sightings waiting for a name come first, unless a link asked for one pet or species.
  const [sel, setSel] = useState<string | null>(() => (oneColumn() && !petInHash(location.hash).pet ? '?' : null));
  const linked = focus.pet ? pets.find(p => p.name.toLowerCase() === focus.pet?.toLowerCase()) : undefined;
  const shown = sel ?? linked?.id ?? null;
  const pet = shown && shown !== '?' ? pets.find(p => p.id === shown) : undefined;
  const [editing, setEditing] = useState<Editing>(null);
  const [said, setSaid] = useState<Said>(null);
  // Sightings labeled here leave the list at once, before it's read again (and come back on a failure or Undo).
  const [gone, setGone] = useState<ReadonlySet<string>>(() => new Set());
  const data = sightings.data;
  const names = data?.names ?? {};
  const groups = byWindow(
    (data?.groups ?? [])
      .filter(g => !focus.species || g.species === focus.species)
      .map(g => ({ ...g, sightings: g.sightings.filter(s => !gone.has(s.track_id)) }))
      .filter(g => g.sightings.length),
  );

  const show = (tracks: string[], on: boolean) =>
    setGone(old => {
      const next = new Set(old);
      for (const t of tracks) on ? next.delete(t) : next.add(t);
      return next;
    });
  const refresh = () => void client.invalidateQueries({ queryKey: SIGHTINGS_KEY });
  const label = async (tracks: string[], value: string) => {
    show(tracks, false);
    setSaid(null);
    try {
      const reply = await labelSightings(tracks, value);
      const who = nameOf(value, { ...names, ...Object.fromEntries(pets.map(p => [p.id, p.name])) });
      setSaid({
        text:
          value === IGNORED
            ? `Ignored ${reply.labeled} sighting${reply.labeled === 1 ? '' : 's'}. They won't be asked about again.`
            : `Labeled ${reply.labeled} sighting${reply.labeled === 1 ? '' : 's'} as ${who}.`,
        undo: async () => {
          await labelSightings(tracks, null);
          show(tracks, true);
        },
      });
    } catch (err) {
      show(tracks, true);
      setSaid({ err: `Could not label those sightings: ${messageOf(err)}` });
    }
    refresh();
  };
  const undo = async (run: () => Promise<void>) => {
    setSaid(null);
    try {
      await run();
      setSaid({ text: 'Undone.' });
    } catch (err) {
      setSaid({ err: `Could not undo: ${messageOf(err)}` });
    }
    refresh();
  };
  // By pet's save: the checked sightings are labeled as the pet, the unchecked ones marked not it. One toast,
  // one Undo for both halves.
  const bulk = async (id: string, checked: string[], unchecked: string[]) => {
    const who = nameOf(id, { ...names, ...Object.fromEntries(pets.map(p => [p.id, p.name])) });
    show(checked, false); // the named ones leave at once; the others still wait, as not this pet
    setSaid(null);
    try {
      const labeled = checked.length ? (await labelSightings(checked, id)).labeled : 0;
      const marked = unchecked.length ? (await markNot(unchecked, id)).marked : 0;
      const parts = [
        labeled ? `Labeled ${labeled} sighting${labeled === 1 ? '' : 's'} as ${who}.` : '',
        marked ? `Marked ${marked} as not ${who}.` : '',
      ];
      setSaid({
        text: parts.filter(Boolean).join(' '),
        undo: async () => {
          if (checked.length) await labelSightings(checked, null);
          if (unchecked.length) await markNot(unchecked, id, false);
          show(checked, true);
        },
      });
    } catch (err) {
      show(checked, true);
      setSaid({ err: `Could not save those sightings: ${messageOf(err)}` });
    }
    refresh();
  };
  // A group in Sightings to name that isn't the pet guessed: it stays, guessed afresh without that pet.
  const notGroup = async (tracks: string[], id: string) => {
    const who = nameOf(id, { ...names, ...Object.fromEntries(pets.map(p => [p.id, p.name])) });
    setSaid(null);
    try {
      const reply = await markNot(tracks, id);
      setSaid({
        text: `Marked ${reply.marked} sighting${reply.marked === 1 ? '' : 's'} as not ${who}.`,
        undo: async () => {
          await markNot(tracks, id, false);
        },
      });
    } catch (err) {
      setSaid({ err: `Could not change those sightings: ${messageOf(err)}` });
    }
    refresh();
  };
  // A sighting on a pet's card that isn't that pet: it leaves the card, and its label too if it had one.
  const notPet = async (p: Pet, track: string) => {
    const labeled = data?.seen?.[p.id]?.find(s => s.track_id === track)?.how === 'labeled';
    setSaid(null);
    try {
      await markNot([track], p.id);
      setSaid({
        text: `Marked that sighting as not ${p.name}.`,
        undo: async () => {
          await markNot([track], p.id, false);
          if (labeled) await labelSightings([track], p.id);
        },
      });
    } catch (err) {
      setSaid({ err: `Could not change that sighting: ${messageOf(err)}` });
    }
    refresh();
  };
  // A pet card's photo choices, with Undo back to how the card was.
  const photo = async (p: Pet, action: 'main' | 'newest' | 'hide', track: string) => {
    setSaid(null);
    const before = { photo: p.photo ?? null, hidden: p.hidden ?? [] };
    try {
      saved((await choosePhoto(p.id, action, track)).pets);
      setSaid({
        text:
          action === 'hide'
            ? 'Hidden from the card.'
            : action === 'main'
              ? 'Now the main photo.'
              : 'The newest is the main photo again.',
        undo: async () => {
          if (action === 'hide') await choosePhoto(p.id, 'show', track);
          const list = before.photo
            ? (await choosePhoto(p.id, 'main', before.photo)).pets
            : (await choosePhoto(p.id, 'newest')).pets;
          saved(list);
        },
      });
    } catch (err) {
      setSaid({ err: `Could not change ${p.name}'s card: ${messageOf(err)}` });
    }
  };
  const saved = (list: Pet[]) => {
    client.setQueryData<PetsView>(PETS_KEY, old => ({ pets: list, species: old?.species ?? options }));
    refresh();
  };
  const remove = async (p: Pet) => {
    // Asked first, as deleting a person is.
    const sure = await confirm({
      id: `remove-pet-${p.id}`,
      title: `Remove ${p.name}`,
      body: `Remove ${p.name}'s name? Recorded observations stay.`,
      confirm: 'Remove',
      danger: true,
    });
    if (!sure) return;
    try {
      saved((await removePet(p.id)).pets);
      setSel(null);
      setSaid({ text: `Removed ${p.name}'s name. Recorded observations stay.` });
    } catch (err) {
      setSaid({ err: messageOf(err) });
    }
  };
  const back = () => {
    setSel(null);
    if (focus.pet) {
      history.replaceState(history.state, '', location.pathname + location.search);
      setFocus({});
    }
  };
  const failed = query.isError && !query.data ? "Couldn't load pets" : null;
  const failedDetail = failed ? messageOf(query.error) : null;
  const check = pet ? data?.recognition.find(c => c.species === pet.species) : undefined;
  return (
    // Not a region of its own: the roster under it is the "Pets" region, by its heading (two "Pets" landmarks, axe)
    <div className={`${classes.page} ${shown != null ? classes.picked : ''}`}>
      {/* On a phone the roster, and its h1, hide while this shows: the page's heading for screen readers then */}
      {narrow && shown != null && <VisuallyHidden component="h1">Pets</VisuallyHidden>}
      <div className={classes.main}>
        <div className={classes.cols}>
          <PetRoster
            pets={pets}
            seen={data?.seen ?? {}}
            waiting={data ? groups.length : null}
            waitingFailed={sightings.isError && !data}
            loaded={query.data != null}
            error={failed}
            errorDetail={failedDetail}
            retrying={query.isFetching}
            onRetry={() => void query.refetch()}
            only={focus.species}
            onShowAll={() => {
              history.replaceState(history.state, '', location.pathname + location.search);
              setFocus({});
            }}
            sel={shown}
            onOpen={setSel}
            onNew={() => setEditing({ pet: null })}
          />
          <div className={classes.detail}>
            {pet ? (
              <PetCard
                key={pet.id}
                pet={pet}
                seen={data?.seen?.[pet.id] ?? NO_SEEN}
                pending={data?.seen_pending?.[pet.id] ?? 0}
                unavailable={data?.seen_unavailable?.[pet.id] ?? 0}
                check={check}
                onBack={back}
                onEdit={() => setEditing({ pet })}
                onRemove={() => void remove(pet)}
                onPhoto={(action, track) => void photo(pet, action, track)}
                onNot={track => void notPet(pet, track)}
                onChanged={refresh}
              />
            ) : (
              <PetInbox
                groups={groups}
                pets={pets}
                names={names}
                speciesList={options}
                loading={sightings.isPending}
                error={sightings.isError ? messageOf(sightings.error) : null}
                onRetry={() => void sightings.refetch()}
                retrying={sightings.isFetching}
                onBack={back}
                onLabel={(tracks, value) => void label(tracks, value)}
                onNot={(tracks, id) => void notGroup(tracks, id)}
                onBulk={(id, checked, unchecked) => void bulk(id, checked, unchecked)}
                onNewPet={(species, then) => setEditing({ pet: null, species, then })}
              />
            )}
          </div>
        </div>
      </div>
      {editing && (
        <PetForm
          opened
          pet={editing.pet}
          species={editing.species}
          options={options}
          onClose={() => setEditing(null)}
          onSaved={(list, p) => {
            saved(list);
            if (editing.then) editing.then(p);
            else setSaid({ text: editing.pet ? `Saved ${p.name}.` : `Added ${p.name}.` });
            if (!editing.pet && !editing.then) {
              setSel(p.id);
            }
          }}
        />
      )}
      <Toast message={said} label="Pets" onUndo={u => void undo(u)} onClose={() => setSaid(null)} />
    </div>
  );
}
