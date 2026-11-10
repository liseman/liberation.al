// The People list, left of the contact card: a count, search, "New person", how many faces wait for a name (a link to
// Live's Who's this?), People's own Who's this? (voices, skips, and the label log), then who's here now and everyone
// else, each with where they are or were last seen.
import { Badge, Button, Group, Modal, Text, TextInput } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, useState } from 'react';
import { messageOf } from '../../shared/http';
import { SafeImage } from '../../shared/SafeImage';
import { hm, plural } from '../lib/format';
import { refreshNames, unknownGroupsQuery, unknownVoicesQuery } from '../livetab/api';
import { viewName } from '../review/views';
import { dropWaiting } from '../whosthis/api';
import { addPerson, type Person } from './api';
import { EntityAvatar } from './entity';
import { claimLiveFace, type LiveFaceDraft } from './liveFace';
import { PersonSearch } from './PersonSearch';
import { personMeta, roleLine, whereabouts } from './profile/role';
import { RosterColumn, RosterRow } from './RosterColumn';

/** The "New person" dialog: a name, then Add. */
export function NewPerson({
  opened,
  onClose,
  onAdded,
  faceDraft,
}: {
  faceDraft?: LiveFaceDraft | null;
  opened: boolean;
  onClose: () => void;
  onAdded: (name: string) => void;
}) {
  const client = useQueryClient();
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const nm = name.trim();
  const close = () => {
    setName('');
    setErr(null);
    onClose();
  };
  const add = async (e: FormEvent) => {
    e.preventDefault();
    if (!nm) return;
    setBusy(true);
    setErr(null);
    try {
      const d = faceDraft ? await claimLiveFace(faceDraft, nm) : await addPerson(nm, '');
      if (faceDraft) {
        dropWaiting(client, 'face', faceDraft.source.group);
        void refreshNames(client); // people and the waiting lists
      }
      setName('');
      onAdded(d.name);
      onClose();
    } catch (e) {
      setErr(messageOf(e) || 'Adding failed. Try again.');
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal opened={opened} onClose={close} title="New person" size={440}>
      <form onSubmit={add}>
        {faceDraft && (
          <SafeImage
            src={faceDraft.image}
            alt="Selected face"
            style={{ maxWidth: '100%', maxHeight: 220, objectFit: 'contain' }}
          />
        )}
        {faceDraft && (
          <Text size="xs" c="dimmed">
            {viewName(faceDraft.source.camera)} · {hm(faceDraft.source.at)}
          </Text>
        )}
        <TextInput
          label="Name"
          maxLength={120}
          data-autofocus
          value={name}
          onChange={e => setName(e.currentTarget.value)}
        />
        {opened && !faceDraft && <PersonSearch name={name} onSelect={setName} />}
        {!faceDraft && (
          <Text size="sm" c="dimmed" mt={8}>
            Add manually at any time. Next you can attach reference photos, audio, or video on their card, or pick what
            has not been named yet.
          </Text>
        )}
        {err && (
          <Text size="sm" c="var(--lib-bad)" role="alert" mt={8}>
            {err}
          </Text>
        )}
        <Group justify="flex-end" gap="xs" mt="lg">
          <Button variant="default" onClick={close}>
            Cancel
          </Button>
          <Button type="submit" disabled={!nm || busy}>
            {nm ? `Add ${nm}` : 'Add'}
          </Button>
        </Group>
      </form>
    </Modal>
  );
}

/** One person's row: avatar, name, and where they are, were last seen, or what's known. */
function Row({ p, current, onPick, onOpen }: { p: Person; current: boolean; onPick: () => void; onOpen: () => void }) {
  // While identity is still rematching faces, an old last-seen time would mislead; personMeta says it's updating.
  const where = p.here_now || !(p.last_seen_pending || p.last_seen_error) ? whereabouts(p) : '';
  const sub = where ? [roleLine(p), where].filter(Boolean).join(', ') : personMeta(p);
  return (
    <RosterRow
      avatar={<EntityAvatar name={p.name} photo={p.photo} here={!!p.here_now} />}
      label={
        <>
          {p.name}{' '}
          {p.owner && (
            <Badge size="xs" variant="light" component="span">
              You
            </Badge>
          )}
        </>
      }
      description={sub}
      current={current}
      onPick={onPick}
      onOpen={onOpen}
    />
  );
}

/** Whether a person matches what was typed in the search: by name, role, or organization. */
const matches = (p: Person, q: string): boolean =>
  [p.name, p.role, p.organization].some(s => s?.toLowerCase().includes(q));

/** The line that opens People's own Who's this?, with how much waits there, or just its name when nothing does (it
 * still holds what was skipped and the label log). */
/** What waits in Who's this?: "3 faces and 2 voices need a name", "1 face needs a name", or just "Who's this?". */
export function waitingText(
  faces: number | null,
  voices: number | null,
  unavailable: readonly ('faces' | 'voices')[] = [],
): string {
  const parts = [faces && plural(faces, 'face', 'faces'), voices && plural(voices, 'voice', 'voices')].filter(Boolean);
  const total = (faces ?? 0) + (voices ?? 0);
  const known = parts.length ? `${parts.join(' and ')} ${total === 1 ? 'needs' : 'need'} a name` : "Who's this?";
  const pending = [
    faces === null && !unavailable.includes('faces') && 'faces',
    voices === null && !unavailable.includes('voices') && 'voices',
  ].filter(Boolean);
  const states = unavailable.map(kind => `${kind === 'faces' ? 'Faces' : 'Voices'} unavailable`);
  if (pending.length) states.push(`Loading ${pending.join(' and ')}…`);
  return [known, ...states].join(' · ');
}

export function Roster({
  people,
  loaded,
  error,
  errorDetail = null,
  retrying,
  onRetry,
  cursor,
  sel,
  onPick,
  onOpen,
  onAdded,
}: {
  people: Person[];
  loaded: boolean;
  error: string | null; // the first load failed
  errorDetail?: string | null; // the server's reason, behind Show details
  retrying: boolean;
  onRetry: () => void;
  cursor: string | null; // the selected roster row
  sel: string | null; // a person's name, '?' for Who's this?, or null
  onPick: (sel: string) => void; // single-click selection
  onOpen: (sel: string) => void; // Enter, I, or double-click
  onAdded: (name: string) => void;
}) {
  const [query, setQuery] = useState('');
  const [adding, setAdding] = useState(false);
  const q = query.trim().toLowerCase();
  const shown = q ? people.filter(p => matches(p, q)) : people;
  const here = shown.filter(p => p.here_now);
  const rest = shown.filter(p => !p.here_now);
  // Counts wait until the roster is on screen. The same queries run on Live immediately; here they must not
  // take a JSON slot before /people, or a cold group read holds the roster until the 20 s deadline.
  const faces = useQuery({ ...unknownGroupsQuery(), enabled: loaded });
  const voices = useQuery({ ...unknownVoicesQuery(), enabled: loaded });
  const unavailable: ('faces' | 'voices')[] = [];
  if (faces.isError) unavailable.push('faces');
  if (voices.isError) unavailable.push('voices');
  const waiting = !loaded
    ? error
      ? "Who's this? · People unavailable"
      : "Who's this? · Waiting for people…"
    : waitingText(
        faces.isError ? null : (faces.data?.count ?? null),
        voices.isError ? null : (voices.data?.count ?? null),
        unavailable,
      );
  const row = (p: Person) => (
    <Row key={p.name} p={p} current={cursor === p.name} onPick={() => onPick(p.name)} onOpen={() => onOpen(p.name)} />
  );
  return (
    <RosterColumn
      title="People"
      titleId="people-h"
      noun="people"
      count={people.length}
      query={query}
      onQuery={setQuery}
      newLabel="New person"
      onNew={() => setAdding(true)}
      waiting={waiting}
      waitingCurrent={sel == null || sel === '?'}
      onWaiting={() => onOpen('?')}
      loaded={loaded}
      error={error}
      errorDetail={errorDetail}
      retrying={retrying}
      onRetry={onRetry}
      empty={{ title: 'No one yet', text: "Add someone with New person, or name a face in Who's this?" }}
      noMatch={{ title: `Nobody matches “${query.trim()}”`, text: 'Check the spelling, or add them with New person.' }}
      groups={[
        { title: 'Here now', list: 'Here now', rows: here.map(row) },
        { title: 'Everyone', list: 'Everyone', rows: rest.map(row) },
      ]}
    >
      <NewPerson opened={adding} onClose={() => setAdding(false)} onAdded={onAdded} />
    </RosterColumn>
  );
}
