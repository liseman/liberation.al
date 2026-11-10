// Live's "Here now", under the cameras: who the cameras see (where, and since when), the named pets
// they see, and who was heard in the last minute, side by side in a row that wraps, each linking to their card.
import { Anchor, Group, Text, Title } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import type { MouseEventHandler } from 'react';
import { hm } from '../lib/format';
import { PageEmpty } from '../lib/page';
import { useLive } from '../live/hooks';
import { useLiveStatus } from '../live/status';
import type { TranscriptEntry } from '../live/types';
import { getPeople, PEOPLE_KEY, type Person } from '../people/api';
import { EntityAvatar } from '../people/entity';
import { getPets, PETS_KEY, type PetsView } from '../people/petLinks';
import { photoUrl } from '../people/pets/api';
import { href, linkClick } from '../router';

export const HEARD_S = 60; // heard this recently and not seen: "Heard just now"
export const PEOPLE_MS = 15_000; // here_now is the live scene's last 30 s

export interface Here {
  key: string;
  name: string;
  photo: string | null;
  where: string; // "Front camera, since 18:20", or "Heard just now"
  href: string; // their contact card, or the pet's card
  seen: boolean; // a camera sees them (the green dot), not only heard
}

const capital = (s: string) => s.charAt(0).toLocaleUpperCase() + s.slice(1);
const whereSeen = (
  cameraName: string | null | undefined,
  cam: string | null | undefined,
  since: number | null | undefined,
): string => {
  const at = capital(cameraName || cam || 'in view');
  return since ? `${at}, since ${hm(since)}` : at;
};

/** A link that opens someone's contact card in People. */
export const cardHref = (name: string): string => `${href('/people')}#card=${encodeURIComponent(name)}`;
const petCard = (name: string): string => `${href('/pets')}#pet=${encodeURIComponent(name)}`;

/** Who is here: everyone a camera sees, the earliest arrival first, then the named pets a camera sees, the same way,
 *  then everyone else heard in the last HEARD_S seconds by a name the People list knows. */
export function hereNow(
  people: readonly Person[],
  lines: readonly TranscriptEntry[],
  now: number,
  pets?: PetsView,
): Here[] {
  const seen = people
    .filter(p => p.here_now)
    .sort((a, b) => (a.here_now?.since ?? now) - (b.here_now?.since ?? now))
    .map(p => ({
      key: `person:${p.name}`,
      name: p.name,
      photo: p.photo ?? null,
      where: whereSeen(p.here_now?.camera_name, p.here_now?.cam, p.here_now?.since),
      href: cardHref(p.name),
      seen: true,
    }));
  const animals = (pets?.pets ?? [])
    .flatMap(p => {
      const h = pets?.here?.[p.id];
      return h ? [{ p, h }] : [];
    })
    .sort((a, b) => a.h.since - b.h.since)
    .map(({ p, h }) => ({
      key: `pet:${p.id}`,
      name: p.name,
      // the photo chosen on its card, else a look at it now
      photo: photoUrl(p.photo || h.track, true),
      where: whereSeen(h.camera_name, h.cam, h.since),
      href: petCard(p.name),
      seen: true,
    }));
  const byName = new Map(people.map(p => [p.name.toLocaleLowerCase(), p]));
  const shown = new Set(seen.map(h => h.name.toLocaleLowerCase()));
  const heard: Here[] = [];
  for (const e of [...lines].reverse()) {
    if (now - e.ts_start > HEARD_S) break;
    const p = byName.get((e.speaker ?? '').trim().toLocaleLowerCase());
    if (!p || shown.has(p.name.toLocaleLowerCase())) continue;
    shown.add(p.name.toLocaleLowerCase());
    heard.push({
      key: `person:${p.name}`,
      name: p.name,
      photo: p.photo ?? null,
      where: 'Heard just now',
      href: cardHref(p.name),
      seen: false,
    });
  }
  return [...seen, ...animals, ...heard];
}

export function HereNow({ onNameFace }: { onNameFace?: MouseEventHandler<HTMLAnchorElement> } = {}) {
  const people = useQuery({ queryKey: PEOPLE_KEY, queryFn: getPeople, refetchInterval: PEOPLE_MS });
  // The Pets list Live already reads for its boxes' links, with which pets are in view now
  const pets = useQuery({ queryKey: PETS_KEY, queryFn: ({ signal }) => getPets(signal), refetchInterval: PEOPLE_MS });
  const lines = useLive('transcript')?.entries ?? [];
  const now = useLiveStatus(s => s.now);
  const here = hereNow(people.data?.people ?? [], lines, now, pets.data);
  return (
    <section className="rail-card here-card" aria-labelledby="here-now-h">
      <Group justify="space-between" gap="xs">
        <Title order={2} id="here-now-h">
          Here now
        </Title>
        <Anchor size="sm" href={`${href('/people')}#whos-this`} onClick={onNameFace ?? linkClick}>
          Who's this?
        </Anchor>
      </Group>
      {here.length ? (
        // biome-ignore lint/a11y/noRedundantRoles: list-style none drops the list from VoiceOver
        <ul className="here-now" role="list">
          {here.map(h => (
            <li key={h.key}>
              <a href={h.href} onClick={linkClick}>
                <EntityAvatar name={h.name} photo={h.photo} here={h.seen} size={40} />
                <span className="here-who">
                  <span className="here-name">{h.name}</span>
                  <span className="here-where">{h.where}</span>
                </span>
              </a>
            </li>
          ))}
        </ul>
      ) : people.isPending ? (
        <Text size="sm" c="dimmed">
          Looking…
        </Text>
      ) : (
        <PageEmpty compact title="Nobody here right now">
          Anyone the cameras see, a named pet in view, or a known voice heard in the last minute, shows here.
        </PageEmpty>
      )}
    </section>
  );
}
