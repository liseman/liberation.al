// A named pet's card, in the contact card's form (../profile/Profile.tsx): its photo, name, species, and where
// it was last seen, Edit and Remove, then its recent sightings and whether its species is recognized yet. Each sighting
// has People's ⋯ menu: Use as main photo, Not <name>, and Hide from the card. Training images (../TrainingSet.tsx)
// lists every sighting labeled as it, to find and remove a wrong one.

import { ActionIcon, Button, Group, Menu, Text, Title } from '@mantine/core';
import { IconDots, IconUserMinus } from '@tabler/icons-react';
import { ProcessingImages } from '../../../shared/ProcessingImages';
import { capitalize } from '../../lib/format';
import { MainPhoto, PhotoGrid } from '../entity';
import { Empty, ItemActions, type MenuAction, Section } from '../profile/parts';
import classes from '../profile/profile.module.css';
import { cameraName, whereabouts } from '../profile/role';
import { VideoTile } from '../profile/VideoTile';
import { petTraining, TrainingSet } from '../TrainingSet';
import {
  type Check,
  type Pet,
  petPhoto,
  petVideos,
  photoUrl,
  type Seen,
  species as Species,
  span,
  videoSightings,
} from './api';

/** Whether a species is recognized by appearance yet, and how its held-out check went. */
export function checkText(check: Check): string {
  const tally = `${check.right} right, ${check.wrong} wrong, ${check.unknown} unsure`;
  return check.on
    ? `recognized by appearance. Checked on ${check.labeled} labeled sightings: ${tally}.`
    : `not recognized yet (${check.reason}).${check.labeled ? ` Checked so far: ${tally}.` : ''}`;
}

function CheckLine({ check }: { check: Check }) {
  const t = checkText(check);
  return <Text size="sm">{capitalize(t)}</Text>;
}

export function PetCard({
  pet,
  seen,
  pending = 0,
  unavailable = 0,
  check,
  onBack,
  onEdit,
  onRemove,
  onPhoto,
  onNot,
  onChanged,
}: {
  pet: Pet;
  seen: Seen[];
  pending?: number;
  unavailable?: number;
  check?: Check;
  onBack: () => void;
  onEdit: () => void;
  onRemove: () => void;
  onPhoto?: (action: 'main' | 'newest' | 'hide', track: string) => void; // the card's photo choices
  onNot?: (track: string) => void; // a sighting that isn't this pet
  onChanged?: () => void; // its training set changed: read its sightings again
}) {
  const newest = seen[0];
  const hidden = new Set(pet.hidden ?? []);
  const photos = seen.filter(s => s.photo_ready !== false && !hidden.has(s.track_id));
  const photo = petPhoto(pet, photos, true);
  const actions = (s: Seen): MenuAction[] => [
    pet.photo === s.track_id
      ? { label: 'Use the newest as main photo', onClick: () => onPhoto?.('newest', s.track_id) }
      : { label: 'Use as main photo', onClick: () => onPhoto?.('main', s.track_id) },
    { label: `Not ${pet.name}`, icon: <IconUserMinus size={14} aria-hidden />, onClick: () => onNot?.(s.track_id) },
    { label: 'Hide from the card', onClick: () => onPhoto?.('hide', s.track_id) },
  ];
  const videos = petVideos(seen);
  const selected = videoSightings(seen);
  const videosPending = selected.filter(s => s.video_status === 'pending').length;
  const videosUnavailable = selected.filter(s => s.video_status === 'unavailable').length;
  const where = newest ? whereabouts({ last_seen: newest.end, last_seen_cam: newest.cam }) : 'Not seen yet';
  return (
    <article className={classes.profile} aria-label={`Card of ${pet.name}`}>
      <div className={classes.back}>
        <Button variant="subtle" size="xs" onClick={onBack}>
          Back to pets
        </Button>
      </div>
      <header className={classes.head}>
        <MainPhoto url={photo ? photoUrl(photo) : null} name={pet.name} />
        <div className={classes.headText}>
          <Title order={2} className={classes.name}>
            {pet.name}
          </Title>
          <Text c="dimmed">{[Species(pet.species), pet.sex].filter(Boolean).join(' · ')}</Text>
          <Text size="sm">{where}</Text>
          <Group gap="xs" mt={10}>
            <Button variant="default" size="xs" onClick={onEdit}>
              Edit
            </Button>
            <Menu position="bottom-start" withinPortal>
              <Menu.Target>
                <ActionIcon variant="default" size={30} aria-label={`More for ${pet.name}`}>
                  <IconDots size={16} />
                </ActionIcon>
              </Menu.Target>
              <Menu.Dropdown>
                <Menu.Item color="red" onClick={onRemove}>
                  Remove
                </Menu.Item>
              </Menu.Dropdown>
            </Menu>
          </Group>
        </div>
      </header>
      <Section title="Recent video">
        {videos.length ? (
          <ul className={classes.videos}>
            {videos.map(v => (
              <VideoTile key={`${v.cam}-${v.start}`} v={v} name={pet.name} person={null} />
            ))}
          </ul>
        ) : videosPending || videosUnavailable ? null : (
          <Empty>{`No recorded video of ${pet.name} yet.`}</Empty>
        )}
        <ProcessingImages kind="video" count={videosPending} unavailable={videosUnavailable} />
      </Section>
      <Section title="Recent sightings">
        {photos.length ? (
          <PhotoGrid
            photos={photos.map(s => {
              const what = `${pet.name} on ${cameraName(s.cam)}, ${span(s.start, s.end)}`;
              return {
                key: s.track_id,
                src: photoUrl(s.track_id),
                alt: what,
                menu: onPhoto && onNot && (
                  <ItemActions label={`the photo of ${what}`} actions={actions(s)} className={classes.tileMenu} />
                ),
                caption: (
                  <Text size="xs" c="dimmed">
                    {span(s.start, s.end)} · {cameraName(s.cam)}
                    {s.how === 'recognized' ? ' · recognized' : ''}
                    {pet.photo === s.track_id ? ' · main photo' : ''}
                  </Text>
                ),
              };
            })}
          />
        ) : pending || unavailable ? null : (
          <Empty>Not seen yet. Name a group of its sightings under “Sightings to name”.</Empty>
        )}
        <ProcessingImages count={pending} unavailable={unavailable} />
      </Section>
      <TrainingSet name={pet.name} source={petTraining(pet.id)} onRemoved={onChanged} />
      {check && (
        <Section title="Recognition">
          <CheckLine check={check} />
        </Section>
      )}
    </article>
  );
}
