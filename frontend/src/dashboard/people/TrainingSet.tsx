// Training images: every example recognition matches against for one person (their face rows) or one pet (the
// sightings labeled as it), on their card. A wrong picture sometimes gets in and pulls matches the wrong way, so the
// grid puts the least typical first, where a mistake is likely to be, and any can be selected and removed.
//
// Closed until asked for, so a card doesn't read a set it won't show. The whole listing comes in one read (a row is
// about 150 bytes, so a couple of thousand is a few hundred kilobytes) and the grid shows PAGE tiles at a time with
// lazy thumbnails, so a large set costs only what is on screen. Removing can't be undone (a pet's example is erased
// from the knowledge store), so it asks first.

import { Button, Checkbox, Group, Text } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf, request } from '../../shared/http';
import { SafeImage } from '../../shared/SafeImage';
import { confirm } from '../lib/confirm';
import { dateTime, plural } from '../lib/format';
import { Empty, Section } from './profile/parts';
import classes from './profile/profile.module.css';
import { cameraName } from './profile/role';

export const PAGE = 60; // tiles shown at first, and added by each "Show more"
const CHUNK = 500; // the most examples one removal request carries

/** One example as the grid shows it, whichever kind it is. */
export interface TrainingImage {
  key: string;
  src: string | null; // null: no picture was kept for it
  at: number | null;
  cam: string | null;
  how: string; // how it got into the set, in a few words
  similarity: number | null; // its cosine to the rest of the set; null when there is nothing to compare with
  unused?: boolean; // a pet label with no example in the current model: listed, but teaching nothing now
}

/** Where the grid's examples come from and how they are removed. */
export interface TrainingSource {
  queryKey: readonly unknown[];
  load: (signal?: AbortSignal) => Promise<TrainingImage[]>;
  remove: (keys: string[]) => Promise<unknown>;
}

interface FaceExample {
  key: string;
  kind: 'owner' | 'learned';
  source: string;
  at: number | null;
  cam: string | null;
  picture: boolean;
  similarity: number | null;
}

const FACE_HOW: Record<string, string> = {
  group: 'you named a face group',
  introduced: 'named from an introduction',
  correction: 'you corrected a sighting',
  reference: 'your reference photo',
  enrolled: 'enrolled from image files',
  learned: 'learned automatically',
  owner: 'added by you',
};

/** A person's face rows (dashboard/face_examples.py). */
export function personTraining(name: string): TrainingSource {
  return {
    queryKey: ['people', 'examples', name],
    load: async signal => {
      const reply = await request<{ examples: FaceExample[] }>(
        `/people/examples?name=${encodeURIComponent(name)}`,
        undefined,
        signal,
      );
      return reply.examples.map(e => ({
        key: e.key,
        src: e.picture ? `/people/examples/photo?key=${e.key}` : null,
        at: e.at,
        cam: e.cam,
        how: FACE_HOW[e.source] ?? 'added by you',
        similarity: e.similarity,
      }));
    },
    remove: keys => request('/people/examples/remove', { name, keys }),
  };
}

interface PetExample {
  track_id: string;
  cam: string;
  start: number | null;
  source: 'owner' | 'carried';
  picture: 'kept' | 'recorded' | null;
  example: boolean;
  similarity: number | null;
}

/** A pet's labeled sightings (dashboard/pets.py). One with no kept photo whose track is still recorded shows the
 *  sighting's own photo, cut from the recording. */
export function petTraining(pet: string): TrainingSource {
  return {
    queryKey: ['pets', 'examples', pet],
    load: async signal => {
      const reply = await request<{ examples: PetExample[] }>(
        `/pets/examples?pet=${encodeURIComponent(pet)}`,
        undefined,
        signal,
      );
      return reply.examples.map(e => {
        const track = encodeURIComponent(e.track_id);
        return {
          key: e.track_id,
          src:
            e.picture === 'kept'
              ? `/pets/examples/photo?track=${track}`
              : e.picture === 'recorded'
                ? `/pets/sightings/photo?track=${track}`
                : null,
          at: e.start,
          cam: e.cam || null,
          how: e.source === 'carried' ? 'carried over from an earlier model' : 'labeled by you',
          similarity: e.similarity,
          unused: !e.example,
        };
      });
    },
    remove: async keys => {
      for (let i = 0; i < keys.length; i += CHUNK)
        await request('/pets/examples/remove', { pet, track_ids: keys.slice(i, i + CHUNK) });
    },
  };
}

/** A tile's caption: when and where it came from, how it got in, and how like the rest it is. */
export function captionOf(image: TrainingImage): string {
  const parts = [image.at != null ? dateTime(image.at) : 'date not kept', cameraName(image.cam), image.how];
  if (image.unused) parts.push('not used by the current model');
  else if (image.similarity != null) parts.push(`${Math.round(Math.max(0, image.similarity) * 100)}% like the rest`);
  return parts.filter(Boolean).join(' · ');
}

export function TrainingSet({
  name,
  source,
  onRemoved,
}: {
  name: string;
  source: TrainingSource;
  onRemoved?: () => void; // the card's own data changed with the set
}) {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [shown, setShown] = useState(PAGE);
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const query = useQuery({
    queryKey: source.queryKey,
    queryFn: ({ signal }) => source.load(signal),
    enabled: open,
  });
  const images = query.data ?? [];
  const chosen = images.filter(i => picked.has(i.key));

  const toggle = (key: string) => {
    const next = new Set(picked);
    if (!next.delete(key)) next.add(key);
    setPicked(next);
  };
  const remove = async () => {
    const sure = await confirm({
      title: `Remove ${plural(chosen.length, 'training image', 'training images')} from ${name}?`,
      body: `Recognition stops matching ${name} against ${chosen.length === 1 ? 'it' : 'them'} right away. This can't be undone.`,
      confirm: 'Remove',
      danger: true,
      id: 'training-remove',
    });
    if (!sure) return;
    setBusy(true);
    setError(null);
    try {
      await source.remove(chosen.map(i => i.key));
      setPicked(new Set());
      onRemoved?.();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
      await client.invalidateQueries({ queryKey: source.queryKey });
    }
  };

  return (
    <Section
      title="Training images"
      aside={
        <Button variant="subtle" size="compact-xs" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? 'Hide' : 'View all'}
        </Button>
      }
    >
      {!open ? (
        <Empty>{`Every picture recognition matches ${name} against. Remove any that are wrong.`}</Empty>
      ) : query.isPending ? (
        <Empty>Loading the training images…</Empty>
      ) : query.isError ? (
        <Text size="sm" c="red" role="alert">
          {messageOf(query.error)}
        </Text>
      ) : !images.length ? (
        <Empty>{`Recognition has no examples of ${name} yet.`}</Empty>
      ) : (
        <>
          <Group gap="xs" mb={8} align="center">
            <Text size="sm" c="dimmed" flex={1} miw={200}>
              {`${plural(images.length, 'example', 'examples')}, the least like the rest first: a wrong one is usually near the top.`}
            </Text>
            {chosen.length > 0 && (
              <>
                <Button variant="subtle" size="compact-xs" onClick={() => setPicked(new Set())} disabled={busy}>
                  Clear
                </Button>
                <Button color="red" size="compact-xs" onClick={() => void remove()} loading={busy}>
                  {`Remove ${chosen.length} selected`}
                </Button>
              </>
            )}
          </Group>
          {error && (
            <Text size="sm" c="red" role="alert" mb={8}>
              {error}
            </Text>
          )}
          <ul className={classes.training}>
            {images.slice(0, shown).map(image => {
              const caption = captionOf(image);
              const selected = picked.has(image.key);
              return (
                <li key={image.key} className={classes.trainingTile} data-selected={selected || undefined}>
                  <label htmlFor={`training-${image.key}`}>
                    {image.src ? (
                      <SafeImage src={image.src} alt="" loading="lazy" retries={1} />
                    ) : (
                      <span className={classes.noPicture}>No picture kept</span>
                    )}
                    <Checkbox
                      id={`training-${image.key}`}
                      className={classes.trainingPick}
                      size="xs"
                      checked={selected}
                      onChange={() => toggle(image.key)}
                      aria-label={`Select the example of ${name}: ${caption}`}
                    />
                    <Text size="xs" c="dimmed" component="span" className={classes.trainingCaption}>
                      {caption}
                    </Text>
                  </label>
                </li>
              );
            })}
          </ul>
          {images.length > shown && (
            <Button variant="default" size="xs" mt={10} onClick={() => setShown(shown + PAGE)}>
              {`Show more (${shown} of ${images.length} shown)`}
            </Button>
          )}
        </>
      )}
    </Section>
  );
}
