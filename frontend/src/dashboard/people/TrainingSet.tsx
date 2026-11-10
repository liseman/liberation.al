// Training images: every example recognition matches against for one person (their face rows) or one pet (the
// sightings labeled as it), on their card. A wrong picture sometimes gets in and pulls matches the wrong way, so the
// grid puts the least typical first, where a mistake is likely to be, and any can be selected and removed.
//
// Closed until asked for, so a card doesn't read a set it won't show. The whole listing comes in one read (a row is
// about 150 bytes, so a couple of thousand is a few hundred kilobytes) and the grid shows PAGE tiles at a time with
// lazy thumbnails, so a large set costs only what is on screen. Removing can't be undone (a pet's example is erased
// from the knowledge store), so it asks first.
//
// A tile's caption is one short line, how like the rest it is and when ("86% alike · today 13:23"); the camera and how
// it got into the set are in its accessible name and hover title. A tile well under the rest of the set (UNLIKE_BY
// below its median, in a set of at least UNLIKE_MIN compared examples) carries an "Unlike the rest" badge, so the
// example the section exists to surface is marked without reading a caption.

import { Badge, Button, Checkbox, Group, Skeleton, Text } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf, request } from '../../shared/http';
import { SafeImage } from '../../shared/SafeImage';
import { confirm } from '../lib/confirm';
import { plural, seenAt } from '../lib/format';
import { type Message, StatusLine } from '../lib/message';
import { LoadError } from '../lib/page';
import { Empty, Section } from './profile/parts';
import classes from './profile/profile.module.css';
import { cameraName } from './profile/role';

export const PAGE = 60; // tiles shown at first, and added by each "Show more"
const CHUNK = 500; // the most examples one removal request carries
// "Unlike the rest": a cosine this far under the set's median. A typical set sits within about 0.1 of its median, and a
// wrong picture far under it (near 0), so 0.2 marks the odd one without marking the ordinary spread.
const UNLIKE_BY = 0.2;
const UNLIKE_MIN = 3; // fewer compared examples than this have no "rest" to be unlike

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

const percent = (similarity: number) => `${Math.round(Math.max(0, similarity) * 100)}%`;
const when = (image: TrainingImage) => (image.at != null ? seenAt(image.at) : 'date not kept');

/** A tile's caption, one short line: how like the rest it is (or that the current model doesn't use it), and when. */
export function captionOf(image: TrainingImage): string {
  const like = image.unused ? 'not used now' : image.similarity != null ? `${percent(image.similarity)} alike` : '';
  return [like, when(image)].filter(Boolean).join(' · ');
}

/** Everything about a tile, for its accessible name and hover title: when, where, how it got in, and how alike. */
export function detailOf(image: TrainingImage): string {
  const parts = [when(image), cameraName(image.cam), image.how];
  if (image.unused) parts.push('not used by the current model');
  else if (image.similarity != null) parts.push(`${percent(image.similarity)} like the rest`);
  return parts.filter(Boolean).join(', ');
}

/** The examples well under the rest of their set: UNLIKE_BY under the median similarity, in a set big enough. */
export function unlikeKeys(images: readonly TrainingImage[]): Set<string> {
  const compared = images.filter(i => !i.unused && i.similarity != null);
  if (compared.length < UNLIKE_MIN) return new Set();
  const sims = compared.map(i => i.similarity as number).sort((a, b) => a - b);
  const mid = sims.length / 2;
  const median =
    sims.length % 2 ? (sims[Math.floor(mid)] as number) : ((sims[mid - 1] as number) + (sims[mid] as number)) / 2;
  return new Set(compared.filter(i => (i.similarity as number) < median - UNLIKE_BY).map(i => i.key));
}

/** Square placeholders in the grid's own shape while the set loads. */
function TilesSkeleton() {
  return (
    <ul className={classes.training} aria-busy="true" aria-label="Loading the training images">
      {Array.from({ length: 6 }, (_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: identical placeholders with no identity of their own
        <li key={i} className={classes.trainingTile}>
          <Skeleton className={classes.trainingSkeleton} />
          <Skeleton height={10} width="70%" mt={6} radius="sm" />
        </li>
      ))}
    </ul>
  );
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
  const [result, setResult] = useState<Message>(null);
  const query = useQuery({
    queryKey: source.queryKey,
    queryFn: ({ signal }) => source.load(signal),
    enabled: open,
  });
  const images = query.data ?? [];
  const chosen = images.filter(i => picked.has(i.key));
  const unlike = unlikeKeys(images);

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
    setResult(null);
    try {
      await source.remove(chosen.map(i => i.key));
      setPicked(new Set());
      onRemoved?.();
    } catch (e) {
      setResult({ err: `Couldn't remove ${chosen.length === 1 ? 'it' : 'them'}: ${messageOf(e)}` });
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
          {open ? 'Hide' : 'Show'}
        </Button>
      }
    >
      {!open ? (
        <Empty>{`Every picture recognition matches ${name} against. Remove any that are wrong.`}</Empty>
      ) : query.isPending ? (
        <TilesSkeleton />
      ) : query.isError ? (
        <LoadError
          message="Couldn't load the training images"
          detail={messageOf(query.error)}
          onRetry={() => void query.refetch()}
          busy={query.isFetching}
        />
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
          {result && (
            <div className={classes.trainingStatus}>
              <StatusLine msg={result} />
            </div>
          )}
          <ul className={classes.training}>
            {images.slice(0, shown).map(image => {
              const caption = captionOf(image);
              const detail = detailOf(image);
              const odd = unlike.has(image.key);
              const selected = picked.has(image.key);
              return (
                <li key={image.key} className={classes.trainingTile} data-selected={selected || undefined}>
                  <label htmlFor={`training-${image.key}`} title={detail}>
                    <span className={classes.trainingPicture}>
                      {image.src ? (
                        <SafeImage src={image.src} alt="" loading="lazy" retries={1} />
                      ) : (
                        <span className={classes.noPicture}>No picture kept</span>
                      )}
                      {odd && (
                        <Badge
                          className={classes.trainingBadge}
                          variant="light"
                          color="yellow"
                          c="var(--lib-warn-text)"
                          tt="none"
                          fw={500}
                          size="sm"
                          aria-hidden
                        >
                          Unlike the rest
                        </Badge>
                      )}
                    </span>
                    <Checkbox
                      id={`training-${image.key}`}
                      className={classes.trainingPick}
                      size="xs"
                      checked={selected}
                      onChange={() => toggle(image.key)}
                      aria-label={`Select the example of ${name}: ${odd ? 'unlike the rest, ' : ''}${detail}`}
                    />
                    <Text size="xs" c="dimmed" component="span" className={classes.trainingCaption} aria-hidden>
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
