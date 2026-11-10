// The naming inbox's bulk view (#2297, #2366): "By pet" or "By person". One name at a time, every waiting thing
// guessed as them, as large pictures, best match first. All start checked; click (or Space) the ones that aren't
// them, then one button saves both halves: the checked ones are named, and the unchecked ones are recorded as not
// them, so they stop being guessed as them (#2364, #2372). One toast, one Undo. Arrows move between pictures.

import { Button, Chip, Group, Text } from '@mantine/core';
import { useState } from 'react';
import { SafeImage } from '../../shared/SafeImage';
import { capitalize } from '../lib/format';
import { SelectableGrid } from '../lib/SelectableGrid';
import classes from './bulk.module.css';

/** One picture in the grid: its key (what the save sends back), its source, and how it reads. */
export interface BulkTile {
  key: string;
  src: string;
  label: string; // its accessible name, which ends with its match ("…, match 90%")
  caption: string;
}

/** A name to sort by: its id (what naming sends), how it reads, and its tiles, best match first. */
export interface BulkTarget {
  id: string;
  name: string;
  tiles: BulkTile[];
}

function remembered(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function remember(key: string, id: string): void {
  try {
    localStorage.setItem(key, id);
  } catch {
    // a private window: the pick just isn't kept
  }
}

function Tiles({
  target,
  noun,
  max,
  onSave,
}: {
  target: BulkTarget;
  noun: string;
  max: number;
  onSave: (target: BulkTarget, checked: string[], unchecked: string[]) => void;
}) {
  const [out, setOut] = useState<ReadonlySet<string>>(() => new Set()); // the ones that aren't them
  // One save covers at most `max`, so one Undo covers it: the best matches first, the rest after they're done.
  const shown = target.tiles.slice(0, max);
  const picked = shown.filter(t => !out.has(t.key));
  const left = shown.filter(t => out.has(t.key));
  const toggle = (key: string) =>
    setOut(have => {
      const next = new Set(have);
      if (!next.delete(key)) next.add(key);
      return next;
    });
  if (!shown.length)
    return (
      <Text size="sm" c="dimmed" mt="sm">
        No waiting {noun} are guessed as {target.name}.
      </Text>
    );
  const save = () =>
    onSave(
      target,
      picked.map(t => t.key),
      left.map(t => t.key),
    );
  return (
    <>
      <Group justify="space-between" gap="xs" my="sm">
        <Text size="sm" c="dimmed">
          {picked.length} of {shown.length} picked. Click the ones that aren't {target.name}
          {left.length ? `; those ${left.length} are saved as not ${target.name}.` : '.'}
          {target.tiles.length > shown.length &&
            ` Showing the best ${shown.length} of ${target.tiles.length}; the rest come up once these are saved.`}
        </Text>
        <Button size="sm" onClick={save}>
          {picked.length ? `Label ${picked.length} as ${target.name}` : `Mark ${left.length} as not ${target.name}`}
        </Button>
      </Group>
      <SelectableGrid
        label={`${title(noun)} guessed as ${target.name}`}
        tiles={shown.map(t => ({
          key: t.key,
          label: t.label,
          content: (
            <>
              <SafeImage className={classes.image} src={t.src} alt="" loading="lazy" retries={2} />
              <span className={classes.caption}>{t.caption}</span>
            </>
          ),
        }))}
        isOn={key => !out.has(key)}
        onToggle={toggle}
        tone="keep"
        marks="always"
        tile="var(--naming-tile, 220px)"
        twoOnPhone
      />
    </>
  );
}

const title = capitalize;

/** The bulk view. `noun` is what the tiles are ("sightings", "faces"), `which` names the chips ("Which pet"), and
 *  `storageKey` keeps the last name picked in this browser. */
export function Bulk({
  label,
  which,
  noun,
  storageKey,
  targets,
  max,
  onSave,
}: {
  label: string; // the section's name ("By pet")
  which: string;
  noun: string;
  storageKey: string;
  targets: BulkTarget[];
  max: number;
  onSave: (target: BulkTarget, checked: string[], unchecked: string[]) => void;
}) {
  const [chosen, setChosen] = useState<string | null>(() => remembered(storageKey));
  // The remembered name while it's still one, else the one with the most waiting.
  const busiest = [...targets].sort((a, b) => b.tiles.length - a.tiles.length)[0];
  const target = targets.find(t => t.id === chosen) ?? busiest;
  if (!target) return null;
  return (
    <section aria-label={label}>
      <Chip.Group
        value={target.id}
        onChange={id => {
          setChosen(id);
          remember(storageKey, id);
        }}
      >
        <Group gap={6} role="radiogroup" aria-label={which}>
          {targets.map(t => (
            <Chip key={t.id} value={t.id} type="radio" size="sm">
              {t.name} ({t.tiles.length})
            </Chip>
          ))}
        </Group>
      </Chip.Group>
      <Tiles key={target.id} target={target} noun={noun} max={max} onSave={onSave} />
    </section>
  );
}
