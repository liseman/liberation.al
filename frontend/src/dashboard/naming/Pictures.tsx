// Every picture of a selected naming card, to choose which ones its answer covers. They all start selected;
// clicking one (or Space or Enter on it) deselects it, and that alone leaves it out of whatever the card's buttons
// do. A deselected picture stays unnamed and waits again. Only the deselected ones are kept track of, so pictures
// not loaded yet count as selected.
import { Text } from '@mantine/core';
import type { ReactNode } from 'react';
import { SafeImage } from '../../shared/SafeImage';
import { plural } from '../lib/format';
import { SelectableGrid } from '../lib/SelectableGrid';
import who from '../whosthis/whosthis.module.css';

/** One picture: its key (what deselecting keeps), its accessible name, and its image. */
export interface Picture {
  key: string;
  label: string;
  src: string;
}

export function Pictures({
  label,
  pictures,
  total = pictures.length,
  out,
  onToggle,
  disabled = false,
  children,
}: {
  label: string; // what they are pictures of ("Face 3")
  pictures: readonly Picture[];
  total?: number; // how many there are in all, when only some are loaded
  out: ReadonlySet<string>; // the deselected ones
  onToggle: (key: string) => void;
  disabled?: boolean;
  children?: ReactNode; // under the grid: pictures still being made, a way to load more
}) {
  const selected = Math.max(0, total - out.size);
  return (
    <section className={who.picker} aria-label={`Every picture of ${label}`}>
      <Text size="xs" c="dimmed" role="status">
        {selected} of {plural(total, 'picture', 'pictures')} selected.{' '}
        {out.size ? 'The rest stay unnamed.' : 'Click one to leave it out.'}
      </Text>
      <SelectableGrid
        label={`Pictures of ${label}: deselect the ones to leave out`}
        tiles={pictures.map(p => ({
          key: p.key,
          label: p.label,
          content: <SafeImage className={who.square} src={p.src} alt="" loading="lazy" />,
        }))}
        isOn={key => !out.has(key)}
        onToggle={onToggle}
        tone="keep"
        marks="always"
        disabled={disabled}
      />
      {children}
    </section>
  );
}

/** A set with `key` flipped: in if it was out, out if it was in. */
export function toggled(set: ReadonlySet<string>, key: string): ReadonlySet<string> {
  const next = new Set(set);
  if (!next.delete(key)) next.add(key);
  return next;
}
