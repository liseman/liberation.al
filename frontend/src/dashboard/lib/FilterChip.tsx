// One filter chip for the app (#2450 board item 6, duplicates item 8): Mantine's Chip (a checkbox to assistive tech),
// drawn neutral with a dot in its series' colour and an optional count, dashed and dimmed when off.
import { Chip } from '@mantine/core';
import type { ReactNode } from 'react';
import classes from './FilterChip.module.css';

/** A Mantine colour name ("teal") or any CSS colour, as the dot's fill. */
const fill = (color: string) => (/^[a-z]+$/.test(color) ? `var(--mantine-color-${color}-filled)` : color);

export function FilterChip({
  checked,
  onChange,
  color,
  count,
  size = 'xs',
  children,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  /** The colour of what it filters, for its dot. */
  color?: string;
  count?: number;
  size?: 'xs' | 'sm';
  children: ReactNode;
}) {
  return (
    <Chip
      size={size}
      checked={checked}
      onChange={onChange}
      classNames={{ label: classes.label, iconWrapper: classes.iconWrapper }}
      // Inline, so they outrank Mantine's filled and checked styles: neutral when on, dashed and dimmed when off.
      styles={{
        label: {
          background: 'transparent',
          borderColor: 'var(--mantine-color-default-border)',
          borderStyle: checked ? 'solid' : 'dashed',
          color: checked ? 'var(--mantine-color-text)' : 'var(--mantine-color-dimmed)',
        },
      }}
    >
      {color && <span className={classes.dot} style={{ background: fill(color) }} aria-hidden />}
      {children}
      {count != null && <span className={classes.count}>{count.toLocaleString()}</span>}
    </Chip>
  );
}
