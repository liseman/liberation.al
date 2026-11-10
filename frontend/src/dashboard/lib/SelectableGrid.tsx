// One grid of pictures to pick from (duplicated frontend components, row 3): Who's this?'s and a pet sighting's "Every
// picture" (take the ones that aren't them out), the naming inbox's bulk view (uncheck the ones that aren't them), and
// a card's "Pick what's theirs" (faces and videos to claim). Each tile is a toggle button with a check mark; the arrow
// keys move between tiles (up and down by a row), Tab walks them in order, and Space or Enter toggles one.
import { Checkbox, UnstyledButton } from '@mantine/core';
import type { CSSProperties, KeyboardEvent, ReactNode } from 'react';
import classes from './selectableGrid.module.css';

/** One tile: its key (what a toggle sends back), its accessible name, and what it shows. */
export interface GridTile {
  key: string;
  label: string;
  content: ReactNode;
}

/** The arrow keys' step from the focused tile: one along a row, a whole row up or down. A layout that puts every
 *  tile in one row (or none, as in a test) steps one for up and down too. */
export function arrowStep(key: string, columns: number): number {
  const row = columns > 0 ? columns : 1;
  return { ArrowRight: 1, ArrowLeft: -1, ArrowDown: row, ArrowUp: -row }[key] ?? 0;
}

function onArrows(e: KeyboardEvent<HTMLDivElement>) {
  const tiles = [...e.currentTarget.querySelectorAll<HTMLElement>('[data-tile]')];
  const first = tiles[0]?.offsetTop;
  const columns = tiles.filter(t => t.offsetTop === first).length;
  const step = arrowStep(e.key, columns === tiles.length ? 1 : columns);
  if (!step) return;
  const at = tiles.indexOf(document.activeElement as HTMLElement);
  const next = tiles[Math.min(tiles.length - 1, Math.max(0, at + step))];
  if (next) {
    e.preventDefault();
    next.focus();
  }
}

export function SelectableGrid({
  label,
  tiles,
  isOn,
  onToggle,
  tone = 'pick',
  marks = 'on',
  tile = 64,
  twoOnPhone = false,
  disabled = false,
}: {
  label: string; // the group's accessible name
  tiles: readonly GridTile[];
  isOn: (key: string) => boolean;
  onToggle: (key: string) => void;
  /** 'pick' rings a chosen tile in the accent; 'remove' rings it in red and dims it (it's about to be taken out);
   *  'keep' dims the tiles that are off instead (all start on, and the ones turned off are left behind). */
  tone?: 'pick' | 'remove' | 'keep';
  marks?: 'on' | 'always'; // the check mark on chosen tiles only, or on every tile (checked or not)
  tile?: number | string; // the narrowest a tile gets, in px (or a CSS length)
  twoOnPhone?: boolean; // two a row on a phone, whatever `tile` says (large pictures)
  disabled?: boolean;
}) {
  return (
    // biome-ignore lint/a11y/useSemanticElements: a group of toggle pictures, moved between with the arrows
    <div
      className={classes.grid}
      role="group"
      aria-label={label}
      onKeyDown={onArrows}
      data-tone={tone}
      data-two-on-phone={twoOnPhone || undefined}
      style={{ '--grid-tile': typeof tile === 'number' ? `${tile}px` : tile } as CSSProperties}
    >
      {tiles.map(t => {
        const on = isOn(t.key);
        return (
          <UnstyledButton
            key={t.key}
            type="button"
            data-tile
            className={classes.tile}
            data-on={on || undefined}
            aria-pressed={on}
            aria-label={t.label}
            disabled={disabled}
            onClick={() => onToggle(t.key)}
          >
            {t.content}
            {/* The corner is a box of our own: Mantine positions its indicator itself, which put it in the flow. */}
            {(on || marks === 'always') && (
              <span className={classes.mark} aria-hidden="true">
                <Checkbox.Indicator checked={on} size="md" radius="xl" color={tone === 'remove' ? 'red' : undefined} />
              </span>
            )}
          </UnstyledButton>
        );
      })}
    </div>
  );
}
