// The top bar's way into the command palette (#2586, #2598): a labelled button like Ask's, with its shortcut shown
// (Ctrl K, or ⌘K on a Mac), and on a phone just its icon, as Ask drops its key there (app.module.css). Opening goes
// through the palette's store, so this button doesn't load the palette itself.
import { Button, Kbd } from '@mantine/core';
import { IconSearch } from '@tabler/icons-react';
import { MAC } from '../../shared/platform';
import { palette } from './store';

export function PaletteButton() {
  return (
    <Button
      size="sm"
      variant="default"
      leftSection={<IconSearch size={18} aria-hidden />}
      rightSection={
        <Kbd size="xs" aria-hidden>
          {MAC ? '⌘K' : 'Ctrl K'}
        </Kbd>
      }
      onClick={() => palette.open()}
      aria-label={`Go to ${MAC ? '⌘K' : 'Ctrl K'}`} // its visible words, shortcut included (label in name)
      aria-keyshortcuts={MAC ? 'Meta+K' : 'Control+K'}
      aria-haspopup="dialog"
      data-palette-trigger
    >
      Go to
    </Button>
  );
}
