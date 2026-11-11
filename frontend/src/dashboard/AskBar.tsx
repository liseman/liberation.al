// The top bar's Ask: one button and one Ask on every page. On Review the button opens Ask with the day it shows as a
// hint the person can remove; everywhere else it opens Ask as it is.
import { Button, Kbd } from '@mantine/core';
import { useHotkeys } from '@mantine/hooks';
import { IconMessageCircle } from '@tabler/icons-react';
import { useState } from 'react';
import { Ask } from './ask/Ask';
import { closeAsk, openAskHere, useAsk } from './ask/openAsk';
import type { TabId } from './router';
import { shell } from './shell';

const focusAsk = () => document.querySelector<HTMLElement>('[data-ask-panel] textarea')?.focus({ preventScroll: true });

/** `tab` is the page showing (under Settings too), and `settings` whether the Settings drawer is open over it. */
export function AskBar({ tab, setup, settings = false }: { tab: TabId; setup: boolean; settings?: boolean }) {
  const open = useAsk(s => s.open);
  const [visited, setVisited] = useState(false);
  if (open && !visited) setVisited(true);
  // "/" opens or focuses Ask (with dashboard.hotkeys on). useHotkeys skips it while typing in a field or editable text
  // and with Ctrl, Cmd, or Alt held; shift+/ is there for keyboards where "/" takes Shift. A modal keeps the key.
  const slash = (e: KeyboardEvent) => {
    if (setup || !shell().hotkeys) return;
    const modal = [...document.querySelectorAll<HTMLElement>('[aria-modal=true]')].some(el =>
      typeof el.checkVisibility === 'function' ? el.checkVisibility() : getComputedStyle(el).display !== 'none',
    );
    if (modal) return;
    e.preventDefault();
    if (open) focusAsk();
    else openAskHere(tab);
  };
  useHotkeys([
    ['/', slash, { preventDefault: false }],
    ['shift+/', slash, { preventDefault: false }],
  ]);
  if (setup) return null;
  // Under Settings the button is hidden but keeps its space, so the top bar doesn't move as the drawer opens and
  // closes, and Ask's window waits behind the drawer with what it holds.
  return (
    <>
      <Button
        size="sm"
        variant="default"
        leftSection={<IconMessageCircle size={18} aria-hidden />}
        rightSection={
          <Kbd size="xs" aria-hidden>
            /
          </Kbd>
        }
        onClick={() => (open ? closeAsk() : openAskHere(tab))}
        aria-label="Ask"
        aria-keyshortcuts="/"
        aria-haspopup="dialog"
        aria-expanded={open}
        data-ask-trigger
        style={{ visibility: settings ? 'hidden' : undefined }}
      >
        Ask
      </Button>
      {visited && <Ask hidden={settings} />}
    </>
  );
}
