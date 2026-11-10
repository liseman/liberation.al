// Live and Review share one top-bar Ask trigger. Review keeps its day-owned panel and engine context.
import { Button, Kbd } from '@mantine/core';
import { useHotkeys } from '@mantine/hooks';
import { IconMessageCircle } from '@tabler/icons-react';
import { useEffect, useState } from 'react';
import { create } from 'zustand';
import { LiveAskPanel } from './ask/LiveAskPanel';
import { useAskOpener } from './ask/openAsk';
import type { TabId } from './router';
import { shell } from './shell';

/** The top bar's Ask button on Live and Review: default weight, with its / shortcut shown (#2453). */
export function AskTrigger({
  onClick,
  label,
  review = false,
  opened = false,
  hidden = false,
}: {
  onClick: () => void;
  label: string;
  review?: boolean;
  opened?: boolean;
  hidden?: boolean;
}) {
  return (
    <Button
      size="sm"
      variant="default"
      leftSection={<IconMessageCircle size={18} aria-hidden />}
      rightSection={
        <Kbd size="xs" aria-hidden>
          /
        </Kbd>
      }
      onClick={onClick}
      aria-label={label}
      aria-keyshortcuts="/"
      aria-haspopup="dialog"
      aria-expanded={opened}
      data-ask-trigger={review ? 'review' : 'live'}
      data-review-ask-trigger={review || undefined}
      style={{ visibility: hidden ? 'hidden' : undefined }}
    >
      Ask
    </Button>
  );
}

/** Whether Review's Ask about this day is open. The top bar owns the one trigger and Review owns the panel; this
 *  shared state joins them, so there is exactly one Ask entry point on the page (no portal into the bar, which could
 *  leave a second trigger behind). */
export const useReviewAsk = create<{ open: boolean; setOpen: (open: boolean | ((was: boolean) => boolean)) => void }>(
  set => ({ open: false, setOpen: open => set(s => ({ open: typeof open === 'function' ? open(s.open) : open })) }),
);

/** Review's trigger: it opens and closes Review's Ask about this day. */
export function ReviewAskTrigger({ hidden = false }: { hidden?: boolean }) {
  const opened = useReviewAsk(s => s.open);
  const setOpen = useReviewAsk(s => s.setOpen);
  return (
    <AskTrigger
      label="Ask about this day"
      onClick={() => setOpen(was => !was)}
      opened={opened}
      review
      hidden={hidden}
    />
  );
}

const focusAsk = (kind: 'live' | 'review') =>
  document.querySelector<HTMLElement>(`[data-ask-panel="${kind}"] textarea`)?.focus({ preventScroll: true });

/** `tab` is the page showing (under Settings too), and `settings` whether the Settings drawer is open over it. */
export function AskBar({ tab, setup, settings = false }: { tab: TabId; setup: boolean; settings?: boolean }) {
  const [open, setOpen] = useState(false);
  const [visited, setVisited] = useState(false);
  if (open && !visited) setVisited(true);
  // Review has its own Ask about the day; every other tab has the general one, whose window stays open across tabs
  // and under Settings. While Settings is open the triggers are hidden, but nothing is unmounted.
  const onReview = tab === 'review' && !setup;
  const onLive = !onReview && !setup;
  useEffect(() => {
    if (onReview) setOpen(false);
  }, [onReview]);
  // openAsk (ask/openAsk.ts): another page opens Ask with something to start from; the panel takes the request.
  const requested = useAskOpener(s => s.request);
  useEffect(() => {
    if (requested && onLive) setOpen(true);
  }, [requested, onLive]);
  // "/" opens or focuses Ask (with dashboard.hotkeys on). useHotkeys skips it while typing in a field or editable text
  // and with Ctrl, Cmd, or Alt held; shift+/ is there for keyboards where "/" takes Shift. A modal keeps the key.
  const slash = (e: KeyboardEvent) => {
    if ((!onLive && !onReview) || !shell().hotkeys) return;
    const modal = [...document.querySelectorAll<HTMLElement>('[aria-modal=true]')].some(el =>
      typeof el.checkVisibility === 'function' ? el.checkVisibility() : getComputedStyle(el).display !== 'none',
    );
    if (modal) return;
    e.preventDefault();
    if (onLive) {
      if (document.querySelector('[data-ask-panel="live"][data-ask-open="true"]')) focusAsk('live');
      else setOpen(true);
    } else {
      const trigger = document.querySelector<HTMLElement>('[data-review-ask-trigger]');
      if (trigger?.getAttribute('aria-expanded') === 'true') focusAsk('review');
      else trigger?.click();
    }
  };
  useHotkeys([
    ['/', slash, { preventDefault: false }],
    ['shift+/', slash, { preventDefault: false }],
  ]);

  // Under Settings the trigger is hidden but keeps its space, so the top bar doesn't move as the drawer opens and
  // closes. Review's panel state is kept.
  if (!onLive) return onReview ? <ReviewAskTrigger hidden={settings} /> : null;
  return (
    <>
      <AskTrigger label="Ask" opened={open} hidden={settings} onClick={() => setOpen(value => !value)} />
      {visited && <LiveAskPanel opened={open} onClose={() => setOpen(false)} />}
    </>
  );
}
