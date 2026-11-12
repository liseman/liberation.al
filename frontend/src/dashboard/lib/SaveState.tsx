// A control group's save and apply state, in one small fixed slot beside the group's title: a spinner while the change
// is saved and the device applies it, a check for a moment once it has, and an error mark whose reason shows on hover
// (and is read out) when it fails. The slot is always in the page at its full size, whatever the state, so the marks
// coming and going never move the controls beside them; screen readers hear each change through its polite live region,
// which is always there too (a live region that appears, rather than changes, is not read reliably).
import { Tooltip, VisuallyHidden } from '@mantine/core';
import { IconAlertTriangle, IconCheck } from '@tabler/icons-react';
import { useEffect, useState } from 'react';
import classes from './saveState.module.css';

export type SavePhase = 'idle' | 'saving' | 'saved' | 'error';

/** What a group is doing with the change just made. `text` is what a screen reader hears (and, when the change
 *  failed, the reason on the error mark); left out, each phase says its own words. */
export interface SaveStatus {
  phase: SavePhase;
  text?: string;
}

/** Nothing has been changed yet. */
export const SAVE_IDLE: SaveStatus = { phase: 'idle' };
/** A change is queued or being written. */
export const saving = (text?: string): SaveStatus => ({ phase: 'saving', text });
/** It was saved and the device has applied it. */
export const saved = (text?: string): SaveStatus => ({ phase: 'saved', text });
/** It did not save, or did not apply: `text` is why. */
export const failed = (text: string): SaveStatus => ({ phase: 'error', text });

/** How long the check stays before the slot goes back to empty. */
export const SAVE_MARK_MS = 2000;

const WORDS: Record<Exclude<SavePhase, 'idle'>, string> = {
  saving: 'Saving.',
  saved: 'Saved and applied.',
  error: 'Not saved.',
};

export function SaveState({ status }: { status: SaveStatus }) {
  const { phase, text } = status;
  const [shown, setShown] = useState<SavePhase>(phase);
  // Retire the check by itself, so the group says nothing once it is settled. A new change shows its own state at once.
  useEffect(() => {
    setShown(phase);
    if (phase !== 'saved') return;
    const timer = setTimeout(() => setShown('idle'), SAVE_MARK_MS);
    return () => clearTimeout(timer);
  }, [phase]);
  const words = phase === 'idle' ? '' : (text ?? WORDS[phase]);
  return (
    <span className={classes.slot} data-phase={shown} aria-live="polite" aria-atomic="true">
      <span className={classes.mark} aria-hidden="true">
        {shown === 'saving' && <span className={classes.spinner} data-testid="save-spinner" />}
        {shown === 'saved' && <IconCheck className={classes.check} size={14} stroke={3} />}
        {shown === 'error' && (
          <Tooltip label={words} multiline w={240} position="bottom-end">
            <IconAlertTriangle className={classes.error} size={14} data-testid="save-error" />
          </Tooltip>
        )}
      </span>
      <VisuallyHidden>{words}</VisuallyHidden>
    </span>
  );
}
