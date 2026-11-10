// A confirmation that floats at the bottom of the window instead of taking a line in the page, so saying it
// never moves what's under it. Every page's toasts share Mantine's one queue: they stack bottom-centre
// rather than replacing each other, the clock holds while the pointer or focus is on one (so its Undo can be
// reached), and each goes by itself after a while. An error stays until it's closed or replaced. The callers keep
// a controlled API: a Toast says what it is given, and onClose says when it went.
import { Button } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { type ReactNode, useEffect, useId, useRef } from 'react';
import { isErr, type Message, messageWords } from './message';
import classes from './toast.module.css';

/** How long a confirmation stays, in ms; one with an action (Undo) stays longer. */
export const TOAST_MS = 6000;
export const TOAST_ACTION_MS = 10000;

/** Say a Message: its words, as an error when it is one, sticky while in progress, and an Undo (calling onUndo with
 *  the message's token) when it can be undone. Or give text, error, sticky, and action directly. */
export function Toast<U = never>({
  message,
  onUndo,
  busy = false,
  text: givenText = null,
  error: givenError = false,
  action: givenAction,
  sticky: givenSticky = false,
  label,
  onClose,
}: {
  message?: Message<U>;
  onUndo?: (token: U) => void;
  busy?: boolean; // an undo is under way: its button waits
  text?: string | null; // nothing to say: the toast is gone
  label?: string; // its accessible name, when the page has more than one
  error?: boolean;
  action?: ReactNode; // another action beside the text (Retry), after Undo
  sticky?: boolean; // in progress ("Saving"): it waits for what replaces it
  onClose: () => void;
}) {
  const fromMessage = message !== undefined;
  const text = fromMessage ? messageWords(message) : givenText;
  const error = fromMessage ? isErr(message) : givenError;
  const sticky = givenSticky || (!!message && 'sticky' in message && !!message.sticky);
  const token = message && 'text' in message ? message.undo : null;
  // An empty token (the server had nothing to undo) offers no Undo, as People's old ToastLine didn't.
  const canUndo = !!token && !!onUndo;
  const hasAction = canUndo || !!givenAction;
  const latest = useRef({ onClose, onUndo, token });
  latest.current = { onClose, onUndo, token };
  const body = (
    <span className={classes.body}>
      <span className={classes.text}>{text}</span>
      {canUndo && (
        <Button
          size="compact-xs"
          variant="subtle"
          flex="0 0 auto"
          disabled={busy}
          onClick={() => latest.current.onUndo?.(latest.current.token as U)}
        >
          Undo
        </Button>
      )}
      {givenAction}
    </span>
  );

  // Each thing said is its own notification, so a new one starts its own clock (an update would keep the old one's)
  // and the one it replaces slides away. Only a close from the queue (its time ran out, or Dismiss) is onClose: one
  // this Toast took down itself, being replaced or unmounted, isn't.
  const base = useId();
  const count = useRef(0);
  const current = useRef<string | null>(null);
  const ms = hasAction ? TOAST_ACTION_MS : TOAST_MS;
  const autoClose = error || sticky ? false : ms;
  // biome-ignore lint/correctness/useExhaustiveDependencies: only new words are a new toast; the rest updates below
  useEffect(() => {
    if (!text) return;
    count.current += 1;
    const id = `${base}-${count.current}`;
    current.current = id;
    notifications.show({
      id,
      message: body,
      autoClose,
      color: error ? 'red' : undefined,
      role: error ? 'alert' : 'status',
      'aria-label': label,
      className: classes.toast,
      closeButtonProps: { 'aria-label': 'Dismiss' },
      'data-error': error || undefined,
      onClose: () => {
        if (current.current !== id) return;
        current.current = null;
        latest.current.onClose();
      },
    });
    return () => {
      if (current.current === id) current.current = null;
      notifications.hide(id);
    };
  }, [text, error, label]);
  // The words stay; a busy Undo, a new action beside them, or holding while an undo runs (sticky) updates the one
  // showing in place, so its buttons stay the same buttons. Its clock starts over when it lets go.
  useEffect(() => {
    if (current.current) notifications.update({ id: current.current, message: body, autoClose });
  });
  return null;
}
