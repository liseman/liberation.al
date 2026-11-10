// One confirm prompt for the app (#2538), in place of the old window.confirm, through Mantine's modals manager
// (ModalsProvider is in shared/Providers.tsx). Cancel is focused on every prompt, so a stray or held Enter (the Calls
// form submits with Enter) keeps things as they are; a prompt whose action is easily undone can opt into a focused
// confirm button with `focusConfirm`. A destructive one also has a red confirm button. A prompt with an `id` opens once:
// asking again while it's up answers no at once, so a double click can't stack two prompts or run the action twice.
import { Text } from '@mantine/core';
import { modals } from '@mantine/modals';
import type { ReactNode } from 'react';

export interface ConfirmOptions {
  title: ReactNode;
  body?: ReactNode; // a sentence or two under the title; plain text is set as a paragraph
  confirm: string; // the confirm button's words ("Delete", "Add")
  cancel?: string; // the cancel button's words, "Cancel" unless the prompt has better ("Keep editing")
  danger?: boolean; // destructive: a red confirm button
  focusConfirm?: boolean; // focus the confirm button instead of Cancel: only for an action that's easily undone
  id?: string; // one prompt at a time per id
}

const open = new Set<string>();

/** Ask first; resolves true when confirmed, false when cancelled, closed, or already being asked. */
export function confirm({
  title,
  body,
  confirm,
  cancel = 'Cancel',
  danger = false,
  focusConfirm = false,
  id,
}: ConfirmOptions): Promise<boolean> {
  if (id && open.has(id)) return Promise.resolve(false);
  if (id) open.add(id);
  return new Promise(resolve => {
    let answered = false;
    const answer = (yes: boolean) => {
      if (answered) return;
      answered = true;
      if (id) open.delete(id);
      resolve(yes);
    };
    modals.openConfirmModal({
      modalId: id,
      title,
      children: typeof body === 'string' ? <Text size="sm">{body}</Text> : (body ?? null),
      labels: { confirm, cancel },
      confirmProps: { ...(danger ? { color: 'red' } : {}), ...(focusConfirm ? { 'data-autofocus': true } : {}) },
      cancelProps: focusConfirm ? undefined : { 'data-autofocus': true },
      onConfirm: () => answer(true),
      onCancel: () => answer(false),
      onClose: () => answer(false),
    });
  });
}
