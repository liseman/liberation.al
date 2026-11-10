// What a change said: a line of text (with an Undo token when it can be undone, and sticky while it's still in
// progress), or an error. Every page's status line and toast take this one shape (#2457 duplicates, row 1).
import { Text } from '@mantine/core';

export type Message<U = never> = { text: string; undo?: U | null; sticky?: boolean } | { err: string } | null;

/** A message's words, or null for none. */
export const messageWords = (m: Message<unknown>): string | null => (m ? ('err' in m ? m.err : m.text) : null);

/** Whether a message is an error. */
export const isErr = (m: Message<unknown>): m is { err: string } => !!m && 'err' in m;

/** A message as one line under what it's about: dimmed, or in the error colour. Nothing when there's nothing to say. */
export function StatusLine({ msg }: { msg: Message<unknown> }) {
  if (!msg) return null;
  return (
    <Text size="sm" c={isErr(msg) ? 'var(--lib-bad)' : 'dimmed'} role="status">
      {messageWords(msg)}
    </Text>
  );
}
