// Naming a voice from a transcript (#357, #300): Live's turns and Review's do it the same way. A voice nobody has
// named is renamed (or claimed, when the name is someone's already); lines with no voice, or a known voice given
// someone else's name (heard wrong), have just those lines tagged, since claiming the voice would merge all of one
// person's voice into another's, which can't be undone. Naming a voice also confirms only the visible turn's lines.
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { badgeText, relabel } from '../lib/speakers';
import { liveKey } from '../live/feed';
import { reduce } from '../live/reduce';
import type { LineRef, Speakers } from '../live/types';
import { claim, labelVoice, refreshNames, tagLines } from './api';
import { useKnownPeople } from './people';

/** What naming did: nothing (no name, or the same one), tagged the lines (their new name arrives later, through the
 *  live feed or a fresh read), or renamed the voice (speakers are updated at once). */
export type Named = { done: 'nothing' } | { done: 'tagged'; as: string } | { done: 'voice' } | { error: string };

export interface Naming {
  voice: number | string | null; // the voice's id, or null for lines no voice was matched to
  current: string; // the name the lines show now
  lines: readonly LineRef[]; // each by its key, never another microphone's line (#247)
  added?: boolean; // the name was just added as a person, so it counts as known before the roster is read again
}

/** A function that names a voice or its lines. onStart says which it's doing as it starts ("tagging" shows
 *  "Tagging as <name>"), since tagging can take a while. */
export function useNameVoice(): (
  what: Naming,
  name: string | null,
  onStart?: (how: 'tagging' | 'saving', as: string) => void,
) => Promise<Named> {
  const client = useQueryClient();
  const known = useKnownPeople();
  return useCallback(
    async ({ voice, current, lines, added }, n, onStart) => {
      if (!n || n === current) return { done: 'nothing' };
      const exists = added || known.some(k => k.toLowerCase() === n.toLowerCase());
      const heardWrong = badgeText(current) === current.trim() && exists && n.toLowerCase() !== current.toLowerCase();
      const namingVoice = voice != null && !heardWrong;
      if (namingVoice) {
        onStart?.('saving', n);
        const j =
          /^Speaker \d+$/.test(current) || exists ? await claim(client, 'voice', voice, n) : await labelVoice(voice, n);
        if (!j.ok) return { error: `That didn't work: ${j.error}` };
        // A claim replies with people, not the voices: the voice takes its name here, so every transcript shows it at
        // once rather than "Unmatched" until the live feed sends the voices again.
        if (j.speakers) client.setQueryData(liveKey('speakers'), reduce('speakers', undefined, j.speakers));
        else client.setQueryData<Speakers>(liveKey('speakers'), s => (s ? relabel(s, voice, n) : s));
        if (!lines.length) return { done: 'voice' };
      }
      // Naming a cluster does not prove that every other line in it was this person. Confirm only the turn the
      // person actually looked at; the backend checks a multi-line request for mixed-speaker outliers.
      onStart?.('tagging', n);
      const j = await tagLines(lines, n);
      if (!j.ok)
        return {
          error: namingVoice
            ? `The voice was named ${n}, but its lines could not be queued for confirmation: ${j.error}`
            : `Tagging failed: ${j.error}`,
        };
      void refreshNames(client, 4000);
      return { done: 'tagged', as: j.note ? `${n}. ${j.note}` : n };
    },
    [client, known],
  );
}
