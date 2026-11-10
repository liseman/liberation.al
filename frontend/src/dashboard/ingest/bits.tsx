// Small pieces the Ingest tab and Settings' imported sources (settings/sources/) share: a status line, the dimmed
// text style, a time in words, and refreshing the ingest view after a change.

import { Text } from '@mantine/core';
import { useQueryClient } from '@tanstack/react-query';
import { dateTime } from '../lib/format';
import { INGEST_KEY } from './api';

export type { Message } from '../lib/message';

export const muted = { size: 'sm', c: 'dimmed' } as const;

/** Ingest's status line is the shared one. */
export { StatusLine as Status } from '../lib/message';

export const when = (ts: number | null): string => dateTime(ts, 'unknown time');

export function useRefresh(): () => Promise<void> {
  const client = useQueryClient();
  return () => client.invalidateQueries({ queryKey: INGEST_KEY });
}

/** What a failure means, and what to do, for the messages the ingest pipeline and its sources give (#2712). The
 *  first that matches wins; a message none knows is shown as it is. */
const PLAIN: { test: RegExp; what: string; fix: string | null }[] = [
  {
    test: /\b404\b/,
    what: "That link didn't load: there's nothing at that address.",
    fix: 'Check the address.',
  },
  {
    test: /\b40[13]\b|forbidden|sign.?in|log.?in|private video/i,
    what: 'That link needs a sign-in.',
    fix: "Download it where you're signed in, then add the file here.",
  },
  {
    test: /isn't installed/i,
    what: "This machine can't read pages like that one yet.",
    fix: 'Link straight to the file, or download it and add it here.',
  },
  {
    test: /unsupported url|unable to extract|no video formats/i,
    what: 'No photo, video, or audio was found at that link.',
    fix: 'Link straight to the file, or to the page the video plays on.',
  },
  {
    test: /bigger than|too large/i,
    what: "It's bigger than this machine takes in.",
    fix: 'Raise ingest.max_mb in Settings, or trim it.',
  },
  { test: /took too long|timed out|timeout/i, what: 'It took too long.', fix: 'Try again.' },
  {
    test: /not a photo, video, or audio file|no picture or sound|no video stream|no video keyframes/i,
    what: "This isn't a photo, video, or audio file that can be read.",
    fix: null,
  },
  {
    test: /couldn't convert|couldn't read the photo/i,
    what: "It couldn't be converted. The file may be damaged.",
    fix: 'Try another copy of it.',
  },
  { test: /interrupted/i, what: 'The upload was cut off.', fix: 'Add it again.' },
];

/** A failure in plain words, with the original kept under Details; a message none of PLAIN knows, as it is. */
export function Failure({ detail }: { detail: string }) {
  const known = PLAIN.find(p => p.test.test(detail));
  if (!known)
    return (
      <Text size="sm" c="var(--lib-bad)">
        {detail}
      </Text>
    );
  return (
    <div>
      <Text size="sm" c="var(--lib-bad)">
        {known.what}
        {known.fix && ` ${known.fix}`}
      </Text>
      <details>
        <summary>
          <Text span size="xs" c="dimmed">
            Details
          </Text>
        </summary>
        <Text size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>
          {detail}
        </Text>
      </details>
    </div>
  );
}
