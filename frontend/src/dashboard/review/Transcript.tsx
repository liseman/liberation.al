// The day's transcript: the shared transcript (transcript/Transcript.tsx) over a day's lines. A line seeks there and
// plays when clicked; the current line follows the playhead (the engine marks it, so following never re-renders the
// list). Review for one person shows only their turns, or everyone's with the others dimmed. A name or a
// correction reads the day's lines again, so it shows here and on the timeline. A search dims the turns without a match.
import { Switch } from '@mantine/core';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { EMPTY_SPEAKERS, speakerName } from '../lib/speakers';
import { useLive } from '../live/hooks';
import type { TranscriptEntry } from '../live/types';
import type { Turn } from '../transcript/model';
import { type Cursor, Transcript as Lines } from '../transcript/Transcript';
import { getTranscript, reviewKey } from './api';
import { useEngine, useReview } from './context';
import classes from './review.module.css';

/** query: Review's search, dimming the turns without a match and highlighting the matches in place. */
export function Transcript({ entries, query = '' }: { entries: TranscriptEntry[] | null; query?: string }) {
  const engine = useEngine();
  const client = useQueryClient();
  const day = useReview(s => s.day);
  const person = useReview(s => s.person);
  const [others, setOthers] = useState(false); // with a person: everyone's lines too, theirs undimmed
  // The live feed sends the voices afresh with every line heard; the turns are rebuilt only when they change.
  const heard = useLive('speakers') ?? EMPTY_SPEAKERS;
  const heardKey = JSON.stringify(heard);
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed by content, not by the object
  const speakers = useMemo(() => heard, [heardKey]);

  const latest = useRef<Record<string, number>>({}); // per day, the newest read started: an older reply is dropped
  const timers = useRef(new Set<ReturnType<typeof setTimeout>>());
  useEffect(() => {
    const pending = timers.current;
    return () => {
      for (const t of pending) clearTimeout(t);
    };
  }, []);
  /** Read a day's lines again: their names changed. A tag is relabelled by the microphone service a moment later.
   *  Read beside the query, not through it, so the day doesn't show as loading again meanwhile. */
  const reread = useCallback(
    (which: string | null, later = false) => {
      if (!which) return;
      const go = () => {
        const n = (latest.current[which] ?? 0) + 1;
        latest.current[which] = n;
        getTranscript(which).then(
          lines => {
            if (latest.current[which] === n) client.setQueryData(reviewKey.transcript(which), lines);
          },
          () => {}, // the lines shown stay; the next opening reads them again
        );
      };
      go();
      if (later) {
        const t = setTimeout(() => {
          timers.current.delete(t);
          go();
        }, 4000);
        timers.current.add(t);
      }
    },
    [client],
  );
  const changed = useCallback((tagged: boolean) => reread(day, tagged), [reread, day]);

  const select = useCallback(
    (turns: Turn[]) => {
      const theirs = (t: Turn) =>
        !person ||
        speakerName(speakers, t.lines[0]?.e ?? { ts_start: 0 })
          .trim()
          .toLocaleLowerCase() === person.trim().toLocaleLowerCase();
      return turns.filter(t => others || theirs(t)).map(t => ({ t, other: !theirs(t) }));
    },
    [person, others, speakers],
  );
  const attach = useCallback(
    (cursor: Cursor | null, centre: boolean) => engine.attachTranscript(cursor, centre),
    [engine],
  );

  return (
    <Lines
      entries={entries}
      speakers={speakers}
      view={`${day}|${person ?? ''}|${others}`}
      label="Transcript of the day"
      className={classes.transcript}
      select={select}
      query={query}
      empty={person && entries?.length ? `${person} said nothing this day.` : 'Nothing was transcribed this day.'}
      onLine={e => engine.seekPlay(e.ts_start)}
      scope={day ?? ''}
      onChanged={changed}
      attach={attach}
      top={
        person && (
          <Switch
            size="xs"
            mb="xs"
            checked={others}
            onChange={e => setOthers(e.currentTarget.checked)}
            label="Show the others' lines too, dimmed"
          />
        )
      }
    />
  );
}
