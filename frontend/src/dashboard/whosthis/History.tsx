// "Recent decisions" under the inbox: the lines labeled lately (each can be played and undone), everything since a
// time undone at once, and how many things were skipped, with a way to bring them back. Loaded when first opened.
import { Button, Group, Text } from '@mantine/core';
import type { DateTimePickerProps } from '@mantine/dates';
import { useQuery } from '@tanstack/react-query';
import { lazy, useState } from 'react';
import { Deferred } from '../Deferred';
import { Disclosure } from '../lib/Disclosure';
import { hms, plural } from '../lib/format';
import { PlayClip } from '../lib/PlayToggle';
import { someLines } from '../live/types';
import { getRefineLog, getSkips, lineClip, REFINE_LOG_KEY, SKIPS_KEY } from '../people/api';
import { showLabel } from './labels';
import classes from './whosthis.module.css';

const DateTimePicker = lazy(() =>
  import('@mantine/dates').then(m => ({ default: (props: DateTimePickerProps) => <m.DateTimePicker {...props} /> })),
);

const UNDO_WINDOW_MS = 45 * 60 * 1000; // "undo everything since" starts this long ago

/** The local minute cutoff used by the undo calendar. */
export function localInput(d: Date): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

/** The server counts what was put aside per kind (an older one listed them). */
const count = (v: unknown[] | number | undefined): number => (Array.isArray(v) ? v.length : (v ?? 0));

export function History({
  onUndo,
  onUnskip,
}: {
  onUndo: (body: { lines: string[] } | { since: number }) => Promise<boolean>; // true once something was undone
  onUnskip: () => void;
}) {
  const [opened, setOpened] = useState(false);
  const [since, setSince] = useState<string | null>(() => localInput(new Date(Date.now() - UNDO_WINDOW_MS)));
  const [confirm, setConfirm] = useState(false);
  const log = useQuery({ queryKey: REFINE_LOG_KEY, queryFn: getRefineLog, enabled: opened });
  const skips = useQuery({ queryKey: SKIPS_KEY, queryFn: getSkips, enabled: opened });
  // A log that can't be read shows as empty, as the old page did.
  const labels = log.data?.labels ?? [];
  const total = log.data?.total ?? 0;
  const loading = log.isPending;
  const nSkipped = skips.data ? count(skips.data.face) + count(skips.data.voice) + count(skips.data.line) : 0;
  const sinceTs = since ? new Date(since.replace(' ', 'T')).getTime() / 1000 : Number.NaN;
  const sinceN = labels.filter(e => e.ts >= sinceTs).length;

  const undoSince = async () => {
    if (await onUndo({ since: sinceTs })) setConfirm(false);
  };

  return (
    <Disclosure
      className={classes.history}
      summary={<>Recent decisions{total ? ` (${total})` : ''}</>}
      onChange={next => {
        if (next) setOpened(true);
      }}
    >
      {nSkipped > 0 && (
        <Text size="sm" mt="xs">
          {plural(nSkipped, 'thing', 'things')} skipped.{' '}
          <Button size="compact-xs" variant="subtle" onClick={onUnskip}>
            Bring them back
          </Button>
        </Text>
      )}
      {loading ? (
        <Text size="sm" c="dimmed">
          Loading
        </Text>
      ) : labels.length ? (
        <ul className={classes.hist}>
          {labels.map(e => (
            <li key={`${e.line ?? e.ts_start}:${e.ts}`}>
              <span className={classes.when}>{hms(e.ts_start)}</span>
              <PlayClip
                size="sm"
                url={lineClip(e.line ?? e.ts_start)}
                playKey={`hist:${e.line ?? e.ts_start}`}
                ts={e.ts_start}
                label={`the line at ${hms(e.ts_start)}`}
              />
              <span className={classes.histText} title={e.text}>
                {e.text}
              </span>
              <span className={classes.got}>
                <b>{showLabel(e.new?.speaker)}</b> <span className={classes.was}>was {showLabel(e.old?.speaker)}</span>
              </span>
              <Button size="compact-xs" variant="subtle" onClick={() => onUndo(someLines([e.line ?? e.ts_start]))}>
                Undo
              </Button>
            </li>
          ))}
        </ul>
      ) : (
        <Text size="sm" c="dimmed">
          No lines labeled yet. Faces and voices named here are listed under the person.
        </Text>
      )}
      {labels.length > 0 && (
        <Group className={classes.since} gap="xs" wrap="wrap">
          <Deferred name="undo cutoff calendar">
            <DateTimePicker
              label="Undo everything since"
              aria-label="Undo labels given after"
              value={since}
              onChange={setSince}
              valueFormat="MMM D, YYYY HH:mm"
              size="xs"
              timePickerProps={{ hoursInputLabel: 'Hours', minutesInputLabel: 'Minutes' }}
              submitButtonProps={{ 'aria-label': 'Use cutoff' }}
            />
          </Deferred>
          {confirm ? (
            <>
              <span>
                Undo {plural(sinceN, 'label', 'labels')}
                {total > labels.length ? ' or more' : ''}?
              </span>
              <Button size="compact-xs" onClick={undoSince}>
                Undo them
              </Button>
              <Button size="compact-xs" variant="subtle" onClick={() => setConfirm(false)}>
                Cancel
              </Button>
            </>
          ) : (
            <Button size="compact-xs" variant="subtle" disabled={!sinceN} onClick={() => setConfirm(true)}>
              Undo them
            </Button>
          )}
        </Group>
      )}
    </Disclosure>
  );
}
