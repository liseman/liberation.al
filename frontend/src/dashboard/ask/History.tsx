import { Button, Group, Loader, Text, Timeline, UnstyledButton } from '@mantine/core';
import { modals } from '@mantine/modals';
import { useEffect, useId, useRef, useState } from 'react';
import { messageOf } from '../../shared/http';
import { confirm } from '../lib/confirm';
import { Disclosure } from '../lib/Disclosure';
import { dayWords, hm, momentWords } from '../lib/format';
import { PageError } from '../lib/page';
import { viewName } from '../review/views';
import { AppLink, Stamp } from './bits';
import { isRec, plain } from './format';
import classes from './history.module.css';
import { deleteHistory, type HistoryEntry, type HistoryPage, readEntry, readHistory } from './historyApi';
import { citedMoments } from './moments';
import { Result } from './Result';

function moments(entry: HistoryEntry) {
  // Stored citation metadata survives removal of its answer or source.
  return entry.citations.length
    ? entry.citations.map(c => ({
        t: c.ts,
        label: [
          typeof c.ts === 'number' ? hm(c.ts) : 'Time unavailable',
          c.camera ? viewName(c.camera).toLowerCase() : c.source,
          `[${c.label}]`,
        ]
          .filter(Boolean)
          .join(' '),
        removed: c.removed,
        verified: c.verified !== false,
      }))
    : citedMoments(entry.result).map(c => ({ ...c, removed: false, verified: false }));
}
function safeResult(entry: HistoryEntry): unknown {
  if (!isRec(entry.result) || !Array.isArray(entry.result.evidence)) return entry.result;
  return {
    ...entry.result,
    evidence: entry.result.evidence.map(e => {
      if (!isRec(e)) return e;
      const citation = entry.citations.find(c => c.label === e.id);
      if (citation?.removed) return { ...e, href: undefined, text: 'Source removed' };
      if (!citation || citation.verified === false) return { ...e, href: undefined, text: 'Source unavailable' };
      return e;
    }),
  };
}

/** Saved exchanges are read independently of the active job and are never submitted as prompt context. */
/** How many past questions show before Show all. */
const LATEST = 5;

export function History({
  day = null,
  refresh,
  onSeek,
  onMoments,
}: {
  day?: string | null;
  refresh?: unknown;
  onSeek?: (ts: number) => void;
  onMoments?: (times: number[] | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState<HistoryPage | null>(null);
  const [selected, setSelected] = useState<HistoryEntry | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  // The latest few, with Show all for the rest of what's loaded (Older questions then reads further back).
  const [showAll, setShowAll] = useState(false);
  const confirmId = useId();
  const confirmOpen = useRef(false);
  const deleting = useRef(false);
  const mounted = useRef(true);
  // biome-ignore lint/correctness/useExhaustiveDependencies: a different day invalidates its pending confirmation.
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      confirmOpen.current = false;
      modals.close(confirmId);
    };
  }, [confirmId, day]);
  const [revision, setRevision] = useState(0);
  const requestNumber = useRef(0);
  const select = (entry: HistoryEntry | null) => {
    setSelected(entry);
    onMoments?.(
      entry ? moments(entry).flatMap(c => (!c.removed && c.verified && typeof c.t === 'number' ? [c.t] : [])) : null,
    );
  };
  const close = () => {
    requestNumber.current++;
    select(null);
    confirmOpen.current = false;
    modals.close(confirmId);
    setLoading(false);
  };
  useEffect(
    () => () => {
      requestNumber.current++;
    },
    [],
  );
  // The list read in flight (the day's first page or an Older questions page): a new list read aborts it, so a
  // reply for a day that's no longer shown can't replace the current list.
  const listRead = useRef<AbortController | null>(null);
  const startListRead = () => {
    listRead.current?.abort();
    const controller = new AbortController();
    listRead.current = controller;
    return controller;
  };
  useEffect(() => () => listRead.current?.abort(), []);
  // biome-ignore lint/correctness/useExhaustiveDependencies: completed jobs and deletions invalidate the server list.
  useEffect(() => {
    if (!open) {
      listRead.current?.abort();
      return;
    }
    const controller = startListRead();
    setLoading(true);
    setError('');
    void readHistory(day, undefined, controller.signal)
      .then(result => {
        if (controller.signal.aborted) return;
        if (!Array.isArray(result?.items) || !('next_before' in result))
          throw new Error('Invalid saved questions response');
        setPage(result);
      })
      .catch(e => {
        if (!controller.signal.aborted) setError(messageOf(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [open, day, refresh, revision]);

  const show = async (id: string) => {
    select(null);
    const n = ++requestNumber.current;
    setError('');
    setLoading(true);
    confirmOpen.current = false;
    modals.close(confirmId);
    try {
      const entry = await readEntry(id);
      if (n === requestNumber.current) select(entry);
    } catch (e) {
      if (n === requestNumber.current) setError(messageOf(e));
    } finally {
      if (n === requestNumber.current) setLoading(false);
    }
  };
  const remove = async (target: string) => {
    if (deleting.current || !mounted.current) return;
    deleting.current = true;
    setLoading(true);
    setError('');
    try {
      await deleteHistory(target === 'all' ? null : target, day);
      if (target === 'all' || selected?.id === target) select(null);
      confirmOpen.current = false;
      modals.close(confirmId);
      setRevision(n => n + 1);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      deleting.current = false;
      setLoading(false);
    }
  };
  const confirmRemove = (target: string) => {
    if (confirmOpen.current || deleting.current) return;
    confirmOpen.current = true;
    void confirm({
      id: confirmId,
      title: 'Confirm deletion',
      body: `${
        target === 'all'
          ? `Delete all saved questions${day ? ' for this Review day' : ' from every surface'}?`
          : 'Delete this saved question and answer?'
      } This cannot be undone.`,
      confirm: 'Confirm deletion',
      danger: true,
    }).then(yes => {
      confirmOpen.current = false;
      if (yes) void remove(target);
    });
  };
  const more = async () => {
    if (page?.next_before == null) return;
    const controller = startListRead();
    setLoading(true);
    setError('');
    try {
      const next = await readHistory(day, page.next_before, controller.signal);
      if (!controller.signal.aborted) setPage({ items: [...page.items, ...next.items], next_before: next.next_before });
    } catch (e) {
      if (!controller.signal.aborted) setError(messageOf(e));
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  };
  return (
    <Disclosure
      className={classes.root}
      summary="Past questions"
      opened={open}
      onChange={next => {
        setOpen(next);
        if (!next) close();
      }}
    >
      {open && (
        <div className={classes.body}>
          {error && (
            <PageError title="Couldn't load your earlier questions" detail={error}>
              {null}
            </PageError>
          )}
          {loading && (
            <Group gap="xs" role="status" aria-live="polite">
              <Loader size="xs" />
              <Text size="sm" c="dimmed">
                Loading saved questions…
              </Text>
            </Group>
          )}
          {page && !page.items.length && <p>No past questions yet.</p>}
          {!!page?.items.length && (
            <>
              <ul aria-label="Past questions" className={classes.list}>
                {(showAll ? page.items : page.items.slice(0, LATEST)).map(entry => (
                  <li key={entry.id}>
                    <UnstyledButton type="button" onClick={() => void show(entry.id)} disabled={loading}>
                      {entry.question}
                    </UnstyledButton>
                    <small>
                      {[
                        momentWords(entry.created_at),
                        entry.surface,
                        entry.day && `about ${dayWords(entry.day)}`,
                        // A finished question goes without saying; a running or failed one says so.
                        entry.status !== 'completed' && plain(entry.status),
                      ]
                        .filter(Boolean)
                        .join(' · ')}
                    </small>
                    <Button
                      variant="subtle"
                      size="compact-xs"
                      c="var(--lib-bad)"
                      aria-label={`Delete ${entry.question}`}
                      onClick={() => confirmRemove(entry.id)}
                      disabled={loading}
                    >
                      Delete
                    </Button>
                  </li>
                ))}
              </ul>
              <Group gap="xs" className={classes.listActions}>
                {!showAll && page.items.length > LATEST ? (
                  <Button variant="subtle" size="compact-xs" onClick={() => setShowAll(true)}>
                    Show all
                  </Button>
                ) : (
                  page.next_before !== null && (
                    <Button variant="subtle" size="compact-xs" onClick={() => void more()} disabled={loading}>
                      Older questions
                    </Button>
                  )
                )}
                <Button
                  variant="subtle"
                  size="compact-xs"
                  c="var(--lib-bad)"
                  onClick={() => confirmRemove('all')}
                  disabled={loading}
                >
                  Delete all past questions{day ? ' for this day' : ''}
                </Button>
              </Group>
            </>
          )}
          {selected && (
            <section aria-label="Saved answer" className={classes.answer}>
              <Button variant="subtle" size="compact-xs" onClick={close}>
                Close saved answer
              </Button>
              <h3>{selected.question}</h3>
              <p>
                {selected.model ? `Model: ${selected.model}` : 'Model unavailable'} · {selected.status}
              </p>
              <p className={classes.timing}>
                Started: <Stamp value={selected.started_at} /> · Finished: <Stamp value={selected.finished_at} />
              </p>
              {(selected.answer_unverified || selected.provenance_complete === false) && (
                <p>Some sources could not be verified. This saved answer may be incomplete.</p>
              )}
              {selected.answer_removed ? (
                <p>This answer was removed because its source evidence was purged.</p>
              ) : selected.result ? (
                <Result result={safeResult(selected)} history={selected.scope === 'history'} />
              ) : (
                <p>No saved answer is available ({plain(selected.status) || 'unknown status'}).</p>
              )}
              {!!moments(selected).length && (
                <Timeline
                  role="list"
                  aria-label="Saved moments"
                  className={classes.moments}
                  bulletSize={12}
                  lineWidth={2}
                  mt="xs"
                >
                  {moments(selected).map(c => (
                    <Timeline.Item
                      key={`${c.label}-${c.t}`}
                      role="listitem"
                      lineVariant={c.removed || !c.verified ? 'dashed' : 'solid'}
                    >
                      {!c.removed && c.verified && typeof c.t === 'number' && !onSeek ? (
                        <AppLink to={`/review#t=${c.t}`}>{c.label}</AppLink>
                      ) : (
                        <UnstyledButton
                          type="button"
                          disabled={loading || c.removed || !c.verified || typeof c.t !== 'number'}
                          onClick={() => {
                            if (!c.removed && c.verified && typeof c.t === 'number') onSeek?.(c.t);
                          }}
                        >
                          {c.label || 'Moment'}
                          {c.removed ? ' · Source removed' : !c.verified ? ' · Source unavailable' : ''}
                        </UnstyledButton>
                      )}
                    </Timeline.Item>
                  ))}
                </Timeline>
              )}
            </section>
          )}
        </div>
      )}
    </Disclosure>
  );
}
