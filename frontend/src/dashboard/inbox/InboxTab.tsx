// The Inbox: research reports, newest first. Each is an Ask question taken slowly in the background; opening
// a report marks it read. A report leads with its question and its actions, then the answer (whose citations open the
// moment in Review), "How sure is this?" in plain words, and one counted line that opens on what it looked at.
// On a phone the list opens a report on its own, with a way back.
import { Alert, Badge, Button, Group, Loader, NavLink, Stack, Text, Title, VisuallyHidden } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { IconChevronLeft } from '@tabler/icons-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { messageOf } from '../../shared/http';
import { useReviewAsk } from '../AskBar';
import { openAsk } from '../ask/openAsk';
import { Paged } from '../ask/Paged';
import { Evidence, Result } from '../ask/Result';
import { confirm } from '../lib/confirm';
import { Disclosure } from '../lib/Disclosure';
import { dayWords, isoDay, momentWords } from '../lib/format';
import { CountSkeleton } from '../lib/loading';
import { PageEmpty, PageError, PageFrame, PageLoading } from '../lib/page';
import { reviewURL } from '../review/url';
import { href, linkClick, navigate } from '../router';
import { type Job, jobQuery, markRead, type Report, removeJob, researchQuery, startResearch } from './api';
import classes from './inbox.module.css';

const PHONE_MAX = 760; // inbox.module.css: at this width or less, the list and the reader take turns

import { CitedMoments, citedMoments } from './moments';
import { lookedLine, sourceCount, verdict } from './sure';

const STATUS: Record<Job['status'], string> = {
  queued: 'Waiting',
  running: 'Researching',
  done: 'Done',
  failed: 'Failed',
};

function took(seconds: number | undefined): string {
  if (seconds === undefined) return '';
  if (seconds < 90) return `${Math.round(seconds)} s`;
  const minutes = Math.round(seconds / 60);
  return minutes < 90 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

const when = (ts: number) => momentWords(ts); // "Today, 13:31", in the app's 24-hour times

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** What a search was for, from its arguments: the words, the day, the camera. */
function searchText(args: Record<string, unknown> | null): string {
  if (!args) return '';
  return Object.values(args)
    .filter(v => typeof v === 'string' || typeof v === 'number')
    .join(', ');
}

/** Evidence items grouped by where they came from, biggest group first. */
function bySource(evidence: readonly Record<string, unknown>[]): [string, Record<string, unknown>[]][] {
  const groups = new Map<string, Record<string, unknown>[]>();
  for (const e of evidence) {
    const kind = typeof e.source === 'string' && e.source ? e.source : 'other';
    groups.set(kind, [...(groups.get(kind) ?? []), e]);
  }
  return [...groups].sort((x, y) => y[1].length - x[1].length);
}

/** What the research looked at, folded to one counted line: it opens on the evidence grouped by source, then the
 * searches it ran and the model that wrote the answer. */
export function LookedAt({ report }: { report: Report }) {
  const { searches, sources, evidence } = report.looked_at;
  const groups = bySource(report.evidence ?? []);
  return (
    <section className={classes.looked} aria-label="What it looked at">
      <Disclosure summary={lookedLine(evidence, sources)} lazy>
        <div className={classes.lookedBody}>
          {groups.map(([kind, items]) => (
            <section key={kind} className={classes.group} aria-label={sourceCount(kind, items.length)}>
              <Text size="sm" fw={600}>
                {sourceCount(kind, items.length)}
              </Text>
              <Paged items={items} render={e => <Evidence item={e} />} />
            </section>
          ))}
          {searches.length > 0 && (
            <>
              <Text size="sm" fw={600}>
                {plural(searches.length, 'search', 'searches')}
              </Text>
              <ol className={classes.searches}>
                {searches.map((s, i) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: searches are a fixed, ordered list
                  <li key={i}>
                    {s.tool ?? 'search'}
                    {searchText(s.args) && `: ${searchText(s.args)}`}
                    {s.found !== null && ` (${plural(s.found, 'found', 'found')})`}
                  </li>
                ))}
              </ol>
            </>
          )}
          {report.model && (
            <Text size="xs" c="dimmed">
              Written by {report.model.replace(/^ollama\//, '')}
            </Text>
          )}
        </div>
      </Disclosure>
    </section>
  );
}

/** "How sure is this?" as one verdict line: "Partly sure" and why, or "Fairly sure", with the caveats that hold
 *  for every report folded under it. The answer's own "Partial answer" badge gives way to it. */
function HowSure({ status, warnings }: { status: unknown; warnings: readonly unknown[] }) {
  const v = verdict(status, warnings);
  if (!v.why && !v.rest.length) return null;
  return (
    <section className={classes.sure} aria-labelledby="inbox-sure">
      <VisuallyHidden component="h3" id="inbox-sure">
        How sure is this?
      </VisuallyHidden>
      <Text>
        <span className={classes.verdict} data-sure={v.why ? 'partly' : 'fairly'}>
          {v.sure}.
        </span>
        {v.why && ` ${v.why}`}
      </Text>
      {v.rest.length > 0 && (
        <Disclosure summary={v.rest.length === 1 ? '1 more caveat' : `${v.rest.length} more caveats`}>
          <ul>
            {v.rest.map(line => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </Disclosure>
      )}
    </section>
  );
}

function ReportView({
  id,
  onGone,
  onBack,
  onOpen,
}: {
  id: string;
  onGone: () => void;
  onBack: () => void;
  onOpen: (id: string) => void; // a new report (Ask again) to read instead
}) {
  const client = useQueryClient();
  const q = useQuery(jobQuery(id));
  const [error, setError] = useState('');
  const [again, setAgain] = useState(false);
  const job = q.data;
  // Opening a finished report marks it read, once.
  useEffect(() => {
    if (job && (job.status === 'done' || job.status === 'failed') && !job.read)
      void markRead(client, id, true).catch(e => setError(messageOf(e)));
  }, [client, id, job]);
  const back = (
    <Button
      variant="subtle"
      size="xs"
      className={classes.back}
      leftSection={<IconChevronLeft size={16} />}
      onClick={onBack}
    >
      All reports
    </Button>
  );
  if (q.isPending)
    return (
      <>
        {back}
        <PageLoading label="this report" />
      </>
    );
  if (q.isError)
    return (
      <>
        {back}
        <PageError
          title="Couldn't load this report"
          detail={messageOf(q.error)}
          onRetry={() => void q.refetch()}
          busy={q.isFetching}
        />
      </>
    );
  if (!job) return null;
  const report = job.result;
  const finished = job.status === 'done' || job.status === 'failed';
  return (
    <article className={classes.report} aria-label="Research report">
      {back}
      <header className={classes.reportHead}>
        <Title order={2} className={classes.question}>
          {job.question}
        </Title>
        <Text size="sm" c="dimmed">
          {job.day ? `About ${dayWords(job.day)} · ` : ''}Asked {when(job.created_at)}
          {job.took_s !== undefined && ` · took ${took(job.took_s)}`}
        </Text>
        <Group gap="xs">
          {/* The day it's about, else the day it was asked: a report with no day of its own still opens somewhere */}
          <Button
            size="xs"
            variant="default"
            component="a"
            href={reviewURL({ day: job.day ?? isoDay(new Date(job.created_at * 1000)) })}
            onClick={linkClick}
          >
            Open in Review
          </Button>
          {finished && (
            // The same question, researched now: "the last 30 minutes" is about a moment that has passed
            <Button
              size="xs"
              variant="default"
              loading={again}
              onClick={async () => {
                setAgain(true);
                try {
                  onOpen((await startResearch(client, job.question, job.day)).id);
                } catch (e) {
                  setError(messageOf(e));
                } finally {
                  setAgain(false);
                }
              }}
            >
              Ask again
            </Button>
          )}
          {job.status === 'done' && !!report?.answer && (
            // Ask, opened on this report: the server reads the report by its id, so the next question knows it
            <Button
              size="xs"
              variant="default"
              onClick={() =>
                openAsk({
                  context: { kind: 'report', id, question: job.question, ...(job.day ? { day: job.day } : {}) },
                })
              }
            >
              Ask a follow-up
            </Button>
          )}
          {finished && (
            <Button size="xs" variant="default" onClick={() => void markRead(client, id, false)}>
              Mark unread
            </Button>
          )}
          <Button
            size="xs"
            variant="subtle"
            color="red"
            // Gone for good (research.py deletes the job; there's no Undo to offer), so it asks first, Cancel focused
            onClick={async () => {
              const sure = await confirm(
                finished
                  ? {
                      title: 'Delete this report?',
                      body: 'It goes for good. Asking the question again starts the research over.',
                      confirm: 'Delete',
                      cancel: 'Keep it',
                      danger: true,
                      id: `inbox-delete-${id}`,
                    }
                  : {
                      title: 'Stop this research?',
                      body: "What it has looked at so far isn't kept.",
                      confirm: 'Stop',
                      cancel: 'Keep researching',
                      danger: true,
                      id: `inbox-delete-${id}`,
                    },
              );
              if (!sure) return;
              await removeJob(client, id)
                .then(onGone)
                .catch(e => setError(messageOf(e)));
            }}
          >
            {finished ? 'Delete' : 'Cancel'}
          </Button>
        </Group>
      </header>
      {error && (
        <PageError title="That didn't work" detail={error}>
          The report is as it was. Try again.
        </PageError>
      )}
      {job.status === 'queued' && <Text>Waiting for the research before it to finish.</Text>}
      {job.status === 'running' && (
        <Group gap="xs">
          <Loader size="xs" />
          <Text>Researching. The report will be here when it's done.</Text>
        </Group>
      )}
      {job.status === 'failed' && (
        <PageError title="This research didn't finish" detail={job.error}>
          Ask it again from Review to start over.
        </PageError>
      )}
      {report && job.status === 'done' && (
        <>
          <Result
            result={{ ...report, is_history: true, scope: 'history' }}
            history
            folded
            bare
            after={<CitedMoments moments={citedMoments(report.answer ?? '', report.evidence ?? [])} />}
          />
          <HowSure status={report.status} warnings={report.warnings ?? []} />
          <LookedAt report={report} />
        </>
      )}
    </article>
  );
}

/** Review's Ask, opened: its Research button looks into a question in the background, and the report lands here. */
function researchAQuestion() {
  navigate(href('/review'));
  useReviewAsk.getState().setOpen(true);
}

export function InboxTab() {
  const q = useQuery(researchQuery());
  const [open, setOpen] = useState<string | null>(null);
  // On a wide screen the reader beside the list opens on arrival: the newest unread report, else the newest.
  // Closing it ("All reports") or a phone, where the list is the page, leaves it closed.
  const wide = useMediaQuery(`(min-width: ${PHONE_MAX + 1}px)`, undefined, { getInitialValueInEffect: false }) ?? false;
  const [closed, setClosed] = useState(false);
  const jobs = q.data?.jobs ?? [];
  const arrival = jobs.find(j => (j.status === 'done' || j.status === 'failed') && !j.read) ?? jobs[0];
  const picked = open && jobs.some(j => j.id === open) ? open : null;
  const selected = picked ?? (wide && !closed ? (arrival?.id ?? null) : null);
  const close = () => {
    setOpen(null);
    setClosed(true);
  };
  return (
    <PageFrame
      title="Inbox"
      meta={
        q.isPending ? (
          <CountSkeleton chars={10} label="Loading how many reports" />
        ) : jobs.length ? (
          plural(jobs.length, 'report')
        ) : undefined
      }
      lede="Questions looked into slowly, in the background. Each report lands here when it's done."
      actions={
        // Research starts from Review's Ask, which has the day's recordings to look through
        <Button size="sm" onClick={researchAQuestion}>
          Research a question
        </Button>
      }
    >
      <div className={classes.page} data-reading={selected ? 'true' : undefined}>
        <Stack gap="xs" className={classes.list} component="section" aria-label="Research reports">
          {q.data && !q.data.available && q.data.reason && <Alert color="yellow">{q.data.reason}</Alert>}
          {q.isPending && <PageLoading label="the Inbox" shape="rows" lines={4} />}
          {q.isError && !q.data && (
            <PageError
              title="Couldn't load the Inbox"
              detail={messageOf(q.error)}
              onRetry={() => void q.refetch()}
              busy={q.isFetching}
            />
          )}
          {q.data && !jobs.length && (
            <PageEmpty
              title="No research reports yet"
              action={
                <Button size="xs" variant="default" component="a" href={href('/review')} onClick={linkClick}>
                  Open Review
                </Button>
              }
            >
              Ask a question in Review and press Research. The report lands here when it's done.
            </PageEmpty>
          )}
          <ul className={classes.items}>
            {jobs.map(j => {
              const unread = (j.status === 'done' || j.status === 'failed') && !j.read;
              // A finished report whose answer came out short of whole says so on the list
              const partial = j.status === 'done' && !!j.answer_status && j.answer_status !== 'ok';
              return (
                <li key={j.id}>
                  <NavLink
                    component="button"
                    type="button"
                    active={selected === j.id}
                    label={j.question}
                    classNames={{ label: classes.itemQuestion, description: classes.itemMeta }}
                    description={
                      <>
                        {unread && <span className={classes.dot} aria-hidden />}
                        {(j.status !== 'done' || partial) && (
                          <>
                            <Badge
                              size="xs"
                              variant="light"
                              color={j.status === 'failed' ? 'red' : partial ? 'yellow' : undefined}
                              // Mantine's yellow label read at 2.7:1 on its tint; the app's amber text reads at 4.7:1
                              c={partial ? 'var(--lib-warn-text)' : undefined}
                            >
                              {partial ? 'Partial' : STATUS[j.status]}
                            </Badge>{' '}
                          </>
                        )}
                        {when(j.created_at)}
                        {/* What the dot and a finished row's missing badge say, for screen readers; the row is named
                            by its visible words plus these, not an aria-label that left the date out (axe) */}
                        {((j.status === 'done' && !partial) || unread) && (
                          <>
                            {' '}
                            <VisuallyHidden>
                              {[j.status === 'done' && !partial && 'Done', unread && 'unread']
                                .filter(Boolean)
                                .join(', ')}
                            </VisuallyHidden>
                          </>
                        )}
                      </>
                    }
                    className={`${classes.item}${unread ? ` ${classes.unread}` : ''}`}
                    aria-current={selected === j.id || undefined}
                    onClick={() => setOpen(j.id)}
                  />
                </li>
              );
            })}
          </ul>
        </Stack>
        <div className={classes.detail}>
          {selected ? (
            <ReportView key={selected} id={selected} onGone={close} onBack={close} onOpen={setOpen} />
          ) : (
            // Only beside the list: on a phone the list is the page, with nothing to choose into
            wide && jobs.length > 0 && <Text c="dimmed">Choose a report to read it.</Text>
          )}
        </div>
      </div>
    </PageFrame>
  );
}
