// Journal: the journal as a column view, like a Finder window. Months in the first column; picking one opens
// its weeks to the right, then a week's days, a day's hours, and an hour's events. The pane at the far right shows
// what's picked: a month's, week's, or day's page, or an hour's or event's text with what it cites, resolved as the
// day page's citations are: the recording, a promoted clip or still, the audio, or why there's nothing left. Arrow keys
// move like Finder: up and down within a column, right into the next, left back out. On a phone it's one column at a
// time, with Back.
import { Badge, Button, Text, UnstyledButton, VisuallyHidden } from '@mantine/core';
import { useDebouncedValue, useMediaQuery } from '@mantine/hooks';
import { IconChevronLeft, IconChevronRight } from '@tabler/icons-react';
import { useQueries, useQuery } from '@tanstack/react-query';
import { type KeyboardEvent, useEffect, useRef, useState } from 'react';
import { HttpError, messageOf } from '../../shared/http';
import { Prose } from '../ask/prose';
import { hm } from '../lib/format';
import { PageFrame, PageHeader, PHONE_WIDTH } from '../lib/page';
import { linkClick } from '../router';
import {
  type Column,
  columnQuery,
  type Evidence,
  evidenceQuery,
  LEVELS,
  type Level,
  type NoPage,
  type Row,
  type Source,
  todayQuery,
} from './api';
import { SourceMedia } from './evidence';
import classes from './journal.module.css';

/** What each column is called, and what its empty state says. */
const COLUMN: Record<Level, { name: string; empty: string }> = {
  month: { name: 'Months', empty: 'No months yet.' },
  week: { name: 'Weeks', empty: 'No weeks in this month yet.' },
  day: { name: 'Days', empty: 'No days in this week yet.' },
  hour: { name: 'Hours', empty: 'No hours written for this day.' },
  event: { name: 'Events', empty: 'No events written for this hour.' },
};

/** When each level's page is written, for a period that has none yet. */
const NO_PAGE: Record<string, string> = {
  month: "This month's page isn't written yet. It's written on the 1st, from the month's weeks.",
  week: "This week's page isn't written yet. It's written early on Monday, from the week's days.",
  day: "This day's page isn't written yet. It's written the night after, from the day's hours.",
};

/** What a period with no page says, by why it has none (Column's `why`). */
function noPage(level: string, why: NoPage | null | undefined): string {
  const from = level === 'day' ? 'hours' : 'days';
  switch (why) {
    case 'rewriting':
      return `This ${level}'s page is being written again after a transcript correction. It's back within a few minutes.`;
    case 'quiet':
      return `Nothing was recorded this ${level}, so it has no page.`;
    case 'failed':
      return `This ${level}'s page couldn't be written: the model failed on it. It's tried again when its ${from} change.`;
    case 'old':
      return `No page was written for this ${level}, and it's too old to be written now.`;
    default:
      return NO_PAGE[level] ?? '';
  }
}

/** A local date as the day column's keys are written: "2026-11-07". */
const dayKey = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

/** Whether a day may have a today-so-far page: today, or yesterday (until its own page is written). */
function recent(key: string): boolean {
  const now = new Date();
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
  return key === dayKey(now) || key === dayKey(yesterday);
}

/** A page's citations open Review at their moment, as the listing's pages already do. */
const reviewLinks = (md: string) => md.replaceAll('](#t=', '](/review#t=');

/** How long a pick must stay put before the column it opens is read: holding an arrow key through a column reads
 *  only where it stops, not every row it passes (a week's days take a second or more to read). */
export const CHILD_DELAY_MS = 250;

/** The id of a row's option element. */
const optionId = (level: Level, key: string) => `journal-${level}-${key.replace(/[^\w-]/g, '_')}`;

/** "09:00:12" for a source's moment. */
const hms = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });

function RowView({ level, row }: { level: Level; row: Row }) {
  if (level === 'hour' || level === 'event')
    return (
      <span className={classes.entry}>
        <span className={classes.time}>{row.label}</span>
        <span className={classes.snippet}>{row.text}</span>
      </span>
    );
  return (
    <span className={classes.period}>
      <span>{row.label}</span>
      <span className={classes.meta}>
        {row.hours ? <span>{row.hours} h</span> : null}
        {row.page && <span className={classes.dot} role="img" title="Page written" aria-label="Page written" />}
      </span>
    </span>
  );
}

function ColumnView({
  level,
  column,
  loading,
  failed,
  selected,
  active,
  onPick,
  onKey,
  listRef,
}: {
  level: Level;
  column: Column | undefined;
  loading: boolean;
  failed: boolean;
  selected: string | undefined;
  active: boolean;
  onPick: (key: string) => void;
  onKey: (e: KeyboardEvent<HTMLDivElement>) => void;
  listRef: (el: HTMLDivElement | null) => void;
}) {
  const items = column?.items ?? [];
  const deep = level === 'hour' || level === 'event';
  return (
    <div
      ref={listRef}
      role="listbox"
      aria-label={COLUMN[level].name}
      tabIndex={active ? 0 : -1}
      aria-activedescendant={selected ? optionId(level, selected) : undefined}
      className={classes.column}
      onKeyDown={onKey}
      data-level={level}
      data-deep={deep || undefined}
      data-active={active || undefined}
    >
      {loading ? (
        <Text size="sm" c="dimmed" className={classes.note}>
          Loading…
        </Text>
      ) : failed ? (
        <Text size="sm" c="red" className={classes.note}>
          Couldn't load this column.
        </Text>
      ) : items.length === 0 ? (
        <Text size="sm" c="dimmed" className={classes.note}>
          {COLUMN[level].empty}
        </Text>
      ) : (
        items.map(row => (
          // biome-ignore lint/a11y/useKeyWithClickEvents: the listbox around it takes the keys (arrows, like Finder)
          <div
            key={row.key}
            id={optionId(level, row.key)}
            role="option"
            aria-selected={row.key === selected}
            tabIndex={-1}
            className={classes.row}
            onMouseDown={e => e.preventDefault()} // keep focus on the list, which handles the keys
            onClick={() => onPick(row.key)}
          >
            <RowView level={level} row={row} />
            {level !== 'event' && <IconChevronRight size={14} className={classes.chevron} aria-hidden />}
          </div>
        ))
      )}
    </div>
  );
}

const KIND: Record<string, string> = {
  walkby: 'Detected',
  action: 'Detected',
  zone: 'In a zone',
};

/** What a source says it is: heard, seen on a camera, detected. */
const what = (s: Source) => (s.camera ? (KIND[s.kind ?? ''] ?? 'Seen') : 'Heard');

function EvidenceView({ evidence }: { evidence: Evidence }) {
  const sources = evidence.sources;
  if (!sources.length)
    return (
      <Text size="sm" c="dimmed">
        Nothing cited.
      </Text>
    );
  return (
    <>
      {evidence.status === 'unverified' && (
        <Badge variant="light" color="yellow" tt="none" fw={500} c="var(--lib-warn-text)">
          Unverified
        </Badge>
      )}
      <ul className={classes.sources}>
        {sources.map((s, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: an entry's sources have no identity beyond their place
          <li key={i} className={classes.source}>
            <div className={classes.sourceHead}>
              <span>
                {what(s)}
                {s.camera ? ` on ${s.camera}` : ''} at {hms(s.start)}
                {s.people.length ? ` · ${s.people.join(', ')}` : ''}
              </span>
              <a href={s.review} onClick={linkClick}>
                Open in Review
              </a>
            </div>
            {s.caption && <p className={classes.sourceText}>{s.caption}</p>}
            <SourceMedia source={s} />
          </li>
        ))}
      </ul>
    </>
  );
}

/** An hour's or event's evidence, read when it's shown. */
function EntryEvidence({ row }: { row: Row }) {
  const query = useQuery({ ...evidenceQuery(row.id ?? row.key, row.start ?? 0), enabled: !!row.id });
  if (!row.sources)
    return (
      <Text size="sm" c="dimmed">
        Nothing cited.
      </Text>
    );
  if (query.isError)
    return (
      <Text size="sm" c="red">
        Couldn't load what this cites.
      </Text>
    );
  if (!query.data)
    return (
      <Text size="sm" c="dimmed">
        Loading what this cites…
      </Text>
    );
  return <EvidenceView evidence={query.data} />;
}

/** Today's page so far, or yesterday's until its page is written (GET /journal/today). */
function TodayPage({ day }: { day: string }) {
  const query = useQuery(todayQuery(day));
  const page = query.data;
  // A 404 is a day the route doesn't build: its page isn't written yet. Any other failure says what failed, unless a
  // page was already read (it stays), and the query keeps asking (todayQuery), so it clears once the dashboard answers.
  if (query.error instanceof HttpError && query.error.status === 404)
    return (
      <Text size="sm" c="dimmed">
        {NO_PAGE.day}
      </Text>
    );
  if (query.isError && !page)
    return (
      <Text size="sm" c="red">
        Couldn't read the day so far: {messageOf(query.error)}. Trying again.
      </Text>
    );
  if (page?.status === 'error')
    return (
      <Text size="sm" c="dimmed">
        This day's page so far couldn't be built. It's tried again in a few minutes.
      </Text>
    );
  if (!page)
    return (
      <Text size="sm" c="dimmed">
        Loading the day so far…
      </Text>
    );
  if (page.status === 'empty')
    return (
      <Text size="sm" c="dimmed">
        No hours written yet today. The page fills in as the hours are.
      </Text>
    );
  return (
    <>
      {page.status === 'pending' && (
        <Text size="xs" c="dimmed" mb="xs" role="status">
          {page.md ? 'Bringing this up to date…' : 'Writing the day so far…'}
        </Text>
      )}
      {page.md && <Prose text={reviewLinks(page.md)} />}
    </>
  );
}

function Preview({ level, row, child }: { level: Level | null; row: Row | undefined; child: Column | undefined }) {
  if (!level || !row)
    return (
      <Text size="sm" c="dimmed">
        Pick a month to start.
      </Text>
    );
  if (level === 'hour' || level === 'event')
    return (
      <article>
        <h2 className={classes.previewTitle}>
          {row.start !== undefined && row.end !== undefined ? `${hm(row.start)} to ${hm(row.end)}` : row.label}
        </h2>
        <p className={classes.entryText}>{row.text}</p>
        {row.start !== undefined && (
          // Where it happened: the Places plan at this moment, with the paths around it.
          <a className={classes.where} href={`/places?t=${Math.round(row.start)}`} onClick={linkClick}>
            Where, on the plan
          </a>
        )}
        <h3 className={classes.subhead}>Evidence</h3>
        <EntryEvidence row={row} />
      </article>
    );
  return (
    <article>
      <h2 className={classes.previewTitle}>{row.label}</h2>
      {child?.page ? (
        <Prose text={child.page.md} />
      ) : child && level === 'day' && recent(row.key) ? (
        <TodayPage day={row.key} />
      ) : child ? (
        <Text size="sm" c="dimmed">
          {noPage(level, child.why)}
        </Text>
      ) : null}
    </article>
  );
}

export function JournalTab() {
  const phone = useMediaQuery(PHONE_WIDTH) ?? false;
  /** The key picked in each column, coarsest first. */
  const [path, setPath] = useState<string[]>([]);
  /** The column the keys act on. */
  const [focus, setFocus] = useState(0);
  /** On a phone, which pane shows: a column's index, or LEVELS.length for the preview. */
  const [pane, setPane] = useState(0);
  const lists = useRef<(HTMLDivElement | null)[]>([]);
  const browser = useRef<HTMLDivElement>(null);

  // A column shows once the column before it has a pick, and is read once that pick has settled (CHILD_DELAY_MS).
  const [settled] = useDebouncedValue(path, CHILD_DELAY_MS);
  const shown = LEVELS.filter((_, i) => i === 0 || path[i - 1] !== undefined);
  const queries = useQueries({
    queries: shown.map((level, i) => {
      const parent = i ? (settled[i - 1] ?? null) : null;
      return { ...columnQuery(level, parent), enabled: i === 0 || parent !== null };
    }),
  });
  /** Whether column i's reply is for the pick showing now (not one the arrows have since moved past). */
  const current = (i: number) => i === 0 || settled[i - 1] === path[i - 1];
  const columns = queries.map((q, i) => (current(i) ? q.data : undefined));

  // A new column opening scrolls the row to its end, as Finder does, when they don't all fit.
  useEffect(() => {
    const el = browser.current;
    if (el && path.length) el.scrollTo?.({ left: el.scrollWidth, behavior: 'smooth' }); // not in jsdom
  }, [path.length]);

  // Open on the newest month, so the first view has something in it.
  const firstMonth = columns[0]?.items[0]?.key;
  useEffect(() => {
    if (firstMonth && path.length === 0) setPath([firstMonth]);
  }, [firstMonth, path.length]);

  const pick = (i: number, key: string) => {
    setPath(p => (p[i] === key ? p.slice(0, i + 1) : [...p.slice(0, i), key]));
    setFocus(phone && i + 1 < LEVELS.length ? i + 1 : i);
    if (phone) setPane(Math.min(i + 1, LEVELS.length));
    else lists.current[i]?.focus(); // so the arrow keys work from where the mouse picked
  };

  const move = (to: number) => {
    setFocus(to);
    lists.current[to]?.focus();
  };

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const items = columns[focus]?.items ?? [];
    const at = items.findIndex(r => r.key === path[focus]);
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      const next = items[Math.max(0, Math.min(items.length - 1, at + (e.key === 'ArrowDown' ? 1 : -1)))];
      if (next) {
        setPath(p => [...p.slice(0, focus), next.key]);
        document.getElementById(optionId(LEVELS[focus] as Level, next.key))?.scrollIntoView({ block: 'nearest' });
      }
    } else if (e.key === 'ArrowRight') {
      const child = columns[focus + 1]?.items[0];
      if (path[focus] && child) {
        if (path[focus + 1] === undefined) setPath(p => [...p.slice(0, focus + 1), child.key]);
        move(focus + 1);
      }
    } else if (e.key === 'ArrowLeft') {
      if (focus > 0) move(focus - 1);
    } else return;
    e.preventDefault();
  };

  // The preview is of the finest pick.
  const depth = path.length - 1;
  const level = depth >= 0 ? (LEVELS[depth] ?? null) : null;
  const row = depth >= 0 ? columns[depth]?.items.find(r => r.key === path[depth]) : undefined;
  const preview = <Preview level={level} row={row} child={columns[depth + 1]} />;

  const column = (i: number) => (
    <ColumnView
      key={LEVELS[i]}
      level={LEVELS[i] as Level}
      column={columns[i]}
      loading={!current(i) || !!queries[i]?.isPending}
      failed={current(i) && !!queries[i]?.isError}
      selected={path[i]}
      active={focus === i}
      onPick={key => pick(i, key)}
      onKey={onKey}
      listRef={el => {
        lists.current[i] = el;
      }}
    />
  );

  if (phone) {
    const inPreview = pane >= LEVELS.length;
    // Back from the preview: to the column under what it shows, or to the events for an event.
    const back = Math.min(path.length, LEVELS.length - 1);
    const parentIndex = inPreview ? path.length - 1 : pane - 1;
    const parent = parentIndex >= 0 ? columns[parentIndex]?.items.find(r => r.key === path[parentIndex]) : undefined;
    return (
      <PageFrame width="wide">
        <section aria-labelledby="journal-h">
          <PageHeader title="Journal" titleId="journal-h" />
          <div className={classes.phoneBar}>
            {pane > 0 && (
              <Button
                variant="subtle"
                size="compact-sm"
                leftSection={<IconChevronLeft size={16} />}
                onClick={() => {
                  const to = inPreview ? back : pane - 1;
                  setPane(to);
                  setFocus(to);
                }}
              >
                {inPreview ? COLUMN[LEVELS[back] as Level].name : 'Back'}
              </Button>
            )}
            <Text fw={600} size="sm" className={classes.phoneTitle}>
              {inPreview ? (row?.label ?? '') : parent ? parent.label : COLUMN[LEVELS[pane] as Level].name}
            </Text>
            {!inPreview && pane > 0 && parent && (
              <UnstyledButton
                className={classes.phoneOpen}
                onClick={() => {
                  setPath(p => p.slice(0, pane));
                  setPane(LEVELS.length);
                }}
              >
                {pane - 1 <= 2 ? 'Page' : 'Details'}
              </UnstyledButton>
            )}
          </div>
          <div className={classes.phonePane}>
            {inPreview ? <div className={classes.preview}>{preview}</div> : column(Math.min(pane, shown.length - 1))}
          </div>
        </section>
      </PageFrame>
    );
  }

  return (
    <PageFrame width="wide">
      <section aria-labelledby="journal-h">
        <PageHeader
          title="Journal"
          titleId="journal-h"
          lede="What happened, a month at a time down to the moment. Arrow keys move between and within the columns."
        />
        <div className={classes.browser} ref={browser}>
          {shown.map((_, i) => column(i))}
          <div className={classes.preview}>{preview}</div>
          {/* A screen reader hears what's picked, not the whole page the preview shows. */}
          <VisuallyHidden role="status">{row ? `${row.label} shown` : ''}</VisuallyHidden>
        </div>
      </section>
    </PageFrame>
  );
}
