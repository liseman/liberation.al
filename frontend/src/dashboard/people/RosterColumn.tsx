// The list left of a card, People's and Pets' alike (duplicated frontend components, row 4): the title and count,
// search, the New button, the line that opens what waits for a name, then the list in its groups, with one loading,
// error, empty, and no-match look. Each roster keeps its own search rule, waiting line, and rows.
import { Box, Button, NavLink, TextInput, UnstyledButton } from '@mantine/core';
import { IconPlus, IconSearch } from '@tabler/icons-react';
import type { KeyboardEvent } from 'react';
import { Fragment, type ReactNode } from 'react';
import { LoadError, PageEmpty, PageHeader, PageLoading } from '../lib/page';
import classes from './people.module.css';

/** Whether the roster and its card share one column (people.module.css, 820px and under: a phone). There, People and
 *  Pets open on what waits for a name, with the list one tap back (Joe, #2584); side by side, both show. */
export function oneColumn(): boolean {
  return typeof window.matchMedia === 'function' && window.matchMedia('(max-width: 820px)').matches;
}

/** Whether a click landed on a roster card's name, which opens the card rather than only selecting it. */
function opensCard(target: EventTarget): boolean {
  return target instanceof Element && target.closest('[data-opens]') != null;
}

/** One group of rows under its heading ("Here now", "Everyone"); a group with no rows isn't shown. */
export interface RosterGroup {
  title: string;
  list: string; // the list's accessible name
  rows: ReactNode[];
}

/** One row: an avatar, the name (and anything beside it), and a line under it. */
export function RosterRow({
  avatar,
  label,
  description,
  current,
  onPick,
  onOpen,
}: {
  avatar: ReactNode;
  label: ReactNode;
  description: ReactNode;
  current: boolean;
  onPick: () => void;
  onOpen?: () => void;
}) {
  const onKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.nativeEvent.isComposing) return;
    const key = e.key.toLowerCase();
    if (e.key === 'Enter' || key === 'i') {
      e.preventDefault();
      onOpen?.();
    }
  };
  return (
    <li>
      <NavLink
        component="button"
        type="button"
        className={classes.row}
        classNames={{ label: classes.name, description: classes.meta }}
        active={current}
        aria-current={current ? 'true' : undefined}
        aria-keyshortcuts={onOpen ? 'I' : undefined}
        // A click on the name opens the card, as it always did; a click elsewhere on the card only selects it, the
        // cursor Enter and I open (#3117). At one column the roster and the card aren't shown together, and a
        // touchscreen has no Enter or I, so a tap anywhere opens the card there (#3145).
        onClick={e => (onOpen && (oneColumn() || opensCard(e.target)) ? onOpen() : onPick())}
        onDoubleClick={onOpen}
        onKeyDown={onKey}
        leftSection={avatar}
        label={
          onOpen ? (
            <span className={classes.opens} data-opens>
              {label}
            </span>
          ) : (
            label
          )
        }
        description={description}
      />
    </li>
  );
}

export function RosterColumn({
  title,
  titleId,
  noun,
  count,
  query,
  onQuery,
  newLabel,
  onNew,
  waiting,
  waitingCurrent,
  onWaiting,
  notice,
  loaded,
  error,
  errorDetail = null,
  retrying,
  onRetry,
  empty,
  noMatch,
  groups,
  children,
}: {
  title: string; // "People"
  titleId: string;
  noun: string; // what the list holds, for the search and the loading line ("people", "pets")
  count: number;
  query: string;
  onQuery: (query: string) => void;
  newLabel: string; // "New person"
  onNew: () => void;
  waiting: ReactNode; // the line that opens what waits for a name
  waitingCurrent: boolean; // that view is the one open
  onWaiting: () => void;
  notice?: ReactNode; // under the waiting line (Pets' "Showing cats only")
  loaded: boolean;
  error: string | null; // the first load failed
  errorDetail?: string | null; // the server's reason, behind Show details
  retrying: boolean;
  onRetry: () => void;
  empty: { title: string; text: string }; // nothing listed at all
  noMatch: { title: string; text: string }; // the search matched nothing
  groups: RosterGroup[];
  children?: ReactNode; // dialogs the column opens (New person)
}) {
  const shown = groups.filter(g => g.rows.length);
  // "Nobody matches" only when there's someone a search could have found: an empty list says it's empty.
  const searched = count > 0 && !!query.trim();
  return (
    <section className={classes.roster} aria-labelledby={titleId}>
      <PageHeader title={title} titleId={titleId} meta={loaded ? count : null} />
      <TextInput
        aria-label={`Search ${noun}`}
        placeholder={`Search ${noun}`}
        leftSection={<IconSearch size={16} />}
        value={query}
        onChange={e => onQuery(e.currentTarget.value)}
        mb="xs"
      />
      <Button fullWidth leftSection={<IconPlus size={16} />} onClick={onNew}>
        {newLabel}
      </Button>
      {children}
      <UnstyledButton
        type="button"
        className={classes.waiting}
        aria-current={waitingCurrent ? 'true' : undefined}
        onClick={onWaiting}
      >
        {waiting} <span aria-hidden="true">›</span>
      </UnstyledButton>
      {notice}
      {!loaded ? (
        error ? (
          <LoadError
            message={error}
            hint="Nothing was lost. Try again in a moment."
            detail={errorDetail}
            onRetry={onRetry}
            busy={retrying}
          />
        ) : (
          <PageLoading label={noun} shape="rows" />
        )
      ) : !shown.length ? (
        <Box mt="md">
          <PageEmpty title={searched ? noMatch.title : empty.title} compact>
            {searched ? noMatch.text : empty.text}
          </PageEmpty>
        </Box>
      ) : (
        shown.map(g => (
          <Fragment key={g.title}>
            <h2 className={classes.group}>{g.title}</h2>
            <ul className={classes.list} aria-label={g.list}>
              {g.rows}
            </ul>
          </Fragment>
        ))
      )}
    </section>
  );
}
