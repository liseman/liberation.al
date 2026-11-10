// The naming inbox: one component for every list of things waiting for a name (Pets' sightings to name;
// People's face groups next), with an explicit "Not <name>" action and Y/N/I keys.
// The shell owns what every kind shares: the heading with how many wait, the "By <kind>" bulk view, the sort (a
// Certainty toggle, highest first by default so the one-click rows come first) and the picture size, both
// remembered per kind in this browser, the list itself, and where focus goes when a row leaves. Each kind renders its rows with
// NamingRow and supplies its own name picker, pictures, and bulk view.
//
// The guess bands are each kind's own: pets' "probable" is a clear recognition match and "maybe" a suggestion
// (people/pet_recognition.py); faces' are vision/face_guess.py's score floors (probable at 75 points, maybe at 50).
// Both read here as "Probably" and "Maybe".
import { Button, Group, SegmentedControl, Text, Title, Tooltip } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { IconArrowDown, IconArrowUp } from '@tabler/icons-react';
import { type CSSProperties, createContext, type ReactNode, useContext, useMemo, useRef, useState } from 'react';
import { readPrefs, writePrefs } from '../../shared/prefs';
import { focusAfter } from './focus';
import '../whosthis/whosthis.css';
import { PageEmpty, PHONE_WIDTH, type ViewChoice, ViewMenu } from '../lib/page';
import who from '../whosthis/whosthis.module.css';
import classes from './naming.module.css';

export type Band = 'probable' | 'maybe' | 'none';

/** A row's best guess, in the shape every kind's list shares: `id` is what naming sends back. */
export interface InboxGuess {
  id: string;
  name: string;
  score: number; // 0 to 100
  band: Band;
}

// By certainty, highest or lowest first. The old select's third order, Newest, is gone: ties keep the
// newest-first order, and a browser that remembered Newest starts at highest first.
export type InboxSort = 'sure' | 'unsure';
export type PhotoSize = 'S' | 'M' | 'L';
const SORTS: { value: InboxSort; label: string }[] = [
  { value: 'sure', label: 'Highest certainty first' },
  { value: 'unsure', label: 'Lowest certainty first' },
];
/** Each size's list picture, narrowest card, and bulk tile widths, in px. L is the large picture; S is about
 *  the old compact rows', and loads the thumbnail, since a full crop is wasted on it. The cards fill as many columns
 *  as fit at their narrowest, so S fits the most. A card is its picture, the guess, and the 208 px name box
 *  with its buttons at the top right. A card with less than about 180 px left for the words puts the name box
 *  and buttons under them instead (naming.module.css), as one at the narrowest width does. */
export const PHOTO_SIZES: Record<PhotoSize, { row: number; card: number; tile: number; thumb: boolean }> = {
  S: { row: 80, card: 520, tile: 110, thumb: true },
  M: { row: 180, card: 620, tile: 160, thumb: false },
  L: { row: 300, card: 740, tile: 220, thumb: false },
};

/** One row of the list: its key (also its `data-row`), its guess for sorting, and how to draw it. */
export interface InboxItem {
  key: string;
  guess: InboxGuess | null;
  render: (shown: { thumb: boolean }) => ReactNode;
}

/** The order picked: by how sure the guess is (Probably, then Maybe, then none), then its score, highest or lowest
 *  first. Ties keep the order the list came in (newest first). */
export function sortItems<T extends { guess: InboxGuess | null }>(items: T[], sort: InboxSort): T[] {
  const rank = (g: InboxGuess | null) =>
    !g || g.band === 'none' ? 0 : (g.band === 'probable' ? 2 : 1) * 1000 + g.score;
  const ranked = items.map(item => [item, rank(item.guess)] as const);
  return ranked.sort((a, b) => (sort === 'sure' ? b[1] - a[1] : a[1] - b[1])).map(([item]) => item);
}

/** A browser's remembered choice, if it's one of `allowed`. */
function remembered<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  const value = readPrefs()[key];
  return allowed.includes(value as T) ? (value as T) : fallback;
}

/** What a row asks of the list: to move focus on before it leaves. */
const Leaving = createContext<(key: string) => void>(() => {});
export const useLeaving = () => useContext(Leaving);

/** The list's selected row, as the roster has one: a click on a row, or focus coming to it (a pick moves focus
 *  to the next row), selects it, and its keys (Y, N, I, and the icons') act on it. One per list. */
const Cursor = createContext<{ cursor: string | null; select: (key: string) => void }>({
  cursor: null,
  select: () => {},
});
export const useCursor = () => useContext(Cursor);

export function NamingInbox({
  prefs,
  title,
  headingId,
  intro,
  items,
  bulkLabel,
  bulk,
  above,
  status,
  empty,
  onBack,
  backLabel,
  defaultSize = 'L',
  sortLabel = 'Sort sightings',
  compact = false,
  total,
  sectionId,
  className = 'who-wide',
  toast,
  footer,
}: {
  prefs: string; // the kind's prefix for remembered choices ("pet", for petSightingsSort and petPhotoSize)
  title: string;
  headingId: string;
  intro?: ReactNode;
  items: InboxItem[];
  defaultSize?: PhotoSize; // until this browser picks one
  sortLabel?: string;
  // How many wait in all, when the server lists only the first ones (Who's this? lists 24 face groups); the
  // header counts these, as People's line and Known's badge do. Defaults to the listed items.
  total?: number | null;
  compact?: boolean; // Live's rail: the list alone, without the controls, the count, or the introduction
  sectionId?: string;
  className?: string;
  toast?: ReactNode; // the kind's floating confirmation, with its Undo
  footer?: ReactNode; // under the list: processing pictures, the state line, recent decisions
  bulkLabel?: string; // "By pet": the bulk view's toggle, when it has one
  bulk?: (shown: { thumb: boolean }) => ReactNode;
  above?: ReactNode; // shown over the list, not in the bulk view (pets' recognition checks)
  status?: ReactNode; // loading or error, in place of the empty line
  empty?: string; // what an empty list says, when the kind doesn't say it itself (in `footer`)
  onBack?: () => void;
  backLabel?: string;
}) {
  const [byBulk, setByBulk] = useState(false);
  const [sort, setSort] = useState<InboxSort>(() => remembered(`${prefs}SightingsSort`, ['sure', 'unsure'], 'sure'));
  const [chosen, setSize] = useState<PhotoSize>(() => remembered(`${prefs}PhotoSize`, ['S', 'M', 'L'], defaultSize));
  const size = compact ? defaultSize : chosen; // the rail has no size control: it keeps the kind's own
  const sorted = useMemo(() => sortItems(items, sort), [items, sort]);
  const waiting = Math.max(total ?? 0, items.length);
  const shown = PHOTO_SIZES[size];
  const list = useRef<HTMLUListElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const pick =
    <T extends string>(key: string, set: (v: T) => void) =>
    (value: string) => {
      set(value as T);
      writePrefs({ ...readPrefs(), [key]: value });
    };
  const leaving = (key: string) => focusAfter(list.current, key, heading.current);
  const [cursor, select] = useState<string | null>(null);
  const selection = useMemo(() => ({ cursor, select }), [cursor]);
  const bulkOn = byBulk && !!bulk;
  // A phone puts the work first (Known's design pass, P5): the title and count on one line with one View menu
  // holding the list's view, sort, and size, and no introduction, so the first row is on screen without scrolling.
  const phone = (useMediaQuery(PHONE_WIDTH, undefined, { getInitialValueInEffect: false }) ?? false) && !compact;
  const views: ViewChoice[] = [
    ...(bulk && bulkLabel
      ? [
          {
            label: 'Show',
            value: bulkOn ? 'bulk' : 'list',
            options: [
              { value: 'list', label: 'List' },
              { value: 'bulk', label: bulkLabel },
            ],
            onChange: (v: string) => setByBulk(v === 'bulk'),
          },
        ]
      : []),
    ...(bulkOn
      ? []
      : [{ label: 'Sort', value: sort, options: SORTS, onChange: pick<InboxSort>(`${prefs}SightingsSort`, setSort) }]),
    {
      label: 'Picture size',
      value: size,
      options: [
        { value: 'S', label: 'Small' },
        { value: 'M', label: 'Medium' },
        { value: 'L', label: 'Large' },
      ],
      onChange: pick<PhotoSize>(`${prefs}PhotoSize`, setSize),
    },
  ];
  return (
    <section
      className={className}
      id={sectionId}
      aria-labelledby={headingId}
      data-photo-size={size}
      style={
        {
          '--naming-photo': `${shown.row}px`,
          '--naming-card': `${shown.card}px`,
          '--naming-tile': `${shown.tile}px`,
        } as CSSProperties
      }
    >
      {onBack && (
        <div className={who.back}>
          <Button variant="subtle" size="xs" onClick={onBack}>
            {backLabel ?? 'Back'}
          </Button>
        </div>
      )}
      <Group gap="sm" align="center" className={classes.header} wrap={phone ? 'nowrap' : undefined}>
        <Title id={headingId} order={2} ref={heading} tabIndex={-1}>
          {title}
        </Title>
        {!compact && waiting > 0 && (
          <span className={classes.count}>
            {waiting} waiting
            {!phone && waiting > items.length && items.length > 0 ? ` · ${items.length} shown` : ''}
          </span>
        )}
        {phone && (
          <div className={classes.view}>
            <ViewMenu choices={views} />
          </div>
        )}
        {!compact && !phone && bulk && bulkLabel && (
          <SegmentedControl
            size="xs"
            aria-label="View"
            data={[
              { value: 'list', label: 'List' },
              { value: 'bulk', label: bulkLabel },
            ]}
            value={bulkOn ? 'bulk' : 'list'}
            onChange={value => setByBulk(value === 'bulk')}
          />
        )}
        {!compact && !phone && (
          <Group gap="xs" ml="auto" align="center">
            {!bulkOn && (
              <>
                <Text size="xs" c="dimmed" aria-hidden>
                  Sort
                </Text>
                <Tooltip
                  label={
                    sort === 'sure'
                      ? 'Highest first. Click for lowest first.'
                      : 'Lowest first. Click for highest first.'
                  }
                  withArrow
                  events={{ hover: true, focus: true, touch: false }}
                >
                  <Button
                    size="xs"
                    variant="default"
                    aria-label={`${sortLabel} by certainty, ${sort === 'sure' ? 'highest' : 'lowest'} first`}
                    rightSection={
                      sort === 'sure' ? <IconArrowDown size={14} aria-hidden /> : <IconArrowUp size={14} aria-hidden />
                    }
                    onClick={() =>
                      pick<InboxSort>(`${prefs}SightingsSort`, setSort)(sort === 'sure' ? 'unsure' : 'sure')
                    }
                  >
                    Certainty
                  </Button>
                </Tooltip>
              </>
            )}
            <Text size="xs" c="dimmed">
              Size
            </Text>
            <SegmentedControl
              size="xs"
              aria-label="Picture size"
              data={['S', 'M', 'L']}
              value={size}
              onChange={pick<PhotoSize>(`${prefs}PhotoSize`, setSize)}
            />
          </Group>
        )}
      </Group>
      {!compact && !phone && intro && (
        <Text c="dimmed" mb="sm">
          {intro}
        </Text>
      )}
      {toast}
      {status}
      {bulkOn && bulk ? (
        bulk({ thumb: shown.thumb })
      ) : (
        <>
          {above}
          {empty && !status && !items.length && <PageEmpty title={empty} compact />}
          <Leaving value={leaving}>
            <Cursor value={selection}>
              <ul className={`who-list ${classes.list}`} aria-label={title} ref={list}>
                {sorted.map(item => (
                  <InboxSlot key={item.key} item={item} thumb={shown.thumb} />
                ))}
              </ul>
            </Cursor>
          </Leaving>
        </>
      )}
      {footer}
    </section>
  );
}

function InboxSlot({ item, thumb }: { item: InboxItem; thumb: boolean }) {
  return <>{item.render({ thumb })}</>;
}
