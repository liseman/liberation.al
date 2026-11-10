// One transcript list for Live and Review: lines grouped into turns, only the turns in view (plus
// OVERSCAN either side) mounted, and the list held still as lines arrive. At the live end it keeps
// following the newest line; scrolled away, it holds on the first visible line, and a pill says how many lines arrived
// below. A turn mid-naming stays mounted while scrolled away, so its edit isn't lost.
// Live and Review are configurations of it: Live a tail of recent lines that play when clicked, Review a day whose
// current line follows the playhead (marked by setting aria-current, so following never re-renders the list) and
// seeks when clicked.
// A search dims the turns without a match and highlights the matches: every turn stays where it is, so the
// list's height and the line it holds on don't move as the words are typed.
import { Button } from '@mantine/core';
import { IconArrowDown } from '@tabler/icons-react';
import { defaultRangeExtractor, measureElement, observeElementRect, useVirtualizer } from '@tanstack/react-virtual';
import {
  type ReactNode,
  type Ref,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { plural } from '../lib/format';
import { BlockSkeleton } from '../lib/loading';
import type { Speakers, TranscriptEntry } from '../live/types';
import { useCorrections } from './corrections';
import { type LineHit, lineHits, lineKey, lineSearch, type Turn, turnKey, turnsOf } from './model';
import { type TurnHandlers, TurnView } from './TurnView';
import classes from './transcript.module.css';

/** This close to the bottom (px) counts as at the live end. */
export const END_PX = 24;
/** Turns mounted beyond the visible ones, either side. */
export const OVERSCAN = 3;
// A turn's height before it's measured: its header, then a row per line.
export const HEAD_PX = 30;
export const LINE_PX = 24;
/** The height assumed while the list has none (its tab hidden, or before layout): a first screenful is mounted. */
const UNLAID_PX = 600;

const lineRows = (el: HTMLElement) => [...el.querySelectorAll<HTMLElement>('[data-key]')];

/** Where the playhead is: Review's engine calls setCur with a line's index in `entries` (or -1). */
export interface Cursor {
  setCur(i: number, centre: boolean): void;
}

export interface TranscriptControls {
  /** Go to the live end and follow it. */
  toEnd(): void;
}

export function Transcript({
  entries,
  speakers,
  view,
  label,
  className,
  look,
  select,
  query = '',
  empty,
  onLine,
  scope,
  onChanged,
  follow = true,
  attach,
  top,
  more,
  onAtEnd,
  controls,
}: {
  /** The lines, or null while they load. */
  entries: readonly TranscriptEntry[] | null;
  speakers: Speakers;
  /** What the list shows (a day, a filter, a search): a new one is a new list, not new lines. */
  view: string;
  label: string;
  className?: string;
  /** Live's display choices. */
  look?: { balloons?: boolean; compact?: boolean; timestamps?: boolean };
  /** The turns to show, and which are someone else's (dimmed). */
  select?: (turns: Turn[]) => { t: Turn; other: boolean }[];
  /** A search: turns without a match are dimmed, and the matching words highlighted. */
  query?: string;
  /** What to say when no turn is shown. */
  empty: ReactNode;
  onLine: (e: TranscriptEntry, i: number) => void;
  /** Corrections are about this (a day); a new one drops their toast. */
  scope: string;
  /** Lines were renamed or corrected: read them again (tagged: the relabel lands a moment later). */
  onChanged: (tagged: boolean) => void;
  /** Whether to stay at the live end when there; false holds the list where it is. */
  follow?: boolean;
  /** A playhead to follow (Review); called again with null on unmount. */
  attach?: (cursor: Cursor | null, centre: boolean) => void;
  /** Above the turns, inside the scrolling box. */
  top?: ReactNode;
  /** Actions a page adds to every turn. */
  more?: ReactNode;
  /** The list reached or left the live end. */
  onAtEnd?: (atEnd: boolean) => void;
  controls?: Ref<TranscriptControls>;
}) {
  const box = useRef<HTMLElement>(null);
  const changedRef = useRef(onChanged);
  changedRef.current = onChanged;
  const fix = useCorrections(scope, () => changedRef.current(false));

  const shown = useMemo(() => {
    if (!entries?.length) return null;
    const all = turnsOf(entries, speakers);
    const turns = select ? select(all) : all.map(t => ({ t, other: false }));
    const turnOf = new Map<number, number>();
    const keys: string[] = [];
    turns.forEach(({ t }, k) => {
      for (const { e, i } of t.lines) {
        turnOf.set(i, k);
        keys.push(lineKey(e));
      }
    });
    return { turns, turnOf, keys };
  }, [entries, speakers, select]);
  // A search: the lines are indexed once (while searching), and each key typed only looks the words up, so the turns
  // aren't regrouped and only the turns whose matches change render again.
  const searching = !!query.trim();
  const search = useMemo(
    () => (searching && entries?.length ? lineSearch(entries, speakers) : null),
    [searching, entries, speakers],
  );
  const found = useMemo(() => {
    const hits = lineHits(search, query);
    const byTurn = new Map<string, ReadonlyMap<string, LineHit>>();
    if (hits.size)
      for (const { t } of shown?.turns ?? []) {
        const mine = new Map<string, LineHit>();
        for (const { e } of t.lines) {
          const hit = hits.get(lineKey(e));
          if (hit) mine.set(lineKey(e), hit);
        }
        if (mine.size) byTurn.set(turnKey(t), mine);
      }
    return byTurn;
  }, [search, query, shown]);
  const turns = shown?.turns ?? [];

  const [kept, setKept] = useState<ReadonlySet<string>>(() => new Set());
  const onLineRef = useRef(onLine);
  onLineRef.current = onLine;
  const handlers = useMemo<TurnHandlers>(
    () => ({
      onLine: (e, i) => onLineRef.current(e, i),
      onNot: fix.openNot,
      onTrash: lines => void fix.trash(lines),
      onTv: lines => void fix.tv(lines),
      onNotTv: lines => void fix.notTv(lines),
      onNamed: tagged => changedRef.current(tagged),
      onKeep: (key, on) =>
        setKept(prev => {
          if (prev.has(key) === on) return prev;
          const next = new Set(prev);
          if (on) next.add(key);
          else next.delete(key);
          return next;
        }),
    }),
    [fix.openNot, fix.trash, fix.tv, fix.notTv],
  );

  const list = useRef<HTMLDivElement>(null);
  const [margin, setMargin] = useState(0); // the list's offset in the scroll box, below what's on top of it
  const virtual = useVirtualizer({
    count: turns.length,
    getScrollElement: () => box.current,
    estimateSize: k => HEAD_PX + LINE_PX * (turns[k]?.t.lines.length ?? 1),
    getItemKey: k => turnKey(turns[k]?.t),
    overscan: OVERSCAN,
    // Review keeps inactive panels mounted with display:none. Their zero-height turns must not replace estimates:
    // each zero pulls more turns into the fallback viewport, synchronously mounting/measuring until React hits
    // its update-depth limit. Keep the last size (or estimate) until layout exists; ResizeObserver remeasures on show.
    measureElement: (element, entry, instance) => {
      const size = measureElement(element, entry, instance);
      if (size > 0) return size;
      const index = instance.indexFromElement(element);
      return instance.itemSizeCache.get(instance.options.getItemKey(index)) ?? instance.options.estimateSize(index);
    },
    scrollMargin: margin,
    initialRect: { width: 0, height: UNLAID_PX }, // the first screenful mounts with the lines, not a render later
    observeElementRect: (instance, cb) =>
      observeElementRect(instance, rect => cb(rect.height ? rect : { ...rect, height: UNLAID_PX })),
    scrollToFn: (offset, { adjustments = 0 }, instance) => {
      if (instance.scrollElement) instance.scrollElement.scrollTop = offset + adjustments;
    },
    rangeExtractor: range => {
      const base = defaultRangeExtractor(range);
      if (!kept.size) return base;
      const held = turns.flatMap(({ t }, k) => (kept.has(turnKey(t)) ? [k] : []));
      return [...new Set([...base, ...held])].sort((a, b) => a - b);
    },
  });
  // Read only when what's above the list changes: reading offsetTop on every render forced a layout in the middle of
  // each scroll frame.
  const hasList = turns.length > 0;
  const hasTop = top != null && top !== false;
  // biome-ignore lint/correctness/useExhaustiveDependencies: the list's offset changes with these, not with renders
  useLayoutEffect(() => {
    setMargin(list.current?.offsetTop ?? 0);
  }, [hasTop, hasList]);
  // Every scroll this list makes goes through the virtualizer, so the offset it adjusts from (as turns are measured)
  // is the one just set, not the last scroll event's.
  const scrollList = (to: number) => virtual.scrollToOffset(Math.max(0, to), { align: 'start' });
  const total = virtual.getTotalSize();
  const items = virtual.getVirtualItems();

  // Where the list is: at the live end, or held on its first visible line (its key and offset from the top). Lines
  // known are those shown when it was last at the end; the rest landed below since.
  const atEnd = useRef(true);
  const following = useRef(follow);
  following.current = follow;
  const anchor = useRef<{ key: string; offset: number } | null>(null);
  const seenKeys = useRef<Set<string> | null>(null);
  const keysRef = useRef<readonly string[]>([]);
  keysRef.current = shown?.keys ?? [];
  const [fresh, setFresh] = useState(0);
  const viewSeen = useRef<string | null>(null);
  const onAtEndRef = useRef(onAtEnd);
  onAtEndRef.current = onAtEnd;
  const remember = useCallback(() => {
    const el = box.current;
    if (!el) return;
    const was = atEnd.current;
    atEnd.current = el.scrollHeight - el.scrollTop - el.clientHeight <= END_PX;
    if (was !== atEnd.current) onAtEndRef.current?.(atEnd.current);
    const at = el.getBoundingClientRect().top;
    const first = lineRows(el).find(r => r.getBoundingClientRect().bottom > at);
    anchor.current = first ? { key: first.dataset.key ?? '', offset: first.getBoundingClientRect().top - at } : null;
    if (atEnd.current && following.current) {
      seenKeys.current = new Set(keysRef.current);
      setFresh(0);
    }
  }, []);
  const tracking = () => atEnd.current && following.current;
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs after the rows re-render, to put the list back
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    if (viewSeen.current !== view) {
      viewSeen.current = view;
      seenKeys.current = null;
      anchor.current = null;
      if (!attach) atEnd.current = true; // a new list opens at its end; Review's opens on the playhead (below)
    }
    if (!shown) return;
    if (tracking()) scrollList(el.scrollHeight);
    else if (anchor.current) {
      const { key, offset } = anchor.current;
      const row = lineRows(el).find(r => r.dataset.key === key);
      if (row) scrollList(el.scrollTop + row.getBoundingClientRect().top - el.getBoundingClientRect().top - offset);
    }
    if (!seenKeys.current || tracking()) seenKeys.current = new Set(shown.keys);
    const seen = seenKeys.current;
    setFresh(tracking() ? 0 : shown.keys.filter(k => !seen.has(k)).length);
  }, [shown, view]);
  // New turns are measured after they mount: at the live end, stay at the end as they grow.
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs when the measured height changes
  useLayoutEffect(() => {
    const el = box.current;
    if (el && tracking() && shown) scrollList(el.scrollHeight);
  }, [total]);
  // Following again (Live's button) goes to the end.
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs when follow turns on
  useLayoutEffect(() => {
    if (follow) toEnd();
  }, [follow]);
  const toEnd = () => {
    const el = box.current;
    if (!el) return;
    const was = atEnd.current;
    atEnd.current = true;
    // Say so before remember() reads the new position: it reports only a change, and this one is already made, so
    // Live would stay not following after its new-lines pill.
    if (!was) onAtEndRef.current?.(true);
    scrollList(el.scrollHeight);
    remember();
  };
  useImperativeHandle(controls, () => ({ toEnd }));

  // The current line's key: the engine marks it again after every refresh, and that never scrolls by itself. Its
  // row may not be mounted (scrolled away, or not yet): it's marked as it mounts, and a jump to it is finished
  // (centred on the line itself) once its turn has mounted.
  const curKey = useRef<string | null>(null);
  const pendingCentre = useRef<string | null>(null);
  const centred = useRef<string | null>(null); // the view the playhead's line was last centred in, on attaching
  const entriesRef = useRef(entries);
  entriesRef.current = entries;
  const turnOfRef = useRef(shown?.turnOf);
  turnOfRef.current = shown?.turnOf;
  const centreRow = (el: HTMLElement, row: Element) => {
    // Scroll only the list: scrollIntoView would also scroll the page when the list is taller than the window.
    const r = row.getBoundingClientRect();
    const p = el.getBoundingClientRect();
    scrollList(el.scrollTop + r.top - p.top - (el.clientHeight - r.height) / 2);
  };
  // Mark the current line in the rows mounted now, and finish a jump whose line just mounted.
  useLayoutEffect(() => {
    const el = box.current;
    if (!el || !attach) return;
    const key = curKey.current;
    for (const row of el.querySelectorAll('[aria-current="time"]'))
      if (row.getAttribute('data-key') !== key) row.removeAttribute('aria-current');
    const row = key == null ? null : el.querySelector(`[data-key="${CSS.escape(key)}"]`);
    if (!row) return;
    row.setAttribute('aria-current', 'time');
    if (pendingCentre.current === key) {
      pendingCentre.current = null;
      centreRow(el, row);
    }
  });
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs again after the rows re-render, so the current line is marked in the new rows
  useEffect(() => {
    if (!attach) return;
    const centre = !!entries?.length && centred.current !== view;
    if (centre) {
      centred.current = view;
      atEnd.current = false; // a list just shown opens on the playhead's line; its scroll event says if that's the end
    }
    attach(
      {
        setCur(i, centre) {
          const el = box.current;
          if (!el) return;
          const e = i >= 0 ? entriesRef.current?.[i] : undefined;
          const key = e ? lineKey(e) : null;
          const moved = key !== curKey.current;
          curKey.current = key;
          for (const row of el.querySelectorAll('[aria-current="time"]'))
            if (row.getAttribute('data-key') !== key) row.removeAttribute('aria-current');
          if (key == null) return;
          const row = el.querySelector(`[data-key="${CSS.escape(key)}"]`);
          if (row) {
            row.setAttribute('aria-current', 'time');
            const r = row.getBoundingClientRect();
            const p = el.getBoundingClientRect();
            if (centre || (moved && (r.top < p.top || r.bottom > p.bottom))) centreRow(el, row);
            return;
          }
          // Not mounted: bring its turn into view, then centre the line once it has mounted.
          const k = turnOfRef.current?.get(i);
          if (k == null || !(centre || moved)) return;
          pendingCentre.current = key;
          const [offset] = virtual.getOffsetForIndex(k, 'center') ?? [];
          if (offset != null) scrollList(offset);
        },
      },
      centre,
    );
    return () => attach(null, false);
  }, [attach, shown]);

  const rows =
    entries == null ? (
      <BlockSkeleton heading={false} lines={8} label="Loading the transcript" />
    ) : !turns.length ? (
      <div className={classes.empty}>{empty}</div>
    ) : (
      <div ref={list} className={classes.turns} style={{ height: total }} data-turns>
        {items.map(v => {
          const x = turns[v.index];
          if (!x) return null;
          return (
            <div
              key={v.key}
              data-index={v.index}
              ref={virtual.measureElement}
              className={classes.turnSlot}
              style={{ transform: `translateY(${v.start - margin}px)` }}
            >
              <TurnView
                t={x.t}
                other={x.other}
                missed={searching && !found.has(String(v.key))}
                hits={found.get(String(v.key))}
                speakers={speakers}
                saving={fix.saving}
                more={more}
                h={handlers}
              />
            </div>
          );
        })}
      </div>
    );
  return (
    <section
      className={className ? `${classes.box} ${className}` : classes.box}
      ref={box}
      aria-label={label}
      onScroll={remember}
      // biome-ignore lint/a11y/noNoninteractiveTabindex: the list scrolls, so the keyboard must reach it (WCAG)
      tabIndex={0}
      data-look={look?.balloons ? 'balloons' : undefined}
      data-density={look?.compact ? 'compact' : undefined}
      data-ts={look?.timestamps ? 'always' : undefined}
    >
      {fix.overlay}
      {top}
      {rows}
      {fresh > 0 && (
        <div className={classes.newLines}>
          <Button size="compact-xs" radius="xl" leftSection={<IconArrowDown size={12} aria-hidden />} onClick={toEnd}>
            {plural(fresh, 'new line', 'new lines')}
          </Button>
        </div>
      )}
    </section>
  );
}
