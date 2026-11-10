// Review's Summary as one story feed: a sentence for each stretch of what the timeline shows, coarse for
// the whole day and finer as it zooms in (GET /review/summary), under parts of the day, each with the detected
// activities inside it. Every click plays from there: a stretch from its start, an activity from its own.
// Show only's chips highlight the stretches tagged with them, and "Only these" hides the rest. Picking a stretch keeps
// the list and selection in place; deliberate timeline pans and zooms still request the summary for the new range.
// A day's view opens with the day's own sentence, and each part of the day reads as one sentence with its
// activities, its hours folded under it; a part with no sentence yet shows its hours.
import { Switch, UnstyledButton } from '@mantine/core';
import { IconChevronRight, IconPlayerPlayFilled } from '@tabler/icons-react';
import { useQuery } from '@tanstack/react-query';
import { type ReactNode, useEffect, useRef, useState } from 'react';
import { hm, isoDay } from '../lib/format';
import { BlockSkeleton } from '../lib/loading';
import { LoadError } from '../lib/page';
import { useCameraNames } from '../shell';
import { activitiesQuery, type Block, type Blocks, blocksQuery, type DayRollup, type DetectedActivity } from './api';
import { useEngine, useReview } from './context';
import { activityKind } from './lanes';
import classes from './review.module.css';
import { activityWords, type FeedRow, feedRows, foldRuns, spanWords, stepWords } from './summaryFeed';
import type { View } from './time';
import { viewName } from './views';

/** The part of the day a time falls in, for the headings between stretches: the hours the server's part sentences
 *  sum up (review_summary.PARTS), so a heading and its sentence cover the same stretch. */
export function partOfDay(ts: number): string {
  const h = new Date(ts * 1000).getHours();
  return h < 5 ? 'Night' : h < 12 ? 'Morning' : h < 17 ? 'Afternoon' : 'Evening';
}

/** The feed's rows by local day and part of the day, in order. */
export function feedGroups(rows: readonly FeedRow[]): { date: string; part: string; rows: FeedRow[] }[] {
  const groups: { date: string; part: string; rows: FeedRow[] }[] = [];
  for (const row of rows) {
    const date = isoDay(new Date(row.at * 1000));
    const part = partOfDay(row.at);
    const last = groups.at(-1);
    if (last && last.date === date && last.part === part) last.rows.push(row);
    else groups.push({ date, part, rows: [row] });
  }
  return groups;
}

// plain_text.TOPICS: semantic labels, not speech/action evidence kinds or per-camera activity IDs.
const SEMANTIC_TOPICS = new Set([
  'work',
  'meal',
  'conversation',
  'call',
  'visit',
  'decision',
  'rest',
  'exercise',
  'chores',
  'errand',
  'media',
  'music',
  'travel',
  'planning',
]);

/** The tags a picked chip matches: its own id, and an established semantic topic for a day-summary section
 *  (event:decisions matches topic:decision). Other event IDs require their own attribution. */
function matches(chip: string): string[] {
  const event = /^event:(.+)$/.exec(chip)?.[1];
  if (!event) return [chip];
  return [
    chip,
    ...[event, event.replace(/s$/, '')].filter(topic => SEMANTIC_TOPICS.has(topic)).map(topic => `topic:${topic}`),
  ];
}

/** OR within each chip facet, AND across facets and cameras. Only the block's own tags establish attribution;
 *  its journal window and other timeline evidence do not. Topics match semantic event chips, not evidence kinds. */
export function picked(b: Block, sel: ReadonlySet<string>, cameras: readonly string[] = []): boolean {
  const facets = new Map<string, string[]>();
  for (const chip of sel) {
    const facet = chip.split(':')[0] ?? chip;
    facets.set(facet, [...(facets.get(facet) ?? []), ...matches(chip)]);
  }
  if (cameras.length)
    facets.set(
      'camera',
      cameras.map(camera => `camera:${camera}`),
    );
  return facets.size > 0 && [...facets.values()].every(tags => tags.some(tag => b.tags.includes(tag)));
}

/** The range to ask for: the timeline's when it zooms (its span changes) or moves off the stretches shown, else
 *  the one already asked, so the view following the playhead during playback doesn't ask again. */
export function nextAsk(asked: View | null, range: View | null, shown: readonly Block[]): View | null {
  if (!range || !asked) return range;
  const span = (v: View) => v.t1 - v.t0;
  if (Math.abs(span(range) - span(asked)) > 0.02 * span(asked)) return range;
  const from = Math.min(asked.t0, shown[0]?.start ?? asked.t0);
  const to = Math.max(asked.t1, shown.at(-1)?.end ?? asked.t1);
  return range.t0 >= from && range.t1 <= to ? asked : range;
}

export function SummaryBlocks({ day }: { day: string }) {
  const engine = useEngine();
  const cameraNames = useCameraNames(); // a camera's current name, following a rename in Settings
  const range = useReview(s => s.range);
  const rangeIntent = useReview(s => s.rangeIntent);
  const sel = useReview(s => s.sel);
  const cameras = useReview(s => s.cameras);
  const [only, setOnly] = useState(false);
  const [selected, setSelected] = useState<number | null>(null); // a block's stable start, independent of its list index
  const [unfolded, setUnfolded] = useState<ReadonlySet<string>>(() => new Set()); // parts showing their hours
  const [runsOpen, setRunsOpen] = useState<ReadonlySet<string>>(() => new Set()); // unconfirmed runs shown chip by chip
  const pickedNavigation = useRef<{ navigation: number; view: number } | null>(null);
  // The stretches showing: kept while finer ones load or are still being written, then swapped in place.
  const [kept, setKept] = useState<Blocks | null>(null);
  const [asked, setAsked] = useState<View | null>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: on a new range; kept is read, not followed
  useEffect(() => {
    const ownClick = pickedNavigation.current;
    // A known summary item may extend across midnight. Moving its cursor/view keeps that list and selected row;
    // the next deliberate viewport gesture, or an unrelated external jump, can request a different summary.
    if (ownClick && ownClick.navigation === engine.navigationVersion && ownClick.view === rangeIntent) return;
    setAsked(a => nextAsk(a, range, kept?.blocks ?? []));
  }, [range, rangeIntent, engine]);
  const requestedDay = asked ? isoDay(new Date(asked.t0 * 1000)) : day;
  const query = useQuery({ ...blocksQuery(requestedDay, asked?.t0 ?? 0, asked?.t1 ?? 0), enabled: !!asked });
  const fresh = query.data && !query.isPlaceholderData ? query.data : null;
  useEffect(() => {
    if (fresh && (fresh.blocks.length || fresh.status === 'ready' || !kept)) setKept(fresh);
  }, [fresh, kept]);
  const refining = !!kept && kept !== fresh && (query.isFetching || query.isPlaceholderData || !!fresh);
  const all = kept?.blocks ?? [];
  const filtering = sel.size > 0 || cameras.length > 0;
  const blocks = filtering && only ? all.filter(b => picked(b, sel, cameras)) : all;
  const missingCamera = cameras.length > 0 && all.some(b => !b.tags.some(tag => tag.startsWith('camera:')));
  const missingAttribution = [...sel].some(chip => {
    const facet = chip.split(':')[0];
    return all.some(
      b => !b.tags.some(tag => tag.startsWith(`${facet}:`) || (facet === 'event' && matches(chip).includes(tag))),
    );
  });
  // the stretch at the playhead; re-renders only when that changes, not on every tick
  const current = useReview(s =>
    s.t == null ? -1 : blocks.findIndex(b => b.start <= (s.t ?? 0) && (s.t ?? 0) < b.end),
  );

  const select = (b: Block) => {
    setSelected(b.start);
    engine.seekPlay(b.start); // plays from its start on every click, the same row again too
    pickedNavigation.current = { navigation: engine.navigationVersion, view: rangeIntent };
  };
  const play = (item: DetectedActivity) => engine.seekActivity(item.start, item.camera, { play: true });
  const activities = useQuery(activitiesQuery(day));
  const shownActivities = (activities.data?.items ?? []).filter(
    item => (!cameras.length || cameras.includes(item.camera)) && !(filtering && only),
  );

  let body: ReactNode;
  if (!kept && (!range || query.isPending))
    body = <BlockSkeleton heading={false} lines={6} label="Loading the summary" />;
  else if (!kept && query.isError)
    body = (
      <LoadError
        message="Couldn't load the summary of this stretch."
        onRetry={() => void query.refetch()}
        busy={query.isFetching}
      />
    );
  else if (!all.length)
    body = (
      <p className={classes.dim}>
        {kept?.status === 'ready' ? 'No written summary for this stretch.' : 'Summarizing this stretch…'}
      </p>
    );
  else if (!blocks.length) body = <p className={classes.dim}>No confirmed summary matches for these filters.</p>;
  else {
    const rows = feedRows(blocks, shownActivities);
    const groups = feedGroups(rows);
    const multipleDays = new Set(groups.map(g => g.date)).size > 1;
    // Parts of the day read as sentences only over the whole list: "Only these" lists single stretches.
    const days = new Map<string, DayRollup>(filtering && only ? [] : (kept?.days ?? []).map(d => [d.day, d]));
    const items: ReactNode[] = [];
    const chip = (item: DetectedActivity) => (
      <UnstyledButton
        key={item.id}
        type="button"
        className={classes.feedChip}
        data-candidate={item.promoted ? undefined : ''}
        title={`Play from ${hm(item.start)}${item.promoted ? '' : ' (unconfirmed)'}`}
        onClick={() => play(item)}
      >
        {hm(item.start)} {activityWords(item)}
        {item.promoted ? '' : '?'} · {viewName(item.camera, cameraNames)}
      </UnstyledButton>
    );
    // A stretch's chips, with 3 or more unconfirmed ones of a kind folded into one toggle: "read, 14 times (not yet
    // confirmed)" rather than 14 "read?" chips. Confirmed ones always show.
    const chips = (list: readonly DetectedActivity[], at: string): ReactNode[] => {
      const runs = foldRuns(list, a => a.name ?? activityKind(a));
      return runs.map(run => {
        if (run.kind === 'one') return chip(run.item);
        const key = `${at}:${run.name}`;
        const open = runsOpen.has(key);
        const toggle = (
          <UnstyledButton
            key={`run-${key}`}
            type="button"
            className={classes.feedChip}
            data-candidate=""
            aria-expanded={open}
            onClick={() =>
              setRunsOpen(previous => {
                const next = new Set(previous);
                if (!next.delete(key)) next.add(key);
                return next;
              })
            }
          >
            {`${run.name}, ${run.items.length} times (not yet confirmed)`}
          </UnstyledButton>
        );
        return open ? [toggle, ...run.items.map(chip)] : toggle;
      });
    };
    const row = (r: FeedRow): ReactNode => {
      if (r.kind === 'activity')
        return (
          <li key={`activity-${r.activity.id}`} className={classes.feedLoose}>
            <span className={classes.feedTime}>{hm(r.activity.start)}</span>
            <span>{chip(r.activity)}</span>
          </li>
        );
      const b = r.block;
      const i = blocks.indexOf(b);
      return (
        <li key={b.start} className={classes.feedRow} data-current={i === current ? '' : undefined}>
          <UnstyledButton
            type="button"
            className={classes.feedMain}
            data-picked={selected === b.start || (filtering && picked(b, sel, cameras)) ? '' : undefined}
            aria-pressed={selected === b.start}
            aria-current={i === current ? 'time' : undefined}
            title={`Play from ${hm(b.start)}`}
            onClick={() => select(b)}
          >
            <span className={classes.feedTime}>
              {hm(b.start)} to {hm(b.end)}
            </span>
            <span className={classes.feedText}>{b.text}</span>
            <IconPlayerPlayFilled size={14} className={classes.feedPlay} aria-hidden />
          </UnstyledButton>
          {r.activities.length > 0 && <span className={classes.feedChips}>{chips(r.activities, `b${b.start}`)}</span>}
        </li>
      );
    };
    let date = '';
    for (const g of groups) {
      const day = days.get(g.date);
      if (g.date !== date) {
        if (multipleDays)
          items.push(
            <li key={`date-${g.date}`} className={classes.part}>
              {new Date((g.rows[0]?.at ?? 0) * 1000).toLocaleDateString(undefined, {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
              })}
            </li>,
          );
        if (day?.text)
          items.push(
            <li key={`day-${g.date}`} className={classes.daySentence}>
              {day.text}
            </li>,
          );
        date = g.date;
      }
      const rollup = day?.parts.find(p => p.name === g.part);
      if (!rollup) {
        items.push(
          <li key={`part-${g.date}-${g.part}`} className={classes.part}>
            {g.part}
          </li>,
          ...g.rows.map(row),
        );
        continue;
      }
      const key = `${g.date}:${g.part}`;
      const open = unfolded.has(key);
      const stretches = g.rows.flatMap(r => (r.kind === 'block' ? [r.block] : []));
      const inside = g.rows.flatMap(r => (r.kind === 'block' ? r.activities : [r.activity]));
      const from = stretches[0]?.start ?? g.rows[0]?.at ?? rollup.start;
      const to = stretches.at(-1)?.end ?? rollup.end;
      items.push(
        <li key={`part-${key}`} className={classes.partRow}>
          <div className={classes.partHead}>
            <span className={classes.part}>{g.part}</span>
            {stretches.length > 0 && (
              <UnstyledButton
                type="button"
                className={classes.partFold}
                aria-expanded={open}
                onClick={() =>
                  setUnfolded(previous => {
                    const next = new Set(previous);
                    if (!next.delete(key)) next.add(key);
                    return next;
                  })
                }
              >
                <IconChevronRight size={12} className={classes.partChevron} data-open={open || undefined} aria-hidden />
                {open ? 'Hide' : 'Show'} {stretches.length === 1 ? '1 stretch' : `${stretches.length} stretches`}
              </UnstyledButton>
            )}
          </div>
          <UnstyledButton
            type="button"
            className={classes.feedMain}
            title={`Play from ${hm(from)}`}
            onClick={() => engine.seekPlay(from)}
          >
            <span className={classes.feedTime}>
              {hm(from)} to {hm(to)}
            </span>
            <span className={rollup.text ? classes.feedText : `${classes.feedText} ${classes.dim}`}>
              {rollup.text ?? `Summing up the ${g.part.toLocaleLowerCase()}…`}
            </span>
            <IconPlayerPlayFilled size={14} className={classes.feedPlay} aria-hidden />
          </UnstyledButton>
          {!open && inside.length > 0 && <span className={classes.feedChips}>{chips(inside, `p${key}`)}</span>}
        </li>,
      );
      if (open) items.push(...g.rows.map(row));
    }
    body = (
      <ul className={classes.feed} aria-label="Stretches and activities">
        {items}
      </ul>
    );
  }

  return (
    <section className={classes.blocks} aria-label="Summary by time">
      <header className={classes.blocksHead}>
        <h2 className={classes.blocksTitle}>Summary</h2>
        <span className={classes.dim} role="status">
          {refining && !query.isError ? 'Refining…' : ''}
        </span>
        {filtering && (
          <Switch
            size="xs"
            label="Only these"
            checked={only}
            onChange={e => setOnly(e.currentTarget.checked)}
            aria-label="Show only the stretches Show only picked"
          />
        )}
      </header>
      {kept && [kept.start, kept.end, kept.level].every(Number.isFinite) && (
        <p className={classes.dim}>
          {spanWords(kept.start, kept.end)} on the timeline,{' '}
          {kept.days?.some(d => d.parts.length) && !(filtering && only) ? 'by part of the day' : stepWords(kept.level)}
        </p>
      )}
      {(missingCamera || missingAttribution) && (
        <p className={classes.dim}>
          {missingCamera
            ? 'Camera attribution is unavailable for some summary blocks. Unverified blocks do not match camera filters.'
            : 'Some summary blocks lack attribution for the selected filters. Unverified blocks do not match.'}
        </p>
      )}
      <div className={classes.blocksBody} data-refining={refining || undefined}>
        {body}
      </div>
      {kept && query.isError && (
        <LoadError
          message="Couldn't refine the summary for this stretch."
          onRetry={() => void query.refetch()}
          busy={query.isFetching}
        />
      )}
      {kept?.status === 'partial' && all.length > 0 && (
        <p className={classes.dim}>The newest stretch is still being written.</p>
      )}
    </section>
  );
}
