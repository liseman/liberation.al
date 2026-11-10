// Review's Summary as one story feed (#2275): the written stretches of what the timeline shows, in time order, each
// with the detected activities that fall inside it. An activity outside every stretch shown (nothing written there
// yet, or a stretch filtered out) is a row of its own at its time, so every activity in view is in the one list.
import { hm } from '../lib/format';
import type { Block, DetectedActivity } from './api';
import { activityKind } from './lanes';

export type FeedRow =
  | { kind: 'block'; at: number; block: Block; activities: DetectedActivity[] }
  | { kind: 'activity'; at: number; activity: DetectedActivity };

/** The feed's rows in time order: each block with the activities that start inside it (in the first such block when
 *  blocks overlap), and the rest as rows of their own. A block and an activity at the same time list the block first. */
export function feedRows(blocks: readonly Block[], activities: readonly DetectedActivity[]): FeedRow[] {
  const rows: FeedRow[] = blocks.map(block => ({ kind: 'block', at: block.start, block, activities: [] }));
  const loose: FeedRow[] = [];
  for (const activity of [...activities].sort((a, b) => a.start - b.start)) {
    const home = rows.find(
      (row): row is Extract<FeedRow, { kind: 'block' }> =>
        row.kind === 'block' && row.block.start <= activity.start && activity.start < row.block.end,
    );
    if (home) home.activities.push(activity);
    else loose.push({ kind: 'activity', at: activity.start, activity });
  }
  return [...rows, ...loose].sort((a, b) => a.at - b.at || (a.kind === 'block' ? -1 : 1));
}

/** How finely the feed reads, from the summary's journal level in seconds: "by the hour", "in 10-minute steps". */
export function stepWords(level: number): string {
  if (level >= 3600 && level % 3600 === 0) return level === 3600 ? 'by the hour' : `in ${level / 3600}-hour steps`;
  if (level >= 60 && level % 60 === 0) return `in ${level / 60}-minute steps`;
  return `in ${level}-second steps`;
}

/** The stretch the feed covers, in words: clock times under a day, else "the whole day" or "3 days". */
export function spanWords(start: number, end: number): string {
  const days = Math.round((end - start) / 86400);
  if (end - start < 0.95 * 86400) return `${hm(start)} to ${hm(end)}`;
  return days <= 1 ? 'the whole day' : `${days} days`;
}

/** A run of chips as Summary shows it: one chip, or 3 or more unconfirmed ones of a kind folded together. */
export type ChipRun<T> = { kind: 'one'; item: T } | { kind: 'fold'; name: string; items: T[] };

/** What an activity chip says happened: who did it, when the activity knows (its person, or the faces on its track,
 *  #2577 follow-up), then the activity in lower case ("Joe walked past"); else the activity alone ("Walked past"). */
export function activityWords(item: DetectedActivity): string {
  const what = item.name ?? activityKind(item);
  const who = item.person_name?.trim();
  if (!who) return what;
  // Lower only a capital that starts a word ("Walked"), not an acronym ("TV on").
  return `${who} ${/^\p{Lu}\p{Ll}/u.test(what) ? what[0]?.toLowerCase() + what.slice(1) : what}`;
}

/** Fold each kind's unconfirmed (not promoted) items into one run when there are at least `min` of them, placed
 *  where the kind first appears; everything else stays one by one, in order. */
export function foldRuns<T extends { promoted?: boolean }>(
  items: readonly T[],
  nameOf: (item: T) => string,
  min = 3,
): ChipRun<T>[] {
  const unconfirmed = new Map<string, T[]>();
  for (const item of items)
    if (!item.promoted) unconfirmed.set(nameOf(item), [...(unconfirmed.get(nameOf(item)) ?? []), item]);
  const placed = new Set<string>();
  const runs: ChipRun<T>[] = [];
  for (const item of items) {
    const name = nameOf(item);
    const group = item.promoted ? undefined : unconfirmed.get(name);
    if (!group || group.length < min) runs.push({ kind: 'one', item });
    else if (!placed.has(name)) {
      placed.add(name);
      runs.push({ kind: 'fold', name, items: group });
    }
  }
  return runs;
}
