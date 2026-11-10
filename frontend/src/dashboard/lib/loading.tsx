// Loading and error states shared by the tabs: skeletons shaped like what is coming, a loader that shows only
// when a wait runs long (so a quick load doesn't flash one), and an error with a retry, so a failed request never
// leaves a skeleton up for good. The error itself is LoadError in page.tsx.
import { Group, Loader, Skeleton, Stack, Text, VisuallyHidden } from '@mantine/core';
import { useEffect, useState } from 'react';
import classes from './loading.module.css';

/** Counts read once this session and shown again while the same tab's count reloads, so the tab row does not flash. */
const sessionCounts = new Map<string, number>();

export function sessionCount(key: string): number | undefined {
  return sessionCounts.get(key);
}

export function setSessionCount(key: string, n: number): void {
  sessionCounts.set(key, n);
}

/** Inline width for a tab count line, matched to one digit and the words ("9 to name"). */
export const countLineChars = (what: string): number => 2 + what.length;

/** How long a load runs before the loader joins the skeleton. */
export const SLOW_MS = 1200;
/** The space every loading state keeps around it, so none sits flush against what is around it. Skeletons keep it on every side; the loader and the error keep it above and below. */
export const ROOM = 'xs';

/** True once `active` has been true for `ms` without a break. */
export function useSlow(active: boolean, ms = SLOW_MS): boolean {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    setSlow(false);
    if (!active) return;
    const t = setTimeout(() => setSlow(true), ms);
    return () => clearTimeout(t);
  }, [active, ms]);
  return active && slow;
}

/** A loader and a line saying what is loading, shown only once the wait runs long. */
export function SlowLoader({ active, label }: { active: boolean; label: string }) {
  const slow = useSlow(active);
  if (!slow) return null;
  return (
    <Group gap="xs" py={ROOM} role="status" aria-live="polite">
      <Loader size="xs" />
      <Text size="sm" c="dimmed">
        {label}
      </Text>
    </Group>
  );
}

/** A count whose read is still pending ("12 to name", "3 animals need a name"): a skeleton in the line where its
 *  words go, inline, one text line tall and `chars` characters wide (about the words it stands for), so the number
 *  arriving shifts nothing. A span, so it can sit inside a button, a tab, or a heading's line. The skeleton itself is
 *  hidden from screen readers; `label`, when given, is read in its place (a tab's count leaves it out, so the tab is
 *  still named by its title). Show it only while the query is pending: a failed read shows its error treatment. */
export function CountSkeleton({ chars, label }: { chars: number; label?: string }) {
  return (
    <>
      <Skeleton
        renderRoot={props => <span {...props} />}
        height="1em"
        width={`${chars}ch`}
        radius="sm"
        display="inline-block"
        style={{ verticalAlign: 'middle' }}
        aria-hidden="true"
        data-testid="count-skeleton"
      />
      {label && <VisuallyHidden>{label}</VisuallyHidden>}
    </>
  );
}

/** Placeholder cards in the People roster grid: a round picture and two lines, like a roster row. */
export function CardsSkeleton({ cards = 4, label }: { cards?: number; label: string }) {
  return (
    <div
      className={classes.cards}
      role="group"
      aria-busy="true"
      aria-label={label}
      data-testid="cards-skeleton"
    >
      {Array.from({ length: cards }, (_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: identical placeholders with no identity of their own
        <div key={i} className={classes.card}>
          <Skeleton circle height={40} />
          <Stack gap={6} style={{ flex: 1, minWidth: 0 }}>
            <Skeleton height={12} width="70%" radius="sm" />
            <Skeleton height={8} width="50%" radius="sm" />
          </Stack>
        </div>
      ))}
    </div>
  );
}

/** Placeholder rows shaped like a list of people: an avatar and two lines of text. */
export function RowsSkeleton({ rows = 6, label }: { rows?: number; label: string }) {
  return (
    <Stack gap="sm" p={ROOM} role="group" aria-busy="true" aria-label={label} data-testid="rows-skeleton">
      {Array.from({ length: rows }, (_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: identical placeholders with no identity of their own
        <Group key={i} gap="sm" wrap="nowrap">
          <Skeleton circle height={36} />
          <Stack gap={6} style={{ flex: 1 }}>
            <Skeleton height={10} width={`${50 + ((i * 17) % 35)}%`} radius="sm" />
            <Skeleton height={8} width={`${30 + ((i * 23) % 40)}%`} radius="sm" />
          </Stack>
        </Group>
      ))}
    </Stack>
  );
}

/** Placeholder lines shaped like a detail pane or a block of text: an optional heading bar, then lines. */
export function BlockSkeleton({
  lines = 5,
  heading = true,
  label,
}: {
  lines?: number;
  heading?: boolean;
  label: string;
}) {
  return (
    <Stack gap="sm" p={ROOM} role="group" aria-busy="true" aria-label={label} data-testid="block-skeleton">
      {heading && <Skeleton height={22} width="45%" radius="sm" />}
      {Array.from({ length: lines }, (_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: identical placeholders with no identity of their own
        <Skeleton key={i} height={10} width={`${70 + ((i * 13) % 30)}%`} radius="sm" />
      ))}
    </Stack>
  );
}
