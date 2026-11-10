// Loading and error states shared by the tabs (#250): skeletons shaped like what is coming, a loader that shows only
// when a wait runs long (so a quick load doesn't flash one), and an error with a retry, so a failed request never
// leaves a skeleton up for good. The error itself is LoadError in page.tsx (#2450).
import { Group, Loader, Skeleton, Stack, Text } from '@mantine/core';
import { useEffect, useState } from 'react';

/** How long a load runs before the loader joins the skeleton. */
export const SLOW_MS = 1200;
/** The space every loading state keeps around it, so none sits flush against what is around it (#335). Skeletons keep it on every side; the loader and the error keep it above and below. */
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
