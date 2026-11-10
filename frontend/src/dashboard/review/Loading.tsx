// Review's loading states (#250). The chips come from /review/presence, which can take a minute on a long day, so they
// load on their own: the rest of Review (the timeline, the transcript, and the player) works meanwhile.
import { Group, Skeleton, Stack, Text } from '@mantine/core';
import { ROOM, useSlow } from '../lib/loading';
import type { DayStillWorking } from './api';

const STILL_MS = 10_000; // past this, say it is still going rather than look stuck (#334)

const PILLS = [72, 96, 64, 110, 80, 58];

/** Placeholder chips while the day's objects and events are being found. working: an ended day the server is working
 *  through once (#435), with how far along it is, said instead of a loader that never says how long. */
export function ChipsSkeleton({ working }: { working?: DayStillWorking | null } = {}) {
  const still = useSlow(true, STILL_MS);
  return (
    <Stack
      gap={6}
      p={ROOM}
      aria-busy="true"
      aria-label="Loading the day's objects and events"
      data-testid="chips-skeleton"
    >
      <Group gap={6}>
        {PILLS.map(w => (
          <Skeleton key={w} height={22} width={w} radius="xl" />
        ))}
      </Group>
      {/* One line, no spinner: the skeleton already says something is coming (#2450). */}
      <Text size="xs" c="dimmed" role="status" aria-live="polite">
        {working
          ? working.message
          : still
            ? "Still finding the day's objects and events. The rest of Review works meanwhile."
            : "Finding the day's objects and events. This can take a minute; the rest of Review works meanwhile."}
      </Text>
    </Stack>
  );
}
