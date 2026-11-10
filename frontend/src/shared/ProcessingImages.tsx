import { Text } from '@mantine/core';

/** Media is processed off the page; only saved pictures enter its grids. */
export function ProcessingImages({
  count = 0,
  unavailable = 0,
  kind = 'detection',
}: {
  count?: number;
  unavailable?: number;
  kind?: 'detection' | 'video';
}) {
  return (
    <>
      {count > 0 && (
        <Text size="xs" c="dimmed" mt="xs" role="status">
          Processing {count.toLocaleString()} {kind}
          {count === 1 ? '' : 's'} in the background…
        </Text>
      )}
      {unavailable > 0 && (
        <Text size="xs" c="dimmed" mt="xs" role="status">
          {unavailable.toLocaleString()} {kind}
          {unavailable === 1 ? '' : 's'} can't be shown.
        </Text>
      )}
    </>
  );
}
