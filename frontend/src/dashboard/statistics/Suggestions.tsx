// Suggested reports: what the corpus has data for and the page doesn't show yet (dashboard/reports.py's
// suggestions), five at a time with a More button. Built-in reports the owner removed come first, so they can come
// back. Adding one saves it to the page on the server.
import { Button, Group, Stack, Text, Title } from '@mantine/core';
import { useState } from 'react';
import { PageError, PageLoading } from '../lib/page';
import type { ChosenReport, Suggestion } from './api';
import { useSuggestions } from './api';
import { reportTitle } from './title';

export const PAGE = 5;

export function Suggestions({ removed, onAdd }: { removed: ChosenReport[]; onAdd: (r: ChosenReport) => void }) {
  const query = useSuggestions();
  const [shown, setShown] = useState(PAGE);
  const all: Suggestion[] = [
    ...removed.map(r => ({ ...r, why: 'A built-in report, removed from this page' })),
    ...(query.data ?? []),
  ];
  const visible = all.slice(0, shown);
  return (
    <Stack gap="sm">
      <Title order={2}>Suggested reports</Title>
      {query.isPending && !removed.length && <PageLoading label="suggestions" shape="rows" lines={3} />}
      {query.isError && (
        <PageError
          title="Suggestions couldn't load"
          detail={query.error.message}
          onRetry={() => void query.refetch()}
          busy={query.isFetching}
        >
          {null}
        </PageError>
      )}
      {!query.isPending && !all.length && (
        <Text size="sm" c="dimmed">
          Nothing more to suggest yet. Suggestions come from what's been recorded, so more appear as days are read.
        </Text>
      )}
      {visible.map(s => (
        <Group key={s.id} justify="space-between" wrap="nowrap" align="center">
          <div>
            <Text size="sm" fw={600}>
              {reportTitle(s)}
            </Text>
            <Text size="xs" c="dimmed">
              {s.why}
            </Text>
          </div>
          <Button
            variant="default"
            size="xs"
            // The row's text wraps; the button keeps its width (at 390 it was squeezed to "Adc").
            style={{ flexShrink: 0 }}
            onClick={() => onAdd({ id: s.id, kind: s.kind, arg: s.arg, title: s.title })}
            aria-label={`Add ${reportTitle(s)}`}
          >
            Add
          </Button>
        </Group>
      ))}
      {all.length > shown && (
        <Group>
          <Button variant="subtle" size="xs" onClick={() => setShown(shown + PAGE)}>
            More ({all.length - shown} left)
          </Button>
        </Group>
      )}
    </Stack>
  );
}
