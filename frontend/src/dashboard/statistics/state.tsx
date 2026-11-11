// A report's loading, error, and empty states, the same in every report: the shared QueryState, plus the one
// state only reports have (no day read yet, so nothing to draw but progress).
import { Text } from '@mantine/core';
import type { UseQueryResult } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { QueryState } from '../lib/page';

interface Progress {
  pending: string[];
  first: string;
  last: string;
}

export function ReportState<T extends Progress>({
  query,
  children,
}: {
  query: UseQueryResult<T>;
  children: (data: T) => ReactNode;
}) {
  return (
    <QueryState
      query={query}
      label="the report"
      errorTitle="This report couldn't load"
      lines={4}
      // The card reserves its kind's usual height while this is on screen (reports.module.css), so the cards below
      // don't jump as each report's data lands.
      loadingProps={{ 'data-report-loading': '' }}
    >
      {data =>
        data.pending.length && data.pending.includes(data.last) ? (
          // Not even today is read yet: there is nothing to draw, only progress. The server is still building it, so
          // the card keeps its reserved height here too, until the figures arrive.
          <Text size="sm" c="dimmed" role="status" data-report-loading="">
            Reading {data.pending.length} {data.pending.length === 1 ? 'day' : 'days'} of records. This report fills in
            as each one is done.
          </Text>
        ) : (
          children(data)
        )
      }
    </QueryState>
  );
}
