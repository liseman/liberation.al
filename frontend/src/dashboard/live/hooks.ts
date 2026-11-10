import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { ReadPriority } from '../../shared/http';
import { liveQuery } from './feed';
import type { LiveData, LiveKind } from './types';

/** One kind of live data, kept current by the feed (undefined until the first event or fetch). */
export function useLive<K extends LiveKind>(kind: K, priority?: ReadPriority): LiveData[K] | undefined {
  const client = useQueryClient();
  return useQuery(liveQuery(client, kind, priority)).data;
}
