// Research jobs and their reports, for the Inbox (memory/research.py): Ask's questions taken slowly in the
// background, sent through Ask's one route with Deep research on. The list arrives on the live feed, so a job's
// progress shows without polling.
import { type QueryClient, queryOptions } from '@tanstack/react-query';
import { request } from '../../shared/http';
import { STATUS, statusQuery } from '../live/feed';

export type JobStatus = 'queued' | 'running' | 'done' | 'failed';

export interface Search {
  tool: string | null;
  args: Record<string, unknown> | null;
  found: number | null;
}

export interface Report {
  status: string;
  answer: string;
  model: string | null;
  warnings: string[];
  evidence: Record<string, unknown>[];
  looked_at: { sources: Record<string, number>; evidence: number; searches: Search[]; window: string | null };
}

export interface Job {
  id: string;
  question: string;
  day: string | null;
  status: JobStatus;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  took_s?: number;
  error?: string;
  read: boolean;
  result?: Report;
  answer_status?: string | null; // how a finished report's answer came out ("ok", "partial"), for the list
}

/** What an answer looks through (research.reach): its search rounds at most, the seconds they may take, the sources
 *  they read, and the model that answers. */
export interface Reach {
  deep: boolean;
  rounds: number;
  search_s: number;
  sources: string[];
  model: string;
  day: string | null;
}

export interface Jobs {
  jobs: Job[];
  available: boolean;
  reason: string | null;
  unread: number;
  /** How long research usually takes here, in seconds (the median of recent reports), or null before any */
  typical_s?: number | null;
  reach?: { quick: Reach; deep: Reach };
}

export const JOBS_KEY = STATUS.research.key;

/** The jobs list, from the live feed (the server sends it when a job or its read mark changes). */
export const researchQuery = () => statusQuery<Jobs>('research');

export const jobQuery = (id: string) =>
  queryOptions({
    queryKey: [...JOBS_KEY, id],
    queryFn: ({ signal }) => request<Job>(`/research/jobs/${id}`, undefined, signal),
  });

/** Send a question to deep research: Ask's one route with `deep` on. The reply is the queued research job. */
export async function startResearch(client: QueryClient, question: string, day: string | null): Promise<Job> {
  const job = await request<Job>('/pack-ask/jobs', { question, deep: true, ...(day ? { day } : {}) });
  await client.invalidateQueries({ queryKey: JOBS_KEY, exact: true });
  return job;
}

export async function markRead(client: QueryClient, id: string, read: boolean): Promise<void> {
  await request(`/research/jobs/${id}/read`, { read });
  await client.invalidateQueries({ queryKey: JOBS_KEY });
}

export async function removeJob(client: QueryClient, id: string): Promise<void> {
  await request(`/research/jobs/${id}/delete`, {});
  await client.invalidateQueries({ queryKey: JOBS_KEY });
}
