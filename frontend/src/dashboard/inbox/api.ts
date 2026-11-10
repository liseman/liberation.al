// Research jobs and their reports, for the Inbox (memory/research.py): Ask's questions taken slowly in the
// background. The list arrives on the live feed, so a job's progress shows without polling.
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

export interface Jobs {
  jobs: Job[];
  available: boolean;
  reason: string | null;
  unread: number;
}

export const JOBS_KEY = STATUS.research.key;

/** The jobs list, from the live feed (the server sends it when a job or its read mark changes). */
export const researchQuery = () => statusQuery<Jobs>('research');

export const jobQuery = (id: string) =>
  queryOptions({
    queryKey: [...JOBS_KEY, id],
    queryFn: ({ signal }) => request<Job>(`/research/jobs/${id}`, undefined, signal),
  });

export async function startResearch(client: QueryClient, question: string, day: string | null): Promise<Job> {
  const job = await request<Job>('/research/jobs', { question, ...(day ? { day } : {}) });
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
