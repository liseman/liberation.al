import { request } from '../../shared/http';

export interface Citation {
  label: string;
  ts: number | null;
  camera: string | null;
  source: string | null;
  removed: boolean;
  verified?: boolean;
}
export interface HistoryEntry {
  id: string;
  question: string;
  surface: string;
  day: string | null;
  scope: string;
  mode: string;
  status: string;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  model: string | null;
  result: unknown;
  citations: Citation[];
  answer_removed: boolean;
  answer_unverified?: boolean;
  provenance_complete?: boolean;
}
export interface HistoryPage {
  items: HistoryEntry[];
  next_before: string | null;
}
export function historyURL(day: string | null, before?: string): string {
  const params = new URLSearchParams({ limit: '20' });
  if (day) {
    params.set('surface', 'review');
    params.set('day', day);
  }
  if (before !== undefined) params.set('before', String(before));
  return `/pack-ask/history?${params}`;
}
export const readHistory = (day: string | null, before?: string, signal?: AbortSignal): Promise<HistoryPage> =>
  request(historyURL(day, before), undefined, signal);
export const readEntry = (id: string, signal?: AbortSignal): Promise<HistoryEntry> =>
  request(`/pack-ask/history/${encodeURIComponent(id)}`, undefined, signal);

export async function deleteHistory(id: string | null, day: string | null): Promise<void> {
  const url = id
    ? `/pack-ask/history/${encodeURIComponent(id)}`
    : day
      ? `/pack-ask/history?${new URLSearchParams({ surface: 'review', day })}`
      : '/pack-ask/history';
  // Through request(), so a failed delete reaches the page's log like every other request; recover false, since a
  // DELETE must never be sent twice.
  const data = await request<{ deleted?: number }>(url, {}, undefined, 'page', false, 'DELETE');
  if (typeof data?.deleted !== 'number') throw new Error('Invalid deletion response');
}
