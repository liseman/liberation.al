// GET /services, shared by the section and its menu badge. The badge and the other readers that only need the
// current state take it from the live feed (servicesStatusQuery, #52); the open section polls it closely.

import { useQuery } from '@tanstack/react-query';
import { useMemo, useSyncExternalStore } from 'react';
import { messageOf, request } from '../../../shared/http';
import { statusQuery } from '../../live/feed';
import {
  dismissible,
  type HistoryView,
  issueKey,
  SERVICES_KEY,
  type Service,
  type ServicesView,
  servicesBadge,
  unreadable,
} from './model';

export const SECTION_POLL_MS = 3000; // while the section shows

/** The services as the live feed last sent them: no poll of its own. */
export const servicesStatusQuery = () => statusQuery<ServicesView>('services');

export const servicesQuery = (refetchInterval: number) => ({
  queryKey: SERVICES_KEY,
  queryFn: ({ signal }: { signal?: AbortSignal }) => request<ServicesView>('/services', undefined, signal, 'page'),
  refetchInterval,
});

const DISMISSED = 'liberational.services.dismissed-issues';
let memory: string | null = null;
let disk: string | null | undefined;

function stored(): string | null {
  try {
    const value = localStorage.getItem(DISMISSED);
    if (value !== disk) {
      disk = value;
      memory = value;
    }
    return memory;
  } catch {
    return memory;
  }
}

function read(raw: string | null): Record<string, string> {
  try {
    const value: unknown = JSON.parse(raw ?? '{}');
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter(([, v]) => typeof v === 'string'));
  } catch {
    return {};
  }
}

function subscribe(changed: () => void): () => void {
  window.addEventListener(DISMISSED, changed);
  window.addEventListener('storage', changed);
  return () => {
    window.removeEventListener(DISMISSED, changed);
    window.removeEventListener('storage', changed);
  };
}

/** Share acknowledgements between the panel and sidebar, retaining an in-memory fallback if storage is blocked. */
export function useServiceIssues(raw: ServicesView | undefined) {
  const saved = useSyncExternalStore(subscribe, stored, () => null);
  const dismissed = useMemo(() => read(saved), [saved]);
  const view = useMemo(
    () =>
      raw
        ? {
            ...raw,
            services: raw.services.map(s =>
              dismissible(s) && dismissed[s.name] === issueKey(s)
                ? { ...s, issue_acknowledged: true, last_error: null }
                : s,
            ),
          }
        : raw,
    [raw, dismissed],
  );
  const dismiss = (s: Service) => {
    if (!dismissible(s)) return;
    memory = JSON.stringify({ ...read(stored()), [s.name]: issueKey(s) });
    try {
      localStorage.setItem(DISMISSED, memory);
      disk = memory;
    } catch {
      /* Acknowledgement still works for this page. */
    }
    window.dispatchEvent(new Event(DISMISSED));
  };
  return { view, dismiss };
}

/** The shell's health badge uses the small query/model layer, without loading the service controls. */
export function useServicesBadge() {
  const query = useQuery(servicesStatusQuery());
  const { view } = useServiceIssues(query.isError ? unreadable(messageOf(query.error)) : query.data);
  return servicesBadge(view);
}

/** Lifecycle history polls independently from the current service controls. */
export const serviceHistoryQuery = (hours: 24 | 168, enabled: boolean) => ({
  queryKey: ['service-history', hours] as const,
  queryFn: ({ signal }: { signal?: AbortSignal }) =>
    request<HistoryView>(`/services/history?hours=${hours}`, undefined, signal, 'page'),
  enabled,
  refetchInterval: 30_000,
  staleTime: 20_000,
  retry: false,
});
