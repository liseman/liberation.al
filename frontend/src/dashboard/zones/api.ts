// Owner-drawn zones on each camera's view (#1316, see/zones.py): the current set per camera, a camera's saved
// versions, and saving a camera's whole set.
import { queryOptions } from '@tanstack/react-query';
import { request } from '../../shared/http';
import type { Point } from './geometry';

export interface Zone {
  id?: string; // absent until the server gives a new zone one
  name: string;
  polygon: Point[];
  kind?: 'area' | 'screen'; // a screen (#2468): nobody is counted in it, and nobody it shows is anyone
}

/** A camera's zone set as one save left it. */
export interface Definition {
  id: string;
  cam: string;
  zones: Zone[];
  at: number;
}

export const ZONES_KEY = ['zones'] as const;

export const zonesQuery = () =>
  queryOptions({
    queryKey: ZONES_KEY,
    queryFn: ({ signal }) => request<{ cameras: Record<string, Definition> }>('/zones', undefined, signal),
  });

export const zoneHistoryQuery = (cam: string) =>
  queryOptions({
    queryKey: [...ZONES_KEY, 'history', cam],
    queryFn: ({ signal }) =>
      request<{ versions: Definition[] }>(`/zones/history?cam=${encodeURIComponent(cam)}`, undefined, signal),
  });

/** Save `cam`'s whole set, edited from version `base` (null for a camera with none yet). */
export const saveZones = (cam: string, zones: Zone[], base: string | null) =>
  request<{ ok: boolean; definition: Definition }>('/zones', { cam, zones, base });
