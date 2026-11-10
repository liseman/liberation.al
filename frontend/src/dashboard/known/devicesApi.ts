// Known › Devices' reads beyond the link card (#1326): Wi-Fi comings and goings today, and where each linked thing
// and person was last seen (GET /device-comings and /device-whereabouts, dashboard/device_links.py). Both are
// read-only and passive: what the router and the Bluetooth sensors already recorded.
import { queryOptions } from '@tanstack/react-query';
import { request } from '../../shared/http';
import { DEVICE_LINKS_KEY, type DeviceLocation, type Target } from '../settings/devices/links';

export interface Stay {
  start: number;
  end: number;
}

export interface WifiDevice {
  mac: string;
  name: string | null; // the hostname the router reported, when it did
  vendor?: string | null; // who made it, by its address's IEEE block or the router's word (proximity/vendors.py)
  place: string | null; // the access point: the owner's name for it, else the router's, else its MAC
  place_id?: string | null; // the access point's MAC, to rename it by
  linked: { id: string; target: Target; name: string } | null;
  location?: DeviceLocation | null; // where the owner said it lives (#3274)
  stays: Stay[]; // oldest first
}

export interface Comings {
  at: number;
  start: number; // local midnight
  gap_seconds: number; // a device last seen this close to `at` is still here
  rows: number;
  devices: WifiDevice[]; // newest stay first
}

export interface Whereabout {
  target: Target;
  name: string;
  links: number;
  last_seen: number | null;
  place: string | null;
  place_id?: string | null; // a Wi-Fi access point's MAC; null for a Bluetooth sensor
  rssi: number | null;
  source: string | null;
}

export interface Whereabouts {
  at: number;
  days: number; // how far back it looked
  links: Whereabout[]; // things first, then people
  caveat: string;
}

export const comingsQuery = () =>
  queryOptions({
    queryKey: [...DEVICE_LINKS_KEY, 'comings'] as const, // under the links' key: a link saved refreshes it
    queryFn: ({ signal }) => request<Comings>('/device-comings', undefined, signal),
    refetchInterval: 60_000,
  });

export const whereaboutsQuery = () =>
  queryOptions({
    queryKey: [...DEVICE_LINKS_KEY, 'whereabouts'] as const,
    queryFn: ({ signal }) => request<Whereabouts>('/device-whereabouts', undefined, signal),
    refetchInterval: 60_000,
  });

export type HomeStatus = 'recent' | 'not_seen' | 'unknown';

export interface HomeRow {
  name: string;
  status: HomeStatus; // recent: a linked device seen within the gap; not_seen: the sensors were running and didn't
  last_seen: number | null;
  source: string | null;
  place: string | null;
  links: number;
}

export interface Presence {
  at: number;
  start: number;
  gap_seconds: number;
  people: HomeRow[];
  caveat: string; // a linked device seen isn't proof its owner is here
}

export const presenceQuery = () =>
  queryOptions({
    queryKey: [...DEVICE_LINKS_KEY, 'presence'] as const,
    queryFn: ({ signal }) => request<Presence>('/device-presence', undefined, signal),
    refetchInterval: 60_000,
  });
