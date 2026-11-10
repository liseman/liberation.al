// Pausing cameras and microphones for privacy: GET/POST /devices/privacy (dashboard/privacy.py). A paused
// device's capture is stopped where it's owned, so nothing from it is recorded; the header icons toggle it and Live
// says "Paused for privacy" in its place.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { request } from '../shared/http';

export type PrivacyKind = 'camera' | 'microphone';
/** on; pausing (saved, capture hasn't stopped it yet); paused; resuming (saved, capture hasn't started it yet). */
export type PrivacyState = 'on' | 'pausing' | 'paused' | 'resuming';

export interface PrivacyDevice {
  name: string;
  kind: PrivacyKind;
  paused: boolean;
  state: PrivacyState;
}

export interface PrivacyView {
  control: boolean; // this browser may change it (the dashboard's machine)
  ready: boolean; // the built capture follows pauses
  devices: PrivacyDevice[];
}

export interface PrivacyChange {
  kind: PrivacyKind;
  paused: boolean;
  device?: string;
}

export const PRIVACY_KEY = ['privacy'] as const;
const SETTLED_MS = 15_000;
const PENDING_MS = 1_000;

/** One kind's devices: which are paused (or being paused), whether all are, and whether capture is still applying a
 *  change. */
export interface PrivacySummary {
  devices: PrivacyDevice[];
  paused: string[];
  all: boolean;
  pending: boolean;
}

export function summary(view: PrivacyView | undefined, kind: PrivacyKind): PrivacySummary {
  const devices = (view?.devices ?? []).filter(d => d.kind === kind);
  const paused = devices.filter(d => d.paused).map(d => d.name);
  return {
    devices,
    paused,
    all: devices.length > 0 && paused.length === devices.length,
    pending: devices.some(d => d.state === 'pausing' || d.state === 'resuming'),
  };
}

/** Whether this browser can pause devices of a kind now. */
export const canChange = (view: PrivacyView | undefined, kind: PrivacyKind): boolean =>
  !!view?.control && view.ready && summary(view, kind).devices.length > 0;

export function usePrivacy(): PrivacyView | undefined {
  return useQuery({
    queryKey: PRIVACY_KEY,
    queryFn: ({ signal }) => request<PrivacyView>('/devices/privacy', undefined, signal, 'background'),
    // Follow capture closely while it applies a change, so the icon turns as soon as the device has stopped.
    refetchInterval: query =>
      query.state.data?.devices.some(d => d.state === 'pausing' || d.state === 'resuming') ? PENDING_MS : SETTLED_MS,
  }).data;
}

export function usePrivacyChange() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (change: PrivacyChange) => request<PrivacyView>('/devices/privacy', change),
    onSuccess: view => client.setQueryData(PRIVACY_KEY, view),
  });
}

/** The set of cameras paused (or being paused), for Live. */
export function usePausedCameras(): ReadonlySet<string> {
  return new Set(summary(usePrivacy(), 'camera').paused);
}
