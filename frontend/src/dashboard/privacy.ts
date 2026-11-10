// Pausing cameras and microphones for privacy: GET/POST /devices/privacy (dashboard/privacy.py). A paused
// device's capture is stopped where it's owned, so nothing from it is recorded; the header icons toggle it and Live
// says "Paused for privacy" in its place. The view arrives on the live feed (live/feed.ts), not by polling.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { request } from '../shared/http';
import { STATUS, statusQuery } from './live/feed';

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

export const PRIVACY_KEY = STATUS.privacy.key;

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

/** The pauses, from the live feed: the server sends them as capture applies a change, so the icon turns as soon
 *  as the device has stopped. */
export function usePrivacy(): PrivacyView | undefined {
  return useQuery(statusQuery<PrivacyView>('privacy')).data;
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
