import { useSyncExternalStore } from 'react';

// What the page shell tells the app: the server fills {{names_json}} into window.NAMES in index.html.

export interface Shell {
  assistant: string;
  owner: string;
  cams: string[]; // exact native camera/RTSP manifest names
  activeCams: string[]; // Live/status only; disabled cameras stay in cams for recorded history
  views: Record<string, string>; // each named camera's name (vision.camera_names, #313), like "front camera"
  hotkeys: boolean; // dashboard.hotkeys: single-key shortcuts, off by default
}

const DEFAULTS: Shell = {
  assistant: 'assistant',
  owner: 'owner',
  cams: ['cam0'],
  activeCams: ['cam0'],
  views: {},
  hotkeys: false,
};

/** The shell values, read when called (tests set window.NAMES before rendering). */
export function shell(): Shell {
  const given = (globalThis as { NAMES?: Partial<Shell> }).NAMES ?? {};
  const merged = { ...DEFAULTS, ...given };
  const cams = merged.cams ?? DEFAULTS.cams;
  return { ...merged, cams, activeCams: given.activeCams ?? cams };
}

/** A successful local camera-use save changes Live's configured set without hiding recorded camera history. */
export function setActiveCameras(cameras: string[]): void {
  const host = globalThis as { NAMES?: Partial<Shell> };
  host.NAMES = { ...shell(), activeCams: [...cameras] };
  window.dispatchEvent(new Event('liberational:active-cameras'));
}

function subscribeActiveCameras(notify: () => void): () => void {
  window.addEventListener('liberational:active-cameras', notify);
  return () => window.removeEventListener('liberational:active-cameras', notify);
}

/** Subscribe Live consumers to successful camera-use saves. */
export function useActiveCameras(): string[] {
  return useSyncExternalStore(subscribeActiveCameras, () => shell().activeCams);
}

/** A saved change to the camera names (vision.camera_names) renames them on this page too, without waiting for a reload. */
export function setCameraNames(views: Record<string, string>): void {
  const host = globalThis as { NAMES?: Partial<Shell> };
  host.NAMES = { ...shell(), views: { ...views } };
  window.dispatchEvent(new Event('liberational:camera-names'));
}

function subscribeCameraNames(notify: () => void): () => void {
  window.addEventListener('liberational:camera-names', notify);
  return () => window.removeEventListener('liberational:camera-names', notify);
}

/** The camera names, following a save in Settings (Live's "Name cam0" link, #2595). */
export function useCameraNames(): Record<string, string> {
  return useSyncExternalStore(subscribeCameraNames, () => shell().views);
}
