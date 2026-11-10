// How the dashboard looks in this browser (transcript style, density, the waveform, mirroring the cameras), kept
// with the theme in liberational.prefs (shared/prefs.ts) so the old pages read the same choices.
import { create } from 'zustand';
import { type Prefs, readPrefs, writePrefs } from '../../shared/prefs';

/** The Resource monitor (usage/UsagePanel.tsx): open or not, where it floats, and its size, in CSS pixels. */
export interface UsagePanelPref {
  open?: boolean;
  x?: number;
  y?: number;
  w?: number;
  h?: number;
}

export interface LivePrefs extends Prefs {
  transcript?: 'lines' | 'balloons';
  waveform?: boolean;
  density?: 'comfortable' | 'compact';
  timestamps?: 'hover' | 'always';
  statusLine?: boolean;
  mirror?: boolean; // every camera, from before cameras were mirrored one by one; mirrorCams overrides it
  mirrorCams?: Record<string, boolean>; // each camera's own choice (Live's button, Settings' Mirror)
  gettingStartedHidden?: string; // Live's Getting started, hidden with these steps open ("models,voice")
  gettingStartedDone?: boolean; // all six were done once: Getting started stays away for good
  reviewBoxes?: boolean; // Review draws people and things boxes on the large camera (default on)
  reviewMarkers?: ('people' | 'pets' | 'objects')[]; // per-viewer Review overlays; people only when unset
  reviewTimeline?: 'compact' | 'expanded'; // Review's timeline: one strip, or a lane per camera
  reviewObjectsKnownOnly?: boolean; // Review's Objects shows only remembered objects, not the camera's labels (default on)
  usagePanel?: UsagePanelPref; // the Resource monitor, closed when unset
  skeletons?: boolean; // Live and Review draw body skeletons over the camera (default on)
  liveTiles?: boolean; // Live's camera overview plays every camera live, not a picture every few seconds (default on)
}

export const usePrefs = create<LivePrefs>(() => readPrefs());

/** Whether this browser shows camera cam mirrored: its own choice, else the old one for every camera. */
export const mirrored = (p: LivePrefs, cam: string): boolean => {
  const own = p.mirrorCams && Object.hasOwn(p.mirrorCams, cam) ? p.mirrorCams[cam] : undefined;
  return typeof own === 'boolean' ? own : !!p.mirror;
};

/** Mirror camera cam in this browser, or not. */
export function setMirrored(cam: string, on: boolean): void {
  setPref('mirrorCams', { ...(readPrefs() as LivePrefs).mirrorCams, [cam]: on });
}

/** Set one preference. It reads storage first, so a theme picked meanwhile (Mantine writes it) isn't lost. */
export function setPref<K extends keyof LivePrefs>(key: K, value: LivePrefs[K]): void {
  const next = { ...readPrefs(), [key]: value };
  writePrefs(next);
  usePrefs.setState(next, true);
}
