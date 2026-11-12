import { viewName } from '../dashboard/review/views';
import type { Sighting } from './api';

export const timestamp = (t: number): string => new Date(t * 1000).toLocaleString();

const BACKENDS: Record<string, string> = { sam3: 'SAM 3', sam2: 'SAM 2', edgetam: 'EdgeTAM' };

/** The model that found an object, in words: SAM 3's concept prompt, or the point grid on SAM 2 or EdgeTAM. */
export const backendName = (backend: string): string => BACKENDS[backend] ?? 'the object model';

const KINDS: Record<Sighting['kind'], string> = {
  manual_label: 'manually labeled',
  visual_match: 'tentative visual match',
};

/** One line about a sighting: camera, time, and how it was made. */
export function sightingLabel(s: Sighting): string {
  return `${viewName(s.camera)} · ${timestamp(s.captured_at)} · ${KINDS[s.kind] ?? s.kind}${s.stale ? ' · older observation' : ''}`;
}
