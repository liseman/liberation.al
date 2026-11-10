import { viewName } from '../dashboard/review/views';
import type { Sighting } from './api';

export const timestamp = (t: number): string => new Date(t * 1000).toLocaleString();

const KINDS: Record<Sighting['kind'], string> = {
  manual_label: 'manually labeled',
  visual_match: 'tentative visual match',
};

/** One line about a sighting: camera, time, and how it was made. */
export function sightingLabel(s: Sighting): string {
  return `${viewName(s.camera)} · ${timestamp(s.captured_at)} · ${KINDS[s.kind] ?? s.kind}${s.stale ? ' · older observation' : ''}`;
}
