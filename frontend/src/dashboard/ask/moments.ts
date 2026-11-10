import { hm } from '../lib/format';
import { shownName } from '../lib/speakers';
import { viewName } from '../review/views';
import { plain } from './format';
import { citedEvidence } from './MarkAnswer';

const MAX_CHIPS = 8;

/** A moment an answer cites: when, and the camera view or the speaker it came from. */
export interface Cited {
  t: number;
  label: string; // "08:12 front camera", "08:12 Dana", or "08:12"
}

/** The moments an answer cites, in time order, one chip per minute and source, at most eight. */
export function citedMoments(result: unknown): Cited[] {
  const seen = new Set<string>();
  const out: Cited[] = [];
  for (const e of [...citedEvidence(result)].sort((a, b) => a.ts - b.ts)) {
    const cam = plain(e.camera).trim();
    const said = plain(e.speaker).trim();
    const who = said && shownName(said); // never a placeholder like "Speaker 261"
    const label = [hm(e.ts), cam ? viewName(cam).toLowerCase() : who].filter(Boolean).join(' ');
    if (seen.has(label)) continue;
    seen.add(label);
    out.push({ t: e.ts, label });
    if (out.length === MAX_CHIPS) break;
  }
  return out;
}
