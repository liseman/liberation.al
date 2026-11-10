// A day page's cited evidence: what each source behind its citations can show now, from
// GET /journal/evidence?day=, for Review's player. Each one is shown with the Journal tab's SourceMedia (evidence.tsx),
// so both age the same way: the recording while it's kept, then a promoted clip or still, then why not.
import { queryOptions } from '@tanstack/react-query';
import { HttpError, request } from '../../shared/http';
import type { MediaSource } from './evidence';

/** One cited source as it can be shown now (memory/journal_evidence.py's resolve_day). */
export interface EvidenceSource {
  label: string; // the day page's citation, like "h3"
  stream: string | null;
  ts_start: number;
  ts_end: number;
  stage: 'video' | 'clip' | 'still' | 'audio' | 'expired';
  caption: string;
  camera?: string | null;
  t?: number; // the moment (seconds): what video, a clip, or audio plays from
  url?: string; // a clip's or a still's file
  still?: string; // a clip's still, for its poster
  note?: string; // why an expired one can't be shown: "footage expired", "no footage was kept", "audio expired"
}

export const journalEvidenceKey = (day: string) => ['journal-evidence', day] as const;

/** A day's cited sources, resolved. What a source can show changes only when its video ages out or is promoted (each
 *  at most once a night), so one read per opening is enough. */
export const journalEvidenceQuery = (day: string) =>
  queryOptions({
    queryKey: journalEvidenceKey(day),
    queryFn: async ({ signal }): Promise<EvidenceSource[]> =>
      (await request<{ sources?: EvidenceSource[] }>(`/journal/evidence?day=${day}`, undefined, signal)).sources ?? [],
    staleTime: 10 * 60_000,
    refetchOnWindowFocus: false,
    // A refusal (a bad day, 4xx) won't change on asking again; a server or network failure might.
    retry: (count, error) => !(error instanceof HttpError && error.status < 500) && count < 2,
  });

/** How far from the playhead a source's moment may be and still be shown there: a promoted clip spans 6 s around its
 *  moment (CLIP_S), plus a little slack. A source's whole span isn't used: a long one (an action over minutes) would
 *  show a clip from minutes away as if it were now. */
export const NEAR_S = 5;

/** The cited source of `camera` whose moment is nearest `t`, within NEAR_S of it; null for none. `camera` is Review's
 *  camera id, the native camera name (shell().cams), which is the name journal sources carry (their segment rows'
 *  `cam`) and the one memory/journal_evidence.py reads video under. A source with no camera (speech) is audio, never
 *  a camera's picture, so it isn't a candidate. */
export function evidenceAt(sources: readonly EvidenceSource[], camera: string, t: number): EvidenceSource | null {
  let best: EvidenceSource | null = null;
  let bestGap = Number.POSITIVE_INFINITY;
  for (const s of sources) {
    if (!s.camera || s.camera !== camera) continue;
    const gap = Math.abs(t - (s.t ?? s.ts_start));
    if (gap <= NEAR_S && gap < bestGap) {
      best = s;
      bestGap = gap;
    }
  }
  return best;
}

/** A day's source as SourceMedia shows one. */
export const toMedia = (s: EvidenceSource): MediaSource => ({
  stage: s.stage,
  caption: s.caption,
  camera: s.camera,
  url: s.url,
  still: s.still,
  note: s.note,
  start: s.ts_start,
  end: Math.max(s.ts_end, s.ts_start),
});
