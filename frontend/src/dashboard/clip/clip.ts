// One recorded clip played by Review's engine outside Review: an Activities card, its popup, and a People
// card's floating video all open a clip the same way. useClip owns the engine and the listing; the views around it
// (a card, a floating window) only arrange it.
import { useCallback, useEffect, useRef, useState } from 'react';
import { messageOf } from '../../shared/http';
import { ReviewEngine } from '../review/engine';
import { daySegments } from '../review/gatewaySegments';
import type { RawSegment } from '../review/types';

export type LoadSegments = (start: number, end: number, signal: AbortSignal) => Promise<RawSegment[]>;

/** What a clip opened on: its first moment, and its camera's segments. */
export interface Clip {
  at: number;
  segments: RawSegment[];
}

/** Where a clip is and whether it's playing, carried between two views of it (a card and its popup). */
export interface Moment {
  t: number;
  play: boolean;
}

export const GONE = 'This recording is no longer kept.';

/** The shortest window a clip opens with, in seconds. A People card's sighting can be a single moment (0 s long):
 *  a tenth of a second of video holds no frame the decoder can show, so the player said "No video at 01:03:14".
 *  Shorter clips run on to this length. */
export const MIN_CLIP_S = 5;

/** The camera's segments over [start, end] in time order, the last one cut off at end, so playback stops there. */
export function clipSegments(segments: readonly RawSegment[], cam: string, start: number, end: number): RawSegment[] {
  return segments
    .filter(s => s.cam === cam && s.end > start && s.start < end)
    .sort((a, b) => a.start - b.start)
    .map(s => (s.end > end ? { ...s, end } : s));
}

export interface ClipOptions {
  cam: string;
  start: number;
  end: number;
  loadSegments?: LoadSegments;
  clip?: Clip | null; // already listed (a popup over its card): opens without listing again
  moment?: Moment | null; // where to start, and whether to play; else the clip's opening
  attempt?: number; // a retry: bump it to list and open again
}

/** An embedded engine on one camera with the clip open in it, and how far it got. The engine starts when the view
 *  mounts and stops when it unmounts or the clip changes. */
export function useClip({
  cam,
  start,
  end,
  loadSegments,
  clip: given = null,
  moment = null,
  attempt = 0,
}: ClipOptions) {
  const [engine] = useState(
    () => new ReviewEngine({ embedded: true, cams: [cam], fetchNotes: async () => ({ cams: {} }) }),
  );
  const [clip, setClip] = useState<Clip | null>(given);
  const [loading, setLoading] = useState(!given);
  const [error, setError] = useState('');
  const listing = useRef<AbortController | null>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: attempt is a retry, which opens the clip again
  useEffect(() => {
    const ctl = new AbortController();
    listing.current = ctl;
    const open = (found: Clip) => {
      engine.openClip(found.at, found.segments);
      if (moment) engine.seek(moment.t, { play: moment.play });
      setClip(found);
      setError('');
      setLoading(false);
    };
    engine.start();
    if (given) open(given);
    else {
      setLoading(true);
      const to = Math.max(end, start + MIN_CLIP_S);
      // By default only the clip's own camera is listed: one spans request, not one per camera.
      const load: LoadSegments = loadSegments ?? ((a, b, signal) => daySegments(a, b, { signal, cams: [cam] }));
      void load(start, to, ctl.signal)
        .then(segments => {
          if (ctl.signal.aborted) return;
          const found = clipSegments(segments, cam, start, to);
          if (!found.length) throw new Error(GONE);
          open({ at: Math.max(start, found[0]?.start ?? start), segments: found });
        })
        .catch(err => {
          if (ctl.signal.aborted) return;
          setError(messageOf(err));
          setLoading(false);
        });
    }
    return () => {
      ctl.abort();
      engine.stop();
    };
  }, [engine, cam, start, end, loadSegments, given, moment, attempt]);
  /** Drop a listing still on its way (the view is closing), so its late answer never opens the clip. */
  const cancel = useCallback(() => listing.current?.abort(), []);
  return { engine, clip, loading, error, cancel };
}
