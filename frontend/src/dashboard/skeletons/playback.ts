// Skeleton playback's data and timing (#1987): the spans to pick from, one span's pose frames, and which frame shows
// at a moment of playback.
import { request } from '../../shared/http';
import { MIN_CONF, type Pose } from '../pose/skeleton';

/** Lead-in and lead-out around a span, in seconds. */
export const LEAD_S = 2;
/** A frame shows until the next one, or for this long at most: past it the track wasn't seen, so nothing is drawn. */
export const HOLD_S = 0.5;

export interface Span {
  id: string;
  source: 'action' | 'walk-by';
  cam: string;
  kind: string;
  start: number;
  end: number;
  track_id: string;
  person: string | null;
}

export interface Frame extends Pose {
  ts: number;
}

export interface Poses {
  cam: string;
  track_id: string;
  start: number;
  end: number;
  frames: Frame[];
}

export const getSpans = (hours = 24) =>
  request<{ spans: Span[] }>(`/skeletons/api/spans?hours=${hours}`).then(r => r.spans);

export function getPoses(span: Span): Promise<Poses> {
  const q = new URLSearchParams({
    cam: span.cam,
    track: span.track_id,
    start: String(span.start - LEAD_S),
    end: String(span.end + LEAD_S),
  });
  return request<Poses>(`/skeletons/api/poses?${q}`);
}

/** The frame showing at `t`: the last one at or before it, if it's no older than HOLD_S. Frames are in time order. */
export function frameAt(frames: readonly Frame[], t: number): Frame | null {
  let lo = 0;
  let hi = frames.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((frames[mid]?.ts ?? 0) <= t) lo = mid + 1;
    else hi = mid;
  }
  const f = frames[lo - 1];
  return f && t - f.ts <= HOLD_S ? f : null;
}

/** The part of the frame to show to fit the track: the union of the joints drawn (MIN_CONF or surer), padded by PAD of
 *  its size, as [x, y, side] in 0 to 1 of the frame. A square in frame fractions keeps the frame's own aspect, so the
 *  figure isn't stretched. */
export const PAD = 0.15;
export function fitView(frames: readonly Frame[]): [number, number, number] {
  let [x1, y1, x2, y2] = [1, 1, 0, 0];
  for (const { keypoints } of frames)
    for (const [x = -1, y = -1, c = 0] of keypoints)
      if (c >= MIN_CONF && x >= 0 && y >= 0) {
        x1 = Math.min(x1, x);
        y1 = Math.min(y1, y);
        x2 = Math.max(x2, x);
        y2 = Math.max(y2, y);
      }
  if (x2 < x1) return [0, 0, 1];
  const side = Math.min(1, Math.max(x2 - x1, y2 - y1, 0.05) * (1 + 2 * PAD));
  return [(x1 + x2 - side) / 2, (y1 + y2 - side) / 2, side];
}
