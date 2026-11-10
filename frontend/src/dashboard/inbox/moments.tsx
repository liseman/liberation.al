// The moments a research answer cites, as a short timeline under it (#2674, Joe's Inbox board, idea 1): each row is
// the moment's time (opening it in Review), a still of the camera then (or what kind of source it is), the evidence's
// own words on one line, and where it came from. It shows what happened, and when, without opening the folded list of
// everything the research looked at.
import { useRef } from 'react';
import { thumbUrl, useClipThumb, useNearScreen } from '../activities/clipThumb';
import { AppLink } from '../ask/bits';
import { hm } from '../lib/format';
import { viewName } from '../review/views';
import classes from './inbox.module.css';

export interface Moment {
  code: string; // "C6"
  ts: number;
  href: string;
  source: string; // camera, journal, transcript, ...
  camera: string | null;
  speaker: string | null;
  text: string;
}

const CODE = /\[?\b([A-Z]{1,2}\d{1,4})\b\]?/g;
const SOURCE: Record<string, string> = {
  camera: 'Camera note',
  journal: 'Journal',
  transcript: 'Said',
  facts: 'Fact',
  timeline: 'Timeline',
  detections: 'Face sighting',
};
export const sourceWord = (source: string): string => SOURCE[source] ?? 'Source';
const GONE = 'Video no longer kept';

/** The evidence items the answer cites, by their codes ("C6", "[T1]"), once each, oldest first. An item without a time or a
 *  link into Review has no moment to show; a code no item has is the model's own and is left in the text. */
export function citedMoments(answer: string, evidence: readonly Record<string, unknown>[]): Moment[] {
  const byCode = new Map<string, Record<string, unknown>>();
  for (const e of evidence)
    for (const code of [e.id, e.label]) if (typeof code === 'string' && code && !byCode.has(code)) byCode.set(code, e);
  const out: Moment[] = [];
  const seen = new Set<string>();
  for (const [, code] of answer.matchAll(CODE)) {
    const e = code ? byCode.get(code) : undefined;
    if (!code || !e || seen.has(code)) continue;
    const ts = typeof e.ts === 'number' ? e.ts : Number.NaN;
    const href = typeof e.href === 'string' ? e.href : typeof e.link === 'string' ? e.link : '';
    if (!Number.isFinite(ts) || !href) continue;
    seen.add(code);
    const text = [e.text, e.quote].find(v => typeof v === 'string' && v.trim()) as string | undefined;
    out.push({
      code,
      ts,
      href,
      source: typeof e.source === 'string' ? e.source : '',
      camera: typeof e.camera === 'string' && e.camera ? e.camera : null,
      speaker: typeof e.speaker === 'string' && e.speaker ? e.speaker : null,
      text: (text ?? '').replace(/\s+/g, ' ').trim(),
    });
  }
  return out.sort((a, b) => a.ts - b.ts);
}

/** A camera's frame at the moment, once the row is near the screen: the shared scene-thumbnail route, which queues a
 *  decode it hasn't made yet (202) and is asked again with backoff. A 404 means the moment's video isn't kept (past
 *  the retention window, or dropped in a reset), and the box says so rather than naming the source (#3165). */
function Still({ m }: { m: Moment }) {
  const box = useRef<HTMLSpanElement>(null);
  const near = useNearScreen(box);
  const cam = m.source === 'camera' ? m.camera : null;
  const thumb = useClipThumb(cam ?? '', m.ts, m.ts + 5, near && !!cam);
  const gone = !!cam && thumb.state === 'none';
  return (
    <span ref={box} className={classes.still} aria-hidden={!gone || undefined}>
      {cam && thumb.state === 'ready' ? (
        <img src={thumbUrl(cam, m.ts, m.ts + 5)} alt="" />
      ) : (
        <span>{gone ? GONE : sourceWord(m.source)}</span>
      )}
    </span>
  );
}

export function CitedMoments({ moments }: { moments: readonly Moment[] }) {
  if (moments.length < 2) return null; // one cited moment is the answer's own link
  return (
    <ol className={classes.moments} aria-label="The moments it cites">
      {moments.map(m => {
        const where = m.camera ? viewName(m.camera) : m.speaker;
        return (
          <li key={m.code}>
            <AppLink to={m.href}>
              <time dateTime={new Date(m.ts * 1000).toISOString()}>{hm(m.ts)}</time>
            </AppLink>
            <Still m={m} />
            <span className={classes.momentText} title={m.text}>
              {m.text || m.code}
            </span>
            <span className={classes.momentSource}>
              {sourceWord(m.source)}
              {where ? ` · ${where}` : ''}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
