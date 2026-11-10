// What one cited source can show now, as memory/journal_evidence.py resolves it (#2808): the recording's still while
// its video is kept ("video"), a promoted clip or still, the room's audio for speech, or why there's nothing left
// ("footage expired", "no footage was kept", "audio expired"). Shared by everything that shows journal evidence: the
// Journal tab's preview (#2811, from GET /journal?entry=) and a day page's citations (GET /journal/evidence?day=).
import { useClipThumb } from '../activities/clipThumb';
import classes from './journal.module.css';

/** A resolved source, as both routes give it: `resolve`'s stage and fields, and the source's span. */
export interface MediaSource {
  stage: 'video' | 'clip' | 'still' | 'audio' | 'expired';
  caption: string;
  camera?: string | null;
  /** A promoted clip's or still's file. */
  url?: string;
  /** A promoted clip's still, for its poster. */
  still?: string;
  /** Why an expired source has nothing to show. */
  note?: string;
  /** The source's span, seconds. */
  start: number;
  end: number;
  /** The audio's URL, when the route gave one; else it's the room's audio over the span. */
  audio?: string;
}

/** The room's audio over a span (at least a second of it). */
export const audioUrl = (start: number, end: number): string =>
  `/audio/range?${new URLSearchParams({ start: start.toFixed(2), end: Math.max(end, start + 1).toFixed(2) })}`;

/** "Footage expired." from the resolver's note. */
export const sentence = (note: string | undefined): string =>
  note ? `${note[0]?.toUpperCase()}${note.slice(1)}.` : 'Nothing kept.';

/** The recording's still for a moment whose video is kept (the scene thumbnail, prepared on request). */
function Still({ source }: { source: MediaSource }) {
  const { url, state, waiting } = useClipThumb(source.camera ?? '', source.start, source.end, true);
  if (state === 'ready') return <img className={classes.still} src={url} alt={source.caption || `${source.camera}`} />;
  if (waiting) return <div className={classes.stillWait}>Preparing the still…</div>;
  return null;
}

/** One source's media: a still, a clip, a picture, audio, or a line saying why there's none. */
export function SourceMedia({ source: s }: { source: MediaSource }) {
  switch (s.stage) {
    case 'video':
      return <Still source={s} />;
    case 'clip':
      return (
        // biome-ignore lint/a11y/useMediaCaption: a promoted clip has no sound; its caption is the text beside it
        <video className={classes.clip} controls preload="none" src={s.url} poster={s.still} />
      );
    case 'still':
      return <img className={classes.still} src={s.url} alt={s.caption} />;
    case 'audio':
      return (
        // biome-ignore lint/a11y/useMediaCaption: room audio has no captions; its transcript is the text beside it
        <audio controls preload="none" src={s.audio ?? audioUrl(s.start, s.end)} className={classes.audio} />
      );
    default:
      return <p className={classes.expired}>{sentence(s.note)} What was seen and heard is still kept.</p>;
  }
}
