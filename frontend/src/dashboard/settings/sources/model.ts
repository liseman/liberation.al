// Which kind each ingest source is (#1331), so Settings shows it where it belongs: a camera source with the
// cameras, anything else that brings media in under Imported media. Links are pasted on the Ingest tab instead.
// A source not named here (a new adapter) is imported media until it's added to CAMERAS.

export type SourceKind = 'camera' | 'media' | 'none';

/** Ingest adapters that record cameras. Native RTSP discovery uses the separate network-cameras API. */
const CAMERAS: ReadonlySet<string> = new Set(['wyze']);

/** Sources Settings doesn't show: links are one-off, pasted on the Ingest tab. */
const ELSEWHERE: ReadonlySet<string> = new Set(['link']);

export function kindOf(name: string): SourceKind {
  if (ELSEWHERE.has(name)) return 'none';
  return CAMERAS.has(name) ? 'camera' : 'media';
}

/** The id of each kind's card, which a link (/settings#network-cameras) scrolls to. */
export const CARD_ID: Record<Exclude<SourceKind, 'none'>, string> = {
  camera: 'network-cameras',
  media: 'imported-media',
};
