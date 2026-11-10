// Words for speakers in Who's this?: the label that means several people, and clipped quotes.
import { MULTI } from '../people/api';

// The section's id on Live: /#whos-this lands on it. Here, not in WhosThis.tsx, so Live can link to the section
// without pulling the whole panel into the entry bundle.
export const WHOS_THIS = 'whos-this';

/** A label for a speaker: "Multiple speakers" for the label that means several, "unknown" for none. */
export const showLabel = (s: string | null | undefined): string => (s === MULTI ? 'Multiple speakers' : s || 'unknown');

/** Text cut to n characters with an ellipsis. */
export const snip = (t: string | null | undefined, n = 40): string =>
  t && t.length > n ? `${t.slice(0, n - 1)}…` : (t ?? '');
