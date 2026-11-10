// Chip colours and wording. A chip keeps the same colour on every load; selected chips never share one.
import { secondsText } from '../lib/format';
import type { Chip } from './types';

// Mid-tone hues that read on both the light and dark timeline lanes.
const PALETTE = [
  '#e8590c',
  '#1c7ed6',
  '#2f9e44',
  '#ae3ec9',
  '#d6336c',
  '#0c8599',
  '#f08c00',
  '#5f3dc4',
  '#66a80f',
  '#c2255c',
] as const;

const hash = (id: string): number => {
  let h = 0;
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return h % PALETTE.length;
};

/** A chip's usual colour. */
export const chipColor = (id: string): string => PALETTE[hash(id)] ?? PALETTE[0];

/** Colours for the selected chips, in the chips' own order: each chip's usual colour, or the next one along when an
 *  earlier selected chip already has it, so no two selected chips look alike. */
export function chipColors(chips: readonly Chip[], sel: ReadonlySet<string>): Record<string, string> {
  const used = new Set<number>();
  const out: Record<string, string> = {};
  for (const c of chips) {
    if (!sel.has(c.id)) continue;
    let i = hash(c.id);
    for (let n = 0; n < PALETTE.length && used.has(i); n++) i = (i + 1) % PALETTE.length;
    used.add(i);
    out[c.id] = PALETTE[i] ?? PALETTE[0];
  }
  return out;
}

/** How long a chip was around in all: "45 s", "12 min", "2 h 5 min" (lib/format's secondsText). */
export const duration = secondsText;

export const KIND_NAME: Record<string, string> = { person: 'People', object: 'Objects', event: 'Events' };

/** A chip's tooltip: how many times, how long in all (left out when the chip doesn't measure time, like a camera's
 *  "came into view" chip, whose spans are only padded so the moment plays), and its first note, which can say how
 *  sure the estimate is. */
export function chipTitle(c: Chip): string {
  let title = `${c.label}: ${c.count} ${c.count === 1 ? 'time' : 'times'}`;
  if (c.total_s > 0) title += `, ${duration(c.total_s)} in all`;
  const note = c.notes?.[0];
  return note ? `${title}. ${note}` : title;
}
