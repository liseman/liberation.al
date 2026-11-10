// Editing a plan's DSL text (#2859). The text stays the one source of truth, as in the library's playground: every
// edit is a splice into it, and the drawing is made again from it. The library edits the upstream DSL, which has no
// `x-` statements (spatial memory's extensions, plan/dsl.ts), so those lines are set aside while editing and put back
// on save, unchanged.
import { dslSpliceAt, lineOf, parseDsl } from '../../../vendor/floorplan/index.js';

/** A plan's text split into what the library edits and the `x-` lines it doesn't know. */
export function split(text: string): { base: string; ext: string[] } {
  const lines = text.split('\n');
  return {
    base: lines.filter(l => !l.trimStart().startsWith('x-')).join('\n'),
    ext: lines.filter(l => l.trimStart().startsWith('x-')),
  };
}

/** The text again, with the `x-` lines set aside put back at its end. */
export function join(base: string, ext: string[]): string {
  const body = base.replace(/\n+$/, '');
  return ext.length ? `${body}\n\n${ext.join('\n')}\n` : `${body}\n`;
}

/** The text with room `id` named `name` (the DSL writes the name as a JSON string). */
export function renameRoom(text: string, id: string, name: string): string {
  return dslSpliceAt(text, ['rooms', id, 'name'], JSON.stringify(name.trim())).text;
}

/** The text without opening `index` (the plan's `openings[index]`: a door, window, or cased opening). */
export function removeOpening(text: string, index: number): string {
  const line = lineOf(parseDsl(text).positions, `openings[${index}]`);
  if (line === undefined) return text;
  const lines = text.split('\n');
  lines.splice(line - 1, 1);
  return lines.join('\n');
}

/** The text with a door between `a` and `b` (room ids, or "exterior"), `width` metres wide: at `at` (a point on the
 *  wall between them, plan metres) when given, else centred on the one wall they share. A door that doesn't fit, or
 *  a pair that shares several walls with no `at`, shows as a finding, as any edit does. */
export function addDoor(text: string, a: string, b: string, width: number, at?: [number, number]): string {
  const body = text.replace(/\n+$/, '');
  const r = (v: number) => Math.round(v * 100) / 100;
  const where = at ? ` at ${r(at[0])},${r(at[1])}` : '';
  return `${body}\ndoor ${a}>${b}${where} w${r(width)}\n`;
}

/** The middle of the longest wall two spaces share, from a plan's walls (each with the spaces either side), or null
 *  when they share none: where a new door between them goes. */
export function doorSpot(
  walls: {
    start: [number, number];
    end: [number, number];
    neg: { kind: string; id?: string };
    pos: { kind: string; id?: string };
  }[],
  a: string,
  b: string,
): [number, number] | null {
  const side = (o: { kind: string; id?: string }) => (o.kind === 'exterior' ? 'exterior' : (o.id ?? ''));
  let best: [number, number] | null = null;
  let longest = 0;
  for (const w of walls) {
    const pair = [side(w.neg), side(w.pos)];
    if (!(pair.includes(a) && pair.includes(b))) continue;
    const length = Math.hypot(w.end[0] - w.start[0], w.end[1] - w.start[1]);
    if (length > longest) {
      longest = length;
      best = [(w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2];
    }
  }
  return best;
}
