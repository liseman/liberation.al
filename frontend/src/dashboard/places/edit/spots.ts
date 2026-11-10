// Camera spots in a plan's text: the `x-camera <camera> at <x>,<y> [height <m>] [yaw <deg>] [fov <deg>]` lines
// (plan/dsl.ts), edited one line at a time. A change rewrites only that camera's line where it stands (a new camera's
// line goes at the end), so every other byte of the plan stays as it was, and a spot set to what it already says
// leaves the text exactly as it was. The positions placer reads the same lines (spatial/plan_import.py `spots`): an
// indented line is not a spot to it, and the later of two lines for one camera wins, so the same holds here.

export type Pt = [number, number];

export interface Spot {
  at: Pt;
  /** Metres above the floor. */
  height?: number;
  /** Degrees clockwise from up the page: which way the camera looks. */
  yaw?: number;
  /** Horizontal field of view, degrees (the held lens outranks it when there is one, spatial/spots.py). */
  fov?: number;
}

const HEAD = /^x-camera[ \t]+(\S+)/;
const OPTIONS = ['at', 'height', 'yaw', 'fov'] as const;

/** One line's spot, or null for a line that isn't a readable spot. */
function readLine(line: string): [string, Spot] | null {
  const id = line.match(HEAD)?.[1];
  if (!id) return null;
  const words = line.trim().split(/\s+/).slice(2);
  const o: Record<string, string> = {};
  for (let i = 0; i < words.length; i += 2) {
    const k = words[i] as string;
    const v = words[i + 1];
    if (!(OPTIONS as readonly string[]).includes(k) || v === undefined) return null;
    o[k] = v;
  }
  const at = (o.at ?? '').split(',').map(Number);
  if (at.length !== 2 || !at.every(Number.isFinite)) return null;
  const spot: Spot = { at: at as Pt };
  for (const k of ['height', 'yaw', 'fov'] as const) {
    if (o[k] === undefined) continue;
    const n = Number(o[k]);
    if (!Number.isFinite(n)) return null;
    spot[k] = n;
  }
  return [id, spot];
}

/** Every camera's spot in a plan's text, in the order their lines come. */
export function spotsOf(text: string): Record<string, Spot> {
  const out: Record<string, Spot> = {};
  for (const line of text.split('\n')) {
    const found = readLine(line);
    if (found) out[found[0]] = found[1];
  }
  return out;
}

const round = (v: number, step: number) => Math.round(Math.round(v / step) * step * 1000) / 1000;

/** A facing as written: tenths of a degree, from 0 up to 360. */
export const facing = (yaw: number) => {
  const v = round(((yaw % 360) + 360) % 360, 0.1);
  return v >= 360 ? 0 : v;
};

/** A spot's line, rounded as the editor writes it: centimetres and tenths of a degree. */
export function spotLine(camera: string, s: Spot): string {
  const opt = (word: string, v: number | undefined) => (v === undefined ? '' : ` ${word} ${v}`);
  return (
    `x-camera ${camera} at ${round(s.at[0], 0.01)},${round(s.at[1], 0.01)}` +
    `${opt('height', s.height === undefined ? undefined : round(s.height, 0.01))}` +
    `${opt('yaw', s.yaw === undefined ? undefined : facing(s.yaw))}${opt('fov', s.fov === undefined ? undefined : round(s.fov, 0.1))}`
  );
}

const same = (a: Spot, b: Spot) => spotLine('', a) === spotLine('', b); // equal as they'd be written, so a nudge that rounds away changes nothing

/** The text with `camera`'s spot set to `spot`: its line rewritten in place, or a new line at the end. */
export function setSpot(text: string, camera: string, spot: Spot): string {
  const lines = text.split('\n');
  let at = -1;
  lines.forEach((l, i) => {
    if (l.match(HEAD)?.[1] === camera) at = i;
  });
  if (at >= 0) {
    const was = readLine(lines[at] as string);
    if (was && same(was[1], spot)) return text;
    lines[at] = spotLine(camera, spot);
    return lines.join('\n');
  }
  return `${text.replace(/\n+$/, '')}\n${spotLine(camera, spot)}\n`;
}

/** Which way a facing points on the plan (y grows down the page). */
export const heading = (yaw: number): Pt => [Math.sin((yaw * Math.PI) / 180), -Math.cos((yaw * Math.PI) / 180)];

/** The facing from `from` toward `to`, degrees clockwise from up the page. */
export const bearing = (from: Pt, to: Pt) => facing((Math.atan2(to[0] - from[0], -(to[1] - from[1])) * 180) / Math.PI);
