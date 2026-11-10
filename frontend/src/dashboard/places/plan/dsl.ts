// The line DSL for spatial memory's extensions: one `x-` statement per line, beside the upstream statements.
// parsePlaceDsl blanks the `x-` lines before handing the text to the upstream parser (so its line numbers still
// point at the right lines) and reads them itself; toPlaceDsl prints the upstream part with the library's printer and
// appends the `x-` lines in a fixed order, so printing what was parsed gives the same text back.
//
//   x-anchor <frame> yaw <deg> origin <x>,<y>,<z> [scale <s>]
//   x-revision <n> [at <iso time>] [model <name>] [visits <id>,<id>,...]
//   x-history <n> [at <iso time>] [model <name>] [visits <id>,...]       one per earlier revision, oldest first
//   x-source <room|outdoor|opening|wall>:<id> <machine|owner> [<confidence>] [partial] [thickness <m>]
//   x-unexplored <id> ["<name>"] rect <x>,<y> <w>x<h>
//   x-unexplored <id> ["<name>"] poly <x>,<y> <x>,<y> <x>,<y> ...
//   x-camera <id> at <x>,<y> [height <m>] [yaw <deg>] [fov <deg>]
//   x-object <id> "<label>" at <x>,<y> [size <w>x<d>] [yaw <deg>] [machine|owner] [<confidence>]
//
// Names and labels are JSON strings ("Sofa \"big\"").
import { parseDsl, toDsl } from '../../../vendor/floorplan/index.js';
import {
  type Area,
  baseOf,
  type Camera,
  type Extensions,
  emptyExtensions,
  type PlanObject,
  type Provenance,
  type Pt,
  type Revision,
  readExtensions,
  withExtensions,
} from './ext';

export class PlaceDslError extends Error {
  readonly line: number;
  constructor(line: number, message: string) {
    super(`line ${line}: ${message}`);
    this.line = line;
  }
}

const TOKEN = /"(?:[^"\\]|\\.)*"|\S+/g;

function num(t: string | undefined, line: number, what: string): number {
  const n = Number(t);
  if (t === undefined || t === '' || !Number.isFinite(n)) throw new PlaceDslError(line, `${what} is a number`);
  return n;
}

function pair(t: string | undefined, sep: ',' | 'x', line: number, what: string): Pt {
  const parts = (t ?? '').split(sep);
  if (parts.length !== 2) throw new PlaceDslError(line, `${what} is written a${sep}b`);
  return [num(parts[0], line, what), num(parts[1], line, what)];
}

function str(t: string | undefined, line: number, what: string): string {
  if (!t?.startsWith('"')) throw new PlaceDslError(line, `${what} is a quoted string`);
  return JSON.parse(t) as string;
}

/** Options after a statement's fixed part: `key value` pairs, bare flags, a bare source word, and a bare number. */
function options(rest: string[], line: number, valued: readonly string[], flags: readonly string[] = []) {
  const out: Record<string, string> = {};
  const set = new Set<string>();
  let source: 'machine' | 'owner' | undefined;
  let bare: number | undefined;
  for (let i = 0; i < rest.length; i++) {
    const t = rest[i] as string;
    if (valued.includes(t)) {
      if (rest[i + 1] === undefined) throw new PlaceDslError(line, `${t} needs a value`);
      out[t] = rest[++i] as string;
    } else if (flags.includes(t)) set.add(t);
    else if (t === 'machine' || t === 'owner') source = t;
    else if (bare === undefined && Number.isFinite(Number(t))) bare = Number(t);
    else throw new PlaceDslError(line, `unexpected "${t}"`);
  }
  return { out, set, source, bare };
}

function revision(rest: string[], line: number): Revision {
  const n = num(rest[0], line, 'the revision number');
  const o = options(rest.slice(1), line, ['at', 'model', 'visits']);
  const r: Revision = { n, visits: o.out.visits ? o.out.visits.split(',').filter(Boolean) : [] };
  if (o.out.at) r.at = o.out.at;
  if (o.out.model) r.model = o.out.model;
  return r;
}

function statement(ext: Extensions, words: string[], line: number) {
  const [head, ...rest] = words;
  switch (head) {
    case 'x-anchor': {
      const frame = rest[0];
      const o = options(rest.slice(1), line, ['yaw', 'origin', 'scale']);
      const origin = (o.out.origin ?? '').split(',').map(v => num(v, line, 'origin'));
      if (!frame || origin.length !== 3) throw new PlaceDslError(line, 'x-anchor <frame> yaw <deg> origin x,y,z');
      ext.anchor = { frame, yaw: num(o.out.yaw, line, 'yaw'), origin: origin as [number, number, number] };
      if (o.out.scale !== undefined) ext.anchor.scale = num(o.out.scale, line, 'scale');
      return;
    }
    case 'x-revision':
      ext.revision = revision(rest, line);
      return;
    case 'x-history':
      ext.history.push(revision(rest, line));
      return;
    case 'x-source': {
      const key = rest[0] ?? '';
      const o = options(rest.slice(1), line, ['thickness'], ['partial']);
      if (!key.includes(':') || !o.source) throw new PlaceDslError(line, 'x-source <kind>:<id> <machine|owner> ...');
      const p: Provenance = { source: o.source };
      if (o.bare !== undefined) p.confidence = o.bare;
      if (o.set.has('partial')) p.partial = true;
      if (o.out.thickness !== undefined) p.thickness = num(o.out.thickness, line, 'thickness');
      ext.provenance[key] = p;
      return;
    }
    case 'x-unexplored': {
      const [id, ...more] = rest;
      if (!id) throw new PlaceDslError(line, 'x-unexplored <id> rect|poly ...');
      const area: Area = {};
      if (more[0]?.startsWith('"')) area.name = str(more.shift(), line, 'the name');
      const shape = more.shift();
      if (shape === 'rect') {
        const [x, y] = pair(more[0], ',', line, 'the corner');
        const [w, h] = pair(more[1], 'x', line, 'the size');
        if (more.length !== 2) throw new PlaceDslError(line, 'rect <x>,<y> <w>x<h>');
        area.rect = [x, y, w, h];
      } else if (shape === 'poly') area.poly = more.map(t => pair(t, ',', line, 'a point'));
      else throw new PlaceDslError(line, 'an unexplored area is a rect or a poly');
      ext.unexplored[id] = area;
      return;
    }
    case 'x-camera': {
      const id = rest[0];
      const o = options(rest.slice(1), line, ['at', 'height', 'yaw', 'fov']);
      if (!id || !o.out.at) throw new PlaceDslError(line, 'x-camera <id> at <x>,<y> ...');
      const cam: Camera = { at: pair(o.out.at, ',', line, 'at') };
      for (const k of ['height', 'yaw', 'fov'] as const) if (o.out[k] !== undefined) cam[k] = num(o.out[k], line, k);
      ext.cameras[id] = cam;
      return;
    }
    case 'x-object': {
      const id = rest[0];
      const label = str(rest[1], line, 'the label');
      const o = options(rest.slice(2), line, ['at', 'size', 'yaw']);
      if (!id || !o.out.at) throw new PlaceDslError(line, 'x-object <id> "<label>" at <x>,<y> ...');
      const obj: PlanObject = { label, at: pair(o.out.at, ',', line, 'at') };
      if (o.out.size) obj.size = pair(o.out.size, 'x', line, 'size');
      if (o.out.yaw !== undefined) obj.yaw = num(o.out.yaw, line, 'yaw');
      if (o.source) obj.source = o.source;
      if (o.bare !== undefined) obj.confidence = o.bare;
      ext.objects[id] = obj;
      return;
    }
    default:
      throw new PlaceDslError(line, `unknown statement "${head}"`);
  }
}

/** A plan in the line DSL, extensions included, as its JSON document. Throws the upstream DslError for the upstream
 *  lines, and PlaceDslError for an `x-` line. */
export function parsePlaceDsl(text: string): Record<string, unknown> {
  const lines = text.split('\n');
  const ours: [number, string][] = [];
  const upstream = lines.map((l, i) => {
    if (!l.trimStart().startsWith('x-')) return l;
    ours.push([i + 1, l]);
    return '';
  });
  const { doc } = parseDsl(upstream.join('\n'));
  const ext = emptyExtensions();
  for (const [line, l] of ours) statement(ext, l.trim().match(TOKEN) ?? [], line);
  return withExtensions(doc, ext);
}

const n = (v: number) => String(v);
const xy = ([x, y]: Pt) => `${n(x)},${n(y)}`;
const opt = (word: string, v: number | string | undefined) => (v === undefined ? '' : ` ${word} ${v}`);

function revisionLine(head: string, r: Revision): string {
  return `${head} ${r.n}${opt('at', r.at)}${opt('model', r.model)}${r.visits.length ? ` visits ${r.visits.join(',')}` : ''}`;
}

/** The `x-` lines for a document's extensions, in a fixed order. */
export function extensionLines(ext: Extensions): string[] {
  const out: string[] = [];
  const a = ext.anchor;
  if (a) out.push(`x-anchor ${a.frame} yaw ${n(a.yaw)} origin ${a.origin.map(n).join(',')}${opt('scale', a.scale)}`);
  for (const r of ext.history) out.push(revisionLine('x-history', r));
  if (ext.revision) out.push(revisionLine('x-revision', ext.revision));
  for (const [key, p] of Object.entries(ext.provenance))
    out.push(
      `x-source ${key} ${p.source}${p.confidence === undefined ? '' : ` ${n(p.confidence)}`}${p.partial ? ' partial' : ''}${opt('thickness', p.thickness)}`,
    );
  for (const [id, u] of Object.entries(ext.unexplored)) {
    const name = u.name === undefined ? '' : ` ${JSON.stringify(u.name)}`;
    const shape = u.rect
      ? `rect ${n(u.rect[0])},${n(u.rect[1])} ${n(u.rect[2])}x${n(u.rect[3])}`
      : `poly ${(u.poly ?? []).map(xy).join(' ')}`;
    out.push(`x-unexplored ${id}${name} ${shape}`);
  }
  for (const [id, c] of Object.entries(ext.cameras))
    out.push(`x-camera ${id} at ${xy(c.at)}${opt('height', c.height)}${opt('yaw', c.yaw)}${opt('fov', c.fov)}`);
  for (const [id, o] of Object.entries(ext.objects))
    out.push(
      `x-object ${id} ${JSON.stringify(o.label)} at ${xy(o.at)}${o.size ? ` size ${n(o.size[0])}x${n(o.size[1])}` : ''}${opt('yaw', o.yaw)}${o.source ? ` ${o.source}` : ''}${o.confidence === undefined ? '' : ` ${n(o.confidence)}`}`,
    );
  return out;
}

/** A JSON plan document, extensions included, in the line DSL. */
export function toPlaceDsl(doc: Record<string, unknown>): string {
  const lines = extensionLines(readExtensions(doc).ext);
  const base = toDsl(baseOf(doc)).replace(/\n+$/, '');
  return lines.length ? `${base}\n\n${lines.join('\n')}\n` : `${base}\n`;
}
