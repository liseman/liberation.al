// Spatial memory's extensions to the floorplan plan format, kept beside the vendored library
// (src/vendor/floorplan) rather than in it, so they can go upstream later. In the JSON form each is an `x-` key at the
// top of the document: the upstream parser ignores `x-` keys by design, so an extended plan is still a valid upstream
// plan. The line DSL spells them as `x-` statements (dsl.ts).
//
//   x-anchor      where the plan sits in the place's point-cloud frame (anchor.ts)
//   x-revision    this revision: its number, when and by which model it was made, and the visits that fed it
//   x-history     earlier revisions, oldest first, the same shape
//   x-provenance  per entity ("room:<id>", "outdoor:<id>", "opening:<id>", "wall:<id>"): machine-derived or owner-
//                 corrected, how sure, whether only partly seen, and a wall's measured thickness
//   x-unexplored  areas not seen yet, as rect or poly
//   x-cameras     fixed cameras, in plan coordinates
//   x-objects     things on the plan (object memory), as boxes
//
// Wall ids are the library's derived ones (w1, w2, ...): stable for one revision's geometry, which is all a machine
// plan needs, since it is regenerated rather than edited.
import type { Finding } from '../../../vendor/floorplan/index.js';

export type Source = 'machine' | 'owner';
export type Pt = [number, number];

export interface Provenance {
  source: Source;
  /** How sure the model was, 0 to 1. Owner corrections leave it out. */
  confidence?: number;
  /** Only part of it has been seen. */
  partial?: boolean;
  /** A wall's measured thickness in metres. The library draws every wall at its class's thickness. */
  thickness?: number;
}

export interface Anchor {
  /** The point cloud's frame: the place map it was registered against. */
  frame: string;
  /** Degrees, counterclockwise, from the plan's north (up the page) to the frame's +Y. */
  yaw: number;
  /** Where the plan's origin sits in the frame, in metres; the third value is the floor's height. */
  origin: [number, number, number];
  /** Frame metres per plan metre (default 1). */
  scale?: number;
}

export interface Revision {
  n: number;
  /** ISO time it was made. */
  at?: string;
  /** What made it: a floorplan model or "owner". */
  model?: string;
  /** The visits whose data fed it. */
  visits: string[];
}

export interface Area {
  name?: string;
  rect?: [number, number, number, number];
  poly?: Pt[];
}

export interface Camera {
  at: Pt;
  /** Metres above the floor. */
  height?: number;
  /** Degrees clockwise from north, which way it looks. */
  yaw?: number;
  /** Horizontal field of view, degrees. */
  fov?: number;
}

export interface PlanObject {
  label: string;
  at: Pt;
  /** Width and depth in metres. */
  size?: Pt;
  /** Degrees clockwise. */
  yaw?: number;
  source?: Source;
  confidence?: number;
}

export interface Extensions {
  anchor?: Anchor;
  revision?: Revision;
  history: Revision[];
  provenance: Record<string, Provenance>;
  unexplored: Record<string, Area>;
  cameras: Record<string, Camera>;
  objects: Record<string, PlanObject>;
}

/** The document keys this layer owns. */
export const KEYS = {
  anchor: 'x-anchor',
  revision: 'x-revision',
  history: 'x-history',
  provenance: 'x-provenance',
  unexplored: 'x-unexplored',
  cameras: 'x-cameras',
  objects: 'x-objects',
} as const;

const OWN = new Set<string>(Object.values(KEYS));

/** The document without this layer's keys: what the upstream library (and its DSL printer) is given. */
export function baseOf(doc: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(doc).filter(([k]) => !OWN.has(k)));
}

/** A document with ext written into it (empty parts left out), on top of base. */
export function withExtensions(base: Record<string, unknown>, ext: Extensions): Record<string, unknown> {
  const out: Record<string, unknown> = { ...baseOf(base) };
  if (ext.anchor) out[KEYS.anchor] = ext.anchor;
  if (ext.revision) out[KEYS.revision] = ext.revision;
  if (ext.history.length) out[KEYS.history] = ext.history;
  if (Object.keys(ext.provenance).length) out[KEYS.provenance] = ext.provenance;
  if (Object.keys(ext.unexplored).length) out[KEYS.unexplored] = ext.unexplored;
  if (Object.keys(ext.cameras).length) out[KEYS.cameras] = ext.cameras;
  if (Object.keys(ext.objects).length) out[KEYS.objects] = ext.objects;
  return out;
}

export const emptyExtensions = (): Extensions => ({
  history: [],
  provenance: {},
  unexplored: {},
  cameras: {},
  objects: {},
});

/** The entity kinds a provenance key may name. */
export const ENTITY_KINDS = ['room', 'outdoor', 'opening', 'wall'] as const;

/** Records a problem; returns undefined, so a reader can `return issue(...)` for what it couldn't read. */
type Issue = (message: string, path: string) => undefined;

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isPt = (v: unknown): v is Pt => Array.isArray(v) && v.length === 2 && v.every(isNum);

/** Keys of o that aren't in allowed, each reported. */
function onlyKeys(o: Record<string, unknown>, allowed: readonly string[], path: string, issue: Issue) {
  for (const k of Object.keys(o)) if (!allowed.includes(k)) issue(`unknown field "${k}"`, `${path}.${k}`);
}

function readSource(o: Record<string, unknown>, path: string, issue: Issue): Pick<Provenance, 'source' | 'confidence'> {
  const source = o.source === 'owner' ? 'owner' : 'machine';
  if (o.source !== undefined && o.source !== 'machine' && o.source !== 'owner')
    issue('source is "machine" or "owner"', `${path}.source`);
  const out: Pick<Provenance, 'source' | 'confidence'> = { source };
  if (o.confidence !== undefined) {
    if (isNum(o.confidence) && o.confidence >= 0 && o.confidence <= 1) out.confidence = o.confidence;
    else issue('confidence is a number from 0 to 1', `${path}.confidence`);
  }
  return out;
}

function readRevision(v: unknown, path: string, issue: Issue): Revision | undefined {
  if (!isObj(v)) return issue('a revision is an object', path);
  onlyKeys(v, ['n', 'at', 'model', 'visits'], path, issue);
  if (!Number.isInteger(v.n) || (v.n as number) < 1) return issue('n is a whole number from 1', `${path}.n`);
  const visits = Array.isArray(v.visits) && v.visits.every(x => typeof x === 'string') ? (v.visits as string[]) : [];
  if (v.visits !== undefined && visits.length !== (v.visits as unknown[]).length)
    issue('visits is a list of visit ids', `${path}.visits`);
  const out: Revision = { n: v.n as number, visits };
  if (typeof v.at === 'string') out.at = v.at;
  if (typeof v.model === 'string') out.model = v.model;
  return out;
}

function readArea(v: unknown, path: string, issue: Issue): Area | undefined {
  if (!isObj(v)) return issue('an area is an object', path);
  onlyKeys(v, ['name', 'rect', 'poly'], path, issue);
  const out: Area = {};
  if (typeof v.name === 'string') out.name = v.name;
  if (v.rect !== undefined) {
    const r = v.rect;
    if (Array.isArray(r) && r.length === 4 && r.every(isNum) && (r[2] as number) > 0 && (r[3] as number) > 0)
      out.rect = r as Area['rect'];
    else return issue('rect is [x, y, width, height] with a positive size', `${path}.rect`);
  } else if (Array.isArray(v.poly) && v.poly.length >= 3 && v.poly.every(isPt)) {
    if (Math.abs(ringArea(v.poly as Pt[])) === 0) return issue('the poly has no area', `${path}.poly`);
    out.poly = v.poly as Pt[];
  } else return issue('an area has a rect, or a poly of 3 or more [x, y] points', path);
  return out;
}

/** The shoelace area of a ring, signed. */
export function ringArea(ring: Pt[]): number {
  let a = 0;
  for (let i = 0; i < ring.length; i++) {
    const [x0, y0] = ring[i] as Pt;
    const [x1, y1] = ring[(i + 1) % ring.length] as Pt;
    a += x0 * y1 - x1 * y0;
  }
  return a / 2;
}

/** An area's outline: a rect's four corners, or its poly. */
export function areaRing(a: Area): Pt[] {
  if (a.rect) {
    const [x, y, w, h] = a.rect;
    return [
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h],
    ];
  }
  return a.poly ?? [];
}

function each<T>(
  v: unknown,
  path: string,
  issue: Issue,
  read: (x: unknown, p: string) => T | undefined,
): Record<string, T> {
  const out: Record<string, T> = {};
  if (v === undefined) return out;
  if (!isObj(v)) {
    issue('a map of id to entry', path);
    return out;
  }
  for (const [id, x] of Object.entries(v)) {
    const r = read(x, `${path}["${id}"]`);
    if (r !== undefined) out[id] = r;
  }
  return out;
}

/** The extensions a document carries, and what's wrong with how they're written (rule "x.schema", errors). What
 *  can't be read is left out, so the rest still draws. */
export function readExtensions(doc: unknown): { ext: Extensions; issues: Finding[] } {
  const issues: Finding[] = [];
  const issue: Issue = (message, path) => {
    issues.push({ rule: 'x.schema', severity: 'error', message, path });
    return undefined;
  };
  const ext = emptyExtensions();
  if (!isObj(doc)) return { ext, issues };

  const a = doc[KEYS.anchor];
  if (a !== undefined) {
    if (!isObj(a)) issue('the anchor is an object', KEYS.anchor);
    else {
      onlyKeys(a, ['frame', 'yaw', 'origin', 'scale'], KEYS.anchor, issue);
      const origin = a.origin;
      if (typeof a.frame !== 'string' || !a.frame) issue('frame names the point cloud frame', `${KEYS.anchor}.frame`);
      else if (!isNum(a.yaw)) issue('yaw is a number of degrees', `${KEYS.anchor}.yaw`);
      else if (!(Array.isArray(origin) && origin.length === 3 && origin.every(isNum)))
        issue('origin is [x, y, z] in metres', `${KEYS.anchor}.origin`);
      else if (a.scale !== undefined && !(isNum(a.scale) && a.scale > 0))
        issue('scale is a positive number', `${KEYS.anchor}.scale`);
      else {
        ext.anchor = { frame: a.frame, yaw: a.yaw, origin: origin as Anchor['origin'] };
        if (a.scale !== undefined) ext.anchor.scale = a.scale as number;
      }
    }
  }

  if (doc[KEYS.revision] !== undefined) ext.revision = readRevision(doc[KEYS.revision], KEYS.revision, issue);
  const h = doc[KEYS.history];
  if (h !== undefined) {
    if (!Array.isArray(h)) issue('history is a list of revisions', KEYS.history);
    else
      h.forEach((r, i) => {
        const read = readRevision(r, `${KEYS.history}[${i}]`, issue);
        if (read) ext.history.push(read);
      });
  }

  ext.provenance = each(doc[KEYS.provenance], KEYS.provenance, issue, (v, path) => {
    const key = path.slice(KEYS.provenance.length + 2, -2);
    const kind = key.split(':')[0];
    if (!(ENTITY_KINDS as readonly string[]).includes(kind ?? '') || !key.includes(':'))
      return issue(`a provenance key is <${ENTITY_KINDS.join('|')}>:<id>`, path);
    if (!isObj(v)) return issue('a provenance entry is an object', path);
    onlyKeys(v, ['source', 'confidence', 'partial', 'thickness'], path, issue);
    const out: Provenance = readSource(v, path, issue);
    if (v.partial === true) out.partial = true;
    if (v.thickness !== undefined) {
      if (kind === 'wall' && isNum(v.thickness) && v.thickness > 0) out.thickness = v.thickness;
      else issue('thickness is a positive number of metres, on a wall', `${path}.thickness`);
    }
    return out;
  });

  ext.unexplored = each(doc[KEYS.unexplored], KEYS.unexplored, issue, (v, p) => readArea(v, p, issue));

  ext.cameras = each(doc[KEYS.cameras], KEYS.cameras, issue, (v, path) => {
    if (!isObj(v)) return issue('a camera is an object', path);
    onlyKeys(v, ['at', 'height', 'yaw', 'fov'], path, issue);
    if (!isPt(v.at)) return issue('at is [x, y]', `${path}.at`);
    const out: Camera = { at: v.at };
    for (const k of ['height', 'yaw', 'fov'] as const) {
      if (v[k] === undefined) continue;
      if (isNum(v[k]) && (k !== 'fov' || (v[k] > 0 && v[k] < 360))) out[k] = v[k];
      else issue(k === 'fov' ? 'fov is between 0 and 360 degrees' : `${k} is a number`, `${path}.${k}`);
    }
    return out;
  });

  ext.objects = each(doc[KEYS.objects], KEYS.objects, issue, (v, path) => {
    if (!isObj(v)) return issue('an object is an object', path);
    onlyKeys(v, ['label', 'at', 'size', 'yaw', 'source', 'confidence'], path, issue);
    if (typeof v.label !== 'string' || !v.label) return issue('label says what it is', `${path}.label`);
    if (!isPt(v.at)) return issue('at is [x, y]', `${path}.at`);
    const out: PlanObject = { label: v.label, at: v.at };
    if (v.size !== undefined) {
      if (isPt(v.size) && v.size[0] > 0 && v.size[1] > 0) out.size = v.size;
      else issue('size is [width, depth], both positive', `${path}.size`);
    }
    if (v.yaw !== undefined) {
      if (isNum(v.yaw)) out.yaw = v.yaw;
      else issue('yaw is a number', `${path}.yaw`);
    }
    if (v.source !== undefined || v.confidence !== undefined) Object.assign(out, readSource(v, path, issue));
    return out;
  });

  return { ext, issues };
}
