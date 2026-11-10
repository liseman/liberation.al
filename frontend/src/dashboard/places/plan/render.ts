// Drawing a place's plan: the vendored library draws the plan, and this adds one overlay group (class
// "x-layer") on top for the extensions, placed through the library's own projection, so nothing in the vendored files
// changes. What the overlay shows:
//
//   owner-corrected walls, openings, rooms    an accent line along them ("You corrected this")
//   machine-derived ones under UNSURE_BELOW    a dashed amber line ("Machine-derived, 45% sure")
//   unexplored areas                          hatching, labelled "Not seen yet"
//   partly seen rooms                         a dashed outline
//   cameras                                   a dot, and its field of view as a wedge
//   objects                                   a box (or a dot without a size) with its label
//
// Machine-derived entities the model was sure of look as the library draws them: the plan's normal look is the
// machine's, and the overlay marks what's different about the rest.
import {
  type FloorplanResult,
  floorplan,
  type Model,
  type Projection,
  projection,
  type RenderOptions,
} from '../../../vendor/floorplan/index.js';
import { areaRing, type Extensions, type Provenance, type Pt, readExtensions } from './ext';

/** Below this confidence a machine-derived entity is drawn as unsure. */
export const UNSURE_BELOW = 0.6;

/** How long a camera's field-of-view wedge is drawn, in metres. */
const FOV_REACH_M = 1.5;

let drawings = 0;

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const f = (n: number) => String(Math.round(n * 100) / 100);

/** How an entity's provenance is drawn: "owner", "unsure", or nothing (the plan's normal look). */
export function looks(p: Provenance | undefined): 'owner' | 'unsure' | undefined {
  if (!p) return undefined;
  if (p.source === 'owner') return 'owner';
  return p.confidence !== undefined && p.confidence < UNSURE_BELOW ? 'unsure' : undefined;
}

/** What an entity's tooltip says about where it came from. */
export function provenanceText(p: Provenance): string {
  const parts = [
    p.source === 'owner'
      ? 'You corrected this'
      : `Machine-derived${p.confidence === undefined ? '' : `, ${Math.round(p.confidence * 100)}% sure`}`,
  ];
  if (p.partial) parts.push('only partly seen');
  if (p.thickness !== undefined) parts.push(`measured ${p.thickness} m thick`);
  return parts.join(', ');
}

const STYLE = `
.x-layer{--x-unsure:#b45309}
svg.floorplan.theme-dark .x-layer{--x-unsure:#f59e0b}
.x-layer .x-line{fill:none;stroke-width:3;stroke-linecap:round}
.x-layer .x-owner{stroke:var(--accent)}
.x-layer .x-unsure{stroke:var(--x-unsure);stroke-dasharray:5 4}
.x-layer .x-ring{fill:none;stroke-width:1.5}
.x-layer .x-seen{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 3}
.x-layer .x-label{fill:var(--muted);font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.x-layer .x-cam{fill:var(--ink)}
.x-layer .x-fov{fill:var(--accent);fill-opacity:.12;stroke:var(--accent);stroke-opacity:.5;stroke-width:1}
.x-layer .x-obj{fill:none;stroke:var(--ink);stroke-width:1.2}
.x-layer .x-obj.x-unsure{stroke:var(--x-unsure);stroke-dasharray:3 3}
.x-layer .x-objtext{fill:var(--ink);font-size:10px}
`;

function ringPath(proj: Projection, ring: Pt[]): string {
  return `${ring
    .map((p, i) => {
      const [x, y] = proj.toScreen(p);
      return `${i ? 'L' : 'M'}${f(x)} ${f(y)}`;
    })
    .join(' ')} Z`;
}

function line(proj: Projection, a: Pt, b: Pt, cls: string, title: string): string {
  const [x1, y1] = proj.toScreen(a);
  const [x2, y2] = proj.toScreen(b);
  return `<path class="x-line ${cls}" d="M${f(x1)} ${f(y1)} L${f(x2)} ${f(y2)}"><title>${esc(title)}</title></path>`;
}

/** The point `d` metres from `from` toward `to`. */
function toward(from: Pt, to: Pt, d: number): Pt {
  const len = Math.hypot(to[0] - from[0], to[1] - from[1]) || 1;
  return [from[0] + ((to[0] - from[0]) * d) / len, from[1] + ((to[1] - from[1]) * d) / len];
}

/** The centroid of a ring's corners: where its label goes. */
const middle = (ring: Pt[]): Pt => [
  ring.reduce((s, p) => s + p[0], 0) / ring.length,
  ring.reduce((s, p) => s + p[1], 0) / ring.length,
];

/** Which way yaw (degrees clockwise from north) points, on the plan (y grows south). */
const heading = (yaw: number): Pt => [Math.sin((yaw * Math.PI) / 180), -Math.cos((yaw * Math.PI) / 180)];

function overlay(ext: Extensions, model: Model, proj: Projection, hatch: string): string {
  let body = '';
  const hatched = (ring: Pt[], title: string, label?: string) => {
    body += `<g><title>${esc(title)}</title><path d="${ringPath(proj, ring)}" fill="url(#${hatch})"/>`;
    body += `<path class="x-seen" d="${ringPath(proj, ring)}" fill="none"/>`;
    // The label only where it fits across the area (about 7 px a letter at 10 px); the tooltip says it anyway.
    const xs = ring.map(p => proj.toScreen(p)[0]);
    if (label && Math.max(...xs) - Math.min(...xs) >= label.length * 7 + 8) {
      const [x, y] = proj.toScreen(middle(ring));
      body += `<text class="x-label" x="${f(x)}" y="${f(y + 3)}" text-anchor="middle">${esc(label)}</text>`;
    }
    body += '</g>';
  };

  for (const [id, area] of Object.entries(ext.unexplored))
    hatched(areaRing(area), `${area.name ?? id}: not seen yet`, 'Not seen yet');

  for (const [key, p] of Object.entries(ext.provenance)) {
    const [kind, ...rest] = key.split(':');
    const id = rest.join(':');
    const look = looks(p);
    const title = `${kind} ${id}: ${provenanceText(p)}`;
    if (kind === 'room') {
      // Outdoor spaces' provenance is kept and linted but not drawn yet: the model has no rings for them here.
      const ring = model.rooms.find(r => r.room.id === id)?.room.poly as Pt[] | undefined;
      if (!ring) continue;
      if (p.partial)
        body += `<path class="x-seen" d="${ringPath(proj, ring)}" fill="none"><title>${esc(title)}</title></path>`;
      if (look)
        body += `<path class="x-ring x-${look}" d="${ringPath(proj, ring)}"><title>${esc(title)}</title></path>`;
    } else if (kind === 'wall' && look) {
      const wall = model.walls.find(w => w.id === id);
      if (wall?.geometry.kind === 'segment') body += line(proj, wall.start, wall.end, `x-${look}`, title);
    } else if (kind === 'opening' && look) {
      // The opening's span: its centre, half its width each way along the wall (the model's own from/to are positions
      // on the wall's axis, not distances from its start).
      const o = model.openings.find(x => x.spec.id === id);
      if (o?.wall.geometry.kind === 'segment') {
        const half = o.spec.width / 2;
        body += line(
          proj,
          toward(o.center, o.wall.start, half),
          toward(o.center, o.wall.end, half),
          `x-${look}`,
          title,
        );
      }
    }
  }

  for (const [id, cam] of Object.entries(ext.cameras)) {
    const [cx, cy] = proj.toScreen(cam.at);
    body += `<g><title>${esc(`Camera ${id}`)}</title>`;
    if (cam.yaw !== undefined && cam.fov !== undefined) {
      const r = FOV_REACH_M * proj.scale;
      const [ax, ay] = heading(cam.yaw - cam.fov / 2);
      const [bx, by] = heading(cam.yaw + cam.fov / 2);
      const large = cam.fov > 180 ? 1 : 0;
      body += `<path class="x-fov" d="M${f(cx)} ${f(cy)} L${f(cx + ax * r)} ${f(cy + ay * r)} A${f(r)} ${f(r)} 0 ${large} 1 ${f(cx + bx * r)} ${f(cy + by * r)} Z"/>`;
    }
    body += `<circle class="x-cam" cx="${f(cx)}" cy="${f(cy)}" r="4"/></g>`;
  }

  for (const [id, o] of Object.entries(ext.objects)) {
    const [x, y] = proj.toScreen(o.at);
    const unsure = looks(o.source ? { source: o.source, confidence: o.confidence } : undefined) === 'unsure';
    const title = o.source ? `${o.label}: ${provenanceText({ source: o.source, confidence: o.confidence })}` : o.label;
    body += `<g data-x-object="${esc(id)}"><title>${esc(title)}</title>`;
    if (o.size) {
      const w = o.size[0] * proj.scale;
      const d = o.size[1] * proj.scale;
      body += `<rect class="x-obj${unsure ? ' x-unsure' : ''}" x="${f(x - w / 2)}" y="${f(y - d / 2)}" width="${f(w)}" height="${f(d)}"${o.yaw ? ` transform="rotate(${f(o.yaw)} ${f(x)} ${f(y)})"` : ''}/>`;
    } else body += `<circle class="x-obj${unsure ? ' x-unsure' : ''}" cx="${f(x)}" cy="${f(y)}" r="3"/>`;
    body += `<text class="x-objtext" x="${f(x)}" y="${f(y + 3)}" text-anchor="middle">${esc(o.label)}</text></g>`;
  }

  if (!body) return '';
  const defs = `<defs><pattern id="${hatch}" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" stroke="var(--muted)" stroke-width="1.5" stroke-opacity=".55"/></pattern></defs>`;
  return `<g class="x-layer"><style>${STYLE}</style>${defs}${body}</g>`;
}

export interface PlaceDrawing {
  /** The library's drawing with the extensions' overlay in it. */
  svg: string;
  result: FloorplanResult;
  ext: Extensions;
}

/** A place's plan, drawn: the library's SVG with the extensions on top. */
export function drawPlace(doc: Record<string, unknown>, render: RenderOptions = {}): PlaceDrawing {
  const result = floorplan(doc, { render, markFindings: 'none' });
  const { ext } = readExtensions(doc);
  const extra = overlay(ext, result.model, projection(result.model, render), `x-hatch-${++drawings}`);
  return { svg: extra ? result.svg.replace(/<\/svg>\s*$/, `${extra}</svg>`) : result.svg, result, ext };
}
