// Lint for a place's plan (#2797). The upstream linter is written for designing a house: a habitable room needs a
// window, a door must be wide enough, a bedroom shouldn't open off the living room. A plan of a place the house has
// seen is a record, not a design, so those findings say nothing useful about it. What matters is whether the plan
// is sound as geometry (rooms tile, walls and openings resolve) and whether its extensions are. lintPlace keeps the
// upstream findings that are about that (QA_RULES) and adds the extensions' own (rules "x.*").
import { type Finding, lint, type Model, sortFindings } from '../../../vendor/floorplan/index.js';
import { ENTITY_KINDS, type Extensions, readExtensions } from './ext';

/** Upstream rules that check a plan is sound geometry, by id or by prefix (ending in "."). Everything else upstream is
 *  a design judgement. */
export const QA_RULES = [
  'schema.',
  'tiling.',
  'wall.',
  'opening.off_wall',
  'opening.overflow',
  'opening.collision',
  'outdoor.overlap',
  'geometry.',
  'arc.',
  'fixture.outside_space',
  'fixture.overlap',
  'window.not_exterior',
  'stair.misaligned',
  'stair.no_arrival',
  'structure.over_open_sky',
] as const;

export const isQaRule = (rule: string): boolean =>
  QA_RULES.some(r => (r.endsWith('.') ? rule.startsWith(r) : rule === r));

/** A wall measured this far (metres) from its class's thickness is worth saying: the drawing uses the class's. */
export const THICKNESS_NOTE_M = 0.03;

/** The ids each kind of entity has in this plan, for checking what provenance names. */
function entityIds(doc: Record<string, unknown>, model: Model | undefined): Record<string, Set<string>> {
  const keys = (v: unknown) => new Set(typeof v === 'object' && v !== null ? Object.keys(v) : []);
  return {
    room: keys(doc.rooms),
    outdoor: keys(doc.outdoor),
    opening: new Set(model?.openings.map(o => o.spec.id) ?? []),
    wall: new Set(model?.walls.map(w => w.id) ?? []),
  };
}

function extensionFindings(ext: Extensions, doc: Record<string, unknown>, model: Model | undefined): Finding[] {
  const out: Finding[] = [];
  const add = (rule: string, severity: Finding['severity'], message: string, path: string) =>
    out.push({ rule, severity, message, path });
  if (!ext.anchor)
    add(
      'x.anchor.missing',
      'info',
      "the plan has no anchor, so nothing measured in the place's point cloud can be placed on it",
      'x-anchor',
    );
  if (ext.revision && ext.revision.visits.length === 0)
    add('x.revision.no_visits', 'warning', `revision ${ext.revision.n} names no visits that fed it`, 'x-revision');
  const ns = [...ext.history.map(r => r.n), ...(ext.revision ? [ext.revision.n] : [])];
  ns.forEach((n, i) => {
    if (i > 0 && n <= (ns[i - 1] as number))
      add(
        'x.history.order',
        'error',
        `revision ${n} comes after revision ${ns[i - 1]}; revisions count up, oldest first`,
        i < ext.history.length ? `x-history[${i}]` : 'x-revision',
      );
  });
  if (model) {
    const ids = entityIds(doc, model);
    for (const [key, p] of Object.entries(ext.provenance)) {
      const [kind, ...rest] = key.split(':');
      const id = rest.join(':');
      const known = (ENTITY_KINDS as readonly string[]).includes(kind ?? '') && ids[kind as string]?.has(id);
      if (!known)
        add('x.reference', 'warning', `provenance names ${key}, which isn't in this plan`, `x-provenance["${key}"]`);
      if (kind === 'wall' && p.thickness !== undefined) {
        const wall = model.walls.find(w => w.id === id);
        if (wall && Math.abs(wall.thickness - p.thickness) > THICKNESS_NOTE_M)
          add(
            'x.wall.thickness',
            'info',
            `wall ${id} measured ${p.thickness} m thick; it's drawn at its class's ${wall.thickness} m`,
            `x-provenance["${key}"].thickness`,
          );
      }
    }
  }
  return out;
}

/** A place plan's findings: upstream geometry QA, the extensions' schema problems, and the extensions' own checks,
 *  most severe first. `all` keeps every upstream finding (the design rules too). */
export function lintPlace(doc: Record<string, unknown>, { all = false }: { all?: boolean } = {}): Finding[] {
  const upstream = lint(doc);
  const { ext, issues } = readExtensions(doc);
  const kept = all ? upstream.findings : upstream.findings.filter(f => isQaRule(f.rule));
  return sortFindings([...kept, ...issues, ...extensionFindings(ext, doc, upstream.model)]);
}
