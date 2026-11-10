// floorplan — declarative floor-plan renderer and linter.
//
//   const plan = parse(json);                 // throws PlanError on schema problems
//   const { model, findings } = analyze(plan); // geometry + rules, never throws
//   const svg = renderSvg(model, { findings });
//
// or in one go: floorplan(json) → { plan, model, findings, svg, levels, schedule }
//
// and, for a caller that wants every problem a document can have through one channel and
// no exceptions at all: lint(json) → { findings, plan?, model? }, where a schema problem
// is a `schema.*` finding carrying the same document path a geometry finding carries.
import { derive, overlapArea, pointOn, shapeArea } from "./derive.js";
import { parse, PlanError } from "./parse.js";
import { DslError, lineOf, readSource } from "./dsl.js";
import { checkRules, sortFindings } from "./rules.js";
import { renderSvg } from "./svg.js";
export { parse, PlanError, derive, checkRules, renderSvg, sortFindings };
// the parser's own vocabularies, so documentation cannot drift from what it accepts
export { FIXTURE_TYPES, OPENING_TYPES, ROOM_KINDS, SIDES, SWEEPS, VERTICAL_TYPES } from "./parse.js";
// the parser's own id constraint, so `--schema`'s legend prints the regex the parser
// actually enforces rather than a hand-typed copy of it
export { ID_RE } from "./parse.js";
// the parser's own field table, so documentation cannot drift from what it accepts either —
// `floorplan --schema` prints this
export { SCHEMA } from "./parse.js";
export { RULES, ruleById } from "./catalogue.js";
// owner classes: rooms, outdoor spaces, the street and holes, plus the predicates that
// tell them apart — a consumer reading model.walls or model.access needs these
export { EXTERIOR, GAP, GROUND_LEVEL, inLevel, isOpenSky, isStreet, isVoid, outdoorOwner, ownerId, ownerKey, pathTo, roomOwner, sameOwner, voidOwner } from "./types.js";
export { applyDrag, applyHandle, applyMove, applyVertexHandle, draggableFixtureEdges, draggableOutdoorEdges, draggableWalls, movableFixtures, wallHandles, } from "./edit.js";
export { levelOf, projection } from "./svg.js";
// authoring support: keep a document canonical, and edit one value in place
export { formatPlan, formatText } from "./format.js";
// the line DSL: the authoring front-end, and its grammar as data
export { ARC_SYNTAX, COMPASS_LINE, DSL_SCHEMA, DslError, DslPosError, dslSchemaText, dslSpliceAll, dslSpliceAt, isDslText, lineOf, parseDsl, POLY_INLINE_EXAMPLE, readSource, toDsl } from "./dsl.js";
export { appendAt, insertKey, JsonPosError, metres, nodeAt, parseWithPositions, pathToString, removeAt, spliceAll, spliceAt, } from "./jsonpos.js";
/** Derive geometry and run every rule. Findings are sorted error → warning → info. */
export function analyze(plan, rules = {}) {
    const { model, findings } = derive(plan);
    return { model, findings: sortFindings([...findings, ...checkRules(model, rules)]) };
}
function levelSchedule(lm) {
    const rooms = lm.rooms.map((m) => ({
        id: m.room.id,
        name: m.room.name,
        kind: m.room.kind,
        zone: m.room.zone,
        area: m.area,
        clearArea: m.clearArea,
        fixtureArea: m.fixtureArea,
        usableArea: m.usableArea,
    }));
    return {
        rooms,
        interiorArea: lm.interiorArea,
        interiorClearArea: round(rooms.reduce((s, r) => s + r.clearArea, 0)),
        footprint: lm.envelope.area,
        waterArea: round(lm.fixtures.filter((f) => f.fixture.type === "pool").reduce((s, f) => s + f.area, 0)),
        outdoor: lm.level.outdoor.map((o) => {
            const area = round(shapeArea(o));
            // exactly the part of each fixture that stands on this deck, the same way a room
            // deducts its own: the arrangement of the two rings, read with an intersection
            // predicate (docs/gaps-design.md §1.3.2). A pool half off the deck deducts half.
            const fixtureArea = round(lm.fixtures.filter((f) => f.fixture.in === o.id).reduce((s, f) => s + overlapArea(f.fixture, o), 0));
            return {
                id: o.id,
                name: o.name,
                area,
                fixtureArea,
                usableArea: round(Math.max(0, area - fixtureArea)),
                covered: o.covered,
                streetConnected: lm.streetOutdoor.has(o.id),
            };
        }),
    };
}
export function schedule(model) {
    const ground = levelSchedule(model);
    if (!model.plan.levelled)
        return ground;
    const levels = model.levels.map((lm) => ({ id: lm.level.id, name: lm.level.name, ...levelSchedule(lm) }));
    return {
        ...ground,
        levels,
        building: {
            storeys: model.building.storeys,
            grossArea: model.building.grossArea,
            footprint: model.building.footprint,
            interiorArea: round(levels.reduce((s, l) => s + l.interiorArea, 0)),
            interiorClearArea: round(levels.reduce((s, l) => s + l.interiorClearArea, 0)),
            waterArea: round(levels.reduce((s, l) => s + l.waterArea, 0)),
        },
    };
}
/** Everything in one call. Throws PlanError only for schema problems. */
export function floorplan(input, opts = {}) {
    const src = source(input);
    const plan = parse(src.doc);
    const { model, findings: raw } = analyze(plan, opts.rules);
    const findings = withLines(raw, src.positions);
    const rank = { error: 0, warning: 1, info: 2 };
    const mark = opts.markFindings ?? "warning";
    const marked = mark === "none" ? [] : findings.filter((f) => rank[f.severity] <= rank[mark]);
    const levels = model.levels.map((lm) => {
        // A finding with no level is building-wide, so it belongs on every sheet.
        const mine = findings.filter((f) => f.level === undefined || f.level === lm.level.id);
        return {
            id: lm.level.id,
            name: lm.level.name,
            svg: renderSvg(model, { ...opts.render, level: lm.level.id, findings: marked.filter((f) => f.level === undefined || f.level === lm.level.id) }),
            findings: mine,
        };
    });
    const ground = levels.find((l) => l.id === model.level.id) ?? levels[0];
    return { plan, model, findings, svg: ground.svg, levels, schedule: schedule(model) };
}
/**
 * The derived walls, so an agent can see where a wall runs *before* placing an opening on
 * it — today it learns a wall's extent only by tripping `wall.ambiguous`
 * (docs/agent-review.md §B4).
 *
 * INVARIANT: endpoints, never `axis`/`c`. Those two fields describe an axis-aligned
 * segment and nothing else; two points describe any segment, so this shape survives the
 * planar-arrangement rewrite (docs/gaps-design.md §1.3.1) unchanged.
 */
export function walls(model, level) {
    const levels = level === undefined ? model.levels : model.levels.filter((lm) => lm.level.id === level);
    return levels.flatMap((lm) => lm.walls.map((w) => ({
        id: w.id,
        kind: w.kind,
        ...(model.plan.levelled ? { level: lm.level.id } : {}),
        from: pointOn(w, w.from),
        to: pointOn(w, w.to),
        neg: w.neg,
        pos: w.pos,
    })));
}
/** How many findings of each severity. The first thing an agent reads back. */
export function summarize(findings) {
    const s = { error: 0, warning: 0, info: 0 };
    for (const f of findings)
        s[f.severity]++;
    return s;
}
/**
 * A schema problem, as a finding. The `schema.<kind>` vocabulary is `PlanIssue["kind"]`
 * and is documented in the rule catalogue like every other rule, so `catalogue.test.ts`
 * guards it.
 */
const SCHEMA_RULE = {
    syntax: { rule: "schema.syntax" },
    unknown_field: { rule: "schema.unknown_field" },
    missing: { rule: "schema.missing" },
    type: { rule: "schema.type" },
    reference: { rule: "schema.reference" },
    geometry: { rule: "schema.geometry" },
    conflict: { rule: "schema.conflict" },
};
export const isSchemaFinding = (f) => f.rule.startsWith("schema.");
/**
 * Every problem with a document, through one channel and without throwing: schema
 * problems arrive as `schema.*` findings carrying the issue's own document path, geometry
 * and semantic problems as the rules' findings (docs/agent-review.md §B8).
 *
 * `parse()` and `floorplan()` keep throwing, for callers who want that.
 */
export function lint(input, opts = {}) {
    let plan;
    // Source text is sniffed once here rather than inside `parse()`, so the DSL positions
    // survive to annotate the findings; `parse()` would have thrown them away.
    let src;
    try {
        src = source(input);
    }
    catch (e) {
        if (!(e instanceof PlanError))
            throw e;
        return { findings: issueFindings(e), error: e };
    }
    try {
        plan = parse(src.doc);
    }
    catch (e) {
        if (!(e instanceof PlanError))
            throw e;
        return { findings: issueFindings(e, src.positions), error: e };
    }
    const { model, findings } = analyze(plan, opts.rules);
    return { findings: withLines(findings, src.positions), plan, model };
}
/**
 * Read source text as a document, in whichever syntax it is written. A DSL tokenizer
 * error arrives as a `PlanError` of `schema.syntax` issues carrying their own line, so a
 * caller has one exception type to handle whatever the input was written in.
 */
function source(input) {
    if (typeof input !== "string")
        return { doc: input };
    try {
        return readSource(input);
    }
    catch (e) {
        if (e instanceof DslError)
            throw new PlanError(e.issues.map((i) => ({ path: "", message: i.message, kind: "syntax", line: i.line })));
        throw new PlanError([{ path: "", message: `not valid JSON: ${e.message}`, kind: "syntax" }]);
    }
}
const issueFindings = (e, positions) => e.issues.map((i) => {
    const line = i.line ?? (positions ? lineOf(positions, i.path) : undefined);
    return { ...SCHEMA_RULE[i.kind], severity: "error", message: i.message, path: i.path, ...(line === undefined ? {} : { line }) };
});
/**
 * Attach `line` to every finding of a DSL document, by resolving its JSON path against
 * the positions the DSL parser recorded. A JSON document is returned untouched — the
 * array identity included — so nothing about its findings changes.
 */
function withLines(findings, positions) {
    if (!positions)
        return findings;
    return findings.map((f) => {
        const line = lineOf(positions, f.path);
        return line === undefined ? f : { ...f, line };
    });
}
export function worstSeverity(findings) {
    if (findings.some((f) => f.severity === "error"))
        return "error";
    if (findings.some((f) => f.severity === "warning"))
        return "warning";
    if (findings.length)
        return "info";
    return undefined;
}
const round = (n) => Math.round(n * 1000) / 1000;
