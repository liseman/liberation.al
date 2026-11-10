import { derive } from "./derive.ts";
import { parse, PlanError } from "./parse.ts";
import { checkRules, sortFindings } from "./rules.ts";
import { renderSvg } from "./svg.ts";
import type { Analysis, Finding, Model, Owner, Plan, Pt, Severity } from "./types.ts";
import type { RuleOptions } from "./rules.ts";
import type { RenderOptions } from "./svg.ts";
export { parse, PlanError, derive, checkRules, renderSvg, sortFindings };
export { FIXTURE_TYPES, OPENING_TYPES, ROOM_KINDS, SIDES, SWEEPS, VERTICAL_TYPES } from "./parse.ts";
export { ID_RE } from "./parse.ts";
export { SCHEMA } from "./parse.ts";
export type { FieldDoc, ObjectDoc } from "./parse.ts";
export { RULES, ruleById } from "./catalogue.ts";
export { EXTERIOR, GAP, GROUND_LEVEL, inLevel, isOpenSky, isStreet, isVoid, outdoorOwner, ownerId, ownerKey, pathTo, roomOwner, sameOwner, voidOwner } from "./types.ts";
export { applyDrag, applyHandle, applyMove, applyVertexHandle, draggableFixtureEdges, draggableOutdoorEdges, draggableWalls, movableFixtures, wallHandles, } from "./edit.ts";
export type { Draggable, Handle, Movable, OffsetHandle, RadiusHandle, VertexHandle } from "./edit.ts";
export { levelOf, projection } from "./svg.ts";
export type { Projection } from "./svg.ts";
export type { RuleDoc } from "./catalogue.ts";
export { formatPlan, formatText } from "./format.ts";
export { ARC_SYNTAX, COMPASS_LINE, DSL_SCHEMA, DslError, DslPosError, dslSchemaText, dslSpliceAll, dslSpliceAt, isDslText, lineOf, parseDsl, POLY_INLINE_EXAMPLE, readSource, toDsl } from "./dsl.ts";
export type { DslDocument, DslIssue, DslPositions, DslSchemaOptions, DslSpan, DslStatementDoc, DslTokenDoc } from "./dsl.ts";
export { appendAt, insertKey, JsonPosError, metres, nodeAt, parseWithPositions, pathToString, removeAt, spliceAll, spliceAt, } from "./jsonpos.ts";
export type { FormatOptions } from "./format.ts";
export type { JsonKind, JsonNode, JsonPath } from "./jsonpos.ts";
export type * from "./types.ts";
export type { IssueKind, PlanIssue } from "./parse.ts";
export type { RuleOptions } from "./rules.ts";
export type { RenderOptions } from "./svg.ts";
/** Derive geometry and run every rule. Findings are sorted error → warning → info. */
export declare function analyze(plan: Plan, rules?: RuleOptions): Analysis;
export interface ScheduleRow {
    id: string;
    name: string;
    kind: string;
    zone: string | undefined;
    /** centreline polygon area, m² */
    area: number;
    /** area net of half the bounding walls, m² */
    clearArea: number;
    /** floor taken by fixtures standing in the room, m² */
    fixtureArea: number;
    /** clearArea less fixtureArea: floor you can stand on, m² */
    usableArea: number;
}
export interface OutdoorRow {
    id: string;
    name: string;
    area: number;
    /** floor taken by fixtures standing in this outdoor space, m² */
    fixtureArea: number;
    /** area less fixtureArea: a deck net of its pool */
    usableArea: number;
    covered: boolean;
    /** can you walk here from the street? false for an enclosed courtyard */
    streetConnected: boolean;
}
/** One storey's numbers. The same shape a single-level schedule has always had. */
export interface LevelSchedule {
    rooms: ScheduleRow[];
    interiorArea: number;
    interiorClearArea: number;
    footprint: number;
    /** total pool surface, m² */
    waterArea: number;
    outdoor: OutdoorRow[];
}
export interface Schedule extends LevelSchedule {
    /**
     * Per level, ground-up, and the building's totals — present only on a document that
     * authored `levels`. A single-level plan's schedule is byte-identical to what it was
     * before levels existed, and the top-level numbers are always the ground level's.
     */
    levels?: Array<{
        id: string;
        name: string;
    } & LevelSchedule>;
    building?: {
        storeys: number;
        /** every level's floor plate summed: gross floor area, m² */
        grossArea: number;
        /** the largest single plate: what the building stands on, m² */
        footprint: number;
        interiorArea: number;
        interiorClearArea: number;
        waterArea: number;
    };
}
export declare function schedule(model: Model): Schedule;
export interface FloorplanResult {
    plan: Plan;
    model: Model;
    findings: Finding[];
    /**
     * The ground (or only) level's drawing, so a single-level caller is unchanged. Use
     * `levels` to reach the others.
     */
    svg: string;
    /** every level, ground-up, each with its own drawing and its own findings */
    levels: Array<{
        id: string;
        name: string;
        svg: string;
        findings: Finding[];
    }>;
    schedule: Schedule;
}
export interface FloorplanOptions {
    render?: RenderOptions;
    rules?: RuleOptions;
    /** draw markers for findings at or above this severity (default: "warning") */
    markFindings?: Severity | "none";
}
/** Everything in one call. Throws PlanError only for schema problems. */
export declare function floorplan(input: unknown, opts?: FloorplanOptions): FloorplanResult;
/** One derived wall, as the JSON output names it. */
export interface WallRow {
    /** unique within its level; `level` is what makes it unique in the building */
    id: string;
    kind: "exterior" | "partition";
    /** only on a document that authored `levels` */
    level?: string;
    from: Pt;
    to: Pt;
    /** owner on the negative side (north for a horizontal wall, west for a vertical one) */
    neg: Owner;
    pos: Owner;
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
export declare function walls(model: Model, level?: string): WallRow[];
/** How many findings of each severity. The first thing an agent reads back. */
export declare function summarize(findings: Finding[]): Record<Severity, number>;
export declare const isSchemaFinding: (f: Finding) => boolean;
export interface LintResult {
    findings: Finding[];
    /** both absent when the document failed the schema */
    plan?: Plan;
    model?: Model;
    /** what `parse()` would have thrown, for a caller that wants its message or its issues */
    error?: PlanError;
}
/**
 * Every problem with a document, through one channel and without throwing: schema
 * problems arrive as `schema.*` findings carrying the issue's own document path, geometry
 * and semantic problems as the rules' findings (docs/agent-review.md §B8).
 *
 * `parse()` and `floorplan()` keep throwing, for callers who want that.
 */
export declare function lint(input: unknown, opts?: FloorplanOptions): LintResult;
export declare function worstSeverity(findings: Finding[]): Severity | undefined;
