import type { Finding, Model, RoomKind } from "./types.ts";
export interface RuleOptions {
    /** share of interior area above which circulation is flagged (default 0.10) */
    circulationShare?: number;
    /** comfort minimum for the short side of a room's largest clear rectangle, per kind (metres) */
    minDimension?: Partial<Record<RoomKind, number>>;
    /** minimum door widths (metres) */
    doorMinWidth?: {
        interior?: number;
        entrance?: number;
    };
    /** walkable gap required between two fixtures in the same room (default 0.6 m) */
    minClearance?: number;
    /**
     * Comfortable stair geometry: pitch in degrees and the shortest going (default 30–42°,
     * 0.25 m). These are conventions and not a code, which is exactly why they are options.
     */
    stairPitch?: {
        min?: number;
        max?: number;
        going?: number;
    };
    /** clear height under the slab a stair passes through, metres (default 2.0) */
    minHeadroom?: number;
}
/**
 * Semantic rules over the derived model. Geometry/topology errors are already
 * produced by derive(); these are the "is this a good house" checks.
 *
 * Three scopes, and each rule belongs to exactly one: **per level** (tiling, walls,
 * openings, light, privacy, sizes — a storey is its own plan), **building-wide**
 * (`entrance.*` and `reach.*`, because you enter a building once and then walk through
 * all of it), and **cross-level** (the stair rules, which are about the seam itself).
 */
export declare function checkRules(model: Model, opts?: RuleOptions): Finding[];
export declare function sortFindings(findings: Finding[]): Finding[];
