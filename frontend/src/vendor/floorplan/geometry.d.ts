import type { ArcSpec, Pt } from "./types.ts";
/** Snap to a 1 mm grid; removes float noise from authored coordinates. */
export declare const snap: (n: number) => number;
export declare const eq: (a: number, b: number) => boolean;
export declare function shoelace(poly: Pt[]): number;
export interface BBox {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
}
export declare function bbox(poly: Pt[]): BBox;
/** Even-odd ray cast. Points exactly on an edge are unspecified; callers probe cell centres. */
export declare function pointInPoly(p: Pt, poly: Pt[]): boolean;
export type PolyProblem = {
    kind: "too_few_points";
    count: number;
} | {
    kind: "zero_area";
} | {
    kind: "self_intersecting";
    edges: [number, number];
} | {
    kind: "arc_radius";
    edge: number;
    needed: number;
    got: number;
};
/**
 * Snap, drop repeated and collinear corners, and check the ring is a simple closed loop.
 *
 * Any simple polygon is accepted — the rectilinear test that used to sit here is gone
 * (docs/gaps-design.md §1.2) — and an edge may carry an arc. What is still refused, with
 * the reason: fewer than three distinct corners, zero area, an arc whose radius cannot
 * span its chord, and a boundary that crosses or touches itself.
 */
export declare function normalizeRing(rawPts: Pt[], rawArcs: ReadonlyArray<ArcSpec | undefined>): {
    poly: Pt[];
    arcs: Array<ArcSpec | undefined>;
} | {
    problem: PolyProblem;
};
/** The straight-edged form, kept for the grid compiler and for callers with no arcs. */
export declare function normalizePoly(raw: Pt[]): {
    poly: Pt[];
} | {
    problem: PolyProblem;
};
/**
 * Outline of a union of grid cells. `cells` are [col,row] indices; xs/ys are the
 * track coordinates. Returns one polygon per connected component (holes are
 * returned as separate loops, so callers should demand exactly one loop).
 */
export declare function cellsToPolygons(cells: Array<[number, number]>, xs: number[], ys: number[]): Pt[][];
/** Largest axis-aligned rectangle made of fully owned cells. O(cols²·rows²), fine at house scale. */
export declare function largestRect(owned: (i: number, j: number) => boolean, xs: number[], ys: number[]): BBox;
