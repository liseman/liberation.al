import type { Arc, MmRing, P } from "./ring.ts";
import { arcParamAt } from "./ring.ts";
/**
 * Twice the signed area of the triangle abc; > 0 when c is clockwise of a→b on the page.
 *
 * INVARIANT: exact in doubles. Coordinates are integer millimetres and a plan spans at
 * most a few hundred metres, so |cross| ≤ 4 × 10¹⁰ against 2⁵³ ≈ 9 × 10¹⁵ — five orders
 * of headroom (docs/gaps-design.md §1.3.3). Every predicate in this file is degree 2 for
 * that reason. A degree-4 predicate — "does the intersection of A and B lie on C",
 * decided without constructing the point — would reach (2 × 10⁵)⁴ and overflow, so it is
 * never used: constructed points are rounded to the grid and re-tested at degree 2, which
 * is what snap-rounding is for.
 */
export declare const orient: (a: P, b: P, c: P) => number;
/** Does point c lie on the closed segment ab? Exact. */
export declare function onSegment(a: P, b: P, c: P): boolean;
/** Do segments ab and cd share a point? Exact, including touching and collinear overlap. */
export declare function segmentsCross(a: P, b: P, c: P, d: P): boolean;
export interface HalfEdge {
    from: number;
    to: number;
    twin: number;
    next: number;
    face: number;
    /** the arc this edge lies on, when it does */
    circle: {
        c: [number, number];
        r: number;
        arc: Arc;
    } | undefined;
    /**
     * Which authored ring edges this piece is part of — both of them where two spaces
     * share a wall. It is how a room's own edge is matched back to the walls running
     * along it, which is what the per-edge offset distance needs.
     */
    srcs: Array<{
        input: number;
        edge: number;
    }>;
}
export interface Face {
    /** boundary cycles, half-edge indices; for a bounded face the first is its outer ring */
    cycles: number[][];
    /** signed mm², positive for a bounded face */
    area: number;
    /** an interior point with the greatest clearance from the boundary */
    probe: [number, number];
    /** indices of the input rings that contain `probe` */
    tags: number[];
}
export interface Arrangement {
    verts: P[];
    half: HalfEdge[];
    faces: Face[];
    /** the unbounded face */
    outer: number;
    rings: MmRing[];
}
declare const twinOf: (h: number) => number;
/** Build the arrangement of every input ring. */
export declare function arrange(rings: MmRing[]): Arrangement;
/** Any point safely inside these rings; the centroid when it is, the pole when it is not. */
export declare function interiorPoint(rings: P[][]): [number, number];
export interface Pole {
    at: [number, number];
    /** clearance from every boundary, mm */
    r: number;
}
/** Signed distance to a set of rings: positive inside, negative outside. */
export declare function distanceToRings(p: [number, number], rings: Array<Array<[number, number]>>): number;
export declare function distToSeg(p: [number, number], a: [number, number], b: [number, number]): number;
/**
 * The pole of inaccessibility: the centre of the largest circle that fits inside
 * `rings` (outer ring first, holes after), and that circle's radius.
 *
 * Two phases, because one does not do. The quadtree refinement — as Mapbox's polylabel
 * does it, on a deterministic max-heap, no randomness and no dependence on input order —
 * bounds the answer quickly but costs dearly to *place* it: along the medial axis of a
 * long room the distance is flat, so every cell on a 12 m band has to be split down to
 * the requested precision. Measured on casa-t3's 12.08 × 1.28 m corridor, 69 181 cells
 * at half a millimetre against 4 317 at five. So the quadtree runs to a coarse bound,
 * and a deterministic pattern search walks downhill from its winner to the requested
 * precision in a few dozen evaluations more.
 *
 * The pattern search only ever increases the clearance, so the result is never worse
 * than the quadtree's bound and is normally exact. Ties are broken by a fixed search
 * order, so a rectangle — whose optimum is a whole segment of its medial axis — always
 * yields the same answer.
 */
export declare function poleOfInaccessibility(rings: Array<Array<[number, number]>>, precision?: number, relative?: number): Pole;
/** The face on the other side of a half-edge. */
export declare const twin: typeof twinOf;
/** Every face whose tag set satisfies the predicate, as boundary rings in millimetres. */
export declare function facesWhere(a: Arrangement, keep: (tags: number[], face: Face) => boolean): P[][];
/** The boundary of a set of faces: one ring per closed loop, holes as their own loops. */
export declare function traceBoundary(a: Arrangement, faces: ReadonlySet<number>): P[][];
/**
 * Overlay: the boolean operations, as face predicates over one arrangement.
 *
 * INVARIANT: this is why there is no clipper in this repository. Union, intersection and
 * difference are the same arrangement read three ways, so there is one robustness story
 * instead of two — and the degeneracies a clipper is worst at, shared vertices and
 * collinear overlapping edges, are exactly what wall junctions are made of
 * (docs/gaps-design.md §1.3.2, dissent D2).
 */
export declare const overlay: (rings: MmRing[]) => Arrangement;
/** Area, in mm², of the region covered by at least `n` of the input rings. */
export declare function coveredArea(a: Arrangement, n: number): number;
/** Area, in mm², covered by ring `i` and by none of `others`. */
export declare function areaOnly(a: Arrangement, i: number, others: ReadonlySet<number>): number;
/** Area, in mm², covered by both ring `i` and ring `j`. */
export declare function areaBoth(a: Arrangement, i: number, j: number): number;
/** The sub-arc a chain of edges on one circle describes, or undefined if it is straight. */
export declare function arcThrough(circle: {
    arc: Arc;
}, from: P, via: P, to: P): Arc | undefined;
/** Is this arrangement edge on a circle at all? */
export declare const edgeArc: (h: HalfEdge) => Arc | undefined;
export { arcParamAt };
