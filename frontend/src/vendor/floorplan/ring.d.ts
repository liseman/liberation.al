import type { ArcSpec, Pt } from "./types.ts";
/** Integer millimetres. */
export type Mm = number;
/** A point in integer millimetres. */
export type P = [Mm, Mm];
/** metres → integer millimetres. Lossless for any coordinate the parser has snapped. */
export declare const toMm: (m: number) => Mm;
/** integer millimetres → metres, exactly representable for any plan under 9000 km. */
export declare const toM: (mm: Mm) => number;
export declare const ptMm: (p: Pt) => P;
export declare const ptM: (p: P) => Pt;
/**
 * The chord sagitta an arc is flattened to, in millimetres.
 *
 * INVARIANT: this is a constant and not a parameter, and `flattenArc` takes nothing but
 * the arc. Two spaces that share a curved wall each flatten it from their own ring; if
 * they could disagree by half a step the shared boundary would split into a comb of
 * slivers — measured at 61 faces totalling 277 cm² on an R = 3 m semicircle
 * (docs/gaps-design.md §1.3.3). Canonical input, canonical output, no free parameters.
 */
export declare const SAGITTA_MM = 1;
/**
 * A circular arc, resolved: centre, radius and the signed angular span from `a` to `b`.
 * Angles are `atan2(y, x)` in the document's frame, where y grows southwards, so a
 * positive span turns clockwise on the page — the same handedness `shoelace` uses.
 */
export interface Arc {
    c: [number, number];
    r: number;
    /** start angle, radians */
    t0: number;
    /** signed span, radians; |span| < 2π */
    span: number;
    a: P;
    b: P;
}
export type ArcProblem = {
    kind: "radius_too_small";
    needed: number;
    got: number;
} | {
    kind: "degenerate";
};
/**
 * Resolve `{ arc: [x,y], r, sweep, large }` against the previous corner.
 *
 * Endpoints, a radius, a turn direction and the reflex flag are the minimal
 * non-redundant description of a circular arc, and each of the four is a number an
 * agent can compute or edit on its own (docs/gaps-design.md §1.2). The centre is
 * derived, never authored, so a `spliceAt` on `r` cannot make the record inconsistent.
 */
export declare function resolveArc(a: P, b: P, r: Mm, cw: boolean, large: boolean): Arc | {
    problem: ArcProblem;
};
/** Where an arc is, `t` of the way along its span. */
export declare const arcPoint: (arc: Arc, t: number) => [number, number];
export declare const arcLength: (arc: Arc) => number;
/** Unit tangent, pointing the way the arc is travelled. */
export declare function arcTangent(arc: Arc, t: number): [number, number];
/** The arc that is this one run backwards. */
export declare const reverseArc: (arc: Arc) => Arc;
/** Lexicographic order on points: x, then y. */
export declare const before: (p: P, q: P) => boolean;
/**
 * Chord points for an arc, excluding both endpoints, identical for either space that
 * shares it. Canonical in two ways: the subdivision always runs from the
 * lexicographically smaller endpoint, and the step is fixed by the radius alone.
 */
export declare function flattenArc(arc: Arc): P[];
/** How many chords `flattenArc` uses. Reported by `arc.too_shallow` and by the tests. */
export declare const arcChords: (arc: Arc) => number;
/** Height of the arc over its chord: what makes a curve worth writing as one. */
export declare function sagitta(arc: Arc): number;
/**
 * A closed boundary. `pts` are the corners in document order; `arcs[i]`, when present,
 * curves the edge from `pts[i]` to `pts[i+1]`. A ring with no arcs is exactly the
 * polygon the library had before, which is why `Room.poly` can still be `pts`.
 */
export interface MmRing {
    pts: P[];
    arcs: Array<Arc | undefined>;
}
export declare const straightRing: (pts: P[]) => MmRing;
/** The ring's corner + arc description in metres, as the model carries it. */
export interface Ring {
    pts: Pt[];
    arcs: Array<ArcSpec | undefined>;
}
/** Resolve a metre-space ring into millimetres. Arcs whose radius is impossible are dropped. */
export declare function mmRing(ring: Ring): MmRing;
/** Every edge of a ring, as an index plus its two corners and its arc if it has one. */
export declare function ringEdges(ring: MmRing): Array<{
    i: number;
    a: P;
    b: P;
    arc: Arc | undefined;
}>;
/**
 * Signed area in mm², positive when the ring winds clockwise on the page. Exact for
 * arcs: each edge contributes ½∮(x dy − y dx) over its own parameterisation, which for
 * a circular arc closes to cx·Δy − cy·Δx + r²·span.
 */
export declare function ringArea(ring: MmRing): number;
/** Perimeter in mm, arcs by arc length. */
export declare function ringPerimeter(ring: MmRing): number;
export declare function ringPoints(ring: MmRing): P[];
/** Even-odd ray cast against the flattened ring. Points on the boundary are unspecified. */
export declare function pointInRing(p: P, ring: MmRing): boolean;
export declare function pointInPolyMm(p: [number, number], poly: Array<[number, number]>): boolean;
export interface Box {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
}
/** Bounding box in mm, tight over arcs: a bulge past its chord counts. */
export declare function ringBox(ring: MmRing): Box;
/** Is the ray at angle `ang` from the centre within the arc's span? */
export declare function onArc(arc: Arc, ang: number): boolean;
/** Where along the arc's span a ray at `ang` falls, 0..1, or undefined if it misses. */
export declare function arcParamAt(arc: Arc, ang: number): number | undefined;
