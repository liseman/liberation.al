import type { Arc, MmRing } from "./ring.ts";
import type { Analysis, Fixture, Model, Owner, Plan, Pt, Shape, Side, Wall, WallGeometry } from "./types.ts";
/**
 * Turn authored rooms into walls, resolve openings onto walls, compute room
 * metrics and the access graph, one level at a time; then join the levels.
 * Geometry/topology problems are returned as findings, never thrown, so a
 * broken plan still yields a drawable model.
 */
export declare function derive(plan: Plan): Analysis;
export declare function ringOf(s: Shape): MmRing;
/** Centreline area in m², exact over arcs. */
export declare const shapeArea: (s: Shape) => number;
export declare function shapeBox(s: Shape): {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
};
/**
 * How much of `a` stands on `b`, in m². The arrangement of the two rings, read with an
 * intersection predicate — the boolean operation that needs no clipper (§1.3.2).
 */
export declare const overlapArea: (a: Shape, b: Shape) => number;
/** Is `a` entirely on `b`'s floor? The area that is, against the area there is. */
export declare const shapeWithin: (a: Shape, b: Shape) => boolean;
/**
 * The part of `a` that none of `others` covers: how much, in m², and a point in it. One
 * arrangement of every ring involved, read with a "covered by a and by nothing else"
 * predicate — the difference of §1.3.2's table.
 */
export declare function uncovered(a: Shape, others: Shape[]): {
    area: number;
    at: Pt;
};
/**
 * Is this point on the shape's boundary, to the millimetre? Used to decide whether
 * moving a wall's end would drag a corner off a third space's edge — the T-junction that
 * a "is it one of its corners" test misses.
 */
export declare function pointOnShapeBoundary(p: Pt, s: Shape, tol?: number): boolean;
/**
 * Shortest distance between two shapes, in metres; 0 when they touch or overlap.
 *
 * Replaces the distance between two *bounding boxes*, which was already wrong for an
 * L-shaped kitchen run before any wall was angled (docs/gaps-design.md §1.3.6). Arcs are
 * measured on their canonical chords, so a curved fixture is understated by at most the
 * one-millimetre sagitta.
 */
export declare function shapeGap(a: Shape, b: Shape): number;
/**
 * Does the quarter-disc a door leaf sweeps meet this shape?
 *
 * The exact predicate, rather than "do their bounding boxes overlap and is the nearest
 * corner of the overlap within reach": a vertex inside the sector, an edge crossing
 * either radius, an edge crossing the arc, or the hinge inside the shape.
 */
export declare function sectorMeetsShape(hinge: Pt, closed: Pt, open: Pt, s: Shape): boolean;
/**
 * Is this point inside the shape? Arcs are tested on their canonical chords.
 *
 * The bounding box first: the clear-rectangle sweep asks this once per grid cell per room,
 * and a curved room's ring is a hundred chords, so rejecting the cells that are nowhere
 * near it is most of the work saved.
 */
export declare function pointInShape(p: Pt, s: Shape): boolean;
/**
 * Which way an exterior wall faces, seen from the room behind it: degrees clockwise from
 * north. `exteriorFaceOf` rounds this to a compass point, which is exact for an
 * axis-aligned wall and an approximation for any other; a message about an angled room
 * should quote the bearing instead.
 */
export declare function outwardBearing(wall: Wall, id: string): number | undefined;
export declare const label: (o: Owner) => string;
/**
 * How a finding points at a thing standing on the floor. An authored fixture is an index
 * into the document's `fixtures`; a vertical element's synthetic footprint has no such
 * index, so it names the element instead.
 */
export declare const occupantRef: (f: Fixture) => {
    fixture: string;
} | {
    vertical: string;
};
/**
 * Which compass side of the space `ref` this wall lies on. Defined only for an
 * axis-aligned wall: `north`, `south`, `east` and `west` say nothing about a wall at 20°,
 * and `at: [x, y]` is the selector that survives angles (docs/gaps-design.md §1.5).
 */
export declare function sideOf(w: Wall, ref: string): Side | undefined;
/** Where on the wall the parameter `t` is. */
export declare function pointOn(w: Wall, t: number): Pt;
/** The millimetre arc record behind an arc geometry. */
export declare function arcOfGeometry(g: WallGeometry & {
    kind: "arc";
}): Arc;
/** How a wall reads in a message. */
export declare function describe(w: Wall): string;
/** Distance from a point to the wall, in metres. */
export declare function distToWall(p: Pt, w: Wall): number;
/** The wall parameter of the point on the wall nearest `p`. */
export declare function paramAt(w: Wall, p: Pt): number;
/** Unit tangent of a wall at parameter `t`, pointing the way `from` → `to` runs. */
export declare function tangentOn(w: Wall, t: number): Pt;
/**
 * The unit normal of a wall at parameter `t`, pointing toward its `pos` side.
 *
 * For an axis-aligned wall that is south for a horizontal wall and east for a vertical
 * one, which is exactly what the window glazing and the entrance tag used to derive from
 * `axis` by table (`svg.ts`, `doors.ts`).
 */
export declare function normalOn(w: Wall, t: number): Pt;
/** The piece of a wall between two of its parameters, as exact geometry. */
export declare function wallSubGeometry(w: Wall, a: number, b: number): WallGeometry;
/**
 * The same wall geometry moved `d` metres toward its `pos` side. A segment translates; an
 * arc keeps its centre and changes radius, which is why window glazing on a curved wall
 * is two concentric arcs rather than two chords.
 */
export declare function offsetGeometry(g: WallGeometry, d: number): WallGeometry;
/**
 * A whole shape as a closed SVG path: straight edges as `L`, curved ones as `A`. A ring
 * with no arcs is left to the caller to draw as a `<polygon>`, which is what it is.
 */
export declare function shapePath(sh: Shape, X: (m: number) => number, Y: (m: number) => number, S: number): string;
/** Has any edge of this shape a curve on it? Decides polygon versus path. */
export declare const isCurved: (sh: Shape) => boolean;
/** Area centroid of a shape, in metres; where a label without a room goes. */
export declare function shapeCentroid(sh: Shape): Pt;
/** Every point an SVG path needs to draw this geometry, as `M`/`L`/`A` commands. */
export declare function geometryPath(g: WallGeometry, X: (m: number) => number, Y: (m: number) => number, S: number): string;
/** Where a wall geometry ends, in metres. */
export declare function geometryEnd(g: WallGeometry): Pt;
export type { Model };
