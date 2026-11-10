import type { JsonPath } from "./jsonpos.ts";
import type { Model, Pt } from "./types.ts";
export interface Draggable {
    wallId: string;
    axis: "h" | "v";
    /** the wall's current centreline coordinate, metres */
    c: number;
    /** clamp range so a drag cannot make the document nonsense */
    min: number;
    max: number;
    /** what moving it rewrites, said plainly enough for a status line */
    writes: string;
    edits: (next: number) => Array<{
        path: JsonPath;
        literal: string;
    }>;
}
/** Every wall on one level that a drag can express as an edit to the source. */
export declare function draggableWalls(text: string, model: Model, level?: string): Map<string, Draggable>;
export declare function applyDrag(text: string, d: Draggable, rawNext: number, free?: boolean): string;
/**
 * Outdoor spaces have no walls — nothing derives from them — so their own edges are the
 * handles. Dragging one moves the coordinate shared by the two vertices at its ends,
 * which for a rectilinear ring is the whole edge.
 */
export declare function draggableOutdoorEdges(text: string, model: Model, level?: string): Map<string, Draggable>;
/** A body that moves in both axes at once, rather than one coordinate along an axis. */
export interface Movable {
    id: string;
    /** the body's north-west corner today, metres */
    at: Pt;
    writes: string;
    edits: (to: Pt) => Array<{
        path: JsonPath;
        literal: string;
    }>;
}
/** Each side of every fixture, so a pool can be made bigger or smaller. */
export declare function draggableFixtureEdges(text: string, model: Model, level?: string): Map<string, Draggable>;
/** Every fixture's body, so a pool can be picked up and put somewhere else. */
export declare function movableFixtures(text: string, model: Model, level?: string): Map<string, Movable>;
export declare function applyMove(text: string, m: Movable, to: Pt, free?: boolean): string;
/**
 * A grip on the drawing, generalised past "a coordinate on an axis" (docs/gaps-design.md
 * §1.3.8). Three kinds, because three is what the geometry has:
 *
 * - **offset** slides a wall along its own normal. The vertices on it move; the edges
 *   running into them keep their far ends and pivot. On an axis-aligned wall this writes
 *   exactly what `draggableWalls` writes, which is the property `handlesMatchDrags` pins.
 * - **radius** changes a curved wall's bulge by splicing the `r` of the arc that made it.
 *   That this is one number in one place is why §1.2 chose an inline arc object over an
 *   SVG path string, which no `spliceAt` could address.
 * - **vertex** moves one corner in two dimensions.
 *
 * `draggableWalls` is not replaced: a grid wall resizes tracks, which is a different and
 * better edit than moving vertices, and it is expressed entirely in document terms.
 */
export type Handle = OffsetHandle | RadiusHandle | VertexHandle;
export interface OffsetHandle {
    kind: "offset";
    id: string;
    wallId: string;
    /** unit normal the wall slides along, pointing at its `pos` side */
    normal: Pt;
    /** where the wall is now, as a projection onto that normal, metres */
    at: number;
    min: number;
    max: number;
    writes: string;
    edits: (next: number) => Array<{
        path: JsonPath;
        literal: string;
    }>;
}
export interface RadiusHandle {
    kind: "radius";
    id: string;
    wallId: string;
    /** the arc's radius now, metres */
    at: number;
    min: number;
    max: number;
    writes: string;
    edits: (next: number) => Array<{
        path: JsonPath;
        literal: string;
    }>;
}
export interface VertexHandle {
    kind: "vertex";
    id: string;
    /** which space and which corner of its `poly` */
    space: string;
    index: number;
    at: Pt;
    writes: string;
    edits: (to: Pt) => Array<{
        path: JsonPath;
        literal: string;
    }>;
}
/**
 * Every handle one level offers. Offsets for straight walls, a radius for each curved
 * one, and a vertex for every corner of every space authored as a `poly`.
 */
export declare function wallHandles(text: string, model: Model, level?: string): Map<string, Handle>;
/** Apply a scalar handle — an offset or a radius — with the same snapping a drag uses. */
export declare function applyHandle(text: string, h: OffsetHandle | RadiusHandle, rawNext: number, free?: boolean): string;
/** Apply a vertex handle: a corner moves in both directions at once. */
export declare function applyVertexHandle(text: string, h: VertexHandle, to: Pt, free?: boolean): string;
