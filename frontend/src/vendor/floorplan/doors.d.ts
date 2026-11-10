import type { BBox } from "./geometry.ts";
import type { Pt, ResolvedOpening } from "./types.ts";
export interface DoorSwing {
    hinge: Pt;
    /** closed leaf tip: the other jamb */
    closed: Pt;
    /** open leaf tip: 90° into the swing room */
    open: Pt;
    /** SVG arc sweep flag from closed to open (screen coordinates, y down) */
    sweep: 0 | 1;
    /** bounding box of the quarter-disc swept by the leaf */
    box: BBox;
}
/** Geometry of a door leaf opened 90° into its swing room. Undefined for non-doors. */
export declare function doorSwing(o: ResolvedOpening): DoorSwing | undefined;
