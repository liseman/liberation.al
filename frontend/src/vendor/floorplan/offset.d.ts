import type { MmRing } from "./ring.ts";
/**
 * Inward offset of a ring by a per-edge distance, mitred at the corners.
 *
 * An arc offsets to the same arc on the same centre with its radius reduced (when the
 * interior is on the centre's side) or increased (when it is not) — one case, still
 * exact, which is the whole reason arcs are kept as arcs rather than flattened.
 *
 * Where an edge is shorter than the offsets meeting on it the mitred ring turns back on
 * itself. Such edges are dropped and the neighbours re-mitred, which is the explicit
 * form of the `Math.max(0, …)` that used to sit at the end of the closed form and was
 * doing real work undocumented. If nothing survives, the ring is empty and the caller
 * should say so: a room with no clear floor is a room the author wants to hear about.
 */
export declare function offsetRing(ring: MmRing, dist: (edge: number) => number): MmRing;
/** Area in mm² of the clear floor a ring leaves under its walls. Never negative. */
export declare const offsetArea: (ring: MmRing, dist: (edge: number) => number) => number;
