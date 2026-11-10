import { normalOn, tangentOn } from "./derive.js";
import { snap } from "./geometry.js";
import { ownerId } from "./types.js";
/** Geometry of a door leaf opened 90° into its swing room. Undefined for non-doors. */
export function doorSwing(o) {
    if (!o.hinge || o.swingRoom === undefined)
        return undefined;
    const w = o.wall;
    const width = o.to - o.from;
    const hingeAtStart = o.spec.hinge === "start";
    // Unit vector along the wall toward the other jamb, and normal into the swing room —
    // taken from the wall's own tangent at the hinge rather than from a table indexed by
    // axis, so a door on a curved wall swings off its tangent. Its closed leaf is then the
    // chord, which is what a leaf is: straight, while the wall it sits in is not.
    const at = o.spec.hinge === "start" ? o.from : o.to;
    const t = tangentOn(w, at);
    const along = hingeAtStart ? t : [-t[0], -t[1]];
    const intoPos = o.swingRoom === ownerId(w.pos);
    const n = normalOn(w, at);
    const normal = intoPos ? n : [-n[0], -n[1]];
    const [hx, hy] = o.hinge;
    const closed = [snap(hx + along[0] * width), snap(hy + along[1] * width)];
    const open = [snap(hx + normal[0] * width), snap(hy + normal[1] * width)];
    const cross = along[0] * normal[1] - along[1] * normal[0];
    return {
        hinge: o.hinge,
        closed,
        open,
        sweep: cross > 0 ? 1 : 0,
        box: {
            x0: Math.min(hx, closed[0], open[0]),
            y0: Math.min(hy, closed[1], open[1]),
            x1: Math.max(hx, closed[0], open[0]),
            y1: Math.max(hy, closed[1], open[1]),
        },
    };
}
