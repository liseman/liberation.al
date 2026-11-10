// Shared types. Coordinates are metres on wall centrelines, y grows downwards
// (plan-view screen convention: north is up).
export const HABITABLE_KINDS = new Set(["bedroom", "living", "kitchen", "office"]);
export const WET_KINDS = new Set(["bath", "wc"]);
export const CIRCULATION_KINDS = new Set(["hall", "corridor"]);
/** Is any edge of this shape curved or off-axis? Decides which measures a room gets. */
export const isRectilinear = (s) => {
    if (s.arcs.some((a) => a !== undefined))
        return false;
    for (let i = 0; i < s.poly.length; i++) {
        const a = s.poly[i];
        const b = s.poly[(i + 1) % s.poly.length];
        if (a[0] !== b[0] && a[1] !== b[1])
            return false;
    }
    return true;
};
/** The id a document with no `levels` block is normalised onto. */
export const GROUND_LEVEL = "ground";
export const EXTERIOR = { kind: "exterior" };
export const GAP = { kind: "gap" };
export const roomOwner = (id) => ({ kind: "room", id });
export const outdoorOwner = (id) => ({ kind: "outdoor", id });
export const voidOwner = (id) => ({ kind: "void", id });
/** every owner kind that names a declared space in the document */
const DECLARED = new Set(["room", "outdoor", "void"]);
/** the space id an owner names, or undefined for the exterior, a gap or an overlap */
export const ownerId = (o) => (DECLARED.has(o.kind) ? o.id : undefined);
/** Stable key for maps and sets. Kinds are distinct, so a room "exterior" is not the street. */
export const ownerKey = (o) => DECLARED.has(o.kind)
    ? `${o.kind}:${o.id}`
    : o.kind === "overlap"
        ? `overlap:${[...o.ids].sort().join("+")}`
        : o.kind;
export const sameOwner = (a, b) => ownerKey(a) === ownerKey(b);
/**
 * Open sky above: the street, or a declared outdoor space.
 *
 * INVARIANT: this — not "is it the street" — is what makes a wall an exterior wall. A
 * courtyard wall faces open air and is built like any other outside wall, so it keeps
 * exterior thickness and a window onto a patio satisfies habitable.no_window instead of
 * tripping window.not_exterior. Access is a separate question: see isStreet.
 */
export const isOpenSky = (o) => o.kind === "exterior" || o.kind === "outdoor";
/**
 * No floor and no roof: open sky, or an undeclared hole. Two of these never have a wall
 * between them.
 *
 * INVARIANT: a declared `void` is floorless but is NOT one of these, and deliberately so.
 * It has the building over it, so the wall beside a stairwell derives as a partition, the
 * envelope wall still runs past a double-height space that reaches the façade, and a
 * window onto a void is still `window.not_exterior`.
 */
export const isVoid = (o) => isOpenSky(o) || o.kind === "gap";
/**
 * Does this side open onto the street: the exterior itself, or an outdoor space the
 * border flood fill reaches (see Model.streetOutdoor)? An enclosed courtyard does not.
 */
export const isStreet = (o, streetOutdoor) => o.kind === "exterior" || (o.kind === "outdoor" && streetOutdoor.has(o.id));
/**
 * `e.path`, plus the first of `fields` the document actually authored on `e`.
 *
 * This is the whole of "derive the path from the parsed document, never from a template"
 * (docs/gaps-design.md §2.5): the parser records what it read, and a rule asks for the
 * field it cares about — `pathTo(room, "poly", "rect")` for a room's geometry, whichever
 * form the author chose — instead of guessing a key that may not be there.
 */
export const pathTo = (e, ...fields) => {
    const f = fields.find((k) => e.authored.includes(k));
    return f === undefined ? e.path : `${e.path}.${f}`;
};
/** A path under one level's content: `openings` or `levels.piso1.openings`. */
export const inLevel = (level, suffix) => level.path === "" ? suffix : `${level.path}.${suffix}`;
