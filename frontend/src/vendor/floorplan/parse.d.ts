import type { Plan } from "./types.ts";
/**
 * What kind of schema problem this is. The vocabulary is the one the existing messages
 * already fall into, and `lint()` turns each into a `schema.<kind>` finding — so an agent
 * that reads findings sees schema problems and geometry problems through one channel.
 *
 *  - `syntax`        the text is not JSON, or the document is not an object
 *  - `unknown_field` a key the schema does not have (checkKeys)
 *  - `missing`       a required thing is absent: geometry, an id, a room, a footprint
 *  - `type`          present but the wrong type, or outside a fixed vocabulary
 *  - `reference`     names something that is not declared: a space, a level, a side
 *  - `geometry`      a polygon or rectangle that cannot be a shape
 *  - `conflict`      two mutually exclusive forms, or an id used twice
 */
export type IssueKind = "syntax" | "unknown_field" | "missing" | "type" | "reference" | "geometry" | "conflict";
export interface PlanIssue {
    path: string;
    message: string;
    kind: IssueKind;
    /**
     * 1-based line in the source text, for a document authored in the line DSL. Absent for
     * a JSON document, whose `path` is already the way to find the value.
     */
    line?: number;
}
export declare class PlanError extends Error {
    readonly issues: PlanIssue[];
    constructor(issues: PlanIssue[]);
}
export declare const ROOM_KINDS: ReadonlySet<string>;
export declare const SIDES: ReadonlySet<string>;
export declare const OPENING_TYPES: ReadonlySet<string>;
export declare const FIXTURE_TYPES: ReadonlySet<string>;
export declare const VERTICAL_TYPES: ReadonlySet<string>;
/** Which way an arc turns, seen on the page: y grows south, so "cw" is clockwise there. */
export declare const SWEEPS: ReadonlySet<string>;
export declare const ID_RE: RegExp;
/**
 * The document's own schema, machine-readable. `checkKeys` below reads its known-key
 * lists from here (see FIELDS_OF and the per-object constants that follow), so a field
 * this table does not mention is a field the parser rejects, by construction — SCHEMA and
 * the parser cannot drift apart the way the README's hand-written tables and
 * `app/src/routes/reference.tsx` used to (docs/agent-review.md B10). `floorplan --schema`
 * prints this table; that is the ≈600-token replacement for the README's ~2 400-token
 * prose an agent no longer needs to read.
 */
export interface FieldDoc {
    name: string;
    /**
     * A short, actionable type tag. `"number[]"` is not one of the nine shapes the review
     * asked for (number, string, [x,y], [x,y,w,h], point[], string[], enum, boolean,
     * object) — it was added because `grid.cols`/`grid.rows`/`layout.cols`/`layout.rows`
     * are lists of plain numbers, and tagging them `"object"` would tell an agent nothing
     * about how to write one; `"number[]"` follows the same naming the review already used
     * for `point[]` and `string[]`.
     */
    type: "number" | "number[]" | "string" | "[x,y]" | "[x,y,w,h]" | "point[]" | "string[]" | "enum" | "boolean" | "object";
    required: boolean;
    /** For `type: "enum"`: a reference to the exported vocabulary itself, never a copy. */
    enum?: ReadonlySet<string>;
    /**
     * For `type: "object"` only: which of three cardinalities the field actually holds —
     * `"object"` alone cannot say, and that one omission is what made every cold-agent JSON
     * authoring attempt guess "array" for `levels` and fail schema on line one
     * (docs/eval/cold/cold-run.md). The parser already knows this per field (`Object.entries`
     * for a map, `Array.isArray` for a list, a plain nested read for one); `shape` is that
     * same fact surfaced to the field table so `--schema`/`--schema=full`/`--schema=md` can
     * print `{id: room}`, `opening[]`, or a bare `opening.on` instead of three fields that all
     * read `object`.
     *
     *  - "one":  a single nested object (`opening.on`, `plan.grid`)
     *  - "list": an array (`level.openings`, `vertical.at`)
     *  - "map":  an object keyed by an id the author invents, the id itself never a field
     *    (`plan.levels`, `level.rooms`)
     */
    shape?: "one" | "list" | "map";
    /** One sentence: units, default, and what reads it. */
    doc: string;
}
export interface ObjectDoc {
    /** Dotted path names a nested shape that has no id of its own, e.g. an opening's `on`. */
    object: "plan" | "walls" | "grid" | "layout" | "level" | "room" | "outdoor" | "void" | "arc" | "opening" | "opening.on" | "opening.position" | "fixture" | "vertical" | "vertical.footprint";
    fields: readonly FieldDoc[];
    /** Mutual exclusions among this object's fields, one sentence each. */
    oneOf?: readonly string[];
}
export declare const SCHEMA: readonly ObjectDoc[];
/**
 * Parse and normalise a plan document. Throws PlanError listing every schema
 * problem found. Geometry/topology problems are NOT raised here — they become
 * validator findings so the caller can still render a partial drawing.
 */
export declare function parse(input: unknown): Plan;
