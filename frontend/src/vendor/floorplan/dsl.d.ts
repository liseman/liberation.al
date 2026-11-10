export interface DslTokenDoc {
    /** how the token is written, e.g. `w<width>` or `hinge:start|end` */
    token: string;
    /** the SCHEMA field it writes, as `<object>.<field>` — the coverage test walks these */
    field: string;
    required: boolean;
    doc: string;
}
export interface DslStatementDoc {
    /** the statement's leading verb, or the set of verbs that introduce it */
    statement: string;
    /** the whole line, `[…]` around what may be left out */
    syntax: string;
    doc: string;
    tokens: readonly DslTokenDoc[];
    /**
     * True for a thing that is not a line of its own but an element inside another
     * statement's token — today only `arc`, which lives inside a `poly`. It is in this
     * table so the coverage test can see the SCHEMA fields it writes, and the grammar
     * printer lists it under "poly elements" rather than among the statements.
     */
    element?: boolean;
    /**
     * A concrete worked snippet, printed between `syntax` and `doc` — before the prose, not
     * after it, so the shape it demonstrates cannot be skimmed past as a parenthetical. Only
     * the "stairs | lift | ramp" statement has one: `docs/eval/cold/cold-run.md` found 4 of 20
     * cold-agent DSL authors wrote a single `at` line for a multi-level stair despite the
     * syntax's own "(one indented line per level served)" aside, because a sentence about
     * cardinality is exactly the fact a worked example removes any doubt about.
     */
    example?: string;
}
/** An `arc` element inside a `poly`, spelled out once for the grammar table and the docs. */
export declare const ARC_SYNTAX = "arc <x>,<y> r<radius> [cw|ccw] [large]";
/**
 * Every statement kind, with every token it accepts and the SCHEMA field that token
 * writes. `test/dsl-schema.test.ts` walks SCHEMA and fails if a field has no token here,
 * so a field added to the parser without a DSL spelling cannot ship.
 */
export declare const DSL_SCHEMA: readonly DslStatementDoc[];
/**
 * Resolves the one thing neither reference showed a worked example of (docs/eval/cold2/
 * cold-run.md gap 7): whether a `poly`'s corners sit on the entity's own line or in an
 * indented block below it, the way `layout`'s cells and a `vertical`'s footprints do. They
 * are inline; this is the concrete counter-example to that guess, not a restatement of the
 * abstract per-element grammar below it.
 */
export declare const POLY_INLINE_EXAMPLE = "a poly's corners are inline on the entity's own line, never an indented block: outdoor deck poly 0,0 5,0 5,2 0,2";
export interface DslIssue {
    /** 1-based */
    line: number;
    /** 1-based */
    column: number;
    /** already prefixed with `line N: ` — the house style for a text-position error */
    message: string;
}
export declare class DslError extends Error {
    readonly issues: DslIssue[];
    constructor(issues: DslIssue[]);
}
/**
 * Where one JSON path's value sits in the DSL text, and how to write a new one there.
 * The line-based twin of a `jsonpos` node: `start`/`end` are absolute character offsets,
 * so a splice is the same operation it is for JSON, and `line` is what a finding carries
 * so an agent can act on the line rather than reconstruct it.
 */
export interface DslSpan {
    line: number;
    column: number;
    start: number;
    end: number;
    /** render a replacement JSON value as the DSL text for this span */
    write: (value: unknown) => string;
}
/** JSON path (as `pathToString` writes it) → where it is in the text. */
export type DslPositions = Map<string, DslSpan>;
export interface DslDocument {
    /** the JSON document, exactly the shape `parse()` accepts */
    doc: Record<string, unknown>;
    positions: DslPositions;
}
/** Shortest round-trip, which is what `String(number)` already gives in JS. */
export declare const numText: (n: unknown) => string;
/**
 * Compile DSL text to the JSON document `parse()` accepts, collecting *every* line that
 * does not parse rather than stopping at the first — an agent that mistyped three doors
 * should learn about three doors.
 */
export declare function parseDsl(text: string): DslDocument;
/**
 * Print a JSON plan document as canonical DSL. Like `formatPlan`, this is a *document*
 * printer: it writes the fields it is given, so a document authored with `rect` keeps its
 * `rect`. The two synonym pairs `parse()` folds together (`position` as a number versus
 * `{from:"start",distance}`; an explicit `"center"` versus none) have one DSL spelling
 * each, so a conversion picks it — see the header.
 */
export declare function toDsl(doc: unknown): string;
/**
 * Which syntax a source text is in: the first non-space character, and nothing else. `{`
 * is JSON, anything else is the DSL — so a JSON document behaves byte-identically to the
 * way it always has, and an empty file keeps JSON's own "not valid JSON" message rather
 * than becoming an empty plan.
 */
export declare function isDslText(text: string): boolean;
/**
 * Read a source text (or pass a value straight through) as the JSON document shape.
 * `positions` is present only for a DSL document, and is what gives a finding its `line`
 * and an edit its splice range.
 */
export declare function readSource(input: unknown): {
    doc: unknown;
    positions?: DslPositions;
};
/** Error raised when a JSON path names nothing the DSL text can splice. */
export declare class DslPosError extends Error {
    readonly line: number | undefined;
    readonly column: number | undefined;
    constructor(message: string, line?: number, column?: number);
}
/**
 * Replace the value at `path` with `literal` (a JSON literal, as `set` and the drag layer
 * produce) in a DSL document, leaving every other character of the line — and every other
 * line — exactly as it was. The line-based twin of `jsonpos.spliceAt`.
 */
export declare function dslSpliceAt(text: string, path: Array<string | number>, literal: string): {
    text: string;
    start: number;
    removed: number;
    inserted: number;
};
export declare function dslSpliceAll(text: string, edits: Array<{
    path: Array<string | number>;
    literal: string;
}>, detail?: false): string;
export declare function dslSpliceAll(text: string, edits: Array<{
    path: Array<string | number>;
    literal: string;
}>, detail: true): {
    text: string;
    start: number;
    removed: number;
    inserted: number;
};
/**
 * The line a document path sits on, walking up the path until something matches — a
 * finding about `openings[3].width` lands on the opening's line even when only the whole
 * statement was recorded, and one about `rooms.sala` lands on the room's line.
 */
export declare function lineOf(positions: DslPositions, path: string): number | undefined;
/** The x/y ↔ compass fact neither reference used to state at all (docs/eval/cold/cold-run.md
 * item 3: brief 04's cold-agent porch placement guessed the wrong sign for south). Source:
 * `derive.ts`'s per-room wall search reads a room's north wall off its *minimum* y and its
 * south wall off its *maximum* y (`rect.y0`/`rect.y1`), and its west/east the same way on x —
 * so y increases southward and x increases eastward. Shared verbatim by `schemaTerse` (cli.ts)
 * and this function so the two references cannot state it two different ways. */
export declare const COMPASS_LINE = "axes: x east, y south; north = -y";
export interface DslSchemaOptions {
    /** include the 77-row `<object>.<field> → token` index (`--schema=dsl-full`); the default
     * omits it — it is what took `--schema=dsl` from 1 388 to well past a terse reference's
     * budget, and every field it lists is already implied by the statement grammar above it. */
    fieldIndex?: boolean;
    /** appended, converted through `toDsl`, as `example:` — a real, already-lint-clean
     * document (fix 4 for docs/eval/cold/cold-run.md: neither reference had one, and every
     * failure the eval logged is a fact a worked example would have settled). */
    example?: Record<string, unknown>;
    /**
     * The id format and every enum vocabulary, printed as a `## legend` section right after
     * "poly elements" — the same lines `cli.ts`'s shared `vocabLegend()` appends to the JSON
     * terse schema, passed in from there rather than built here, so this module never needs
     * to import the vocabularies twice or the two legends can't say two different things
     * (fix 1, docs/eval/cold2/cold-run.md: `--schema=dsl` never enumerated a room kind or a
     * fixture type at all, which alone caused 5 of 7 DSL parse failures in that eval).
     */
    legend?: readonly string[];
}
/** `--schema=dsl`: one line per statement kind with its tokens; `--schema=dsl-full` adds the
 * field index. Both end with a worked example when `opts.example` is given. */
export declare function dslSchemaText(opts?: DslSchemaOptions): string;
