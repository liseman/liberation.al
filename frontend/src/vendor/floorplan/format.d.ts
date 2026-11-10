export interface FormatOptions {
    /** spaces per block level (default 2) */
    indent?: number;
}
/**
 * Print a JSON document in canonical form. This is a *document* printer: it emits exactly
 * the keys it is given, so a document authored with `rect` keeps its `rect` and one
 * authored with `poly` keeps its `poly`. A parsed `Plan` has already had `rect` expanded
 * to `poly`, so `formatPlan(plan)` cannot recover the shorthand — `formatText`, which goes
 * from source text to source text, is the one that preserves the author's form.
 */
export declare function formatPlan(value: unknown, opts?: FormatOptions): string;
/** Reformat source text; throws whatever JSON.parse throws when the text is not valid. */
export declare const formatText: (text: string, opts?: FormatOptions) => string;
