export type JsonKind = "object" | "array" | "string" | "number" | "boolean" | "null";
export type JsonPath = Array<string | number>;
export interface JsonNode {
    kind: JsonKind;
    /** character range of the value itself, excluding surrounding whitespace */
    start: number;
    end: number;
    /** objects: member name -> value node, in source order */
    members?: Map<string, JsonNode>;
    /** arrays: element nodes */
    items?: JsonNode[];
    /**
     * Set only on a node that is an object member's value: the index of the opening
     * quote of its key, i.e. where the *member* (key, not just the value) begins in
     * the source. removeAt/insertKey use this to splice a whole "key": value pair.
     */
    keyStart?: number;
}
export declare class JsonPosError extends Error {
    readonly index: number;
    readonly line: number;
    readonly column: number;
    constructor(message: string, text: string, index: number);
}
/** Parse JSON text into a tree of nodes carrying their source ranges. */
export declare function parseWithPositions(text: string): JsonNode;
/** The node at a path, or undefined if the path does not exist. */
export declare function nodeAt(root: JsonNode, path: JsonPath): JsonNode | undefined;
/**
 * Replace the value at `path` with `literal`, returning the new text and the change
 * so a caller can keep a caret in place. Throws if the path does not exist — a drag
 * must never silently write nothing.
 */
export declare function spliceAt(text: string, path: JsonPath, literal: string, root?: JsonNode): {
    text: string;
    start: number;
    removed: number;
    inserted: number;
};
/** Apply several edits at once; ranges must not overlap. Applied right to left. */
export declare function spliceAll(text: string, edits: Array<{
    path: JsonPath;
    literal: string;
}>): string;
/**
 * Delete the array element or object member at `path`, eating the separating comma and
 * its surrounding whitespace so the result stays valid JSON, one entity per line:
 * removing the last sibling eats the *preceding* comma, removing any other sibling eats
 * the comma (and newline/indentation) that *follows* it, and removing the only sibling
 * collapses the collection to `[]`/`{}`. Throws `JsonPosError` if `path` does not exist.
 */
export declare function removeAt(text: string, path: JsonPath, root?: JsonNode): string;
/**
 * Insert `literal` as a new element after the array's last one, on its own line at the
 * last sibling's indentation and with its separator style (see `trailingJoint`). An
 * empty array becomes `[literal]`. Throws `JsonPosError` if `path` is not an array.
 */
export declare function appendAt(text: string, path: JsonPath, literal: string, root?: JsonNode): string;
/**
 * Insert a new member `key: literal` after the object's last one, on its own line at
 * the last sibling's indentation and with its separator style. An empty object becomes
 * `{ "key": literal }`. Throws `JsonPosError` if `path` is not an object, or if `key`
 * is already a member (an agent should never silently overwrite a sibling this way —
 * use `spliceAt`/`set` for that).
 */
export declare function insertKey(text: string, path: JsonPath, key: string, literal: string, root?: JsonNode): string;
export declare const pathToString: (path: JsonPath) => string;
/** Metres formatted the way the fixtures write them: no trailing zeros, mm precision. */
export declare const metres: (n: number) => string;
