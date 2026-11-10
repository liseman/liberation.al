import type { Finding, LevelModel, Model, Pt } from "./types.ts";
export interface RenderOptions {
    /**
     * Which level to draw (default: the ground level). One SVG per level is the convention
     * architects read and the token-cheap answer for an agent, which asks for the level it
     * is working on.
     */
    level?: string;
    /**
     * Draw the level below in outline under this one (default true). It is what makes it
     * possible to see at a glance whether the walls upstairs land on the walls downstairs.
     */
    ghost?: boolean;
    /** pixels per metre (default 40) */
    scale?: number;
    theme?: "auto" | "light" | "dark";
    /** override any CSS custom property, e.g. { wall: "#000", bg: "#fff" } */
    colors?: Partial<Record<"bg" | "wall" | "ink" | "muted" | "accent", string>>;
    /** "auto": full label where it fits, numbered key otherwise */
    labels?: "auto" | "full" | "index";
    /** which area to print under the room name */
    areas?: "clear" | "centreline" | "none";
    dimensions?: boolean;
    /** draw markers for these findings */
    findings?: Finding[];
    locale?: string;
    title?: string;
}
/**
 * Where the drawing puts a point. Exported because a caller that wants to interact with
 * the SVG — hit-testing a wall, dragging it — has to turn client pixels back into metres,
 * and must use the very same transform the renderer used or the two will disagree.
 */
export interface Projection {
    /** pixels per metre */
    scale: number;
    /** drawing origin in SVG units */
    ox: number;
    oy: number;
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
    toScreen(p: Pt): Pt;
    /** SVG user units back to metres */
    toModel(x: number, y: number): Pt;
}
/**
 * The level a drawing is about: the one `opts.level` names, or the ground level. An
 * unknown id falls back to the ground level rather than throwing — a render is never the
 * right place to fail a plan that parsed.
 */
export declare function levelOf(model: Model, level: string | undefined): LevelModel;
export declare function projection(model: Model, opts?: RenderOptions): Projection;
/** Render a derived model to a standalone SVG string. Pure; no DOM. */
export declare function renderSvg(model: Model, opts?: RenderOptions): string;
