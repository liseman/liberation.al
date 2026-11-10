// Example plans for the Places tab while the house has none of its own: two made-up homes as Pascal scenes
// (scene/build.ts), the format a place's plan is stored in. The flat is scene/example.ts, with one camera spot; the
// cabin is here.
import type { Scene } from '../../plan-editor/protocol';
import { buildScene, type Pt } from './scene/build';
import { EXAMPLE_SCENE } from './scene/example';

/** A place the tab lists: its name, when it was last visited (null: not yet), and its plan. */
export interface Place {
  id: string;
  name: string;
  /** Epoch seconds of the last visit, or null when nobody has been there yet. */
  lastVisit: number | null;
  /** When anyone was last seen there by its fixed cameras, apart from visits. */
  lastActivity?: number | null;
  /** The plan, as a Pascal scene; null before the place has one, or when the stored plan couldn't be read. */
  scene: Scene | null;
  /** Why the place's plan couldn't be read, when it couldn't. */
  planError?: string;
  /** The revision whose frame the plan is drawn in (what a save of it is sent with). */
  planRevision?: string | null;
  /** The newest machine plan, while the owner's holds over it, when it reads as a scene. */
  machineScene?: Scene | null;
  /** Whose plan it is: the machine's (with when its revision was made) or the owner's. */
  origin?: { source: 'owner' | 'machine'; at: number | null; machineNewer: boolean };
  /** Shown as an example, not something the house mapped. */
  example: boolean;
}

const rect = (x: number, y: number, w: number, h: number): Pt[] => [
  [x, y],
  [x + w, y],
  [x + w, y + h],
  [x, y + h],
];

/** A cabin: one room for living and cooking, with a toilet and a store beside it. */
export const CABIN_SCENE = buildScene({
  height: 2.4,
  walls: [
    { from: [0, 0], to: [6.2, 0], thickness: 0.2 },
    { from: [6.2, 0], to: [6.2, 4], thickness: 0.2 },
    { from: [6.2, 4], to: [0, 4], thickness: 0.2 },
    { from: [0, 4], to: [0, 0], thickness: 0.2 },
    { from: [5, 0], to: [5, 4], thickness: 0.1 },
    { from: [5, 2], to: [6.2, 2], thickness: 0.1 },
  ],
  openings: [
    { kind: 'door', at: [0.9, 4], width: 0.9 },
    { kind: 'door', at: [5, 1.5], width: 0.7 },
    { kind: 'door', at: [5, 2.5], width: 0.7 },
    { kind: 'window', at: [2.5, 0], width: 2.4 },
    { kind: 'window', at: [3.4, 4], width: 2 },
    { kind: 'window', at: [0, 2], width: 1.2 },
    { kind: 'window', at: [6.2, 1], width: 0.6 },
  ],
  rooms: [
    { id: 'living', name: 'Living and kitchen', outline: rect(0, 0, 5, 4) },
    { id: 'wc', name: 'Toilet', outline: rect(5, 0, 1.2, 2) },
    { id: 'store', name: 'Store', outline: rect(5, 2, 1.2, 2) },
  ],
});

export const EXAMPLE_PLACES: Place[] = [
  { id: 'example-flat', name: 'Example flat', lastVisit: null, scene: EXAMPLE_SCENE, example: true },
  { id: 'example-cabin', name: 'Example cabin', lastVisit: null, scene: CABIN_SCENE, example: true },
];
