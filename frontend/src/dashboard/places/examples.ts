// Example plans for the Places tab until spatial memory makes real ones. Each is a plan document in
// the vendored floorplan library's JSON form (src/vendor/floorplan/README.md): metres on wall centrelines, rooms that
// tile the footprint, openings on the walls between them. The flat also carries one of each of spatial memory's
// extensions (plan/ext.ts) to show how they draw. Both lint clean (plan/lint.ts).

/** A place the tab lists: its name, when it was last visited (null: not yet), and its plan document. */
export interface Place {
  id: string;
  name: string;
  /** Epoch seconds of the last visit, or null when nobody has been there yet. */
  lastVisit: number | null;
  /** When anyone was last seen there by its fixed cameras, apart from visits. */
  lastActivity?: number | null;
  /** The plan document, as floorplan() takes it, with spatial memory's extensions (plan/ext.ts); null before the
   *  place has a plan. */
  plan: Record<string, unknown> | null;
  /** Why the place's plan couldn't be read, when it couldn't. */
  planError?: string;
  /** The plan as its DSL text (a real place's), what the editor works on. */
  planText?: string;
  /** The revision whose frame the plan is drawn in (what a correction of it is saved against). */
  planRevision?: string | null;
  /** The newest machine plan's text, while the owner's correction holds over it. */
  machinePlanText?: string | null;
  /** Whose plan it is: the machine's (with when its revision was made) or the owner's correction. */
  origin?: { source: 'owner' | 'machine'; at: number | null; machineNewer: boolean };
  /** How many walkthroughs went into it, and its plan's rooms no LiDAR walk covered (null: can't tell), #3. */
  walks?: number;
  unwalkedRooms?: { id: string; name: string }[] | null;
  /** Shown as an example, not something the house mapped. */
  example: boolean;
}

const FLAT = {
  title: 'Example flat',
  walls: { exterior: 0.3, partition: 0.12 },
  rooms: {
    bedroom: { name: 'Bedroom', kind: 'bedroom', rect: [0, 0, 3, 3.6] },
    hall: { name: 'Hall', kind: 'hall', rect: [0, 3.6, 3, 2.4] },
    bath: { name: 'Bathroom', kind: 'bath', rect: [0, 6, 3, 3] },
    living: { name: 'Living room', kind: 'living', rect: [3, 0, 5, 4.4] },
    kitchen: { name: 'Kitchen', kind: 'kitchen', rect: [3, 4.4, 5, 2.2] },
    study: { name: 'Study', kind: 'office', rect: [3, 6.6, 5, 2.4] },
  },
  openings: [
    {
      type: 'door',
      between: ['exterior', 'hall'],
      on: { room: 'hall', side: 'west' },
      width: 0.9,
      hinge: 'start',
      swingInto: 'hall',
      entrance: true,
    },
    { type: 'door', between: ['hall', 'bedroom'], width: 0.8, hinge: 'start', swingInto: 'bedroom' },
    { type: 'cased', between: ['hall', 'living'], width: 0.6 },
    { type: 'cased', between: ['living', 'kitchen'], width: 2.4 },
    { type: 'door', between: ['hall', 'bath'], width: 0.7, hinge: 'end', swingInto: 'bath' },
    { id: 'study_door', type: 'door', between: ['kitchen', 'study'], width: 0.8, hinge: 'end', swingInto: 'study' },
    { type: 'window', between: ['exterior', 'living'], on: { room: 'living', side: 'north' }, width: 2.4 },
    { type: 'window', between: ['exterior', 'living'], on: { room: 'living', side: 'east' }, width: 1.6 },
    { type: 'window', between: ['exterior', 'bedroom'], on: { room: 'bedroom', side: 'north' }, width: 1.2 },
    { type: 'window', between: ['exterior', 'kitchen'], on: { room: 'kitchen', side: 'east' }, width: 1.2 },
    { type: 'window', between: ['exterior', 'study'], on: { room: 'study', side: 'east' }, width: 1.2 },
    { type: 'window', between: ['exterior', 'bath'], on: { room: 'bath', side: 'west' }, width: 0.6 },
  ],
  'x-anchor': { frame: 'place:example-flat', yaw: 0, origin: [0, 0, 0] },
  'x-history': [{ n: 1, at: '2026-11-01T18:20:00Z', model: 'baseline', visits: ['visit-1'] }],
  'x-revision': { n: 2, at: '2026-11-05T09:10:00Z', model: 'roomformer', visits: ['visit-1', 'visit-2'] },
  'x-provenance': {
    'wall:w12': { source: 'owner' },
    'wall:w8': { source: 'machine', confidence: 0.42 },
    'wall:w19': { source: 'machine', confidence: 0.5 },
    'room:study': { source: 'machine', confidence: 0.7, partial: true },
    'opening:study_door': { source: 'machine', confidence: 0.4 },
  },
  'x-unexplored': { study_back: { name: 'Back of the study', rect: [5.6, 7.8, 2.4, 1.2] } },
  'x-cameras': { hall: { at: [0.4, 4.2], height: 2.2, yaw: 90, fov: 100 } },
  'x-objects': {
    sofa: { label: 'Sofa', at: [5.5, 1.1], size: [2.2, 0.9], source: 'owner' },
    table: { label: 'Table', at: [7, 5.4], size: [1, 0.8], source: 'machine', confidence: 0.5 },
  },
};

// The library's own cabin fixture (fixtures/cabin.json upstream), with its rooms named in English.
const CABIN = {
  title: 'Example cabin',
  walls: { exterior: 0.2, partition: 0.1 },
  rooms: {
    living: { name: 'Living and kitchen', kind: 'living', rect: [0, 0, 5, 4] },
    wc: { name: 'Toilet', kind: 'wc', rect: [5, 0, 1.2, 2] },
    store: { name: 'Store', kind: 'storage', rect: [5, 2, 1.2, 2] },
  },
  outdoor: { deck: { name: 'Deck', rect: [0, 4, 5, 2] } },
  openings: [
    { type: 'door', between: ['deck', 'living'], at: [0.9, 4], width: 0.9, hinge: 'start', swingInto: 'living' },
    {
      type: 'door',
      between: ['living', 'wc'],
      position: { from: 'end', distance: 0.5 },
      width: 0.7,
      hinge: 'end',
      swingInto: 'wc',
    },
    {
      type: 'door',
      between: ['living', 'store'],
      position: { from: 'start', distance: 0.5 },
      width: 0.7,
      hinge: 'start',
      swingInto: 'store',
    },
    {
      type: 'window',
      between: ['exterior', 'living'],
      on: { room: 'living', side: 'north' },
      position: 2.5,
      width: 2.4,
    },
    { type: 'window', between: ['deck', 'living'], on: { room: 'living', side: 'south' }, position: 3.4, width: 2 },
    { type: 'window', between: ['exterior', 'living'], on: { room: 'living', side: 'west' }, width: 1.2 },
    { type: 'window', between: ['exterior', 'wc'], on: { room: 'wc', side: 'east' }, width: 0.6 },
  ],
};

export const EXAMPLE_PLACES: Place[] = [
  { id: 'example-flat', name: 'Example flat', lastVisit: null, plan: FLAT, example: true },
  { id: 'example-cabin', name: 'Example cabin', lastVisit: null, plan: CABIN, example: true },
];
