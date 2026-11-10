// The example flat (../examples.ts) as a Pascal scene: the same six made-up rooms, their doors and windows, and
// the one camera spot: an example for the Places tab while the house has no plan of its own.
import { buildScene, type Pt } from './build';

const EXTERIOR = 0.3;
const PARTITION = 0.12;
const rect = (x: number, y: number, w: number, h: number): Pt[] => [
  [x, y],
  [x + w, y],
  [x + w, y + h],
  [x, y + h],
];

export const EXAMPLE_SCENE = buildScene({
  height: 2.4,
  walls: [
    { from: [0, 0], to: [8, 0], thickness: EXTERIOR },
    { from: [8, 0], to: [8, 9], thickness: EXTERIOR },
    { from: [8, 9], to: [0, 9], thickness: EXTERIOR },
    { from: [0, 9], to: [0, 0], thickness: EXTERIOR },
    { from: [3, 0], to: [3, 9], thickness: PARTITION },
    { from: [0, 3.6], to: [3, 3.6], thickness: PARTITION },
    { from: [0, 6], to: [3, 6], thickness: PARTITION },
    { from: [3, 4.4], to: [8, 4.4], thickness: PARTITION },
    { from: [3, 6.6], to: [8, 6.6], thickness: PARTITION },
  ],
  openings: [
    { kind: 'door', at: [0, 4.8], width: 0.9 },
    { kind: 'door', at: [1.5, 3.6], width: 0.8 },
    { kind: 'cased', at: [3, 4], width: 0.6 },
    { kind: 'cased', at: [5.5, 4.4], width: 2.4 },
    { kind: 'door', at: [1.5, 6], width: 0.7 },
    { kind: 'door', at: [5.5, 6.6], width: 0.8 },
    { kind: 'window', at: [5.5, 0], width: 2.4 },
    { kind: 'window', at: [8, 2.2], width: 1.6 },
    { kind: 'window', at: [1.5, 0], width: 1.2 },
    { kind: 'window', at: [8, 5.5], width: 1.2 },
    { kind: 'window', at: [8, 7.8], width: 1.2 },
    { kind: 'window', at: [0, 7.5], width: 0.6 },
  ],
  rooms: [
    { id: 'bedroom', name: 'Bedroom', outline: rect(0, 0, 3, 3.6) },
    { id: 'hall', name: 'Hall', outline: rect(0, 3.6, 3, 2.4) },
    { id: 'bath', name: 'Bathroom', outline: rect(0, 6, 3, 3) },
    { id: 'living', name: 'Living room', outline: rect(3, 0, 5, 4.4) },
    { id: 'kitchen', name: 'Kitchen', outline: rect(3, 4.4, 5, 2.2) },
    { id: 'study', name: 'Study', outline: rect(3, 6.6, 5, 2.4) },
  ],
  extras: { version: 1, cameras: { hall: { at: [0.4, 4.2], height: 2.2, yaw: 90, fov: 100 } } },
});
