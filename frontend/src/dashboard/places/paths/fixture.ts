// Example positions for the example flat (places/examples.ts), until spatial memory has real ones to show: two people
// walking between rooms and a dog about the living room, as footprints, and a cat asleep in the bedroom, known only by
// its zone. Generated for any window, so the scrubber has something to move through.
import type { Point, Positions } from '../api';

type Leg = [number, number]; // a point on the plan, metres

// Waypoints through the example flat's rooms (its plan: bedroom, hall, bath on the left; living, kitchen, study).
const DANA: Leg[] = [
  [1.2, 4.8],
  [2.6, 4.0],
  [4.5, 2.0],
  [6.5, 1.6],
  [6.8, 3.5],
  [5.0, 5.5],
  [6.5, 5.5],
  [5.5, 7.8],
  [4.0, 7.4],
  [5.0, 5.4],
  [3.5, 2.5],
  [1.5, 4.6],
];
const SAM: Leg[] = [
  [6.0, 7.8],
  [5.0, 7.2],
  [4.2, 5.2],
  [6.5, 5.0],
  [7.0, 2.5],
  [4.2, 1.2],
  [3.6, 3.8],
  [5.6, 7.6],
];
const BISCUIT: Leg[] = [
  [4.0, 3.5],
  [5.5, 2.5],
  [7.0, 3.6],
  [6.0, 1.0],
  [4.4, 1.6],
  [5.2, 3.8],
  [6.8, 2.4],
];

/** Points along waypoints from start to end, one every `step` seconds, with a little wander. */
function walk(legs: Leg[], start: number, end: number, room: (x: number, y: number) => string, step = 30): Point[] {
  const n = Math.max(2, Math.floor((end - start) / step));
  const out: Point[] = [];
  for (let i = 0; i <= n; i++) {
    const f = (i / n) * (legs.length - 1);
    const k = Math.min(legs.length - 2, Math.floor(f));
    const [ax, ay] = legs[k] as Leg;
    const [bx, by] = legs[k + 1] as Leg;
    const u = f - k;
    const x = ax + (bx - ax) * u + 0.12 * Math.sin(i * 1.7);
    const y = ay + (by - ay) * u + 0.12 * Math.cos(i * 1.3);
    out.push({ t: start + (i / n) * (end - start), x, y, room: room(x, y), confidence: 0.8, source: 'fixed-camera' });
  }
  return out;
}

/** The example flat's room at a point (its rects, plan/examples.ts). */
function flatRoom(x: number, y: number): string {
  if (x < 3) return y < 3.6 ? 'bedroom' : y < 6 ? 'hall' : 'bath';
  return y < 4.4 ? 'living' : y < 6.6 ? 'kitchen' : 'study';
}

export function examplePositions(place: string, start: number, end: number): Positions {
  const span = end - start;
  return {
    place,
    start,
    end,
    entities: [
      { id: 'person:dana', name: 'Dana', kind: 'person' },
      { id: 'person:sam', name: 'Sam', kind: 'person' },
      { id: 'pet:biscuit', name: 'Biscuit', kind: 'pet' },
      { id: 'pet:miso', name: 'Miso', kind: 'pet' },
    ],
    tracks: [
      { entity: 'person:dana', points: walk(DANA, start, end, flatRoom) },
      { entity: 'person:sam', points: walk(SAM, start + span * 0.35, end, flatRoom) },
      { entity: 'pet:biscuit', points: walk(BISCUIT, start, start + span * 0.8, flatRoom, 45) },
    ],
    rooms: [{ entity: 'pet:miso', room: null, name: 'Bedroom', start, end, confidence: 0.9, source: 'zone' }],
  };
}
