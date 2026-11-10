import { afterEach, describe, expect, it, vi } from 'vitest';
import { takeLiveObject } from '../../objects/handoff';
import type { BoxPick } from '../camera/boxes';
import { follow, relabel } from './Stage';

const at = { left: 0, top: 0, width: 10, height: 10 };

describe('face boxes on the large camera (#352)', () => {
  afterEach(() => history.replaceState(null, '', '/'));

  it("labels an unknown face Who's this? and leaves a named one and objects as they are", () => {
    const box = [0.1, 0.1, 0.2, 0.2];
    expect(relabel({ id: 'f1', box, label: 'Face 4', kind: 'face', tone: 'unknown' }).label).toBe("Who's this?");
    expect(relabel({ id: 'f2', box, label: 'Kara', kind: 'face', tone: 'known' }).label).toBe('Kara');
    expect(relabel({ id: 'o1', box, label: 'cup', kind: 'object' }).label).toBe('cup');
  });

  it("opens a named face's card, and Who's this? for an unknown one", () => {
    const face = (label: string, tone: 'known' | 'unknown'): BoxPick => ({
      id: 'f',
      box: [0, 0, 1, 1],
      label,
      kind: 'face',
      tone,
      ...at,
    });
    follow(face('Kara Lee', 'known'));
    expect(location.pathname + location.hash).toBe('/people#card=Kara%20Lee');
    follow(face("Who's this?", 'unknown'));
    expect(location.pathname + location.hash).toBe('/people#whos-this');
  });
});

it('opens a clicked Live object crop with its suggested name while preserving the original pixels', () => {
  const canvas = document.createElement('canvas');
  canvas.width = 400;
  canvas.height = 200;
  const drawImage = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage,
  } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,AAAA');
  follow({ id: 'o1', kind: 'object', label: 'cup', box: [0.25, 0.25, 0.75, 0.75], ...at }, 'cam2', canvas);
  expect(location.pathname).toBe('/objects');
  expect(drawImage).toHaveBeenCalledWith(canvas, 90, 45, 220, 110, 0, 0, 220, 110);
  expect(takeLiveObject()).toMatchObject({ camera: 'cam2', name: 'cup' });
  vi.restoreAllMocks();
});

it('preserves the clicked unknown face group, track, camera, and its own capture time through relabeling', async () => {
  const { liveBoxes } = await import('../camera/boxes');
  const [face] = liveBoxes({
    cam: 'cam1',
    ts: 105,
    face_ts: 101,
    faces: [{ track: 'run:cam1:9', name: 'Face 5', box: [0.1, 0.2, 0.3, 0.4] }],
  });
  expect(face).toBeDefined();
  if (!face) throw new Error('missing fixture face');
  expect(relabel(face).faceSource).toEqual({ camera: 'cam1', at: 101, track: 'run:cam1:9', group: 5 });
  expect(relabel(face).label).toBe("Who's this?");
});

// An animal's box on Live's large view opens Pets, never an Objects draft.
describe("following a large-view animal's box", () => {
  afterEach(() => history.replaceState(null, '', '/'));
  const animal = (label: string): BoxPick => ({ id: 'o', box: [0.1, 0.1, 0.3, 0.3], label, kind: 'object', ...at });
  const PETS = { pets: [{ id: '1', name: 'Bess', species: 'dog' }], species: ['cat', 'dog'] };

  it("opens a taught pet's own entry on Pets", () => {
    follow(animal('Bess'), 'cam0', null, PETS);
    expect(location.pathname + location.hash).toBe('/pets#pet=Bess');
  });

  it("opens Pets for an animal's species", () => {
    follow(animal('dog'), 'cam0', null, PETS);
    expect(location.pathname + location.hash).toBe('/pets#species=dog');
  });

  it('still takes any other object to an Objects draft, which needs the displayed picture', () => {
    expect(() => follow(animal('coffee mug'), 'cam0', null, PETS)).toThrow('No displayed picture is available.');
    expect(location.pathname).toBe('/'); // the Objects path, not Pets
  });
});
