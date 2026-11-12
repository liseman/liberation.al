import { describe, expect, it } from 'vitest';
import type { TranscriptEntry } from '../live/types';
import type { Vitals } from '../live/vitals';
import type { Person } from '../people/api';
import type { PetsView } from '../people/petLinks';
import { HEARD_S, hereNow } from './HereNow';

const NOW = 1_793_000_000;
const clock = (ts: number) =>
  new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
const person = (p: Omit<Person, 'voices'>): Person => ({ voices: [], ...p }) as Person;
const line = (speaker: string, ago: number) => ({ ts_start: NOW - ago, speaker, text: 'hi' }) as TranscriptEntry;

describe('who is here now', () => {
  it('lists who the cameras see, the earliest arrival first, with where and since when', () => {
    const people: Person[] = [
      person({ name: 'Kara', here_now: { cam: 'cam0', camera_name: 'front camera', since: NOW - 60 } }),
      person({ name: 'Luca', here_now: { cam: 'cam1', camera_name: null, since: NOW - 600 } }),
    ];
    expect(hereNow(people, [], NOW).map(h => [h.name, h.where])).toEqual([
      ['Luca', `Cam1, since ${clock(NOW - 600)}`],
      ['Kara', `Front camera, since ${clock(NOW - 60)}`],
    ]);
  });

  it('adds who was heard in the last minute and is not in view, by a name People knows', () => {
    const people: Person[] = [
      person({ name: 'Kara', here_now: { cam: 'cam0', since: NOW - 30 } }),
      person({ name: 'Luca' }),
      person({ name: 'Dana' }),
    ];
    const lines = [line('Dana', HEARD_S + 5), line('Speaker 4', 3), line('kara', 2), line('Luca', 1)];
    expect(hereNow(people, lines, NOW).map(h => [h.name, h.where])).toEqual([
      ['Kara', `Cam0, since ${clock(NOW - 30)}`],
      ['Luca', 'Heard just now'],
    ]);
  });

  it('puts the named pets a camera sees after the people, with a photo of them and a link to their card', () => {
    const people: Person[] = [
      person({ name: 'Kara', here_now: { cam: 'cam0', since: NOW - 30 } }),
      person({ name: 'Luca' }),
    ];
    const pets: PetsView = {
      species: ['cat', 'dog'],
      pets: [
        { id: 'p1', name: 'Bess', species: 'dog' },
        { id: 'p2', name: 'Mia', species: 'cat', photo: 'run:cat:4' },
        { id: 'p3', name: 'Pip', species: 'cat' },
      ],
      here: {
        p2: { cam: 'cam1', camera_name: 'desk', since: NOW - 300, track: 'run:cat:7' },
        p1: { cam: 'cam0', camera_name: '', since: NOW - 20, track: 'run:dog:2' },
      },
    };
    const here = hereNow(people, [line('Luca', 1)], NOW, pets);
    expect(here.map(h => [h.name, h.where, h.seen])).toEqual([
      ['Kara', `Cam0, since ${clock(NOW - 30)}`, true],
      ['Mia', `Desk, since ${clock(NOW - 300)}`, true],
      ['Bess', `Cam0, since ${clock(NOW - 20)}`, true],
      ['Luca', 'Heard just now', false],
    ]);
    // Mia's photo is the one chosen on her card; Bess has none, so it's a look at her now
    expect(here[1]?.photo).toContain('track=run%3Acat%3A4');
    expect(here[2]?.photo).toContain('track=run%3Adog%3A2');
    expect(here[1]?.href).toBe('/pets#pet=Mia');
    expect(here[0]?.href).toBe('/people#card=Kara');
  });

  it('shows a heart rate only for the people a close camera has a confident recent reading for', () => {
    const people: Person[] = [
      person({ name: 'Kara', here_now: { cam: 'cam0', since: NOW - 30 } }),
      person({ name: 'Luca', here_now: { cam: 'cam0', since: NOW - 30 } }),
      person({ name: 'Dana' }),
    ];
    const vitals: Vitals = {
      at: NOW,
      // The server only sends readings it trusts and that are recent; the name is the one it keys by, case folded.
      people: { kara: { bpm: 72, confidence: 0.81, face_px: 160, at: NOW - 2 } },
    };
    const here = hereNow(people, [line('Dana', 5)], NOW, undefined, vitals);
    expect(here.map(h => [h.name, h.heart])).toEqual([
      ['Kara', 72],
      ['Luca', undefined],
      ['Dana', undefined],
    ]);
  });
});
