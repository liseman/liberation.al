import { expect, it } from 'vitest';
import type { Place } from './examples';
import { namesWords, walkNeed } from './walks';

const place = (over: Partial<Place>): Place => ({
  id: 'p',
  name: 'P',
  lastVisit: null,
  plan: null,
  example: false,
  ...over,
});

it.each([
  ['never walked', { walks: 0, unwalkedRooms: [{ id: 'den', name: 'Den' }] }, { kind: 'place' }],
  ['walks missing from the reply', {}, { kind: 'place' }],
  [
    'walked with rooms left',
    { walks: 1, unwalkedRooms: [{ id: 'den', name: 'Den' }] },
    { kind: 'rooms', rooms: ['Den'] },
  ],
  ['walked everywhere', { walks: 1, unwalkedRooms: [] }, null],
  ["walked, rooms can't be told", { walks: 1, unwalkedRooms: null }, null],
  ['an example', { example: true }, null],
] as [string, Partial<Place>, unknown][])('%s', (_, over, need) => {
  expect(walkNeed(place(over))).toEqual(need);
});

it('lists room names with the Oxford comma', () => {
  expect(namesWords(['Den'])).toBe('Den');
  expect(namesWords(['Den', 'Hall'])).toBe('Den and Hall');
  expect(namesWords(['Den', 'Hall', 'Study'])).toBe('Den, Hall, and Study');
});
