import { expect, it } from 'vitest';
import { dayOf } from './day';

// Friday, November 6, 2026, 7 PM.
const NOW = new Date(2026, 10, 6, 19, 0);

it.each([
  ['today', '2026-11-06'],
  ['Yesterday', '2026-11-05'],
  ['monday', '2026-11-02'],
  ['wed', '2026-11-04'],
  ['friday', '2026-11-06'], // today is a Friday
  ['2026-11-03', '2026-11-03'],
  ['nov 3', '2026-11-03'],
  ['3 November', '2026-11-03'],
  ['11/3', '2026-11-03'],
  ['dec 25', '2025-12-25'], // this year's is still ahead: last year's
  ['feb 30', null], // no such day
  ['2026-11-07', null], // tomorrow
  ['tu', null], // too short to be a weekday
  ['Dana', null],
  ['', null],
])('reads %j as %j', (query, day) => {
  expect(dayOf(query, NOW)).toBe(day);
});

it("reads today's date as today before noon too, not last year's (the review)", () => {
  const morning = new Date(2026, 10, 6, 9, 0);
  expect(dayOf('nov 6', morning)).toBe('2026-11-06');
  expect(dayOf('11/6', morning)).toBe('2026-11-06');
  expect(dayOf('nov 7', morning)).toBe('2025-11-07'); // tomorrow's date is last year's
});
