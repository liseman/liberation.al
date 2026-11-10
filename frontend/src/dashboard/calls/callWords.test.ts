import { describe, expect, it } from 'vitest';
import type { Call, Person } from '../people/api';
import { callMeta, callTitle, dayHeading } from './callWords';

const people: Person[] = [
  { name: 'Dana', voices: [], number: '+1 415 555 0123' },
  { name: 'Fay', voices: [], number: null },
];
const call = (more: Partial<Call>): Call => ({ day: '2026-11-03', dir: 'c', ...more });

describe('callTitle', () => {
  it.each<[string, Partial<Call>, string]>([
    [
      'a placed call to someone by name',
      { direction: 'out', to: { name: 'Dana', number: '+14155550123' } },
      'Called Dana',
    ],
    [
      'a typed number that is someone in People',
      { direction: 'out', to: { name: null, number: '+14155550123' } },
      'Called Dana',
    ],
    [
      'a typed number nobody has',
      { direction: 'out', to: { name: null, number: '+14155550177' } },
      'Called +14155550177',
    ],
    ['a placed call with no target saved', { direction: 'out', to: null }, 'Called someone'],
    ['an answered call from someone in People', { direction: 'in', from: '+14155550123' }, 'Dana called'],
    ['an answered call from a number nobody has', { direction: 'in', from: '+14155550177' }, '+14155550177 called'],
    ['an answered call with no number', { direction: 'in', from: null }, 'Someone called'],
    ['an older call, which way unknown', { direction: 'unknown', from: '+14155550123' }, 'Dana'],
  ])('%s', (_case, more, title) => {
    expect(callTitle(call(more), people)).toBe(title);
  });
});

describe('callMeta', () => {
  it('says the time in 24 hours, the length as people say it, and who spoke, without turns', () => {
    expect(
      callMeta(call({ started: '2026-11-03T17:40:00', durationSeconds: 182, turns: 14, speakers: { Dana: {} } })),
    ).toBe('17:40 · 3 min · with Dana');
    expect(callMeta(call({ durationSeconds: 4 }))).toBe('4 s');
    expect(callMeta(call({ durationSeconds: 48, speakers: { Sam: {}, Alice: {} } }), 'Sam called')).toBe(
      '48 s · with Alice',
    );
  });
});

describe('dayHeading', () => {
  const now = new Date('2026-11-06T12:00:00');
  it('says Today and Yesterday, and a year only when it is not this one', () => {
    expect(dayHeading('2026-11-06', now)).toBe('Today');
    expect(dayHeading('2026-11-05', now)).toBe('Yesterday');
    expect(dayHeading('2026-11-03', now)).not.toMatch(/2026/);
    expect(dayHeading('2025-12-30', now)).toMatch(/2025$/);
  });
});
