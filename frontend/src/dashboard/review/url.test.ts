import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { readHash, reviewHash } from './url';

describe('Review place hashes', () => {
  it('keeps the old single-person URL and its day, time, camera, and playback fields', () => {
    const hash = reviewHash({
      person: 'Alice Smith',
      day: '2026-11-03',
      t: 123.456,
      cam: 'front',
      cameras: ['rear', 'rear'],
      play: true,
    });
    expect(hash).toBe('#person=Alice+Smith&cam=front&cameras=rear&play=1&day=2026-11-03&t=123.46');
    expect(readHash(hash)).toEqual({
      person: 'Alice Smith',
      people: ['Alice Smith'],
      day: '2026-11-03',
      t: 123.46,
      cam: 'front',
      cameras: ['rear'],
      play: true,
    });
  });

  it('preserves selection order and opens the last selected person', () => {
    const hash = reviewHash({ people: ['Alice', 'Dana', 'Alice'], person: 'ignored', day: '2026-11-03', t: 15 });
    expect(hash).toBe('#person=Alice,Dana&day=2026-11-03&t=15');
    expect(readHash(hash)).toMatchObject({ person: 'Dana', people: ['Alice', 'Dana'] });
  });

  it('distinguishes a literal comma in one name from the list delimiter before decoding', () => {
    const people = ['Smith, Alice', 'Dana + Lee', 'A&B % 鳥'];
    const hash = reviewHash({ people });
    expect(hash).toContain('Smith%2C+Alice,Dana+%2B+Lee,A%26B+%25+%E9%B3%A5');
    expect(readHash(hash)).toMatchObject({ people, person: people[2] });
    expect(readHash('#person=Smith%2C+Alice')).toMatchObject({ people: ['Smith, Alice'], person: 'Smith, Alice' });
  });

  it('can clear all people explicitly and keeps absent or invalid place fields empty', () => {
    expect(reviewHash({ people: [], person: 'Alice', t: Number.NaN })).toBe('#');
    expect(readHash('#person=&day=yesterday&t=oops')).toEqual({ person: '', people: [], day: undefined, t: undefined });
    expect(readHash('#')).toMatchObject({ person: '', people: [] });
    expect(readHash('#person')).toMatchObject({ person: '', people: [] });
  });

  it('round trips arbitrary name lists without treating encoded punctuation as separators', () => {
    fc.assert(
      fc.property(fc.array(fc.string({ minLength: 1 }), { maxLength: 8 }), people => {
        const unique = [...new Set(people)];
        expect(readHash(reviewHash({ people }))).toMatchObject({ people: unique, person: unique.at(-1) ?? '' });
      }),
    );
  });
});
