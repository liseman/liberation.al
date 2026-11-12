import { describe, expect, it } from 'vitest';
import type { Check, Learning } from './api';
import { checkText, learningText } from './PetCard';

const learning = (extra: Partial<Learning> = {}): Learning => ({
  tested: 20,
  right: 18,
  wrong: 0,
  unknown: 2,
  days: 7,
  start: 100,
  end: 200,
  automatic_percent: 90,
  references: 120,
  stranded: 0,
  no_photo: 0,
  ...extra,
});

describe('the recognition paragraph', () => {
  it('says one taught sighting with no saved picture in the singular, and what the count costs', () => {
    expect(learningText(learning({ no_photo: 1 }))).toContain(
      "1 taught sighting has no saved picture, so it can't be relearned if the recognition model changes.",
    );
  });

  it('says several taught sightings with no saved picture in the plural', () => {
    expect(learningText(learning({ no_photo: 3 }))).toContain(
      "3 taught sightings have no saved picture, so they can't be relearned if the recognition model changes.",
    );
  });

  it('plurals the earlier taught sightings that have no usable reference', () => {
    expect(learningText(learning({ stranded: 1 }))).toContain('1 earlier taught sighting has no usable reference.');
    expect(learningText(learning({ stranded: 30 }))).toContain('30 earlier taught sightings have no usable reference.');
  });

  it('plurals the usable teaching examples, the held-out sightings, and the days of the check', () => {
    expect(learningText(learning({ references: 1 }))).toContain('1 usable teaching example.');
    expect(learningText(learning({ references: 120 }))).toContain('120 usable teaching examples.');
    expect(learningText(learning({ tested: 1 }))).toContain('in 1 held-out sighting:');
    expect(learningText(learning({ tested: 20 }))).toContain('in 20 held-out sightings:');
    expect(learningText(learning({ automatic_percent: null, days: 1 }))).toContain(
      'No time-separated check is available within the last 1 day.',
    );
  });
});

describe('the recognition check line', () => {
  const check = (extra: Partial<Check> = {}): Check => ({
    species: 'cat',
    labeled: 5,
    right: 4,
    wrong: 1,
    unknown: 0,
    on: true,
    reason: null,
    ...extra,
  });

  it('plurals the labeled sightings the check ran on', () => {
    expect(checkText(check({ tested: 1 }))).toContain('Checked on 1 labeled sighting:');
    expect(checkText(check({ tested: 20 }))).toContain('Checked on 20 labeled sightings:');
  });
});
