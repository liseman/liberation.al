import { describe, expect, it } from 'vitest';
import { calendarStep, kinds, landingDay, stepDay } from './days';

const DAYS = ['2026-10-20', '2026-10-24', '2026-10-25'];

describe('stepDay', () => {
  it('skips the days with nothing recorded', () => {
    expect(stepDay(DAYS, '2026-10-24', -1)).toBe('2026-10-20');
    expect(stepDay(DAYS, '2026-10-24', 1)).toBe('2026-10-25');
    expect(stepDay(DAYS, '2026-10-22', 1)).toBe('2026-10-24'); // from an empty day too
  });

  it('has nowhere to go past the first or last recorded day', () => {
    expect(stepDay(DAYS, '2026-10-20', -1)).toBeNull();
    expect(stepDay(DAYS, '2026-10-28', 1)).toBeNull();
  });

  it('steps one calendar day while the recorded days are unknown', () => {
    expect(stepDay(undefined, '2026-10-24', -1)).toBe('2026-10-23');
    expect(stepDay(undefined, '2026-10-31', 1)).toBe('2026-11-01');
  });
});

describe('landingDay', () => {
  it('opens the latest recorded day when the day has nothing', () => {
    expect(landingDay(DAYS, '2026-10-28')).toBe('2026-10-25');
  });

  it('stays on a day with recordings, or when nothing is recorded', () => {
    expect(landingDay(DAYS, '2026-10-24')).toBeNull();
    expect(landingDay([], '2026-10-28')).toBeNull();
  });

  it('never opens a day after the empty one while an earlier day has something', () => {
    expect(landingDay(DAYS, '2026-10-22')).toBe('2026-10-20');
    expect(landingDay(DAYS, '2026-10-01')).toBe('2026-10-25'); // the clock is behind every recording
  });
});

it('steps the calendar across a month and a year', () => {
  expect(calendarStep('2026-12-31', 1)).toBe('2027-01-01');
  expect(calendarStep('2026-03-01', -1)).toBe('2026-02-28');
});

it('names what a day has as a list', () => {
  expect(kinds({ day: 'd', video: true, audio: true, transcript: true })).toBe('video, room audio, and transcript');
  expect(kinds({ day: 'd', video: false, audio: true, transcript: true })).toBe('room audio and transcript');
});
