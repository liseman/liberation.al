import { describe, expect, it } from 'vitest';
import { lookedLine, plainWarnings, verdict } from './sure';

describe('plainWarnings', () => {
  it('rewrites known warnings once each, keeps unknown ones, and drops blanks', () => {
    expect(
      plainWarnings([
        'Recorded frames are sparse VLM observations, not verified events.',
        '1 gate-skipped/error rows excluded: these are not fresh visual observations.',
        'Recorded frames are sparse VLM observations, again.',
        '  ',
        42,
        'A new caveat.',
      ]),
    ).toEqual([
      'Based on camera snapshots every few minutes, so short moments can be missed.',
      '1 snapshot was skipped or unreadable and left out.',
      'A new caveat.',
    ]);
  });
});

describe('lookedLine', () => {
  it('counts the sources biggest first, with an Oxford comma, and names unknown kinds as given', () => {
    expect(lookedLine(24, { transcript: 1, camera: 20, journal: 3 })).toBe(
      'Looked at 24 sources: 20 camera notes, 3 journal entries, and 1 transcript line',
    );
    expect(lookedLine(2, { camera: 1, radar: 1 })).toBe('Looked at 2 sources: 1 camera note and 1 radar');
    expect(lookedLine(1, {})).toBe('Looked at 1 source');
  });
});

describe('verdict', () => {
  const ALWAYS = [
    'Recorded frames are sparse VLM observations, not verified events.',
    'Transcripts are sampled ASR, not a complete record.',
  ];
  it('is fairly sure with only the caveats every report has, folded', () => {
    expect(verdict('ok', ALWAYS)).toEqual({
      sure: 'Fairly sure',
      why: null,
      rest: [
        'Based on camera snapshots every few minutes, so short moments can be missed.',
        'Who said what comes from automatic transcripts, and can be wrong.',
      ],
    });
  });
  it("is partly sure when snapshots couldn't be read, folding the unreadable line into one count", () => {
    const v = verdict('partial', [
      ...ALWAYS,
      'Frame decode or local VLM unavailable for some rows.',
      '468 gate-skipped/error rows excluded: these are not fresh visual observations.',
    ]);
    expect(v.sure).toBe('Partly sure');
    expect(v.why).toBe("468 snapshots couldn't be read, so some of it is missing.");
    expect(v.rest).toHaveLength(2);
    expect(verdict('ok', ['1 gate-skipped/error rows excluded.']).why).toBe(
      "1 snapshot couldn't be read, so some of it is missing.",
    );
  });
  it("leads with a caveat it doesn't know, and says a partial answer without one", () => {
    expect(verdict('ok', ['Something new the server says.', ...ALWAYS])).toMatchObject({
      sure: 'Partly sure',
      why: 'Something new the server says.',
    });
    expect(verdict('partial', ALWAYS)).toMatchObject({
      sure: 'Partly sure',
      why: "Some of what it needed wasn't there.",
    });
  });
});
