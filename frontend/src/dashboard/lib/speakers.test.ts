import { describe, expect, it } from 'vitest';
import { badgeText, PALETTE, speakerName, spkColor } from './speakers';

describe('spkColor', () => {
  it('indexes the palette by a numeric id, wrapping around', () => {
    expect(spkColor(2)).toBe(PALETTE[2]);
    expect(spkColor('12')).toBe(PALETTE[2]);
  });

  it('gives the same colour every time for a named id', () => {
    expect(spkColor('dana')).toBe(spkColor('dana'));
    expect(PALETTE).toContain(spkColor('dana'));
  });

  it('greys out an unknown speaker', () => {
    expect(spkColor(null)).toBe('hsl(220 10% 70%)');
  });
});

describe('badgeText', () => {
  it.each([
    ['Speaker 4', '4'],
    ['Dana', 'Dana'],
    ['  Dana  ', 'Dana'],
    ['unknown', '?'],
    ['Multiple speakers', '?'],
    ['', '?'],
    [null, '?'],
    ['Speaker 4b', 'Speaker 4b'],
  ])('%j shows %j', (label, text) => {
    expect(badgeText(label)).toBe(text);
  });
});

describe('speakerName', () => {
  const speakers = {
    list: [],
    byId: { '4': { id: 4, label: 'Dana' } },
    merged: { '9': 4 },
  };

  it("names a line by its voice's current name, following merges", () => {
    expect(speakerName(speakers, { ts_start: 1, speaker_id: 9, speaker: 'Speaker 9' })).toBe('Dana');
  });

  it("falls back to the line's own name, then unknown", () => {
    expect(speakerName(speakers, { ts_start: 1, speaker_id: 7, speaker: 'Speaker 7' })).toBe('Speaker 7');
    expect(speakerName(speakers, { ts_start: 1 })).toBe('unknown');
  });
});
