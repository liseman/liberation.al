import { describe, expect, it } from 'vitest';
import { recordedLine, voiceName, zoneLabel, zones } from './controls';

describe('the You and the assistant controls', () => {
  it('names a Kokoro voice in words', () => {
    expect([voiceName('af_heart'), voiceName('bm_george'), voiceName('zf_xiao_bei'), voiceName('odd')]).toEqual([
      'Heart (American English, female)',
      'George (British English, male)',
      'Xiao bei (Mandarin, female)',
      'odd',
    ]);
  });

  it('says when the voice reference was recorded and how long it is', () => {
    const now = new Date(2026, 9, 29);
    const oct23 = new Date(2026, 9, 23, 15).getTime() / 1000;
    const lastYear = new Date(2025, 2, 1).getTime() / 1000;
    expect([
      recordedLine({ exists: true, seconds: 14.6, recorded: oct23 }, now),
      recordedLine({ exists: true, seconds: 11, recorded: lastYear }, now),
      recordedLine({ exists: false, seconds: 0, problem: 'missing' }, now),
      recordedLine({ exists: false, seconds: 0, problem: 'unreadable' }, now),
    ]).toEqual([
      `Recorded ${new Date(oct23 * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}, 15 s`,
      `Recorded ${new Date(lastYear * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}, 11 s`,
      'Not recorded yet',
      "The saved reference can't be read. Record it again",
    ]);
  });

  it('lists UTC among the zones, and finds an old zone name by its new one', () => {
    expect(zones()).toContain('UTC');
    expect([zoneLabel('Asia/Calcutta'), zoneLabel('America/Los_Angeles')]).toEqual([
      'Asia/Calcutta (Kolkata)',
      'America/Los Angeles',
    ]);
  });

  it("says when the voice reference couldn't be loaded, not that it is loading", () => {
    expect([recordedLine(undefined), recordedLine(undefined, new Date(), true)]).toEqual([
      'Loading',
      "Couldn't load your voice reference",
    ]);
  });
});
