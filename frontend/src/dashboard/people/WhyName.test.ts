import { describe, expect, it } from 'vitest';
import type { Why } from './api';
import { explain } from './WhyName';

const base: Why = {
  name: 'Dana',
  subject: 'line',
  via: 'voice',
  basis: 'owner_tag',
  asserted_at: null,
  source: 'dashboard',
  confidence: 1,
  similarity: null,
  threshold: 0.55,
  voice: null,
  mark: {},
};

describe('explain', () => {
  it.each([
    [{ via: 'voice', basis: 'owner_tag' }, 'Its voice was named Dana.'],
    [{ via: 'voice', basis: 'cluster_merge' }, "Its voice was merged into Dana's."],
    [{ via: 'voice', basis: 'backfill_match' }, "Its voice was matched to Dana's."],
    [{ via: 'voice', basis: 'introduction' }, 'Its voice was named Dana from an introduction.'],
    [{ via: 'line', basis: 'correction' }, 'You corrected this line to Dana.'],
    [{ via: 'line', basis: 'owner_tag' }, 'You named this line Dana.'],
    [{ via: 'shot', basis: 'correction', subject: 'shot' }, 'You corrected this photo to Dana.'],
    [{ via: 'face', basis: null, subject: 'shot' }, "The face was recognized as Dana's when it was seen."],
    [{ via: 'heard', basis: null }, 'It was heard as Dana; nothing has named this line or its voice since.'],
    [
      { via: 'heard', basis: null, voice: { id: 1, label: 'Eve' } },
      "It was heard as Dana. Its voice is Eve's now, which doesn't rename lines already heard.",
    ],
    [{ via: 'detached', basis: null }, 'This line was taken off its voice.'],
    [{ via: 'hand', basis: null }, 'Dana was written on this line by hand, before names were kept in the log.'],
    [{ via: 'none', basis: null }, 'Nobody has named its voice yet.'],
  ] as [Partial<Why>, string][])('%o says %s', (w, said) => {
    expect(explain({ ...base, ...w }).said).toBe(said);
  });

  it('gives the match against what a match needs, and nothing without a score', () => {
    expect(explain({ ...base, similarity: 0.712 }).match).toBe('Voice match 0.71 (a match needs 0.55).');
    expect(explain({ ...base, subject: 'shot', similarity: 0.5, threshold: 0.45 }).match).toBe(
      'Face match 0.50 (a match needs 0.45).',
    );
    expect(explain(base).match).toBeNull();
  });

  it('says when the deciding assertion was made', () => {
    expect(explain({ ...base, asserted_at: 1_793_000_000 }).said).toMatch(/^Its voice was named Dana on .+\.$/);
  });
});
