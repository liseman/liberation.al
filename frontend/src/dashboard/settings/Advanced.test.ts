import { describe, expect, it } from 'vitest';
import { matchesFilter, NOTE_MAX, sectionNote, sectionTitle } from './Advanced';
import type { Setting } from './model';
import { sameAsDefault } from './Row';

describe("Advanced's cards (#387)", () => {
  it('titles a config section in words', () => {
    expect([sectionTitle('presence'), sectionTitle('models.files.vad_silero')]).toEqual([
      'Presence',
      'Models / Files / Vad silero',
    ]);
  });

  it("notes a section by its comment's first sentence", () => {
    expect([
      sectionNote('How visits are counted. Each one is an episode, e.g. a walk-in.'),
      sectionNote('Naming people (people/identity.py). A tag can carry a role.'),
      sectionNote('No full stop at all'),
      sectionNote(''),
    ]).toEqual(['How visits are counted.', 'Naming people (people/identity.py).', 'No full stop at all', '']);
    const long = sectionNote(`Timeline: ${'encounters, the stretches each person was seen '.repeat(4)}and more.`);
    expect(long.length).toBeLessThanOrEqual(NOTE_MAX + 1);
    expect(long.endsWith('\u2026')).toBe(true);
    expect(long.startsWith('Timeline: encounters, the stretches')).toBe(true);
  });

  it("doesn't end a sentence at e.g., nor leave an aside open where it cuts", () => {
    expect(sectionNote('Paths (e.g. state) live here. More.')).toBe('Paths (e.g. state) live here.');
    const timeline = sectionNote(
      'Timeline (memory/timeline.py): encounters, the stretches of time each person was seen (face visits in recorded video) or heard.',
    );
    expect(timeline).toBe(
      'Timeline (memory/timeline.py): encounters, the stretches of time each person was seen\u2026',
    );
  });

  const setting = (s: Partial<Setting>): Setting => ({
    key: 'presence.gap_s',
    type: 'float',
    value: 30,
    default: 30,
    secret: false,
    description: '',
    ...s,
  });

  it('counts a number typed another way as its default', () => {
    expect([
      sameAsDefault(setting({}), '30.0'),
      sameAsDefault(setting({}), '3e1'),
      sameAsDefault(setting({}), '31'),
    ]).toEqual([true, true, false]);
    const list = setting({ type: 'list', default: ['USB Audio: - (hw:1,0)'] });
    expect(sameAsDefault(list, '["USB Audio: - (hw:1,0)"]')).toBe(true); // the draft holds it as JSON
  });

  it('filters a new feature by its name too', () => {
    expect(matchesFilter(setting({ key: 'x.shiny', type: 'bool', feature: 'Shiny things' }), 'shiny things')).toBe(
      true,
    );
  });
});
