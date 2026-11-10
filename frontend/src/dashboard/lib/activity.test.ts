import { describe, expect, it } from 'vitest';
import { activityName } from './activity';

describe('activityName', () => {
  it.each([
    ['crossing', 'Walked past'], // a kind with a name of its own
    ['eat', 'Eating'],
    ['read', 'Read'], // an unnamed kind reads as a name beside "Eating", not a raw label
    ['sleep', 'Sleep'],
    ['watch tv', 'Watch TV'], // known acronyms keep their capitals
    ['watch_tv', 'Watch TV'],
    ['use_pc', 'Use PC'],
    ['play  the   piano', 'Play the piano'], // extra spaces fold
    ['Coffee break', 'Coffee break'], // a name already given its capitals keeps them
    ['tv', 'TV'],
    ['', ''],
  ])('%j reads as %j', (kind, name) => {
    expect(activityName(kind)).toBe(name);
  });
});
