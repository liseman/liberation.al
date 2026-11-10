import { describe, expect, it } from 'vitest';
import { petHref, petInHash } from './petLinks';

const VIEW = {
  pets: [
    { id: '1', name: 'Bess', species: 'dog' },
    { id: '2', name: 'Moe Meerkat', species: 'cat' },
  ],
  species: ['cat', 'dog', 'rabbit'],
};

// An animal word opens Pets, at the pet or for its species; anything else isn't Pets' to open.
describe('petHref', () => {
  it.each([
    ['a taught pet, by name whatever the case', 'moe meerkat', '/pets#pet=Moe%20Meerkat'],
    ['a species', 'dog', '/pets#species=dog'],
    ['a species in the plural', 'Dogs', '/pets#species=dog'],
    ['a species only the server lists', 'rabbit', '/pets#species=rabbit'],
    ['a species before /pets has answered', 'cat', '/pets#species=cat'],
  ])('sends %s to Pets', (_what, label, href) => {
    expect(petHref(label, label === 'cat' ? undefined : VIEW)).toBe(href);
  });

  it('leaves an object alone', () => {
    expect(petHref('coffee mug', VIEW)).toBeNull();
  });
});

describe('petInHash', () => {
  it.each([
    ['#pet=Moe%20Meerkat', { pet: 'Moe Meerkat' }],
    ['#species=Dog', { species: 'dog' }],
    ['#card=Sam', {}],
    ['', {}],
  ])('reads %s', (hash, wanted) => {
    expect(petInHash(hash)).toEqual(wanted);
  });
});
