import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hashTiming, ReviewEngine } from './engine';
import type { Chip } from './types';

beforeEach(() => {
  hashTiming.everyMs = 0; // each link write lands at once here
});

const DAY = '2026-10-27';
const AT = new Date(2026, 9, 27, 10, 0, 15).getTime() / 1000;
const PLACE = `cam=patio&cameras=patio&day=${DAY}&t=${AT}`;
const STATE = { route: 'review', source: 'people-test' };

const CHIPS: Chip[] = [
  { id: 'person:alice', kind: 'person', label: 'Alice', spans: [[AT, AT + 60]], total_s: 60, count: 1 },
  { id: 'person:dana', kind: 'person', label: 'Dana', spans: [[AT, AT + 60]], total_s: 60, count: 1 },
  { id: 'object:container', kind: 'object', label: 'Container', spans: [[AT, AT + 60]], total_s: 60, count: 1 },
  { id: 'event:meals', kind: 'event', label: 'Meals', spans: [[AT, AT + 60]], total_s: 60, count: 1 },
  { id: 'person:speaker-12', kind: 'person', label: 'Speaker 12', spans: [[AT, AT + 60]], total_s: 60, count: 1 },
  { id: 'person:face-4', kind: 'person', label: 'Face 4', spans: [[AT, AT + 60]], total_s: 60, count: 1 },
  { id: 'person:unknown', kind: 'person', label: 'Unknown faces', spans: [[AT, AT + 60]], total_s: 60, count: 1 },
];

const engines: ReviewEngine[] = [];

function open(hash = `#${PLACE}`): ReviewEngine {
  history.replaceState(STATE, '', `/review${hash}`);
  const engine = new ReviewEngine({
    cams: ['patio', 'kitchen'],
    fetchNotes: async () => ({ cams: {} }),
    now: () => AT + 86400,
  });
  engines.push(engine);
  engine.start();
  return engine;
}

function arrive(engine: ReviewEngine, chips: Chip[] = CHIPS): void {
  engine.setChips(DAY, engine.ui.getState().gen, chips);
}

beforeEach(() => history.replaceState(null, '', '/review'));
afterEach(() => {
  for (const engine of engines.splice(0)) engine.stop();
});

describe('Review people selection and the URL', () => {
  it('writes raw comma-separated people in selection order without changing the place or adding history', () => {
    const engine = open();
    arrive(engine);
    const length = history.length;
    const replace = vi.spyOn(history, 'replaceState');
    const push = vi.spyOn(history, 'pushState');

    engine.toggleChip('person:alice');
    expect(location.hash).toBe(`#person=Alice&${PLACE}`);
    engine.toggleChip('person:dana');
    expect(location.hash).toBe(`#person=Alice,Dana&${PLACE}`);
    engine.toggleChip('person:alice');
    engine.toggleChip('person:alice');

    expect(location.hash).toBe(`#person=Dana,Alice&${PLACE}`);
    expect(engine.ui.getState().people).toEqual(['Dana', 'Alice']);
    expect(engine.ui.getState().person).toBe('Alice');
    expect(engine.ui.getState().day).toBe(DAY);
    expect(engine.t).toBe(AT);
    expect(engine.ui.getState().camera).toBe('patio');
    expect(engine.ui.getState().cameras).toEqual(['patio']);
    expect(history.state).toEqual(STATE);
    expect(history.length).toBe(length);
    expect(push).not.toHaveBeenCalled();
    expect(replace).toHaveBeenLastCalledWith(STATE, '', `/review#person=Dana,Alice&${PLACE}`);
  });

  it('falls back to the previous selected person and removes the hash field when the last one is deselected', () => {
    const engine = open();
    arrive(engine);
    engine.toggleChip('person:alice');
    engine.toggleChip('person:dana');
    engine.toggleChip('person:dana');

    expect(engine.ui.getState().people).toEqual(['Alice']);
    expect(engine.ui.getState().person).toBe('Alice');
    expect([...engine.ui.getState().sel]).toEqual(['person:alice']);
    expect(location.hash).toBe(`#person=Alice&${PLACE}`);

    engine.toggleChip('person:alice');
    expect(engine.ui.getState().people).toEqual([]);
    expect(engine.ui.getState().person).toBe('');
    expect([...engine.ui.getState().sel]).toEqual([]);
    expect(location.hash).toBe(`#${PLACE}`);
  });

  it('keeps object and event selections when pickPerson adds people or clears the people facet', () => {
    const engine = open();
    arrive(engine);
    engine.toggleChip('object:container');
    engine.toggleChip('event:meals');
    engine.pickPerson('Alice');
    engine.pickPerson('Dana');

    expect([...engine.ui.getState().sel]).toEqual(['object:container', 'event:meals', 'person:alice', 'person:dana']);
    expect(engine.ui.getState().person).toBe('Dana');
    expect(location.hash).toBe(`#person=Alice,Dana&${PLACE}`);

    engine.pickPerson('');
    expect([...engine.ui.getState().sel]).toEqual(['object:container', 'event:meals']);
    expect(engine.ui.getState().people).toEqual([]);
    expect(engine.ui.getState().person).toBe('');
    expect(location.hash).toBe(`#${PLACE}`);
  });

  it.each([
    ['person:speaker-12', 'Speaker 12'],
    ['person:face-4', 'Face 4'],
    ['person:unknown', 'Unknown faces'],
  ])('keeps anonymous chip %s out of the confirmed person hash', (id, label) => {
    const engine = open();
    arrive(engine);
    engine.toggleChip(id);
    expect(engine.ui.getState().sel.has(id)).toBe(true);
    expect(engine.ui.getState().people).toEqual([]);
    expect(engine.ui.getState().person).toBe('');
    expect(location.hash).toBe(`#${PLACE}`);

    engine.pickPerson(label);
    expect(engine.ui.getState().sel.has(id)).toBe(true);
    expect(engine.ui.getState().people).toEqual([]);
    expect(engine.ui.getState().person).toBe('');
    expect(location.hash).toBe(`#${PLACE}`);
  });

  it('applies every linked person when chips arrive and preserves an explicit deselection through refresh and remount', () => {
    const engine = open(`#person=Alice,Dana&${PLACE}`);
    expect(engine.ui.getState().people).toEqual(['Alice', 'Dana']);
    expect([...engine.ui.getState().sel]).toEqual([]);
    arrive(engine);
    expect([...engine.ui.getState().sel]).toEqual(['person:alice', 'person:dana']);

    engine.toggleChip('person:dana');
    engine.stop();
    engine.start();
    arrive(engine, [...CHIPS].reverse());
    expect(engine.ui.getState().people).toEqual(['Alice']);
    expect(engine.ui.getState().person).toBe('Alice');
    expect([...engine.ui.getState().sel]).toEqual(['person:alice']);
    expect(location.hash).toBe(`#person=Alice&${PLACE}`);

    engine.stop();
    const reopened = open(location.hash);
    arrive(reopened);
    expect(reopened.ui.getState().people).toEqual(['Alice']);
    expect([...reopened.ui.getState().sel]).toEqual(['person:alice']);
  });

  it.each(['hashchange', 'popstate'])('accepts an incoming single-person link on %s and keeps other facets', event => {
    const engine = open(`#person=Alice,Dana&${PLACE}`);
    arrive(engine);
    engine.toggleChip('object:container');
    history.replaceState(STATE, '', `/review#person=Dana&${PLACE}`);
    window.dispatchEvent(new Event(event));

    expect(engine.ui.getState().people).toEqual(['Dana']);
    expect(engine.ui.getState().person).toBe('Dana');
    expect([...engine.ui.getState().sel]).toEqual(['object:container', 'person:dana']);
    expect(engine.ui.getState().camera).toBe('patio');
    expect(engine.t).toBe(AT);
  });

  it('selects a linked person when a later neighbor chunk adds their chip beside an existing selection', () => {
    const engine = open(`#person=Alice,Dana&${PLACE}`);
    engine.setDayChips(
      DAY,
      CHIPS.filter(c => c.id === 'person:alice'),
    );
    expect([...engine.ui.getState().sel]).toEqual(['person:alice']);
    engine.setDayChips(
      '2026-10-28',
      CHIPS.filter(c => c.id === 'person:dana'),
    );
    expect([...engine.ui.getState().sel]).toEqual(['person:alice', 'person:dana']);
    expect(engine.ui.getState().people).toEqual(['Alice', 'Dana']);
  });
});
