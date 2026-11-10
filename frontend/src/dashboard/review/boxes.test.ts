import { describe, expect, it } from 'vitest';
import { asBox, boxesAt, faceToName, type ReviewBox, shownBoxes } from './boxes';
import type { Chip } from './types';

const b = (ts: number, id: string, extra: Partial<ReviewBox> = {}): ReviewBox => ({
  ts,
  id,
  kind: 'face',
  box: [0.1, 0.1, 0.2, 0.2],
  label: id,
  tone: 'known',
  name: id,
  ...extra,
});

describe('boxesAt', () => {
  it('keeps pet samples independent from face and object cadence', () => {
    const pet = b(100.7, 'pet:cat', { kind: 'pet', species: 'cat', label: 'Mia · cat' });
    const face = b(100.8, 'Dana');
    const object = b(98, 'Mug', { kind: 'object' });
    expect(boxesAt([object, pet, face], 101).map(box => box.id)).toEqual(['pet:cat', 'Dana', 'Mug']);
    expect(boxesAt([pet], 101.3)).toEqual([]); // a pet ends GAP_S after its last sample
    expect(asBox(pet)).toMatchObject({ kind: 'pet', label: 'Mia · cat' });
  });

  it("moves a person's body in a straight line between samples a tenth of a second apart", () => {
    const at = (ts: number, x: number) =>
      b(ts, 'person:7', { kind: 'person', track: '7', box: [x, 0.2, x + 0.2, 0.8] });
    const list = [at(100, 0.1), at(100.1, 0.2), at(100.2, 0.4)];
    expect(boxesAt(list, 100.05)[0]?.box).toEqual([0.15, 0.2, 0.35, 0.8].map(v => expect.closeTo(v, 6)));
    expect(boxesAt(list, 100.15)[0]?.box[0]).toBeCloseTo(0.3, 6);
    expect(boxesAt(list, 100.2)[0]?.box[0]).toBeCloseTo(0.4, 6);
  });

  it('ends a body at a gap instead of sliding it across the picture', () => {
    const at = (ts: number, x: number) =>
      b(ts, 'person:7', { kind: 'person', track: '7', box: [x, 0.2, x + 0.2, 0.8] });
    const list = [at(100, 0.1), at(102, 0.7)]; // the track went unseen for two seconds
    expect(boxesAt(list, 100.3)[0]?.box[0]).toBeCloseTo(0.1, 6); // held, not moved toward the later sample
    expect(boxesAt(list, 100.6)).toEqual([]);
    expect(boxesAt(list, 101.98)[0]?.box[0]).toBeCloseTo(0.7, 6); // back as soon as it's seen again
    expect(boxesAt(list, 99.9)).toEqual([]); // not before it was seen
  });

  it('draws a face only for a person who has no body box, so nobody gets two', () => {
    const body = b(100, 'person:7', { kind: 'person', track: '7', label: 'Dana' });
    const covered = b(100, 'face:dana', { track: '7' });
    const old = b(100, 'face:eli', { track: 'run:cam0:3', label: 'Eli' }); // an older recording: faces only
    expect(boxesAt([body, covered, old], 100.1).map(x => x.id)).toEqual(['person:7', 'face:eli']);
  });

  it('holds a face up to a second after its last sample, then lets it go', () => {
    const list = [b(100, 'Dana')];
    expect(boxesAt(list, 100.6).map(x => x.id)).toEqual(['Dana']);
    expect(boxesAt(list, 101.2)).toEqual([]);
    expect(boxesAt(list, 99.5)).toEqual([]); // not before it was seen
  });

  it('keeps each face on its own samples', () => {
    const list = [b(100, 'Dana'), b(100.5, 'Eli'), b(100.5, 'Kara')];
    expect(boxesAt(list, 100.7).map(x => x.id)).toEqual(['Dana', 'Eli', 'Kara']);
  });

  it('holds a thing longer, since object memory samples about every 5 s', () => {
    const list = [b(100, 'mug', { kind: 'object' })];
    expect(boxesAt(list, 104).map(x => x.id)).toEqual(['mug']);
    expect(boxesAt(list, 107)).toEqual([]);
  });

  it("holds the camera's labels from their grounding until its next one", () => {
    const sofa = b(100, 'label:sofa:0', { kind: 'label', tone: 'unknown', until: 110 });
    const lamp = b(100, 'label:lamp:1', { kind: 'label', tone: 'unknown', until: 110 });
    const later = b(110, 'label:sofa:0', { kind: 'label', tone: 'unknown', until: 125 });
    const list = [sofa, lamp, later];
    expect(boxesAt(list, 99.5)).toEqual([]); // not before it was grounded
    expect(boxesAt(list, 109.9)).toEqual([sofa, lamp]); // well past a thing's 6 s hold
    expect(boxesAt(list, 110)).toEqual([later]); // the next record takes over at once
    expect(boxesAt(list, 125)).toEqual([]);
  });

  it('takes things from their newest sample alongside the people', () => {
    const list = [b(100, 'mug', { kind: 'object' }), b(100.8, 'Dana'), b(100.9, 'face:1', { tone: 'unknown' })];
    expect(boxesAt(list, 101).map(x => x.id)).toEqual(['Dana', 'face:1', 'mug']);
  });
});

describe('faceToName', () => {
  const face = (ts: number, extra: Partial<ReviewBox> = {}) =>
    b(ts, 'face:#3', { tone: 'unknown', track: '7', group: 5, label: "Who's this?", ...extra });

  it("takes the track's newest unknown face with a group within a second before the frame", () => {
    const list = [face(99), face(99.6, { box: [0.3, 0.3, 0.4, 0.4] }), face(99.7, { track: '8' })];
    expect(faceToName(list, '7', 100)?.box).toEqual([0.3, 0.3, 0.4, 0.4]);
  });

  it('crops the face where it is drawn on the frame, between its samples', () => {
    const list = [face(100, { box: [0.1, 0.1, 0.2, 0.2] }), face(101, { box: [0.5, 0.1, 0.6, 0.2] })];
    const got = faceToName(list, '7', 100.5);
    expect(got?.box).toEqual([0.3, 0.1, 0.4, 0.2].map(v => expect.closeTo(v, 6)));
    expect(got?.ts).toBe(100); // still the sample's evidence: its time, track, and group
  });

  it("carries a person's face along with their body since the face was sampled", () => {
    const body = (ts: number, x: number) =>
      b(ts, 'person:7', { kind: 'person', track: '7', tone: 'unknown', box: [x, 0.1, x + 0.3, 0.9] });
    const list = [face(100, { box: [0.1, 0.1, 0.2, 0.2] }), body(100, 0.05), body(100.5, 0.45)];
    expect(faceToName(list, '7', 100.5)?.box).toEqual([0.5, 0.1, 0.6, 0.2].map(v => expect.closeTo(v, 6)));
  });

  it('finds nothing when the face is too old, named, or has no group', () => {
    expect(faceToName([face(98.5)], '7', 100)).toBeNull();
    expect(faceToName([face(99.8, { tone: 'known' })], '7', 100)).toBeNull();
    expect(faceToName([face(99.8, { group: undefined })], '7', 100)).toBeNull();
  });
});

describe('shownBoxes', () => {
  const chip = (id: string, kind: string, label: string): Chip => ({
    id,
    kind,
    label,
    spans: [],
    total_s: 0,
    count: 1,
  });
  const chips = [
    chip('person:dana', 'person', 'Dana'),
    chip('person:unknown', 'person', 'Unknown faces'),
    chip('object:mug', 'object', 'mug'),
  ];
  const list = [
    b(1, 'Dana'),
    b(1, 'Eli'),
    b(1, 'face:#7', { tone: 'unknown', label: "Who's this?", name: undefined }),
    b(1, 'mug', { kind: 'object', label: 'mug' }),
  ];

  it('shows everything while nothing is picked', () => {
    expect(shownBoxes(list, chips, new Set())).toHaveLength(4);
  });

  it('shows only what Show only picks', () => {
    expect(shownBoxes(list, chips, new Set(['person:dana'])).map(x => x.id)).toEqual(['Dana']);
    expect(shownBoxes(list, chips, new Set(['person:unknown', 'object:mug'])).map(x => x.id)).toEqual([
      'face:#7',
      'mug',
    ]);
  });

  it('picks bodies like faces: a named person by name, an unknown one under Unknown faces', () => {
    const dana = b(1, 'person:7', { kind: 'person', label: 'Dana', name: 'Dana' });
    const stranger = b(1, 'person:8', { kind: 'person', tone: 'unknown', label: "Who's this?", name: undefined });
    expect(shownBoxes([dana, stranger], chips, new Set(['person:dana'])).map(x => x.id)).toEqual(['person:7']);
    expect(shownBoxes([dana, stranger], chips, new Set(['person:unknown'])).map(x => x.id)).toEqual(['person:8']);
    expect(shownBoxes([dana, stranger], chips, new Set(['object:mug']))).toEqual([]);
  });

  it('matches a remembered thing to its chip by the chip’s words ("mug" picks "Red mug")', () => {
    const red = [b(1, 'object:o1', { kind: 'object', label: 'Red mug', name: 'Red mug' })];
    expect(shownBoxes(red, chips, new Set(['object:mug']))).toHaveLength(1);
    expect(
      shownBoxes([b(1, 'object:o2', { kind: 'object', label: 'Muggle' })], chips, new Set(['object:mug'])),
    ).toEqual([]);
  });
});

describe('asBox', () => {
  it('draws people (bodies and faces) as faces and things as objects, keeping the tone', () => {
    expect(asBox(b(1, 'face:1', { tone: 'unknown' }))).toMatchObject({ kind: 'face', tone: 'unknown' });
    expect(asBox(b(1, 'person:7', { kind: 'person', tone: 'known' }))).toMatchObject({ kind: 'face', tone: 'known' });
    expect(asBox(b(1, 'mug', { kind: 'object' })).kind).toBe('object');
  });
  it("draws the camera's labels as Live does, untoned, so remembered things stand out", () => {
    const label = asBox(b(1, 'label:sofa:0', { kind: 'label', tone: 'unknown', label: 'sofa' }));
    expect(label).toEqual({ id: 'label:sofa:0', box: [0.1, 0.1, 0.2, 0.2], label: 'sofa', kind: 'object' });
    expect(asBox(b(1, 'object:1', { kind: 'object' })).tone).toBe('known');
  });
});
