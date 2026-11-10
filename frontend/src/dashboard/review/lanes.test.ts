import { describe, expect, it } from 'vitest';
import { spkColor } from '../lib/speakers';
import type { Speakers } from '../live/types';
import type { ActivityPromotion } from './api';
import { chipTitle } from './chipStyle';
import {
  activityChips,
  camAt,
  cameraChips,
  compactLayout,
  eventTicks,
  hitAt,
  hitTitle,
  layout,
  MAX_ROWS,
  markBands,
  nameList,
  type PresenceChip,
  personColor,
  pickedPeople,
  presenceRows,
  shownPerson,
  UNKNOWN_FACES,
} from './lanes';
import { nextSpanStart, presentFrom, presentSpans } from './spans';
import { RULER } from './time';

const speakers = (list: { id: number; label: string }[], merged: Record<string, number> = {}): Speakers =>
  ({ list, byId: Object.fromEntries(list.map(s => [String(s.id), s])), merged }) as unknown as Speakers;
const person = (label: string, cams: Record<string, [number, number][]> = {}): PresenceChip => ({
  id: `person:${label.toLowerCase()}`,
  kind: 'person',
  label,
  spans: Object.values(cams).flat(),
  total_s: 0,
  count: 1,
  cams,
});

it.each(['__proto__', 'constructor', 'toString'])('keeps presence on %s separate from absent camera evidence', cam => {
  const chips = [person('Dana', JSON.parse(`{"${cam}":[[10,20]]}`)), person('Alice')];
  const rows = presenceRows(chips, [cam], () => 'blue');
  expect(Object.keys(rows)).toEqual([cam]);
  expect(rows[cam]?.map(row => row.label)).toEqual(['Dana']);
  expect(layout([cam], rows).cams[0]?.rows[0]?.row.label).toBe('Dana');
  expect(layout([cam], {}).cams[0]?.rows).toEqual([]);
});

describe('personColor', () => {
  it('is the colour of the voice with that name, matched without case, after merges', () => {
    const spk = speakers([{ id: 4, label: 'Dana' }], { '4': 7 });
    expect(personColor('dana ', spk)).toBe(spkColor(7));
  });

  it("is grey for speech nobody is known for, never a person's colour", () => {
    expect(personColor('unknown', speakers([]))).toBe(spkColor(null));
    expect(personColor('  ', speakers([]))).toBe(spkColor(null));
  });

  it('comes from the name when no voice has it, the same every time', () => {
    const spk = speakers([]);
    expect(personColor('Kara', spk)).toBe(personColor('kara', spk));
    expect(personColor('Kara', spk)).toBe(spkColor('kara'));
  });
});

describe('presenceRows', () => {
  const color = () => '#123456';

  it('lists the people seen on each camera, most time first, and only people', () => {
    const chips = [
      person('Eli', { cam0: [[0, 10]] }),
      person('Dana', { cam0: [[0, 100]], cam1: [[5, 6]] }),
      { ...person('Mug'), kind: 'object', cams: { cam0: [[0, 500]] } } as PresenceChip,
    ];
    const rows = presenceRows(chips, ['cam0', 'cam1', 'cam2'], color);
    expect(rows.cam0?.map(r => r.label)).toEqual(['Dana', 'Eli']);
    expect(rows.cam1?.map(r => r.label)).toEqual(['Dana']);
    expect(rows.cam2).toEqual([]);
  });

  it(`folds everyone past ${MAX_ROWS - 1} into one Others row with all their time`, () => {
    const chips = ['A', 'B', 'C', 'D', 'E', 'F'].map((n, i) => person(n, { cam0: [[0, 60 - i * 10]] }));
    const rows = presenceRows(chips, ['cam0'], color).cam0 ?? [];
    expect(rows.map(r => r.label)).toEqual(['A', 'B', 'C', 'D', '2 others']);
    expect(rows.at(-1)?.spans).toEqual([
      [0, 20],
      [0, 10],
    ]);
  });

  it('keeps exactly the max as rows, with no Others row', () => {
    const chips = ['A', 'B', 'C', 'D', 'E'].map(n => person(n, { cam0: [[0, 1]] }));
    expect(presenceRows(chips, ['cam0'], color).cam0?.map(r => r.label)).toEqual(['A', 'B', 'C', 'D', 'E']);
  });

  it('draws nothing for a person chip with no per-camera presence (an older server)', () => {
    expect(presenceRows([person('Dana')], ['cam0'], color).cam0).toEqual([]);
  });
});

describe('layout', () => {
  const rows = (n: number) =>
    Array.from({ length: n }, (_, i) => ({ id: `p${i}`, label: `P${i}`, color: '#000', spans: [] }));

  it('labels Who spoke, each camera, the audio, and Events, top to bottom', () => {
    const L = layout(['cam0', 'cam1'], {}, cam => (cam === 'cam0' ? 'Front camera' : cam));
    expect(L.labels.map(l => l.text)).toEqual(['Who spoke', 'Front camera', 'cam1', 'Audio kept', 'Events']);
    const ys = L.labels.map(l => l.y);
    expect([...ys].sort((a, b) => a - b)).toEqual(ys);
    expect(L.marks[0]).toBeGreaterThan(RULER);
    expect(L.events[0] + L.events[1]).toBeLessThanOrEqual(L.height);
  });

  it('packs the lanes a couple of pixels apart, each beside its label and as tall as one at least', () => {
    const L = layout(['cam0', 'cam1'], {});
    for (const [a, b] of L.labels.slice(1).map((l, i) => [L.labels[i], l] as const))
      expect((b?.y ?? 0) - ((a?.y ?? 0) + (a?.h ?? 0))).toBe(2);
    const bands = [L.speech, ...L.cams.map(c => c.video), L.audio, L.events];
    L.labels.forEach((l, i) => {
      const [y, h] = bands[i] ?? [0, 0];
      expect(l.h).toBeGreaterThanOrEqual(14);
      expect(y).toBeGreaterThanOrEqual(l.y);
      expect(y + h).toBeLessThanOrEqual(l.y + l.h);
    });
  });

  it("keeps every presence row inside its camera's video band, and grows the band to fit", () => {
    const one = layout(['cam0'], { cam0: rows(1) });
    const five = layout(['cam0'], { cam0: rows(5) });
    const lane = five.cams[0];
    if (!lane) throw new Error('no lane');
    const [vy, vh] = lane.video;
    for (const { band } of lane.rows) {
      expect(band[0]).toBeGreaterThanOrEqual(vy);
      expect(band[0] + band[1]).toBeLessThanOrEqual(vy + vh);
    }
    expect(five.cams[0]?.video[1]).toBeGreaterThan(one.cams[0]?.video[1] ?? 0);
    expect(five.height).toBeGreaterThan(one.height);
  });
});

describe('compact timeline', () => {
  const cams = ['cam0', 'cam1'];
  const people = ['Alice', 'Dana', 'Eli', 'Finn', 'Gail', 'Hana'].map((name, i) =>
    person(name, { cam0: [[100 + i * 20, 110 + i * 20]] }),
  );
  const mug: PresenceChip = {
    id: 'object:mug',
    kind: 'object',
    label: 'Mug',
    spans: [[300, 320]],
    total_s: 20,
    count: 1,
    camera: 'cam1',
  };
  const pet: PresenceChip = {
    id: 'pet:cat',
    kind: 'pet',
    label: 'Cat',
    spans: [[400, 420]],
    total_s: 20,
    count: 1,
  };
  const chips = [...people, mug, pet];
  const L = compactLayout(
    cams,
    presenceRows(chips, cams, () => '#000'),
  );

  it('has a fixed thin height with no labels, independent of cameras or people', () => {
    expect(L.compact).toBe(true);
    expect(L.height).toBe(RULER + 52);
    expect(compactLayout([], {}).height).toBe(L.height);
    expect(
      compactLayout(
        Array.from({ length: 20 }, (_, i) => `cam${i}`),
        {},
      ).height,
    ).toBe(L.height);
    expect(L.labels).toEqual([]);
    expect(L.cams[0]?.video).toEqual(L.cams[1]?.video);
    expect(new Set(L.cams.flatMap(lane => lane.rows.map(({ band }) => band[0]))).size).toBe(1);
    for (const y of [RULER, L.marks[0], L.speech[0], L.height - 1]) expect(camAt(L, y)).toBeNull();
  });

  it.each(people)('picks $label by time even when its row is overlaid or folded into Others', chip => {
    const start = chip.spans[0]?.[0] ?? 0;
    for (const y of [L.marks[0], L.speech[0], L.audio[0]])
      expect(hitAt(L, y, start + 5, 0, chips, [])).toMatchObject({
        kind: 'presence',
        chip: { id: chip.id },
        cam: 'cam0',
        start,
      });
  });

  it('picks the nearest object across selected cameras ahead of overlapping speech', () => {
    const said = [{ ts_start: 315, ts_end: 330, who: 'Alice' }];
    expect(hitAt(L, L.speech[0], 321, 2, chips, said)).toMatchObject({
      kind: 'presence',
      chip: { id: mug.id },
      cam: 'cam1',
      start: 300,
      end: 320,
    });
    expect(hitAt(compactLayout(['cam0'], {}), L.speech[0], 321, 2, chips, said)).toMatchObject({ kind: 'speech' });
    expect(hitAt(L, L.marks[0], 350, 2, chips, said)).toBeNull();
  });

  it('keeps camera-less pet evidence and its generic title without inventing a camera', () => {
    const hit = hitAt(L, L.audio[0], 410, 0, chips, []);
    expect(hit).toEqual({ kind: 'thing', chip: pet, start: 400, end: 420 });
    expect(hit && hitTitle(hit)).toBe('Cat');
  });

  it('uses the closest span regardless of chip order and excludes other camera presence', () => {
    const distant = person('Dana', { cam0: [[500, 510]], cam1: [[600, 620]] });
    const closer = person('Alice', { cam0: [[520, 530]] });
    expect(hitAt(L, L.events[0], 519, 10, [distant, closer], [])).toMatchObject({
      chip: { id: closer.id },
      start: 520,
    });
    expect(hitAt(compactLayout(['cam0'], {}), L.marks[0], 610, 0, [distant], [])).toBeNull();
  });

  it('prioritizes event starts and actual activity duration using the caller tolerance', () => {
    const event: PresenceChip = {
      id: 'event:bell',
      kind: 'event',
      label: 'Bell',
      spans: [[310, 350]],
      total_s: 40,
      count: 1,
    };
    const activities = activityChips([
      {
        id: 'read',
        camera: 'cam0',
        start: 400,
        end: 430,
        count: null,
        name: null,
        kind: 'reading',
        person: null,
        promoted: false,
      },
    ]);
    expect(hitAt(L, L.audio[0], 311, 2, [...chips, event], [])).toMatchObject({ kind: 'event', start: 310, end: 350 });
    expect(hitAt(L, L.events[0], 318, 2, [...chips, event], [])).toMatchObject({
      kind: 'presence',
      chip: { id: mug.id },
    });
    expect(hitAt(L, L.speech[0], 425, 0, [...chips, ...activities], [])).toMatchObject({
      kind: 'activity',
      activity: { id: 'read' },
      start: 400,
      end: 430,
    });
  });
});

describe('who shows at full strength', () => {
  const chips = [person('Dana'), person('Eli'), { ...person('Unknown faces'), id: UNKNOWN_FACES }];

  it('is everyone while no person is picked', () => {
    expect(shownPerson('Eli', pickedPeople(chips, new Set(['object:mug'])))).toBe(true);
  });

  it('is only the picked people, by name without case, and never unknown faces as a name', () => {
    const picked = pickedPeople(chips, new Set(['person:dana', UNKNOWN_FACES]));
    expect([...picked]).toEqual(['dana']);
    expect(shownPerson('DANA', picked)).toBe(true);
    expect(shownPerson('Eli', picked)).toBe(false);
  });
});

describe('eventTicks', () => {
  it('ticks the start of each event span, in time order, in its chip colour', () => {
    const ev = (id: string, spans: [number, number][]) =>
      ({ id, kind: 'event', label: id, spans, total_s: 0, count: spans.length }) as PresenceChip;
    const ticks = eventTicks(
      [
        ev('e1', [[30, 40]]),
        ev('e2', [
          [10, 12],
          [50, 60],
        ]),
        person('Dana', {}),
      ],
      c => c.id,
    );
    expect(ticks).toEqual([
      { t: 10, color: 'e2' },
      { t: 30, color: 'e1' },
      { t: 50, color: 'e2' },
    ]);
  });
});

describe('markBands', () => {
  it('bands picked things and events, and picked people only when the server sent no per-camera presence', () => {
    const mug = { ...person('Mug'), id: 'object:mug', kind: 'object' } as PresenceChip;
    const dana = person('Dana', { cam0: [[0, 10]] });
    const eli = person('Eli'); // no cams from this server
    const unknown = { ...person('Unknown faces'), id: UNKNOWN_FACES };
    const sel = new Set(['object:mug', 'person:dana', 'person:eli', UNKNOWN_FACES]);
    expect(markBands([mug, dana, eli, unknown], sel, c => c.id).map(b => b.color)).toEqual([
      'object:mug',
      'person:eli',
      UNKNOWN_FACES,
    ]);
  });

  it('bands nothing that is not picked', () => {
    expect(markBands([person('Eli')], new Set(), c => c.id)).toEqual([]);
  });
});

describe('what the pointer is over on the timeline', () => {
  const lunch = {
    id: 'event:lunch',
    kind: 'event',
    label: 'Lunch',
    spans: [[1000, 1600]],
    total_s: 0,
    count: 1,
  } as PresenceChip;
  const dana = person('Dana', { cam0: [[500, 900]] });
  const chips = [lunch, dana];
  const L = layout(
    ['cam0'],
    presenceRows(chips, ['cam0'], () => '#000'),
  );
  const said = [
    { ts_start: 100, ts_end: 103, who: 'Kara' },
    { ts_start: 108, ts_end: 110, who: 'Luca' }, // 5 s after: the same stretch of talk
    { ts_start: 112, ts_end: 113, who: 'Kara' },
    { ts_start: 200, ts_end: 202, who: 'Joe' }, // long after: another
  ];
  const mid = (band: readonly number[]) => (band[0] as number) + (band[1] as number) / 2;

  it('picks an event tick a click lands on, within a few pixels', () => {
    expect(hitAt(L, mid(L.events), 1003, 5, chips, said)).toMatchObject({ kind: 'event', start: 1000, end: 1600 });
    expect(hitAt(L, mid(L.events), 1020, 5, chips, said)).toBeNull();
  });

  it('picks the whole stretch of talk around a line, with who spoke', () => {
    const h = hitAt(L, mid(L.speech), 109, 0.5, chips, said);
    expect(h).toEqual({ kind: 'speech', names: ['Kara', 'Luca'], start: 100, end: 113 });
    expect(hitAt(L, mid(L.speech), 150, 0.5, chips, said)).toBeNull(); // a quiet gap
  });

  it('picks a person in one camera’s view, by their row', () => {
    const row = L.cams[0]?.rows[0];
    if (!row) throw new Error('no row');
    expect(hitAt(L, mid(row.band), 600, 1, chips, said)).toMatchObject({
      kind: 'presence',
      cam: 'cam0',
      start: 500,
      end: 900,
    });
    expect(hitAt(L, mid(row.band), 950, 1, chips, said)).toBeNull();
  });

  it('picks the nearest stretch of talk or bar, not the first one in reach (zoomed out, reach is wide)', () => {
    const far = [
      { ts_start: 100, ts_end: 102, who: 'Kara' },
      { ts_start: 130, ts_end: 132, who: 'Joe' }, // 28 s later: another stretch
    ];
    expect(hitAt(L, mid(L.speech), 131, 40, chips, far)).toMatchObject({ names: ['Joe'], start: 130 });
    const twice = person('Eli', {
      cam0: [
        [500, 520],
        [540, 560],
      ],
    });
    const L2 = layout(
      ['cam0'],
      presenceRows([twice], ['cam0'], () => '#000'),
    );
    const row = L2.cams[0]?.rows[0];
    if (!row) throw new Error('no row');
    expect(hitAt(L2, mid(row.band), 545, 30, [twice], [])).toMatchObject({ start: 540, end: 560 });
    expect(hitAt(L2, mid(row.band), 525, 30, [twice], [])).toMatchObject({ start: 500 }); // nearer the first
  });

  it("picks from the camera lane clicked, with that camera's own spans", () => {
    const both = person('Dana', { cam0: [[500, 900]], cam1: [[1000, 1100]] });
    const L2 = layout(
      ['cam0', 'cam1'],
      presenceRows([both], ['cam0', 'cam1'], () => '#000'),
    );
    const row = L2.cams[1]?.rows[0];
    if (!row) throw new Error('no row');
    expect(hitAt(L2, mid(row.band), 1050, 1, [both], [])).toMatchObject({ cam: 'cam1', start: 1000, end: 1100 });
    expect(hitAt(L2, mid(row.band), 600, 1, [both], [])).toBeNull(); // cam0 saw her then, not cam1
  });

  it('picks nothing on empty lane', () => {
    expect(hitAt(L, L.height - 1, 600, 1, chips, said)).toBeNull();
  });

  it('names what it picked', () => {
    expect(hitTitle({ kind: 'event', chip: lunch, start: 0, end: 1 })).toBe('Lunch');
    expect(hitTitle({ kind: 'presence', chip: dana, cam: 'cam0', start: 0, end: 1 })).toBe('Dana in view');
    expect(hitTitle({ kind: 'speech', names: ['Kara', 'Luca', 'Joe'], start: 0, end: 1 })).toBe(
      'Kara, Luca, and Joe talking',
    );
    expect(nameList(['Kara', 'Luca'])).toBe('Kara and Luca');
  });
});

describe('camAt', () => {
  it('finds the camera whose lane a height is in, and none in the gap between lanes', () => {
    const L = layout(['cam0', 'cam1'], {});
    const [c0, c1] = L.cams;
    expect(camAt(L, (c0?.video[0] ?? 0) + 1)).toBe('cam0');
    expect(camAt(L, (c0?.video[0] ?? 0) + (c0?.video[1] ?? 0))).toBeNull(); // the 2 px between the lanes
    expect(camAt(L, (c1?.video[0] ?? 0) + (c1?.video[1] ?? 0) - 1)).toBe('cam1');
    expect(camAt(L, L.speech[0])).toBeNull(); // the speech lane belongs to no camera
    expect(camAt(L, L.audio[0])).toBeNull();
  });
});

it('groups activity filters by kind while keeping individual lane identities and hit spans', () => {
  const base = {
    camera: 'cam0',
    start: 10,
    end: 20,
    count: null,
    name: null,
    kind: 'reading',
    person: null,
    promoted: false,
  };
  const items = [
    { ...base, id: 'one', track_id: 'track-a', person_name: 'Alice' },
    { ...base, id: 'two', start: 30, end: 45, track_id: 'track-b', person_name: 'Dana' },
    { ...base, id: 'three', start: 50, end: 51, kind: null },
  ];
  const chips = activityChips(items);
  expect(chips).toHaveLength(2);
  expect(chips[0]).toMatchObject({
    label: 'Reading',
    count: 2,
    spans: [
      [10, 20],
      [30, 45],
    ],
  });
  const ticks = eventTicks(chips, () => '#abc');
  expect(ticks).toHaveLength(3);
  expect(ticks[1]?.activity).toMatchObject({ id: 'two', track_id: 'track-b', person_name: 'Dana' });
  const L = layout([], {});
  const hit = hitAt(L, L.events[0] + 1, 35, 0.1, chips, []);
  expect(hit).toMatchObject({ kind: 'activity', activity: { id: 'two' }, start: 30, end: 45 });
  expect(hit && hitTitle(hit)).toContain('Dana');
  expect(markBands(chips, new Set([chips[0]?.id ?? '']), () => '#abc')).toEqual([
    {
      color: '#abc',
      spans: [
        [10, 20],
        [30, 45],
      ],
    },
  ]);
});

it('draws and hits a zero-duration crossing at its exact source time', () => {
  const chips = activityChips([
    {
      id: 'crossing',
      camera: 'cam0',
      start: 10,
      end: 10,
      count: null,
      name: null,
      kind: 'crossing',
      person: null,
      promoted: false,
    },
  ]);
  expect(eventTicks(chips, () => '#abc')[0]).toMatchObject({ t: 10, end: 10, activity: { id: 'crossing' } });
  const L = layout([], {});
  expect(hitAt(L, L.events[0] + 1, 10, 0, chips, [])).toMatchObject({ kind: 'activity', start: 10, end: 10 });
});

const ratingItem = {
  id: 'candidate',
  camera: 'cam0',
  start: 10,
  end: 20,
  count: null,
  name: null,
  kind: 'reading',
  person: null,
  promoted: false,
};

it.each([
  ['reading', 'needs_ratings', 3, 7, 6, 6 / 7, 'Needs 3 more ratings'],
  ['reading', 'needs_ratings', 1, 9, 8, 8 / 9, 'Needs 1 more rating'],
  ['reading', 'needs_ratings', 0, 10, 7, 0.7, 'Not counted in Ask: more ratings needed'],
  ['reading', 'needs_correct_ratings', null, 10, 7, 0.7, 'Not counted in Ask: needs more correct ratings'],
  ['reading', 'disabled', null, 10, 9, 0.9, 'Not counted in Ask'],
  ['crossing', 'counted', null, 0, 0, null, 'Counted in Ask; Wrong events excluded'],
  ['crossing', 'needs_correct_ratings', null, 10, 7, 0.7, 'Not counted in Ask: needs more correct ratings'],
  [null, 'needs_ratings', 10, 0, 0, null, 'Needs 10 more ratings'],
] as const)(
  'shows authoritative %s / %s counting status without deriving it from displayed candidates',
  (kind, status, remaining, rated, right, precision, expected) => {
    const policy: ActivityPromotion = {
      kind,
      camera: null,
      status,
      remaining_ratings: remaining,
      rated,
      right,
      precision,
    };
    const [chip] = activityChips([{ ...ratingItem, kind, feedback: 'wrong' }], [policy]);
    if (!chip) throw new Error('missing chip');
    expect(chipTitle(chip)).toContain(expected);
    expect(chipTitle(chip)).not.toContain('Needs 0');
    expect(chip.activities?.[0]?.promoted).toBe(false);
  },
);

it('lists effective angle overrides without claiming all cameras are counted', () => {
  const policy: ActivityPromotion[] = [
    {
      kind: 'reading',
      camera: null,
      status: 'needs_correct_ratings',
      remaining_ratings: null,
      rated: 30,
      right: 23,
      precision: 23 / 30,
    },
    {
      kind: 'reading',
      camera: 'cam0',
      status: 'counted',
      remaining_ratings: null,
      rated: 10,
      right: 9,
      precision: 0.9,
    },
    {
      kind: 'reading',
      camera: 'cam1',
      status: 'needs_correct_ratings',
      remaining_ratings: null,
      rated: 10,
      right: 4,
      precision: 0.4,
    },
    { kind: 'reading', camera: 'cam2', status: 'counted', remaining_ratings: null, rated: 10, right: 10, precision: 1 },
  ];
  const [chip] = activityChips(
    [ratingItem, { ...ratingItem, id: 'two', camera: 'cam1' }],
    policy,
    cam => ({ cam0: 'Front camera', cam1: 'Rear camera', cam2: 'Other camera' })[cam] ?? cam,
  );
  if (!chip) throw new Error('missing chip');
  expect(chipTitle(chip)).toContain(
    'Default: Not counted in Ask: needs more correct ratings. Front camera: Counted in Ask; Wrong events excluded. Rear camera: Not counted in Ask: needs more correct ratings.',
  );
  expect(chipTitle(chip)).not.toContain('Other camera');
});

it('does not assume crossing defaults when the backend supplies no promotion status', () => {
  const [chip] = activityChips([{ ...ratingItem, kind: 'crossing' }]);
  if (!chip) throw new Error('missing chip');
  expect(chipTitle(chip)).toContain('Counting status unavailable');
  expect(chipTitle(chip)).not.toContain('Counted in Ask');
});

it('labels a crossing as walked past and explains it in the chip tooltip, keeping its kind and filter id', () => {
  const [chip] = activityChips([{ ...ratingItem, kind: 'crossing' }]);
  if (!chip) throw new Error('missing chip');
  expect(chip.id).toBe('event:activity:crossing');
  expect(chip.label).toBe('Walked past');
  expect(chipTitle(chip)).toContain("A person crossed the camera's counting line.");
  expect(chip.activities?.[0]?.kind).toBe('crossing');
});

it('keeps only selected camera evidence in marks, retaining speech and events with no camera', () => {
  const chips = cameraChips(
    [
      {
        id: 'person:dana',
        kind: 'person',
        label: 'Dana',
        count: 2,
        total_s: 20,
        spans: [
          [10, 20],
          [30, 40],
        ],
        cams: { cam0: [[10, 20]], cam1: [[30, 40]] },
      },
      {
        id: 'event:presence:cam0',
        kind: 'event',
        label: 'Came into view',
        camera: 'cam0',
        count: 1,
        total_s: 0,
        spans: [[10, 11]],
      },
      { id: 'event:summary', kind: 'event', label: 'Conversation', count: 1, total_s: 10, spans: [[50, 60]] },
    ],
    ['cam1'],
  );
  expect(chips.map(c => c.id)).toEqual(['person:dana', 'event:summary']);
  expect(chips[0]?.spans).toEqual([[30, 40]]);
  expect(chips[0]?.cams).toEqual({ cam1: [[30, 40]] });
  expect(chips[1]?.spans).toEqual([[50, 60]]);
});

it('keeps speech reachable by stepping and skip playback when camera evidence is filtered', () => {
  const filtered = cameraChips(
    [
      {
        id: 'person:alice',
        kind: 'person',
        label: 'Alice',
        count: 3,
        total_s: 30,
        spans: [
          [10, 20],
          [100, 110],
          [200, 210],
        ],
        speech: [[100, 110]],
        cams: { cam0: [[10, 20]], cam1: [[200, 210]] },
      },
    ],
    ['cam0'],
  );
  expect(filtered[0]?.spans).toEqual([
    [10, 20],
    [100, 110],
  ]);
  const selected = presentSpans(filtered, new Set(['person:alice']));
  expect(nextSpanStart(selected, 20)).toBe(100);
  expect(presentFrom(selected, 100)).toBe(100);
  expect(filtered[0]?.count).toBe(2);
  expect(filtered[0]?.total_s).toBe(20);
});

it('unions overlapping speech and selected views without counting the same time twice', () => {
  const filtered = cameraChips(
    [
      {
        id: 'person:alice',
        kind: 'person',
        label: 'Alice',
        count: 2,
        total_s: 30,
        spans: [
          [10, 30],
          [100, 110],
        ],
        speech: [
          [15, 30],
          [100, 110],
        ],
        cams: { cam0: [[10, 20]], cam1: [[18, 25]], cam2: [[200, 210]] },
      },
    ],
    ['cam0', 'cam1'],
  );
  expect(filtered[0]?.spans).toEqual([
    [10, 30],
    [100, 110],
  ]);
  expect(filtered[0]?.count).toBe(2);
  expect(filtered[0]?.total_s).toBe(30);
  expect(filtered[0]?.cams).toEqual({ cam0: [[10, 20]], cam1: [[18, 25]] });
});
