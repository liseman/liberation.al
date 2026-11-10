import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { lint, projection } from '../../../vendor/floorplan/index.js';
import { EXAMPLE_PLACES } from '../examples';
import { placeToPlan, planToPlace } from './anchor';
import { PlaceDslError, parsePlaceDsl, toPlaceDsl } from './dsl';
import { type Anchor, readExtensions } from './ext';
import { isQaRule, lintPlace } from './lint';
import { drawPlace } from './render';

const BASE = {
  title: 'T',
  walls: { exterior: 0.3, partition: 0.12 },
  rooms: {
    a: { name: 'A', kind: 'living', rect: [0, 0, 4, 3] },
    b: { name: 'B', kind: 'bedroom', rect: [4, 0, 3, 3] },
  },
  openings: [{ id: 'd1', type: 'door', between: ['a', 'b'], width: 0.9 }],
};
const flat = EXAMPLE_PLACES.find(p => p.id === 'example-flat')?.plan as Record<string, unknown>;
const rules = (doc: Record<string, unknown>, opts?: { all?: boolean }) => lintPlace(doc, opts).map(f => f.rule);

describe('the DSL', () => {
  it('prints and reads back every extension of the example flat unchanged', () => {
    const text = toPlaceDsl(flat);
    expect(parsePlaceDsl(text)).toEqual(flat);
    expect(toPlaceDsl(parsePlaceDsl(text))).toBe(text);
  });

  const id = fc.stringMatching(/^[a-z][a-z0-9_]{0,6}$/);
  const n = fc.integer({ min: -500, max: 500 }).map(v => v / 10);
  const pos = fc.integer({ min: 1, max: 500 }).map(v => v / 10);
  const conf = fc.integer({ min: 0, max: 100 }).map(v => v / 100);
  const ext = fc.record(
    {
      'x-anchor': fc.record({ frame: id, yaw: n, origin: fc.tuple(n, n, n) }),
      'x-revision': fc.record({ n: fc.integer({ min: 1, max: 9 }), visits: fc.array(id, { maxLength: 3 }) }),
      'x-provenance': fc.dictionary(
        fc.tuple(fc.constantFrom('room', 'wall', 'opening', 'outdoor'), id).map(([k, i]) => `${k}:${i}`),
        fc.record({ source: fc.constantFrom('machine', 'owner'), confidence: conf }, { requiredKeys: ['source'] }),
        { maxKeys: 3 },
      ),
      'x-unexplored': fc.dictionary(id, fc.record({ rect: fc.tuple(n, n, pos, pos) }), { maxKeys: 2 }),
      'x-cameras': fc.dictionary(id, fc.record({ at: fc.tuple(n, n), yaw: n }), { maxKeys: 2 }),
      'x-objects': fc.dictionary(
        id,
        fc.record({ label: fc.string({ minLength: 1, maxLength: 8 }), at: fc.tuple(n, n), size: fc.tuple(pos, pos) }),
        { maxKeys: 2 },
      ),
    },
    { requiredKeys: [] },
  );

  it('round-trips any extensions through the DSL', () => {
    fc.assert(
      fc.property(ext, e => {
        const doc = { ...BASE, ...e };
        const back = parsePlaceDsl(toPlaceDsl(doc));
        expect(readExtensions(back).ext).toEqual(readExtensions(doc).ext);
      }),
    );
  });

  it('keeps the upstream line numbers past x- lines', () => {
    expect(() => parsePlaceDsl('plan "T"\nx-anchor f yaw 0 origin 0,0,0\nfrobnicate a')).toThrow(/line 3/);
  });

  it('names the line of a bad x- statement', () => {
    expect(() => parsePlaceDsl('plan "T"\nroom a living rect 0,0 3x3\nx-camera c yaw 9')).toThrow(PlaceDslError);
    expect(() => parsePlaceDsl('plan "T"\nroom a living rect 0,0 3x3\nx-camera c yaw 9')).toThrow(/^line 3:/);
  });

  it('leaves an extended plan valid upstream', () => {
    expect(lint(flat).findings.filter(f => f.rule.startsWith('schema.'))).toEqual([]);
  });
});

describe('the anchor', () => {
  it('turns and shifts plan points into the frame, north up', () => {
    const a: Anchor = { frame: 'f', yaw: 90, origin: [10, 20, 1.5] };
    // (1, 0): east on the plan; turned 90 degrees counterclockwise it points along +Y.
    expect(planToPlace(a, [1, 0]).map(v => +v.toFixed(9))).toEqual([10, 21, 1.5]);
    // (0, 2): 2 m south on the plan is -Y before turning, +X after.
    expect(planToPlace(a, [0, 2]).map(v => +v.toFixed(9))).toEqual([12, 20, 1.5]);
    expect(planToPlace({ frame: 'f', yaw: 0, origin: [0, 0, 0], scale: 2 }, [1, 1])).toEqual([2, -2, 0]);
  });

  it('brings any point back where it was', () => {
    const r = fc.double({ min: -1000, max: 1000, noNaN: true });
    fc.assert(
      fc.property(
        r,
        r,
        fc.double({ min: -720, max: 720, noNaN: true }),
        fc.double({ min: 0.1, max: 10, noNaN: true }),
        r,
        r,
        (x, y, yaw, scale, ox, oy) => {
          const a: Anchor = { frame: 'f', yaw, origin: [ox, oy, 0], scale };
          const [bx, by] = placeToPlan(a, planToPlace(a, [x, y]));
          expect(bx).toBeCloseTo(x, 6);
          expect(by).toBeCloseTo(y, 6);
        },
      ),
    );
  });
});

describe('the drawing', () => {
  it('adds nothing to a plan without extensions', () => {
    const { svg, result } = drawPlace(BASE);
    expect(svg).toBe(result.svg);
    expect(svg).not.toContain('x-layer');
  });

  it.each([
    ['an owner-corrected wall', { 'wall:w1': { source: 'owner' } }, 'x-line x-owner', 'You corrected this'],
    ['an unsure machine wall', { 'wall:w1': { source: 'machine', confidence: 0.3 } }, 'x-line x-unsure', '30% sure'],
    ['an unsure opening', { 'opening:d1': { source: 'machine', confidence: 0.2 } }, 'x-line x-unsure', 'opening d1'],
    ['a partly seen room', { 'room:a': { source: 'machine', partial: true } }, 'class="x-seen"', 'only partly seen'],
  ])('marks %s', (_, provenance, cls, title) => {
    const { svg } = drawPlace({ ...BASE, 'x-provenance': provenance });
    expect(svg).toContain(cls);
    expect(svg).toContain(title);
  });

  it("draws an opening's mark over the door itself", () => {
    // d1 sits on the wall x = 4 between the rooms, centred at y = 1.5 and 0.9 m wide: from y = 1.05 to y = 1.95.
    const { svg, result } = drawPlace({ ...BASE, 'x-provenance': { 'opening:d1': { source: 'owner' } } });
    const proj = projection(result.model, {});
    const ends = [...(svg.match(/class="x-line x-owner" d="M([\d.]+) ([\d.]+) L([\d.]+) ([\d.]+)"/) ?? [])]
      .slice(1)
      .map(Number);
    const [x1, y1] = ends;
    const [x2, y2] = ends.slice(2);
    const a = proj.toModel(x1 as number, y1 as number);
    const b = proj.toModel(x2 as number, y2 as number);
    expect(
      [a, b].map(([x, y]): [number, number] => [+x.toFixed(2), +y.toFixed(2)]).sort((p, q) => p[1] - q[1]),
    ).toEqual([
      [4, 1.05],
      [4, 1.95],
    ]);
  });

  it('leaves a machine wall it was sure of as the library drew it', () => {
    const { svg, result } = drawPlace({
      ...BASE,
      'x-provenance': { 'wall:w1': { source: 'machine', confidence: 0.9 } },
    });
    expect(svg).toBe(result.svg);
  });

  it('labels an unexplored area only where the label fits, and titles it either way', () => {
    const { svg } = drawPlace({ ...BASE, 'x-unexplored': { nook: { rect: [5, 1, 1, 1] } } });
    expect(svg).toContain('nook: not seen yet');
    expect(svg).not.toContain('>Not seen yet<');
  });

  it('hatches an unexplored area, and draws cameras and objects', () => {
    const { svg } = drawPlace({
      ...BASE,
      'x-unexplored': { u: { rect: [4.2, 0.5, 2.6, 2] } },
      'x-cameras': { c: { at: [1, 1], yaw: 90, fov: 90 } },
      'x-objects': { o: { label: 'Sofa <&>', at: [2, 2], size: [1, 0.5] } },
    });
    expect(svg).toContain('>Not seen yet<');
    expect(svg).toMatch(/fill="url\(#x-hatch-\d+\)"/);
    expect(svg).toContain('class="x-cam"');
    expect(svg).toContain('class="x-fov"');
    expect(svg).toContain('data-x-object="o"');
    expect(svg).toContain('Sofa &lt;&amp;&gt;');
    expect(svg).not.toContain('Sofa <&>');
  });
});

describe('the lint', () => {
  it("drops the upstream design rules and keeps geometry's", () => {
    const anchored = { ...BASE, 'x-anchor': { frame: 'f', yaw: 0, origin: [0, 0, 0] } };
    expect(rules(anchored, { all: true })).toContain('habitable.no_window');
    expect(rules(anchored)).toEqual([]);
    const overlap = { ...anchored, rooms: { ...BASE.rooms, b: { name: 'B', kind: 'bedroom', rect: [3.9, 0, 3, 3] } } };
    expect(rules(overlap)).toContain('tiling.overlap');
    expect(isQaRule('room.min_dimension')).toBe(false);
    expect(isQaRule('wall.ambiguous')).toBe(true);
  });

  it.each([
    [
      'a confidence past 1',
      { 'x-provenance': { 'room:a': { source: 'machine', confidence: 1.5 } } },
      'x.schema',
      'x-provenance["room:a"].confidence',
    ],
    ['an unknown field', { 'x-cameras': { c: { at: [1, 1], zoom: 2 } } }, 'x.schema', 'x-cameras["c"].zoom'],
    ['a provenance key with no kind', { 'x-provenance': { a: { source: 'owner' } } }, 'x.schema', 'x-provenance["a"]'],
    [
      'a wall that is not there',
      { 'x-provenance': { 'wall:w99': { source: 'owner' } } },
      'x.reference',
      'x-provenance["wall:w99"]',
    ],
    ['a revision with no visits', { 'x-revision': { n: 1, visits: [] } }, 'x.revision.no_visits', 'x-revision'],
    [
      'revisions out of order',
      { 'x-history': [{ n: 2, visits: ['v'] }], 'x-revision': { n: 2, visits: ['v'] } },
      'x.history.order',
      'x-revision',
    ],
    ['no anchor', {}, 'x.anchor.missing', 'x-anchor'],
  ])('reports %s', (_, extra, rule, path) => {
    expect(lintPlace({ ...BASE, ...extra })).toContainEqual(expect.objectContaining({ rule, path }));
  });

  it("notes a wall measured off its class's thickness, past 3 cm only", () => {
    const at = (t: number) => rules({ ...BASE, 'x-provenance': { 'wall:w1': { source: 'machine', thickness: t } } });
    expect(at(0.22)).toContain('x.wall.thickness');
    expect(at(0.28)).not.toContain('x.wall.thickness');
  });

  it.each(EXAMPLE_PLACES.map(p => [p.name, p] as const))('finds nothing wrong with %s', (_, place) => {
    expect(lintPlace(place.plan ?? {}).filter(f => f.severity !== 'info')).toEqual([]);
  });
});
