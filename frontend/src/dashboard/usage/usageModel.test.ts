import { describe, expect, it } from 'vitest';
import type { UsageHistory, UsageSample } from '../sources/api';
import { BANDS, bands, breakdown, merge, OTHER, resources, rows } from './usageModel';

const GB = 1073741824;
const sample = (ts: number, cpu: Record<string, number> = {}, mem: Record<string, number> = {}): UsageSample => ({
  ts,
  cpu,
  mem,
  gpu: [],
});
const history = (gpus: UsageHistory['gpus'] = []): UsageHistory => ({
  step: 5,
  window: 1800,
  cpu_count: 4,
  mem_total: 32 * GB,
  gpus,
  samples: [],
});

describe('merge', () => {
  it('adds only newer samples and drops those past the window', () => {
    const held = [sample(0), sample(5), sample(10)];
    expect(merge(held, [sample(10), sample(15), sample(25)], 20).map(s => s.ts)).toEqual([5, 10, 15, 25]);
    expect(merge([], [], 20)).toEqual([]);
  });
});

describe('resources', () => {
  it('always offers the processor and memory, and one choice per GPU', () => {
    expect(resources(history()).map(r => r.label)).toEqual(['Processor', 'Memory']);
    const gpus = resources(history([{ index: 1, name: 'RTX', total: 24 * GB }]));
    expect(gpus.map(r => [r.key, r.label, r.total, r.unit])).toEqual([
      ['cpu', 'Processor', 100, 1],
      ['mem', 'Memory', 32 * GB, GB],
      ['gpu1', 'GPU 1', 24 * GB, GB],
    ]);
  });
});

describe('bands', () => {
  it('gives its own band to the busiest services by share of any resource, in name order', () => {
    const cpu = Object.fromEntries(Array.from({ length: BANDS + 2 }, (_, i) => [`s${i}`, i + 1]));
    // s0 uses little processor but most of the memory, so it keeps a band.
    const got = bands([sample(0, { ...cpu, [OTHER]: 50 }, { s0: 30 * GB })], resources(history()));
    expect(got).toEqual(['s0', 's3', 's4', 's5', 's6', 's7']);
  });

  it('gives no band to a service that never used anything', () => {
    expect(bands([sample(0, { idle: 0, busy: 3 })], resources(history()))).toEqual(['busy']);
  });
});

describe('rows', () => {
  it('folds services without a band into everything else, and breaks the stack where samples are missing', () => {
    const got = rows(
      [sample(0, { a: 10, b: 2, [OTHER]: 5 }), sample(5, { a: 11, [OTHER]: 5 }), sample(60, { a: 12 })],
      'cpu',
      ['a'],
      5,
    );
    expect(got.map(r => [r.ts, r.gap, r.values])).toEqual([
      [0, false, { a: 10, [OTHER]: 7 }],
      [5, false, { a: 11, [OTHER]: 5 }],
      [10, true, { a: 0, [OTHER]: 0 }],
      [60, false, { a: 12, [OTHER]: 0 }],
    ]);
  });

  it('reads a GPU the sample does not have as nothing', () => {
    expect(rows([sample(0, { a: 1 })], 'gpu3', ['a'], 5)[0]?.values).toEqual({ a: 0, [OTHER]: 0 });
  });
});

describe('breakdown', () => {
  it('lists every service busiest first and everything else last, leaving out zeros', () => {
    expect(breakdown(sample(0, { b: 2, a: 9, idle: 0, [OTHER]: 4 }), 'cpu')).toEqual([
      { name: 'a', value: 9 },
      { name: 'b', value: 2 },
      { name: OTHER, value: 4 },
    ]);
  });
});
