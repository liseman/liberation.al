// The data behind This machine's stacked usage chart: the server's samples (GET /stats/history), merged as
// they arrive, cut into one stack per resource. The services that get their own band are the few busiest across every
// resource, so a service keeps its band and colour when the resource changes; their colours go in name order, so the
// same services get the same colours on every load. Everything else (the rest of the machine and the smaller
// services) is one neutral band on top, and the hover still lists each of them.

import { size } from '../settings/services/machineMeters';
import type { UsageHistory, UsageSample } from '../sources/api';

export const BANDS = 6; // services with their own colour; the palette's validated order has six before it repeats
export const OTHER = 'other'; // the server's name for the rest of the machine
export const GAP_STEPS = 3; // samples further apart than this many steps are a gap in the chart, not a slope

export interface Resource {
  key: string; // 'cpu', 'mem', or 'gpu<index>'
  label: string; // the toggle's words
  title: string; // the axis and summary's words
  total: number; // what the stack fills up to
  unit: number; // the axis ticks at round numbers of this: 1 for percent, a GB for memory
  format: (value: number) => string;
}

const GB = 1073741824;
const percent = (n: number) => `${n < 10 ? n.toFixed(1) : Math.round(n)}%`;

/** The resources a history can show: the processor and memory always, and each GPU it reports. */
export function resources(h: UsageHistory): [Resource, ...Resource[]] {
  return [
    { key: 'cpu', label: 'Processor', title: 'Processor, % of the machine', total: 100, unit: 1, format: percent },
    { key: 'mem', label: 'Memory', title: 'Memory', total: h.mem_total || 1, unit: GB, format: size },
    ...h.gpus.map(g => ({
      key: `gpu${g.index}`,
      label: `GPU ${g.index}`,
      title: `GPU ${g.index} memory (${g.name})`,
      total: g.total || 1,
      unit: GB,
      format: size,
    })),
  ];
}

/** One sample's value per service for a resource; empty when the sample has no such GPU. */
export function valuesOf(sample: UsageSample, key: string): Record<string, number> {
  if (key === 'cpu') return sample.cpu;
  if (key === 'mem') return sample.mem;
  return sample.gpu.find(g => `gpu${g.index}` === key)?.apps ?? {};
}

/** One GPU's overall utilization at a sample, when the server read it. */
export function utilOf(sample: UsageSample, key: string): number | null {
  return sample.gpu.find(g => `gpu${g.index}` === key)?.util ?? null;
}

/** Newly fetched samples added to those held, oldest dropped past the window. */
export function merge(held: UsageSample[], fresh: UsageSample[], window: number): UsageSample[] {
  const last = held.at(-1)?.ts ?? Number.NEGATIVE_INFINITY;
  const all = [...held, ...fresh.filter(s => s.ts > last)];
  const newest = all.at(-1)?.ts;
  return newest == null ? all : all.filter(s => s.ts >= newest - window);
}

/** The services with their own band: the BANDS busiest by their peak share of any resource, in name order. */
export function bands(samples: UsageSample[], all: Resource[]): string[] {
  const peak = new Map<string, number>();
  for (const s of samples)
    for (const r of all)
      for (const [name, value] of Object.entries(valuesOf(s, r.key)))
        if (name !== OTHER) peak.set(name, Math.max(peak.get(name) ?? 0, value / r.total));
  return [...peak.entries()]
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, BANDS)
    .map(([name]) => name)
    .sort((a, b) => a.localeCompare(b));
}

export interface Row {
  ts: number;
  gap: boolean; // a break in the record: drawn as nothing
  values: Record<string, number>; // each band, and OTHER for everything else
  sample?: UsageSample;
}

/** The stack's rows for one resource: each band's value, OTHER as everything else, and a gap row wherever samples are
 *  further apart than GAP_STEPS steps, so a time nothing was recorded never reads as a slope or a zero. */
export function rows(samples: UsageSample[], key: string, names: string[], step: number): Row[] {
  const out: Row[] = [];
  const zero = Object.fromEntries([...names, OTHER].map(n => [n, 0]));
  for (const s of samples) {
    const previous = out.at(-1);
    if (previous && s.ts - previous.ts > GAP_STEPS * step)
      out.push({ ts: previous.ts + step, gap: true, values: zero });
    const raw = valuesOf(s, key);
    const values: Record<string, number> = { ...zero };
    for (const [name, value] of Object.entries(raw)) {
      if (names.includes(name)) values[name] = value;
      else values[OTHER] = (values[OTHER] ?? 0) + value;
    }
    out.push({ ts: s.ts, gap: false, values, sample: s });
  }
  return out;
}

/** Every service's value at a sample for the hover, busiest first, Everything else last; zeros left out. */
export function breakdown(sample: UsageSample, key: string): { name: string; value: number }[] {
  const raw = valuesOf(sample, key);
  const listed = Object.entries(raw)
    .filter(([name, value]) => name !== OTHER && value > 0)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([name, value]) => ({ name, value }));
  return raw[OTHER] ? [...listed, { name: OTHER, value: raw[OTHER] }] : listed;
}

const NAMES: Record<string, string> = { [OTHER]: 'Everything else', ollama: 'Ollama' };
export const nameOf = (service: string): string => NAMES[service] ?? service;
