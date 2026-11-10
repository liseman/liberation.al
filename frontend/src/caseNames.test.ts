// macOS checkouts are case-insensitive: './UsageChart' resolved to usageChart.ts there and the dashboard build
// failed (and again for usage/). Two sources whose names differ only in case must not share a folder.
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { expect, it } from 'vitest';

const SOURCE = /\.(ts|tsx|js|jsx|mts|mjs)$/;

function clashes(dir: string): string[] {
  const out: string[] = [];
  const seen = new Map<string, string>();
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      out.push(...clashes(path));
      continue;
    }
    if (!SOURCE.test(name)) continue;
    const stem = name.replace(SOURCE, '').replace(/\.test$/, '');
    const key = stem.toLowerCase();
    const other = seen.get(key);
    if (other !== undefined && other !== stem) out.push(join(dir, stem) + ' vs ' + other);
    else seen.set(key, stem);
  }
  return out;
}

it('no two source files in a folder differ only in case (they collide on macOS)', () => {
  expect(clashes(join(__dirname))).toEqual([]);
});
