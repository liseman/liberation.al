// A Mac's filesystem ignores case, so two modules in one folder whose import names differ only by case (Chips.tsx
// and chips.ts, both imported without an extension) resolve to either one there, and the build broke on a Mac while
// passing on Linux (2026-10-28). This keeps any such pair out of the tree.
import { readdirSync } from 'node:fs';
import { join, relative } from 'node:path';
import { expect, it } from 'vitest';

const SRC = join(import.meta.dirname, '..');
const MODULE = /\.(tsx|ts|jsx|js|mts)$/; // imported without their extension
const TEST = /\.test\.[a-z]+$/;

it('has no two modules in a folder whose import names differ only by case', () => {
  const clashes: string[] = [];
  const walk = (dir: string) => {
    const seen = new Map<string, string>();
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        walk(join(dir, entry.name));
        continue;
      }
      if (!MODULE.test(entry.name) || TEST.test(entry.name)) continue;
      const name = entry.name.replace(MODULE, '');
      const other = seen.get(name.toLowerCase());
      if (other && other !== name) clashes.push(`${relative(SRC, dir)}: ${other} and ${name}`);
      seen.set(name.toLowerCase(), name);
    }
  };
  walk(SRC);
  expect(clashes).toEqual([]);
});
