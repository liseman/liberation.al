import { expect, it } from 'vitest';
import { iconCollections, rebase } from '../../vite.pascal.ts';

it("moves the editor's root paths under the build's base, and nothing else", () => {
  const code = [
    "const a = '/icons/wall.webp';",
    'const b = "/audios/click.mp3";',
    'const c = "url(\'/cursor.svg\') 4 2, default";',
    "const d = '/places/icons/x';",
    "const e = 'https://example.com/icons/y.webp';",
    'const f = `/icons/door.webp`;',
  ].join('\n');
  expect(rebase(code, '/static/app/').split('\n')).toEqual([
    "const a = '/static/app/pascal/icons/wall.webp';",
    'const b = "/static/app/pascal/audios/click.mp3";',
    'const c = "url(\'/static/app/pascal/cursor.svg\') 4 2, default";',
    "const d = '/places/icons/x';",
    "const e = 'https://example.com/icons/y.webp';",
    'const f = `/static/app/pascal/icons/door.webp`;',
  ]);
});

it("gathers the icons the editor's code names, so none is fetched from the icon service", () => {
  const collections = iconCollections() as { prefix: string; icons: Record<string, unknown> }[];
  const lucide = collections.find(c => c.prefix === 'lucide');
  expect(lucide && Object.keys(lucide.icons).length).toBeGreaterThan(10);
  expect(lucide?.icons.house).toBeTruthy(); // one the editor's source names
  for (const c of collections) expect(Object.keys(c.icons).length).toBeGreaterThan(0);
});
