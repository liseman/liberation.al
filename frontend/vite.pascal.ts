// What building the plan editor's page (plan-editor/index.html) takes. Pascal Editor (@pascal-app/editor) ships as TSX
// written for a Next host with Tailwind, and expects its static files at the site's root and a few more on public
// CDNs. This plugin makes it build here and stay on our own origin:
//
// - `next/image` and `next/link` resolve to two small stand-ins (src/plan-editor/shims);
// - the root paths its source names (`/icons/...`, `/audios/...`) are rewritten to where the dashboard serves them
//   (`<base>pascal/...`, from public/pascal/);
// - the Iconify icons its source names are gathered into `virtual:pascal-icons`, registered at start, so none is
//   fetched from the Iconify API;
// - three's Draco and Basis decoders are served at `<base>pascal/decoders/` (copied from node_modules into the
//   build, and answered by the dev server), instead of from the CDNs the loaders default to.
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import type { Plugin } from 'vite';

const here = import.meta.dirname;
const require = createRequire(join(here, 'package.json'));
const modules = join(here, 'node_modules');

/** The Pascal packages whose code names icons and root paths: the editor's source, and the built ones. */
const SOURCES = [
  '@pascal-app/editor/src',
  '@pascal-app/nodes/dist',
  '@pascal-app/viewer/dist',
  '@pascal-app/core/dist',
];
const ICON_SETS = ['lucide', 'ph', 'qlementine-icons', 'mdi'] as const;
const ICON_NAME = /['"`](lucide|ph|qlementine-icons|mdi):([a-z0-9-]+)['"`]/g;
/** A root path to one of the editor's static files (a folder of them, or its cursor), where a string or a CSS
 *  `url('...')` starts. */
const ROOT_PATH = /(['"`])\/(icons\/|audios\/|cursor\.svg)/g;
const ICONS_ID = 'virtual:pascal-icons';

/** The decoders three's loaders fetch at run time: the folder under `pascal/decoders/`, and where it comes from. */
const DECODERS: Record<string, string> = {
  draco: 'three/examples/jsm/libs/draco/gltf',
  basis: 'three/examples/jsm/libs/basis',
};

function files(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) files(path, out);
    else if (/\.(tsx?|js)$/.test(name) && !/\.(test|d)\.[a-z]+$/.test(name)) out.push(path);
  }
  return out;
}

/** The Iconify collections holding just the icons Pascal's code names, as `addCollection` takes them. */
export function iconCollections(): unknown[] {
  const { getIcons } = require('@iconify/utils') as { getIcons: (set: unknown, names: string[]) => unknown };
  const names = new Map<string, Set<string>>(ICON_SETS.map(set => [set, new Set()]));
  for (const source of SOURCES)
    for (const file of files(join(modules, source)))
      for (const m of readFileSync(file, 'utf8').matchAll(ICON_NAME)) names.get(m[1] as string)?.add(m[2] as string);
  return ICON_SETS.map(set => getIcons(require(`@iconify-json/${set}/icons.json`), [...(names.get(set) ?? [])])).filter(
    Boolean,
  );
}

/** `code` with the editor's root paths moved under `base` (the build's own base, ending in a slash). */
export function rebase(code: string, base: string): string {
  return code.replace(ROOT_PATH, (_, quote: string, path: string) => `${quote}${base}pascal/${path}`);
}

function decoderFile(path: string): string | null {
  const [kind, name, ...rest] = path.split('/');
  const from = kind ? DECODERS[kind] : undefined;
  if (!from || !name || rest.length || !/^[\w.-]+\.(js|wasm)$/.test(name)) return null;
  const file = join(modules, from, name);
  return existsSync(file) ? file : null;
}

/** `decoders`: copy the decoders into this build (the plan editor's own build asks for it; the dev server answers
 *  for them either way). */
export function pascal({ decoders = false }: { decoders?: boolean } = {}): Plugin {
  let base = '/';
  return {
    name: 'pascal-editor-host',
    enforce: 'pre',
    config: () => ({
      resolve: {
        alias: {
          'next/image': resolve(here, 'src/plan-editor/shims/next-image.tsx'),
          'next/link': resolve(here, 'src/plan-editor/shims/next-link.tsx'),
        },
      },
    }),
    configResolved(config) {
      base = config.base;
    },
    resolveId: id => (id === ICONS_ID ? `\0${ICONS_ID}` : null),
    load: id => (id === `\0${ICONS_ID}` ? `export default ${JSON.stringify(iconCollections())};` : null),
    transform(code, id) {
      if (!id.includes('/@pascal-app/') || !ROOT_PATH.test(code)) return null;
      ROOT_PATH.lastIndex = 0;
      return { code: rebase(code, base), map: null };
    },
    configureServer(server) {
      const prefix = `${base}pascal/decoders/`;
      server.middlewares.use((req, res, next) => {
        const path = (req.url ?? '').split('?')[0] ?? '';
        const file = path.startsWith(prefix) ? decoderFile(path.slice(prefix.length)) : null;
        if (!file) return next();
        res.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
        res.end(readFileSync(file));
      });
    },
    generateBundle() {
      if (!decoders) return;
      for (const [kind, from] of Object.entries(DECODERS))
        for (const name of readdirSync(join(modules, from)))
          if (/\.(js|wasm)$/.test(name))
            this.emitFile({
              type: 'asset',
              fileName: `pascal/decoders/${kind}/${name}`,
              source: readFileSync(join(modules, from, name)),
            });
    },
  };
}
