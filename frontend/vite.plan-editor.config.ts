// Builds the plan editor's page (plan-editor/index.html: Pascal Editor, src/plan-editor/) into the dashboard's build,
// after it (`npm run build`). It is a build of its own, not a second page of vite.config.ts, so that the dashboard's
// chunks are exactly what they are without it: two pages in one build share chunks, which split the dashboard's
// start into twice as many files. The page lands at /static/app/plan-editor/index.html, its scripts beside the
// dashboard's in assets/ (content-hashed, so they can't collide), and the decoders in pascal/decoders/.
import { resolve } from 'node:path';
import tailwind from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { pascal } from './vite.pascal.ts';

const here = import.meta.dirname;

export default defineConfig({
  base: '/static/app/',
  plugins: [react(), tailwind(), pascal({ decoders: true })],
  publicDir: false, // public/ went out with the dashboard's build
  build: {
    outDir: resolve(here, '../src/liberational/dashboard/static/app'),
    emptyOutDir: false, // the dashboard's build is already there
    target: 'esnext', // the page's entry awaits at its top level
    sourcemap: false, // 14 MB of someone else's code: its maps would double the package
    chunkSizeWarningLimit: 4000,
    rolldownOptions: {
      input: { 'plan-editor': resolve(here, 'plan-editor/index.html') },
    },
  },
});
