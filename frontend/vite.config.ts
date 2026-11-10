// Builds the dashboard frontend into the Python package, where the dashboard serves it
// at /static/app/. `mise run frontend` builds it; `mise run dev` runs Vite's dev server next to the dashboard.
import { resolve } from 'node:path';
import tailwind from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
import { pascal } from './vite.pascal.ts';

const here = import.meta.dirname;
// ports.vite in config/defaults.toml; `mise run dev` exports it. --port alone wouldn't move HMR's client port.
const port = Number(process.env.VITE_PORT ?? 5173);

export default defineConfig({
  base: '/static/app/',
  // tailwind and pascal() are for the plan editor's page (plan-editor/index.html), which the dev server serves from
  // this config; its build is its own (vite.plan-editor.config.ts), so this build's chunks are the dashboard's alone
  plugins: [react(), tailwind(), pascal()],
  build: {
    outDir: resolve(here, '../src/liberational/dashboard/static/app'),
    emptyOutDir: true,
    manifest: true, // .vite/manifest.json: the dev-mode check for a stale build reads its time
    sourcemap: true,
    rolldownOptions: {
      input: { index: resolve(here, 'index.html') },
    },
  },
  // In dev the dashboard serves the pages and proxies /static/app/ here; HMR's websocket comes straight to this port.
  server: { port, strictPort: true, hmr: { clientPort: port } },
  test: {
    environment: 'jsdom',
    setupFiles: ['src/test/setup.ts'],
    css: false,
    restoreMocks: true,
    testTimeout: 30_000, // a loaded shared machine, not a slow test: see asyncUtilTimeout in src/test/setup.ts
    maxWorkers: 4, // a shared machine: AGENTS.md "Share the machine"
  },
});
