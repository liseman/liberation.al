// Builds the dashboard frontend into the Python package, where the dashboard serves it
// at /static/app/. `mise run frontend` builds it; `mise run dev` runs Vite's dev server next to the dashboard.
import { resolve } from 'node:path';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

const here = import.meta.dirname;
// ports.vite in config/defaults.toml; `mise run dev` exports it. --port alone wouldn't move HMR's client port.
const port = Number(process.env.VITE_PORT ?? 5173);

export default defineConfig({
  base: '/static/app/',
  plugins: [react()],
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
    testTimeout: 30_000, // a loaded shared machine, not a slow test: see asyncUtilTimeout in src/test/setup.ts (#1321)
    maxWorkers: 4, // a shared machine: AGENTS.md "Share the machine"
  },
});
