import type { QueryClient } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import type { ReactElement } from 'react';
import { newQueryClient, Providers } from '../shared/Providers';

/** Every query client renderPage made in this test, for the teardown (setup.ts) to cancel and clear. */
const clients = new Set<QueryClient>();

/** Cancel each rendered client's reads still in flight and drop its cache. Unmounting cancels only the reads a mounted
 *  component was watching: one started with fetchQuery (Review's camera notes) runs on, and with it the shared request
 *  gate's slot (shared/http.ts has two), so the next test's reads queued behind it. */
export async function releaseClients(): Promise<void> {
  const all = [...clients];
  clients.clear();
  await Promise.all(all.map(c => c.cancelQueries()));
  for (const c of all) c.clear();
}

/** Render ui inside the app's providers, with a fresh query client that never retries. */
export function renderPage(ui: ReactElement) {
  const client = newQueryClient();
  clients.add(client);
  client.setDefaultOptions({ queries: { retry: false }, mutations: { retry: false } });
  return {
    client,
    ...render(
      <Providers client={client} test>
        {ui}
      </Providers>,
    ),
  };
}
