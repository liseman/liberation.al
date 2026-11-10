// What every page renders inside: Mantine with our theme, one TanStack Query client for server data, and Mantine's
// modals manager (modals.openConfirmModal from @mantine/modals, in place of window.confirm).
import '@mantine/core/styles.css';
import '@mantine/dates/styles.css';
import '@mantine/notifications/styles.css';
import './fonts.css';
import './theme.css';
import { MantineProvider } from '@mantine/core';
import { ModalsProvider } from '@mantine/modals';
import { Notifications } from '@mantine/notifications';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { type ReactNode, useState } from 'react';
import { isNetworkFailure } from './http';
import { prefsColorSchemeManager } from './prefs';
import { cssVariablesResolver, theme } from './theme';

// Data a tab read in the last 30 s is shown again without asking the server (a tab switch, a dialog opened twice).
// Queries that must be fresher poll (refetchInterval) or are invalidated by what changes them; the live feed sets its
// own data. Data nobody shows is kept 10 minutes, so a tab hidden that long still opens on what it had (#330).
export const STALE_MS = 30_000;
export const KEEP_MS = 10 * 60_000;

export function newQueryClient(): QueryClient {
  // The server is local, so a failed request is worth one quick retry, not three slow ones. A network failure has
  // already waited out a restart inside request (shared/http.ts), so it isn't retried again.
  const retry = (failures: number, error: Error) => failures < 1 && !isNetworkFailure(error);
  return new QueryClient({
    defaultOptions: { queries: { retry, refetchOnWindowFocus: true, staleTime: STALE_MS, gcTime: KEEP_MS } },
  });
}

export function Providers({
  children,
  client,
  test = false,
}: {
  children: ReactNode;
  client?: QueryClient;
  test?: boolean; // Mantine's test mode: no transitions or portals, for jsdom
}) {
  const [queryClient] = useState(() => client ?? newQueryClient());
  const [manager] = useState(prefsColorSchemeManager);
  return (
    <MantineProvider
      theme={theme}
      cssVariablesResolver={cssVariablesResolver}
      colorSchemeManager={manager}
      defaultColorScheme="auto"
      env={test ? 'test' : 'default'}
    >
      <QueryClientProvider client={queryClient}>
        <ModalsProvider>{children}</ModalsProvider>
        {/* Every page's toasts (lib/Toast, #2596): bottom-centre, over the page and the Ask window (150), under
            Mantine's modals (200) */}
        <Notifications
          position="bottom-center"
          zIndex={160}
          limit={4}
          containerWidth={560}
          transitionDuration={test ? 0 : undefined}
        />
      </QueryClientProvider>
    </MantineProvider>
  );
}
