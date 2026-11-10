import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Providers } from '../shared/Providers';
import { startTelemetry } from '../shared/telemetry';
import { App } from './App';
import { followPreloadErrors } from './lib/staleChunk';

startTelemetry(); // errors, failed and slow requests, and slow navigations, to logs/frontend.log (#2007)
followPreloadErrors(); // a chunk a redeploy removed reloads once for the new build (#2005)

const root = document.getElementById('app');
if (!root) throw new Error('index.html has no #app element');
createRoot(root).render(
  <StrictMode>
    <Providers>
      <App />
    </Providers>
  </StrictMode>,
);
// The page shell's loading fallback waits for this before it goes away.
window.dispatchEvent(new Event('liberational:ready'));
