import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { server } from './server';

// This expected-failure probe has no failing assertion: only the shared teardown
// can reject it. Removing the strict guard makes Vitest report "Expect test to fail".
it.fails('fails an unhandled request even when the caller catches its fetch rejection', async () => {
  await fetch('/__msw_strict_probe__').catch(() => undefined);
});

it.fails('fails a fire-and-forget unhandled request before the test finishes', () => {
  void fetch('/__msw_unawaited_probe__').catch(() => undefined);
});

it('clears the previous test failure and accepts an explicitly handled request', async () => {
  server.use(http.get('/__msw_strict_probe__', () => HttpResponse.json({ handled: true })));
  const response = await fetch('/__msw_strict_probe__');
  expect(await response.json()).toEqual({ handled: true });
});
