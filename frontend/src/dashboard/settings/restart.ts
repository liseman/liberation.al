// Follow a configuration restart without leaving a poll alive after the form unmounts.
import { request } from '../../shared/http';

export const RESTART_DEADLINE_MS = 120_000;

function pause(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();
    const cancel = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', cancel);
      resolve();
    }, 700);
    signal.addEventListener('abort', cancel, { once: true });
  });
}

/** Resolve only after the destination reports a different boot, or reject on cancellation/deadline. */
export async function followRestart(oldBoot: string, url: string, signal: AbortSignal): Promise<void> {
  const controller = new AbortController();
  const cancel = () => controller.abort(signal.reason);
  signal.addEventListener('abort', cancel, { once: true });
  if (signal.aborted) cancel();
  const deadline = setTimeout(
    () =>
      controller.abort(
        new Error(
          'The dashboard has not come back after 120 seconds. Check its terminal or log, then reload this page.',
        ),
      ),
    RESTART_DEADLINE_MS,
  );
  try {
    while (true) {
      await pause(controller.signal);
      try {
        const health = await request<{ boot?: string }>(
          new URL('/health', url).href,
          undefined,
          controller.signal,
          'page',
          false, // this loop is the wait for the restart
        );
        if (health.boot && health.boot !== oldBoot) return;
      } catch {
        // A refused connection while the old process exits is expected. Cancellation still ends the poll.
        controller.signal.throwIfAborted();
      }
    }
  } finally {
    clearTimeout(deadline);
    signal.removeEventListener('abort', cancel);
  }
}
