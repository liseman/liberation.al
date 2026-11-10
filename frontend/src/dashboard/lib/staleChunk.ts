import { reportError } from '../../shared/telemetry';

// A tab left open across a redeploy: `vite build` empties its output, so the old hashed chunks the tab still
// asks for are gone, and the first visit to a page it hasn't loaded yet fails its dynamic import (a 404). The page
// itself is served no-store, so one reload fetches the new build. Reload once, never in a loop: a chunk that fails
// again within RELOAD_GAP_MS of the last reload is a real failure, and its page says so.

/** The least time between two automatic reloads. */
export const RELOAD_GAP_MS = 60_000;
const KEY = 'liberational:chunk-reload-at';

// A reload this page already started. Vite's preload handler claims it first, and handling its event makes Vite
// resolve the import with nothing, so the page's lazy wrapper then fails with "Cannot read properties of undefined
// (reading 'default')", which is no chunk error. While the page is on its way out, any boundary error is
// part of it: show the loader, not a failure.
let reloading = false;

/** For tests: forget that a reload started. */
export function resetReloading(): void {
  reloading = false;
}

/** Whether an error is a failed dynamic import of a code chunk, as each browser words it. */
export function isChunkLoadError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  return (
    error.name === 'ChunkLoadError' ||
    /dynamically imported module|Importing a module script failed|module script failed|Unable to preload CSS/i.test(
      error.message,
    )
  );
}

/** Whether an automatic reload may happen now (none in the last RELOAD_GAP_MS), recording it if so. Without session
 *  storage (a private window that refuses it) it never reloads, since nothing could stop a loop. */
export function claimReload(now = Date.now(), storage: Pick<Storage, 'getItem' | 'setItem'> = sessionStorage): boolean {
  try {
    const last = Number(storage.getItem(KEY));
    if (Number.isFinite(last) && last > 0 && now - last < RELOAD_GAP_MS) return false;
    storage.setItem(KEY, String(now));
    return true;
  } catch {
    return false;
  }
}

/** Reload for the new build if `error` is a failed chunk import and no reload happened just now. Returns whether it
 *  reloaded, so a caller can show its own error otherwise. */
export function reloadForNewBuild(error: unknown, reload: () => void = () => location.reload()): boolean {
  if (reloading) return true; // already reloading: this error is that reload's side effect
  if (!isChunkLoadError(error) || !claimReload()) return false;
  reloading = true;
  reload();
  return true;
}

/** Vite's own signal for a failed preload of a dynamic import's dependencies: reload once instead of failing. */
export function followPreloadErrors(target: Window = window, reload: () => void = () => location.reload()): () => void {
  const onError = (event: Event) => {
    const payload = (event as Event & { payload?: unknown }).payload;
    if (reloading || claimReload()) {
      event.preventDefault(); // handled: Vite won't rethrow it, and resolves the import with nothing
      if (!reloading) {
        reloading = true;
        reload();
      }
    } else {
      // No reload this time, so the import fails for real: that goes into the page's log. A preload error
      // that reloads isn't logged, since the reload is its recovery.
      reportError('chunk', payload ?? 'preload failed');
      if (payload instanceof Error) console.error(payload);
    }
  };
  target.addEventListener('vite:preloadError', onError);
  return () => target.removeEventListener('vite:preloadError', onError);
}
