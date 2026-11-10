import { beforeEach, describe, expect, it, vi } from 'vitest';
import { reportError } from '../../shared/telemetry';
import {
  claimReload,
  followPreloadErrors,
  isChunkLoadError,
  RELOAD_GAP_MS,
  reloadForNewBuild,
  resetReloading,
} from './staleChunk';

vi.mock('../../shared/telemetry', async importOriginal => {
  const real = await importOriginal<typeof import('../../shared/telemetry')>();
  return { ...real, reportError: vi.fn() };
});

/** A session storage of its own, so tests don't share the real one. */
function storage(): Pick<Storage, 'getItem' | 'setItem'> {
  const kept = new Map<string, string>();
  return { getItem: k => kept.get(k) ?? null, setItem: (k, v) => void kept.set(k, v) };
}

// #2005: a tab left open across a redeploy asks for chunks the build removed.
describe('isChunkLoadError', () => {
  it.each([
    ['Chrome', new TypeError('Failed to fetch dynamically imported module: /static/app/assets/Known-OLD.js')],
    ['Firefox', new TypeError('error loading dynamically imported module: /static/app/assets/Known-OLD.js')],
    ['Safari', new TypeError('Importing a module script failed.')],
    ['a CSS preload', new Error('Unable to preload CSS for /static/app/assets/Known-OLD.css')],
  ])("knows %s's failed chunk import", (_browser, error) => {
    expect(isChunkLoadError(error)).toBe(true);
  });

  it.each([new Error('Could not load people: 500'), 'a string', null])('leaves anything else alone: %s', error => {
    expect(isChunkLoadError(error)).toBe(false);
  });
});

describe('claimReload', () => {
  it('allows one reload, then none until the gap has passed', () => {
    const kept = storage();
    expect(claimReload(1_000_000, kept)).toBe(true);
    expect(claimReload(1_000_000 + RELOAD_GAP_MS - 1, kept)).toBe(false);
    expect(claimReload(1_000_000 + RELOAD_GAP_MS, kept)).toBe(true);
  });

  it('never reloads without session storage, since nothing could stop a loop', () => {
    const refusing = {
      getItem: () => {
        throw new DOMException('denied', 'SecurityError');
      },
      setItem: () => {},
    };
    expect(claimReload(1_000_000, refusing)).toBe(false);
  });
});

// #2054, from Joe's frontend log: Vite's preload error came first and its handler claimed the reload; Vite then
// resolved the import with nothing, and the lazy wrapper's "reading 'default'" reached the page's boundary.
describe('while a reload is under way', () => {
  beforeEach(() => {
    sessionStorage.clear();
    resetReloading();
  });

  it("keeps the loader for the lazy wrapper's error after Vite's preload error, without reloading twice", () => {
    const reload = vi.fn();
    const stop = followPreloadErrors(window, reload);
    const event = new Event('vite:preloadError', { cancelable: true });
    window.dispatchEvent(event);
    stop();
    expect(event.defaultPrevented).toBe(true);
    expect(reload).toHaveBeenCalledOnce();
    const again = vi.fn();
    const wrapper = new TypeError("Cannot read properties of undefined (reading 'default')");
    expect(reloadForNewBuild(wrapper, again)).toBe(true); // the boundary shows "Loading the new version"
    expect(again).not.toHaveBeenCalled();
  });

  it('shows that same error as a failure when no reload is under way', () => {
    const reload = vi.fn();
    expect(reloadForNewBuild(new TypeError("Cannot read properties of undefined (reading 'default')"), reload)).toBe(
      false,
    );
    expect(reload).not.toHaveBeenCalled();
  });
});

// Joe's frontend log filled with "Failed to fetch dynamically imported module" from tabs that then reloaded fine.
describe('what a preload error logs', () => {
  beforeEach(() => {
    sessionStorage.clear();
    resetReloading();
  });

  const preloadError = (payload: Error) => {
    const event = Object.assign(new Event('vite:preloadError', { cancelable: true }), { payload });
    window.dispatchEvent(event);
    return event;
  };

  it('logs nothing when it reloads for the new build', () => {
    const reload = vi.fn();
    const stop = followPreloadErrors(window, reload);
    preloadError(new TypeError('Failed to fetch dynamically imported module: /static/app/assets/Known-OLD.js'));
    stop();
    expect(reload).toHaveBeenCalledOnce();
    expect(reportError).not.toHaveBeenCalled();
  });

  it('logs the failure when a reload just happened, so the import fails for real', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(claimReload()).toBe(true); // the reload a moment ago
    resetReloading(); // this is the page that reload loaded
    const reload = vi.fn();
    const stop = followPreloadErrors(window, reload);
    const failed = new TypeError('Failed to fetch dynamically imported module: /static/app/assets/Known-NEW.js');
    const event = preloadError(failed);
    stop();
    expect(event.defaultPrevented).toBe(false); // Vite rethrows it, so the page's boundary shows it
    expect(reload).not.toHaveBeenCalled();
    expect(reportError).toHaveBeenCalledWith('chunk', failed);
  });
});
