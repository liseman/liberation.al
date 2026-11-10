// Display preferences, kept in this browser only under the key the old dashboard used, so choices carry over.
import type { MantineColorScheme, MantineColorSchemeManager } from '@mantine/core';

export const PREFS_KEY = 'liberational.prefs';

export type ThemePref = 'system' | 'light' | 'dark';

export interface Prefs {
  theme?: ThemePref;
  [key: string]: unknown;
}

export function readPrefs(): Prefs {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as Prefs) : {};
  } catch {
    return {}; // storage can be unavailable (private mode) or hold junk; the defaults still work
  }
}

export function writePrefs(prefs: Prefs): void {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    // the choice still holds until the page reloads
  }
}

/** Mantine's colour scheme for a theme preference: 'system' (or anything unknown) follows the OS. */
export const schemeFor = (theme: unknown): MantineColorScheme =>
  theme === 'light' || theme === 'dark' ? theme : 'auto';

/** Keeps Mantine's colour scheme in prefs.theme, and follows changes made in other tabs. */
export function prefsColorSchemeManager(): MantineColorSchemeManager {
  let listener: ((e: StorageEvent) => void) | null = null;
  return {
    get: () => schemeFor(readPrefs().theme),
    set: value => writePrefs({ ...readPrefs(), theme: value === 'auto' ? 'system' : value }),
    subscribe: onUpdate => {
      listener = e => {
        if (e.key === PREFS_KEY) onUpdate(schemeFor(readPrefs().theme));
      };
      window.addEventListener('storage', listener);
    },
    unsubscribe: () => {
      if (listener) window.removeEventListener('storage', listener);
      listener = null;
    },
    clear: () => {
      const { theme: _, ...rest } = readPrefs();
      writePrefs(rest);
    },
  };
}
