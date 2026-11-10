import { describe, expect, it, vi } from 'vitest';
import { PREFS_KEY, prefsColorSchemeManager, readPrefs, schemeFor } from './prefs';

describe('schemeFor', () => {
  it.each([
    ['light', 'light'],
    ['dark', 'dark'],
    ['system', 'auto'],
    [undefined, 'auto'],
    ['sepia', 'auto'],
  ])('%s gives %s', (theme, scheme) => {
    expect(schemeFor(theme)).toBe(scheme);
  });
});

describe('readPrefs', () => {
  it('falls back to no prefs when storage holds junk', () => {
    localStorage.setItem(PREFS_KEY, '{not json');
    expect(readPrefs()).toEqual({});
    localStorage.setItem(PREFS_KEY, '[1, 2]');
    expect(readPrefs()).toEqual({});
  });
});

describe('prefsColorSchemeManager', () => {
  it('reads the theme the old dashboard saved', () => {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ theme: 'dark' }));
    expect(prefsColorSchemeManager().get('auto')).toBe('dark');
  });

  it('saves auto as system, keeping the other prefs', () => {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ theme: 'light', hotkeys: true }));
    prefsColorSchemeManager().set('auto');
    expect(JSON.parse(localStorage.getItem(PREFS_KEY) ?? '')).toEqual({ theme: 'system', hotkeys: true });
  });

  it('clear drops only the theme', () => {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ theme: 'dark', hotkeys: true }));
    prefsColorSchemeManager().clear();
    expect(readPrefs()).toEqual({ hotkeys: true });
  });

  it('follows a theme changed in another tab until unsubscribed', () => {
    const manager = prefsColorSchemeManager();
    const onUpdate = vi.fn();
    manager.subscribe(onUpdate);
    localStorage.setItem(PREFS_KEY, JSON.stringify({ theme: 'light' }));
    window.dispatchEvent(new StorageEvent('storage', { key: PREFS_KEY }));
    window.dispatchEvent(new StorageEvent('storage', { key: 'something.else' }));
    expect(onUpdate.mock.calls).toEqual([['light']]);
    manager.unsubscribe();
    window.dispatchEvent(new StorageEvent('storage', { key: PREFS_KEY }));
    expect(onUpdate).toHaveBeenCalledTimes(1);
  });
});
