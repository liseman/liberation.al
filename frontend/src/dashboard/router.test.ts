import { waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { closeSettings, followHistory, knownTabOf, navigate, tabOf, useRoute } from './router';

afterEach(() => history.pushState(null, '', '/'));

describe('tabOf', () => {
  it.each([
    ['/', 'live'],
    ['/review', 'review'],
    ['/people/', 'known'],
    ['/pets', 'known'],
    ['/calls', 'calls'],
    ['/calls/', 'calls'],
    ['/ingest', 'known'],
    ['/statistics/', 'statistics'],
    ['/reports', 'live'],
    ['/settings', 'settings'],
    ['/debug', 'debug'],
    ['/skeletons', 'skeletons'],
    ['/objects/', 'known'],
    ['/nowhere', 'live'],
  ])('%s is %s', (path, tab) => {
    expect(tabOf(path)).toBe(tab);
  });
});

describe('Known', () => {
  it.each([
    ['/people', 'people'],
    ['/pets/', 'pets'],
    ['/objects', 'objects'],
    ['/review', null],
    ['/ingest', 'ingest'],
  ])('%s is the %s tab', (path, tab) => {
    expect(knownTabOf(path)).toBe(tab);
  });

  it('keeps the tab its path last named while another page shows, and Back returns to it', async () => {
    const stop = followHistory();
    navigate('/pets');
    expect(useRoute.getState()).toMatchObject({ tab: 'known', known: 'pets' });
    navigate('/review');
    expect(useRoute.getState()).toMatchObject({ tab: 'review', known: 'pets' });
    history.back();
    await waitFor(() => expect(useRoute.getState()).toMatchObject({ tab: 'known', known: 'pets' }));
    stop();
  });
});

describe('Known under Settings', () => {
  it("takes the tab from Settings' return route after a reload", async () => {
    history.replaceState({ settingsReturnTo: '/objects#x', settingsFromApp: true }, '', '/settings');
    vi.resetModules(); // a reload: the router module starts over with no memory of the last tab
    const fresh = await import('./router');
    expect(fresh.useRoute.getState()).toMatchObject({ tab: 'settings', known: 'objects' });
  });

  it('takes it from the return route again when Back reaches Settings from another tab', async () => {
    const stop = followHistory();
    navigate('/pets');
    navigate('/settings');
    navigate('/people');
    expect(useRoute.getState().known).toBe('people');
    history.back();
    await waitFor(() => expect(useRoute.getState()).toMatchObject({ tab: 'settings', known: 'pets' }));
    stop();
  });
});

describe('navigation', () => {
  it('changes the URL and the tab without a page load, keeping the hash', () => {
    let hashChanges = 0;
    const onHash = () => hashChanges++;
    window.addEventListener('hashchange', onHash);
    navigate('/review#day=2026-10-28');
    window.removeEventListener('hashchange', onHash);
    expect(location.pathname + location.hash).toBe('/review#day=2026-10-28');
    expect(useRoute.getState().tab).toBe('review');
    expect(hashChanges).toBe(1);
  });

  it('closes repeated Settings deep links back to their original view and supports Forward', async () => {
    const stop = followHistory();
    navigate('/review#day=2026-10-28');
    navigate('/settings#services');
    navigate('/settings#advanced');
    expect(useRoute.getState().settingsReturnTo).toBe('/review#day=2026-10-28');
    closeSettings();
    await waitFor(() => expect(useRoute.getState().tab).toBe('review'));
    expect(location.hash).toBe('#day=2026-10-28');
    history.forward();
    await waitFor(() => expect(useRoute.getState().tab).toBe('settings'));
    expect(location.hash).toBe('#advanced');
    stop();
  });

  it('follows Back and Forward until stopped', () => {
    const stop = followHistory();
    history.pushState(null, '', '/people');
    window.dispatchEvent(new PopStateEvent('popstate'));
    expect(useRoute.getState().tab).toBe('known');
    stop();
    history.pushState(null, '', '/review');
    window.dispatchEvent(new PopStateEvent('popstate'));
    expect(useRoute.getState().tab).toBe('known');
  });
});
