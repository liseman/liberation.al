// Tabs are real paths, so links and reloads land on the right tab. /settings is a route without a tab. Known (#1970)
// is one page whose own tabs (People, Pets, Objects) are paths too, so their links, reloads, and Back work the same.
import type { MouseEvent } from 'react';
import { create } from 'zustand';

export type TabId =
  | 'live'
  | 'review'
  | 'journal'
  | 'known'
  | 'calls'
  | 'inbox'
  | 'reports'
  | 'ingest'
  | 'debug'
  | 'skeletons'
  | 'settings';

export const TABS: { id: Exclude<TabId, 'settings'>; path: string; label: string }[] = [
  { id: 'live', path: '/', label: 'Live' },
  { id: 'review', path: '/review', label: 'Review' },
  { id: 'journal', path: '/journal', label: 'Journal' },
  { id: 'known', path: '/people', label: 'Known' },
  { id: 'calls', path: '/calls', label: 'Calls' },
  { id: 'inbox', path: '/inbox', label: 'Inbox' },
  { id: 'reports', path: '/reports', label: 'Reports' },
  { id: 'ingest', path: '/ingest', label: 'Ingest' },
  { id: 'debug', path: '/debug', label: 'Debug' },
];

/** Known's tabs: who and what the house knows by name. Each is its own path; the first is where Known opens. */
export const KNOWN_TABS = [
  { id: 'people', path: '/people', label: 'People' },
  { id: 'pets', path: '/pets', label: 'Pets' },
  { id: 'objects', path: '/objects', label: 'Objects' },
  { id: 'activities', path: '/activities', label: 'Activities' },
  { id: 'devices', path: '/devices', label: 'Devices' },
  { id: 'places', path: '/places', label: 'Places' },
] as const;
export type KnownTab = (typeof KNOWN_TABS)[number]['id'];

/** Which of Known's tabs a path is, or null for a path outside Known. */
export function knownTabOf(pathname: string): KnownTab | null {
  const rest = pathname.replace(/\/+$/, '') || '/';
  return KNOWN_TABS.find(t => t.path === rest)?.id ?? null;
}

// /skeletons is a test page (#1987) with no tab in the nav.
const ALIASES: Record<string, TabId> = { '/settings': 'settings', '/skeletons': 'skeletons' };

/** The URL of an in-app path. Every in-app link goes through here, so a prefix (like the /next the app was served
 *  at while it was ported) would only ever need adding in one place. */
export const href = (path: string): string => path;

export function tabOf(pathname: string): TabId {
  const rest = pathname.replace(/\/+$/, '') || '/';
  if (knownTabOf(rest)) return 'known';
  return ALIASES[rest] ?? TABS.find(t => t.path === rest)?.id ?? 'live';
}

interface Route {
  tab: TabId;
  /** Known's tab: the last one its path named, so it stays put while another page (or Settings) shows. */
  known: KnownTab;
  settingsReturnTo?: string;
}

let lastKnown: KnownTab = 'people';

function currentRoute(): Route {
  const returnTo: string | undefined = history.state?.settingsReturnTo;
  // Under Settings, the page behind the drawer is its return route, so Known's tab comes from that path, which
  // survives a reload and Back, where this module's memory doesn't.
  const shown =
    tabOf(location.pathname) === 'settings' && returnTo
      ? (returnTo.split(/[?#]/)[0] ?? location.pathname)
      : location.pathname;
  lastKnown = knownTabOf(shown) ?? lastKnown;
  return {
    tab: tabOf(location.pathname),
    known: lastKnown,
    settingsReturnTo: returnTo,
  };
}

export const useRoute = create<Route>(currentRoute);

/** Go to an in-app URL (path plus optional hash) without a page load. */
export function navigate(url: string): void {
  const u = new URL(url, location.href);
  const settings = tabOf(u.pathname) === 'settings';
  const alreadyOpen = tabOf(location.pathname) === 'settings';
  const state = settings
    ? alreadyOpen
      ? history.state
      : { settingsReturnTo: location.pathname + location.search + location.hash, settingsFromApp: true }
    : null;
  const target = u.pathname + u.search + u.hash;
  if (settings && alreadyOpen) history.replaceState(state, '', target);
  else history.pushState(state, '', target);
  useRoute.setState(currentRoute());
  window.dispatchEvent(new HashChangeEvent('hashchange'));
}

/** Go to an in-app URL in place of the current entry, for an old address that now lives elsewhere (Settings'
 *  #device-links is Known › Devices): Back doesn't return to the old address. */
export function redirect(url: string): void {
  const u = new URL(url, location.href);
  history.replaceState(null, '', u.pathname + u.search + u.hash);
  useRoute.setState(currentRoute());
  window.dispatchEvent(new HashChangeEvent('hashchange'));
}

/** Write the current page's hash, for a page that keeps its place in the URL (Review, #2391). A push adds a Back
 *  step; otherwise the entry is replaced. No hashchange is dispatched (and the browser fires none for history writes),
 *  so the page never hears its own write as a link to follow. An unchanged URL writes nothing. */
export function setHash(hash: string, { push = false }: { push?: boolean } = {}): void {
  const target = location.pathname + location.search + hash;
  if (target === location.pathname + location.search + location.hash) return;
  if (push) history.pushState(null, '', target);
  else history.replaceState(history.state, '', target);
  useRoute.setState(currentRoute());
}

/** Close an in-app drawer with Back; a direct /settings link has no background entry, so it returns to Live. */
export function closeSettings(): void {
  if (history.state?.settingsFromApp) {
    history.back();
    return;
  }
  history.replaceState(null, '', '/');
  useRoute.setState(currentRoute());
  window.dispatchEvent(new HashChangeEvent('hashchange'));
}

/** Click handler for in-app links: plain left clicks navigate in place, modified clicks open normally. */
export function linkClick(e: MouseEvent<HTMLAnchorElement>): void {
  if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  navigate(e.currentTarget.href);
}

/** Follow the browser's Back and Forward; returns a function that stops. */
export function followHistory(): () => void {
  const pop = () => useRoute.setState(currentRoute());
  window.addEventListener('popstate', pop);
  window.addEventListener('hashchange', pop);
  return () => {
    window.removeEventListener('popstate', pop);
    window.removeEventListener('hashchange', pop);
  };
}
