// The dashboard app: a top bar with the tabs, status, and buttons, and the current tab below it.
import { Anchor, Menu, UnstyledButton } from '@mantine/core';
import { useDocumentTitle, useMediaQuery } from '@mantine/hooks';
import { IconBroadcast, IconDots, IconInbox, IconPlayerPlay, IconUsers, type TablerIcon } from '@tabler/icons-react';
import { useQueryClient } from '@tanstack/react-query';
import { Activity, lazy, type ReactNode, Suspense, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { AskBar } from './AskBar';
import classes from './app.module.css';
import { ThumbFeed } from './camera/ThumbFeed';
import { Deferred } from './Deferred';
import { InboxLabel } from './inbox/InboxLabel';
import { startLive } from './live/feed';
import { LiveTab } from './livetab/LiveTab';
import { PaletteButton } from './palette/PaletteButton';
import { closeSettings, followHistory, href, KNOWN_TABS, linkClick, TABS, type TabId, tabOf, useRoute } from './router';
import { ShellMenu } from './ShellMenu';
import { StatusPill } from './StatusIcons';
import { SettingsDrawer, useNeedsSetup } from './settings/Settings';
import { UsagePanel } from './usage/UsagePanel';

// Live stays eager; every other pane loads on first visit and keeps its Activity state afterward.
const ReportsTab = lazy(() => import('./reports/ReportsTab'));
// The command palette loads after the page, so its Settings and people reads don't hold the first view up.
const Palette = lazy(() => import('./palette/Palette').then(m => ({ default: m.Palette })));
const Review = lazy(() => import('./review/Review').then(m => ({ default: m.Review })));
const Known = lazy(() => import('./known/Known').then(m => ({ default: m.Known })));
const JournalTab = lazy(() => import('./journal/JournalTab').then(m => ({ default: m.JournalTab })));
const CallsTab = lazy(() => import('./calls/Calls').then(m => ({ default: m.CallsTab })));
const InboxTab = lazy(() => import('./inbox/InboxTab').then(m => ({ default: m.InboxTab })));
const IngestTab = lazy(() => import('./ingest/IngestTab').then(m => ({ default: m.IngestTab })));
const DebugTab = lazy(() => import('./debug/DebugTab').then(m => ({ default: m.DebugTab })));
const SkeletonsPage = lazy(() => import('./skeletons/SkeletonsPage').then(m => ({ default: m.SkeletonsPage })));
const SettingsForm = lazy(() => import('./settings/SettingsForm').then(m => ({ default: m.SettingsForm })));

const TITLES: Record<TabId, string> = {
  live: 'Live',
  review: 'Review',
  journal: 'Journal',
  known: 'Known',
  calls: 'Calls',
  inbox: 'Inbox',
  reports: 'Reports',
  ingest: 'Ingest',
  debug: 'Debug',
  skeletons: 'Skeletons',
  settings: 'Settings',
};

type Pane = Exclude<TabId, 'settings'>;

/** The top bar's places: Debug is a developer page, reached from the menu, not a main place. */
const NAV = TABS.filter(t => t.id !== 'debug');

/** A phone's bottom bar: the four places used most at the thumb, and the rest under More. */
const BOTTOM: { id: Pane; icon: TablerIcon }[] = [
  { id: 'live', icon: IconBroadcast },
  { id: 'review', icon: IconPlayerPlay },
  { id: 'known', icon: IconUsers },
  { id: 'inbox', icon: IconInbox },
];
/** The phone breakpoint, as in app.module.css. */
const PHONE_NAV = '(max-width: 700px)';
const MORE = NAV.filter(t => !BOTTOM.some(b => b.id === t.id));

const pathOf = (id: Pane, known: string): string => {
  const t = TABS.find(x => x.id === id);
  if (!t) return '/';
  return id === 'known' ? (KNOWN_TABS.find(k => k.id === known)?.path ?? t.path) : t.path;
};

function BottomNav({ content, known }: { content: TabId; known: string }) {
  // Only on a phone: elsewhere the top bar has the places, and two navigations named the same would confuse.
  const phone = useMediaQuery(PHONE_NAV) ?? false;
  const inMore = MORE.some(t => t.id === content);
  if (!phone) return null;
  return (
    <nav className={classes.bottom} aria-label="Sections">
      {BOTTOM.map(b => {
        const t = TABS.find(x => x.id === b.id);
        return (
          <a
            key={b.id}
            className={classes.bottomItem}
            href={href(pathOf(b.id, known))}
            onClick={linkClick}
            aria-current={content === b.id ? 'page' : undefined}
          >
            <b.icon size={22} stroke={1.75} aria-hidden />
            <span>{b.id === 'inbox' ? <InboxLabel /> : t?.label}</span>
          </a>
        );
      })}
      <Menu position="top-end" withinPortal>
        <Menu.Target>
          <UnstyledButton className={classes.bottomItem} aria-current={inMore ? 'page' : undefined}>
            <IconDots size={22} stroke={1.75} aria-hidden />
            <span>More</span>
          </UnstyledButton>
        </Menu.Target>
        <Menu.Dropdown>
          {MORE.map(t => (
            <Menu.Item
              key={t.id}
              component="a"
              href={href(t.path)}
              onClick={linkClick}
              aria-current={content === t.id ? 'page' : undefined}
            >
              {t.label}
            </Menu.Item>
          ))}
        </Menu.Dropdown>
      </Menu>
    </nav>
  );
}
// The same element every render, so a tab switch re-renders only the Activity boundaries, never the tabs inside.
const PANES: Record<Pane, ReactNode> = {
  live: <LiveTab />,
  review: (
    <Deferred name="Review">
      <Review />
    </Deferred>
  ),
  journal: (
    <Deferred name="Journal">
      <JournalTab />
    </Deferred>
  ),
  known: (
    <Deferred name="Known">
      <Known />
    </Deferred>
  ),
  calls: (
    <Deferred name="Calls">
      <CallsTab />
    </Deferred>
  ),
  inbox: (
    <Deferred name="Inbox">
      <InboxTab />
    </Deferred>
  ),
  reports: (
    <Deferred name="Reports">
      <ReportsTab />
    </Deferred>
  ),
  ingest: (
    <Deferred name="Ingest">
      <IngestTab />
    </Deferred>
  ),
  debug: (
    <Deferred name="Debug">
      <DebugTab />
    </Deferred>
  ),
  skeletons: (
    <Deferred name="Skeletons">
      <SkeletonsPage />
    </Deferred>
  ),
};

/** Each tab stays mounted once visited, so switching back shows it as it was instead of building it again.
 *  A hidden tab is an Activity in hidden mode: display none, with its effects cleaned up until it shows again, so its
 *  streams, polling queries, animation frames, and key handlers stop while it's out of sight. */
function Views({ tab: current }: { tab: Pane }): ReactNode {
  const [visited, setVisited] = useState<Pane[]>([current]);
  if (!visited.includes(current)) setVisited([...visited, current]);
  return visited.map(p => (
    <Activity key={p} mode={p === current ? 'visible' : 'hidden'}>
      {PANES[p]}
    </Activity>
  ));
}

export function App() {
  const client = useQueryClient();
  const tab = useRoute(s => s.tab);
  const known = useRoute(s => s.known);
  const returnTo = useRoute(s => s.settingsReturnTo);
  const background = tabOf(returnTo?.split(/[?#]/)[0] ?? '/');
  const content = tab === 'settings' ? (background === 'settings' ? 'live' : background) : tab;
  const setup = useNeedsSetup();
  // A layout effect: the feed is open before any query subscribes (in a passive effect), so a stream-fed query
  // waits for the stream's opening snapshot instead of asking its URL (#52).
  useLayoutEffect(() => startLive(client), [client]);
  const top = useRef<HTMLElement>(null);
  const tabs = useRef<HTMLElement>(null);
  // On a phone the bottom bar is the one Sections navigation; the top one is hidden outright, not just by CSS.
  const phone = useMediaQuery(PHONE_NAV) ?? false;
  useLayoutEffect(() => {
    // A deep link or history navigation can select a tab outside the phone navigation viewport.
    const reveal = () =>
      tabs.current?.querySelector<HTMLElement>(`[data-tab="${content}"]`)?.scrollIntoView({
        block: 'nearest',
        inline: 'nearest',
      });
    reveal();
    window.addEventListener('resize', reveal);
    return () => window.removeEventListener('resize', reveal);
  }, [content]);
  // The header's real height (it wraps on narrow screens) is --top-h, which Live's layout and the waveform use.
  useLayoutEffect(() => {
    const measure = () => {
      const height = top.current?.getBoundingClientRect().height ?? 0;
      if (height > 0) document.documentElement.style.setProperty('--top-h', `${Math.ceil(height)}px`);
    };
    measure();
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(measure) : null;
    if (top.current) observer?.observe(top.current);
    window.addEventListener('resize', measure);
    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, []);
  useEffect(followHistory, []);
  useDocumentTitle(`Liberational: ${TITLES[tab]}`);

  return (
    <>
      <ThumbFeed />
      <Anchor
        className={classes.skip}
        href="#main-content"
        onClick={e => {
          e.preventDefault();
          document.getElementById('main-content')?.focus();
        }}
      >
        Skip to content
      </Anchor>
      <header className={classes.top} ref={top}>
        <a className={classes.brand} href={href('/')} onClick={linkClick}>
          Liberational
        </a>
        {/* On a phone the bottom bar has the places, so the top names the page instead of the app. */}
        <span className={classes.pageName}>{TITLES[content]}</span>
        <nav className={classes.tabs} aria-label="Sections" ref={tabs} hidden={phone || undefined}>
          {NAV.map(t => (
            <a
              key={t.id}
              className={classes.tab}
              data-tab={t.id}
              href={href(t.id === 'known' ? (KNOWN_TABS.find(k => k.id === known)?.path ?? t.path) : t.path)}
              onClick={linkClick}
              aria-current={content === t.id ? 'page' : undefined}
            >
              {t.id === 'inbox' ? <InboxLabel /> : t.label}
            </a>
          ))}
        </nav>
        <div className={classes.right}>
          {!setup && <PaletteButton />}
          {/* The page showing under Settings decides which Ask the bar has, so Settings never closes it. */}
          <AskBar tab={content} setup={setup} settings={tab === 'settings'} />
          <StatusPill />
          <ShellMenu setup={setup} routed={tab === 'settings'} />
        </div>
      </header>
      {!setup && (
        <Suspense fallback={null}>
          <Palette />
        </Suspense>
      )}
      <main id="main-content" tabIndex={-1}>
        {setup ? (
          <Deferred name="setup">
            <SettingsForm setup />
          </Deferred>
        ) : (
          <Views tab={content} />
        )}
      </main>
      {!setup && <BottomNav content={content} known={known} />}
      {!setup && <SettingsDrawer opened={tab === 'settings'} onClose={closeSettings} />}
      {!setup && <UsagePanel />}
    </>
  );
}
