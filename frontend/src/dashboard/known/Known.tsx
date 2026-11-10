// Known (#1970): who and what the house knows by name, one page with a tab each for People, Pets, Objects, Activities
// (#2283), Devices (phones, watches, and tags, Joe 2026-11-06), and Places (floorplans, #2796). Each tab is its own path
// (/people, /pets, /objects, /activities, /devices, /places), so a link or a reload opens
// the right one and Back steps through them. A tab's content is the page it was before, as is. Like the app's own pages, each mounts on its first visit
// and then stays mounted while hidden (an Activity), so switching back shows it as it was (#330). The page is one wide
// PageFrame (#2469), so its tabs start at the same left edge as every other page, each with its own PageHeader.
import { Tabs } from '@mantine/core';
import { useElementSize, useIntersection } from '@mantine/hooks';
import { useQuery } from '@tanstack/react-query';
import { Activity, lazy, type ReactNode, useCallback, useEffect, useRef, useState } from 'react';
import { toCheck, useRecentActivities } from '../activities/recent';
import { Deferred } from '../Deferred';
import { PageFrame } from '../lib/page';
import { sightingsQuery, waitingGroups } from '../people/pets/api';
import { KNOWN_TABS, type KnownTab, navigate, useRoute } from '../router';
import { unknownGroupsQuery, unknownVoicesQuery, waitingTotal } from '../whosthis/api';
import classes from './known.module.css';

const PeopleTab = lazy(() => import('../people/PeopleTab').then(m => ({ default: m.PeopleTab })));
const Pets = lazy(() => import('../people/Pets').then(m => ({ default: m.Pets })));
const ObjectMemoryPage = lazy(() => import('../../objects/ObjectMemory').then(m => ({ default: m.ObjectMemoryPage })));
const ActivitiesTab = lazy(() => import('../activities/ActivitiesTab').then(m => ({ default: m.ActivitiesTab })));
const DevicesTab = lazy(() => import('./DevicesTab').then(m => ({ default: m.DevicesTab })));
const PlacesTab = lazy(() => import('../places/PlacesTab').then(m => ({ default: m.PlacesTab })));

// The same element every render, so a tab switch re-renders only the Activity boundaries.
const PANELS: Record<KnownTab, ReactNode> = {
  people: (
    <Deferred name="People">
      <PeopleTab />
    </Deferred>
  ),
  pets: (
    <Deferred name="Pets">
      <Pets />
    </Deferred>
  ),
  objects: (
    <Deferred name="Objects">
      <ObjectMemoryPage />
    </Deferred>
  ),
  activities: (
    <Deferred name="Activities">
      <ActivitiesTab />
    </Deferred>
  ),
  devices: (
    <Deferred name="Devices">
      <DevicesTab />
    </Deferred>
  ),
  places: (
    <Deferred name="Places">
      <PlacesTab />
    </Deferred>
  ),
};

/** A sub-tab's count of what waits there, the same number its list shows (#2451), saying what it counts ("201 to
 *  name"): a bare number beside "People" read as how many people there are. It sits on its own line under the tab's
 *  title (#3243), a line every tab has from the first paint, loaded or not and counted or not, and every tab is one
 *  fixed width, so a count arriving late (they read their lists seconds after Known opens) never widens a tab or slides
 *  the row under a pointer that hadn't moved (#2828). Nothing shows when none waits. */
export function Waiting({ n, what }: { n: number | null | undefined; what: string }) {
  return <span className={classes.count}>{n ? `${n} ${what}` : null}</span>;
}

// Each badge reads the list its tab reads, so the badge and the list agree. On its own tab the badge only follows
// that read (the tab decides when to start it: People's roster must come first, #2236). On another tab it reads the
// list itself, but only BADGE_DELAY_MS after Known opens, so a cold read never takes a slot from the open tab's own.
const BADGE_DELAY_MS = 4000;

/** Whether this browser has IntersectionObserver (every one Known runs in; not the tests' jsdom), which useIntersection
 *  needs to attach. */
const OBSERVES = typeof IntersectionObserver === 'function';

function useSettled(): boolean {
  const [settled, setSettled] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setSettled(true), BADGE_DELAY_MS);
    return () => clearTimeout(t);
  }, []);
  return settled;
}

function PeopleBadge({ read }: { read: boolean }) {
  const faces = useQuery({ ...unknownGroupsQuery(), enabled: read });
  const voices = useQuery({ ...unknownVoicesQuery(), enabled: read });
  return <Waiting n={waitingTotal(faces, voices)} what="to name" />;
}

function PetsBadge({ read }: { read: boolean }) {
  const sightings = useQuery({ ...sightingsQuery(), enabled: read });
  return <Waiting n={waitingGroups(sightings.data)} what="to name" />;
}

/** The clips of the last week Clips to check lists (#2283): counted kinds by their spot-check sample (#2441). */
function ActivitiesBadge() {
  const recent = useRecentActivities();
  return <Waiting n={toCheck(recent.items, recent.promotion).length} what="to check" />;
}

const badge = (id: KnownTab, tab: KnownTab, settled: boolean): ReactNode =>
  id === 'people' ? (
    <PeopleBadge read={settled && tab !== 'people'} />
  ) : id === 'pets' ? (
    <PetsBadge read={settled && tab !== 'pets'} />
  ) : id === 'activities' ? (
    <ActivitiesBadge />
  ) : (
    <Waiting n={null} what="" />
  );

/** Which ends of the tab row have more tabs past them (on a phone the row scrolls sideways): those ends fade, so the
 *  row says it scrolls. The open tab is scrolled into view, so Devices isn't left off the edge once picked. */
export function useRowEdges(tab: KnownTab) {
  const list = useRef<HTMLDivElement>(null);
  const open = useRef(tab);
  open.current = tab;
  const [edges, setEdges] = useState({ start: false, end: false });
  /** Scroll the row (only the row: scrollIntoView would move the page too) so the open tab is clear of the faded ends. */
  const showOpen = useCallback(() => {
    const el = list.current;
    const tabEl = el?.querySelector<HTMLElement>(`#known-tab-${open.current}`);
    if (!el || !tabEl || el.scrollWidth <= el.clientWidth) return;
    const row = el.getBoundingClientRect();
    const t = tabEl.getBoundingClientRect();
    if (t.left < row.left + FADE_PX) el.scrollLeft -= row.left + FADE_PX - t.left;
    else if (t.right > row.right - FADE_PX) el.scrollLeft += t.right - (row.right - FADE_PX);
  }, []);
  const measure = useCallback(() => {
    const el = list.current;
    if (!el) return;
    const start = el.scrollLeft > 1;
    const end = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
    setEdges(e => (e.start === start && e.end === end ? e : { start, end }));
  }, []);
  useEffect(() => {
    const el = list.current;
    if (!el) return;
    measure();
    // The row narrowing (a phone turned, a window resized) can push the open tab past the edge again.
    const observer = new ResizeObserver(() => {
      showOpen();
      measure();
    });
    observer.observe(el);
    el.addEventListener('scroll', measure, { passive: true });
    return () => {
      observer.disconnect();
      el.removeEventListener('scroll', measure);
    };
  }, [measure, showOpen]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new open tab (read through `open`) is what re-runs it
  useEffect(() => {
    showOpen();
    measure();
  }, [tab, showOpen, measure]);
  return { list, edges };
}

/** How far the row's ends fade (known.module.css's 48px). */
const FADE_PX = 48;

export function Known() {
  const tab = useRoute(s => s.known);
  const [visited, setVisited] = useState<KnownTab[]>([tab]);
  if (!visited.includes(tab)) setVisited([...visited, tab]);
  const settled = useSettled();
  // Stuck once the line at the row's top (the frame's top edge: the row holds the frame's top space) has scrolled
  // under the app bar (--top-h, read once); the row's height is published as --known-h for what sits under it.
  const [top] = useState(
    () => Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--top-h')) || 56,
  );
  const row = useRowEdges(tab);
  const sentinel = useIntersection({ rootMargin: `-${top}px 0px 0px 0px` });
  const stuck = !!sentinel.entry && !sentinel.entry.isIntersecting && sentinel.entry.boundingClientRect.top < top + 1;
  const bar = useElementSize();
  useEffect(() => {
    if (bar.height > 0) document.documentElement.style.setProperty('--known-h', `${Math.ceil(bar.height)}px`);
  }, [bar.height]);
  return (
    <PageFrame width="wide" className={classes.root}>
      <div ref={OBSERVES ? sentinel.ref : undefined} className={classes.sentinel} aria-hidden="true" />
      <div ref={bar.ref} className={classes.bar} data-stuck={stuck ? '' : undefined}>
        <Tabs
          value={tab}
          onChange={value => {
            const next = KNOWN_TABS.find(t => t.id === value);
            if (next && next.id !== tab) navigate(next.path);
          }}
          classNames={{ list: classes.list, tab: classes.tab, tabLabel: classes.label }}
        >
          <Tabs.List
            aria-label="Known"
            ref={row.list}
            data-more-start={row.edges.start || undefined}
            data-more-end={row.edges.end || undefined}
          >
            {KNOWN_TABS.map(t => (
              <Tabs.Tab key={t.id} value={t.id} id={`known-tab-${t.id}`} aria-controls={`known-panel-${t.id}`}>
                <span className={classes.title}>{t.label}</span>
                {badge(t.id, tab, settled)}
              </Tabs.Tab>
            ))}
          </Tabs.List>
        </Tabs>
      </div>
      {visited.map(id => (
        <Activity key={id} mode={id === tab ? 'visible' : 'hidden'}>
          <div role="tabpanel" id={`known-panel-${id}`} aria-labelledby={`known-tab-${id}`}>
            {PANELS[id]}
          </div>
        </Activity>
      ))}
    </PageFrame>
  );
}
