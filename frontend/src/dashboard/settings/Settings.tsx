// First-run setup owns the page; configured installations open Settings in a Mantine Drawer.
import { Drawer } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { Activity, lazy, useEffect, useState } from 'react';
import { request } from '../../shared/http';
import { Deferred } from '../Deferred';
import { CONFIG_KEY, type ConfigView } from './model';
import { SearchSlot } from './searchSlot';
import settingsClasses from './settings.module.css';

const SettingsPage = lazy(() => import('./SettingsPage').then(m => ({ default: m.SettingsPage })));

export function useNeedsSetup(): boolean {
  const query = useQuery({
    queryKey: CONFIG_KEY,
    queryFn: ({ signal }) => request<ConfigView>('/config', undefined, signal),
  });
  // A first-run flow owns the page until restart navigation. A focus refetch can
  // see pack.toml before services finish restarting; it must not unmount that flow.
  const [started, setStarted] = useState(false);
  useEffect(() => {
    if (query.data?.needs_setup) setStarted(true);
  }, [query.data?.needs_setup]);
  return started || (query.data?.needs_setup ?? false);
}

export function SettingsDrawer({ opened, onClose }: { opened: boolean; onClose: () => void }) {
  const [visited, setVisited] = useState(opened);
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  if (opened && !visited) setVisited(true);
  return (
    <Drawer.Root
      opened={opened}
      onClose={() => {
        // Settings controls open their own portaled Mantine Modals. Their Escape
        // closes that dialog first, leaving this drawer and its draft in place.
        const childOpen = [...document.querySelectorAll<HTMLElement>('[role="dialog"]')].some(
          dialog =>
            !dialog.classList.contains(settingsClasses.drawerContent ?? '') &&
            (typeof dialog.checkVisibility === 'function'
              ? dialog.checkVisibility()
              : getComputedStyle(dialog).display !== 'none'),
        );
        if (childOpen) return;
        onClose();
      }}
      position="right"
      size="min(100%, 1100px)"
      keepMounted
      classNames={{
        content: settingsClasses.drawerContent,
        header: settingsClasses.drawerHeader,
        body: settingsClasses.drawerBody,
      }}
    >
      <Drawer.Overlay />
      <Drawer.Content>
        {/* One header row: the title (where focus lands on opening, so the close button wears no ring after a
            click), Settings' search, and close. */}
        {/* Not a banner landmark: inside the dialog its <header> would be a second one beside the app's (axe). */}
        <Drawer.Header role="none">
          <Drawer.Title className={settingsClasses.drawerTitle} tabIndex={-1} data-autofocus>
            Settings
          </Drawer.Title>
          <div ref={setSlot} className={settingsClasses.searchSlot} />
          <Drawer.CloseButton aria-label="Close settings" />
        </Drawer.Header>
        <Drawer.Body>
          {visited && (
            <Activity mode={opened ? 'visible' : 'hidden'}>
              <Deferred name="Settings">
                <SearchSlot value={slot}>
                  <SettingsPage />
                </SearchSlot>
              </Deferred>
            </Activity>
          )}
        </Drawer.Body>
      </Drawer.Content>
    </Drawer.Root>
  );
}
