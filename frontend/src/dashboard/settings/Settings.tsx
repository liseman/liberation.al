// First-run setup owns the page; configured installations open Settings in a Mantine Drawer.
import { Drawer } from '@mantine/core';
import { Activity, lazy, useState } from 'react';
import { Deferred } from '../Deferred';
import { shell } from '../shell';
import { SearchSlot } from './searchSlot';
import settingsClasses from './settings.module.css';

const SettingsPage = lazy(() => import('./SettingsPage').then(m => ({ default: m.SettingsPage })));

/** First run (no pack.toml when the page was served): setup owns the page until its own restart navigation loads
 *  the page again, so a config refresh that sees the new file can't unmount the flow. The server fills it into the
 *  page, so the shell asks nothing to know. */
export function useNeedsSetup(): boolean {
  return shell().needsSetup;
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
