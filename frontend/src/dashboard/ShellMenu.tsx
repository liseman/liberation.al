// The top bar's one menu: Settings, the Resource monitor, the theme, Live's display choices, and the Debug log,
// under a gear. They were five separate controls in four styles; none is used often enough to sit in the bar. A
// failed or crash-looping service still shows on the gear as a dot, from every tab.
import { ActionIcon, type MantineColorScheme, Menu, useMantineColorScheme } from '@mantine/core';
import { useHotkeys } from '@mantine/hooks';
import {
  IconActivity,
  IconAdjustmentsHorizontal,
  IconCheck,
  IconDeviceDesktop,
  IconMoon,
  IconSettings,
  IconSun,
  IconTerminal2,
} from '@tabler/icons-react';
import { useRef, useState } from 'react';
import classes from './app.module.css';
import { LiveDisplay } from './livetab/Appearance';
import { href, linkClick, navigate } from './router';
import { useServicesBadge } from './settings/services/api';
import { shell } from './shell';
import { TITLE as MONITOR, setUsagePanelOpen, useUsagePanelOpen } from './usage/UsagePanel';

const THEMES: { value: MantineColorScheme; label: string; icon: typeof IconSun }[] = [
  { value: 'auto', label: 'System', icon: IconDeviceDesktop },
  { value: 'light', label: 'Light', icon: IconSun },
  { value: 'dark', label: 'Dark', icon: IconMoon },
];

/** `setup`: first-run setup, where only the theme applies. `routed`: Settings is open. */
export function ShellMenu({ setup = false, routed = false }: { setup?: boolean; routed?: boolean }) {
  const { colorScheme, setColorScheme } = useMantineColorScheme();
  const [display, setDisplay] = useState(false);
  const monitor = useUsagePanelOpen();
  const issue = useServicesBadge();
  const gear = useRef<HTMLButtonElement>(null);
  // "," opens Settings from anywhere, as "/" opens Ask, when single-key shortcuts are on (dashboard.hotkeys). Typing in
  // a box never triggers it (useHotkeys ignores inputs). Focus goes to the gear first, so closing returns there.
  useHotkeys([
    [
      ',',
      () => {
        if (setup || routed || !shell().hotkeys) return;
        gear.current?.focus();
        navigate(href('/settings'));
      },
    ],
  ]);
  return (
    <>
      <Menu position="bottom-end" withinPortal width={240}>
        <Menu.Target>
          <ActionIcon
            variant={routed ? 'light' : 'subtle'}
            color="gray"
            size="lg"
            aria-label={issue ? `Menu (services: ${issue.text})` : 'Menu'}
            data-settings-trigger
            ref={gear}
            className={classes.gear}
          >
            <IconSettings size={20} stroke={1.75} aria-hidden />
            {issue && !setup && <span aria-hidden className={classes.gearDot} />}
          </ActionIcon>
        </Menu.Target>
        <Menu.Dropdown>
          {!setup && (
            <>
              <Menu.Item
                component="a"
                href={href('/settings')}
                onClick={e => {
                  // The drawer gives focus back to what had it when it opened: the gear, not this item, which goes.
                  gear.current?.focus();
                  linkClick(e);
                }}
                leftSection={<IconSettings size={16} aria-hidden />}
                rightSection={issue ? <span aria-hidden className={classes.gearDotInline} /> : null}
              >
                Settings
              </Menu.Item>
              {issue && <Menu.Label className={classes.statusLine}>Services: {issue.text}</Menu.Label>}
              <Menu.Item
                leftSection={<IconActivity size={16} aria-hidden />}
                rightSection={monitor ? <IconCheck size={14} aria-hidden /> : null}
                aria-pressed={monitor}
                onClick={() => setUsagePanelOpen(!monitor)}
              >
                {MONITOR}
              </Menu.Item>
              <Menu.Divider />
            </>
          )}
          <Menu.Label>Theme</Menu.Label>
          {THEMES.map(t => (
            <Menu.Item
              key={t.value}
              leftSection={<t.icon size={16} aria-hidden />}
              rightSection={t.value === colorScheme ? <IconCheck size={14} aria-hidden /> : null}
              onClick={() => setColorScheme(t.value)}
            >
              {t.label}
            </Menu.Item>
          ))}
          {!setup && (
            <>
              <Menu.Item
                leftSection={<IconAdjustmentsHorizontal size={16} aria-hidden />}
                onClick={() => setDisplay(true)}
              >
                Live display…
              </Menu.Item>
              <Menu.Divider />
              <Menu.Item
                component="a"
                href={href('/debug')}
                onClick={linkClick}
                leftSection={<IconTerminal2 size={16} aria-hidden />}
              >
                Debug log
              </Menu.Item>
            </>
          )}
        </Menu.Dropdown>
      </Menu>
      <LiveDisplay opened={display} onClose={() => setDisplay(false)} />
    </>
  );
}
