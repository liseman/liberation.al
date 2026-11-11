// The Resource monitor: the usage chart (UsageChart) in a panel opened from the header's activity
// button. It's mounted by the app shell, not a page, so it stays open on every page until it's closed. On a wide screen
// it floats: dragged by its title bar and resized from its corner. On a phone it docks as a bottom sheet. Open or
// closed, where it sits, and its size are kept per viewer (liberational.prefs). The dashboard samples usage all the
// time (machine_usage.Sampler), so opening the panel shows the whole last half hour; the chart polls only while
// it's open.
import { CloseButton, Paper, Text, Title } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { IconGripHorizontal } from '@tabler/icons-react';
import { lazy, type PointerEvent, useEffect, useId, useRef } from 'react';
import { Deferred } from '../Deferred';
import { setPref, type UsagePanelPref, usePrefs } from '../lib/prefs';
import classes from './usagePanel.module.css';

// The chart and D3 are needed only when the monitor opens, not when the shell loads.
const UsageChart = lazy(() => import('./UsageChart').then(module => ({ default: module.UsageChart })));

export const PHONE = '(max-width: 700px)'; // the header's and the page frame's phone breakpoint (app.module.css)
export const DEFAULT_SIZE = { w: 560, h: 460 };
const MIN = { w: 320, h: 240 }; // as small as the CSS lets it be resized
const EDGE = 16; // a new panel's gap from the window's right edge
const TOP = 64; // and from its top, under the header
const KEEP = 96; // this much of the title bar always stays on screen
const SAVE_MS = 250; // a resize is saved once it settles
export const TITLE = 'Resource monitor';

const current = (): UsagePanelPref => usePrefs.getState().usagePanel ?? {};
const update = (patch: UsagePanelPref) => setPref('usagePanel', { ...current(), ...patch });

/** Where a panel of width w goes: its saved place, else the top right, always with its title bar reachable. */
export function placed(pref: UsagePanelPref, w: number, view: { width: number; height: number }) {
  const x = pref.x ?? view.width - w - EDGE;
  const y = pref.y ?? TOP;
  return {
    x: Math.round(Math.min(Math.max(x, KEEP - w), view.width - KEEP)),
    y: Math.round(Math.min(Math.max(y, 0), view.height - 40)),
  };
}

/** Whether the panel is open, for the top bar's menu. */
export const useUsagePanelOpen = (): boolean => usePrefs(s => s.usagePanel?.open === true);

/** Open or close the panel. */
export const setUsagePanelOpen = (open: boolean): void => update({ open });

export function UsagePanel() {
  const pref = usePrefs(s => s.usagePanel) ?? {};
  const phone = useMediaQuery(PHONE) ?? false;
  const heading = useId();
  const panel = useRef<HTMLElement>(null);
  const drag = useRef<{ dx: number; dy: number } | null>(null);
  const open = pref.open === true;
  const w = Math.max(MIN.w, pref.w ?? DEFAULT_SIZE.w);
  const h = Math.max(MIN.h, pref.h ?? DEFAULT_SIZE.h);

  // A resize from the corner is the browser's (CSS resize), so its size is read back and saved once it settles.
  useEffect(() => {
    const el = panel.current;
    if (!open || phone || !el || typeof ResizeObserver === 'undefined') return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const observer = new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const box = el.getBoundingClientRect();
        const size = { w: Math.round(box.width), h: Math.round(box.height) };
        if (size.w && size.h && (size.w !== current().w || size.h !== current().h)) update(size);
      }, SAVE_MS);
    });
    observer.observe(el);
    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [open, phone]);

  if (!open) return null;
  const view = { width: window.innerWidth, height: window.innerHeight };
  const at = placed(pref, w, view);

  const start = (e: PointerEvent<HTMLDivElement>) => {
    if (phone || e.button !== 0 || (e.target as HTMLElement).closest('button')) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { dx: e.clientX - at.x, dy: e.clientY - at.y };
  };
  const move = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    const el = panel.current;
    if (!d || !el) return;
    const next = placed({ x: e.clientX - d.dx, y: e.clientY - d.dy }, w, view);
    el.style.left = `${next.x}px`; // moved directly while dragging; saved (and rendered) when it's let go
    el.style.top = `${next.y}px`;
  };
  const end = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    drag.current = null;
    update(placed({ x: e.clientX - d.dx, y: e.clientY - d.dy }, w, view));
  };

  return (
    <Paper
      ref={panel}
      component="section"
      aria-labelledby={heading}
      className={phone ? `${classes.panel} ${classes.sheet}` : classes.panel}
      shadow="lg"
      withBorder
      style={phone ? undefined : { left: at.x, top: at.y, width: w, height: h }}
    >
      <div
        className={classes.bar}
        data-draggable={!phone || undefined}
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={end}
      >
        {!phone && <IconGripHorizontal size={16} className={classes.grip} aria-hidden />}
        <Title order={2} id={heading} className={classes.title}>
          {TITLE}
        </Title>
        <CloseButton aria-label={`Close ${TITLE}`} onClick={() => update({ open: false })} />
      </div>
      <div className={classes.body}>
        <Text size="sm" c="dimmed" mb="sm">
          Each of Liberational's programs over the last half hour, stacked to the whole machine. Everything else is
          other programs and the smaller services.
        </Text>
        <Deferred name="resource use">
          <UsageChart />
        </Deferred>
      </div>
    </Paper>
  );
}
