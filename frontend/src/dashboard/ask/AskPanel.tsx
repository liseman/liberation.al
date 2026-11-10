// The Ask window (#1358): it floats above the page on any tab, nonmodal, so the page stays usable beside it. Drag it
// by its header (or move it with the arrow keys on its move handle); it stays inside the viewport, also on a resize,
// and remembers where it was per browser. On a phone-width screen it is a bottom sheet instead. Escape or the close
// button closes it, and focus goes back to the Ask button that opened it.
import { ActionIcon, Text } from '@mantine/core';
import { IconGripHorizontal, IconX } from '@tabler/icons-react';
import {
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useEffectEvent,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { createPortal } from 'react-dom';
import classes from './panel.module.css';

export const SHEET_MAX = 640; // at or under this width, a bottom sheet
const MARGIN = 8;
const WIDTH = 440;
const HEIGHT = 620;
const KEEP_HEADER = 48; // at least this much of the window stays on screen, so its header can be grabbed again
const STEP = 16; // arrow keys move the window this far (Shift: four times as far)

export interface Spot {
  left: number;
  top: number;
}

export interface Box extends Spot {
  width: number;
  height: number;
}

/** The window's box for a wanted top-left corner in a viewport: as large as fits, wholly on screen where it can be,
 *  and never above the page's header (`headerBottom`). With no wanted corner, it sits at the top right. */
export function fitWindow(wanted: Spot | null, viewport: { width: number; height: number }, headerBottom: number): Box {
  const width = Math.min(WIDTH, viewport.width - 2 * MARGIN);
  const minTop = Math.max(MARGIN, headerBottom + MARGIN);
  const spot = wanted ?? { left: viewport.width - width - 2 * MARGIN, top: minTop };
  const left = Math.min(Math.max(spot.left, MARGIN), Math.max(MARGIN, viewport.width - width - MARGIN));
  const top = Math.min(Math.max(spot.top, minTop), Math.max(minTop, viewport.height - KEEP_HEADER - MARGIN));
  const height = Math.max(KEEP_HEADER, Math.min(HEIGHT, viewport.height - top - MARGIN));
  return { left: Math.round(left), top: Math.round(top), width, height };
}

const storageKey = (kind: string) => `liberational.ask-window.${kind}`;

/** Where this browser last left the window, or null. */
export function savedSpot(kind: string): Spot | null {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey(kind)) ?? 'null');
    return value && Number.isFinite(value.left) && Number.isFinite(value.top)
      ? { left: value.left, top: value.top }
      : null;
  } catch {
    return null;
  }
}

function saveSpot(kind: string, spot: Spot): void {
  try {
    localStorage.setItem(storageKey(kind), JSON.stringify({ left: spot.left, top: spot.top }));
  } catch {
    // no storage (a private window, or blocked): it opens in its usual place next time
  }
}

const headerBottom = () => document.querySelector('header')?.getBoundingClientRect().bottom ?? 0;
const viewport = () => ({ width: window.innerWidth, height: window.innerHeight });
const isSheet = () => window.innerWidth <= SHEET_MAX;

export function AskPanel({
  opened,
  onClose,
  title,
  kind,
  children,
  className = '',
}: {
  opened: boolean;
  onClose: () => void;
  title: string;
  kind: 'live' | 'review';
  children: ReactNode;
  className?: string;
}) {
  const root = useRef<HTMLElement>(null);
  const [sheet, setSheet] = useState(isSheet);
  const [box, setBox] = useState<Box>(() => fitWindow(savedSpot(kind), viewport(), headerBottom()));
  const drag = useRef<{ x: number; y: number; from: Spot } | null>(null);

  const place = useCallback((wanted: Spot | null) => {
    setSheet(isSheet());
    setBox(fitWindow(wanted, viewport(), headerBottom()));
  }, []);
  // On open, from where it was saved; on a resize, from where it is, but always on screen.
  const refit = useEffectEvent(() => place({ left: box.left, top: box.top }));
  useLayoutEffect(() => {
    if (!opened) return;
    place(savedSpot(kind));
    const resize = () => refit();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [opened, kind, place]);

  const moveTo = (spot: Spot, save: boolean) => {
    const next = fitWindow(spot, viewport(), headerBottom());
    setBox(next);
    if (save) saveSpot(kind, next);
  };
  const dragged = (e: PointerEvent<HTMLElement>): Spot | null => {
    const d = drag.current;
    return d ? { left: d.from.left + e.clientX - d.x, top: d.from.top + e.clientY - d.y } : null;
  };
  const onPointerDown = (e: PointerEvent<HTMLElement>) => {
    if (sheet || e.button !== 0 || (e.target as HTMLElement).closest('button:not([data-ask-move])')) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, from: { left: box.left, top: box.top } };
  };
  const onPointerMove = (e: PointerEvent<HTMLElement>) => {
    const spot = dragged(e);
    if (spot) moveTo(spot, false);
  };
  const onPointerUp = (e: PointerEvent<HTMLElement>) => {
    const spot = dragged(e);
    drag.current = null;
    if (spot) moveTo(spot, true);
  };
  const onMoveKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    const step = e.shiftKey ? STEP * 4 : STEP;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const move = moves[e.key];
    if (!move) return;
    e.preventDefault();
    moveTo({ left: box.left + move[0], top: box.top + move[1] }, true);
  };

  const close = () => {
    onClose();
    document.querySelector<HTMLElement>(`[data-ask-trigger="${kind}"]`)?.focus({ preventScroll: true });
  };
  const closeFromKey = useEffectEvent(close);
  const focusInside = useEffectEvent(() => {
    root.current?.querySelector<HTMLElement>('textarea, input')?.focus({ preventScroll: true });
  });
  useEffect(() => {
    if (!opened) return;
    const frame = requestAnimationFrame(focusInside);
    const handleEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      const modal = [...document.querySelectorAll<HTMLElement>('[aria-modal="true"]')].some(el =>
        typeof el.checkVisibility === 'function' ? el.checkVisibility() : getComputedStyle(el).display !== 'none',
      );
      if (!modal) {
        event.preventDefault();
        closeFromKey();
      }
    };
    document.addEventListener('keydown', handleEscape, true);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('keydown', handleEscape, true);
    };
  }, [opened]);

  const style = sheet
    ? { display: opened ? 'flex' : 'none' }
    : // Its height is a cap, not a size: the window fits what's in it (#2450 board item 3: no blank half-window under
      // the question box), and scrolls inside once an answer runs past it.
      { display: opened ? 'flex' : 'none', left: box.left, top: box.top, width: box.width, maxHeight: box.height };
  return createPortal(
    <section
      ref={root}
      role="dialog"
      aria-modal="false"
      aria-label={title}
      hidden={!opened}
      data-ask-panel={kind}
      data-ask-open={opened ? 'true' : 'false'}
      data-ask-sheet={sheet ? 'true' : undefined}
      className={`${classes.panel} ${sheet ? classes.sheet : classes.window} ${className}`}
      style={style}
    >
      {/* Dragging is a pointer convenience; the move handle inside does the same from the keyboard. */}
      <header
        className={`${classes.heading}${sheet ? '' : ` ${classes.handle}`}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        {!sheet && (
          <ActionIcon
            variant="subtle"
            size="sm"
            data-ask-move
            aria-label={`Move ${title} with the arrow keys`}
            onKeyDown={onMoveKey}
            className={classes.grip}
          >
            <IconGripHorizontal size={16} />
          </ActionIcon>
        )}
        <Text fw={650} size="sm" className={classes.title}>
          {title}
        </Text>
        <ActionIcon variant="subtle" aria-label={`Close ${title}`} onClick={close}>
          <IconX size={18} />
        </ActionIcon>
      </header>
      <div className={classes.body}>{children}</div>
    </section>,
    document.body,
  );
}
