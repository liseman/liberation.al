// A part of a page whose height arrives with its data holds its last height in this browser while it loads,
// so what's under it doesn't move when the data lands. Per viewer and per window width band, and only a hint: storage
// can be missing or throw (a private window), and then the part just takes its natural height.
import { type RefCallback, useMemo } from 'react';

const band = (): string => (typeof innerWidth === 'number' && innerWidth <= 700 ? 'narrow' : 'wide');
const keyOf = (name: string): string => `liberational.height.${name}.${band()}`;

/** The height `name` last had at this width, in px, if it was remembered. */
export function rememberedHeight(name: string): number | undefined {
  try {
    const h = Number(localStorage.getItem(keyOf(name)));
    return h > 0 ? h : undefined;
  } catch {
    return undefined;
  }
}

/** A ref that keeps `name`'s height remembered while the element is on the page (skipped while `holding`, so a
 *  placeholder's own height is never what is remembered). */
export function useRememberHeight<T extends HTMLElement>(name: string, holding = false): RefCallback<T> {
  return useMemo(
    () => (el: T | null) => {
      if (!el || holding || typeof ResizeObserver !== 'function') return;
      const ro = new ResizeObserver(() => {
        const h = Math.round(el.getBoundingClientRect().height);
        try {
          if (h > 0) localStorage.setItem(keyOf(name), String(h));
        } catch {
          // not remembered this time
        }
      });
      ro.observe(el);
      return () => ro.disconnect();
    },
    [name, holding],
  );
}
