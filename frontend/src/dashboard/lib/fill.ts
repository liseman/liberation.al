// A view that is the window's own height: from where it starts down to the page frame's gutter above the window's
// bottom edge (less a phone's bottom bar, which the CSS takes off). The start is measured, because the header wraps and
// what sits above the view is the page's to decide, and it is measured again after every render and when the window
// changes, so a longer heading or a wrapped bar moves the view instead of overflowing it.
//
// Such a view marks itself data-fills-page, which is how the page frame (page.module.css) and the card it sits in
// (places.module.css) know to give up their own room under it: without that, the document would end that much below the
// window and the page would scroll under a view meant to fill it.
import { type CSSProperties, useLayoutEffect, useState } from 'react';

/** The style carrying where `ref`'s element starts, for the view's own CSS (--fills-top, places.module.css). */
export function useFillsPage<T extends HTMLElement>(ref: { current: T | null }): CSSProperties {
  const [top, setTop] = useState(0);
  useLayoutEffect(() => {
    const measure = () => {
      const el = ref.current;
      // Round the start up, so the view is never a fraction taller than the room it has and the page never scrolls by a
      // sub-pixel.
      if (el) setTop(Math.ceil(el.getBoundingClientRect().top + window.scrollY));
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  });
  return { '--fills-top': `${top}px` } as CSSProperties;
}
