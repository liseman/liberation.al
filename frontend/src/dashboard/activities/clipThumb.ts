// A clip card's start-frame still, from the People cards' scene thumbnails (/people/video-thumb.jpg).
// The route serves a prepared file at once; a miss answers 202 and queues the decode in the dashboard's background
// crop worker, under the shared decode cap and its admission limit, so many cards never decode at once. A card asks
// only once it's near the screen, and asks again with backoff while the still is pending. 404 means no frame is kept.
import { useQuery } from '@tanstack/react-query';
import { type RefObject, useEffect, useState } from 'react';

export type ThumbState = 'ready' | 'pending' | 'none';

export const thumbUrl = (cam: string, start: number, end: number): string =>
  `/people/video-thumb.jpg?${new URLSearchParams({ cam, start: start.toFixed(3), end: end.toFixed(3) })}`;

/** 200 is a ready still (read through, so the image below comes from the browser's cache), 202 is queued. */
export async function fetchThumb(url: string, signal?: AbortSignal): Promise<ThumbState> {
  const response = await fetch(url, { signal });
  if (response.status === 200) {
    await response.blob();
    return 'ready';
  }
  await response.body?.cancel();
  if (response.status === 202) return 'pending';
  if (response.status === 404) return 'none';
  throw new Error(`The clip's still couldn't be read (${response.status}).`);
}

const FIRST_WAIT_MS = 2000; // the route's Retry-After
const LONGEST_WAIT_MS = 30_000;

/** How long to wait before asking again after `asked` pending answers: 2 s, growing by half each time, to 30 s. */
export const thumbWait = (asked: number): number =>
  Math.min(LONGEST_WAIT_MS, FIRST_WAIT_MS * 1.5 ** Math.max(0, asked - 1));

export function useClipThumb(cam: string, start: number, end: number, enabled: boolean) {
  const url = thumbUrl(cam, start, end);
  const query = useQuery({
    queryKey: ['activities', 'thumb', cam, start.toFixed(3)],
    queryFn: ({ signal }) => fetchThumb(url, signal),
    enabled,
    staleTime: Number.POSITIVE_INFINITY,
    retry: 1,
    refetchInterval: q => (q.state.data === 'pending' ? thumbWait(q.state.dataUpdateCount) : false),
  });
  return { url, state: query.data, waiting: enabled && (query.isPending || query.data === 'pending') };
}

/** True once the element has come within a screen of view; always true where IntersectionObserver is missing. */
export function useNearScreen(ref: RefObject<Element | null>): boolean {
  const [near, setNear] = useState(() => typeof IntersectionObserver === 'undefined');
  useEffect(() => {
    const element = ref.current;
    if (near || !element) return;
    const observer = new IntersectionObserver(
      entries => {
        if (entries.some(e => e.isIntersecting)) setNear(true);
      },
      { rootMargin: '100% 0px' },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, near]);
  return near;
}
