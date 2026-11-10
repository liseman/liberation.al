import { Skeleton } from '@mantine/core';
import { IconPhotoOff } from '@tabler/icons-react';
import { type ComponentPropsWithoutRef, type ReactNode, useEffect, useState } from 'react';

/** How long a failed image waits before each of its `retries` (a face crop the dashboard was too busy to cut, #2018). */
export const RETRY_MS = [2000, 5000];

/** Missing retained media gets a neutral placeholder, and a new URL gets its own load attempt. With `retries`, a
 *  failed load is tried again that many times (at most RETRY_MS.length) before the placeholder.
 *  An explicit null fallback omits an unavailable image.
 *
 *  While it loads, and through its retries, a skeleton holds the image's place (#2049): the box the caller sizes
 *  (its className, width, and height, or a `[data-safe-image]` rule, as for the placeholder) shows Mantine's
 *  Skeleton until the picture arrives, so nothing shifts and no box sits blank. An image the caller sizes fills that
 *  box; an unsized one keeps its own size, set by the caller's `img` rules or the picture itself. */
export function SafeImage({
  src,
  alt = '',
  className,
  style,
  width,
  height,
  onError,
  fallback,
  retries = 0,
  ...props
}: ComponentPropsWithoutRef<'img'> & { fallback?: ReactNode; retries?: number }) {
  const [failed, setFailed] = useState<string | undefined>();
  const [tried, setTried] = useState<{ src?: string; n: number; waiting: boolean }>({ n: 0, waiting: false });
  const attempt = tried.src === src ? tried : { src, n: 0, waiting: false };
  const shown = src && attempt.n ? `${src}${src.includes('?') ? '&' : '?'}retry=${attempt.n}` : src;
  const [loaded, setLoaded] = useState<string | undefined>();
  useEffect(() => {
    if (!attempt.waiting) return;
    const t = setTimeout(() => setTried({ src, n: attempt.n + 1, waiting: false }), RETRY_MS[attempt.n] ?? 0);
    return () => clearTimeout(t);
  }, [attempt.waiting, attempt.n, src]);
  if ((!src || failed === src) && fallback === null) return null;
  if (!src || failed === src)
    return (
      <span
        data-safe-image
        className={className}
        style={{
          display: 'inline-grid',
          placeItems: 'center',
          background: 'var(--mantine-color-default-hover)',
          width,
          height,
          ...style,
        }}
        // A caller's own fallback describes the unavailable image; the box stays out of its way.
        {...(fallback
          ? {}
          : {
              role: 'img',
              'aria-label': alt ? `${alt}: image unavailable` : undefined,
              'aria-hidden': !alt || undefined,
              title: 'Image unavailable',
            })}
      >
        {fallback ?? <IconPhotoOff size="1.5em" aria-hidden="true" />}
      </span>
    );
  const sized = Boolean(className || width || height);
  return (
    <Skeleton
      data-safe-image
      visible={loaded !== shown}
      className={className}
      style={{ ...(sized ? { width, height, overflow: 'hidden' } : { display: 'inline-block' }), ...style }}
    >
      <img
        {...props}
        key={attempt.n}
        src={shown}
        alt={alt}
        width={width}
        height={height}
        style={
          sized
            ? { display: 'block', width: '100%', height: '100%', objectFit: 'inherit', borderRadius: 'inherit' }
            : undefined
        }
        onLoad={event => {
          setLoaded(shown);
          props.onLoad?.(event);
        }}
        onError={event => {
          if (attempt.n < Math.min(retries, RETRY_MS.length) && !attempt.waiting)
            return setTried({ src, n: attempt.n, waiting: true });
          setFailed(src);
          onError?.(event);
        }}
      />
    </Skeleton>
  );
}
