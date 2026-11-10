// A clip before it plays (#2476): its start frame, a play badge, and its length as m:ss. An Activities card, a People
// card's recent video, and the Pickers' unnamed videos all show this; the button around it is the caller's.
import { Skeleton } from '@mantine/core';
import { IconPlayerPlayFilled } from '@tabler/icons-react';
import type { CSSProperties } from 'react';
import { SafeImage } from '../../shared/SafeImage';
import { clipLength } from '../lib/format';
import classes from './clipPoster.module.css';
import { ratioOf } from './ratio';

export function ClipPoster({
  src,
  seconds,
  waiting = false,
  badge = true,
  ratio = null,
  onShape,
}: {
  src: string | null | undefined; // the start frame, when there is one
  seconds: number;
  waiting?: boolean; // the frame is still being cut: a skeleton holds its place
  badge?: boolean; // the play badge (off on a tile that picks rather than plays)
  ratio?: number | null; // the camera's width over height, when known; 16:9 until then
  onShape?: (ratio: number) => void; // the frame's own shape, once it loads
}) {
  return (
    <span className={classes.poster} style={ratio ? ({ '--clip-ratio': ratio } as CSSProperties) : undefined}>
      {src ? (
        <SafeImage
          className={classes.still}
          src={src}
          alt=""
          loading="lazy"
          fallback={null}
          onLoad={e => {
            const found = ratioOf(e.currentTarget.naturalWidth, e.currentTarget.naturalHeight);
            if (found) onShape?.(found);
          }}
        />
      ) : waiting ? (
        <Skeleton className={classes.still} radius={0} />
      ) : null}
      {badge && (
        <span className={classes.badge} aria-hidden="true">
          <IconPlayerPlayFilled size={18} />
        </span>
      )}
      <span className={classes.length}>{clipLength(seconds)}</span>
    </span>
  );
}
