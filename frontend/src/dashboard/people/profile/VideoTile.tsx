// One recent video on a card (#2058): its recorded frame, a play badge, its length, what was happening, and when and
// on which camera. Playing it opens the floating player. People's card and a pet's card both show these. The player
// (Review's engine underneath) loads on the first play, so a page of cards doesn't fetch it (#3127).
import { Text, UnstyledButton } from '@mantine/core';
import { lazy, type ReactNode, Suspense, useState } from 'react';
import { ClipPoster } from '../../clip/ClipPoster';
import { cameraRatio } from '../../clip/ratio';
import type { CardVideo } from '../api';
import classes from './profile.module.css';
import { cameraName, dayTime } from './role';

const VideoFloat = lazy(() => import('./VideoFloat').then(m => ({ default: m.VideoFloat })));

export function VideoTile({
  v,
  name,
  menu,
  person,
}: {
  v: CardVideo;
  name: string; // whose card it's on, for the player's label
  menu?: ReactNode; // the card's own actions for it (People's Not them / Hide)
  person?: string | null; // Review's person filter from the player; defaults to `name`, null for none (a pet)
}) {
  const when = dayTime(v.start);
  const camera = cameraName(v.cam, v.camera_name);
  const what = v.what || `Seen by ${camera}`;
  const [floating, setFloating] = useState<{ origin: HTMLElement; video: CardVideo } | null>(null);
  return (
    <li className={classes.video}>
      <UnstyledButton
        type="button"
        className={classes.link}
        onClick={event => setFloating({ origin: event.currentTarget, video: v })}
        aria-label={`Play video: ${what}, ${when}`}
      >
        <ClipPoster src={v.thumb} seconds={v.duration_s ?? v.end - v.start} ratio={cameraRatio(v.cam)} />
      </UnstyledButton>
      {floating && (
        <Suspense fallback={null}>
          <VideoFloat
            video={floating.video}
            name={name}
            person={person}
            origin={floating.origin}
            onClose={() => setFloating(null)}
          />
        </Suspense>
      )}
      <span className={classes.videoHead}>
        <span className={classes.videoWhat}>{what}</span>
        {menu}
      </span>
      <Text span size="xs" c="dimmed">
        {`${when}, ${camera}`}
      </Text>
    </li>
  );
}
