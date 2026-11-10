// Keeps the latest picture of every live camera from app start (thumbs.ts), so Live's tiles paint at once. Its own
// component so the status clock it reads re-renders only it, never the app around it.
import { useEffect } from 'react';
import { useLive } from '../live/hooks';
import { useLiveStatus } from '../live/status';
import { usePausedCameras } from '../privacy';
import { useActiveCameras } from '../shell';
import { VISION_FRESH_S } from '../status';
import { keepThumb } from './thumbs';

/** The cameras worth a picture: active, not paused for privacy, and on by vision's fresh word (Live's own test). A
 *  camera that's off would only fail each request, and a failure reopens the shared connection. */
export function thumbCams(
  active: readonly string[],
  paused: ReadonlySet<string>,
  states: Record<string, { state?: string } | undefined> | undefined,
): string[] {
  return active.filter(cam => !paused.has(cam) && states?.[cam]?.state === 'on');
}

/** Keeps a picture of each of cams; cameras that leave the list lose theirs. */
function useThumbFeed(cams: readonly string[]): void {
  const key = [...cams].sort().join('\n');
  useEffect(() => {
    const offs = key ? key.split('\n').map(keepThumb) : [];
    return () => {
      for (const off of offs) off();
    };
  }, [key]);
}

export function ThumbFeed() {
  const active = useActiveCameras();
  const paused = usePausedCameras();
  const vision = useLive('vision');
  // Seconds; only whether vision's word is fresh matters, so this re-renders when that flips, not every tick.
  const fresh = useLiveStatus(s => !!vision?.ts && s.now - vision.ts <= VISION_FRESH_S);
  useThumbFeed(fresh ? thumbCams(active, paused, vision?.cameras) : []);
  return null;
}
