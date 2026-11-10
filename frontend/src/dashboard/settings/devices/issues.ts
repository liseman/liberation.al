// Lightweight Settings menu health: importing a badge must not load device editors.
import { useQuery } from '@tanstack/react-query';
import { useLive } from '../../live/hooks';
import { useLiveStatus } from '../../live/status';
import type { AmbientDevice } from '../../live/types';
import { unnamed } from '../../review/views';
import { shell } from '../../shell';
import { VISION_FRESH_S } from '../../status';
import type { SectionBadge } from '../catalog';
import { CONFIG_KEY, type ConfigView, listText } from '../model';
import { parsePlaces } from './model';

/** Vision's word on each camera, while its status is fresh (vision is running). */
export function useCameraStates() {
  const vision = useLive('vision');
  const now = useLiveStatus(s => s.now);
  return vision?.ts && now - vision.ts <= VISION_FRESH_S ? (vision.cameras ?? {}) : null;
}

/** The camera names saved: Settings' copy of the config once it has loaded (so a name just saved counts), else the
 *  page shell's. Reads the cache only; the badge never fetches. */
function useSavedViews(): Readonly<Record<string, string>> {
  const { data } = useQuery<ConfigView>({ queryKey: CONFIG_KEY, enabled: false });
  const row = data?.sections.flatMap(s => s.settings).find(r => r.key === 'vision.camera_names');
  if (!row) return shell().views;
  return parsePlaces(Array.isArray(row.value) ? listText(row.value) : String(row.value ?? ''), shell().cams);
}

/** The menu badge: cameras being reopened and microphones that aren't responding, else cameras still to name. */
export function useDeviceIssues(): SectionBadge | null {
  const cams = useCameraStates();
  const ambient = useLive('ambient');
  const views = useSavedViews();
  const n =
    Object.values(cams ?? {}).filter(c => c.state === 'retrying').length +
    Object.values((ambient?.devices ?? {}) as Record<string, AmbientDevice>).filter(d => d.healthy === false).length;
  if (n) return { text: n === 1 ? '1 issue' : `${n} issues`, tone: 'warn' };
  // A camera with no name goes by its id on every page
  const missing = unnamed(shell().activeCams, views).length;
  return missing ? { text: missing === 1 ? 'Name 1 camera' : `Name ${missing} cameras`, tone: 'dim' } : null;
}
