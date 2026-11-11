// The scene's objects and unexplored areas for the editor's own components: read from the editor's scene store, and
// written back to the level's metadata as one change, so each is a step the editor undoes and saves like any other.
import { useScene } from '@pascal-app/core';
import { useCallback, useMemo } from 'react';
import { level } from '../dashboard/places/scene/draw';
import { type Area, type AreaKind, areas, withArea } from './areas';
import type { Scene } from './protocol';
import { currentScene } from './useSpots';

/** Set an object's or an unexplored area's record in the editor's scene, or take it away with null. */
export function setArea(kind: AreaKind, id: string, area: Area | null): void {
  const scene = currentScene();
  const storey = level(scene);
  const metadata = withArea(scene, kind, id, area);
  if (!storey || !metadata) return;
  useScene.getState().updateNode(storey.id as never, { metadata } as never);
}

export function useAreas(): { scene: Scene; areas: Record<AreaKind, Area[]>; set: typeof setArea } {
  const nodes = useScene(state => state.nodes);
  const rootNodeIds = useScene(state => state.rootNodeIds);
  const scene = useMemo(() => ({ nodes, rootNodeIds }) as unknown as Scene, [nodes, rootNodeIds]);
  const found = useMemo(() => areas(scene), [scene]);
  const set = useCallback(setArea, []);
  return { scene, areas: found, set };
}
