// The scene's camera spots for the editor's own components: read from the editor's scene store, and written back
// to the level's metadata as one change, so each is a step the editor undoes and saves like any other.
import { useScene } from '@pascal-app/core';
import { useCallback, useMemo } from 'react';
import { level } from '../dashboard/places/scene/draw';
import type { Scene } from './protocol';
import { type Spot, spots, withSpot } from './spots';

/** The editor's scene as our own readers take it. */
export function currentScene(): Scene {
  const state = useScene.getState();
  return { nodes: state.nodes, rootNodeIds: state.rootNodeIds } as unknown as Scene;
}

/** Set a camera's spot in the editor's scene, or take it away with null. */
export function setSpot(camera: string, spot: Spot | null): void {
  const scene = currentScene();
  const storey = level(scene);
  const metadata = withSpot(scene, camera, spot);
  if (!storey || !metadata) return;
  useScene.getState().updateNode(storey.id as never, { metadata } as never);
}

export function useSpots(): { scene: Scene; spots: Record<string, Spot>; set: typeof setSpot } {
  const nodes = useScene(state => state.nodes);
  const rootNodeIds = useScene(state => state.rootNodeIds);
  const scene = useMemo(() => ({ nodes, rootNodeIds }) as unknown as Scene, [nodes, rootNodeIds]);
  const found = useMemo(() => spots(scene), [scene]);
  const set = useCallback(setSpot, []);
  return { scene, spots: found, set };
}
