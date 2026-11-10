// Shared live-scene shape. Who's this? uses saved shot URLs only.
import type { SceneEntry } from '../live/types';
import type { SceneFace } from './model';

export interface LiveScene extends SceneEntry {
  caption?: string;
  faces?: SceneFace[];
}
