import { createContext, useContext } from 'react';
import { useStore } from 'zustand';
import type { ReviewEngine, ReviewUi } from './engine';

export const EngineContext = createContext<ReviewEngine | null>(null);

/** The Review page's engine. */
export function useEngine(): ReviewEngine {
  const engine = useContext(EngineContext);
  if (!engine) throw new Error('useEngine needs a Review page around it');
  return engine;
}

/** One piece of the engine's UI state; the component re-renders when it changes. */
export function useReview<T>(pick: (s: ReviewUi) => T): T {
  return useStore(useEngine().ui, pick);
}
