// What the objects' and unexplored areas' marks and panel share that isn't in the scene: which one is selected, and
// which kind is waiting to be put on the plan.
import { create } from 'zustand';
import type { AreaKind } from './areas';

export interface SelectedArea {
  kind: AreaKind;
  id: string;
}

interface AreasUi {
  selected: SelectedArea | null;
  adding: AreaKind | null;
  select: (kind: AreaKind | null, id?: string | null) => void;
  setAdding: (kind: AreaKind | null) => void;
}

export const useAreasUi = create<AreasUi>(set => ({
  selected: null,
  adding: null,
  select: (kind, id) => set({ selected: kind && id ? { kind, id } : null }),
  setAdding: adding => set({ adding }),
}));

/** The kind's word as the page shows it: "object" or "unexplored area". */
export const kindName = (kind: AreaKind): string => (kind === 'object' ? 'object' : 'unexplored area');

/** Whether `id` is the selected area of `kind`. */
export const isSelected = (selected: SelectedArea | null, kind: AreaKind, id: string): boolean =>
  selected?.kind === kind && selected.id === id;
