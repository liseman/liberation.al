// What the objects' and unexplored areas' marks and panel share that isn't in the scene: which one is selected, and
// which kind is waiting to be put on the plan.
import { create } from 'zustand';
import type { Area, AreaKind } from './areas';

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

/** The kind's name as the page shows it, as a heading or a name: "Object" or "Unexplored area". */
export const kindName = (kind: AreaKind): string => (kind === 'object' ? 'Object' : 'Unexplored area');

/** The kind's word inside a sentence: "object" or "unexplored area". */
export const kindWord = (kind: AreaKind): string => (kind === 'object' ? 'object' : 'unexplored area');

/** What a mark is called where it has no label of its own: "Unnamed object", or the kind's name. */
export const markName = (area: Area): string =>
  area.label || (area.kind === 'object' ? 'Unnamed object' : kindName(area.kind));

/** Whether `id` is the selected area of `kind`. */
export const isSelected = (selected: SelectedArea | null, kind: AreaKind, id: string): boolean =>
  selected?.kind === kind && selected.id === id;
