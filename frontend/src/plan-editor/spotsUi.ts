// What the camera spots' marks and panel share that isn't in the scene: which spot is selected, which camera is
// waiting to be put on the plan, and the cameras the host says the house has.
import { create } from 'zustand';

export interface CameraName {
  id: string;
  /** The name the page shows: the camera's friendly name when it has one. */
  name: string;
}

interface SpotsUi {
  selected: string | null;
  adding: string | null;
  cameras: CameraName[];
  select: (camera: string | null) => void;
  setAdding: (camera: string | null) => void;
  setCameras: (cameras: CameraName[]) => void;
}

export const useSpotsUi = create<SpotsUi>(set => ({
  selected: null,
  adding: null,
  cameras: [],
  select: selected => set({ selected }),
  setAdding: adding => set({ adding }),
  setCameras: cameras => set({ cameras }),
}));

/** A camera's name as shown, from the host's list, or its id. */
export const cameraName = (cameras: CameraName[], id: string): string => cameras.find(c => c.id === id)?.name ?? id;
