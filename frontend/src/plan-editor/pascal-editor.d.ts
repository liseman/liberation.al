// The part of `@pascal-app/editor` this app uses, typed here. The package ships as TSX source written for its own
// compiler settings, so `tsconfig.json` points the type checker at this file instead of at that source (Vite builds
// the real thing). Keep it to what `src/plan-editor/` imports, and check it against the source when the pin moves.
import type { ComponentType, ReactNode } from 'react';
import type { Scene } from './protocol';

export interface SidebarTab {
  id: string;
  label: string;
  icon?: ReactNode;
  mobileIcon?: ReactNode;
}

export interface EditorProps {
  layoutVersion?: 'v1' | 'v2';
  projectId?: string | null;
  sidebarTabs?: (SidebarTab & { component: ComponentType })[];
  /** Host-owned SVG content mounted in the 2D plan, in the plan's own coordinates (metres). */
  floorplanSceneSlot?: ReactNode;
  /** Show the scene with editing off: the editor's own mode for looking at a saved version. */
  isVersionPreviewMode?: boolean;
  onLoad?: () => Promise<Scene | null>;
  onSave?: (scene: Scene, options?: { keepalive?: boolean }) => Promise<void>;
}
export const Editor: ComponentType<EditorProps>;

/** How the wall tool draws: one wall after another, or a whole room at once. */
export type WallDrawVariant = string;

export interface EditorState {
  mode: string;
  tool: string | null;
  setPhase: (phase: 'site' | 'building') => void;
  setMode: (mode: 'select' | 'build') => void;
  setTool: (tool: string | null) => void;
  setCatalogCategory: (category: null) => void;
  setToolDefaults: (kind: string, defaults: null) => void;
  showView: (view: string, options?: { beside?: boolean }) => void;
}
export const useEditor: {
  <T>(selector: (state: EditorState) => T): T;
  getState: () => EditorState;
};

/** Select the scene's first building and its ground level, as the editor does when it enters its building phase. */
export function selectDefaultBuildingAndLevel(): void;

export const useAudio: { getState: () => { setMuted: (muted: boolean) => void } };

export function useWallDrawVariant(): WallDrawVariant;
export function selectWallDrawVariant(variant: WallDrawVariant): void;

export const BuildPanelRoomsSection: ComponentType<{
  activeVariant: WallDrawVariant | null;
  onSelect: (variant: WallDrawVariant) => void;
}>;
export const BuildPanelSection: ComponentType<{ id: string; title: string; hint?: string; children: ReactNode }>;
export const BuildToolGrid: ComponentType<{ columns: 3 | 4; children: ReactNode }>;
export const BuildToolTile: ComponentType<{
  active: boolean;
  iconSrc: string;
  label: string;
  onClick: () => void;
  title?: string;
}>;

/** From `@pascal-app/editor/locales`: registers the editor's own message catalogues. */
export function registerBundledLocales(): void;
