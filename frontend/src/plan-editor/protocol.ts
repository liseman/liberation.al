// What the Places page and the plan editor's page say to each other. The editor is a page of its own
// (plan-editor/index.html) in an iframe on the same origin; both sides check a message's origin and its `type`.

/** A Pascal scene as its editor loads and saves it: nodes by id, and the ids at the root. */
export interface Scene {
  nodes: Record<string, SceneNode>;
  rootNodeIds: string[];
}

/** One node of a scene. Every kind has these; the rest depends on `type`. */
export interface SceneNode {
  id: string;
  type: string;
  parentId: string | null;
  metadata?: Record<string, unknown>;
  children?: string[];
  [field: string]: unknown;
}

/** Which of the editor's views a frame opens on: its 2D editor, its 3D view, or both side by side. */
export type PlanView = '2d' | '3d' | 'split';

/** Editor to host: the page is up and waits for a scene. */
export interface Ready {
  type: 'plan-editor:ready';
}

/** Host to editor: the scene to open, and the house's cameras (id and the name the page shows), for the Cameras
 *  panel to offer the ones with no spot yet. */
export interface Open {
  type: 'plan-editor:open';
  scene: Scene;
  cameras?: { id: string; name: string }[];
}

/** Editor to host: the scene as it stands after an edit. */
export interface Changed {
  type: 'plan-editor:changed';
  scene: Scene;
}

export type FromEditor = Ready | Changed;

/** The editor page's address for a view and a colour scheme, under the build's base. */
export function editorUrl(base: string, view: PlanView, scheme: 'light' | 'dark'): string {
  return `${base}plan-editor/index.html#view=${view}&scheme=${scheme}`;
}

/** What an editor page's address asks for; anything else reads as the 2D editor, light. */
export function fromHash(hash: string): { view: PlanView; scheme: 'light' | 'dark' } {
  const q = new URLSearchParams(hash.replace(/^#/, ''));
  const view = q.get('view');
  return {
    view: view === '3d' || view === 'split' ? view : '2d',
    scheme: q.get('scheme') === 'dark' ? 'dark' : 'light',
  };
}

/** A message from this origin that is one of ours, or null. */
export function ours<T extends { type: string }>(event: MessageEvent, types: readonly T['type'][]): T | null {
  if (event.origin !== location.origin) return null;
  const data = event.data as { type?: unknown } | null;
  return data && typeof data.type === 'string' && (types as readonly string[]).includes(data.type) ? (data as T) : null;
}
