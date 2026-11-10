// What the browser runs on, where the dashboard's words depend on it.

/** A Mac (or iPhone or iPad): Cmd is its shortcut key, shown as ⌘, where other systems use Ctrl. */
export const MAC =
  typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
