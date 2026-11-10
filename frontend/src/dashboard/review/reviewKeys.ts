// Review's keyboard shortcuts, in one table (#2285): the engine dispatches on it (engine.ts onKey and onSpace), and
// the control bar's "?" lists it (ReviewHelp.tsx), so the two can't drift apart. A key added here needs a handler
// for its id in the engine, or the type check fails; one removed here goes from both at once.
// Shortcuts that belong to one focused control (a search box, the ratings group, the timeline's marking tool) stay
// with that control and aren't listed.

export interface ReviewKey {
  id: 'play' | 'nudge' | 'ends' | 'lines' | 'mark' | 'clear' | 'moments' | 'skip' | 'zoom';
  group: 'play' | 'move' | 'mark'; // its section in the "?" table (#2577)
  keys: readonly string[]; // KeyboardEvent.key values
  shown: readonly string[]; // as the "?" table prints them
  does: string; // what they do, beside the keys ("Play or pause")
}

/** The "?" table's sections, in order. */
export const KEY_GROUPS: readonly { id: ReviewKey['group']; title: string }[] = [
  { id: 'play', title: 'Play' },
  { id: 'move', title: 'Move around' },
  { id: 'mark', title: 'Mark an event' },
];

// In the order the "?" table lists them: by section, then as below.
export const REVIEW_KEYS: readonly ReviewKey[] = [
  { id: 'play', group: 'play', keys: [' '], shown: ['Space'], does: 'Play or pause' },
  {
    id: 'nudge',
    group: 'play',
    keys: ['ArrowLeft', 'ArrowRight'],
    shown: ['←', '→'],
    does: 'Nudge back or on (Shift: further)',
  },
  { id: 'ends', group: 'play', keys: ['Home', 'End'], shown: ['Home', 'End'], does: 'Go to the start or end' },
  { id: 'skip', group: 'play', keys: ['s'], shown: ['s'], does: 'Skip quiet time on or off' },
  { id: 'lines', group: 'move', keys: ['j', 'k'], shown: ['j', 'k'], does: 'Previous or next transcript line' },
  {
    id: 'moments',
    group: 'move',
    keys: ['[', ']'],
    shown: ['[', ']'],
    does: 'Previous or next moment Show only picks',
  },
  { id: 'zoom', group: 'move', keys: ['+', '=', '-'], shown: ['+', '-'], does: 'Zoom the timeline in or out' },
  { id: 'mark', group: 'mark', keys: ['i', 'o'], shown: ['I', 'O'], does: "Mark the event's start or end" },
  { id: 'clear', group: 'mark', keys: ['Escape'], shown: ['Escape'], does: 'Clear the marks' },
];

const BY_KEY = new Map(REVIEW_KEYS.flatMap(k => k.keys.map(key => [key, k] as const)));

/** The shortcut a KeyboardEvent.key is, if it's one of Review's. */
export const shortcut = (key: string): ReviewKey | undefined => BY_KEY.get(key);
