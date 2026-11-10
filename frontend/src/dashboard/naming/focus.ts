// Where focus goes when a naming row leaves the list (#2328, #2329): shared by every kind of naming inbox.

/** Where focus goes when a row leaves for saving: the next row still waiting, else the one before, else the heading,
 *  without scrolling, so the owner stays where they were in the list and can carry on down it (#2328). */
export function focusAfter(list: HTMLElement | null, id: string, fallback: HTMLElement | null) {
  const rows = [...(list?.querySelectorAll<HTMLElement>(':scope > li[data-row]') ?? [])];
  const at = rows.findIndex(r => r.dataset.row === id);
  const open = (r: HTMLElement) => r.dataset.row !== id && r.style.display !== 'none';
  const next = at < 0 ? undefined : (rows.slice(at + 1).find(open) ?? rows.slice(0, at).reverse().find(open));
  (next ?? fallback)?.focus({ preventScroll: true });
}

/** Where focus goes after an Undo brings a row back (or keeps it): the row, else the list's heading. The toast's Undo
 *  button went with the toast, so focus was on nothing; it only moves when it is still there (on the page or in a
 *  toast), never away from where the owner has gone since. The row can come back a moment later (its list is read
 *  again): until `give_up` the call reports false and does nothing while it isn't drawn, so the caller tries again on
 *  its next render; then the heading takes focus. True once settled. */
export function focusRestored(rowKey: string | null, headingId: string, giveUp = false): boolean {
  const at = document.activeElement;
  const lost =
    !at || at === document.body || !!at.closest('.mantine-Notifications-root, [role="status"], [role="alert"]');
  if (!lost) return true;
  const row = rowKey ? document.querySelector<HTMLElement>(`li[data-row="${CSS.escape(rowKey)}"]`) : null;
  if (rowKey && !row && !giveUp) return false;
  (row ?? document.getElementById(headingId))?.focus();
  return true;
}

export type Direction = 'up' | 'down' | 'left' | 'right';

/** The card beside, above, or below `from` on screen (#3241), by the cards' drawn rectangles, not their list order,
 *  so in a grid of several columns the arrows follow what the owner sees. Left and Right stay in `from`'s row (in one
 *  column they find nothing); Up and Down take the nearest row that way, and in it the card closest across. Hidden
 *  cards (a row saving, or folded away) are skipped. Null when there is none that way. */
export function neighbour(cards: HTMLElement[], from: HTMLElement, dir: Direction): HTMLElement | null {
  const at = from.getBoundingClientRect();
  const across = (r: DOMRect) => Math.abs(r.left + r.width / 2 - (at.left + at.width / 2));
  const shown = cards
    .filter(c => c !== from && c.style.display !== 'none')
    .map(card => ({ card, r: card.getBoundingClientRect() }));
  let found: { card: HTMLElement; r: DOMRect }[];
  if (dir === 'left' || dir === 'right') {
    const sameRow = shown.filter(({ r }) => r.top < at.bottom - 1 && r.bottom > at.top + 1);
    found = sameRow.filter(({ r }) => (dir === 'right' ? r.left >= at.right - 1 : r.right <= at.left + 1));
    found.sort((a, b) => across(a.r) - across(b.r));
  } else {
    const beyond = shown.filter(({ r }) => (dir === 'down' ? r.top >= at.bottom - 1 : r.bottom <= at.top + 1));
    if (!beyond.length) return null;
    const tops = beyond.map(({ r }) => r.top);
    const nearest = dir === 'down' ? Math.min(...tops) : Math.max(...tops);
    found = beyond.filter(({ r }) => Math.abs(r.top - nearest) < 1);
    found.sort((a, b) => across(a.r) - across(b.r));
  }
  return found[0]?.card ?? null;
}
