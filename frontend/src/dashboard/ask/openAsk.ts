// The one Ask: whether it is open, and what the page that opened it asked it to start from. Every way in (the top
// bar's button, the "/" key, the command palette, Review, and the Inbox) calls openAsk, so Ask is the same component
// with the same behaviour wherever it is opened from; a page only passes a starting question, a hint of what is on
// screen, or Deep research switched on. A hint shows as a chip over the box, which the person can remove.
//
// Review also tells Ask about itself (useAskPage): the day it shows, so the top bar's Ask opened there starts with
// that day as a hint, and how to play a moment and mark the timeline, so an answer's moments play in place.
import { create } from 'zustand';
import { type TabId, useRoute } from '../router';

/** What a question starts from, shown as a removable chip over the box: a research report to follow up on (the
 *  server reads the report by its id, memory/ask.py followup_report), or the day on screen in Review. */
export type AskHint =
  | { kind: 'report'; id: string; question: string; day?: string }
  | { kind: 'day'; day: string; label: string; today: boolean };

export interface AskRequest {
  /** null: start from no hint (the top bar off Review); left out: keep the hint the box has */
  hint?: AskHint | null;
  draft?: string;
  /** Deep research switched on for the next question (the Inbox's "Research a question") */
  deep?: boolean;
}

interface AskState {
  open: boolean;
  /** The latest request, until the Ask box takes it; `n` tells two identical requests apart. */
  request: (AskRequest & { n: number }) | null;
  take: () => (AskRequest & { n: number }) | null;
}

export const useAsk = create<AskState>((set, get) => ({
  open: false,
  request: null,
  take: () => {
    const request = get().request;
    if (request) set({ request: null });
    return request;
  },
}));

let count = 0;

/** Open Ask, starting from `request` (an empty one keeps what the box already holds). */
export function openAsk(request: AskRequest = {}): void {
  count += 1;
  useAsk.setState({ open: true, request: { ...request, n: count } });
}

export function closeAsk(): void {
  useAsk.setState({ open: false });
}

/** What Review tells Ask while it shows a day. */
export interface AskPage {
  day: { day: string; label: string; today: boolean } | null;
  seek: ((t: number) => void) | null;
  mark: ((times: number[]) => void) | null;
}

export const useAskPage = create<AskPage>(() => ({ day: null, seek: null, mark: null }));

/** The hint for the day Review shows, or none. */
export function dayHint(page: AskPage = useAskPage.getState()): AskHint | undefined {
  return page.day ? { kind: 'day', ...page.day } : undefined;
}

/** Open Ask from the top bar, the "/" key, or the palette: on Review with the day it shows as a hint, elsewhere with
 *  none. The draft stays. */
export function openAskHere(tab: TabId = useRoute.getState().tab): void {
  openAsk({ hint: (tab === 'review' && dayHint()) || null });
}
