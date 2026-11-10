// Open the top bar's Ask from anywhere, with something to start from (#2594 follow-up, for the Inbox's "Ask a
// follow-up"). The Ask panel opens, shows the context as a chip over the box ("Following up on: <question>"), and puts
// the draft in the box. Questions asked while the chip is there carry the report's id, and the server reads the
// report itself (memory/ask.py followup_report), so the page never sends the report's text.
import { create } from 'zustand';

/** A research report a question follows up on: its id (the server reads it by this), its question, and its day. */
export interface AskContext {
  kind: 'report';
  id: string;
  question: string;
  day?: string;
}

export interface AskRequest {
  context?: AskContext;
  draft?: string;
}

interface Opener {
  /** The latest request, until the Ask panel takes it; `n` tells two identical requests apart. */
  request: (AskRequest & { n: number }) | null;
  take: () => (AskRequest & { n: number }) | null;
}

export const useAskOpener = create<Opener>((set, get) => ({
  request: null,
  take: () => {
    const request = get().request;
    if (request) set({ request: null });
    return request;
  },
}));

let count = 0;

/** Open Ask with a starting context and draft. On Review the day's own Ask stays as it is; this opens the top bar's. */
export function openAsk(request: AskRequest = {}): void {
  count += 1;
  useAskOpener.setState({ request: { ...request, n: count } });
}
