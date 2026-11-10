// Clone state shared by the Voice clones sidebar and the Clone buttons on the roster: the voice picked in the sidebar,
// and the voice whose reference is being built or removed right now (one at a time, so every clone button waits).
// The candidates themselves are server data (CLONES_KEY). Ported from static/dashboard/voice-tools.js.
import type { QueryClient } from '@tanstack/react-query';
import { create } from 'zustand';
import { messageOf } from '../../../shared/http';
import { buildClone, CLONES_KEY, type Clone, type Person, removeClone } from '../api';

interface CloneState {
  sel: number | null;
  busy: number | null;
  pick: (id: number) => void;
}

/** The text box the picked voice speaks from, which pick() focuses. */
export const CLONE_TEXT_ID = 'clone-text';

export const useClones = create<CloneState>(set => ({
  sel: null,
  busy: null,
  pick: id => {
    set({ sel: id });
    requestAnimationFrame(() => {
      const el = document.getElementById(CLONE_TEXT_ID);
      el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      el?.focus({ preventScroll: true });
    });
  },
}));

/** A person's clone candidate: the first of their voices that is built, else the first that is ready, else the
 * first known one. */
export function cloneOf(p: Pick<Person, 'voices'>, list: readonly Clone[]): Clone | null {
  const mine = p.voices.map(id => list.find(c => c.id === id)).filter((c): c is Clone => !!c);
  return mine.find(c => c.built) ?? mine.find(c => c.ready) ?? mine[0] ?? null;
}

type Done = { ok: true; utterances?: number; seconds?: number; removed?: string[] } | { ok: false; error: string };

/** Run one build or removal at a time (synchronous on the server: a build takes 5 to 20 s), then reload the list.
 * A second call while one runs is refused, so a double click builds once. */
async function exclusive(
  client: QueryClient,
  id: number,
  run: () => Promise<unknown>,
  fallback: string,
): Promise<Done> {
  if (useClones.getState().busy != null) return { ok: false, error: 'Another voice is being worked on; wait for it.' };
  useClones.setState({ busy: id });
  try {
    const r = (await run()) as { ok?: boolean; error?: string } & Record<string, unknown>;
    return r.ok ? ({ ...r, ok: true } as Done) : { ok: false, error: r.error || fallback };
  } catch (e) {
    return { ok: false, error: `${fallback.replace(/\.$/, '')}: ${messageOf(e)}` };
  } finally {
    useClones.setState({ busy: null });
    void client.invalidateQueries({ queryKey: CLONES_KEY });
  }
}

export const build = (client: QueryClient, id: number) =>
  exclusive(client, id, () => buildClone(id), 'Building failed.');

export const remove = (client: QueryClient, id: number) =>
  exclusive(client, id, () => removeClone(id), 'Removing failed.');
