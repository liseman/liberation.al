// Settings' draft (#382): what has been typed and not saved, shared by every section, the search, and the save bar.
// Server values are query data; the draft is the page's own, so moving between sections (or tabs) keeps it. A
// section's own panel (catalog.ts Body) reads and changes it through useDraft.
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createContext, type ReactNode, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { HttpError, messageOf, request } from '../../shared/http';
import { setCameraNames, shell } from '../shell';
import { parsePlaces } from './devices/model';
import {
  blank,
  CONFIG_KEY,
  type ConfigReply,
  type ConfigView,
  initialValues,
  isRequired,
  restartDestination,
  type Setting,
  sameDocument,
  type Values,
} from './model';
import { followRestart } from './restart';

export interface Draft {
  view: ConfigView;
  all: Setting[]; // every setting, in defaults.toml order
  rows: (section: string) => Setting[]; // a section's settings (Advanced: all of them)
  value: (key: string) => string | boolean | undefined;
  set: (key: string, value: string | boolean) => void;
  reset: (key: string) => void; // back to what is saved
  changed: (key: string) => boolean; // differs from what is saved
  changedKeys: string[];
  missing: Setting[]; // required and blank
  discard: () => void;
  save: () => void;
  busy: boolean;
  note: string; // "Saving", "Saved to pack.toml. Restarting..."
  failure: { error: string; field?: string } | null;
}

const DraftContext = createContext<Draft | null>(null);

export function useDraft(): Draft {
  const d = useContext(DraftContext);
  if (!d) throw new Error('useDraft is for Settings sections, inside DraftProvider');
  return d;
}

export function DraftProvider({ view, children }: { view: ConfigView; children: ReactNode }) {
  const client = useQueryClient();
  const saved = useMemo(() => initialValues(view), [view]); // what the server has: the view, as drafts read it
  const [values, setValues] = useState<Values>(saved);
  const [note, setNote] = useState('');
  const [failure, setFailure] = useState<{ error: string; field?: string } | null>(null);
  const controller = useRef<AbortController | null>(null);
  // The page stays mounted (#330), so the settings can change elsewhere meanwhile (Sources writes the devices): a
  // fresh view is what is saved now, and the draft keeps only what was changed here.
  const before = useRef(saved);
  useEffect(() => {
    const old = before.current;
    if (old === saved) return;
    before.current = saved;
    setValues(current => ({
      ...saved,
      ...Object.fromEntries(Object.entries(current).filter(([k, v]) => v !== old[k])),
    }));
  }, [saved]);
  // No abort on cleanup: a hidden tab cleans its effects up too (#330), and leaving Settings mid-save must not stop
  // the save or the restart it follows.
  const all = useMemo(() => view.sections.flatMap(s => s.settings), [view]);
  const mutation = useMutation({
    mutationFn: async () => {
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;
      const { boot } = await request<{ boot: string }>('/health', undefined, abort.signal);
      // Only what was changed here, so a setting changed elsewhere since this page opened is never written back.
      const changes = Object.fromEntries(Object.entries(values).filter(([k, v]) => v !== saved[k]));
      const reply = await request<ConfigReply>('/config', { values: changes }, abort.signal);
      if (!reply.ok) throw new HttpError(reply.error ?? 'Saving failed', 400, reply);
      // Camera names show on every page from the page shell: a saved rename reaches them now, not at the next load
      if ('vision.camera_names' in changes) setCameraNames(parsePlaces(changes['vision.camera_names'], shell().cams));
      await client.invalidateQueries({ queryKey: CONFIG_KEY, refetchType: 'none' });
      const overrides = reply.overridden?.length
        ? ` ${reply.overridden.join(', ')} stay as the environment sets them.`
        : '';
      if (reply.restarting === false) {
        // The dashboard stays up (#400): show the saved settings here, and say what restarts in the background. A
        // secret is never sent back, so the one just saved is cleared rather than left looking unsaved.
        await client.invalidateQueries({ queryKey: CONFIG_KEY });
        const secrets = all.filter(r => r.secret && r.key in changes).map(r => r.key);
        if (secrets.length) setValues(current => ({ ...current, ...Object.fromEntries(secrets.map(k => [k, ''])) }));
        const restarted = reply.restarted?.length ? ` Restarting ${reply.restarted.join(', ')}.` : '';
        setNote(`Saved to ${view.path}.${restarted}${overrides}`);
        return;
      }
      setNote(`Saved to ${view.path}. Restarting with the new settings.${overrides}`);
      const url = restartDestination(reply.url);
      await followRestart(boot, url, abort.signal);
      if (abort.signal.aborted) return;
      if (sameDocument(url))
        location.reload(); // assigning only a new #hash would not reload
      else location.assign(url);
    },
    onError: error => {
      if (controller.current?.signal.aborted) return;
      const data = error instanceof HttpError ? error.data : null;
      const field =
        data && typeof data === 'object' && 'field' in data && typeof data.field === 'string' ? data.field : undefined;
      setNote('');
      setFailure({ error: messageOf(error), field });
    },
  });
  const busy = mutation.isPending;
  const changedKeys = Object.keys(values).filter(key => values[key] !== saved[key]);
  const missing = all.filter(row => isRequired(row) && blank(values[row.key]));
  const draft: Draft = {
    view,
    all,
    // A section lists its settings in labels.toml's order; Advanced lists every one in defaults.toml's
    rows: section =>
      section === 'advanced'
        ? all
        : all.filter(row => row.section === section).sort((a, b) => (a.order ?? 1e4) - (b.order ?? 1e4)),
    value: key => values[key],
    set: (key, value) => {
      setNote(''); // a "Saved" note from a save that kept this page (#400) gives way to the new change
      setValues(current => ({ ...current, [key]: value }));
    },
    reset: key => setValues(current => ({ ...current, [key]: saved[key] as string | boolean })),
    changed: key => values[key] !== saved[key],
    changedKeys,
    missing,
    discard: () => {
      setValues(saved);
      setFailure(null);
      setNote('');
    },
    save: () => {
      if (!view.control || busy || missing.length || !changedKeys.length) return;
      setFailure(null);
      setNote('Saving');
      mutation.mutate();
    },
    busy,
    note,
    failure,
  };
  return <DraftContext value={draft}>{children}</DraftContext>;
}
