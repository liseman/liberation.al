// Corrections from a transcript, the same on Live and Review: a turn put down to the wrong person opens Who is this?
// (move its lines to someone, or take them off this person, #300), a line's label comes off, or lines are marked not
// transcribed correctly. One change at a time, each with one Undo in a toast; lines that failed can be tried again.
import { Button } from '@mantine/core';
import { useQueryClient } from '@tanstack/react-query';
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { HttpError, messageOf } from '../../shared/http';
import { Toast as FloatingToast } from '../lib/Toast';
import type { TranscriptEntry } from '../live/types';
import { markTv, notTv, refreshTranscript, rejectLines, undoRejected } from '../livetab/api';
import { useKnownPeople } from '../livetab/people';
import { correct, undoChange } from '../people/api';
import type { Toast } from '../people/profile/parts';
import { WhoIsThis } from '../people/WhoIsThis';

/** A turn to correct: its exact visible lines and their current names, as /people/correct validates. */
export interface Not {
  lines: { key: string; name: string }[];
  name: string;
}
interface RetryCorrection {
  target: Not;
  action: 'move' | 'remove';
  to?: string;
  total: number;
  changed: number;
}
type Undo = { kind: 'people'; tokens: string[] } | { kind: 'transcript'; token: string };

export interface Corrections {
  saving: boolean;
  /** Open Who is this? for a turn; `from` gets focus back when it closes. */
  openNot: (n: Not, from: HTMLElement | null) => void;
  /** Mark these lines not transcribed correctly. */
  trash: (lines: readonly TranscriptEntry[]) => Promise<void>;
  /** "This is TV" (#2468): mark these lines as a TV's, with an Undo. */
  tv: (lines: readonly TranscriptEntry[]) => Promise<void>;
  /** "Not TV": take the TV marks off these lines. */
  notTv: (lines: readonly TranscriptEntry[]) => Promise<void>;
  /** The toast and Who is this?, to render once beside the list. */
  overlay: ReactNode;
}

/** `scope`: what the list shows (a day); a new one drops a toast about the last. `onChanged`: read the lines again,
 *  as the page reads them (the live feed is always read again too). */
export function useCorrections(scope: string, onChanged: () => void): Corrections {
  const client = useQueryClient();
  const known = useKnownPeople();
  const people = useMemo(() => known.map(name => ({ name })), [known]);
  const [not, setNot] = useState<Not | null>(null);
  const opener = useRef<HTMLElement | null>(null);
  const [toast, setToast] = useState<Toast>(null);
  const undoRef = useRef<Undo | null>(null);
  const [retry, setRetry] = useState<RetryCorrection | null>(null);
  const working = useRef(false);
  const [saving, setSaving] = useState(false);
  const changedRef = useRef(onChanged);
  changedRef.current = onChanged;
  const changed = useCallback(() => {
    refreshTranscript(client);
    changedRef.current();
  }, [client]);

  // Another day: a toast, and Who is this?, were about the last one.
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs when the scope changes, by design
  useEffect(() => {
    setToast(null);
    setNot(null);
    setRetry(null);
    undoRef.current = null;
  }, [scope]);

  // Focus goes back to what opened Who is this? once the page has drawn the change: a turn's buttons are disabled
  // while a correction saves, and focusing a disabled button does nothing. A timer raced React's render under load, so
  // focus was sometimes lost for good (the flaky Why this name test in Review.test).
  const [returnTo, setReturnTo] = useState<HTMLElement | null>(null);
  const restoreFocus = () => {
    const el = opener.current;
    opener.current = null;
    if (el) setReturnTo(el);
  };
  const closeWho = () => {
    setNot(null);
    if (!working.current) restoreFocus();
  };
  const fix = async (action: 'move' | 'remove', to?: string, previous?: RetryCorrection) => {
    const c = previous?.target ?? not;
    if (!c || working.current) return;
    working.current = true;
    closeWho();
    setSaving(true);
    setRetry(null);
    setToast({ text: 'Saving', sticky: true });
    if (!previous) undoRef.current = { kind: 'people', tokens: [] };
    const tokens = undoRef.current?.kind === 'people' ? undoRef.current.tokens : [];
    const failed: Not['lines'] = [];
    const errors: string[] = [];
    let done = previous?.changed ?? 0;
    const total = previous?.total ?? c.lines.length;
    for (const line of c.lines) {
      try {
        const j = await correct(
          action === 'move'
            ? { name: line.name, kind: 'voice', id: line.key, action, to: to ?? '' }
            : { name: line.name, kind: 'voice', id: line.key, action },
        );
        if (!j.ok) throw new Error(j.error || "That didn't work.");
        done++;
        if (j.undo) tokens.push(j.undo);
      } catch (e) {
        failed.push(line);
        errors.push(e instanceof HttpError && e.reason ? e.reason : messageOf(e));
      }
    }
    undoRef.current = tokens.length ? { kind: 'people', tokens } : null;
    const undo = tokens.length ? 'current' : undefined;
    if (failed.length) {
      setRetry({ target: { ...c, lines: failed }, action, to, total, changed: done });
      setToast({ text: `Changed ${done} of ${total} lines. ${[...new Set(errors)].join(' ')}`, undo, sticky: true });
    } else {
      setToast({
        text:
          action === 'move'
            ? total === 1
              ? `Moved to ${to}`
              : `Moved ${total} lines to ${to}`
            : total === 1
              ? `Removed from ${c.name}`
              : `Removed ${total} lines from ${c.name}`,
        undo,
      });
    }
    changed();
    setSaving(false);
    working.current = false;
    restoreFocus();
  };
  const trash = useCallback(
    async (lines: readonly TranscriptEntry[]) => {
      if (working.current) return;
      working.current = true;
      setSaving(true);
      setRetry(null);
      undoRef.current = null;
      setToast({ text: 'Saving correction', sticky: true });
      const result = await rejectLines(lines.map(e => e.line ?? e.ts_start));
      if (result.ok && typeof result.undo === 'string') {
        undoRef.current = { kind: 'transcript', token: result.undo };
        setToast({ text: 'Marked as not transcribed correctly', undo: 'current' });
        changed();
      } else setToast({ err: result.error || 'Could not correct the transcript' });
      setSaving(false);
      working.current = false;
    },
    [changed],
  );
  const tv = useCallback(
    async (lines: readonly TranscriptEntry[]) => {
      if (working.current) return;
      working.current = true;
      setSaving(true);
      setRetry(null);
      undoRef.current = null;
      setToast({ text: 'Marking as TV', sticky: true });
      const result = await markTv(lines.map(e => e.line ?? e.ts_start));
      if (result.ok && typeof result.undo === 'string') {
        undoRef.current = { kind: 'transcript', token: result.undo };
        setToast({ text: 'Marked as TV: left out of people and the journal', undo: 'current' });
        changed();
      } else setToast({ err: result.error || 'Could not mark it as TV' });
      setSaving(false);
      working.current = false;
    },
    [changed],
  );
  const offTv = useCallback(
    async (lines: readonly TranscriptEntry[]) => {
      if (working.current) return;
      working.current = true;
      setSaving(true);
      setRetry(null);
      undoRef.current = null;
      setToast({ text: 'Taking the TV mark off', sticky: true });
      const result = await notTv(lines.map(e => e.line ?? e.ts_start));
      if (result.ok) {
        // An Undo marks the lines a TV's again (#2643): a misclick could otherwise put a TV back into people.
        if (typeof result.undo === 'string') undoRef.current = { kind: 'transcript', token: result.undo };
        setToast({
          text: 'Not TV: it counts as said here again',
          undo: typeof result.undo === 'string' ? 'current' : undefined,
        });
        changed();
      } else setToast({ err: result.error || 'Could not take the TV mark off' });
      setSaving(false);
      working.current = false;
    },
    [changed],
  );
  const undo = async () => {
    const current = undoRef.current;
    if (!current || working.current) return;
    working.current = true;
    setSaving(true);
    setRetry(null);
    if (current.kind === 'transcript') {
      const result = await undoRejected(current.token);
      if (result.ok) {
        undoRef.current = null;
        setToast({ text: 'Undone' });
      } else setToast({ text: result.error || 'Undo failed. Try again.', undo: 'current', sticky: true });
    } else {
      const failed: string[] = [];
      const errors: string[] = [];
      for (const token of [...current.tokens].reverse()) {
        try {
          const j = await undoChange(token);
          if (!j.ok) throw new Error(j.errors?.join(' ') || 'Undo failed.');
        } catch (e) {
          failed.push(token);
          errors.push(messageOf(e));
        }
      }
      undoRef.current = failed.length ? { kind: 'people', tokens: failed.reverse() } : null;
      setToast(
        failed.length
          ? {
              text: `Undo failed for ${failed.length} of ${current.tokens.length} lines. ${errors.join(' ')}`,
              undo: 'current',
              sticky: true,
            }
          : { text: 'Undone' },
      );
    }
    changed();
    setSaving(false);
    working.current = false;
  };

  useEffect(() => {
    if (!returnTo || saving) return; // still saving: its buttons are disabled until the next commit
    if (returnTo.isConnected) returnTo.focus();
    setReturnTo(null);
  }, [returnTo, saving]);
  const savingNow = useRef(saving);
  savingNow.current = saving;
  const openNot = useCallback((n: Not, from: HTMLElement | null) => {
    if (savingNow.current) return; // one change at a time, so no Undo is lost
    opener.current = from;
    setNot(n);
  }, []);

  const overlay = (
    <>
      <FloatingToast
        text={toast && ('err' in toast ? toast.err : toast.text)}
        error={!!toast && 'err' in toast}
        sticky={!!toast && 'sticky' in toast && !!toast.sticky}
        onClose={() => {
          setToast(null);
          setRetry(null);
          undoRef.current = null;
        }}
        action={
          <>
            {retry && (
              <Button
                size="compact-xs"
                variant="subtle"
                disabled={saving}
                onClick={() => void fix(retry.action, retry.to, retry)}
              >
                Retry failed lines
              </Button>
            )}
            {toast && 'text' in toast && toast.undo && (
              <Button size="compact-xs" variant="subtle" disabled={saving} onClick={() => void undo()}>
                Undo
              </Button>
            )}
          </>
        }
      />
      <WhoIsThis
        name={not?.name ?? ''}
        label={not ? (not.lines.length === 1 ? 'this line' : 'this turn') : null}
        people={people}
        onClose={closeWho}
        onMove={to => void fix('move', to)}
        onRemove={() => void fix('remove')}
      />
    </>
  );
  return {
    saving,
    openNot,
    trash,
    tv,
    notTv: offTv,
    overlay,
  };
}
