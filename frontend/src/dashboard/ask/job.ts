// An Ask question is a server job: POST it, then poll /pack-ask/jobs/<id> until it is done. The job in progress and
// the draft question are saved in sessionStorage, so a reload (or switching between the old and new pages) picks the
// job back up. The keys and the saved shape, {active, question}, are the old widget's (static/lib/ask.js).
import { useEffect, useRef, useState } from 'react';
import { HttpError, messageOf, request } from '../../shared/http';
import { dateOf, isRec, plain, reason } from './format';
import { jobURL } from './links';

/** What each scope does, in one sentence, shown under Live's scope choice. */
export const MODES = {
  local: 'Searches your recordings on this machine. Nothing leaves it.',
  friends: 'Asks a few chosen models. They see your question, not your recordings.',
  everyone: 'Asks every model that is set up. They see your question, not your recordings.',
} as const;

export type Mode = keyof typeof MODES;

/** Each scope's name on Live's segmented choice and in its progress line. */
export const MODE_NAMES: Record<Mode, string> = { local: 'My recordings', friends: 'Friends', everyone: 'Everyone' };
export type Scope = 'history' | 'general';

/** Local searches the recordings (a Review day, or on Live the window the question names); friends and everyone
 * are general questions, so no recorded evidence is sent off this machine (#2307). */
export const scopeOf = (day: string | null, mode: Mode): Scope => (day || mode === 'local' ? 'history' : 'general');

export const isMode = (value: unknown): value is Mode => typeof value === 'string' && Object.hasOwn(MODES, value);

// Review (history scope) keeps the v1 key so a recorded-evidence job survives a reload. General Asks use v2 keys, so
// an old evidence job saved under v1 can never be resumed (or polled) by a general question; only its draft is kept.
const KEY = 'pack-ask.job.v1';
const GENERAL_KEY = 'pack-ask.job.v2.general';

export const storageKey = (day: string | null, showModes: boolean): string =>
  day ? `${KEY}.review.${day}` : showModes ? GENERAL_KEY : `${GENERAL_KEY}.ask`;

// Where a pre-scope build saved its Live / single Ask draft; only the question text is migrated from these.
const legacyKey = (showModes: boolean): string => (showModes ? KEY : `${KEY}.ask`);

export interface ActiveJob {
  id: string;
  poll_url: string;
  mode: Mode;
  scope?: Scope; // missing on a job saved before scopes existed
  question: string;
  created_at: number;
}

export interface Status {
  text: string;
  finished?: unknown; // the job's finished_at, once it is done
}

/** The question behind a result, as the server took it: what it said, when it was answered, and the Review day. */
export interface Asked {
  question: string;
  at: number; // epoch seconds: when the job started, the "now" its answer read "today" and "the last hour" from
  day: string | null;
}

function askedOf(j: Record<string, unknown>): Asked | null {
  // A job can wait in the queue behind another; its answer is read from when it started, so a label is dated then.
  const at = typeof j.started_at === 'number' ? j.started_at : j.created_at;
  if (typeof j.question !== 'string' || !j.question.trim() || typeof at !== 'number' || !Number.isFinite(at))
    return null;
  return { question: j.question, at, day: typeof j.day === 'string' ? j.day : null };
}

export const modeLabel = (mode: Mode, showModes: boolean): string => (showModes ? MODE_NAMES[mode] : 'Ask');
// An Ask says nothing at rest: its status line is for progress while an answer is coming (#2452).
const readyText = (_showModes: boolean) => '';

/** An error's text for the viewer: the server's reason when it gave one, else the old widget's "HTTP <status>". */
export function failure(e: unknown): string {
  return e instanceof HttpError ? (e.reason ?? `HTTP ${e.status}`) : messageOf(e);
}

export function saveJob(key: string, active: ActiveJob | null, question: string): void {
  try {
    sessionStorage.setItem(key, JSON.stringify({ active, question }));
  } catch {
    // Private mode or storage quotas must not break a job.
  }
}

const read = (key: string): unknown => JSON.parse(sessionStorage.getItem(key) || 'null');

export interface Restored {
  question: string;
  active: ActiveJob | null;
  status: Status;
}

/** What a fresh widget starts with: the saved draft, and the saved job when it belongs to this widget. */
export function restore(day: string | null, showModes: boolean): Restored {
  const out: Restored = { question: '', active: null, status: { text: readyText(showModes) } };
  try {
    let saved = read(storageKey(day, showModes));
    if (!saved && !day) {
      // First visit after the scope change: keep the draft question, never the old job (it may be an evidence job).
      const old = read(legacyKey(showModes));
      if (isRec(old) && typeof old.question === 'string' && old.question) {
        saved = { active: null, question: old.question };
        out.status = {
          text: 'Previous draft kept; nothing was sent',
        };
      }
    }
    if (!isRec(saved)) return out;
    if (typeof saved.question === 'string') out.question = saved.question;
    const job = saved.active;
    if (!isRec(job)) return out;
    // A saved job resumes only inside its own scope. Review also accepts a job saved before scopes existed; a general
    // Ask never does.
    const url = jobURL(job.poll_url);
    if (!url || !isMode(job.mode)) return out;
    if ((job.scope ?? (day ? 'history' : null)) !== scopeOf(day, job.mode)) {
      out.status = { text: 'A previous question of a different kind was not resumed. Your question is preserved.' };
      return out;
    }
    if (!showModes && job.mode !== 'local') {
      out.status = {
        text: 'Previous multi-model question not resumed. Your question is preserved; mode choices are on Live.',
      };
      return out;
    }
    // The saved record is kept whole (spread), so saving it again writes back what was read.
    out.active = { ...job, id: plain(job.id), poll_url: url, mode: job.mode } as ActiveJob;
    out.status = { text: `${modeLabel(job.mode, showModes)} · recovering job ${plain(job.id)}` };
  } catch {
    // Ignore invalid or unavailable storage.
  }
  return out;
}

const STATUSES = new Set(['queued', 'running', 'completed', 'failed']);

/** The Ask widget's state: the draft, the chosen mode, the job in progress, and the latest result. */
export function useAskJob(day: string | null, showModes: boolean) {
  const key = storageKey(day, showModes);
  const [initial] = useState(() => restore(day, showModes));
  const [question, setQuestion] = useState(initial.question);
  const [mode, setMode] = useState<Mode>(initial.active?.mode ?? 'local');
  const [active, setActiveState] = useState(initial.active);
  // the job as it is now, for saving the draft: change() can run before a re-render shows a job just taken
  const activeNow = useRef(initial.active);
  const setActive = (job: ActiveJob | null) => {
    activeNow.current = job;
    setActiveState(job);
  };
  const [busy, setBusy] = useState(initial.active !== null);
  const [status, setStatus] = useState<Status>(initial.status);
  const [elapsed, setElapsed] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState<unknown>(null);
  // The answer as the model writes it (#1336), until the finished result replaces it.
  const [partial, setPartial] = useState('');
  const [asked, setAsked] = useState<Asked | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  // The duplicate-submit guard must not wait for a re-render; the others are read by the poll loop.
  const busyNow = useRef(busy);
  const draft = useRef(initial.question);
  const started = useRef(initial.active ? Number(initial.active.created_at) || Date.now() : 0);
  const shown = useRef<string | null>(null);

  useEffect(() => {
    const update = () =>
      setElapsed(started.current ? `${Math.max(0, Math.floor((Date.now() - started.current) / 1000))}s elapsed` : '');
    update();
    if (!busy) return;
    const tick = setInterval(update, 1000);
    return () => clearInterval(tick);
  }, [busy]);

  // Poll the job in progress: three quick reads a second apart, then every 3 s, and every 3 s after a failed read.
  useEffect(() => {
    if (!active) return;
    const label = modeLabel(active.mode, showModes);
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    const writingNow = { current: false };
    const finish = () => {
      setPartial('');
      saveJob(key, null, draft.current);
      busyNow.current = false;
      setBusy(false);
      activeNow.current = null;
      setActiveState(null);
    };
    const poll = async () => {
      try {
        // recover false: this poll has its own retry and says so, so a dropped connection isn't waited out twice.
        const j = await request<unknown>(active.poll_url, undefined, controller.signal, 'page', false);
        if (controller.signal.aborted) return;
        if (!isRec(j) || typeof j.status !== 'string' || !STATUSES.has(j.status))
          throw new Error('Invalid job status from server');
        setError('');
        const when = dateOf(j.created_at);
        if (when) started.current = when.getTime();
        const done = j.status === 'completed' || j.status === 'failed';
        setStatus({
          text: `${label} · ${j.status}${j.result && !done ? ' · partial result' : ''}`,
          finished: done ? j.finished_at : null,
        });
        if (j.result) {
          const signature = JSON.stringify(j.result);
          if (signature !== shown.current) {
            shown.current = signature;
            setResult(j.result);
            setAsked(askedOf(j));
          }
        }
        const writing = !done && typeof j.partial === 'string' && j.partial ? j.partial : '';
        writingNow.current = writing !== '';
        setPartial(writing);
        if (done) {
          if (j.status === 'failed') setError(reason(j.error || 'Job failed without an error description.'));
          if (j.status === 'completed' && !j.result) setError('Job completed without a result.');
          finish();
          return;
        }
      } catch (e) {
        if (controller.signal.aborted) return;
        if (e instanceof HttpError && e.status === 404) {
          if (!draft.current.trim()) {
            // the box was emptied when it was asked (Review's Ask): put the question back, as the message says
            draft.current = active.question;
            setQuestion(active.question);
          }
          finish();
          setStatus({ text: `${label} · interrupted or expired` });
          setError(
            'This question was lost during a server restart or expired. Your question is preserved; submit it again.',
          );
          return;
        }
        attempts = 3;
        setError(
          `Cannot read job status (${failure(e)}). Still tracking this job; retrying in 3 seconds. Job ID: ${active.id}`,
        );
      }
      attempts += 1;
      // Every second while the answer is being written, so it reads as it grows.
      timer = setTimeout(poll, attempts < 3 || writingNow.current ? 1000 : 3000);
    };
    timer = setTimeout(poll, 1000);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [active, key, showModes]);

  const lock = (value: boolean, m: Mode) => {
    busyNow.current = value;
    setBusy(value);
    setMode(m);
  };

  const change = (value: string) => {
    setQuestion(value);
    draft.current = value;
    saveJob(key, activeNow.current, value);
  };

  /** Ask the question in the box. True once the server has taken the job. `extra` is a follow-up on a research
   *  report: its id (the server reads the report) and its day, when this Ask has no day of its own. */
  const submit = async (
    picked: Mode,
    extra: { followup?: { report: string }; day?: string } = {},
  ): Promise<boolean> => {
    if (busyNow.current) return false;
    const m: Mode = showModes ? picked : 'local';
    const scope = scopeOf(day, m);
    const q = draft.current.trim();
    if (!q) {
      setError('Please enter a question first.');
      input.current?.focus();
      return false;
    }
    setError('');
    setResult(null);
    setAsked(null);
    shown.current = null;
    started.current = Date.now();
    lock(true, m);
    setStatus({ text: `${modeLabel(m, showModes)} · submitting` });
    try {
      const asOf = day ?? extra.day;
      const j = await request<unknown>('/pack-ask/jobs', {
        question: q,
        mode: m,
        scope,
        ...(asOf ? { day: asOf } : {}),
        ...(extra.followup ? { followup: extra.followup } : {}),
      });
      const url = isRec(j) ? jobURL(j.poll_url) : null;
      if (!isRec(j) || !url || !j.id || (j.status !== 'queued' && j.status !== 'running'))
        throw new Error('Invalid job acknowledgement');
      const job: ActiveJob = {
        id: plain(j.id),
        poll_url: url,
        mode: m,
        scope,
        question: q,
        created_at: started.current,
      };
      saveJob(key, job, draft.current);
      setStatus({ text: `${modeLabel(m, showModes)} · ${j.status}` });
      setActive(job);
      return true;
    } catch (e) {
      setStatus({ text: `${modeLabel(m, showModes)} · submission not confirmed` });
      setError(
        `Could not confirm submission (${failure(e)}). The server may have accepted it. No automatic resubmit; retrying may create another job.`,
      );
      lock(false, m);
      return false;
    }
  };

  // pending: the question being answered now, as it was asked (the box may hold the next one)
  const pending = active?.question ?? null;
  return {
    scope: scopeOf(day, mode),
    question,
    change,
    mode,
    busy,
    status,
    elapsed,
    error,
    result,
    partial,
    asked,
    pending,
    submit,
    input,
  };
}
