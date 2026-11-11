// An Ask question is a server job: POST it, then poll /pack-ask/jobs/<id> until it is done. The job in progress and
// the draft question are saved in sessionStorage as {active, question}, so a reload picks the job back up.
import { useEffect, useRef, useState } from 'react';
import { HttpError, messageOf, request } from '../../shared/http';
import { dateOf, isRec, plain, reason } from './format';
import { jobURL } from './links';

// The job in progress and the draft, saved per browser tab. One Ask, one key; a job saved under an older key (a
// scope or a Review day of its own) is not resumed.
export const KEY = 'pack-ask.job.v3';

export interface ActiveJob {
  id: string;
  poll_url: string;
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

/** What a fresh Ask starts with: the saved draft, and the saved job when there is one. */
export function restore(): Restored {
  const out: Restored = { question: '', active: null, status: { text: '' } };
  try {
    const saved = read(KEY);
    if (!isRec(saved)) return out;
    if (typeof saved.question === 'string') out.question = saved.question;
    const job = saved.active;
    const url = isRec(job) ? jobURL(job.poll_url) : null;
    if (!isRec(job) || !url) return out;
    // The saved record is kept whole (spread), so saving it again writes back what was read.
    out.active = { ...job, id: plain(job.id), poll_url: url } as ActiveJob;
    out.status = { text: `Ask · recovering job ${plain(job.id)}` };
  } catch {
    // Ignore invalid or unavailable storage.
  }
  return out;
}

const STATUSES = new Set(['queued', 'running', 'completed', 'failed']);

/** Ask's state: the draft, the job in progress, and the latest result. */
export function useAskJob() {
  const [initial] = useState(restore);
  const [question, setQuestion] = useState(initial.question);
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
  // The answer as the model writes it, until the finished result replaces it.
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
    const label = 'Ask';
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    const writingNow = { current: false };
    const finish = () => {
      setPartial('');
      saveJob(KEY, null, draft.current);
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
          // A failed job whose result says why (the answer's own failure) needs no second, emptier line.
          if (j.status === 'failed' && (j.error || !j.result))
            setError(reason(j.error || 'Job failed without an error description.'));
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
  }, [active]);

  const lock = (value: boolean) => {
    busyNow.current = value;
    setBusy(value);
  };

  const change = (value: string) => {
    setQuestion(value);
    draft.current = value;
    saveJob(KEY, activeNow.current, value);
  };

  /** Ask the question in the box, quickly. True once the server has taken the job. `day` is the day hint (Review's
   *  day, or a followed-up report's), and `followup` a research report's id (the server reads the report). */
  const submit = async (extra: { followup?: { report: string }; day?: string } = {}): Promise<boolean> => {
    if (busyNow.current) return false;
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
    lock(true);
    setStatus({ text: 'Ask · submitting' });
    try {
      const j = await request<unknown>('/pack-ask/jobs', {
        question: q,
        ...(extra.day ? { day: extra.day } : {}),
        ...(extra.followup ? { followup: extra.followup } : {}),
      });
      const url = isRec(j) ? jobURL(j.poll_url) : null;
      if (!isRec(j) || !url || !j.id || (j.status !== 'queued' && j.status !== 'running'))
        throw new Error('Invalid job acknowledgement');
      const job: ActiveJob = { id: plain(j.id), poll_url: url, question: q, created_at: started.current };
      saveJob(KEY, job, draft.current);
      setStatus({ text: `Ask · ${j.status}` });
      setActive(job);
      return true;
    } catch (e) {
      setStatus({ text: 'Ask · submission not confirmed' });
      setError(
        `Could not confirm submission (${failure(e)}). The server may have accepted it. No automatic resubmit; retrying may create another job.`,
      );
      lock(false);
      return false;
    }
  };

  /** Forget the answer shown, as when the question is sent to deep research instead. */
  const clear = () => {
    setResult(null);
    setAsked(null);
    setError('');
    setStatus({ text: '' });
    shown.current = null;
  };

  // pending: the question being answered now, as it was asked (the box may hold the next one)
  const pending = active?.question ?? null;
  return {
    question,
    change,
    busy,
    status,
    elapsed,
    error,
    result,
    partial,
    asked,
    pending,
    submit,
    clear,
    input,
  };
}
