// What the Live tab asks the dashboard server for: known people, naming voices, faces, and lines, and starting a
// service.
import type { QueryClient } from '@tanstack/react-query';
import { HttpError, messageOf, request } from '../../shared/http';
import { liveKey } from '../live/feed';
import { reduce } from '../live/reduce';
import { type LineRef, oneLine, someLines, type Transcript } from '../live/types';
import { PEOPLE_KEY } from '../people/api';
import { SERVICES_KEY, type ServicesReply } from '../settings/services/model';
import { dropWaiting, UNKNOWN_GROUPS_KEY, UNKNOWN_VOICES_KEY } from '../whosthis/api';

export interface Reply {
  ok: boolean;
  error?: string | null;
  note?: string | null;
  name?: string;
  [field: string]: unknown;
}

/** POST and give back the server's reply, with a failure turned into {ok: false, error}. */
async function post(url: string, body: unknown, fallback: string): Promise<Reply> {
  try {
    return await request<Reply>(url, body);
  } catch (e) {
    const reason = e instanceof HttpError ? e.reason : messageOf(e);
    return { ok: false, error: reason || fallback };
  }
}

// Who's this? lives in ../whosthis; its queries are re-exported for the callers that import them here.
export {
  type Band,
  type Guess,
  ignoreFace,
  UNKNOWN_GROUPS_KEY,
  UNKNOWN_VOICES_KEY,
  type UnknownGroup,
  type UnknownVoice,
  unknownGroupsQuery,
  unknownVoicesQuery,
} from '../whosthis/api';

interface Refresh {
  all: boolean;
  timer?: ReturnType<typeof setTimeout>;
  done: Promise<void>;
  finish: () => void;
}
const pending = new WeakMap<QueryClient, Map<number, Refresh>>();
const generations = new WeakMap<QueryClient, number>();
/** Every refresh still waiting on its timer, and every one reading, for cancelQueuedRefreshes. */
const queued = new Set<Refresh>();
const reading = new Set<AbortController>();

/**
 * Drop every refresh still waiting on its timer (their promises resolve, so nothing waits forever). The test teardown
 * calls this: a correction's refresh fires 50 ms (or a tag's 4 s) later from a timer of its own, and under load it
 * could outlive its test and read /transcript after that test's mocks were gone (WhosThis.test's overnight failure).
 */
export function cancelQueuedRefreshes(): void {
  for (const batch of queued) {
    clearTimeout(batch.timer);
    batch.finish();
  }
  queued.clear();
  // One already reading holds one of the page's two request slots until it answers (shared/http.ts): stop it too.
  for (const read of reading) read.abort();
  reading.clear();
}

/** Coalesce a burst per query client and delay. A later correction also supersedes an in-flight tail read. */
function queueRefresh(client: QueryClient, delay: number, all: boolean): Promise<void> {
  generations.set(client, (generations.get(client) ?? 0) + 1);
  const batches = pending.get(client) ?? new Map<number, Refresh>();
  pending.set(client, batches);
  let batch = batches.get(delay);
  if (!batch) {
    let finish = () => {};
    const done = new Promise<void>(resolve => {
      finish = resolve;
    });
    batch = { all, done, finish };
    batches.set(delay, batch);
  }
  batch.all ||= all;
  if (batch.timer !== undefined) clearTimeout(batch.timer);
  const work = batch;
  queued.add(work);
  work.timer = setTimeout(() => {
    queued.delete(work);
    batches.delete(delay);
    if (!batches.size) pending.delete(client);
    const generation = (generations.get(client) ?? 0) + 1;
    generations.set(client, generation);
    const reads: Promise<unknown>[] = work.all
      ? [client.invalidateQueries()]
      : [liveKey('speakers'), PEOPLE_KEY, UNKNOWN_GROUPS_KEY, UNKNOWN_VOICES_KEY].map(queryKey =>
          client.invalidateQueries({ queryKey }),
        );
    const stop = new AbortController();
    reading.add(stop);
    reads.push(
      request('/transcript?n=300', undefined, stop.signal)
        .then(d => {
          if (generations.get(client) !== generation) return;
          client.setQueryData<Transcript>(liveKey('transcript'), prev => reduce('transcript', prev, d) as Transcript);
        })
        .finally(() => reading.delete(stop)),
    );
    void Promise.allSettled(reads).then(work.finish);
  }, delay);
  return work.done;
}

/** Refresh names and their dependent views once after a burst; delayed tag followups also share one timer. */
export function refreshNames(client: QueryClient, delay = 50): Promise<void> {
  return queueRefresh(client, delay, false);
}

/** A voice or face belongs to a person (one who exists already is merged into). */
export async function claim(client: QueryClient, kind: 'voice' | 'face', id: number | string, name: string) {
  const n = Number.parseInt(String(id), 10);
  const r = await post('/people/claim', { kind, id: n, name }, 'unknown error');
  if (r.ok) dropWaiting(client, kind, n);
  void refreshNames(client);
  return r;
}

/** These lines were said by name: the microphone service relabels them and learns the voice. */
export const tagLines = (lines: readonly LineRef[], name: string) =>
  post('/speakers/tag', { ...someLines(lines), name }, 'unknown error');

/** Rename a voice. */
export const labelVoice = (id: number | string, name: string) =>
  post('/speakers/label', { id: Number.parseInt(String(id), 10), name }, 'unknown error');

/** This line isn't the voice it was put down to. */
export const unlabelLine = (line: LineRef) => post('/transcript/unlabel', oneLine(line), 'Could not remove the label');

/** Mark this visible turn as incorrectly transcribed, preserving raw audio and one batch Undo. */
export const rejectLines = (lines: readonly LineRef[]) =>
  post('/transcript/reject', someLines(lines), 'Could not correct the transcript');

/** "This is TV": the lines stay, marked as a TV's, and leave people, voices, the journal, and Ask. Undone with
 *  undoRejected, as a correction. */
export const markTv = (lines: readonly LineRef[]) =>
  post('/transcript/media', someLines(lines), 'Could not mark it as TV');

/** "Not TV": take back these lines' TV marks, the owner's or the audio tagger's. */
export const notTv = (lines: readonly LineRef[]) =>
  post('/transcript/not-media', someLines(lines), 'Could not take the TV mark off');

/** "TV or radio" on an unnamed voice: every line it has, marked as above. */
export const markTvVoice = (group: number) => post('/voices/unknown/media', { group }, 'Could not mark it as TV');

export const undoRejected = (undo: string) =>
  post('/transcript/undo', { undo }, 'Could not undo the transcript correction');

/** A reviewed transcript changes every view derived from it, including saved answers and voice references. */
export function refreshTranscript(client: QueryClient): Promise<void> {
  return queueRefresh(client, 50, true);
}

/** Turn a service on; gives back an error, or null when it worked. */
export async function startService(client: QueryClient, name: string): Promise<string | null> {
  const r = (await post('/services', { name, action: 'on' }, `Turning ${name} on failed`)) as ServicesReply;
  if (r.services) client.setQueryData(SERVICES_KEY, r);
  return r.ok ? null : r.error || `Turning ${name} on failed`;
}
