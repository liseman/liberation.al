// Requests to the local servers. A failed request throws an HttpError carrying the server's own message ({"error"} or
// {"detail"}), which pages show as is, and the parsed body. A GET gives up after TIMEOUT_MS; a POST has no deadline.
// A GET that gets no answer at all (the dashboard restarting) waits for the server to come back, up to RECOVER_MS.

import { requestEnded, requestStarted } from './telemetry';

export const TIMEOUT_MS = 20_000;
// How long a GET waits out a server that dropped the connection, polling /health every BACKOFF_MS step (the last
// step repeats). A deploy restarts the dashboard in a few seconds; this covers a slow start with room to spare.
export const RECOVER_MS = 30_000;
const BACKOFF_MS = [250, 500, 1000, 2000];

export class HttpError extends Error {
  readonly status: number;
  readonly reason: string | null; // the server's own message, when it gave one
  readonly data: unknown; // the parsed JSON body, when there was one (a refusal can still carry fresh state)
  constructor(reason: string | null, status: number, data: unknown = null) {
    super(reason ?? `Request failed (${status})`);
    this.name = 'HttpError';
    this.reason = reason;
    this.status = status;
    this.data = data;
  }
}

function serverMessage(data: unknown): string | null {
  if (!data || typeof data !== 'object') return null;
  for (const field of ['error', 'detail'] as const)
    if (field in data) {
      const value = (data as Record<string, unknown>)[field];
      if (typeof value === 'string' && value) return value;
    }
  return null;
}

const isBinary = (body: unknown): body is Blob | ArrayBuffer | ArrayBufferView =>
  body instanceof Blob || body instanceof ArrayBuffer || ArrayBuffer.isView(body);

function bodyInit(body: unknown): Pick<RequestInit, 'method' | 'headers' | 'body'> {
  if (body === undefined) return {};
  if (isBinary(body)) {
    const type = body instanceof Blob && body.type ? body.type : 'application/octet-stream';
    return { method: 'POST', headers: { 'Content-Type': type }, body: body as BodyInit };
  }
  return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
}

/** A GET that got no answer within TIMEOUT_MS. The server may still be working on it. */
export class RequestTimeout extends Error {
  constructor() {
    super('The server took too long to answer');
    this.name = 'RequestTimeout';
  }
}

// HTTP/1.1 browsers have six connections per origin. Keep JSON reads within two
// slots so the four long-lived streams retain theirs. At most one background poll
// can run, leaving page data a slot even when status endpoints are slow.
const READ_SLOTS = 2;
export type ReadPriority = 'page' | 'background';
let activeReads = 0;
let activePolls = 0;
const waitingReads: { priority: ReadPriority; start: () => void }[] = [];

function drainReads(): void {
  while (activeReads < READ_SLOTS) {
    let index = waitingReads.findIndex(read => read.priority === 'page');
    if (index < 0 && activePolls === 0) index = waitingReads.findIndex(read => read.priority === 'background');
    if (index < 0) return;
    waitingReads.splice(index, 1)[0]?.start();
  }
}

function readSlot(signal: AbortSignal, priority: ReadPriority): Promise<() => void> {
  return new Promise((resolve, reject) => {
    const cancel = () => {
      const index = waitingReads.indexOf(waiting);
      if (index >= 0) waitingReads.splice(index, 1);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const start = () => {
      signal.removeEventListener('abort', cancel);
      activeReads++;
      if (priority === 'background') activePolls++;
      resolve(() => {
        activeReads--;
        if (priority === 'background') activePolls--;
        drainReads();
      });
    };
    const waiting = { priority, start };
    if (signal.aborted) return cancel();
    waitingReads.push(waiting);
    signal.addEventListener('abort', cancel, { once: true });
    drainReads();
  });
}

/**
 * True for a request that got no response at all: fetch rejects with a TypeError when the connection is refused or
 * dropped (Firefox words it "NetworkError when attempting to fetch resource."). An HTTP error, a timeout, or an abort
 * is something else.
 */
export const isNetworkFailure = (e: unknown): boolean => e instanceof TypeError;

/**
 * True for a refusal that will clear by itself shortly: the server's 503 with `starting` (its data broker restarting,
 * or a first read still loading).
 */
export const isStarting = (e: unknown): boolean =>
  e instanceof HttpError &&
  e.status === 503 &&
  typeof e.data === 'object' &&
  e.data !== null &&
  (e.data as { starting?: unknown }).starting === true;

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const done = () => {
      signal?.removeEventListener('abort', cancel);
      resolve();
    };
    const cancel = () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const timer = setTimeout(done, ms);
    if (signal?.aborted) return cancel();
    signal?.addEventListener('abort', cancel, { once: true });
  });
}

/** Poll the server's /health until it answers (true), or until `deadline` passes (false). */
async function backUp(url: string, deadline: number, signal?: AbortSignal): Promise<boolean> {
  const health = new URL('/health', new URL(url, globalThis.location?.href)).href;
  for (let step = 0; Date.now() < deadline; step++) {
    await wait(Math.min(BACKOFF_MS[Math.min(step, BACKOFF_MS.length - 1)] ?? 0, deadline - Date.now()), signal);
    try {
      const r = await fetch(health, { cache: 'no-store', signal });
      if (r.ok) return true;
    } catch (e) {
      if (!isNetworkFailure(e)) throw e;
    }
  }
  return false;
}

/**
 * GET url, or POST body when there is one, and return the parsed JSON response (see send). A GET that gets no
 * response at all waits until /health answers again and asks once more, for up to RECOVER_MS, so a dashboard restart
 * shows as a slow load (with the page's last data still up) rather than an error. A GET answered "starting" (503,
 * isStarting) is asked again after each BACKOFF_MS step, within the same RECOVER_MS. A POST is never sent twice,
 * since it may have reached the server. `recover` false skips the wait, for a caller that polls /health itself.
 */
export async function request<T>(
  url: string,
  body?: unknown,
  signal?: AbortSignal,
  priority: ReadPriority = 'page',
  recover = true,
): Promise<T> {
  // Every request is timed for the page's log: a failed or slow one is reported, and each joins the trace of
  // the navigation that waited on it. Only its path is kept.
  const method = body === undefined ? 'GET' : 'POST';
  const started = performance.now();
  requestStarted();
  try {
    const data = await retried<T>(url, body, signal, priority, recover);
    requestEnded(url, method, 200, performance.now() - started);
    return data;
  } catch (e) {
    const aborted = signal?.aborted === true || (e instanceof DOMException && e.name === 'AbortError');
    requestEnded(url, method, e instanceof HttpError ? e.status : null, performance.now() - started, aborted);
    throw e;
  }
}

/** A file POSTed as its request's body, as request() sends one, with the upload's progress, which fetch can't give:
 *  `onProgress` gets the share sent so far, 0 to 1 (Ingest). A refusal is an HttpError with the server's
 *  message, as request()'s is. */
export function upload<T>(url: string, file: Blob, onProgress?: (share: number) => void): Promise<T> {
  const started = performance.now();
  requestStarted();
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', file.type || 'application/octet-stream');
    xhr.upload.onprogress = e => {
      if (e.lengthComputable && e.total) onProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      let data: unknown = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        // not JSON: the status code says what happened
      }
      const ok = xhr.status >= 200 && xhr.status < 300;
      requestEnded(url, 'POST', ok ? 200 : xhr.status, performance.now() - started);
      if (ok) resolve(data as T);
      else reject(new HttpError(serverMessage(data), xhr.status, data));
    };
    xhr.onerror = () => {
      requestEnded(url, 'POST', null, performance.now() - started);
      reject(new TypeError('Failed to fetch'));
    };
    xhr.send(file);
  });
}

async function retried<T>(
  url: string,
  body: unknown,
  signal: AbortSignal | undefined,
  priority: ReadPriority,
  recover: boolean,
): Promise<T> {
  const deadline = Date.now() + RECOVER_MS;
  for (let step = 0; ; step++) {
    try {
      return await send<T>(url, body, signal, priority);
    } catch (e) {
      if (!recover || body !== undefined || signal?.aborted) throw e;
      if (isStarting(e)) {
        const pause = BACKOFF_MS[Math.min(step, BACKOFF_MS.length - 1)] ?? 0;
        if (Date.now() + pause >= deadline) throw e;
        await wait(pause, signal);
        continue;
      }
      if (!isNetworkFailure(e)) throw e;
      if (!(await backUp(url, deadline, signal))) throw e;
    }
  }
}

/**
 * Send one request: GET url, or POST body when there is one, and return the parsed JSON response. A Blob (for example a WAV
 * recording) or other binary body is sent as is with its own type; anything else is sent as JSON. signal cancels
 * it, and a GET also gives up after TIMEOUT_MS.
 */
async function send<T>(
  url: string,
  body: unknown,
  signal: AbortSignal | undefined,
  priority: ReadPriority,
): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const cancel = () => controller.abort();
  if (signal?.aborted) controller.abort();
  signal?.addEventListener('abort', cancel);
  let release: (() => void) | undefined;
  try {
    if (body === undefined) {
      release = await readSlot(controller.signal, priority);
      // Only reads get a deadline, as getJSON had; a POST can wait on the server's own work, like a service restart.
      // The deadline is the server's: it starts once this read has a slot, so a page that asks for many at once
      // doesn't time out the ones queued behind a few slow answers.
      timer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, TIMEOUT_MS);
    }
    const response = await fetch(url, { cache: 'no-store', signal: controller.signal, ...bodyInit(body) });
    let data: unknown = null;
    try {
      data = await response.json(); // the deadline holds while the body arrives
    } catch (e) {
      if (controller.signal.aborted) throw e;
      // not JSON: fall through to the status code
    }
    if (!response.ok) throw new HttpError(serverMessage(data), response.status, data);
    return data as T;
  } catch (e) {
    if (timedOut) throw new RequestTimeout();
    throw e;
  } finally {
    release?.();
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}

/** An error's message, for showing to the viewer. */
export const messageOf = (e: unknown): string => (e instanceof Error ? e.message : String(e));
