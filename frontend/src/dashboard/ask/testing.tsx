// Test helpers for the Ask widget: a fake job server on MSW, and fake timers that tests advance to run a poll.
import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { PackAsk } from './PackAsk';

/** A poll reply: a job body, an HTTP failure, or a dropped connection. */
export type Reply = Record<string, unknown> | { httpStatus: number } | 'offline';

export function jobServer() {
  const posts: unknown[] = [];
  const polls: string[] = [];
  const acks: Reply[] = [];
  const replies: Reply[] = [];
  const answer = (r: Reply) =>
    r === 'offline'
      ? HttpResponse.error()
      : 'httpStatus' in r && typeof r.httpStatus === 'number'
        ? HttpResponse.json({}, { status: r.httpStatus })
        : HttpResponse.json(r);
  server.use(
    http.post('/pack-ask/jobs', async ({ request }) => {
      posts.push(await request.json());
      const n = posts.length;
      return answer(acks.shift() ?? { id: `j${n}`, status: 'queued', poll_url: `/pack-ask/jobs/j${n}` });
    }),
    http.get('/pack-ask/jobs/:id', ({ params }) => {
      polls.push(String(params.id));
      return answer(replies.shift() ?? { status: 'running' });
    }),
  );
  return { posts, polls, acks, replies };
}

/** Fake timers that also follow real time, so Testing Library's waits still run. */
export function fakeTimers() {
  vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  return userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
}

/** Let ms of fake time pass, running any poll that falls due. */
export const wait = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));

export { KEY } from './job';

export const saved = (key: string): unknown => JSON.parse(sessionStorage.getItem(key) ?? 'null');
export const store = (key: string, value: unknown) => sessionStorage.setItem(key, JSON.stringify(value));

export const question = () => screen.getByRole<HTMLTextAreaElement>('textbox');
export const status = () => screen.getByRole('status');
export const alertText = () => screen.getAllByRole('alert')[0]?.textContent ?? '';
export const buttons = () => screen.getAllByRole('button').filter(b => b.closest('[aria-busy]'));

export function renderAsk() {
  return renderPage(<PackAsk />);
}

/** Research jobs as the live feed's `research` event serves them: research can run, with its reach and timing. */
export function researchServer(over: Record<string, unknown> = {}) {
  const reach = (deep: boolean) => ({
    deep,
    rounds: deep ? 8 : 3,
    search_s: deep ? 1800 : 80,
    sources: ['journal', 'transcript', 'camera'],
    model: deep ? 'big-model' : 'small-model',
    day: null,
  });
  server.use(
    http.get('/research/jobs', () =>
      HttpResponse.json({
        jobs: [],
        available: true,
        reason: null,
        unread: 0,
        typical_s: 600,
        reach: { quick: reach(false), deep: reach(true) },
        ...over,
      }),
    ),
  );
}
