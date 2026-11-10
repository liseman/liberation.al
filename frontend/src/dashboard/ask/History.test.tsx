import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it, vi } from 'vitest';
import { Providers } from '../../shared/Providers';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { History } from './History';
import type { HistoryEntry } from './historyApi';
import { PackAsk } from './PackAsk';

/** The saved answer's line, whose "[C1]" is now a link to its moment: matched as the paragraph reads. */
const visitorLeft = (_: string, el: Element | null) =>
  el?.tagName === 'P' && el.textContent === 'The visitor left at C1.';
const citation = { label: 'C1', ts: 1200, camera: 'cam0', source: 'camera', removed: false, verified: true };
const entry: HistoryEntry = {
  id: 'saved-1',
  question: 'When did the visitor leave?',
  surface: 'review',
  day: '2026-10-27',
  scope: 'history',
  mode: 'local',
  status: 'completed',
  created_at: 5000,
  started_at: 5001,
  finished_at: 5002,
  model: 'fixture-model',
  result: {
    status: 'ok',
    scope: 'history',
    answer: 'The visitor left at [C1].',
    evidence: [{ id: 'C1', ts: 1200, camera: 'cam0', href: '/review#t=1200' }],
  },
  citations: [citation],
  answer_removed: false,
};
beforeEach(() => sessionStorage.clear());
function historyServer(value = entry) {
  const reads: URL[] = [];
  const mutations: { url: string; body: unknown; type: string | null }[] = [];
  let kept = [value];
  server.use(
    http.get('/pack-ask/history', ({ request }) => {
      reads.push(new URL(request.url));
      return HttpResponse.json({ items: kept, next_before: null });
    }),
    http.get('/pack-ask/history/:id', () => HttpResponse.json(value)),
    http.delete('/pack-ask/history/:id', async ({ request }) => {
      mutations.push({ url: request.url, body: await request.json(), type: request.headers.get('Content-Type') });
      kept = [];
      return HttpResponse.json({ deleted: 1 });
    }),
    http.delete('/pack-ask/history', async ({ request }) => {
      mutations.push({ url: request.url, body: await request.json(), type: request.headers.get('Content-Type') });
      kept = [];
      return HttpResponse.json({ deleted: 1 });
    }),
  );
  return { reads, mutations };
}
async function open() {
  await userEvent.click(screen.getByRole('button', { name: 'Past questions' }));
  return screen.findByRole('button', { name: entry.question });
}

it('fetches past answers after remount without posting or changing the new prompt', async () => {
  const net = historyServer();
  const first = renderPage(<PackAsk showModes />);
  await open();
  expect(net.reads[0]?.searchParams.has('surface')).toBe(false);
  first.unmount();
  renderPage(<PackAsk showModes />);
  const input = screen.getByRole('textbox', { name: 'What would you like to ask?' });
  await userEvent.type(input, 'A different question');
  await userEvent.click(await open());
  expect(await screen.findByText(visitorLeft)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'C1' })).toHaveAttribute('href', '/review#t=1200'); // its citation
  expect(input).toHaveValue('A different question');
  expect(net.reads).toHaveLength(2);
  // No POST handler: a model submission would fail strict network teardown.
});

it('filters Review by day and seeks a stored moment without resubmitting', async () => {
  const net = historyServer();
  const seekPlay = vi.fn();
  const onMarks = vi.fn();
  renderPage(
    <PackAsk day="2026-10-27" bare compact dayLabel="Tuesday" today={false} onSeek={seekPlay} onMarks={onMarks} />,
  );
  await userEvent.click(await open());
  const saved = await screen.findByRole('region', { name: 'Saved answer' });
  await userEvent.click(within(saved).getByRole('button', { name: /\[C1\]/ }));
  expect(seekPlay).toHaveBeenCalledExactlyOnceWith(1200);
  expect(onMarks).toHaveBeenLastCalledWith([1200]);
  expect(net.reads[0]?.searchParams.get('surface')).toBe('review');
  expect(net.reads[0]?.searchParams.get('day')).toBe('2026-10-27');
  expect(screen.getByRole('textbox', { name: 'Ask about Tuesday' })).toHaveValue('');
});

it.each(['one', 'all'])('confirms before deleting %s with a JSON DELETE', async kind => {
  const net = historyServer();
  renderPage(<History day="2026-10-27" />);
  await open();
  const name = kind === 'one' ? `Delete ${entry.question}` : 'Delete all past questions for this day';
  await userEvent.click(screen.getByRole('button', { name }));
  expect(net.mutations).toEqual([]);
  const confirmation = screen.getByRole('dialog', { name: 'Confirm deletion' });
  expect(confirmation).toHaveTextContent('This cannot be undone.');
  await userEvent.click(within(confirmation).getByRole('button', { name: 'Cancel' }));
  expect(net.mutations).toEqual([]);
  await userEvent.click(screen.getByRole('button', { name }));
  await userEvent.click(screen.getByRole('button', { name: 'Confirm deletion' }));
  await screen.findByText('No past questions yet.');
  expect(net.mutations).toHaveLength(1);
  expect(net.mutations[0]?.body).toEqual({});
  expect(net.mutations[0]?.type).toBe('application/json');
  const url = new URL(net.mutations[0]?.url ?? '');
  expect(url.pathname).toBe(kind === 'one' ? '/pack-ask/history/saved-1' : '/pack-ask/history');
  if (kind === 'all') expect(url.searchParams.get('day')).toBe('2026-10-27');
});

it('discards a pending deletion when its history unmounts', async () => {
  const net = historyServer();
  const view = renderPage(<History day="2026-10-27" />);
  await open();
  await userEvent.click(screen.getByRole('button', { name: `Delete ${entry.question}` }));
  expect(screen.getByRole('dialog', { name: 'Confirm deletion' })).toBeInTheDocument();
  view.unmount();
  renderPage(<History day="2026-10-28" />);
  expect(screen.queryByRole('dialog', { name: 'Confirm deletion' })).not.toBeInTheDocument();
  expect(net.mutations).toEqual([]);
});

it('keeps removed and unverified moments disabled and hides a purged answer', async () => {
  historyServer({
    ...entry,
    result: null,
    answer_removed: true,
    citations: [
      { ...citation, removed: true },
      { label: 'J2', ts: 1800, camera: null, source: 'journal', removed: false, verified: false },
    ],
  });
  const seek = vi.fn();
  renderPage(<History onSeek={seek} />);
  await userEvent.click(await open());
  expect(await screen.findByText(/answer was removed/)).toBeInTheDocument();
  const moments = screen.getByRole('list', { name: 'Saved moments' });
  expect(within(moments).getByRole('button', { name: /Source removed/ })).toBeDisabled();
  expect(within(moments).getByRole('button', { name: /Source unavailable/ })).toBeDisabled();
  expect(screen.queryByText(visitorLeft)).not.toBeInTheDocument();
  expect(seek).not.toHaveBeenCalled();
});

it('reports unavailable history and refreshes when an answer finishes', async () => {
  historyServer();
  server.use(
    http.get('/pack-ask/history', () => HttpResponse.json({ error: 'Local history unavailable' }, { status: 503 })),
  );
  const view = renderPage(<History refresh={null} />);
  await userEvent.click(screen.getByRole('button', { name: 'Past questions' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Local history unavailable');
  expect(screen.queryByText('No past questions yet.')).not.toBeInTheDocument();
  server.use(http.get('/pack-ask/history', () => HttpResponse.json({ items: [entry], next_before: null })));
  view.rerender(
    <Providers client={view.client} test>
      <History refresh="completed-job" />
    </Providers>,
  );
  await waitFor(() => expect(screen.getByRole('button', { name: entry.question })).toBeInTheDocument());
});

it('pages older questions with the server cursor and keeps the same day filter', async () => {
  const cursors: (string | null)[] = [];
  server.use(
    http.get('/pack-ask/history', ({ request }) => {
      const url = new URL(request.url);
      expect(url.searchParams.get('day')).toBe('2026-10-27');
      cursors.push(url.searchParams.get('before'));
      return HttpResponse.json(
        url.searchParams.has('before')
          ? { items: [{ ...entry, id: 'older', question: 'An older question', created_at: 4000 }], next_before: null }
          : { items: [entry], next_before: 'saved-1' },
      );
    }),
  );
  renderPage(<History day="2026-10-27" />);
  await open();
  await userEvent.click(screen.getByRole('button', { name: 'Older questions' }));
  await screen.findByRole('button', { name: 'An older question' });
  expect(cursors).toEqual([null, 'saved-1']);
  const questions = screen.getByRole('list', { name: 'Past questions' });
  expect(
    within(questions)
      .getAllByRole('listitem')
      .map(li => li.querySelector('button')?.textContent),
  ).toEqual([entry.question, 'An older question']);
});

it("drops an older-questions read that lands after the day changed, keeping the new day's list (#3202)", async () => {
  let release: () => void = () => {};
  const held = new Promise<void>(resolve => {
    release = resolve;
  });
  let olderAnswered = false;
  server.use(
    http.get('/pack-ask/history', async ({ request }) => {
      const url = new URL(request.url);
      const day = url.searchParams.get('day');
      if (url.searchParams.has('before')) {
        await held;
        olderAnswered = true;
        return HttpResponse.json({
          items: [{ ...entry, id: 'older', question: 'An older Monday question' }],
          next_before: null,
        });
      }
      return HttpResponse.json(
        day === '2026-10-27'
          ? { items: [{ ...entry, question: 'A Monday question' }], next_before: 'saved-1' }
          : { items: [{ ...entry, id: 'tue', question: 'A Tuesday question' }], next_before: null },
      );
    }),
  );
  const view = renderPage(<History day="2026-10-27" />);
  await userEvent.click(screen.getByRole('button', { name: 'Past questions' }));
  await screen.findByRole('button', { name: 'A Monday question' });
  await userEvent.click(screen.getByRole('button', { name: 'Older questions' }));
  view.rerender(
    <Providers client={view.client} test>
      <History day="2026-10-28" />
    </Providers>,
  );
  await screen.findByRole('button', { name: 'A Tuesday question' });
  release();
  await waitFor(() => expect(olderAnswered).toBe(true));
  // Let the stale reply's handlers run before checking the list it would have replaced.
  await new Promise(resolve => setTimeout(resolve, 20));
  const questions = screen.getByRole('list', { name: 'Past questions' });
  expect(
    within(questions)
      .getAllByRole('listitem')
      .map(li => li.querySelector('button')?.textContent),
  ).toEqual(['A Tuesday question']);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});

it('does not retain a working evidence link for an unverified saved citation', async () => {
  historyServer({ ...entry, citations: [{ ...citation, verified: false }] });
  renderPage(<History />);
  await userEvent.click(await open());
  const saved = await screen.findByRole('region', { name: 'Saved answer' });
  expect(within(saved).queryByRole('link')).not.toBeInTheDocument();
  expect(within(saved).getByRole('button', { name: /Source unavailable/ })).toBeDisabled();
});

it('lists the latest five past questions, and Show all lists the rest', async () => {
  const many = Array.from({ length: 7 }, (_, i) => ({ ...entry, id: `saved-${i}`, question: `Question ${i + 1}` }));
  server.use(http.get('/pack-ask/history', () => HttpResponse.json({ items: many, next_before: 3000 })));
  renderPage(<History day={null} />);
  await userEvent.click(screen.getByRole('button', { name: 'Past questions' }));
  const list = await screen.findByRole('list', { name: 'Past questions' });
  expect(within(list).getAllByRole('listitem')).toHaveLength(5);
  expect(screen.queryByRole('button', { name: 'Older questions' })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Show all' }));
  expect(within(list).getAllByRole('listitem')).toHaveLength(7);
  expect(within(list).getByRole('button', { name: 'Question 7' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Older questions' })).toBeInTheDocument(); // then further back
});

it("dates past questions in the app's 24-hour words, and names only a state that isn't finished", async () => {
  const failed = { ...entry, id: 'saved-2', question: 'A question that failed', status: 'failed' };
  server.use(http.get('/pack-ask/history', () => HttpResponse.json({ items: [entry, failed], next_before: null })));
  renderPage(<History day={null} />);
  await userEvent.click(screen.getByRole('button', { name: 'Past questions' }));
  const list = await screen.findByRole('list', { name: 'Past questions' });
  const [done, broke] = within(list).getAllByRole('listitem');
  const meta = (li: HTMLElement | undefined) => li?.querySelector('small')?.textContent ?? '';
  expect(meta(done)).toMatch(/, \d{2}:\d{2} · review · about \w/);
  expect(meta(done)).not.toMatch(/completed|AM|PM/);
  expect(meta(broke)).toMatch(/ · failed$/);
});
