import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { delay, HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { AskBar } from '../AskBar';
import { useAsk } from '../ask/openAsk';
import type { Job } from './api';
import { InboxLabel } from './InboxLabel';
import { count, InboxTab } from './InboxTab';

const T = 1_793_000_000;
const done: Job = {
  id: 'a'.repeat(32),
  question: 'What did Dana say about the garden?',
  day: '2026-10-22',
  status: 'done',
  created_at: T,
  started_at: T + 1,
  finished_at: T + 181,
  took_s: 180,
  read: false,
};
const waiting: Job = { ...done, id: 'b'.repeat(32), question: 'Who came by?', status: 'queued', took_s: undefined };
const report = {
  status: 'ok',
  answer: 'Dana wants to plant tomatoes [T1].',
  model: 'ollama/qwen3:4b-instruct',
  warnings: [],
  evidence: [
    {
      id: 'T1',
      ts: T - 3600,
      source: 'transcript',
      speaker: 'Dana',
      text: 'Tomatoes this year.',
      href: `/review#day=2026-10-22&t=${T - 3600}`,
    },
  ],
  looked_at: {
    sources: { transcript: 12, camera: 3 },
    evidence: 15,
    searches: [{ tool: 'transcript', args: { words: 'garden' }, found: 9 }],
    window: '2026-10-22',
  },
};
let posts: { url: string; body: unknown }[];

beforeEach(() => {
  posts = [];
  server.use(
    http.get('/research/jobs', () =>
      HttpResponse.json({ jobs: [waiting, done], available: true, reason: null, unread: 1 }),
    ),
    http.get(`/research/jobs/${done.id}`, () => HttpResponse.json({ ...done, result: report })),
    http.post('/research/jobs/:id/:action', async ({ request, params }) => {
      posts.push({ url: `${params.id}/${params.action}`, body: await request.json() });
      return HttpResponse.json({ ok: true });
    }),
  );
});

it('lists reports newest first, with the unread one marked new', async () => {
  renderPage(<InboxTab />);
  expect(
    await screen.findByRole('button', { name: /What did Dana say about the garden\?.*Done, unread/ }),
  ).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Who came by\?.*Waiting/ })).toBeInTheDocument();
  // Reports apart from the questions still waiting for one
  expect(screen.getByText('1 report · 1 waiting')).toBeInTheDocument();
});

it.each([
  [[], ''],
  [[{ status: 'done' }, { status: 'failed' }], '2 reports'],
  [[{ status: 'queued' }, { status: 'running' }], '2 waiting'],
  [[{ status: 'done' }, { status: 'queued' }], '1 report · 1 waiting'],
] as const)('counts %o as %o', (jobs, words) => {
  expect(count(jobs)).toBe(words);
});

it('opens a waiting report without a way into Review, since nothing of it exists yet', async () => {
  server.use(http.get(`/research/jobs/${waiting.id}`, () => HttpResponse.json(waiting)));
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /Who came by\?.*Waiting/ }));
  expect(await screen.findByText('Waiting for the research before it to finish.')).toBeInTheDocument();
  expect(screen.queryByRole('link', { name: 'Open in Review' })).not.toBeInTheDocument();
});

it('opens a report: its answer, the time it took, what it looked at, and marks it read', async () => {
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /garden/ }));
  const article = await screen.findByRole('article', { name: 'Research report' });
  expect(await within(article).findByText(/Dana wants to plant tomatoes/)).toBeInTheDocument();
  expect(within(article).getByText(/took 3 min/)).toBeInTheDocument();
  const looked = within(article).getByRole('region', { name: 'What it looked at' });
  const fold = within(looked).getByRole('button', {
    name: 'Looked at 15 sources: 12 transcript lines and 3 camera notes',
  });
  expect(looked).not.toHaveTextContent('Tomatoes this year.');
  await userEvent.click(fold);
  expect(within(looked).getByRole('region', { name: '1 transcript line' })).toHaveTextContent('Tomatoes this year.');
  expect(looked).toHaveTextContent('transcript: garden (9 found)');
  expect(looked).toHaveTextContent('Written by qwen3:4b-instruct');
  await waitFor(() => expect(posts).toContainEqual({ url: `${done.id}/read`, body: { read: true } }));
});

it('leads with the question and its actions, and says how sure it is in plain words', async () => {
  server.use(
    http.get(`/research/jobs/${done.id}`, () =>
      HttpResponse.json({
        ...done,
        result: {
          ...report,
          warnings: [
            'Transcripts are sampled ASR, not a complete conversation record; speaker labels may be wrong.',
            '468 gate-skipped/error rows excluded: these are not fresh visual observations.',
            'Something new the server says.',
          ],
        },
      }),
    ),
  );
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /garden/ }));
  const article = await screen.findByRole('article', { name: 'Research report' });
  await within(article).findByText(/Dana wants to plant tomatoes/);
  const text = article.textContent ?? '';
  // The actions come before the answer, not 4,000 px under it.
  expect(text.indexOf('Delete')).toBeLessThan(text.indexOf('Dana wants to plant tomatoes'));
  expect(within(article).getByRole('link', { name: 'Open in Review' })).toHaveAttribute(
    'href',
    expect.stringContaining('2026-10-22'),
  );
  // One verdict line for this report's own caveat; the ones that hold for every report fold under it
  const sure = within(article).getByRole('region', { name: 'How sure is this?' });
  expect(sure).toHaveTextContent("Partly sure. 468 snapshots couldn't be read, so some of it is missing.");
  await userEvent.click(within(sure).getByText('2 more caveats'));
  expect(
    within(sure)
      .getAllByRole('listitem')
      .map(li => li.textContent),
  ).toEqual(['Who said what comes from automatic transcripts, and can be wrong.', 'Something new the server says.']);
  expect(article).not.toHaveTextContent(/ASR|gate-skipped/);
  // The verdict says it, so the answer has no "Answer" heading or "Partial answer" badge of its own
  expect(within(article).queryByRole('heading', { name: 'Answer' })).toBeNull();
});

it('marks a report unread again, and deletes one', async () => {
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /garden/ }));
  await userEvent.click(await screen.findByRole('button', { name: 'Mark unread' }));
  await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
  // It asks first, with Cancel ("Keep it") focused, so Enter keeps the report
  const ask = await screen.findByRole('dialog', { name: 'Delete this report?' });
  expect(within(ask).getByRole('button', { name: 'Keep it' })).toHaveFocus();
  await userEvent.click(within(ask).getByRole('button', { name: 'Delete' }));
  await waitFor(() =>
    expect(posts).toEqual(
      expect.arrayContaining([
        { url: `${done.id}/read`, body: { read: false } },
        { url: `${done.id}/delete`, body: {} },
      ]),
    ),
  );
});

it('keeps the report when Delete is answered Keep it', async () => {
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /garden/ }));
  await userEvent.click(await screen.findByRole('button', { name: 'Delete' }));
  await userEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Keep it' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  expect(posts.filter(p => p.url.endsWith('/delete'))).toEqual([]);
  expect(screen.getByRole('article', { name: 'Research report' })).toBeInTheDocument();
});

it("links the answer's citation to its moment in Review, as the evidence list does", async () => {
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /garden/ }));
  const cite = await screen.findByRole('link', { name: 'T1' });
  expect(cite).toHaveAttribute('href', `/review#day=2026-10-22&t=${T - 3600}`);
  expect(cite.closest('p')).toHaveTextContent('Dana wants to plant tomatoes T1.');
});

it("says why research can't run here", async () => {
  server.use(
    http.get('/research/jobs', () =>
      HttpResponse.json({ jobs: [], available: false, reason: 'Research needs Ask’s own Ollama.', unread: 0 }),
    ),
  );
  renderPage(<InboxTab />);
  expect(await screen.findByText(/Research needs Ask/)).toBeInTheDocument();
  expect(screen.getByText('No research reports yet')).toBeInTheDocument();
  expect(screen.getByText(/Ask a question with Deep research on\./)).toBeInTheDocument();
});

it('shows how many reports are unread on the tab', async () => {
  renderPage(<InboxLabel />);
  expect(await screen.findByLabelText('1 unread')).toHaveTextContent('1');
});

it('holds the unread badge as a skeleton while the count loads, and the link stays "Inbox"', () => {
  server.use(http.get('/research/jobs', () => delay('infinite')));
  renderPage(
    <a href="/inbox">
      <InboxLabel />
    </a>,
  );
  expect(screen.getByTestId('inbox-count-skeleton')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Inbox' })).toBeInTheDocument();
});

it.each([
  ['none is unread', () => HttpResponse.json({ jobs: [], available: true, unread: 0 })],
  ['the count failed', () => HttpResponse.json({ error: 'down' }, { status: 500 })],
])('shows no badge and no skeleton once %s', async (_, answer) => {
  let asked = 0;
  server.use(
    http.get('/research/jobs', () => {
      asked += 1;
      return answer();
    }),
  );
  renderPage(<InboxLabel />);
  await waitFor(() => expect(screen.queryByTestId('inbox-count-skeleton')).not.toBeInTheDocument(), {
    timeout: 5000,
  });
  expect(asked).toBeGreaterThan(0);
  expect(screen.queryByLabelText(/unread/)).not.toBeInTheDocument();
});

it('holds the report count as a skeleton while the Inbox loads', () => {
  server.use(http.get('/research/jobs', () => delay('infinite')));
  renderPage(<InboxTab />);
  expect(screen.getByText('Loading how many reports')).toBeInTheDocument();
  expect(screen.getByTestId('count-skeleton')).toBeInTheDocument();
});

/** A wide screen: the reader sits beside the list. */
function asDesktop() {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    query =>
      ({
        matches: query.includes('min-width'),
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList,
  );
}
afterEach(() => vi.restoreAllMocks());

it('opens the newest unread report on arrival on a wide screen, and stays closed once closed', async () => {
  asDesktop();
  renderPage(<InboxTab />);
  // The waiting job is newer, but the unread finished one is what there is to read
  expect(await screen.findByRole('heading', { name: done.question })).toBeInTheDocument();
  await waitFor(() => expect(posts).toContainEqual({ url: `${done.id}/read`, body: { read: true } }));
  await userEvent.click(screen.getByRole('button', { name: 'All reports' }));
  expect(screen.queryByRole('article', { name: 'Research report' })).toBeNull();
  expect(screen.getByText('Choose a report to read it.')).toBeInTheDocument();
});

it('leaves a phone on the list, with nothing to choose into beside it', async () => {
  renderPage(<InboxTab />);
  await screen.findByRole('button', { name: /garden/ });
  expect(screen.queryByRole('article', { name: 'Research report' })).toBeNull();
  expect(screen.queryByText('Choose a report to read it.')).toBeNull();
});

it('marks a partial report on the list, and says when in 24-hour words', async () => {
  server.use(
    http.get('/research/jobs', () =>
      HttpResponse.json({ jobs: [{ ...done, answer_status: 'partial' }], available: true, reason: null, unread: 1 }),
    ),
  );
  renderPage(<InboxTab />);
  const row = await screen.findByRole('button', { name: /garden/ });
  expect(row).toHaveTextContent('Partial');
  expect(row).not.toHaveTextContent(/AM|PM/);
  expect(row).toHaveAccessibleName(/, \d\d:\d\d unread$/);
});

it('Research a question opens the one Ask here with Deep research on, and the pending report joins the list', async () => {
  history.replaceState(null, '', '/inbox');
  useAsk.setState({ open: false, request: null });
  const fresh: Job = { ...waiting, id: 'c'.repeat(32), question: 'What changed in the garden?', day: null };
  const sent: unknown[] = [];
  let jobs = [waiting, done];
  server.use(
    http.get('/research/jobs', () =>
      HttpResponse.json({ jobs, available: true, reason: null, unread: 1, typical_s: 300 }),
    ),
    http.post('/pack-ask/jobs', async ({ request }) => {
      sent.push(await request.json());
      jobs = [fresh, waiting, done];
      return HttpResponse.json({ ...fresh, deep: true }, { status: 202 });
    }),
  );
  renderPage(
    <>
      <AskBar tab="inbox" setup={false} />
      <InboxTab />
    </>,
  );
  await userEvent.click(await screen.findByRole('button', { name: 'Research a question' }));
  const ask = await screen.findByRole('dialog', { name: 'Ask' });
  const toggle = await within(ask).findByRole('switch', { name: /^Deep research/ });
  await waitFor(() => expect(toggle).toBeChecked());
  await userEvent.type(
    within(ask).getByRole('textbox', { name: 'Your question' }),
    'What changed in the garden?{Enter}',
  );
  await waitFor(() => expect(sent).toEqual([{ question: 'What changed in the garden?', deep: true }]));
  const note = (await within(ask).findByText('Research started')).closest('[data-ask-sent]');
  expect(note).toHaveTextContent('The report will be in the Inbox. Usually ready in about 5 minutes.');
  // The pending report is in the list at once, and the page never moved
  expect(await screen.findByRole('button', { name: /^What changed in the garden\?.*Waiting/ })).toBeInTheDocument();
  expect(location.pathname).toBe('/inbox');
  useAsk.setState({ open: false, request: null });
});

it('shows the moments the answer cites as a timeline, each opening Review', async () => {
  server.use(
    http.get(`/research/jobs/${done.id}`, () =>
      HttpResponse.json({
        ...done,
        result: {
          ...report,
          answer: 'Dana sat down at C2, and said so in T1.',
          evidence: [
            ...report.evidence,
            {
              id: 'C2',
              ts: T - 7200,
              source: 'camera',
              camera: 'cam0',
              text: 'One person seated.',
              href: '/review#t=1',
            },
          ],
        },
      }),
    ),
  );
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /garden/ }));
  const moments = await screen.findByRole('list', { name: 'The moments it cites' });
  const rows = within(moments).getAllByRole('listitem');
  // Oldest first: the camera note two hours before, then what Dana said
  expect(rows.map(r => r.textContent)).toEqual([
    expect.stringContaining('One person seated.'),
    expect.stringContaining('Tomatoes this year.'),
  ]);
  expect(within(rows[0] as HTMLElement).getByRole('link')).toHaveAttribute('href', '/review#t=1');
  expect(rows[1]).toHaveTextContent('Said · Dana');
  // The camera moment's still is a 404 (its video was dropped): its box says so, not just "Camera note"
  expect(await within(rows[0] as HTMLElement).findByText('Video no longer kept')).toBeInTheDocument();
  expect(rows[1]).not.toHaveTextContent('Video no longer kept');
});

it('asks the same question again, and opens the new report', async () => {
  const fresh: Job = { ...waiting, id: 'c'.repeat(32), question: done.question, day: done.day, status: 'queued' };
  const sent: unknown[] = [];
  server.use(
    http.post('/pack-ask/jobs', async ({ request }) => {
      sent.push(await request.json());
      return HttpResponse.json({ ...fresh, deep: true }, { status: 202 });
    }),
    http.get(`/research/jobs/${fresh.id}`, () => HttpResponse.json(fresh)),
    http.get('/research/jobs', () =>
      HttpResponse.json({ jobs: [fresh, waiting, done], available: true, reason: null, unread: 1 }),
    ),
  );
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /^What did Dana say about the garden\?.*Done/ }));
  await userEvent.click(await screen.findByRole('button', { name: 'Ask again' }));
  expect(sent).toEqual([{ question: done.question, deep: true, day: done.day }]);
  // The new one is open: waiting for the research before it
  expect(await screen.findByText('Waiting for the research before it to finish.')).toBeInTheDocument();
});

it('opens the one Ask on the report for a follow-up', async () => {
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /garden/ }));
  await userEvent.click(await screen.findByRole('button', { name: 'Ask a follow-up' }));
  expect(useAsk.getState()).toMatchObject({
    open: true,
    request: { hint: { kind: 'report', id: done.id, question: done.question, day: done.day } },
  });
  useAsk.setState({ open: false, request: null });
});

it("opens Review on the day a report was asked when it isn't about a day of its own", async () => {
  server.use(
    http.get('/research/jobs', () =>
      HttpResponse.json({ jobs: [{ ...done, day: null }], available: true, reason: null, unread: 1 }),
    ),
    http.get(`/research/jobs/${done.id}`, () => HttpResponse.json({ ...done, day: null, result: report })),
  );
  renderPage(<InboxTab />);
  await userEvent.click(await screen.findByRole('button', { name: /garden/ }));
  const day = new Date(done.created_at * 1000).toLocaleDateString('sv');
  expect(await screen.findByRole('link', { name: 'Open in Review' })).toHaveAttribute(
    'href',
    expect.stringContaining(day),
  );
});
