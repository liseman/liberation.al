import { act, screen, waitFor, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { openAsk, useAsk } from './openAsk';
import { deepWords, PackAsk, thin } from './PackAsk';
import {
  alertText,
  buttons,
  fakeTimers,
  jobServer,
  KEY,
  question,
  renderAsk,
  researchServer,
  saved,
  status,
  store,
  wait,
} from './testing';

let user: ReturnType<typeof fakeTimers>;
beforeEach(() => {
  sessionStorage.clear();
  history.replaceState(null, '', '/');
  useAsk.setState({ open: false, request: null });
  user = fakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

const names = () => buttons().map(b => b.textContent);
// Mantine's Switch is a label around its input, its description included: the name starts with the label.
const deep = () => screen.getByRole('switch', { name: /^Deep research/ });
const evidenceCards = () => screen.queryAllByRole('article').filter(a => !a.querySelector('header'));

/** Wait for the server to acknowledge the question just sent. */
const queued = () => waitFor(() => expect(status()).toHaveTextContent(/ · (queued|running)$/));

/** Type a question and submit it with Enter. */
async function ask(text: string) {
  await user.type(question(), text);
  await user.keyboard('{Enter}');
}

/** Ask again with the Ask button, then run the first poll. */
async function again() {
  await user.click(screen.getByRole('button', { name: 'Ask' }));
  await queued();
  await wait(1000);
}

const DAY = { kind: 'day', day: '2026-10-20', label: 'Tue 20 Oct', today: false } as const;

it('focuses the question when the deferred panel contents mount', () => {
  renderPage(<PackAsk />);
  expect(question()).toHaveFocus();
});

describe('one Ask: a quick answer by default', () => {
  it('opens on the box with Deep research off, one Ask, and nothing said at rest', () => {
    renderAsk();
    const section = screen.getByRole('region', { name: 'Ask' });
    expect(names()).toEqual(['Ask']);
    expect(deep()).not.toBeChecked();
    expect(screen.queryByRole('radiogroup')).not.toBeInTheDocument(); // no choice of who to ask
    expect(question()).toHaveAccessibleName('Your question');
    expect(question()).toHaveAccessibleDescription(
      'Searches your recordings for the time the question names. Nothing leaves this machine.',
    );
    expect(status()).toHaveTextContent(/^$/);
    expect(section).not.toHaveTextContent(/Friends|Everyone|My recordings/);
  });

  it('asks for a question instead of sending an empty one, and focuses the box', async () => {
    const net = jobServer();
    renderAsk();
    question().blur();
    await user.click(screen.getByRole('button', { name: 'Ask' }));
    expect(alertText()).toBe('Please enter a question first.');
    expect(question()).toHaveFocus();
    expect(net.posts).toEqual([]);
  });

  it('sends Enter as a quick question once, however often it is pressed or clicked', async () => {
    const net = jobServer();
    renderAsk();
    await ask('Question <img onerror=alert(1)>');
    await waitFor(() => expect(net.posts).toHaveLength(1));
    // No day: the server reads the question for its own window
    expect(net.posts[0]).toEqual({ question: 'Question <img onerror=alert(1)>' });
    expect(buttons().every(b => (b as HTMLButtonElement).disabled)).toBe(true);
    await user.keyboard('{Enter}');
    await wait(0);
    expect(net.posts).toHaveLength(1);
    await waitFor(() => expect(status()).toHaveTextContent('Ask · queued'));
    expect(saved(KEY)).toEqual({
      active: {
        id: 'j1',
        poll_url: '/pack-ask/jobs/j1',
        question: 'Question <img onerror=alert(1)>',
        created_at: expect.any(Number),
      },
      question: 'Question <img onerror=alert(1)>',
    });
  });

  it('offers questions to start from, and asks one when picked', async () => {
    const net = jobServer();
    renderAsk();
    const start = screen.getByRole('list', { name: 'Suggested questions' });
    await user.click(within(start).getByRole('button', { name: 'Who came by today?' }));
    await waitFor(() => expect(net.posts.at(-1)).toEqual({ question: 'Who came by today?' }));
    expect(screen.queryByRole('list', { name: 'Suggested questions' })).not.toBeInTheDocument();
  });

  it('adds a line on Shift+Enter and sends nothing', async () => {
    const net = jobServer();
    renderAsk();
    await user.type(question(), 'two{Shift>}{Enter}{/Shift}lines');
    expect(question()).toHaveValue('two\nlines');
    await wait(0);
    expect(net.posts).toEqual([]);
  });

  it('keeps tracking a job when a status read fails, backing off to 3 s', async () => {
    const net = jobServer();
    net.replies.push('offline');
    renderAsk();
    await ask('Q');
    await queued();
    await wait(1000);
    await waitFor(() =>
      expect(alertText()).toMatch(
        /^Cannot read job status \(.+\)\. Still tracking this job; retrying in 3 seconds\. Job ID: j1$/,
      ),
    );
    expect(net.polls).toEqual(['j1']);
    await wait(2500);
    expect(net.polls).toEqual(['j1']);
    await wait(600);
    await waitFor(() => expect(net.polls).toEqual(['j1', 'j1']));
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
  });

  it('polls a second apart three times, then every 3 s', async () => {
    const net = jobServer();
    renderAsk();
    await ask('Q');
    await queued();
    for (const n of [1, 2, 3]) {
      await wait(1000);
      await waitFor(() => expect(net.polls).toHaveLength(n));
    }
    await wait(2500);
    expect(net.polls).toHaveLength(3);
    await wait(600);
    await waitFor(() => expect(net.polls).toHaveLength(4));
  });

  it('shows the answer as it is written, polling every second, until the finished result replaces it', async () => {
    const net = jobServer();
    net.replies.push(
      { status: 'running' },
      { status: 'running' },
      { status: 'running', partial: 'Dana came' },
      { status: 'running', partial: 'Dana came by' },
      {
        status: 'completed',
        finished_at: 1700000100,
        result: { answer: 'Dana came by at noon.', status: 'ok', evidence: [{ id: 'T1', text: 'Hello' }] },
      },
    );
    renderAsk();
    await ask('Q');
    await queued();
    for (const n of [1, 2, 3]) {
      await wait(1000);
      await waitFor(() => expect(net.polls).toHaveLength(n));
    }
    expect(await screen.findByLabelText('Answer so far')).toHaveTextContent('Dana came');
    await wait(1000); // a fourth poll a second later, not 3 s, while the answer grows
    await waitFor(() => expect(screen.getByLabelText('Answer so far')).toHaveTextContent('Dana came by'));
    await wait(1000);
    expect(await screen.findByText('Dana came by at noon.')).toBeInTheDocument();
    expect(screen.queryByLabelText('Answer so far')).not.toBeInTheDocument();
    expect(status()).toHaveTextContent(/^Ask · completed · /);
    expect(status().querySelector('time')).toHaveAttribute('datetime', '2023-11-14T22:15:00.000Z');
    expect(screen.queryByRole('button', { name: 'Research it in depth' })).not.toBeInTheDocument();
  });

  it("reports a failed job's structured error", async () => {
    const net = jobServer();
    net.replies.push({ status: 'failed', error: { message: 'Structured job failure' } });
    renderAsk();
    await ask('Q');
    await queued();
    await wait(1000);
    await waitFor(() => expect(alertText()).toBe('Structured job failure'));
    expect(status()).toHaveTextContent(/^Ask · failed$/);
  });

  it('says so when a completed job has no result', async () => {
    const net = jobServer();
    net.replies.push({ status: 'completed' });
    renderAsk();
    await ask('Q');
    await queued();
    await wait(1000);
    await waitFor(() => expect(alertText()).toBe('Job completed without a result.'));
  });

  it('stops and keeps the question when the job expired (404)', async () => {
    const net = jobServer();
    net.replies.push({ httpStatus: 404 });
    renderAsk();
    await ask('Preserve this question');
    await queued();
    await wait(1000);
    await waitFor(() => expect(alertText()).toMatch(/server restart or expired/));
    expect(status()).toHaveTextContent(/^Ask · interrupted or expired$/);
    expect(question()).toHaveValue('Preserve this question');
    expect(buttons().every(b => !(b as HTMLButtonElement).disabled)).toBe(true);
    await wait(10_000);
    expect(net.polls).toEqual(['j1']);
  });

  it("explains a refused submission, with the server's reason or its status", async () => {
    const net = jobServer();
    net.acks.push({ httpStatus: 500 });
    renderAsk();
    await ask('Q');
    await waitFor(() =>
      expect(alertText()).toBe(
        'Could not confirm submission (HTTP 500). The server may have accepted it. No automatic resubmit; retrying may create another job.',
      ),
    );
    expect(status()).toHaveTextContent('Ask · submission not confirmed');
    expect(buttons().every(b => !(b as HTMLButtonElement).disabled)).toBe(true);
  });

  it('rejects an acknowledgement whose poll URL is not a job URL', async () => {
    const net = jobServer();
    net.acks.push({ id: 'x', status: 'queued', poll_url: 'https://evil.test/pack-ask/jobs/x' });
    renderAsk();
    await ask('Q');
    await waitFor(() => expect(alertText()).toMatch(/^Could not confirm submission \(Invalid job acknowledgement\)/));
    await wait(5000);
    expect(net.polls).toEqual([]);
  });

  it('bounds evidence to 20 cards, drops unsafe links, and shows coverage', async () => {
    const net = jobServer();
    const evidence = Array.from({ length: 25 }, (_, i) => ({
      id: i,
      timestamp: 1700000000,
      text: '<img>',
      href: i === 0 ? 'javascript:alert(1)' : i === 1 ? '//evil.test/' : '/review#t=1700000000',
    }));
    net.replies.push({
      status: 'completed',
      result: {
        answer: 'An estimate.',
        status: 'ok',
        coverage: { partial: true, available: 8, included: 5 },
        evidence,
      },
    });
    renderAsk();
    await ask('Evidence question');
    await queued();
    await wait(1000);
    expect(await screen.findByRole('heading', { name: 'Evidence (25)' })).toBeInTheDocument();
    expect(evidenceCards()).toHaveLength(20);
    const links = screen.getAllByRole('link').filter(a => !a.textContent?.startsWith('New'));
    expect(links).toHaveLength(18);
    expect(links.every(a => a.getAttribute('href') === '/review#t=1700000000')).toBe(true);
    expect(screen.getByText('This answer is based on partial recorded evidence.')).toBeInTheDocument();
    await user.click(screen.getByText('History coverage'));
    expect(screen.getByText('Included').nextElementSibling).toHaveTextContent('5');
    await user.click(screen.getByText('More (5 remaining)'));
    expect(evidenceCards()).toHaveLength(25);
  });

  it('lists the search rounds with a New tag that opens its Settings switch in place', async () => {
    const net = jobServer();
    net.replies.push({
      status: 'completed',
      result: {
        answer: 'An estimate from recorded frames.',
        evidence: [{ id: 'C1', text: 'Frame evidence', href: '/review#day=2026-10-20&t=1792526400' }],
        rounds: [
          { round: 1, decision: 'search', tool: 'journal', args: { text: 'lights' }, found: 2, new: ['J1', 'J2'] },
          { round: 2, decision: 'error', error: 'unknown tool' },
          { round: 3, decision: 'answer' },
          null,
        ],
      },
    });
    renderAsk();
    await ask('Q');
    await queued();
    await wait(1000);
    const summary = await screen.findByRole('button', { name: 'Search rounds (3)' });
    await user.click(summary);
    const rounds = within(summary.closest('[data-disclosure]') as HTMLElement).getAllByRole('listitem');
    expect(rounds.map(r => r.textContent)).toEqual([
      'journal (text lights): 2 found, new J1 J2',
      'Skipped: unknown tool',
      'Enough evidence; answered.',
    ]);
    const tag = screen.getByRole('link', { name: "New: Ask's search loop. Its switch is in Settings." });
    await user.click(tag);
    expect(location.pathname + location.hash).toBe('/settings#set-recall.enabled');
  });

  it('tags journal, facts, timeline, detections, and forgotten answers, and nothing else', async () => {
    const net = jobServer();
    const tags = (root: HTMLElement) =>
      within(root)
        .queryAllByRole('link', { name: /^New:/ })
        .map(a => a.getAttribute('href'));
    renderAsk();
    await user.type(question(), 'Who was here?');

    net.replies.push({
      status: 'completed',
      result: {
        answer: 'Recognized in today: Dana (heard).',
        paths: ['timeline'],
        evidence: [
          { id: 'J1', source: 'journal', text: 'Dana made tea.' },
          { id: 'F1', source: 'facts', text: 'Dana likes tea.' },
          { id: 'T1', source: 'transcript', text: 'Morning.' },
        ],
      },
    });
    await again();
    const answer = (await screen.findByRole('heading', { name: 'Answer' })).closest('section') as HTMLElement;
    expect(tags(answer)).toEqual(['/settings#set-timeline.enabled']);
    expect(evidenceCards().map(tags)).toEqual([['/settings#set-journal.enabled'], ['/settings#set-journal.facts'], []]);

    net.replies.push({
      status: 'completed',
      result: { answer: 'Dana was last seen at 10:00.', paths: ['detections'], evidence: [{ id: 'C1' }] },
    });
    await again();
    await screen.findByText('Dana was last seen at 10:00.');
    expect(tags(document.body)).toEqual(['/settings#set-detections.last_seen']);

    net.replies.push({
      status: 'completed',
      result: { answer: "Doesn't look like anything to me.", is_history: false, paths: ['forgotten'] },
    });
    await again();
    await screen.findByText("Doesn't look like anything to me.");
    expect(tags(document.body)).toEqual(['/settings#set-assistant.forgotten_reply']);
  });

  it('opens an evidence link into Review in place, and fires hashchange', async () => {
    const net = jobServer();
    net.replies.push({
      status: 'completed',
      result: {
        answer: 'See [the moment](#t=1792526400).',
        evidence: [{ id: 'C1', timestamp: 1792526400, href: '/review#day=2026-10-20&t=1792526400' }],
      },
    });
    const changes: string[] = [];
    const listen = () => changes.push(location.hash);
    window.addEventListener('hashchange', listen);
    renderAsk();
    await ask('Q');
    await queued();
    await wait(1000);
    const card = (await screen.findAllByRole('article'))[0] as HTMLElement;
    await user.click(within(card).getByRole('link'));
    expect(location.pathname + location.hash).toBe('/review#day=2026-10-20&t=1792526400');
    await user.click(screen.getByRole('link', { name: 'the moment' }));
    await waitFor(() => expect(changes).toEqual(['#day=2026-10-20&t=1792526400', '#t=1792526400']));
    window.removeEventListener('hashchange', listen);
  });
});

describe('a hint from the page that opened Ask', () => {
  it("asks about Review's day while its chip is there, and about the question's own time once it is closed", async () => {
    const net = jobServer();
    renderAsk();
    act(() => openAsk({ hint: DAY }));
    expect(await screen.findByText('Tue 20 Oct')).toBeInTheDocument();
    expect(question()).toHaveAccessibleDescription(/^Searches your recordings from Tue 20 Oct, unless/);
    // the day's own starters
    expect(screen.getByRole('button', { name: 'Who was here that day?' })).toBeInTheDocument();
    await ask('How long were lights on?');
    await waitFor(() => expect(net.posts).toEqual([{ question: 'How long were lights on?', day: '2026-10-20' }]));
    await user.click(screen.getByRole('button', { name: 'Stop asking about Tue 20 Oct' }));
    expect(screen.queryByText('Tue 20 Oct')).not.toBeInTheDocument();
    expect(question()).toHaveFocus();
  });

  it('follows up on a report, sending its id and day, with Deep research off, until the chip is closed', async () => {
    const net = jobServer();
    researchServer();
    renderAsk();
    act(() =>
      openAsk({
        hint: { kind: 'report', id: 'r1', question: 'Who came by this week?', day: '2026-11-05' },
        draft: 'And when',
      }),
    );
    expect(await screen.findByText('Who came by this week?')).toBeInTheDocument();
    expect(question()).toHaveValue('And when');
    expect(deep()).toBeDisabled();
    expect(deep()).toHaveAccessibleDescription('A follow-up on a report gets a quick answer.');
    expect(screen.queryByRole('list', { name: 'Suggested questions' })).not.toBeInTheDocument();
    await user.type(question(), '?{Enter}');
    await waitFor(() =>
      expect(net.posts).toEqual([{ question: 'And when?', day: '2026-11-05', followup: { report: 'r1' } }]),
    );
    await user.click(screen.getByRole('button', { name: 'Stop following up on this report' }));
    expect(screen.queryByText('Who came by this week?')).not.toBeInTheDocument();
  });

  it('starts again from no hint when the top bar opens it off Review, keeping the draft', async () => {
    renderAsk();
    act(() => openAsk({ hint: DAY }));
    await screen.findByText('Tue 20 Oct');
    await user.type(question(), 'Half');
    act(() => openAsk({ hint: null }));
    await waitFor(() => expect(screen.queryByText('Tue 20 Oct')).not.toBeInTheDocument());
    expect(question()).toHaveValue('Half');
  });
});

describe('Deep research', () => {
  /** Deep research's posts, apart from quick ones. */
  function deepServer(reply: Record<string, unknown> | { httpStatus: number } = {}) {
    const posts: unknown[] = [];
    server.use(
      http.post('/pack-ask/jobs', async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        posts.push(body);
        if ('httpStatus' in reply) return HttpResponse.json({ error: 'Research is turned off.' }, { status: 409 });
        return HttpResponse.json(
          body.deep
            ? { id: 'r1', question: body.question, status: 'queued', deep: true, ...reply }
            : { id: 'j1', status: 'queued', poll_url: '/pack-ask/jobs/j1' },
          { status: 202 },
        );
      }),
    );
    return posts;
  }

  it('says what it looks through when switched on, and sends the question to research, not the quick path', async () => {
    researchServer();
    const posts = deepServer();
    renderAsk();
    act(() => openAsk({ hint: DAY }));
    await user.click(deep());
    await waitFor(() =>
      expect(deep()).toHaveAccessibleDescription(
        'Up to 8 search rounds over up to 30 minutes (a quick answer: 3 rounds in 80 s), answered by big-model. The report lands in the Inbox. Research here usually takes about 10 minutes.',
      ),
    );
    expect(screen.getByRole('button', { name: 'Research' })).toBeInTheDocument();
    await ask('What changed in the garden?');
    await waitFor(() =>
      expect(posts).toEqual([{ question: 'What changed in the garden?', deep: true, day: '2026-10-20' }]),
    );
    const sent = await screen.findByText('Research started');
    const note = sent.closest('[data-ask-sent]') as HTMLElement;
    expect(note).toHaveTextContent(
      '“What changed in the garden?” is being looked into in the background. The report will be in the Inbox in about 10 minutes.',
    );
    expect(within(note).getByRole('link', { name: 'Inbox' })).toHaveAttribute('href', '/inbox');
    // Not sticky: the next question is quick unless switched on again; the box is empty for it
    expect(deep()).not.toBeChecked();
    expect(question()).toHaveValue('');
    expect(screen.queryByText(/^Ask · /)).not.toBeInTheDocument();
  });

  it('opens switched on from the Inbox, for that question only', async () => {
    researchServer();
    renderAsk();
    act(() => openAsk({ deep: true }));
    await waitFor(() => expect(deep()).toBeChecked());
    act(() => openAsk({ hint: null }));
    await waitFor(() => expect(deep()).not.toBeChecked());
  });

  it("can't be switched on where research can't run, and says why", async () => {
    researchServer({ available: false, reason: 'Research is turned off in Settings.' });
    renderAsk();
    await waitFor(() => expect(deep()).toBeDisabled());
    expect(deep()).toHaveAccessibleDescription('Research is turned off in Settings.');
  });

  it("says so when the server won't take it, and keeps the question", async () => {
    researchServer();
    deepServer({ httpStatus: 409 });
    renderAsk();
    await user.click(deep());
    await ask('Q');
    await waitFor(() => expect(alertText()).toBe('Research is turned off.'));
    expect(question()).toHaveValue('Q');
    expect(deep()).toBeChecked();
  });
});

describe('no quick answer: research offered', () => {
  it('says so plainly and researches the same question in one action', async () => {
    researchServer();
    const net = jobServer();
    net.replies.push({
      status: 'completed',
      result: {
        answer: 'No matching safe, timestamped evidence in the available history.',
        status: 'ok',
        evidence: [],
      },
    });
    renderAsk();
    await ask('When did Rex last go out?');
    await queued();
    await wait(1000);
    const offer = (await screen.findByText('Nothing in your recordings answered this quickly.')).closest(
      '[data-ask-offer]',
    ) as HTMLElement;
    await user.clear(question());
    await user.type(question(), 'something else'); // the offer researches the question answered, not a new draft
    await user.click(within(offer).getByRole('button', { name: 'Research it in depth' }));
    await waitFor(() => expect(net.posts.at(-1)).toEqual({ question: 'When did Rex last go out?', deep: true }));
    expect(await screen.findByText('Research started')).toBeInTheDocument();
    expect(question()).toHaveValue('something else');
  });

  it.each([
    [{ status: 'error', answer: 'x', evidence: [{ id: 1 }] }, "Ask couldn't answer this quickly."],
    [{ status: 'ok', answer: 'x', evidence: [] }, 'Nothing in your recordings answered this quickly.'],
    [
      { status: 'partial', answer: 'x', evidence: [{ id: 1 }] },
      'This quick answer rests on only part of the evidence.',
    ],
    [{ status: 'ok', answer: 'x', evidence: [{ id: 1 }] }, null],
    [{ status: 'ok', answer: 'Forgotten.', is_history: false }, null],
  ])('thin(%o) is %s', (result, why) => {
    expect(thin(result)).toBe(why);
  });
});

it('words deep research from what the server says, without a typical time before any report', () => {
  const reach = (rounds: number, search_s: number, model: string) => ({
    deep: true,
    rounds,
    search_s,
    sources: [],
    model,
    day: null,
  });
  expect(deepWords(reach(1, 90, 'ollama/m'), undefined, null)).toBe(
    'Up to 1 search round over up to 2 minutes, answered by m. The report lands in the Inbox.',
  );
});

describe('restoring a saved job', () => {
  const active = { id: 'g1', poll_url: '/pack-ask/jobs/g1', question: 'Q', created_at: 1 };

  it('recovers a running job after a reload without resubmitting, then shows its end', async () => {
    const net = jobServer();
    store(KEY, { active, question: 'Q draft' });
    renderAsk();
    expect(status()).toHaveTextContent('Ask · recovering job g1');
    expect(question()).toHaveValue('Q draft');
    expect(buttons().every(b => (b as HTMLButtonElement).disabled)).toBe(true);
    net.replies.push({ status: 'completed', result: { answer: 'Recovered answer.', evidence: [{ id: 1 }] } });
    await wait(1000);
    expect(await screen.findByText('Recovered answer.')).toBeInTheDocument();
    expect(net.posts).toEqual([]);
    expect(saved(KEY)).toEqual({ active: null, question: 'Q draft' });
  });

  it('never resumes a job saved under an older key, and saves the draft as it is typed', async () => {
    const net = jobServer();
    store('pack-ask.job.v2.general', { active, question: 'Old' });
    renderAsk();
    expect(question()).toHaveValue('');
    await wait(5000);
    expect(net.polls).toEqual([]);
    await user.type(question(), 'Half a thought');
    expect(saved(KEY)).toEqual({ active: null, question: 'Half a thought' });
  });
});
