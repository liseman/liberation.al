import { act, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Providers } from '../../shared/Providers';
import { renderPage } from '../../test/render';
import { openAsk } from './openAsk';
import { PackAsk } from './PackAsk';
import {
  alertText,
  buttons,
  fakeTimers,
  GENERAL,
  jobServer,
  question,
  REVIEW,
  renderAsk,
  saved,
  status,
  store,
  wait,
} from './testing';

let user: ReturnType<typeof fakeTimers>;
beforeEach(() => {
  sessionStorage.clear();
  history.replaceState(null, '', '/');
  user = fakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

const names = () => buttons().map(b => b.textContent);
const scope = (name: string) => screen.getByRole('radio', { name });

/** Pick a scope on Live, then press Ask. */
async function askIn(name: 'My recordings' | 'Friends' | 'Everyone') {
  await user.click(scope(name));
  await user.click(screen.getByRole('button', { name: 'Ask' }));
}
const cards = () => screen.queryAllByRole('article').filter(a => a.querySelector('header'));
const evidenceCards = () => screen.queryAllByRole('article').filter(a => !a.querySelector('header'));

/** Wait for the server to acknowledge the question just sent. */
const queued = () => waitFor(() => expect(status()).toHaveTextContent(/ · (queued|running)$/));

/** Type a question and submit it with Enter. */
async function ask(text: string) {
  await user.type(question(), text);
  await user.keyboard('{Enter}');
}

/** Ask again with the first button, then run the first poll. */
async function again() {
  await user.click(buttons()[0] as HTMLElement);
  await queued();
  await wait(1000);
}

it('focuses the compact question when the deferred panel contents mount', () => {
  renderPage(<PackAsk showModes bare compact />);
  expect(question()).toHaveFocus();
});

describe('Live: My recordings searches recordings, Friends and Everyone are general', () => {
  it('opens on the box with one scope choice on My recordings, one Ask, and nothing said at rest', () => {
    renderAsk();
    const section = screen.getByRole('region', { name: 'Ask' });
    expect(within(section).getByRole('heading', { name: 'Ask' })).toBeInTheDocument();
    expect(names()).toEqual(['Ask']);
    expect(within(screen.getByRole('radiogroup', { name: 'Who to ask' })).getAllByRole('radio')).toHaveLength(3);
    expect(scope('My recordings')).toBeChecked();
    const input = screen.getByLabelText('What would you like to ask?');
    expect(input).toHaveAttribute('data-ask-question', 'general'); // what the / key finds on Review, by kind
    expect(input).toHaveAccessibleDescription('Searches your recordings on this machine. Nothing leaves it.');
    expect(status()).toHaveTextContent(/^$/);
    expect(section).not.toHaveTextContent(/selected day|LLMs|local selected/i);
  });

  it('says what each scope shares as it is picked', async () => {
    renderAsk();
    await user.click(scope('Friends'));
    expect(question()).toHaveAccessibleDescription(/They see your question, not your recordings\.$/);
    await user.click(scope('Everyone'));
    expect(question()).toHaveAccessibleDescription(/^Asks every model/);
  });

  it('asks for a question instead of sending an empty one, and focuses the box', async () => {
    const net = jobServer();
    renderAsk();
    question().blur();
    await askIn('Friends');
    expect(alertText()).toBe('Please enter a question first.');
    expect(question()).toHaveFocus();
    expect(net.posts).toEqual([]);
  });

  it('sends Enter as a local recordings question once, however often it is pressed or clicked', async () => {
    const net = jobServer();
    renderAsk();
    await ask('Question <img onerror=alert(1)>');
    await waitFor(() => expect(net.posts).toHaveLength(1));
    // Local runs the recordings search with no day, so the question names its own window ("today")
    expect(net.posts[0]).toEqual({ question: 'Question <img onerror=alert(1)>', mode: 'local', scope: 'history' });
    expect(buttons().every(b => (b as HTMLButtonElement).disabled)).toBe(true);
    expect(scope('Friends')).toBeDisabled();
    await user.keyboard('{Enter}');
    await wait(0);
    expect(net.posts).toHaveLength(1);
    await waitFor(() => expect(status()).toHaveTextContent('My recordings · queued'));
    expect(saved(GENERAL)).toEqual({
      active: {
        id: 'j1',
        poll_url: '/pack-ask/jobs/j1',
        mode: 'local',
        scope: 'history',
        question: 'Question <img onerror=alert(1)>',
        created_at: expect.any(Number),
      },
      question: 'Question <img onerror=alert(1)>',
    });
  });

  it('sends a friends question only once Friends is picked, from Ask or Enter', async () => {
    const net = jobServer();
    renderAsk();
    await user.type(question(), 'explain tides');
    await askIn('Friends');
    await waitFor(() => expect(net.posts).toEqual([{ question: 'explain tides', mode: 'friends', scope: 'general' }]));
    expect(scope('Friends')).toBeChecked();
  });

  it('follows up on a report opened with openAsk, sending its id and day, until the chip is closed', async () => {
    const net = jobServer();
    renderAsk();
    act(() =>
      openAsk({
        context: { kind: 'report', id: 'r1', question: 'Who came by this week?', day: '2026-11-05' },
        draft: 'And when',
      }),
    );
    expect(await screen.findByText('Who came by this week?')).toBeInTheDocument();
    expect(question()).toHaveValue('And when');
    expect(scope('Friends')).toBeDisabled(); // a follow-up asks My recordings
    expect(screen.queryByRole('list', { name: 'Suggested questions' })).not.toBeInTheDocument();
    await user.type(question(), '?{Enter}');
    await waitFor(() =>
      expect(net.posts).toEqual([
        { question: 'And when?', mode: 'local', scope: 'history', day: '2026-11-05', followup: { report: 'r1' } },
      ]),
    );
    await user.click(screen.getByRole('button', { name: 'Stop following up on this report' }));
    expect(screen.queryByText('Who came by this week?')).not.toBeInTheDocument();
    expect(question()).toHaveFocus();
  });

  it('offers questions about your recordings to start from, asking one on My recordings whatever was picked', async () => {
    const net = jobServer();
    renderAsk();
    await user.click(scope('Everyone'));
    const start = screen.getByRole('list', { name: 'Suggested questions' });
    await user.click(within(start).getByRole('button', { name: 'Who came by today?' }));
    await waitFor(() =>
      expect(net.posts.at(-1)).toEqual({ question: 'Who came by today?', mode: 'local', scope: 'history' }),
    );
    expect(scope('My recordings')).toBeChecked();
    expect(screen.queryByRole('list', { name: 'Suggested questions' })).not.toBeInTheDocument();
  });

  it('sends Enter in the scope picked', async () => {
    const net = jobServer();
    renderAsk();
    await user.click(scope('Everyone'));
    await ask('explain tides');
    await waitFor(() => expect(net.posts).toEqual([{ question: 'explain tides', mode: 'everyone', scope: 'general' }]));
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
      { status: 'running', partial: 'Gravity bends' },
      { status: 'running', partial: 'Gravity bends light' },
      { status: 'completed', finished_at: 1700000100, result: { answer: 'Gravity bends light, a lot.', status: 'ok' } },
    );
    renderAsk();
    await ask('Q');
    await queued();
    for (const n of [1, 2, 3]) {
      await wait(1000);
      await waitFor(() => expect(net.polls).toHaveLength(n));
    }
    expect(await screen.findByLabelText('Answer so far')).toHaveTextContent('Gravity bends');
    await wait(1000); // a fourth poll a second later, not 3 s, while the answer grows
    await waitFor(() => expect(screen.getByLabelText('Answer so far')).toHaveTextContent('Gravity bends light'));
    await wait(1000);
    expect(await screen.findByText('Gravity bends light, a lot.')).toBeInTheDocument();
    expect(screen.queryByLabelText('Answer so far')).not.toBeInTheDocument();
  });

  it('shows a finished general result as inert text, without recorded evidence, and stops polling', async () => {
    const net = jobServer();
    net.replies.push({
      status: 'completed',
      finished_at: 1700000100,
      result: {
        answer: '<script>alert(1)</script>',
        status: 'partial',
        coverage: { partial: true, available: 8, included: 5 },
        warnings: ['missing camera'],
        evidence: [{ id: 1, timestamp: 1700000000, text: 'SECRET RECORDING', href: '/review#t=1700000000' }],
        panel: [{ model: 'x', error: 'unavailable' }],
      },
    });
    renderAsk();
    await user.type(question(), 'Q');
    await askIn('Friends'); // a general question
    await waitFor(() => expect(status()).toHaveTextContent('Friends · queued'));
    await wait(1000);
    expect(await screen.findByText('<script>alert(1)</script>')).toBeInTheDocument();
    const section = screen.getByRole('region', { name: 'Ask' });
    expect(section.querySelector('script, img, a')).toBeNull();
    expect(section).toHaveTextContent('Partial answer');
    expect(section).not.toHaveTextContent(/SECRET RECORDING|Evidence \(|partial recorded evidence|History coverage/);
    expect(status()).toHaveTextContent(/^Friends · completed · /);
    expect(status().querySelector('time')).toHaveAttribute('datetime', '2023-11-14T22:15:00.000Z');
    expect(screen.getByRole('listitem')).toHaveTextContent(/^missing camera$/);
    expect(cards()).toHaveLength(1);
    expect(cards()[0]).toHaveTextContent('Unavailable');
    expect(buttons().every(b => !(b as HTMLButtonElement).disabled)).toBe(true);
    expect(saved(GENERAL)).toEqual({ active: null, question: 'Q' });
    await wait(10_000);
    expect(net.polls).toEqual(['j1']);
  });

  it("reports a failed job's structured error", async () => {
    const net = jobServer();
    net.replies.push({ status: 'failed', error: { message: 'Structured job failure' } });
    renderAsk();
    await ask('Q');
    await queued();
    await wait(1000);
    await waitFor(() => expect(alertText()).toBe('Structured job failure'));
    expect(status()).toHaveTextContent(/^My recordings · failed$/);
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
    expect(status()).toHaveTextContent(/^My recordings · interrupted or expired$/);
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
    expect(status()).toHaveTextContent('My recordings · submission not confirmed');
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
});

describe('restoring a saved job', () => {
  const general = {
    id: 'g1',
    poll_url: '/pack-ask/jobs/g1',
    mode: 'friends',
    scope: 'general',
    question: 'Q',
    created_at: 1,
  };

  it('recovers a running job after a reload without resubmitting, then shows its end', async () => {
    const net = jobServer();
    store(GENERAL, { active: general, question: 'Q draft' });
    renderAsk();
    expect(status()).toHaveTextContent('Friends · recovering job g1');
    expect(question()).toHaveValue('Q draft');
    expect(buttons().every(b => (b as HTMLButtonElement).disabled)).toBe(true);
    expect(scope('Friends')).toBeChecked();
    net.replies.push({ status: 'failed', error: 'provider_unavailable' });
    await wait(1000);
    await waitFor(() => expect(alertText()).toBe('This model is currently unavailable.'));
    expect(net.posts).toEqual([]);
    expect(net.polls).toEqual(['g1']);
    expect(saved(GENERAL)).toEqual({ active: null, question: 'Q draft' });
  });

  it('recovers a local recordings job on the single Ask too', async () => {
    const net = jobServer();
    store(`${GENERAL}.ask`, { active: { ...general, mode: 'local', scope: 'history' }, question: 'Q' });
    net.replies.push({ status: 'completed', result: { answer: 'General recovered', is_history: false } });
    renderAsk({});
    expect(status()).toHaveTextContent('Ask · recovering job g1');
    await wait(1000);
    expect(await screen.findByText('General recovered')).toBeInTheDocument();
  });

  it.each([
    ['a friends job with recordings', { ...general, scope: 'history' }],
    ['a job saved before scopes', { ...general, scope: undefined }],
    ['an older local general job', { ...general, mode: 'local' }],
  ])('never resumes %s on Live', async (_, active) => {
    const net = jobServer();
    store(GENERAL, { active, question: 'Q' });
    renderAsk();
    expect(status()).toHaveTextContent(
      'A previous question of a different kind was not resumed. Your question is preserved.',
    );
    expect(question()).toHaveValue('Q');
    expect(buttons().every(b => !(b as HTMLButtonElement).disabled)).toBe(true);
    await wait(5000);
    expect(net.polls).toEqual([]);
  });

  it.each([
    ['Live', { showModes: true }, 'pack-ask.job.v1'],
    ['the single Ask', {}, 'pack-ask.job.v1.ask'],
  ])('keeps an old draft on %s but never its old job', async (_, props, legacy) => {
    const net = jobServer();
    const old = {
      active: {
        id: 'old',
        poll_url: '/pack-ask/jobs/old',
        mode: 'local',
        question: 'What happened today?',
        created_at: 1,
      },
      question: 'What happened today?',
    };
    store(legacy, old);
    renderAsk(props);
    expect(question()).toHaveValue('What happened today?');
    expect(status()).toHaveTextContent(/[Pp]revious draft kept; nothing was sent$/);
    await wait(5000);
    expect(net.polls).toEqual([]);
    expect(saved(legacy)).toEqual(old);
  });

  it('prefers the general draft to an old one', () => {
    store('pack-ask.job.v1', { active: null, question: 'Old draft' });
    store(GENERAL, { active: null, question: 'New general draft' });
    renderAsk();
    expect(question()).toHaveValue('New general draft');
    expect(status()).toHaveTextContent(/^$/);
  });

  it('saves the draft as it is typed', async () => {
    renderAsk({});
    await user.type(question(), 'Half a thought');
    expect(saved(`${GENERAL}.ask`)).toEqual({ active: null, question: 'Half a thought' });
    expect(saved(GENERAL)).toBeNull();
    expect(saved('pack-ask.job.v1.ask')).toBeNull();
  });
});

describe('Review: a question about one day', () => {
  const day = '2026-10-20';

  it('has one plain Ask and day-local copy', () => {
    renderAsk({ day });
    const section = screen.getByRole('region', { name: 'Ask about recordings' });
    expect(within(section).getByRole('heading', { name: 'Ask about video & conversations' })).toBeInTheDocument();
    expect(names()).toEqual(['Ask', 'Research']);
    expect(buttons()[0]).not.toHaveAttribute('aria-pressed');
    const input = screen.getByLabelText('What would you like to know?');
    expect(input).toHaveAccessibleDescription(
      'Searches recordings and raw transcripts for 2026-10-20. An explicit time such as "this week" overrides the selected day. Enter asks · Shift+Enter adds a line.',
    );
    expect(status()).toHaveTextContent(/^$/);
    expect(section).not.toHaveTextContent(/Ask local|friends|everyone|local selected/i);
  });

  it('sends the day with a history scope and saves under that day only', async () => {
    const net = jobServer();
    renderAsk({ day });
    await ask('How long were lights on?');
    await waitFor(() =>
      expect(net.posts).toEqual([{ question: 'How long were lights on?', mode: 'local', scope: 'history', day }]),
    );
    await waitFor(() => expect(status()).toHaveTextContent('Ask · queued'));
    expect(saved(REVIEW)).toMatchObject({ active: { id: 'j1', scope: 'history', mode: 'local' } });
    expect(saved(GENERAL)).toBeNull();
    expect(saved('pack-ask.job.v1')).toBeNull();
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
        status: 'partial',
        coverage: { partial: true, available: 8, included: 5 },
        evidence,
      },
    });
    renderAsk({ day });
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
    renderAsk({ day });
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
    expect(tag).toHaveTextContent(/^New$/);
    expect(tag).toHaveAttribute('href', '/settings#set-recall.enabled');
    await user.click(tag);
    expect(location.pathname + location.hash).toBe('/settings#set-recall.enabled');
  });

  it('tags journal, facts, timeline, detections, and forgotten answers, and nothing else', async () => {
    const net = jobServer();
    const tags = (root: HTMLElement) =>
      within(root)
        .queryAllByRole('link', { name: /^New:/ })
        .map(a => a.getAttribute('href'));
    renderAsk({ day });
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
    expect(answer).toHaveTextContent('From the timeline: who was seen and heard. New');
    expect(evidenceCards().map(tags)).toEqual([['/settings#set-journal.enabled'], ['/settings#set-journal.facts'], []]);

    net.replies.push({
      status: 'completed',
      result: {
        answer: 'Dana was last seen at 10:00.',
        paths: ['detections'],
        evidence: [{ id: 'C1', source: 'camera' }],
      },
    });
    await again();
    await screen.findByText('Dana was last seen at 10:00.');
    expect(tags(document.body)).toEqual(['/settings#set-detections.last_seen']);

    net.replies.push({
      status: 'completed',
      result: { answer: 'Nobody.', evidence: [{ id: 'C1', source: 'camera' }] },
    });
    await again();
    await screen.findByText('Nobody.');
    expect(tags(document.body)).toEqual([]);

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
    renderAsk({ day });
    await ask('Q');
    await queued();
    await wait(1000);
    const card = (await screen.findAllByRole('article'))[0] as HTMLElement;
    await user.click(within(card).getByRole('link'));
    expect(location.pathname + location.hash).toBe('/review#day=2026-10-20&t=1792526400');
    await user.click(screen.getByRole('link', { name: 'the moment' }));
    await waitFor(() => expect(location.hash).toBe('#t=1792526400'));
    await waitFor(() => expect(changes).toEqual(['#day=2026-10-20&t=1792526400', '#t=1792526400']));
    window.removeEventListener('hashchange', listen);
  });

  it('recovers its own job, including one saved before scopes existed', async () => {
    const net = jobServer();
    const active = { id: 'review', poll_url: '/pack-ask/jobs/review', mode: 'local', question: 'Q', created_at: 1 };
    store(REVIEW, { active, question: 'How long were lights on?' });
    net.replies.push({ status: 'completed', result: { answer: 'Recovered evidence answer.' } });
    renderAsk({ day });
    expect(status()).toHaveTextContent('Ask · recovering job review');
    expect(question()).toHaveValue('How long were lights on?');
    await wait(1000);
    expect(await screen.findByText('Recovered evidence answer.')).toBeInTheDocument();
    expect(net.polls).toEqual(['review']);
    expect(net.posts).toEqual([]);
  });

  it.each(['friends', 'everyone'])('does not resume an old %s job and leaves it saved', async mode => {
    const net = jobServer();
    const old = {
      active: { id: 'review', poll_url: '/pack-ask/jobs/review', mode, scope: 'history', question: 'Q', created_at: 1 },
      question: 'Q',
    };
    store(REVIEW, old);
    renderAsk({ day });
    expect(status()).toHaveTextContent(
      'Previous multi-model question not resumed. Your question is preserved; mode choices are on Live.',
    );
    expect(names()).toEqual(['Ask', 'Research']);
    expect(buttons()[0]).toBeEnabled();
    await wait(5000);
    expect(net.polls).toEqual([]);
    expect(saved(REVIEW)).toEqual(old);
    await user.click(buttons()[0] as HTMLElement);
    await waitFor(() => expect(net.posts).toEqual([{ question: 'Q', mode: 'local', scope: 'history', day }]));
  });

  it('never resumes a general job', async () => {
    const net = jobServer();
    store(REVIEW, {
      active: {
        id: 'g1',
        poll_url: '/pack-ask/jobs/g1',
        mode: 'local',
        scope: 'general',
        question: 'Q',
        created_at: 1,
      },
      question: 'Q',
    });
    renderAsk({ day });
    expect(status()).toHaveTextContent('different kind was not resumed');
    await wait(5000);
    expect(net.polls).toEqual([]);
  });
});

describe('scope isolation between Live and Review', () => {
  it('never shares a question or a job between Review days and Live', async () => {
    const net = jobServer();
    const review = renderPage(<PackAsk day="2026-10-20" />);
    await ask('who visited?');
    await waitFor(() =>
      expect(net.posts.at(-1)).toEqual({
        question: 'who visited?',
        mode: 'local',
        scope: 'history',
        day: '2026-10-20',
      }),
    );
    review.unmount();

    const live = renderPage(<PackAsk showModes />);
    expect(question()).toHaveValue('');
    await user.type(question(), 'explain tides');
    await askIn('Friends');
    await waitFor(() =>
      expect(net.posts.at(-1)).toEqual({ question: 'explain tides', mode: 'friends', scope: 'general' }),
    );
    live.unmount();

    renderPage(<PackAsk day="2026-10-21" />);
    expect(question()).toHaveValue('');
    expect(status()).not.toHaveTextContent('recovering job');
  });

  it('starts clean when the day changes under a mounted widget', async () => {
    jobServer();
    const view = renderPage(<PackAsk day="2026-10-20" />);
    await ask('who visited?');
    await waitFor(() => expect(status()).toHaveTextContent('Ask · queued'));
    view.rerender(
      <Providers client={view.client} test>
        <PackAsk day="2026-10-21" />
      </Providers>,
    );
    expect(question()).toHaveValue('');
    expect(status()).toHaveTextContent(/^$/);
    expect(buttons()[0]).toBeEnabled();
  });
});
