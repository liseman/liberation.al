import { screen, waitFor, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createStore } from 'zustand/vanilla';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { EngineContext } from '../review/context';
import type { ReviewEngine, ReviewUi } from '../review/engine';
import { citedRecords } from './MarkAnswer';
import { PackAsk } from './PackAsk';
import { fakeTimers, jobServer, renderAsk, wait } from './testing';

const DAY = '2026-10-20';
const ASKED = 1792540800.25;
const Q = 'What did Dana say about the garden?';

let user: ReturnType<typeof fakeTimers>;
let marks: unknown[];
beforeEach(() => {
  sessionStorage.clear();
  user = fakeTimers();
  marks = [];
  server.use(
    http.post('/measurement/mark', async ({ request }) => {
      marks.push(await request.json());
      return HttpResponse.json({ ok: true });
    }),
  );
});
afterEach(() => {
  vi.useRealTimers();
});

/** Ask Q and let the job finish with result. */
async function answered(result: Record<string, unknown>, job: Record<string, unknown> = {}) {
  const jobs = jobServer();
  jobs.replies.push({
    status: 'completed',
    question: Q,
    created_at: ASKED,
    day: DAY,
    result: { status: 'ok', scope: 'history', evidence: [], ...result },
    ...job,
  });
  await user.type(screen.getByRole('textbox', { name: /what would you like/i }), Q);
  await user.keyboard('{Enter}');
  await wait(1000); // the first poll falls due; its reply arrives later
  // The Ask button comes back in the same render as the answer, so the answer (and its marks) are on the page.
  await waitFor(() => expect(screen.getByRole('button', { name: 'Ask' })).toBeEnabled());
}

const mark = () => screen.queryByRole('region', { name: 'Mark this answer' });
const button = (name: RegExp) => within(mark() as HTMLElement).getByRole('button', { name });

describe('citedRecords', () => {
  it('keeps only the records the answer cites, each as its time and a source tools/measure.py knows', () => {
    const result = {
      answer: 'Dana planted tomatoes [T1, J2] and said so again [T4].',
      evidence: [
        { id: 'T1', ts: 100, source: 'transcript' },
        { id: 'J2', ts: 200, kind: 'journal', source: 'recall' },
        { id: 'C3', ts: 300, source: 'camera' }, // not cited
        { id: 'T4', ts: 400, source: 'detections' }, // a source measure doesn't check
        { id: 'T5', ts: 'noon', source: 'transcript' },
      ],
    };
    expect(citedRecords(result)).toEqual([
      { ts: 100, source: 'transcript' },
      { ts: 200, source: 'journal' },
      { ts: 400 },
    ]);
  });

  it('reads a cited range, and keeps each record once', () => {
    const result = {
      answer: 'Dana spoke three times [T1-T3], once again [T2].',
      evidence: [
        { id: 'T1', ts: 100, source: 'transcript' },
        { id: 'T2', ts: 200, source: 'transcript' },
        { id: 'T3', ts: 300, source: 'transcript' },
        { id: 'T4', ts: 300, source: 'transcript' }, // the same line again, under another id
        { id: 'T5', ts: 500, source: 'transcript' },
      ],
    };
    expect(citedRecords({ ...result, answer: `${result.answer} [T4]` })).toEqual([
      { ts: 100, source: 'transcript' },
      { ts: 200, source: 'transcript' },
      { ts: 300, source: 'transcript' },
    ]);
  });
});

describe('marking a Review answer', () => {
  it('saves a right answer with the records it cited', async () => {
    renderAsk({ day: DAY });
    await answered({
      answer: 'Dana said to plant tomatoes [T1].',
      evidence: [
        { id: 'T1', ts: 1792530005, source: 'transcript' },
        { id: 'C2', ts: 1792530100, source: 'camera' },
      ],
    });
    await user.click(button(/^yes$/i));
    await waitFor(() => expect(marks).toHaveLength(1));
    expect(marks[0]).toEqual({
      kind: 'answer',
      question: Q,
      asked_at: ASKED,
      day: DAY,
      right: true,
      cited: [{ ts: 1792530005, source: 'transcript' }],
    });
    expect(await within(mark() as HTMLElement).findByRole('status')).toHaveTextContent(
      'Marked right: saved the question with the record it cited.',
    );
  });

  it('dates the label from when the answer started, not when it was queued', async () => {
    renderAsk({ day: DAY });
    await answered({ answer: 'Nothing was said.' }, { started_at: ASKED + 90 });
    await user.click(button(/^no, correct it$/i));
    await user.type(screen.getByRole('textbox', { name: /a right answer says/i }), 'Dana');
    await user.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(marks).toHaveLength(1));
    expect(marks[0]).toMatchObject({ asked_at: ASKED + 90 });
  });

  it('saves Yes at once, with no form, when the answer cited nothing', async () => {
    renderAsk({ day: DAY });
    await answered({ answer: 'Dana was last seen at 3 pm.' });
    await user.click(button(/^yes$/i));
    await waitFor(() => expect(marks).toHaveLength(1));
    expect(marks[0]).toMatchObject({ right: true });
    expect(marks[0]).not.toHaveProperty('cited');
    expect(marks[0]).not.toHaveProperty('says');
    expect(screen.queryByRole('textbox', { name: /a right answer says/i })).toBeNull();
  });

  it("saves a wrong answer's correction and never its citations", async () => {
    renderAsk({ day: DAY });
    await answered({
      answer: 'Dana said to plant tulips [T1].',
      evidence: [{ id: 'T1', ts: 1792530005, source: 'transcript' }],
    });
    await user.click(button(/^no, correct it$/i));
    await user.type(screen.getByRole('textbox', { name: /a right answer says/i }), 'tomatoes');
    await user.type(screen.getByRole('textbox', { name: /shouldn't say/i }), 'tulips');
    await user.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(marks).toHaveLength(1));
    expect(marks[0]).toEqual({
      kind: 'answer',
      question: Q,
      asked_at: ASKED,
      day: DAY,
      right: false,
      says: 'tomatoes',
      not_says: 'tulips',
    });
    expect(await within(mark() as HTMLElement).findByRole('status')).toHaveTextContent(
      "Marked wrong: saved the question with what a right answer says and what it shouldn't say.",
    );
  });

  it('can say a wrong answer should cite the moment Review is on', async () => {
    const ui = createStore<Pick<ReviewUi, 't'>>(() => ({ t: 1792530005 }));
    const engine = { ui } as unknown as ReviewEngine;
    renderPage(
      <EngineContext value={engine}>
        <PackAsk day={DAY} />
      </EngineContext>,
    );
    await answered({ answer: 'Nobody spoke.' });
    await user.click(button(/^no, correct it$/i));
    const box = screen.getByRole('checkbox', { name: /where review is now/i });
    await user.click(box);
    ui.setState({ t: 1792539999 }); // the playhead moves on: the picked moment stays
    await user.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(marks).toHaveLength(1));
    expect(marks[0]).toMatchObject({ right: false, moment: 1792530005 });
  });

  it("shows the server's refusal and keeps the correction open", async () => {
    server.use(http.post('/measurement/mark', () => HttpResponse.json({ ok: false, error: 'nope' }, { status: 403 })));
    renderAsk({ day: DAY });
    await answered({ answer: 'Nobody spoke.' });
    await user.click(button(/^no, correct it$/i));
    await user.type(screen.getByRole('textbox', { name: /a right answer says/i }), 'Dana');
    await user.click(screen.getByRole('button', { name: 'Save' }));
    expect(await within(mark() as HTMLElement).findByRole('alert')).toHaveTextContent('nope');
    expect(screen.getByRole('textbox', { name: /a right answer says/i })).toHaveValue('Dana');
  });

  it('offers no mark on a failed answer or one that is not about recordings', async () => {
    renderAsk({ day: DAY });
    await answered({ answer: 'Question processing failed.', status: 'error' }, { status: 'failed' });
    expect(mark()).toBeNull();
    await answered({ answer: 'Paris.', is_history: false });
    expect(screen.getByText('Paris.')).toBeInTheDocument();
    expect(mark()).toBeNull();
  });
});

describe('a general question', () => {
  it('offers no mark, since the measurement set only asks about recordings', async () => {
    renderAsk({ showModes: true });
    const jobs = jobServer();
    jobs.replies.push({ status: 'completed', question: Q, created_at: ASKED, result: { status: 'ok', answer: 'Hi.' } });
    await user.type(screen.getByRole('textbox'), Q);
    await user.click(screen.getByRole('radio', { name: 'Friends' })); // friends is general; local searches (#2307)
    await user.click(screen.getByRole('button', { name: 'Ask' }));
    await waitFor(() => expect(jobs.posts).toHaveLength(1));
    await wait(1000);
    expect(screen.getByText('Hi.')).toBeInTheDocument();
    expect(mark()).toBeNull();
  });
});
