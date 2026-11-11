import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { citedMoments } from '../ask/moments';
import { openAsk, useAsk, useAskPage } from '../ask/openAsk';
import { PackAsk } from '../ask/PackAsk';
import { jobServer } from '../ask/testing';
import { hm } from '../lib/format';
import { useRoute } from '../router';

afterEach(() => {
  vi.unstubAllGlobals();
  useAskPage.setState({ day: null, seek: null, mark: null });
  useAsk.setState({ open: false, request: null });
});

const DAY = { day: '2026-10-27', label: 'Tue 27 Oct', today: false };

describe('citedMoments', () => {
  it('chips the cited records in time order, one per minute and source, by camera view or speaker', () => {
    vi.stubGlobal('NAMES', { views: { cam0: 'front camera' } });
    const result = {
      answer: 'Dana came in [T2], the dog left [C1, C3], and later [C4].',
      evidence: [
        { id: 'C1', ts: 2000, camera: 'cam0' },
        { id: 'T2', ts: 1000, speaker: 'Dana' },
        { id: 'C3', ts: 2010, camera: 'cam0' }, // the same minute and camera as C1
        { id: 'C4', ts: 3000 }, // no camera or speaker
        { id: 'C5', ts: 4000, camera: 'cam0' }, // not cited
      ],
    };
    expect(citedMoments(result)).toEqual([
      { t: 1000, label: `${hm(1000)} Dana` },
      { t: 2000, label: `${hm(2000)} front camera` },
      { t: 3000, label: hm(3000) },
    ]);
  });

  it('keeps at most eight, and none for a result with no answer', () => {
    const evidence = Array.from({ length: 12 }, (_, i) => ({ id: `C${i + 1}`, ts: i * 600, camera: 'cam0' }));
    expect(citedMoments({ answer: '[C1-C12]', evidence })).toHaveLength(8);
    expect(citedMoments(null)).toEqual([]);
  });
});

describe("Ask on Review's day", () => {
  const answered = () => {
    const jobs = jobServer();
    jobs.replies.push({
      status: 'completed',
      question: 'When did the dog leave?',
      created_at: 5000,
      day: '2026-10-27',
      result: {
        status: 'ok',
        scope: 'history',
        answer: 'At [C2] and [C1].',
        evidence: [
          { id: 'C1', ts: 3000, camera: 'cam0' },
          { id: 'C2', ts: 1200, camera: 'cam0' },
        ],
      },
    });
    return jobs;
  };

  /** The one Ask on Review, opened on its day, with Review's player and timeline as fakes. */
  function onReview() {
    const seek = vi.fn();
    const mark = vi.fn();
    useRoute.setState({ tab: 'review' });
    useAskPage.setState({ day: DAY, seek, mark });
    const view = renderPage(<PackAsk />);
    act(() => openAsk({ hint: { kind: 'day', ...DAY } }));
    return { seek, mark, view };
  }

  it("marks the newest answer's cited moments on the timeline, plays one from its chip, and clears them with the hint", async () => {
    sessionStorage.clear();
    answered();
    const { seek, mark } = onReview();
    await screen.findByText('Tue 27 Oct');
    await userEvent.type(screen.getByRole('textbox', { name: 'Your question' }), 'When did the dog leave?{Enter}');
    const chips = await screen.findByRole('list', { name: 'Moments it cites' }, { timeout: 3000 });
    await waitFor(() => expect(mark).toHaveBeenLastCalledWith([1200, 3000]));
    await userEvent.click(within(chips).getAllByRole('button')[0] as HTMLElement);
    expect(seek).toHaveBeenCalledWith(1200);
    await userEvent.click(screen.getByRole('button', { name: 'Stop asking about Tue 27 Oct' }));
    await waitFor(() => expect(mark).toHaveBeenLastCalledWith([]));
    expect(screen.queryByRole('list', { name: 'Moments it cites' })).not.toBeInTheDocument();
  });

  it("offers the day's questions to start from, and asks one about the day when picked", async () => {
    sessionStorage.clear();
    const jobs = answered();
    onReview();
    const start = await screen.findByRole('list', { name: 'Suggested questions' });
    await userEvent.click(await within(start).findByRole('button', { name: 'Who was here that day?' }));
    await waitFor(() => expect(jobs.posts).toEqual([{ question: 'Who was here that day?', day: '2026-10-27' }]));
    expect(screen.queryByRole('list', { name: 'Suggested questions' })).not.toBeInTheDocument();
  });
});
