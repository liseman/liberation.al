import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { citedMoments } from '../ask/moments';
import { PackAsk } from '../ask/PackAsk';
import { jobServer } from '../ask/testing';
import { hm } from '../lib/format';

afterEach(() => vi.unstubAllGlobals());

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

describe("Review's Ask (the one PackAsk, scoped to the day)", () => {
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

  it("marks the newest answer's cited moments on the timeline, plays one from its chip, and clears them when it goes", async () => {
    answered();
    const onMarks = vi.fn();
    const onSeek = vi.fn();
    const view = renderPage(
      <PackAsk day="2026-10-27" bare compact dayLabel="Tue 27 Oct" today={false} onSeek={onSeek} onMarks={onMarks} />,
    );
    await userEvent.type(
      screen.getByRole('textbox', { name: 'Ask about Tue 27 Oct' }),
      'When did the dog leave?{Enter}',
    );
    const chips = await screen.findByRole('list', { name: 'Moments it cites' }, { timeout: 3000 });
    await waitFor(() => expect(onMarks).toHaveBeenLastCalledWith([1200, 3000]));
    await userEvent.click(within(chips).getAllByRole('button')[0] as HTMLElement);
    expect(onSeek).toHaveBeenCalledWith(1200);
    view.unmount();
    expect(onMarks).toHaveBeenLastCalledWith([]);
  });

  it('offers questions to start from, and asks one when picked, with Research beside Ask', async () => {
    const jobs = answered();
    renderPage(<PackAsk day="2026-10-27" bare compact dayLabel="Tue 27 Oct" today={false} onSeek={() => {}} />);
    expect(screen.getByRole('button', { name: 'Research' })).toBeInTheDocument();
    const start = screen.getByRole('list', { name: 'Suggested questions' });
    await userEvent.click(within(start).getByRole('button', { name: 'Who was here that day?' }));
    await waitFor(() =>
      expect(jobs.posts).toEqual([
        { question: 'Who was here that day?', mode: 'local', scope: 'history', day: '2026-10-27' },
      ]),
    );
    expect(screen.queryByRole('list', { name: 'Suggested questions' })).not.toBeInTheDocument();
  });
});
