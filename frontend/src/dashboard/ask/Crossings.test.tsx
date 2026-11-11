import { screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fakeTimers, jobServer, question, renderAsk, status, wait } from './testing';

let user: ReturnType<typeof fakeTimers>;
beforeEach(() => {
  sessionStorage.clear();
  history.replaceState(null, '', '/');
  user = fakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

const result = {
  partial: true,
  cross_camera: 'Camera counts are shown separately and are never summed.',
  notice: 'Walked past counts line crossings, not distinct people.',
  cameras: {
    cam0: {
      line_description: 'default vertical line at 50% of the frame width',
      crossings: { left_to_right: 2, right_to_left: 1, total: 3 },
      scanned_seconds: 3600,
      coverage_of_requested: 0.5,
      sample_rates_hz: [5],
      camera_motion: { uncertain: true, flagged_files: ['x.mp4', 'y.mp4'] },
      histogram: [{ start: 1700000000, end: 1700003600, count: 3 }],
      examples: [{ ts: 1700000200, direction: 'left_to_right', link: '/review#day=2026-10-20&t=1700000200' }],
      examples_truncated: false,
      unrecorded_or_unscanned_intervals: [[1700003600, 1700007200]],
    },
  },
};

/** Ask a Review question whose answer carries the walk-by count as the server gives it. */
async function askForCrossings(pathBValue: Record<string, unknown>) {
  const net = jobServer();
  net.replies.push({
    status: 'completed',
    result: {
      answer: 'cam0: people walked past about 3 times.',
      scope: 'history',
      is_history: true,
      crossings: { path_b: pathBValue },
    },
  });
  renderAsk();
  await user.type(question(), 'how many times did someone walk by');
  await user.keyboard('{Enter}');
  await waitFor(() => expect(status()).toHaveTextContent(/^Looking through your recordings/));
  await wait(1000);
}

const done = { status: 'done', result, answer: 'cam0: people walked past about 3 times.' };
const pathB = () => screen.getByRole('heading', { name: 'Walked past' }).closest('section') as HTMLElement;

describe('crossing answers', () => {
  it('shows the line crossings per camera with the answer', async () => {
    await askForCrossings(done);
    await waitFor(() => expect(within(pathB()).getByText('Walked past 3 times (estimate)')).toBeInTheDocument());
    const b = pathB();
    expect(b).toHaveTextContent('left to right: 2 · right to left: 1');
    expect(b).toHaveTextContent('Line: default vertical line at 50% of the frame width');
    expect(b).toHaveTextContent('Scanned 60 min of the requested window (50%), sampled at 5 frames per second.');
    expect(b).toHaveTextContent(
      'Camera motion was detected in 2 of these crossings, so a fixed line is unreliable there. Treat this count with extra caution.',
    );
    expect(b).toHaveTextContent("Partial: see-actions wasn't watching for some of the requested window");
    expect(b).toHaveTextContent('never summed');
    expect(b).toHaveTextContent('cam0: people walked past about 3 times.');
    await user.click(within(b).getByText('Times walked past (1)'));
    expect(within(b).getByRole('link')).toHaveAttribute('href', '/review#day=2026-10-20&t=1700000200');
    await user.click(within(b).getByText('Unscanned or unrecorded intervals (1)'));
  });

  it('says when the line-crossing count is unavailable', async () => {
    await askForCrossings({ status: 'unavailable', reason: 'the broker is down' });
    expect(await screen.findByText('Walked-past count unavailable: the broker is down')).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Walked past' })).not.toBeInTheDocument();
  });
});

it('shows no caption estimate', async () => {
  await askForCrossings(done);
  await waitFor(() => expect(pathB()).toBeInTheDocument());
  expect(screen.queryByText(/caption/i)).not.toBeInTheDocument();
});
