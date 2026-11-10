import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { History } from './History';

it('sends the selected local minute as the undo cutoff, with confirmation first', async () => {
  const noon = new Date(2026, 10, 4, 12, 0).getTime() / 1000;
  vi.spyOn(Date, 'now').mockReturnValue(new Date(2026, 10, 4, 12, 45, 59).getTime());
  server.use(
    http.get('/speakers/refine/log', () =>
      HttpResponse.json({
        total: 2,
        labels: [
          {
            ts: noon,
            ts_start: noon - 10,
            text: 'First label',
            new: { speaker: 'Alice' },
            old: { speaker: 'unknown' },
          },
          {
            ts: noon + 1800,
            ts_start: noon + 1790,
            text: 'Second label',
            new: { speaker: 'Dana' },
            old: { speaker: 'unknown' },
          },
        ],
      }),
    ),
    http.get('/inbox/skips', () => HttpResponse.json({ face: 0, voice: 0, line: 0 })),
  );
  const undo = vi.fn().mockResolvedValue(true);
  renderPage(<History onUndo={undo} onUnskip={() => {}} />);
  await userEvent.click(screen.getByRole('button', { name: 'Recent decisions' }));
  const picker = await screen.findByRole('button', { name: 'Undo labels given after' });
  expect(picker).toHaveTextContent('Nov 4, 2026 12:00');
  await userEvent.click(picker);
  await userEvent.clear(screen.getByLabelText('Minutes'));
  await userEvent.type(screen.getByLabelText('Minutes'), '15');
  await userEvent.click(screen.getByRole('button', { name: 'Use cutoff' }));
  await userEvent.click(screen.getByRole('button', { name: 'Undo them' }));
  expect(undo).not.toHaveBeenCalled();
  expect(screen.getByText('Undo 1 label?')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Undo them' }));
  await waitFor(() => expect(undo).toHaveBeenCalledExactlyOnceWith({ since: noon + 900 }));
});
