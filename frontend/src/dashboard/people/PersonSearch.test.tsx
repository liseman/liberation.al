import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { PersonSearch } from './PersonSearch';

it('refines context, previews/backtracks, and only explicitly selects a candidate', async () => {
  const queries: unknown[] = [];
  server.use(
    http.post('/people/search', async ({ request }) => {
      queries.push(await request.json());
      return HttpResponse.json({
        candidates: [
          {
            name: 'Dana North',
            detail: 'Engineer at Example, Seattle',
            url: 'https://example.com/north',
            source_title: 'Example team biography',
            provider: 'Brave Web Search',
          },
          { name: 'Dana South', detail: 'Artist, Boston', url: 'https://example.com/south' },
        ],
      });
    }),
  );
  const selected = vi.fn();
  const user = userEvent.setup();
  renderPage(<PersonSearch name="Dana" onSelect={selected} />);
  await user.type(screen.getByLabelText('Affiliation, location, or other public context'), 'Engineer');
  await user.click(screen.getByRole('button', { name: 'Search / refine' }));
  await user.click(await screen.findByRole('button', { name: 'Preview Example team biography' }));
  expect(selected).not.toHaveBeenCalled();
  expect(screen.getByRole('link', { name: 'Check source' })).toHaveAttribute('href', 'https://example.com/north');
  await user.click(screen.getByRole('button', { name: 'Back to candidates' }));
  await user.type(screen.getByLabelText('Affiliation, location, or other public context'), ' Seattle');
  await user.click(screen.getByRole('button', { name: 'Search / refine' }));
  await user.click(await screen.findByRole('button', { name: 'Preview Example team biography' }));
  await user.click(screen.getByRole('button', { name: 'Use this name' }));
  expect(selected).toHaveBeenCalledExactlyOnceWith('Dana North');
  expect(queries).toEqual([
    { name: 'Dana', context: 'Engineer' },
    { name: 'Dana', context: 'Engineer Seattle' },
  ]);
});

it('ignores a delayed response after context refinement', async () => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => {
    release = resolve;
  });
  server.use(
    http.post('/people/search', async () => {
      await gate;
      return HttpResponse.json({ candidates: [{ name: 'Stale', detail: '', url: 'https://example.com' }] });
    }),
  );
  const user = userEvent.setup();
  renderPage(<PersonSearch name="Dana" onSelect={vi.fn()} />);
  await user.click(screen.getByRole('button', { name: 'Search / refine' }));
  await user.type(screen.getByLabelText('Affiliation, location, or other public context'), 'Boston');
  await act(async () => {
    release();
    await gate;
  });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Search / refine' })).toBeEnabled());
  expect(screen.queryByText('Stale')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Affiliation, location, or other public context')).toHaveValue('Boston');
});

it('keeps refinement input on failure and permits retry', async () => {
  server.use(http.post('/people/search', () => HttpResponse.json({ error: 'Unavailable' })));
  const user = userEvent.setup();
  renderPage(<PersonSearch name="Dana" onSelect={vi.fn()} />);
  await user.type(screen.getByLabelText('Affiliation, location, or other public context'), 'Boston');
  await user.click(screen.getByRole('button', { name: 'Search / refine' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Unavailable');
  expect(screen.getByLabelText('Affiliation, location, or other public context')).toHaveValue('Boston');
  expect(screen.getByRole('button', { name: 'Search / refine' })).toBeEnabled();
});
