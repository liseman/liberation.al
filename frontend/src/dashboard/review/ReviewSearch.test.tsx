import { act, cleanup, fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { Providers } from '../../shared/Providers';
import { renderPage } from '../../test/render';
import type { Block } from './api';
import { ReviewSearch, SEMANTIC_DELAY_MS } from './ReviewSearch';

const DAY = '2026-10-27';
const ENTRIES = [{ ts_start: 100, speaker: 'Alice', text: 'Café after lunch' }];
const BLOCKS: Block[] = [{ start: 200, end: 220, text: 'Café at the window', tags: [] }];
let pending: ((response: Response) => void)[];

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  pending = [];
});
afterEach(async () => {
  cleanup();
  await act(async () => {
    for (const resolve of pending) resolve(Response.json({ results: [] }));
    if (vi.isFakeTimers()) await vi.advanceTimersByTimeAsync(0);
  });
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function show(props: Partial<Parameters<typeof ReviewSearch>[0]> = {}) {
  const seek = vi.fn();
  const searching = vi.fn();
  return {
    seek,
    searching,
    ...renderPage(
      <ReviewSearch day={DAY} entries={ENTRIES} blocks={BLOCKS} onSeek={seek} onSearching={searching} {...props} />,
    ),
  };
}
const input = () => screen.getByRole('combobox', { name: 'Search transcript and summary' });
const change = (value: string) => fireEvent.change(input(), { target: { value } });
const advance = (ms = 0) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });

it('searches both sources instantly, preserves accent highlights, and keeps player hotkeys out of navigation', async () => {
  const fetcher = vi.fn<typeof fetch>();
  vi.stubGlobal('fetch', fetcher);
  const { seek, searching } = show();
  change('cafe');
  const options = screen.getAllByRole('option');
  expect(options).toHaveLength(2);
  expect(input()).toHaveAttribute('aria-expanded', 'true');
  expect(input()).toHaveAttribute('aria-controls', screen.getByRole('listbox').id);
  expect(within(options[0] as HTMLElement).getByText('Café', { selector: 'mark' })).toBeInTheDocument();
  expect(options[1]).toHaveTextContent('Summary');
  const globalKey = vi.fn();
  window.addEventListener('keydown', globalKey);
  input().focus();
  try {
    fireEvent.keyDown(input(), { key: 'ArrowDown', code: 'ArrowDown' });
    expect(options[1]).toHaveAttribute('aria-selected', 'true');
    expect(input()).toHaveAttribute('aria-activedescendant', options[1]?.id);
    fireEvent.keyDown(input(), { key: 'Enter', code: 'Enter' });
    expect(seek).toHaveBeenLastCalledWith(200);
    fireEvent.keyDown(input(), { key: 'ArrowUp', code: 'ArrowUp' });
    fireEvent.keyDown(input(), { key: 'Enter', code: 'Enter' });
    expect(seek).toHaveBeenLastCalledWith(100);
    expect(input()).toHaveFocus();
    await advance(SEMANTIC_DELAY_MS * 2);
    expect(fetcher).not.toHaveBeenCalled();
    fireEvent.keyDown(input(), { key: 'Escape' });
    expect(input()).toHaveValue('');
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    expect(searching).toHaveBeenLastCalledWith('');
    expect(globalKey).not.toHaveBeenCalled();
  } finally {
    window.removeEventListener('keydown', globalKey);
  }
});

it('labels a transcript hit with the name the transcript shows, never a placeholder', () => {
  show({ entries: [{ ts_start: 100, speaker: 'Speaker 250', text: 'Broke it down into three' }] });
  change('three');
  expect(screen.getByRole('option')).toHaveTextContent('Unmatched: Broke it down into three');
  expect(screen.getByRole('option')).not.toHaveTextContent('Speaker 250');
});

it('debounces semantic search and aborts/rejects stale query and day responses', async () => {
  const calls: { url: string; signal?: AbortSignal | null }[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn<typeof fetch>(
      (url, init) =>
        new Promise<Response>(resolve => {
          calls.push({ url: String(url), signal: init?.signal });
          pending.push(resolve); // deliberately ignores abort, to exercise stale-response rejection too
        }),
    ),
  );
  const view = show({ entries: [], blocks: [] });
  change('distant planets');
  await advance(SEMANTIC_DELAY_MS - 1);
  expect(calls).toEqual([]);
  await advance(1);
  expect(calls[0]?.url).toContain('q=distant%20planets');
  change('galaxies');
  expect(calls[0]?.signal?.aborted).toBe(true);
  await advance(SEMANTIC_DELAY_MS);
  expect(calls).toHaveLength(2);
  await act(async () => {
    pending[0]?.(
      Response.json({ results: [{ id: 'old', kind: 'semantic', ts: 5, end: 8, text: 'Old query answer' }] }),
    );
    if (vi.isFakeTimers()) await vi.advanceTimersByTimeAsync(0);
  });
  expect(screen.queryByText('Old query answer')).not.toBeInTheDocument();
  view.rerender(
    <Providers client={view.client} test>
      <ReviewSearch day="2026-10-28" entries={[]} blocks={[]} onSeek={view.seek} />
    </Providers>,
  );
  expect(calls[1]?.signal?.aborted).toBe(true);
  await advance(SEMANTIC_DELAY_MS);
  expect(calls[2]?.url).toContain('day=2026-10-28');
  await act(async () => {
    pending[1]?.(
      Response.json({ results: [{ id: 'old-day', kind: 'semantic', ts: 6, end: 9, text: 'Old day answer' }] }),
    );
    pending[2]?.(
      Response.json({ results: [{ id: 'new-day', kind: 'semantic', ts: 7, end: 10, text: 'Current day answer' }] }),
    );
    if (vi.isFakeTimers()) await vi.advanceTimersByTimeAsync(0);
  });
  expect(screen.queryByText('Old day answer')).not.toBeInTheDocument();
  expect(screen.getByText('Current day answer')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('option'));
  expect(view.seek).toHaveBeenCalledWith(7);
});

it('shows local Ollama failures and retries by explicit intent without an automatic retry loop', async () => {
  let failed = true;
  const fetcher = vi.fn<typeof fetch>(() =>
    Promise.resolve(
      failed
        ? Response.json({ error: 'Local Ollama is unavailable' }, { status: 503 })
        : Response.json({ results: [{ id: 'match', kind: 'semantic', ts: 42, end: 43, text: 'Recovered result' }] }),
    ),
  );
  vi.stubGlobal('fetch', fetcher);
  show({ entries: [], blocks: [] });
  change('unrelated question');
  await advance(SEMANTIC_DELAY_MS);
  expect(screen.getByRole('alert')).toHaveTextContent('Local Ollama is unavailable');
  await advance(SEMANTIC_DELAY_MS * 3);
  expect(fetcher).toHaveBeenCalledTimes(1);
  failed = false;
  fireEvent.click(within(screen.getByRole('alert')).getByRole('button', { name: 'Retry' }));
  await advance();
  expect(screen.getByText('Recovered result')).toBeInTheDocument();
  expect(screen.getByRole('group', { name: 'Similar in meaning' })).toBeInTheDocument();
  expect(fetcher).toHaveBeenCalledTimes(2);
});

it('allows Enter on Search by meaning with fuzzy hits and while the full-day summary is pending', async () => {
  const fetcher = vi.fn<typeof fetch>(() =>
    Promise.resolve(
      Response.json({ results: [{ id: 'meaning', kind: 'semantic', ts: 300, end: 301, text: 'A related moment' }] }),
    ),
  );
  vi.stubGlobal('fetch', fetcher);
  show({ loading: true, summary: 'loading' });
  change('cafe');
  expect(screen.getByText(/Loading the full-day summary/)).toBeInTheDocument();
  await advance(SEMANTIC_DELAY_MS * 2);
  expect(fetcher).not.toHaveBeenCalled();
  screen.getByRole('button', { name: 'Search by meaning' }).focus();
  vi.useRealTimers();
  const user = userEvent.setup();
  await user.keyboard('{Enter}');
  await screen.findByText('A related moment');
  expect(screen.getByText('A related moment')).toBeInTheDocument();
  expect(screen.getAllByRole('option')).toHaveLength(3);
});

it('never starts semantic requests for punctuation alone or before initial local text is available', async () => {
  const fetcher = vi.fn<typeof fetch>();
  vi.stubGlobal('fetch', fetcher);
  show({ entries: [], blocks: [], loading: true, summary: 'error' });
  change('!!!');
  await advance(SEMANTIC_DELAY_MS * 2);
  expect(fetcher).not.toHaveBeenCalled();
  expect(screen.getByText(/summary could not be loaded/)).toBeInTheDocument();
  change('not loaded yet');
  await advance(SEMANTIC_DELAY_MS * 2);
  expect(fetcher).not.toHaveBeenCalled();
});

it('bounds pasted and programmatic input to the API query limit', async () => {
  const fetcher = vi.fn<typeof fetch>(() => Promise.resolve(Response.json({ results: [] })));
  vi.stubGlobal('fetch', fetcher);
  show({ entries: [], blocks: [] });
  expect(input()).toHaveAttribute('maxlength', '500');
  change('x'.repeat(501));
  expect(input()).toHaveValue('x'.repeat(500));
  await advance(SEMANTIC_DELAY_MS);
  expect(new URL(String(fetcher.mock.calls[0]?.[0]), 'http://localhost').searchParams.get('q')).toHaveLength(500);
});
