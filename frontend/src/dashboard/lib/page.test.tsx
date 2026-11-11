import { useQuery } from '@tanstack/react-query';
import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { PAGE_SLOW_MS, PageError, PageFrame, PageLoading, PHONE_WIDTH, QueryState } from './page';

afterEach(() => {
  vi.useRealTimers();
});

it('can show a card grid skeleton shaped like the People roster', () => {
  renderPage(<PageLoading label="people" shape="cards" lines={2} />);
  expect(screen.getByTestId('cards-skeleton')).toBeInTheDocument();
});

it('shows the skeleton under the real heading, and says it is still loading only once the wait runs long', async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  renderPage(
    <PageFrame title="People">
      <PageLoading label="people" />
    </PageFrame>,
  );
  expect(screen.getByRole('heading', { level: 1, name: 'People' })).toBeInTheDocument();
  expect(screen.getByLabelText('Loading people')).toBeInTheDocument();
  await act(async () => {
    await vi.advanceTimersByTimeAsync(PAGE_SLOW_MS - 100);
  });
  expect(screen.getByRole('status')).toHaveTextContent('');
  await act(async () => {
    await vi.advanceTimersByTimeAsync(200);
  });
  expect(screen.getByRole('status')).toHaveTextContent('Still loading people.');
});

it('keeps the raw reason behind Show details and retries on request', async () => {
  const retry = vi.fn();
  renderPage(<PageError title="Couldn't load people" detail="disk" onRetry={retry} />);
  const alert = screen.getByRole('alert');
  expect(alert).toHaveTextContent("Couldn't load people");
  expect(alert).toHaveTextContent('Nothing was lost.');
  expect(screen.getByText('disk')).not.toBeVisible();
  await userEvent.click(screen.getByText('Show details'));
  expect(screen.getByText('disk')).toBeVisible();
  await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
  expect(retry).toHaveBeenCalledOnce();
});

function Fetched({ fetch }: { fetch: () => Promise<string> }) {
  const query = useQuery({ queryKey: ['query-state', fetch], queryFn: fetch, retry: false });
  return (
    <QueryState query={query} label="the notes">
      {text => <p>{text}</p>}
    </QueryState>
  );
}

it('QueryState shows the skeleton, then the content', async () => {
  let done: (v: string) => void = () => {};
  renderPage(<Fetched fetch={() => new Promise(r => (done = r))} />);
  expect(screen.getByLabelText('Loading the notes')).toBeInTheDocument();
  await act(async () => done('Hello'));
  expect(await screen.findByText('Hello')).toBeInTheDocument();
  expect(screen.queryByLabelText('Loading the notes')).not.toBeInTheDocument();
});

it('QueryState says what failed, keeps the reason behind Show details, and retries', async () => {
  let calls = 0;
  renderPage(
    <Fetched
      fetch={async () => {
        calls += 1;
        if (calls === 1) throw new Error('disk');
        return 'Back';
      }}
    />,
  );
  expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't load the notes");
  expect(screen.getByText('disk')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
  expect(await screen.findByText('Back')).toBeInTheDocument();
});

function Held() {
  const query = useQuery({ queryKey: ['held'], queryFn: () => new Promise<string>(() => {}) });
  return (
    <QueryState query={query} label="the chart" loadingProps={{ 'data-report-loading': '' }}>
      {text => <p>{text}</p>}
    </QueryState>
  );
}

it('QueryState marks its loading skeleton for a parent that reserves room while it loads', () => {
  renderPage(<Held />);
  expect(screen.getByLabelText('Loading the chart').closest('[data-report-loading]')).not.toBeNull();
});

// The header and the page frame both fold at 700px in their stylesheets (app.module.css, page.module.css). A page
// that folded at a wider width would hide its own controls while the header still showed the full tab bar.
it("folds at the header's own breakpoint", () => {
  expect(PHONE_WIDTH).toBe('(max-width: 700px)');
});
