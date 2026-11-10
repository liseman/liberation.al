import { fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { type HistoryView, historyOrder, observedUptime, type Service, type ServiceHistory } from './model';
import { ServiceHistoryPanel } from './ServiceHistory';

const svc = (name: string, state = 'active'): Service => ({
  name,
  description: 'Worker',
  state,
  installed: true,
  pids: [],
  selected: true,
});
const alpha: ServiceHistory = {
  name: 'alpha',
  spans: [
    { start: 0, end: 10, state: 'unknown' },
    { start: 10, end: 40, state: 'up' },
    { start: 40, end: 70, state: 'stopped' },
    { start: 70, end: 100, state: 'up' },
  ],
  events: [
    { at: 10, kind: 'started', reason: null },
    { at: 40, kind: 'crashed', reason: 'exit-code: exited, status 7' },
    { at: 50, kind: 'restart', reason: null },
    { at: 70, kind: 'started', reason: null },
  ],
};
const body: HistoryView = {
  start: 0,
  end: 100,
  source: 'journal',
  available: true,
  sample_s: null,
  truncated: false,
  note: null,
  services: [alpha],
};
function show(data: HistoryView = body) {
  server.use(http.get('/services/history', () => HttpResponse.json(data)));
  return renderPage(<ServiceHistoryPanel services={[svc('alpha')]} />);
}

it('excludes unknown time from observed uptime and leaves entirely unknown history unmeasured', () => {
  expect(
    observedUptime({
      name: 'alpha',
      events: [],
      spans: [
        { start: 0, end: 50, state: 'unknown' },
        { start: 50, end: 75, state: 'up' },
        { start: 75, end: 100, state: 'stopped' },
      ],
    }),
  ).toBe(50);
  expect(observedUptime({ name: 'alpha', events: [], spans: [{ start: 0, end: 100, state: 'unknown' }] })).toBeNull();
});

it('sorts current trouble, recorded failures, restarts, and downtime before healthy rows with stable ties', () => {
  const quiet = { name: 'quiet', spans: [], events: [] };
  expect(
    historyOrder(
      [
        { ...quiet, name: 'zeta' },
        { ...alpha, name: 'crash' },
        { ...quiet, name: 'failed' },
        { name: 'job', spans: [], events: [{ at: 40, kind: 'start-failed', reason: 'dependency' }] },
        quiet,
        { ...quiet, name: 'alpha' },
      ],
      [svc('zeta'), svc('crash'), svc('failed', 'failed'), svc('job'), svc('quiet'), svc('alpha')],
    ).map(r => r.name),
  ).toEqual(['failed', 'crash', 'job', 'alpha', 'quiet', 'zeta']);
});

it('shows recorded spans and crash reasons through keyboard focus and arrow navigation', async () => {
  show();
  const plot = await screen.findByRole('button', { name: 'alpha uptime history' });
  fireEvent.focus(plot);
  expect(await screen.findByText('alpha: Started')).toBeVisible();
  fireEvent.keyDown(plot, { key: 'ArrowLeft' });
  fireEvent.keyDown(plot, { key: 'ArrowLeft' });
  fireEvent.keyDown(plot, { key: 'ArrowLeft' });
  expect(await screen.findByText('alpha: Crashed')).toBeVisible();
  expect(screen.getByText('exit-code: exited, status 7')).toBeVisible();
  expect(screen.getByText('Observed run: 30 s')).toBeVisible();
  fireEvent.keyDown(plot, { key: 'Escape' });
  expect(screen.queryByText('alpha: Crashed')).not.toBeInTheDocument();
});

it('opens details on a tap/click without changing any service controls', async () => {
  show();
  const plot = await screen.findByRole('button', { name: 'alpha uptime history' });
  fireEvent.click(plot, { detail: 0 });
  expect(await screen.findByText('alpha: Started')).toBeVisible();
  expect(screen.queryByRole('switch')).not.toBeInTheDocument();
});

it('requests the selected seven-day window separately from the 24-hour cache', async () => {
  const windows: string[] = [];
  server.use(
    http.get('/services/history', ({ request }) => {
      windows.push(new URL(request.url).searchParams.get('hours') ?? '');
      return HttpResponse.json(body);
    }),
  );
  renderPage(<ServiceHistoryPanel services={[svc('alpha')]} />);
  await screen.findByRole('button', { name: 'alpha uptime history' });
  await userEvent.click(screen.getByRole('radio', { name: '7 days' }));
  await screen.findByRole('button', { name: 'alpha uptime history' });
  expect(windows).toEqual(['24', '168']);
});

it('labels sampled and unavailable history without converting an unknown window into healthy time', async () => {
  show({
    ...body,
    source: 'sampled',
    sample_s: 300,
    note: 'Exact start, restart, and exit details are unavailable.',
    services: [{ name: 'alpha', events: [], spans: [{ start: 0, end: 100, state: 'unknown' }] }],
  });
  expect(await screen.findByText('Five-minute samples')).toBeVisible();
  const row = screen.getByRole('listitem', { name: 'alpha history' });
  expect(within(row).getByText('Not observed')).toBeVisible();
  expect(within(row).queryByText(/100%/)).not.toBeInTheDocument();
});

it('includes a manually running service even when it is neither installed nor selected', () => {
  const empty = { spans: [], events: [] };
  expect(
    historyOrder(
      [
        { name: 'manual', ...empty },
        { name: 'off', ...empty },
      ],
      [
        { ...svc('manual', 'not installed'), installed: false, selected: false, pids: [42] },
        { ...svc('off', 'not installed'), installed: false, selected: false, pids: [] },
      ],
    ).map(row => row.name),
  ).toEqual(['manual']);
});

it('does not show a permanently loading history card without a service inventory', () => {
  renderPage(<ServiceHistoryPanel services={[]} />);
  expect(screen.queryByRole('region', { name: 'Service uptime' })).not.toBeInTheDocument();
  expect(screen.queryByText('Loading service history…')).not.toBeInTheDocument();
});

it('shows a failed start job as a distinct failure without claiming a crash or stopped uptime', async () => {
  show({
    ...body,
    services: [
      {
        name: 'alpha',
        spans: [{ start: 0, end: 100, state: 'unknown' }],
        events: [{ at: 50, kind: 'start-failed', reason: 'dependency' }],
      },
    ],
  });
  const row = await screen.findByRole('listitem', { name: 'alpha history' });
  expect(within(row).getByText(/1 failure/)).toBeVisible();
  expect(within(row).getByText(/Not observed/)).toBeVisible();
  fireEvent.focus(within(row).getByRole('button', { name: 'alpha uptime history' }));
  expect(screen.getByText('alpha: Start failed')).toBeVisible();
  expect(screen.getByText('dependency')).toBeVisible();
  expect(screen.queryByText('alpha: Crashed')).not.toBeInTheDocument();
});
