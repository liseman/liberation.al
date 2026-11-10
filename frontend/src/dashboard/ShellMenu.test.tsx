import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it } from 'vitest';
import { renderPage } from '../test/render';
import { server } from '../test/server';
import { setPref, usePrefs } from './lib/prefs';
import { ShellMenu } from './ShellMenu';
import type { UsageHistory } from './sources/api';
import { UsagePanel } from './usage/UsagePanel';

const EMPTY: UsageHistory = { step: 5, window: 1800, cpu_count: 4, mem_total: 1, gpus: [], samples: [] };

beforeEach(() => {
  server.use(
    http.get('/services', () => HttpResponse.json({ ok: true, control: true, services: [] })),
    http.get('/stats/history', () => HttpResponse.json(EMPTY)),
  );
});
afterEach(() => {
  setPref('usagePanel', undefined);
});

it('holds Settings and the Debug log as links', async () => {
  renderPage(<ShellMenu />);
  await userEvent.click(screen.getByRole('button', { name: 'Menu' }));
  expect(await screen.findByRole('menuitem', { name: 'Settings' })).toHaveAttribute('href', '/settings');
  expect(screen.getByRole('menuitem', { name: 'Debug log' })).toHaveAttribute('href', '/debug');
});

it('marks the gear while a service has failed, as the Settings button did', async () => {
  const failed = { name: 'vision', description: '', state: 'failed', installed: true, pids: [], selected: true };
  server.use(http.get('/services', () => HttpResponse.json({ ok: true, control: true, services: [failed] })));
  renderPage(<ShellMenu />);
  expect(await screen.findByRole('button', { name: 'Menu (services: 1 issue)' })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Menu (services: 1 issue)' }));
  expect(await screen.findByText('Services: 1 issue')).toBeInTheDocument();
});

it('opens and closes the Resource monitor, remembering which', async () => {
  renderPage(
    <>
      <ShellMenu />
      <UsagePanel />
    </>,
  );
  expect(screen.queryByRole('region', { name: 'Resource monitor' })).toBeNull();
  await userEvent.click(screen.getByRole('button', { name: 'Menu' }));
  await userEvent.click(await screen.findByRole('menuitem', { name: 'Resource monitor' }));
  expect(await screen.findByRole('region', { name: 'Resource monitor' })).toBeInTheDocument();
  expect(JSON.parse(localStorage.getItem('liberational.prefs') ?? '{}').usagePanel).toEqual({ open: true });
  await userEvent.click(screen.getByRole('button', { name: 'Menu' }));
  await userEvent.click(await screen.findByRole('menuitem', { name: 'Resource monitor' }));
  await waitFor(() => expect(screen.queryByRole('region', { name: 'Resource monitor' })).toBeNull());
  expect(usePrefs.getState().usagePanel?.open).toBe(false);
});

it('keeps only the theme during first-run setup', async () => {
  renderPage(<ShellMenu setup />);
  await userEvent.click(screen.getByRole('button', { name: 'Menu' }));
  expect(await screen.findByRole('menuitem', { name: 'Dark' })).toBeInTheDocument();
  expect(screen.queryByRole('menuitem', { name: 'Settings' })).toBeNull();
  expect(screen.queryByRole('menuitem', { name: 'Debug log' })).toBeNull();
});
