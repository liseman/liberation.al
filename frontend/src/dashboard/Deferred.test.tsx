import { act, screen } from '@testing-library/react';
import { lazy } from 'react';
import { expect, it, vi } from 'vitest';
import { reportError } from '../shared/telemetry';
import { renderPage } from '../test/render';
import { Deferred } from './Deferred';
import { reloadForNewBuild } from './lib/staleChunk';

// Reloading the page isn't something jsdom can do: the boundary's decision is what's under test (#2005).
vi.mock('./lib/staleChunk', async importOriginal => {
  const real = await importOriginal<typeof import('./lib/staleChunk')>();
  return { ...real, reloadForNewBuild: vi.fn((error: unknown) => real.isChunkLoadError(error)) };
});
vi.mock('../shared/telemetry', async importOriginal => {
  const real = await importOriginal<typeof import('../shared/telemetry')>();
  return { ...real, reportError: vi.fn() };
});

it('keeps a named loading state until deferred code resolves', async () => {
  let finish!: (value: { default: () => React.ReactNode }) => void;
  const Page = lazy(
    () =>
      new Promise<{ default: () => React.ReactNode }>(resolve => {
        finish = resolve;
      }),
  );
  renderPage(
    <>
      <button type="button">Shell control</button>
      <Deferred name="Reports">
        <Page />
      </Deferred>
    </>,
  );
  expect(screen.getByLabelText('Loading Reports')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Shell control' })).toBeEnabled();
  await act(async () => finish({ default: () => <h1>Reports ready</h1> }));
  expect(await screen.findByRole('heading', { name: 'Reports ready' })).toBeInTheDocument();
  expect(screen.queryByLabelText('Loading Reports')).not.toBeInTheDocument();
});

it('contains a repeated chunk failure and provides an explicit reload action', async () => {
  vi.mocked(reloadForNewBuild).mockReturnValueOnce(false);
  vi.spyOn(console, 'error').mockImplementation(() => {});
  const Page = lazy(async () => {
    throw new TypeError('Failed to fetch dynamically imported module: /static/app/assets/Objects-OLD.js');
  });
  renderPage(
    <>
      <button type="button">Shell control</button>
      <Deferred name="Objects">
        <Page />
      </Deferred>
    </>,
  );
  expect(await screen.findByText("Couldn't load Objects")).toBeInTheDocument();
  expect(screen.getByText(/code didn't download/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Reload' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Shell control' })).toBeEnabled();
});

it('reloads for the new build when a redeploy removed the chunk, instead of showing a failure (#2005)', async () => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
  const Page = lazy(async () => {
    throw new TypeError('Failed to fetch dynamically imported module: /static/app/assets/Known-OLD.js');
  });
  renderPage(
    <Deferred name="Known">
      <Page />
    </Deferred>,
  );
  expect(await screen.findByLabelText('Loading the new version of Known')).toBeInTheDocument();
  expect(screen.queryByText("Couldn't load Known")).not.toBeInTheDocument();
  expect(reloadForNewBuild).toHaveBeenCalledOnce();
  expect(reportError).not.toHaveBeenCalled(); // the reload is its recovery, not an error in the page's log
});

it('logs runtime failures without blaming a code download or automatically reloading', async () => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
  const failed = new Error('Maximum update depth exceeded');
  const Page = lazy(async () => {
    throw failed;
  });
  renderPage(
    <Deferred name="Objects">
      <Page />
    </Deferred>,
  );
  expect(await screen.findByText("Couldn't load Objects")).toBeInTheDocument();
  expect(reportError).toHaveBeenCalledWith('boundary', failed, 'Objects');
  expect(screen.getByText(/encountered an unexpected error/)).toBeInTheDocument();
  expect(screen.queryByText(/code didn't download/)).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Loading the new version of Objects')).not.toBeInTheDocument();
});
