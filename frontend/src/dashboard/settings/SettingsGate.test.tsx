import { act, render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { newQueryClient, Providers } from '../../shared/Providers';
import { CONFIG_KEY, type ConfigView } from './model';
import { useNeedsSetup } from './Settings';

const config = (needs_setup: boolean): ConfigView => ({
  ok: true,
  control: true,
  needs_setup,
  path: 'pack.toml',
  sections: [],
});

function SetupGate() {
  const setup = useNeedsSetup();
  return <h1>{setup ? 'Set up Liberational' : 'Dashboard'}</h1>;
}

function renderGate(needsSetup: boolean) {
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false, staleTime: Infinity } });
  client.setQueryData(CONFIG_KEY, config(needsSetup));
  render(
    <Providers client={client} test>
      <SetupGate />
    </Providers>,
  );
  return client;
}

it('keeps first-run setup mounted when a post-save config refresh sees the new file', async () => {
  const client = renderGate(true);
  expect(screen.getByRole('heading', { name: 'Set up Liberational' })).toBeInTheDocument();

  // Saving creates pack.toml before the dashboard restarts. A focus/background refresh can observe it.
  await act(async () => {
    client.setQueryData(CONFIG_KEY, config(false));
    await new Promise(resolve => setTimeout(resolve, 0));
  });

  expect(screen.getByRole('heading', { name: 'Set up Liberational' })).toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: 'Dashboard' })).not.toBeInTheDocument();
});

it('opens the dashboard normally when a configured installation first loads', () => {
  renderGate(false);
  expect(screen.getByRole('heading', { name: 'Dashboard' })).toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: 'Set up Liberational' })).not.toBeInTheDocument();
});
