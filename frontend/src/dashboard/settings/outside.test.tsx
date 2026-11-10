import { screen, waitFor, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { DraftProvider } from './draft';
import type { ConfigView } from './model';
import { PhoneStatus, WhatLeaves } from './outside';

const service = (name: string, state: string, selected = true) => ({ name, description: '', state, selected });

function serve(services: object[], sources: object[] = []) {
  server.use(
    http.get('/services', () => HttpResponse.json({ ok: true, control: true, services })),
    http.get('/ingest/api/items', () => HttpResponse.json({ items: [], jobs: [], control: true, sources })),
  );
}

const view = (agentUrl: string): ConfigView =>
  ({
    ok: true,
    control: true,
    needs_setup: false,
    path: 'pack.toml',
    sections: [
      {
        name: 'agent',
        description: '',
        settings: [
          {
            key: 'agent.url',
            value: agentUrl,
            default: '',
            type: 'str',
            secret: false,
            description: '',
            section: 'advanced',
          },
        ],
      },
    ],
  }) as ConfigView;

describe("Phone's status", () => {
  it.each([
    ['active', true, 'Calls work: the phone bridge is running.', null],
    ['failed', true, 'Calls are off: the phone bridge stopped.', 'See Services'],
    [
      'not installed',
      false,
      "Calls are off: the phone bridge isn't set to run on this machine.",
      'Turn it on in Services',
    ],
  ])(
    'says calls work or are off when the bridge is %s, with the way to Services',
    async (state, selected, words, link) => {
      serve([service('phone-bridge', state, selected)]);
      renderPage(<PhoneStatus />);
      await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(words));
      if (link) expect(screen.getByRole('link', { name: link })).toHaveAttribute('href', '#services');
      else expect(screen.queryByRole('link')).toBeNull();
    },
  );
});

describe('What leaves this machine', () => {
  const row = (label: string) => screen.getByText(label).closest('div')?.parentElement as HTMLElement;

  it('says each way out on or off, and that everything else stays', async () => {
    serve(
      [service('phone-bridge', 'active')],
      [{ name: 'google_photos', label: 'Google Photos', ongoing: false, configured: true, message: '', seen: 0 }],
    );
    renderPage(
      <DraftProvider view={view('https://agent.example:18789/v1/chat/completions')}>
        <WhatLeaves />
      </DraftProvider>,
    );
    await screen.findByText(/go to the agent at agent\.example:18789/);
    expect(await within(row('Phone calls')).findByText('On')).toBeInTheDocument();
    expect(within(row('Agent')).getByText('On')).toBeInTheDocument();
    expect(await within(row('Google Photos')).findByText('On')).toBeInTheDocument();
    expect(screen.getByText(/Everything else stays on this machine/)).toBeInTheDocument();
  });

  it('counts an agent on this machine, or none, as nothing leaving', async () => {
    serve([]);
    const page = renderPage(
      <DraftProvider view={view('http://127.0.0.1:18789/v1/chat/completions')}>
        <WhatLeaves />
      </DraftProvider>,
    );
    expect(await screen.findByText(/an agent on this machine, so they stay here/)).toBeInTheDocument();
    expect(within(row('Agent')).getByText('Off')).toBeInTheDocument();
    page.unmount();
    renderPage(
      <DraftProvider view={view('')}>
        <WhatLeaves />
      </DraftProvider>,
    );
    expect(await screen.findByText(/No agent is set/)).toBeInTheDocument();
  });
});
