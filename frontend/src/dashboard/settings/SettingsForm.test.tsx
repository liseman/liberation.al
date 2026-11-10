import { modals } from '@mantine/modals';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { useState } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { ConfigView, Setting } from './model';
import { SettingsForm } from './SettingsForm';
import { FEATURES, OPEN_KEY } from './sections';

vi.mock('./restart', () => ({ followRestart: vi.fn(() => new Promise<void>(() => {})) }));
const row = (key: string, value: Setting['value'], extra: Partial<Setting> = {}): Setting => ({
  key,
  value,
  default: value,
  type: typeof value === 'boolean' ? 'bool' : Array.isArray(value) ? 'list' : typeof value === 'number' ? 'int' : 'str',
  secret: false,
  description: `About ${key}`,
  ...extra,
});
const view = (extra: Partial<ConfigView> = {}): ConfigView => ({
  ok: true,
  control: true,
  needs_setup: false,
  path: 'pack.toml',
  sections: [
    {
      name: 'owner',
      description: '',
      settings: [
        row('owner.name', 'Alice'),
        row('owner.phone', '+15550123456'),
        row('owner.speaker_label', '', { default: '' }),
      ],
    },
    {
      name: 'agent',
      description: '',
      settings: [row('agent.token', 'must-not-render-secret', { secret: true, set: true })],
    },
    {
      name: 'journal',
      description: '',
      settings: [row('journal.enabled', true, { feature: 'The journal' }), row('journal.levels', [30, 180])],
    },
  ],
  ...extra,
});
/** Most tests here are about the fields, so every section starts open; the accordion's own tests clear this. */
const ALL_OPEN = JSON.stringify([FEATURES, 'owner', 'agent', 'journal']);
beforeEach(() => {
  localStorage.setItem(OPEN_KEY, ALL_OPEN);
  history.replaceState(null, '', '/settings');
  server.use(
    http.get('/config', () => HttpResponse.json(view())),
    http.get('/health', () => HttpResponse.json({ boot: 'old' })),
  );
});

it('shows secret presence but never puts a saved secret into the DOM or draft payload', async () => {
  let submitted: { values: Record<string, unknown> } | undefined;
  server.use(
    http.post('/config', async ({ request }) => {
      submitted = (await request.json()) as { values: Record<string, unknown> };
      return HttpResponse.json({ ok: true });
    }),
  );
  const { container, client } = renderPage(<SettingsForm />);
  await screen.findByLabelText('Token agent.token');
  const secret = screen.getByPlaceholderText('Set. Leave blank to keep it.');
  expect(secret).toHaveValue('');
  expect(container.innerHTML).not.toContain('must-not-render-secret');
  await userEvent.click(screen.getByRole('button', { name: 'Save and restart' }));
  await waitFor(() => expect(submitted?.values['agent.token']).toBe(''));
  expect(client.getQueryState(['config'])?.isInvalidated).toBe(true);
});

it('binds each new-feature switch once and sends changes with the rest of the form', async () => {
  let submitted: { values: Record<string, unknown> } | undefined;
  server.use(
    http.post('/config', async ({ request }) => {
      submitted = (await request.json()) as { values: Record<string, unknown> };
      return HttpResponse.json({ ok: true });
    }),
  );
  renderPage(<SettingsForm />);
  const feature = await screen.findByRole('checkbox', { name: 'The journal' });
  expect(feature).toBeChecked();
  expect(screen.getAllByRole('checkbox')).toHaveLength(1);
  await userEvent.click(feature);
  await userEvent.click(screen.getByRole('button', { name: 'Save and restart' }));
  await waitFor(() => expect(submitted?.values['journal.enabled']).toBe(false));
  expect(submitted?.values['journal.levels']).toBe('30, 180');
});

it('requires owner fields and respects read-only settings', async () => {
  server.use(http.get('/config', () => HttpResponse.json(view({ control: false }))));
  renderPage(<SettingsForm />);
  const name = await screen.findByRole('textbox', { name: 'Name owner.name' });
  expect(name).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Save and restart' })).toBeDisabled();
  expect(screen.getByRole('checkbox', { name: 'The journal' })).toBeDisabled();
});

it('disables saving a blank required field and computes the speaker-label fallback', async () => {
  renderPage(<SettingsForm />);
  const name = await screen.findByRole('textbox', { name: /Name owner.name/ });
  expect(screen.getByRole('textbox', { name: 'Speaker label owner.speaker_label' })).toHaveAttribute(
    'placeholder',
    'alice',
  );
  await userEvent.clear(name);
  expect(screen.getByRole('button', { name: 'Save and restart' })).toBeDisabled();
  await userEvent.type(name, 'Dana');
  expect(screen.getByRole('textbox', { name: 'Speaker label owner.speaker_label' })).toHaveAttribute(
    'placeholder',
    'dana',
  );
  expect(screen.getByRole('button', { name: 'Save and restart' })).toBeEnabled();
});

it('asks before discarding edits, and Keep editing preserves the draft', async () => {
  const close = vi.fn();
  renderPage(<SettingsForm onClose={close} />);
  const name = await screen.findByRole('textbox', { name: /Name owner.name/ });
  await userEvent.clear(name);
  await userEvent.type(name, 'Dana');
  await userEvent.click(screen.getByRole('button', { name: 'Close' }));
  expect(await screen.findByRole('dialog', { name: 'Unsaved changes' })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Keep editing' }));
  expect(close).not.toHaveBeenCalled();
  expect(name).toHaveValue('Dana');
  await userEvent.click(screen.getByRole('button', { name: 'Close' }));
  await userEvent.click(await screen.findByRole('button', { name: 'Discard' }));
  expect(close).toHaveBeenCalledOnce();
});

it('shows server validation and focuses its field without losing the draft', async () => {
  server.use(
    http.post('/config', () =>
      HttpResponse.json({ ok: false, error: 'Use an international number', field: 'owner.phone' }, { status: 400 }),
    ),
  );
  renderPage(<SettingsForm />);
  const phone = await screen.findByRole('textbox', { name: 'Phone owner.phone' });
  await userEvent.clear(phone);
  await userEvent.type(phone, '123');
  await userEvent.click(screen.getByRole('button', { name: 'Save and restart' }));
  await waitFor(() => expect(phone).toHaveFocus());
  expect(phone).toHaveValue('123');
  expect(screen.getByRole('alert')).toHaveTextContent('Use an international number');
  expect(screen.getByRole('button', { name: 'Save and restart' })).toBeEnabled();
});

it('offers the first-run save action without a close action', async () => {
  renderPage(<SettingsForm setup />);
  expect(await screen.findByRole('heading', { name: 'Set up Liberational' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Save and start' })).toBeEnabled();
  expect(screen.queryByRole('button', { name: 'Close' })).not.toBeInTheDocument();
});

it('adds list entries as tags and submits the edited list', async () => {
  let submitted: { values: Record<string, unknown> } | undefined;
  server.use(
    http.post('/config', async ({ request }) => {
      submitted = (await request.json()) as { values: Record<string, unknown> };
      return HttpResponse.json({ ok: true });
    }),
  );
  renderPage(<SettingsForm />);
  const levels = await screen.findByRole('combobox', { name: 'Levels journal.levels' });
  await userEvent.type(levels, '720{Enter}');
  expect(screen.getByText('720', { exact: true })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Save and restart' }));
  await waitFor(() => expect(submitted?.values['journal.levels']).toBe('30, 180, 720'));
});

it('lands a setting deep link on its single feature switch, opening its section', async () => {
  localStorage.removeItem(OPEN_KEY); // every section starts closed
  history.replaceState(null, '', '/settings#set-journal.enabled');
  renderPage(<SettingsForm />);
  const feature = await screen.findByRole('checkbox', { name: 'The journal' });
  await waitFor(() => expect(feature).toHaveFocus());
  // The link back sits in the Journal section, which stays closed.
  expect(screen.getByRole('link', { name: 'The journal: on or off under New features', hidden: true })).toHaveAttribute(
    'href',
    '#set-journal.enabled',
  );
});

it('shows a load failure and retries when asked', async () => {
  server.use(http.get('/config', () => HttpResponse.json({ error: 'Settings unavailable' }, { status: 503 })));
  renderPage(<SettingsForm />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Settings unavailable');
  server.use(http.get('/config', () => HttpResponse.json(view())));
  await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
  expect(await screen.findByRole('textbox', { name: 'Name owner.name' })).toHaveValue('Alice');
});

function UnmountHarness({ close }: { close: () => void }) {
  const [visible, setVisible] = useState(true);
  return (
    <>
      <button type="button" onClick={() => setVisible(false)}>
        Leave settings
      </button>
      <button
        type="button"
        onClick={() => modals.open({ title: 'Other task', children: 'Other settings remain open' })}
      >
        Open other task
      </button>
      {visible && <SettingsForm onClose={close} />}
    </>
  );
}
function DismissHarness() {
  const [closeRequest, setCloseRequest] = useState(0);
  return (
    <>
      <button type="button" onClick={() => setCloseRequest(value => value + 1)}>
        Dismiss drawer
      </button>
      <SettingsForm onClose={() => {}} closeRequest={closeRequest} />
    </>
  );
}
it('removes the pending discard confirmation when the settings form unmounts', async () => {
  const close = vi.fn();
  renderPage(<UnmountHarness close={close} />);
  const leave = screen.getByRole('button', { name: 'Leave settings' });
  const name = await screen.findByRole('textbox', { name: 'Name owner.name' });
  await userEvent.type(name, ' changed');
  await userEvent.click(screen.getByRole('button', { name: 'Close' }));
  await screen.findByRole('dialog', { name: 'Unsaved changes' });
  fireEvent.click(leave);
  await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Unsaved changes' })).not.toBeInTheDocument());
  expect(close).not.toHaveBeenCalled();
});
it('keeps one confirmation when drawer close is requested repeatedly', async () => {
  renderPage(<DismissHarness />);
  const dismiss = screen.getByRole('button', { name: 'Dismiss drawer' });
  const name = await screen.findByRole('textbox', { name: 'Name owner.name' });
  await userEvent.type(name, ' changed');
  fireEvent.click(dismiss);
  await screen.findByRole('dialog', { name: 'Unsaved changes' });
  fireEvent.click(dismiss);
  await userEvent.click(screen.getByRole('button', { name: 'Keep editing' }));
  await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Unsaved changes' })).not.toBeInTheDocument());
});

it('unmounting one form leaves another task modal open', async () => {
  renderPage(<UnmountHarness close={vi.fn()} />);
  const leave = screen.getByRole('button', { name: 'Leave settings' });
  const openOther = screen.getByRole('button', { name: 'Open other task' });
  const name = await screen.findByRole('textbox', { name: 'Name owner.name' });
  await userEvent.type(name, ' changed');
  await userEvent.click(screen.getByRole('button', { name: 'Close' }));
  await screen.findByRole('dialog', { name: 'Unsaved changes' });
  fireEvent.click(openOther);
  await screen.findByRole('dialog', { name: 'Other task' });
  fireEvent.click(leave);
  expect(screen.getByRole('dialog', { name: 'Other task' })).toBeInTheDocument();
});

// ---- the accordion ----

const control = (name: RegExp) => screen.getByRole('button', { name });

it('starts with every section closed, each saying how many settings it holds', async () => {
  localStorage.removeItem(OPEN_KEY);
  renderPage(<SettingsForm />);
  await screen.findByRole('button', { name: /^Owner/ });
  expect(control(/^Owner/)).toHaveTextContent('Owner3 settings');
  expect(control(/^Agent/)).toHaveTextContent('Agent1 setting');
  expect([/^New features/, /^Owner/, /^Agent/, /^Journal/].map(n => control(n).getAttribute('aria-expanded'))).toEqual([
    'false',
    'false',
    'false',
    'false',
  ]);
  expect(screen.queryByRole('textbox', { name: 'Name owner.name' })).not.toBeInTheDocument();
});

it('opens several sections at once and remembers them in this browser', async () => {
  localStorage.removeItem(OPEN_KEY);
  const first = renderPage(<SettingsForm />);
  await userEvent.click(await screen.findByRole('button', { name: /^Owner/ }));
  await userEvent.click(control(/^Agent/));
  expect(await screen.findByRole('textbox', { name: 'Name owner.name' })).toBeVisible();
  expect(JSON.parse(localStorage.getItem(OPEN_KEY) ?? '[]')).toEqual(['owner', 'agent']);
  first.unmount();
  renderPage(<SettingsForm />);
  await screen.findByRole('button', { name: /^Owner/ });
  expect(control(/^Owner/)).toHaveAttribute('aria-expanded', 'true');
  expect(control(/^Agent/)).toHaveAttribute('aria-expanded', 'true');
  expect(control(/^Journal/)).toHaveAttribute('aria-expanded', 'false');
});

it('still works when the browser refuses storage', async () => {
  const get = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
    throw new Error('denied');
  });
  const set = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new Error('denied');
  });
  try {
    renderPage(<SettingsForm />);
    await userEvent.click(await screen.findByRole('button', { name: /^Owner/ }));
    expect(await screen.findByRole('textbox', { name: 'Name owner.name' })).toBeVisible();
  } finally {
    get.mockRestore();
    set.mockRestore();
  }
});

it('opens a section with something to fix and marks it', async () => {
  localStorage.removeItem(OPEN_KEY);
  server.use(
    http.get('/config', () =>
      HttpResponse.json(
        view({
          needs_setup: true,
          sections: [{ name: 'owner', description: '', settings: [row('owner.name', '', { default: '' })] }],
        }),
      ),
    ),
  );
  renderPage(<SettingsForm setup />);
  await waitFor(() => expect(screen.getByRole('button', { name: /^Owner/ })).toHaveAttribute('aria-expanded', 'true'));
  expect(control(/^Owner/)).toHaveTextContent('Needs attention');
});

it('marks a section with an unsaved change, and only that one', async () => {
  renderPage(<SettingsForm />);
  const name = await screen.findByRole('textbox', { name: 'Name owner.name' });
  expect(control(/^Owner/)).not.toHaveTextContent('Unsaved');
  await userEvent.type(name, 'x');
  expect(control(/^Owner/)).toHaveTextContent('Unsaved');
  expect(control(/^Agent/)).not.toHaveTextContent('Unsaved');
});

// ---- closed-section and focus edge cases ----

it('focuses a refused field even when its section was closed', async () => {
  localStorage.setItem(OPEN_KEY, JSON.stringify(['agent']));
  server.use(
    http.post('/config', () =>
      HttpResponse.json({ ok: false, error: 'Use an international number', field: 'owner.phone' }, { status: 400 }),
    ),
  );
  renderPage(<SettingsForm />);
  await userEvent.click(await screen.findByRole('button', { name: 'Save and restart' }));
  const phone = await screen.findByRole('textbox', { name: 'Phone owner.phone' });
  await waitFor(() => expect(phone).toHaveFocus());
  expect(control(/^Owner/)).toHaveTextContent('Needs attention');
});

it('leaves a section closed after its owner closes it, when another section changes', async () => {
  localStorage.setItem(OPEN_KEY, JSON.stringify(['owner', FEATURES]));
  renderPage(<SettingsForm />);
  await userEvent.type(await screen.findByRole('textbox', { name: 'Name owner.name' }), 'x');
  await userEvent.click(control(/^Owner/));
  await waitFor(() => expect(control(/^Owner/)).toHaveAttribute('aria-expanded', 'false'));
  await userEvent.click(screen.getByRole('checkbox', { name: 'The journal' }));
  expect(control(/^Owner/)).toHaveAttribute('aria-expanded', 'false');
});

it('marks only New features for a feature switch, not the section it belongs to', async () => {
  localStorage.setItem(OPEN_KEY, JSON.stringify([FEATURES]));
  renderPage(<SettingsForm />);
  await userEvent.click(await screen.findByRole('checkbox', { name: 'The journal' }));
  expect(control(/^New features/)).toHaveTextContent('Unsaved');
  expect(control(/^Journal/)).not.toHaveTextContent('Unsaved');
  expect(control(/^Journal/)).toHaveAttribute('aria-expanded', 'false');
});

it("opens a closed section when the browser finds one of its fields invalid, so Save isn't silently blocked", async () => {
  localStorage.removeItem(OPEN_KEY);
  renderPage(<SettingsForm />);
  await screen.findByRole('button', { name: /^Owner/ });
  const phone = document.getElementById('set-owner.phone');
  expect(phone).not.toBeNull();
  fireEvent.invalid(phone as HTMLElement);
  await waitFor(() => expect(control(/^Owner/)).toHaveAttribute('aria-expanded', 'true'));
  await waitFor(() => expect(phone).toHaveFocus());
});
