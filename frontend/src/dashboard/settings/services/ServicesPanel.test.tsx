import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import type { ConfigView, Setting } from '../model';
import { SettingsPage } from '../SettingsPage';
import type { Service, ServicesView } from './model';

async function expectTooltip(target: HTMLElement, text: string) {
  await userEvent.hover(target);
  expect(await screen.findByText(text, { selector: '[role="tooltip"]' })).toBeInTheDocument();
  await userEvent.unhover(target);
  await waitFor(() => expect(screen.queryByText(text, { selector: '[role="tooltip"]' })).not.toBeInTheDocument());
}

const svc = (over: Partial<Service> = {}): Service => ({
  name: 'voice',
  description: 'Listens',
  state: 'not running',
  installed: false,
  pids: [],
  selected: true,
  ...over,
});

const SERVICES: Service[] = [
  svc({
    name: 'dashboard',
    description: 'This page',
    group: 'dashboard',
    installed: true,
    state: 'active',
    pids: [10],
    default: true,
  }),
  svc({
    name: 'voice',
    description: 'Listens and talks',
    group: 'workers',
    installed: true,
    state: 'active',
    pids: [11, 12],
    up_s: 540,
    default: true,
  }),
  svc({
    name: 'vision',
    description: 'Watches the cameras',
    group: 'workers',
    installed: true,
    state: 'failed',
    default: true,
  }),
  svc({ name: 'notes', description: 'Writes the journal', default: true }),
  svc({ name: 'phone', description: 'Answers calls', selected: false }),
];

const setting = (key: string, label: string): Setting => ({
  key,
  type: 'str',
  value: '',
  default: '',
  secret: false,
  description: '',
  label,
  help: '',
  section: 'services',
});
const CONFIG: ConfigView = {
  ok: true,
  control: true,
  needs_setup: false,
  path: 'pack.toml',
  restarts: [],
  sections: [
    {
      name: 'services',
      description: '',
      settings: [setting('services.enable', 'Also run'), setting('services.disable', "Don't run")],
    },
  ],
};
let configPosts: unknown[];
beforeEach(() => {
  localStorage.clear();
  configPosts = [];
  server.use(
    http.get('/config', () => HttpResponse.json(CONFIG)),
    http.post('/config', async ({ request }) => {
      configPosts.push(await request.json());
      return HttpResponse.json({ ok: true });
    }),
    http.get('/services/history', () =>
      HttpResponse.json({
        start: 0,
        end: 86400,
        source: 'unavailable',
        available: false,
        sample_s: null,
        truncated: false,
        note: 'No service history is available for this window.',
        services: [],
      }),
    ),
    http.get('/health', () => HttpResponse.json({ boot: 'b1' })),
    http.get('/vision', () => HttpResponse.json({})), // Cameras and microphones' menu badge
    http.get('/ambient', () => HttpResponse.json({})),
    http.get('/stats', () =>
      HttpResponse.json({ cpu: 12, cpu_count: 8, load: [1, 1, 1], mem_used: 2e9, mem_total: 8e9 }),
    ),
    http.get('/stats/history', () =>
      HttpResponse.json({ step: 5, window: 1800, cpu_count: 8, mem_total: 8e9, gpus: [], samples: [] }),
    ),
  );
});
afterEach(() => history.replaceState(null, '', '/'));

const view = (over: Partial<ServicesView> = {}): ServicesView => ({
  ok: true,
  control: true,
  platform: 'linux-x86_64',
  services: SERVICES,
  ...over,
});

/** The view with the named services installed and running. */
const withUp = (...names: string[]): ServicesView =>
  view({
    services: SERVICES.map(s => (names.includes(s.name) ? { ...s, installed: true, state: 'active', pids: [99] } : s)),
  });

/** A promise the test resolves itself, to hold a request in flight. */
function gate() {
  let open = () => {};
  const shut = new Promise<void>(r => {
    open = r;
  });
  return { shut, open };
}

function serve(v: ServicesView = view()) {
  server.use(http.get('/services', () => HttpResponse.json(v)));
}

/** Settings, opened on Services (or the section the hash names). */
async function openSection(hash = '#services') {
  history.replaceState(null, '', `/settings${hash}`);
  const page = renderPage(<SettingsPage />);
  const name = { '#services': 'Services', '#machine': 'This machine' }[hash] ?? /./;
  const section = await screen.findByRole('region', { name });
  if (hash === '#services') await showAll();
  return { ...page, section };
}

/** Every group of services open: a quiet one starts folded; these tests read its rows. */
async function showAll() {
  const toggles = await screen.findAllByRole('button', { name: /^(Show|Hide) / }).catch(() => []);
  for (const t of toggles) if (t.getAttribute('aria-expanded') === 'false') await userEvent.click(t);
}
const badge = () => screen.getByRole('link', { name: /^Services/ });
const row = (name: string) => screen.getByRole('listitem', { name });
const health = () => screen.getByRole('region', { name: 'Health' });

/** Disable from a row's menu. */
async function disable(name: string) {
  await userEvent.click(await screen.findByRole('button', { name: `More for ${name}` }));
  await userEvent.click(await screen.findByRole('menuitem', { name: /^Disable/ }));
}

describe('the menu badge', () => {
  it('counts failed and crash-looping services, and clears once they recover', async () => {
    serve(view({ services: [...SERVICES, svc({ name: 'ambient', installed: true, state: 'active', restarts: 5 })] }));
    const { client } = await openSection('#you');
    await waitFor(() => expect(badge()).toHaveTextContent('2 issues')); // vision failed, ambient keeps restarting
    client.setQueryData(['services'], withUp('vision', 'notes'));
    await waitFor(() => expect(badge()).toHaveTextContent(/^Services$/));
  });

  it('says when status cannot be read, and the section says why', async () => {
    server.use(http.get('/services', () => HttpResponse.json({ error: 'boom' }, { status: 500 })));
    const { section } = await openSection();
    await waitFor(() => expect(badge()).toHaveTextContent('Unreadable'));
    expect(await within(section).findByRole('alert')).toHaveTextContent('Could not read service status: boom');
    expect(health()).toHaveTextContent('Status unknown');
  });
});

describe('the summary', () => {
  it('counts what runs, what needs attention, and what is stopped, and takes you to the trouble', async () => {
    serve();
    await openSection();
    await waitFor(() => expect(health()).toHaveTextContent('2 running, 1 needs attention, 1 stopped'));
    expect(health()).toHaveTextContent('4 set to run here. 1 turned off.');
    await userEvent.click(within(health()).getByRole('button', { name: '1 needs attention' }));
    expect(document.activeElement).toBe(row('vision'));
  });

  it('says only what runs when all is well', async () => {
    serve(withUp('vision', 'notes'));
    await openSection();
    await waitFor(() => expect(health()).toHaveTextContent(/^4 running4 set to run here\. 1 turned off\./));
    expect(within(health()).queryByRole('button', { name: /Start/ })).not.toBeInTheDocument(); // nothing to start
  });
});

describe('recent error notices', () => {
  it('dismisses only this notice, keeps it dismissed after remount, and shows a new error', async () => {
    const recovered = svc({
      installed: true,
      state: 'active',
      up_s: 1800,
      last_error: { at: '09:02:00', ts: 1000, text: 'temporary connection failure' },
    });
    serve(view({ services: [recovered] }));
    const { unmount } = await openSection();
    const service = await screen.findByRole('listitem', { name: 'voice' });
    await userEvent.click(within(service).getByRole('button', { name: 'Dismiss issue' }));
    expect(service).not.toHaveTextContent('temporary connection failure');
    expect(service).toHaveTextContent('running');
    unmount();
    const next = await openSection();
    expect(await screen.findByRole('listitem', { name: 'voice' })).not.toHaveTextContent(
      'temporary connection failure',
    );
    next.client.setQueryData(
      ['services'],
      view({
        services: [
          {
            ...recovered,
            last_error: { at: '09:03:00', ts: 1060, text: 'temporary connection failure' },
          },
        ],
      }),
    );
    await waitFor(() => expect(row('voice')).toHaveTextContent('temporary connection failure'));
  });

  it.each(['failed', 'activating'])('never lets dismissal hide an active problem (%s)', async state => {
    localStorage.setItem(
      'liberational.services.dismissed.voice',
      JSON.stringify({
        at: '09:02:00',
        ts: 1000,
        text: 'temporary connection failure',
      }),
    );
    serve(
      view({
        services: [
          svc({
            installed: true,
            state,
            restarts: 4,
            up_s: 40,
            last_error: { at: '09:02:00', ts: 1000, text: 'temporary connection failure' },
          }),
        ],
      }),
    );
    await openSection();
    const service = await screen.findByRole('listitem', { name: 'voice' });
    expect(service).toHaveAttribute('data-trouble');
    expect(service).toHaveTextContent('temporary connection failure');
    expect(within(service).queryByRole('button', { name: 'Dismiss issue' })).not.toBeInTheDocument();
  });
});

describe('recovered restart issues', () => {
  it('dismisses a running restart-only issue and its badges, then restores a new restart', async () => {
    const recovered = svc({ installed: true, state: 'active', restarts: 6, up_s: 415, run_id: 'run-1' });
    serve(view({ services: [recovered] }));
    const { client } = await openSection();
    await waitFor(() => expect(badge()).toHaveTextContent('1 issue'));
    await userEvent.click(within(row('voice')).getByRole('button', { name: 'Dismiss issue' }));
    await waitFor(() => expect(badge()).toHaveTextContent(/^Services$/));
    expect(row('voice')).toHaveTextContent('running');
    expect(row('voice')).not.toHaveAttribute('data-trouble');
    expect(health()).not.toHaveTextContent('needs attention');
    client.setQueryData(['services'], view({ services: [{ ...recovered, restarts: 7, run_id: 'run-2' }] }));
    await waitFor(() => expect(badge()).toHaveTextContent('1 issue'));
    expect(row('voice')).toHaveAttribute('data-trouble');
  });

  it('brings back identical text in the same second when its log occurrence changes', async () => {
    const recovered = svc({
      installed: true,
      state: 'active',
      up_s: 1800,
      run_id: 'run-1',
      last_error: { id: 'error-1', at: '09:02:00', ts: 1000, text: 'temporary connection failure' },
    });
    serve(view({ services: [recovered] }));
    const { client } = await openSection();
    await userEvent.click(
      within(await screen.findByRole('listitem', { name: 'voice' })).getByRole('button', { name: 'Dismiss issue' }),
    );
    expect(row('voice')).not.toHaveTextContent('temporary connection failure');
    client.setQueryData(
      ['services'],
      view({
        services: [
          {
            ...recovered,
            last_error: { id: 'error-2', at: '09:02:00', ts: 1000, text: 'temporary connection failure' },
          },
        ],
      }),
    );
    await waitFor(() => expect(row('voice')).toHaveTextContent('temporary connection failure'));
  });
});

describe('issue acknowledgement safety', () => {
  it('keeps a subsequent failed state visible even after its running history was dismissed', async () => {
    const recovered = svc({ installed: true, state: 'active', restarts: 6, up_s: 415, run_id: 'failure-transition' });
    serve(view({ services: [recovered] }));
    const { client } = await openSection();
    await userEvent.click(
      within(await screen.findByRole('listitem', { name: 'voice' })).getByRole('button', { name: 'Dismiss issue' }),
    );
    client.setQueryData(['services'], view({ services: [{ ...recovered, state: 'failed' }] }));
    await waitFor(() => expect(row('voice')).toHaveAttribute('data-trouble'));
    expect(row('voice')).toHaveTextContent('failed');
    expect(within(row('voice')).queryByRole('button', { name: 'Dismiss issue' })).not.toBeInTheDocument();
    expect(badge()).toHaveTextContent('1 issue');
  });

  it('still dismisses when browser storage reads work but writes are blocked', async () => {
    localStorage.setItem('liberational.services.dismissed-issues', '{malformed');
    const recovered = svc({ installed: true, state: 'active', restarts: 6, up_s: 415, run_id: 'blocked-storage' });
    serve(view({ services: [recovered] }));
    await openSection();
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota exceeded');
    });
    await userEvent.click(
      within(await screen.findByRole('listitem', { name: 'voice' })).getByRole('button', { name: 'Dismiss issue' }),
    );
    await waitFor(() => expect(badge()).toHaveTextContent(/^Services$/));
    expect(row('voice')).not.toHaveAttribute('data-trouble');
  });
});

describe('a crash loop', () => {
  it('flags a loop and opens a box with what happened and the error, and keeps older errors quiet', async () => {
    serve(
      view({
        services: [
          svc({
            name: 'ambient',
            description: 'Hears the room',
            installed: true,
            state: 'active',
            restarts: 7,
            up_s: 40,
            last_error: { at: '09:02:11', text: 'the model would not load' },
          }),
          svc({ name: 'voice', description: 'Listens and talks', installed: true, state: 'active', restarts: 1 }),
          svc({
            name: 'vision',
            description: 'Watches',
            installed: true,
            state: 'active',
            restarts: 9,
            up_s: 86400, // looped once, but has run a day since
            last_error: { at: '', text: 'RuntimeError: the camera vanished' }, // a crash's bare traceback: no time
          }),
        ],
      }),
    );
    await openSection();
    const ambient = await screen.findByRole('listitem', { name: 'ambient' });
    expect(within(ambient).getByText('keeps restarting')).toBeInTheDocument();
    expect(within(ambient).getByText('at 09:02')).toBeInTheDocument();
    expect(
      within(ambient).getByText(
        'It is running after 7 automatic restarts. Dismiss this issue if it has recovered; a new error or restart brings it back.',
      ),
    ).toBeInTheDocument();
    expect(within(ambient).getByText('the model would not load')).toBeInTheDocument();
    expect(within(ambient).getByRole('link', { name: 'See the log' })).toHaveAttribute('href', '/debug');
    expect(within(ambient).queryByRole('button', { name: 'Start again' })).not.toBeInTheDocument(); // it runs
    const voice = row('voice');
    expect(within(voice).getByText('running')).toBeInTheDocument(); // one restart isn't a loop
    expect(voice).not.toHaveTextContent(/restarted/);
    const vision = row('vision');
    expect(within(vision).getByText('running')).toBeInTheDocument();
    expect(within(vision).getByText(/^Recent error:/).textContent).toBe(
      'Recent error: RuntimeError: the camera vanished See the log Dismiss issue',
    );
    expect(within(vision).queryByText(/stopped trying/)).not.toBeInTheDocument();
    await waitFor(() => expect(badge()).toHaveTextContent('1 issue')); // ambient only
  });

  it('says a failed service stopped, when, and how often it was restarted, and starts it again', async () => {
    const failed = svc({
      name: 'vision',
      group: 'workers',
      installed: true,
      state: 'failed',
      restarts: 4,
      last_error: { at: '09:41:02', text: 'CUDA out of memory' },
    });
    serve(view({ services: [failed] }));
    let sent: unknown;
    server.use(
      http.post('/services', async ({ request }) => {
        sent = await request.json();
        return HttpResponse.json(view({ services: [{ ...failed, state: 'active', restarts: 0, up_s: 2 }] })); // a start by hand resets the count
      }),
    );
    await openSection();
    const vision = await screen.findByRole('listitem', { name: 'vision' });
    expect(vision).toHaveAttribute('data-trouble');
    expect(within(vision).getByText('failed')).toBeInTheDocument();
    expect(within(vision).getByText('at 09:41')).toBeInTheDocument();
    expect(
      within(vision).getByText(
        'Stopped at 09:41 after an error. The service manager restarted it 4 times, then stopped trying.',
      ),
    ).toBeInTheDocument();
    expect(within(vision).getByText('CUDA out of memory')).toBeInTheDocument();
    await userEvent.click(within(vision).getByRole('button', { name: 'Start again' }));
    await waitFor(() => expect(sent).toEqual({ name: 'vision', action: 'start' }));
    await waitFor(() => expect(row('vision')).not.toHaveAttribute('data-trouble'));
  });
});

describe('which services run here', () => {
  /** Settings, searching for the two lists, which Services leaves to search and Advanced. */
  async function searchRun() {
    await openSection();
    expect(screen.queryByRole('combobox', { name: 'Also run' })).not.toBeInTheDocument();
    await userEvent.type(screen.getByRole('searchbox', { name: 'Search settings' }), 'run');
  }

  it("picks Also run from the services off by default, and Don't run from the ones on, saved with the draft", async () => {
    serve(view({ services: [...SERVICES, svc({ name: 'whisper', description: 'Transcribes', required: true })] }));
    await searchRun();
    await userEvent.click(await screen.findByRole('combobox', { name: 'Also run' }));
    expect((await screen.findAllByRole('option')).map(o => o.textContent)).toEqual(['phone']); // never the dashboard or whisper
    await userEvent.click(await screen.findByRole('option', { name: 'phone' }));
    await userEvent.click(screen.getByRole('combobox', { name: "Don't run" }));
    expect((await screen.findAllByRole('option')).map(o => o.textContent)).toEqual(['voice', 'vision', 'notes']);
    await userEvent.click(await screen.findByRole('option', { name: 'notes' }));

    const bar = screen.getByRole('region', { name: 'Unsaved changes' });
    await userEvent.click(within(bar).getByRole('button', { name: /Save/ }));
    await waitFor(() => expect(configPosts).toHaveLength(1));
    expect(configPosts[0]).toMatchObject({ values: { 'services.enable': 'phone', 'services.disable': 'notes' } });
  });

  it('opens a link to one of the lists in Advanced, where its field is', async () => {
    // Services no longer draws them, so a link that opened Services found nothing to focus.
    serve();
    for (const key of ['services.enable', 'services.disable']) {
      history.replaceState(null, '', `/settings#set-${key}`);
      const { unmount } = renderPage(<SettingsPage />);
      expect(await screen.findByRole('heading', { level: 2, name: 'Advanced' })).toBeInTheDocument();
      await waitFor(() => expect(document.activeElement?.id).toBe(`set-${key}`));
      unmount();
    }
  });

  it('takes a name saved twice as one pick', async () => {
    serve();
    const saved = { ...setting('services.enable', 'Also run'), value: 'phone, phone' };
    server.use(
      http.get('/config', () =>
        HttpResponse.json({
          ...CONFIG,
          sections: [
            { name: 'services', description: '', settings: [saved, setting('services.disable', "Don't run")] },
          ],
        }),
      ),
    );
    await searchRun();
    await userEvent.click(await screen.findByRole('combobox', { name: 'Also run' }));
    expect(await screen.findByRole('option', { name: 'phone' })).toHaveAttribute('aria-selected', 'true');
  });

  it('keeps a saved name this machine no longer offers', async () => {
    serve();
    const saved = { ...setting('services.enable', 'Also run'), value: 'phone, retired' };
    server.use(
      http.get('/config', () =>
        HttpResponse.json({
          ...CONFIG,
          sections: [
            { name: 'services', description: '', settings: [saved, setting('services.disable', "Don't run")] },
          ],
        }),
      ),
    );
    await searchRun();
    await userEvent.click(await screen.findByRole('combobox', { name: 'Also run' }));
    await userEvent.click(await screen.findByRole('option', { name: 'phone' })); // unpick phone
    const bar = screen.getByRole('region', { name: 'Unsaved changes' });
    await userEvent.click(within(bar).getByRole('button', { name: /Save/ }));
    await waitFor(() => expect(configPosts).toHaveLength(1));
    expect(configPosts[0]).toMatchObject({ values: { 'services.enable': 'retired' } });
  });
});

describe('this machine', () => {
  it('is a section of its own, with the platform the services report', async () => {
    serve();
    const { section } = await openSection('#machine');
    expect(
      await within(section).findByText('Linux on x86-64, with 8 processor cores and 7.5 GB of memory.'),
    ).toBeInTheDocument();
    expect(within(section).queryByRole('switch')).not.toBeInTheDocument();
  });
});

describe('the services', () => {
  it('lists the services in a card per group, with their state, and the turned-off ones apart', async () => {
    serve();
    const { section } = await openSection();
    expect(await within(section).findByText('Listens and talks')).toBeInTheDocument();

    expect(
      within(section)
        .getAllByRole('heading', { level: 3 })
        .map(h => h.textContent),
    ).toEqual(['Service uptime', 'Workers', 'Dashboard', 'Other']);
    const workers = within(section).getByRole('region', { name: 'Workers' });
    expect(workers).toHaveTextContent('Turn the recordings into words, faces, and people, and answer when spoken to.');
    expect(within(workers).getByText('1 of 2 running')).toBeInTheDocument();
    expect(within(section).getByRole('region', { name: 'Dashboard' })).toHaveTextContent('Running');
    expect(
      within(workers)
        .getAllByRole('listitem')
        .map(li => li.getAttribute('aria-label')),
    ).toEqual(['voice', 'vision']);
    expect(within(row('voice')).getByTitle('pid 11, 12')).toHaveTextContent('running');
    expect(within(row('vision')).getByText('failed')).toBeInTheDocument();
    expect(within(row('vision')).getByText('Stopped after an error.')).toBeInTheDocument();
    expect(within(row('notes')).getByText('off')).toBeInTheDocument();

    // One switch per service: on while it runs, and it says which way it goes.
    const voice = screen.getByRole('switch', { name: 'Run voice' });
    expect(voice).toBeChecked();
    await expectTooltip(voice, 'Stops it now. It starts again when the computer restarts, or on Repair.');
    expect(screen.getByRole('switch', { name: 'Run vision' })).not.toBeChecked(); // failed: installed, but not up
    await expectTooltip(
      screen.getByRole('switch', { name: 'Run notes' }),
      'Installs and starts it. It runs at every login.',
    );
    // Its menu has the log and Disable, which says what it does.
    await userEvent.click(screen.getByRole('button', { name: 'More for voice' }));
    expect(await screen.findByRole('menuitem', { name: 'See the log' })).toHaveAttribute('href', '/debug');
    expect(screen.getByRole('menuitem', { name: /^Disable/ })).toHaveTextContent(
      'DisableStops it and keeps it off, even after a restart, until you enable it.',
    );
    await userEvent.keyboard('{Escape}');
    expect(screen.getByText(/^The switch starts or stops a service now/)).toHaveTextContent(
      'To keep one off, choose Disable from its menu.',
    );
    // The dashboard is always on: no switch and no menu.
    expect(screen.queryByRole('switch', { name: 'Run dashboard' })).not.toBeInTheDocument();
    expect(within(row('dashboard')).getByText('Always on')).toHaveAttribute(
      'title',
      'This page runs on the dashboard; use pack-services in a terminal',
    );
    expect(screen.queryByRole('button', { name: 'More for dashboard' })).not.toBeInTheDocument();

    // vision (failed) and notes (off) are what Start would start.
    await expectTooltip(screen.getByRole('button', { name: 'Start 2 stopped' }), 'Start vision, notes');

    // phone is stopped and not enabled here: under Turned off, open (this isn't a phone), with its group.
    const off = within(section).getByRole('region', { name: 'Turned off' });
    const head = within(off).getByRole('button', { name: /^Turned off/ });
    expect(head).toHaveAttribute('aria-expanded', 'true');
    expect(within(off).getByRole('listitem', { name: 'phone' })).toHaveTextContent('Answers callsOther');
    await userEvent.click(head);
    expect(head).toHaveAttribute('aria-expanded', 'false');
    expect(within(off).queryByRole('listitem', { name: 'phone' })).not.toBeInTheDocument();
    expect(within(section).getByRole('region', { name: 'Other' })).not.toHaveTextContent('phone');
  });

  it('shows a service not enabled here in the main list while it still runs', async () => {
    serve(view({ services: [svc({ name: 'phone', selected: false, installed: true, state: 'active' })] }));
    const { section } = await openSection();
    expect(await within(section).findByRole('region', { name: 'Other' })).toHaveTextContent('phone');
    expect(within(section).queryByRole('region', { name: 'Turned off' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Enable phone' })).toBeEnabled();
  });

  it.each([
    ['switch', 'Run notes', true],
    ['button', 'Enable notes', false],
  ] as const)(
    'says on the row why a service started by hand has no working %s, for keyboards too',
    async (role, name, selected) => {
      serve(view({ services: [svc({ name: 'notes', selected, installed: false, pids: [42] })] }));
      await openSection();
      const control = await screen.findByRole(role, { name });
      expect(control).toBeDisabled();
      expect(
        within(row('notes')).getByText(
          'Started outside the service manager, so it can only be stopped where it was started.',
        ),
      ).toBeVisible();
      expect(control).toHaveAccessibleDescription(
        'Started outside the service manager, so it can only be stopped where it was started.',
      );
    },
  );

  it('is view only away from the machine, with no switches or buttons', async () => {
    serve(view({ control: false }));
    const { section } = await openSection();
    expect(
      await within(section).findByText('View only. Services can be switched only on the machine Liberational runs on.'),
    ).toBeInTheDocument();
    expect(
      within(section).queryByRole('button', { name: /Start|Repair|Turn all off|More for|Enable|Disable/ }),
    ).toBeNull();
    expect(within(section).queryByRole('switch')).toBeNull();
    expect(within(row('voice')).getByText('running')).toBeInTheDocument();
    expect(within(section).queryByText(/^The switch starts/)).not.toBeInTheDocument();
  });
});

describe("a service's switch and Disable", () => {
  it('posts the switch, shows it starting, then shows the fresh state', async () => {
    serve();
    const reply = gate();
    let sent: unknown;
    server.use(
      http.post('/services', async ({ request }) => {
        sent = await request.json();
        await reply.shut;
        return HttpResponse.json({ ...withUp('notes'), out: 'installed notes' });
      }),
    );
    await openSection();
    await userEvent.click(await screen.findByRole('switch', { name: 'Run notes' }));

    await waitFor(() => expect(sent).toEqual({ name: 'notes', action: 'on' }));
    expect(within(row('notes')).getByText('starting')).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: 'Run notes' })).toBeDisabled();
    expect(screen.getByRole('switch', { name: 'Run notes' })).toBeChecked();
    // Other bulk actions wait for the switch to finish.
    expect(screen.getByRole('button', { name: 'Repair' })).toBeDisabled();
    await expectTooltip(
      screen.getByRole('button', { name: 'Repair' }).parentElement as HTMLElement,
      'Remove and reinstall every enabled service except the dashboard',
    );

    reply.open();
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Run notes' })).toBeEnabled());
    expect(screen.getByRole('switch', { name: 'Run notes' })).toBeChecked();
    expect(within(row('notes')).getByText('running')).toBeInTheDocument();
  });

  it("shows the server's reason when a switch is refused", async () => {
    serve();
    server.use(
      http.post('/services', () =>
        HttpResponse.json(
          { ok: false, error: 'notes is already running outside the service manager' },
          { status: 409 },
        ),
      ),
    );
    await openSection();
    await userEvent.click(await screen.findByRole('switch', { name: 'Run notes' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('notes is already running outside the service manager');
    expect(within(row('notes')).getByText('off')).toBeInTheDocument();
  });

  it('says which action failed when the server gives no reason', async () => {
    serve();
    server.use(http.post('/services', () => HttpResponse.json({ ...view(), ok: false, error: '' }, { status: 500 })));
    await openSection();
    await disable('voice');
    expect(await screen.findByRole('alert')).toHaveTextContent(/^Disable voice failed$/);
  });

  it('stops a running service with its switch, and disables it from its menu', async () => {
    serve();
    const sent: unknown[] = [];
    server.use(
      http.post('/services', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json(view());
      }),
    );
    await openSection();
    await userEvent.click(await screen.findByRole('switch', { name: 'Run voice' }));
    await waitFor(() => expect(sent).toEqual([{ name: 'voice', action: 'stop' }]));
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Run voice' })).toBeEnabled());
    await disable('voice');
    await waitFor(() =>
      expect(sent).toEqual([
        { name: 'voice', action: 'stop' },
        { name: 'voice', action: 'disable' },
      ]),
    );
  });

  it("holds every row's controls while one action runs", async () => {
    // Two Disables at once each rewrote pack.toml, and one change was lost.
    serve();
    const reply = gate();
    server.use(
      http.post('/services', async () => {
        await reply.shut;
        return HttpResponse.json(view());
      }),
    );
    await openSection();
    await disable('voice');
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Run vision' })).toBeDisabled());
    expect(screen.getByRole('switch', { name: 'Run notes' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Enable phone' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Start again' })).toBeDisabled();
    reply.open();
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Run vision' })).toBeEnabled());
  });

  it("reads the settings again after Disable, so Also run and Don't run show the change", async () => {
    serve();
    let reads = 0;
    server.use(
      http.get('/config', () => {
        reads += 1;
        return HttpResponse.json(CONFIG);
      }),
      http.post('/services', () => HttpResponse.json(view())),
    );
    await openSection();
    await screen.findByRole('switch', { name: 'Run voice' });
    const before = reads;
    await userEvent.click(screen.getByRole('switch', { name: 'Run voice' })); // a stop leaves pack.toml alone
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Run voice' })).toBeEnabled());
    expect(reads).toBe(before);
    await disable('voice');
    await waitFor(() => expect(reads).toBe(before + 1));
  });

  it('enables a turned-off service from its line', async () => {
    serve();
    let sent: unknown;
    server.use(
      http.post('/services', async ({ request }) => {
        sent = await request.json();
        return HttpResponse.json(view());
      }),
    );
    const { section } = await openSection();
    await userEvent.click(await within(section).findByRole('button', { name: 'Enable phone' }));
    await waitFor(() => expect(sent).toEqual({ name: 'phone', action: 'enable' }));
  });
});

describe('Start n stopped', () => {
  it('starts each stopped service in turn and reports every failure', async () => {
    serve();
    const sent: unknown[] = [];
    let inFlight = 0;
    let most = 0;
    const gates = [gate(), gate()];
    server.use(
      http.post('/services', async ({ request }) => {
        const body = (await request.json()) as { name: string };
        const step = sent.push(body) - 1;
        inFlight += 1;
        most = Math.max(most, inFlight);
        await gates[step]?.shut;
        inFlight -= 1;
        return body.name === 'vision'
          ? HttpResponse.json({ ok: false, error: 'vision would not start' }, { status: 500 })
          : HttpResponse.json(withUp('notes'));
      }),
    );
    await openSection();
    await userEvent.click(await screen.findByRole('button', { name: 'Start 2 stopped' }));

    expect(await screen.findByRole('button', { name: 'Starting vision' })).toBeDisabled();
    expect(within(row('vision')).getByText('starting')).toBeInTheDocument();
    gates[0]?.open();
    expect(await screen.findByRole('button', { name: 'Starting notes' })).toBeInTheDocument();
    gates[1]?.open();

    expect(await screen.findByRole('alert')).toHaveTextContent('vision would not start');
    expect(sent).toEqual([
      { name: 'vision', action: 'start' }, // installed, so started
      { name: 'notes', action: 'on' }, // not installed, so turned on
    ]);
    expect(most).toBe(1);
    expect(within(row('notes')).getByText('running')).toBeInTheDocument();
  });
});

describe('Repair and Turn all off', () => {
  it('asks before Repair, posts nothing on Cancel, and shows the note after', async () => {
    serve();
    const reply = gate();
    const sent: unknown[] = [];
    server.use(
      http.post('/services', async ({ request }) => {
        sent.push(await request.json());
        await reply.shut;
        return HttpResponse.json({ ...view(), note: 'Left alone, running by hand: notes.' });
      }),
    );
    const { section } = await openSection();
    await userEvent.click(await within(section).findByRole('button', { name: 'Repair' }));
    let ask = await screen.findByRole('dialog', { name: 'Repair services?' });
    expect(ask).toHaveTextContent(
      'Remove and reinstall every enabled service except the dashboard? Each one restarts, so the microphone and cameras drop out for a moment.',
    );
    await userEvent.click(within(ask).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Repair services?' })).not.toBeInTheDocument());
    expect(sent).toEqual([]);
    expect(section).toBeInTheDocument();

    await userEvent.click(within(section).getByRole('button', { name: 'Repair' }));
    ask = await screen.findByRole('dialog', { name: 'Repair services?' });
    await userEvent.click(within(ask).getByRole('button', { name: 'Repair' }));
    await waitFor(() => expect(sent).toEqual([{ all: 'repair' }]));
    expect(within(section).getByRole('button', { name: 'Repairing...' })).toBeDisabled();
    // Every selected service but the dashboard shows it is being started.
    expect(within(row('voice')).getByText('starting')).toBeInTheDocument();
    expect(within(row('notes')).getByText('starting')).toBeInTheDocument();
    expect(within(row('dashboard')).getByText('running')).toBeInTheDocument();

    reply.open();
    expect(await within(section).findByText('Left alone, running by hand: notes.')).toHaveAttribute('role', 'status');
    expect(within(section).queryByRole('alert')).not.toBeInTheDocument();
    expect(within(section).getByRole('button', { name: 'Repair' })).toBeEnabled();
  });

  it('asks before Turn all off, closes only the confirm on Escape, and shows a failure', async () => {
    serve();
    const sent: unknown[] = [];
    server.use(
      http.post('/services', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ...view(), ok: false, error: 'pack-services uninstall timed out' }, { status: 504 });
      }),
    );
    const { section } = await openSection();
    await userEvent.click(await within(section).findByRole('button', { name: 'Turn all off' }));
    const ask = await screen.findByRole('dialog', { name: 'Turn every service off?' });
    expect(ask).toHaveTextContent(
      'Turn off every service except the dashboard? Nothing is listened to, watched, or recorded until you turn them back on.',
    );
    await userEvent.keyboard('{Escape}');
    await waitFor(() =>
      expect(screen.queryByRole('dialog', { name: 'Turn every service off?' })).not.toBeInTheDocument(),
    );
    expect(section).toBeInTheDocument();
    expect(sent).toEqual([]);

    await userEvent.click(within(section).getByRole('button', { name: 'Turn all off' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Turn every service off?' })).getByRole('button', {
        name: 'Turn off',
      }),
    );
    expect(await within(section).findByRole('alert')).toHaveTextContent('pack-services uninstall timed out');
    expect(sent).toEqual([{ all: 'off' }]);
  });
});

describe('groups that need nothing fold away', () => {
  const settle = () => {
    history.replaceState(null, '', '/settings#services');
    return renderPage(<SettingsPage />);
  };
  const toggle = (group: RegExp) => screen.findByRole('button', { name: group });

  it('folds a group that is all running, opens one with a problem, and opens a folded one when a problem comes', async () => {
    const ok = svc({ name: 'notes', group: 'workers', installed: true, state: 'active' });
    const down = svc({ name: 'voice', group: 'core', installed: true, state: 'failed' });
    serve(view({ services: [ok, down] }));
    const { client } = settle();
    const workers = await toggle(/^Show Workers/);
    expect(workers).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('listitem', { name: 'notes' })).toBeNull(); // folded to its heading and count
    expect(await toggle(/^Hide /)).toHaveAttribute('aria-expanded', 'true'); // the group with voice failed
    expect(screen.getByRole('listitem', { name: 'voice' })).toBeInTheDocument();
    await userEvent.click(workers);
    expect(screen.getByRole('listitem', { name: 'notes' })).toBeInTheDocument();
    await userEvent.click(await toggle(/^Hide Workers/));
    expect(screen.queryByRole('listitem', { name: 'notes' })).toBeNull();
    // notes fails: its group opens on its own
    client.setQueryData(['services'], view({ services: [{ ...ok, state: 'failed' }, down] }));
    expect(await screen.findByRole('listitem', { name: 'notes' })).toBeInTheDocument();
  });
});
