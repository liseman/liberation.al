import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { CONFIG_KEY, type ConfigView, type Setting } from './model';
import { groups, SettingsPage } from './SettingsPage';

// A save that restarts would poll GET /health until the dashboard came back, long after its test ended, and
// trip a later test's unhandled-request check: wait forever instead, as SettingsForm.test.tsx does.
vi.mock('./restart', () => ({ followRestart: vi.fn(() => new Promise<void>(() => {})) }));

const row = (s: Partial<Setting> & Pick<Setting, 'key'>): Setting => ({
  type: 'str',
  value: '',
  default: '',
  secret: false,
  description: '',
  label: '',
  help: '',
  section: '',
  ...s,
});
const VIEW: ConfigView = {
  ok: true,
  control: true,
  needs_setup: false,
  path: 'pack.toml',
  restarts: ['voice', 'dashboard'],
  sections: [
    {
      name: 'owner',
      description: '',
      settings: [
        row({
          key: 'owner.name',
          value: 'Joe',
          label: 'Your name',
          help: 'What the assistant calls you',
          section: 'you',
        }),
      ],
    },
    {
      name: 'assistant',
      description: '',
      settings: [
        row({
          key: 'assistant.name',
          value: 'Bob',
          default: 'Bob',
          label: "Assistant's name",
          help: 'What you call it',
          section: 'you',
        }),
      ],
    },
    {
      name: 'phone',
      description: '',
      settings: [
        row({ key: 'phone.public_host', label: 'Public address', help: 'Reaches the phone bridge', section: 'phone' }),
      ],
    },
    {
      name: 'presence',
      description: '',
      settings: [
        row({ key: 'presence.gap_s', type: 'int', value: 45, default: 30, description: 'Longest break in one visit' }),
      ],
    },
    {
      name: 'journal',
      description: '',
      settings: [
        row({
          key: 'journal.enabled',
          type: 'bool',
          value: false,
          default: false,
          feature: 'The journal',
          section: 'features',
          description: 'Writes a diary of the day.',
        }),
        row({ key: 'agent.token', secret: true, set: false, description: 'The agent token' }),
        row({
          key: 'owner.voice_ref',
          value: 'data/voices/owner_ref.wav',
          label: 'Your voice reference',
          help: 'About 15 seconds of you',
          section: 'you',
        }),
      ],
    },
  ],
};
let posts: unknown[];

function open(hash = '', view: ConfigView = VIEW, reply: Record<string, unknown> = { ok: true }) {
  history.replaceState(null, '', `/settings${hash}`);
  server.use(
    http.get('/config', () => HttpResponse.json(view)),
    http.get('/health', () => HttpResponse.json({ boot: 'b1' })),
    http.get('/config/voice_ref', () => HttpResponse.json({ exists: false })),
    // Cameras and microphones: its menu badge reads vision's and the microphones' live status
    http.get('/vision', () => HttpResponse.json({})),
    http.get('/ambient', () => HttpResponse.json({})),
    http.get('/devices', () => HttpResponse.json({ ok: true, control: true, notes: {}, mics: [], cameras: [] })),
    http.get('/services', () => HttpResponse.json({ ok: true, control: true, services: [] })), // the menu badge
    // Network cameras and Imported media read the ingest sources
    http.get('/ingest/api/items', () =>
      HttpResponse.json({ items: [], jobs: [], control: true, sources: [], wyze: { cameras: [], rtsp_url: '' } }),
    ),
    http.post('/config', async ({ request }) => {
      posts.push(await request.json());
      return HttpResponse.json(reply);
    }),
  );
  return renderPage(<SettingsPage />);
}
const nav = () => screen.getByRole('navigation', { name: 'Settings sections' });
const bar = () => screen.queryByRole('region', { name: 'Unsaved changes' });
/** Opens Advanced's sections by their titles: each folds to its heading and count. */
const showAdvanced = async (...titles: string[]) => {
  for (const t of titles) await userEvent.click(await screen.findByRole('button', { name: `Show ${t}` }));
};

beforeEach(() => {
  posts = [];
  // A named camera, so the Devices badge doesn't ask for one (devices/issues.test.tsx covers that)
  vi.stubGlobal('NAMES', { views: { cam0: 'front camera' } });
});
afterEach(() => {
  history.replaceState(null, '', '/');
  vi.unstubAllGlobals();
});

describe('the Settings page', () => {
  it('lists the sections, and shows a setting by its plain label and help', async () => {
    open();
    const items = within(await screen.findByRole('navigation', { name: 'Settings sections' })).getAllByRole('link');
    expect(items.map(a => a.textContent)).toEqual([
      'You and the assistant',
      'Devices',
      'Imported media',
      'Features',
      'People and privacy',
      'Voice clones',
      'Phone',
      'Services',
      'This machine',
      'Advanced',
    ]);
    expect(screen.getByRole('textbox', { name: 'Your name' })).toHaveValue('Joe');
    expect(screen.getByText('What the assistant calls you')).toBeInTheDocument();
    expect(bar()).toBeNull(); // nothing changed yet
  });

  it('shows every setting in Advanced by its key, Default or Changed', async () => {
    open('#advanced');
    await showAdvanced('Presence', 'Assistant');
    const gap = await screen.findByRole('spinbutton', { name: 'presence.gap_s' });
    expect(gap).toHaveValue(45);
    expect(screen.getByRole('group', { name: 'presence.gap_s' })).toHaveAccessibleDescription(
      'Longest break in one visit',
    );
    expect(within(screen.getByRole('group', { name: 'presence.gap_s' })).getByText('Changed')).toBeInTheDocument();
    expect(within(screen.getByRole('group', { name: 'assistant.name' })).getByText('Default')).toBeInTheDocument();
  });

  it('marks a change, resets it, and says in the save bar how many and what a save restarts', async () => {
    open();
    const name = await screen.findByRole('textbox', { name: 'Your name' });
    await userEvent.clear(name);
    await userEvent.type(name, 'Jordan');
    expect(bar()).toHaveTextContent('1 unsaved change. Saving restarts: voice, dashboard');
    expect(within(nav()).getByRole('link', { name: /You and the assistant/ })).toHaveTextContent('1 unsaved');
    await userEvent.click(screen.getByRole('button', { name: 'Reset Your name' }));
    expect(name).toHaveValue('Joe');
    expect(bar()).toBeNull();
  });

  it('throws the draft away on Discard', async () => {
    open();
    await userEvent.type(await screen.findByRole('textbox', { name: "Assistant's name" }), 'by');
    await userEvent.click(within(bar() as HTMLElement).getByRole('button', { name: 'Discard' }));
    expect(screen.getByRole('textbox', { name: "Assistant's name" })).toHaveValue('Bob');
  });

  it('saves the draft, with Save and restart', async () => {
    open();
    await userEvent.type(await screen.findByRole('textbox', { name: "Assistant's name" }), 'by');
    await userEvent.click(within(bar() as HTMLElement).getByRole('button', { name: 'Save and restart' }));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toEqual({ values: { 'assistant.name': 'Bobby' } }); // only what changed here
  });

  it('will not save with a required setting blank, and says which and where', async () => {
    open();
    await userEvent.clear(await screen.findByRole('textbox', { name: 'Your name' }));
    expect(bar()).toHaveTextContent('Fill in Your name to save.');
    expect(within(bar() as HTMLElement).getByRole('button', { name: 'Save and restart' })).toBeDisabled();
    expect(within(nav()).getByRole('link', { name: /You and the assistant/ })).toHaveTextContent('Needs attention');
  });

  it('finds settings across every section, by their words or their key', async () => {
    open();
    await userEvent.type(await screen.findByRole('searchbox', { name: 'Search settings' }), 'gap');
    expect(screen.getByRole('heading', { name: '1 setting match' })).toBeInTheDocument();
    expect(screen.getByRole('spinbutton', { name: 'presence.gap_s' })).toBeInTheDocument();
    await userEvent.clear(screen.getByRole('searchbox', { name: 'Search settings' }));
    await userEvent.type(screen.getByRole('searchbox', { name: 'Search settings' }), 'bridge');
    expect(screen.getByRole('heading', { name: 'Phone' })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Public address' })).toBeInTheDocument();
  });

  it('opens the section of a setting a link names, and focuses it', async () => {
    open('#set-phone.public_host');
    const host = await screen.findByRole('textbox', { name: 'Public address' });
    await waitFor(() => expect(host).toHaveFocus());
    expect(within(nav()).getByRole('link', { name: 'Phone' })).toHaveAttribute('aria-current', 'page');
  });

  it('opens and focuses the setting a refused save names, and says why', async () => {
    open('', VIEW, { ok: false, error: 'phone.public_host needs a port', field: 'phone.public_host' });
    await userEvent.type(await screen.findByRole('textbox', { name: "Assistant's name" }), 'by');
    await userEvent.click(within(bar() as HTMLElement).getByRole('button', { name: 'Save and restart' }));
    const host = await screen.findByRole('textbox', { name: 'Public address' });
    await waitFor(() => expect(host).toHaveFocus());
    expect(within(bar() as HTMLElement).getByRole('alert')).toHaveTextContent('phone.public_host needs a port');
  });
});

describe('review cases', () => {
  it('names a new feature by its name, says what it does (not its default), and finds it by name', async () => {
    open('#features');
    const journal = await screen.findByRole('switch', { name: 'The journal' });
    expect(screen.getByText('Writes a diary of the day.')).toBeInTheDocument();
    expect(screen.queryByText(/by default/)).toBeNull();
    // Switched from how it comes, it says so
    await userEvent.click(journal);
    expect(screen.getByText('Changed: off by default')).toBeInTheDocument();
    await userEvent.click(journal);
    expect(screen.queryByText(/by default/)).toBeNull();
    await userEvent.type(screen.getByRole('searchbox', { name: 'Search settings' }), 'the journal');
    expect(screen.getByRole('heading', { name: '1 setting match' })).toBeInTheDocument();
  });

  it('takes a fresh view as what is saved, keeping only what was changed here', async () => {
    const { client } = open();
    await userEvent.type(await screen.findByRole('textbox', { name: "Assistant's name" }), 'by');
    const fresh = structuredClone(VIEW);
    const phone = fresh.sections[2]?.settings[0] as Setting;
    phone.value = 'pack.example:443'; // changed elsewhere meanwhile
    client.setQueryData(CONFIG_KEY, fresh);
    await userEvent.click(within(nav()).getByRole('link', { name: 'Phone' }));
    expect(screen.getByRole('textbox', { name: 'Public address' })).toHaveValue('pack.example:443');
    expect(bar()).toHaveTextContent('1 unsaved change'); // still the assistant's name, and only that
  });

  it('lands a link to the voice reference on the recorder', async () => {
    open('#set-owner.voice_ref');
    await screen.findByRole('group', { name: 'Your voice reference' });
    await waitFor(() => expect(document.activeElement?.id).toBe('set-owner.voice_ref'));
  });

  it('says it is read only for a viewer not on this machine', async () => {
    open('', { ...VIEW, control: false });
    expect(
      await screen.findByText('Settings can be changed only from the machine Liberational runs on.'),
    ).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Your name' })).toBeDisabled();
  });

  it('lets Discard clear a refused save once its changes are reset by hand', async () => {
    open('', VIEW, { ok: false, error: 'No.' });
    await userEvent.type(await screen.findByRole('textbox', { name: "Assistant's name" }), 'by');
    await userEvent.click(within(bar() as HTMLElement).getByRole('button', { name: 'Save and restart' }));
    await within(bar() as HTMLElement).findByRole('alert');
    await userEvent.click(screen.getByRole('button', { name: "Reset Assistant's name" }));
    await userEvent.click(within(bar() as HTMLElement).getByRole('button', { name: 'Discard' }));
    expect(bar()).toBeNull();
  });

  it('shows a secret typed in Advanced as changed', async () => {
    open('#advanced');
    await showAdvanced('Journal');
    const secret = await screen.findByRole('group', { name: 'agent.token' });
    expect(within(secret).getByText('Default')).toBeInTheDocument();
    await userEvent.type(within(secret).getByPlaceholderText('Not set'), 'tok');
    expect(within(secret).getByText('Changed')).toBeInTheDocument();
  });
});

describe('You and the assistant', () => {
  const YOU: ConfigView = {
    ...VIEW,
    sections: [
      ...VIEW.sections,
      {
        name: 'you',
        description: '',
        settings: [
          row({
            key: 'tts.voice',
            value: 'af_heart',
            default: 'af_heart',
            label: 'Fallback voice',
            help: 'Kokoro fallback voice',
            section: 'you',
          }),
          row({ key: 'time.timezone', label: 'Time zone', help: 'For dates in history questions', section: 'you' }),
          row({
            key: 'assistant.wake_words',
            type: 'list',
            value: [],
            default: [],
            label: 'Wake words',
            help: 'Say one',
            section: 'you',
          }),
        ],
      },
    ],
  };
  const voices = (clone: boolean) =>
    server.use(
      http.get('/config/voices', () => HttpResponse.json({ ok: true, clone, kokoro: ['af_heart', 'bm_george'] })),
    );

  it("lists a section's settings in labels.toml's order, not defaults.toml's", async () => {
    voices(false);
    const ordered = structuredClone(YOU);
    for (const sec of ordered.sections)
      for (const r of sec.settings) r.order = { 'time.timezone': 0, 'owner.name': 1 }[r.key] ?? 5;
    open('', ordered);
    await screen.findByRole('textbox', { name: 'Your name' });
    const labels = screen.getAllByRole('group').map(g => g.getAttribute('aria-labelledby'));
    expect(labels.slice(0, 2)).toEqual(['set-time.timezone-label', 'set-owner.name-label']);
  });

  it('offers the fallback voice where it can clone one, saying it speaks in the clone', async () => {
    voices(true);
    open('', YOU);
    const pick = await screen.findByRole('combobox', { name: 'Fallback voice' });
    await waitFor(() => expect(pick).toHaveValue('Heart (American English, female)'));
    // The value is the draft's at once; the note comes with the voices (its line held blank until then)
    await waitFor(() =>
      expect(pick).toHaveAccessibleDescription('It speaks in your cloned voice. This one is for when it can’t.'),
    );
  });

  it('offers the Kokoro voices by name where it cannot clone one', async () => {
    voices(false);
    open('', YOU);
    const pick = await screen.findByRole('combobox', { name: 'Fallback voice' });
    await waitFor(() => expect(pick).toHaveValue('Heart (American English, female)'));
    await userEvent.click(pick);
    await userEvent.click(await screen.findByRole('option', { name: 'George (British English, male)' }));
    expect(bar()).toHaveTextContent('1 unsaved change');
  });

  it("offers every time zone, and this machine's first", async () => {
    voices(false);
    open('', YOU);
    const zone = await screen.findByRole('combobox', { name: 'Time zone' });
    expect(zone).toHaveValue(`This machine's (${Intl.DateTimeFormat().resolvedOptions().timeZone})`);
    await userEvent.clear(zone);
    await userEvent.type(zone, 'Los Ang');
    await userEvent.click(await screen.findByRole('option', { name: 'America/Los Angeles' }));
    await userEvent.click(within(bar() as HTMLElement).getByRole('button', { name: 'Save and restart' }));
    await waitFor(() => expect(posts[0]).toEqual({ values: { 'time.timezone': 'America/Los_Angeles' } }));
  });

  it('suggests "hey" and the assistant\'s name for the wake words', async () => {
    voices(false);
    open('', YOU);
    expect(await screen.findByPlaceholderText('hey bob')).toBeInTheDocument();
  });

  it('keeps the time zone when the one already picked is picked again', async () => {
    voices(false);
    const set = structuredClone(YOU);
    for (const sec of set.sections)
      for (const r of sec.settings) if (r.key === 'time.timezone') r.value = 'America/New_York';
    open('', set);
    const zone = await screen.findByRole('combobox', { name: 'Time zone' });
    await userEvent.click(zone);
    await userEvent.click(await screen.findByRole('option', { name: 'America/New York' }));
    expect(zone).toHaveValue('America/New York');
    expect(bar()).toBeNull();
  });

  it('takes a voice as free text where there are no voices to list', async () => {
    server.use(http.get('/config/voices', () => HttpResponse.json({ ok: true, clone: false, kokoro: [] })));
    open('', YOU);
    const box = await screen.findByRole('textbox', { name: 'Fallback voice' });
    await userEvent.clear(box);
    await userEvent.type(box, 'am_adam');
    expect(bar()).toHaveTextContent('1 unsaved change');
  });

  it("offers to try again when the voice reference can't be loaded", async () => {
    voices(false);
    open();
    server.use(http.get('/config/voice_ref', () => HttpResponse.json({ ok: false, error: 'boom' }, { status: 500 })));
    expect(await screen.findByText("Couldn't load your voice reference")).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
  });

  it('shows the voice reference in a line, and records it again in a dialog', async () => {
    voices(false);
    open();
    server.use(
      http.get('/config/voice_ref', () =>
        HttpResponse.json({
          ok: true,
          exists: true,
          seconds: 15,
          recorded: 1792800000,
          path: 'x.wav',
          prompt: 'Read this.',
        }),
      ),
    );
    const again = await screen.findByRole('button', { name: 'Record again' });
    expect(screen.getByRole('button', { name: 'Play your voice reference' })).toBeEnabled();
    expect(screen.getByText(/^Recorded .*, 15 s$/)).toBeInTheDocument();
    await userEvent.click(again);
    const dialog = await screen.findByRole('dialog', { name: 'Record your voice' });
    expect(within(dialog).getByText('Read this.')).toBeInTheDocument();
    expect(within(dialog).queryByText(/is saved at/)).toBeNull(); // the row above says that
  });
});

describe('restarting only what a change affects', () => {
  const TARGETED: ConfigView = structuredClone(VIEW);
  for (const sec of TARGETED.sections)
    for (const r of sec.settings) r.restarts = r.key === 'assistant.name' ? [] : ['voice', 'dashboard'];

  it("names what the changes restart, and nothing for a live setting's", async () => {
    open('', TARGETED);
    const name = await screen.findByRole('textbox', { name: "Assistant's name" });
    await userEvent.clear(name);
    await userEvent.type(name, 'Ada');
    expect(bar()).toHaveTextContent('1 unsaved change. Saving restarts nothing.');
    expect(within(bar() as HTMLElement).getByRole('button', { name: 'Save' })).toBeEnabled();
    const you = screen.getByRole('textbox', { name: 'Your name' });
    await userEvent.type(you, 'x');
    expect(bar()).toHaveTextContent('2 unsaved changes. Saving restarts: voice, dashboard');
    expect(within(bar() as HTMLElement).getByRole('button', { name: 'Save and restart' })).toBeEnabled();
  });

  it("doesn't say what restarts when the server can't tell", async () => {
    open('', { ...VIEW, restarts: [] }); // an unknown service in pack.toml: no rows say either
    const name = await screen.findByRole('textbox', { name: "Assistant's name" });
    await userEvent.type(name, 'x');
    expect(within(bar() as HTMLElement).getByRole('status')).toHaveTextContent(/^1 unsaved change\.$/);
    expect(within(bar() as HTMLElement).getByRole('button', { name: 'Save and restart' })).toBeEnabled();
  });

  it('clears a secret just saved on a save that stays on the page', async () => {
    const withSecret: ConfigView = structuredClone(TARGETED);
    withSecret.sections[0]?.settings.push(
      row({
        key: 'phone.api_secret',
        secret: true,
        label: 'API secret',
        help: 'Its key',
        section: 'you',
        restarts: [],
      }),
    );
    open('', withSecret, { ok: true, restarted: [], restarting: false });
    await screen.findAllByText('API secret');
    const token = document.getElementById('set-phone.api_secret') as HTMLInputElement;
    await userEvent.type(token, 'sekrit');
    await userEvent.click(within(bar() as HTMLElement).getByRole('button', { name: 'Save' }));
    expect(await screen.findByText('Saved to pack.toml.')).toBeInTheDocument();
    await waitFor(() => expect(token).toHaveValue(''));
  });

  it('stays on the page after a save that leaves the dashboard running', async () => {
    let saved = false;
    open('', TARGETED, { ok: true, restarted: [], restarting: false });
    server.use(
      http.get('/config', () => {
        const v = structuredClone(TARGETED);
        if (saved)
          for (const sec of v.sections) for (const r of sec.settings) if (r.key === 'assistant.name') r.value = 'Ada';
        return HttpResponse.json(v);
      }),
      http.post('/config', async ({ request }) => {
        posts.push(await request.json());
        saved = true;
        return HttpResponse.json({ ok: true, restarted: [], restarting: false });
      }),
    );
    const name = await screen.findByRole('textbox', { name: "Assistant's name" });
    await userEvent.clear(name);
    await userEvent.type(name, 'Ada');
    await userEvent.click(within(bar() as HTMLElement).getByRole('button', { name: 'Save' }));
    // Exactly this: a save that restarts says "Restarting with the new settings" and waits for the dashboard
    expect(await screen.findByText('Saved to pack.toml.')).toBeInTheDocument();
    expect(name).toHaveValue('Ada');
    await userEvent.type(name, 'm'); // a new change shows as one again
    expect(bar()).toHaveTextContent('1 unsaved change.');
  });
});

describe('Features, People and privacy, and Phone', () => {
  const GROUPED: ConfigView = {
    ...VIEW,
    sections: [
      {
        name: 'journal',
        description: '',
        settings: [
          row({
            key: 'journal.enabled',
            type: 'bool',
            value: true,
            default: true,
            label: 'The journal',
            help: 'A diary',
            section: 'features',
            group: 'Memory and answers',
          }),
          row({
            key: 'dashboard.hotkeys',
            type: 'bool',
            value: false,
            default: false,
            label: 'Keyboard shortcuts',
            help: 'Keys',
            section: 'features',
            group: 'Dashboard',
          }),
          row({
            key: 'journal.facts',
            type: 'bool',
            value: true,
            default: true,
            label: 'Facts about people',
            help: 'Facts',
            section: 'features',
            group: 'Memory and answers',
          }),
          row({
            key: 'journal.other',
            type: 'bool',
            value: true,
            default: true,
            label: 'Ungrouped',
            help: 'Plain',
            section: 'features',
          }),
        ],
      },
    ],
  };

  it('keeps one Other card when labels.toml has a group of that name', () => {
    const r = (key: string, group?: string) => row({ key, group });
    expect(
      groups([r('a.x', 'Other'), r('a.y'), r('a.z', 'Memory')]).map(([g, rows]) => [g, rows.map(x => x.key)]),
    ).toEqual([
      ['Other', ['a.x', 'a.y']],
      ['Memory', ['a.z']],
    ]);
    expect(groups([r('a.y')]).map(([g]) => g)).toEqual(['']); // a section with no groups: one untitled card
  });

  it('keeps each group heading inside its card, with ungrouped settings last under Other', async () => {
    open('#features', GROUPED);
    await screen.findByRole('switch', { name: 'The journal' });
    const memory = screen.getByRole('region', { name: 'Memory and answers' });
    expect(within(memory).getByRole('heading', { level: 3, name: 'Memory and answers' })).toBeInTheDocument();
    expect(
      within(memory)
        .getAllByRole('switch')
        .map(s => s.getAttribute('aria-labelledby')),
    ).toEqual(['set-journal.enabled-label', 'set-journal.facts-label']);
    expect(
      within(screen.getByRole('region', { name: 'Dashboard' })).getByRole('switch', { name: 'Keyboard shortcuts' }),
    ).toBeInTheDocument();
    const other = screen.getByRole('region', { name: 'Other' }); // a setting with no group comes last
    expect(within(other).getByRole('switch', { name: 'Ungrouped' })).toBeInTheDocument();
    expect(screen.getAllByRole('switch').at(-1)?.getAttribute('aria-labelledby')).toBe('set-journal.other-label');
  });
});

describe('Advanced', () => {
  const RAW: ConfigView = {
    ...VIEW,
    sections: [
      {
        name: 'presence',
        description: 'How visits are counted. More words follow.',
        settings: [
          row({
            key: 'presence.gap_s',
            type: 'float',
            value: 45,
            default: 30,
            description: 'Longest break in a visit',
          }),
          row({
            key: 'presence.chip_context_s',
            type: 'float',
            value: 10,
            default: 10,
            description: 'Context around a chip',
          }),
        ],
      },
      {
        name: 'timeline',
        description: 'Encounters for Ask.',
        settings: [
          row({ key: 'timeline.gap_s', type: 'float', value: 120, default: 120, description: 'Break inside one' }),
        ],
      },
    ],
  };
  const keys = () => screen.queryAllByRole('group').map(g => g.getAttribute('aria-labelledby'));

  it('shows a card per config section, titled and noted', async () => {
    open('#advanced', RAW);
    const presence = await screen.findByRole('region', { name: 'Presence' });
    expect(within(presence).getByText('How visits are counted.')).toBeInTheDocument();
    expect(within(presence).getByText('2 settings')).toBeInTheDocument(); // folded to its count
    await showAdvanced('Presence');
    expect(within(presence).getByText('Changed')).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Timeline' })).toBeInTheDocument();
  });

  it('folds each section until it is opened, a filter is on, or a link names a setting in it', async () => {
    open('#advanced', RAW);
    const toggle = await screen.findByRole('button', { name: 'Show Presence' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('group', { name: 'presence.gap_s' })).toBeNull();
    await userEvent.click(toggle);
    expect(screen.getByRole('group', { name: 'presence.gap_s' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Hide Presence' }));
    expect(screen.queryByRole('group', { name: 'presence.gap_s' })).toBeNull();
    // a link to one setting opens its section
    act(() => {
      location.hash = '#set-timeline.gap_s';
      window.dispatchEvent(new HashChangeEvent('hashchange'));
    });
    expect(await screen.findByRole('group', { name: 'timeline.gap_s' })).toBeInTheDocument();
  });

  it('filters by key or words', async () => {
    open('#advanced', RAW);
    const filter = await screen.findByRole('textbox', { name: 'Filter settings' });
    await userEvent.type(filter, 'gap');
    expect(keys()).toEqual(['set-presence.gap_s-label', 'set-timeline.gap_s-label']);
    await userEvent.clear(filter);
    await userEvent.type(filter, 'around a chip');
    expect(keys()).toEqual(['set-presence.chip_context_s-label']);
    expect(screen.queryByRole('region', { name: 'Timeline' })).toBeNull(); // a section with nothing left goes
    await userEvent.type(filter, 'zzz');
    expect(screen.getByRole('status')).toHaveTextContent('No setting matches.');
  });

  it('keeps showing a row being edited back to its default under Only changed', async () => {
    open('#advanced', RAW);
    await userEvent.click(await screen.findByRole('checkbox', { name: 'Only changed from default' }));
    const gap = document.getElementById('set-presence.gap_s') as HTMLInputElement;
    await userEvent.clear(gap);
    await userEvent.type(gap, '30');
    expect(keys()).toContain('set-presence.gap_s-label');
    expect(gap).toHaveFocus();
  });

  it('drops the filter for a link to a setting it hides', async () => {
    open('#advanced', RAW);
    await userEvent.type(await screen.findByRole('textbox', { name: 'Filter settings' }), 'timeline');
    expect(keys()).toEqual(['set-timeline.gap_s-label']);
    location.hash = '#set-presence.gap_s';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    await waitFor(() => expect(keys()).toContain('set-presence.gap_s-label'));
    expect(screen.getByRole('textbox', { name: 'Filter settings' })).toHaveValue('');
  });

  it('shows only the settings changed from default', async () => {
    open('#advanced', RAW);
    await userEvent.click(await screen.findByRole('checkbox', { name: 'Only changed from default' }));
    expect(keys()).toEqual(['set-presence.gap_s-label']);
    const filter = screen.getByRole('textbox', { name: 'Filter settings' });
    await userEvent.type(filter, 'timeline');
    expect(screen.getByRole('status')).toHaveTextContent('No setting matches among those changed from default.');
  });
});

describe("Settings' sources", () => {
  it('opens Imported media from a link to its card, and focuses the card', async () => {
    open('#imported-media');
    expect(await screen.findByRole('heading', { name: 'Imported media', level: 2 })).toBeInTheDocument();
    const card = await screen.findByRole('region', { name: 'Accounts and apps' });
    await waitFor(() => expect(card).toHaveFocus());
  });

  it('opens Devices from a link to its Network cameras card', async () => {
    open('#network-cameras');
    expect(await screen.findByRole('heading', { name: 'Devices', level: 2 })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole('region', { name: 'Network cameras' })).toHaveFocus());
  });

  it('sends an old link to the phones, watches, and tags card (/settings#device-links) to Known › Devices', async () => {
    open('#device-links');
    await waitFor(() => expect(location.pathname).toBe('/devices'));
    expect(location.hash).toBe('');
  });

  it("indexes Devices' cards at its top, and #devices still opens the section", async () => {
    open('#devices');
    expect(await screen.findByRole('heading', { name: 'Devices', level: 2 })).toBeInTheDocument();
    expect(screen.getByText(/Phones, watches, and tags are in Known › Devices\./)).toBeInTheDocument();
    expect(await screen.findByRole('link', { name: 'Known › Devices' })).toHaveAttribute('href', '/devices');
    expect(screen.queryByRole('region', { name: 'Phones, watches, and tags' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('link', { name: 'Microphones' }));
    await waitFor(() => expect(screen.getByRole('region', { name: 'Microphones' })).toHaveFocus());
  });
});
