import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { newQueryClient, Providers } from '../shared/Providers';
import { PREFS_KEY } from '../shared/prefs';
import { server } from '../test/server';
import { App } from './App';
import { useReviewAsk } from './AskBar';
import { liveKey } from './live/feed';
import { useLiveStatus } from './live/status';
import { knownTabOf, TABS, tabOf, useRoute } from './router';
import { setActiveCameras, shell } from './shell';

// Review lists the day's video from the gateway; there is none in a test, so the day has no video.
vi.mock('./review/gatewaySegments', async original => ({
  ...(await original<typeof import('./review/gatewaySegments')>()),
  daySegments: vi.fn(async () => []),
}));

const NOW = 1_800_000_000;

/** An EventSource that never connects: the shell renders from what each test puts in the cache. */
class QuietEventSource {
  static readonly CLOSED = 2;
  readyState = 0;
  onopen = null;
  onerror = null;
  addEventListener() {}
  close() {}
}

function renderApp(path = '/', live: { ambient?: unknown; scene?: unknown } = {}) {
  history.pushState(null, '', path);
  useRoute.setState({
    tab: tabOf(location.pathname),
    known: knownTabOf(location.pathname) ?? 'people',
    settingsReturnTo: undefined,
  });
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { ...client.getDefaultOptions().queries, retry: false } });
  client.setQueryData(liveKey('ambient'), live.ambient ?? {});
  client.setQueryData(liveKey('vision'), {});
  client.setQueryData(liveKey('scene'), live.scene ?? []);
  client.setQueryData(liveKey('transcript'), { entries: [], resets: 1 });
  client.setQueryData(liveKey('speakers'), { list: [], byId: {}, merged: {} });
  return render(
    <Providers client={client} test>
      <App />
    </Providers>,
  );
}

beforeEach(() => {
  useReviewAsk.setState({ open: false }); // the one Review Ask state outlives a render, as it does in the app
  vi.stubGlobal('EventSource', QuietEventSource);
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Luca', cams: ['cam0', 'cam1'], hotkeys: false });
  useLiveStatus.setState({ now: NOW, conn: { state: 'on', text: 'live' } });
  server.use(
    http.get('/config', () =>
      HttpResponse.json({ ok: true, control: true, needs_setup: false, path: 'pack.toml', sections: [] }),
    ),
    // Getting started's reads (#296): nothing set up, so it shows; tests about it set their own.
    http.get('/stats', () => HttpResponse.json({})),
    http.get('/devices', () => HttpResponse.json({ control: true, mics: [], cameras: [], notes: {} })),
    http.get('/config/voice_ref', () => HttpResponse.json({ exists: false })),
    http.get('/services', () => HttpResponse.json({ ok: true, control: true, platform: 'linux', services: [] })),
    http.get('/people', () => HttpResponse.json({ people: [] })),
    http.get('/pets', () => HttpResponse.json({ pets: [], species: ['cat', 'dog', 'bird'] })),
    http.get('/objects/api/status', () => HttpResponse.json({ cameras: [], objects: 0, observer: null })),
    http.get('/objects/api/objects', () => HttpResponse.json({ objects: [] })),
    http.get('/live/happening', () => HttpResponse.json({ items: [], updated: null })),
    http.get('/speakers/refine', () => HttpResponse.json({ items: [], remaining: 0 })),
    http.get('/calls', () => HttpResponse.json([])),
    http.get('/clones', () =>
      HttpResponse.json({
        clones: [
          { id: 1, label: 'Alice', ready: false, built: false },
          { id: 3, label: 'Dana', ready: true, built: false, clean_s: 14 },
        ],
      }),
    ),
    http.get('/transcript/stats', () => HttpResponse.json({ utterances: 0, speakers: 0, speech_s: 0 })),
    http.get('/summarize/status', () => HttpResponse.json({ running: false })),
    http.get('/summary', ({ request }) =>
      HttpResponse.json({
        day: new URL(request.url).searchParams.get('day') ?? '',
        md: '',
        status: 'empty',
      }),
    ),
    http.get('/transcript/day', ({ request }) =>
      HttpResponse.json({
        day: new URL(request.url).searchParams.get('day') ?? '',
        entries: [],
      }),
    ),
    http.get('/audio/spans', ({ request }) =>
      HttpResponse.json({
        day: new URL(request.url).searchParams.get('day') ?? '',
        spans: [],
      }),
    ),
    http.get('/review/presence', () => HttpResponse.json({ chips: [] })),
    http.get('/activities', () => HttpResponse.json({ items: [], truncated: false })),
    http.get('/review/summary', () =>
      HttpResponse.json({ start: 0, end: 0, level: 3600, status: 'ready', blocks: [] }),
    ),
    http.get('/review/days', () => HttpResponse.json({ days: [], today: '2026-10-28' })),
    http.get('/voices/unknown/groups', () => HttpResponse.json({ groups: [], count: 0, skipped: [] })),
    http.get('/faces/unknown/groups', () => HttpResponse.json({ groups: [], count: 0, skipped: [] })),
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
  history.pushState(null, '', '/');
});

/** Settings is in the top bar's menu (#2453). */
async function openSettings() {
  await userEvent.click(screen.getByRole('button', { name: 'Menu' }));
  await userEvent.click(await screen.findByRole('menuitem', { name: 'Settings' }));
}

describe('top bar', () => {
  it('brings the active section into view on direct load and in-app navigation', async () => {
    const scroll = vi.spyOn(Element.prototype, 'scrollIntoView');
    renderApp('/ingest');
    const nav = screen.getByRole('navigation', { name: 'Sections' });
    const ingest = within(nav).getByRole('link', { name: 'Ingest' });
    expect(scroll.mock.contexts).toContain(ingest);
    expect(scroll).toHaveBeenCalledWith({ block: 'nearest', inline: 'nearest' });
    scroll.mockClear();
    const live = within(nav).getByRole('link', { name: 'Live' });
    await userEvent.click(live);
    expect(scroll.mock.contexts).toContain(live);
    expect(live).toHaveAttribute('aria-current', 'page');
    scroll.mockClear();
    act(() => window.dispatchEvent(new Event('resize')));
    expect(scroll.mock.contexts).toContain(live);
  });

  it('keeps Ask, the status, and one menu in the bar, with Settings in the menu (#2453)', async () => {
    renderApp('/');
    const header = document.querySelector('header') as HTMLElement;
    expect(within(header).getByRole('button', { name: 'Ask' })).toHaveAttribute('aria-keyshortcuts', '/');
    expect(within(header).getByRole('button', { name: /Microphone: / })).toBeInTheDocument();
    expect(within(header).queryByRole('button', { name: /Appearance|Resource monitor/ })).toBeNull();
    await userEvent.click(within(header).getByRole('button', { name: 'Menu' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Settings' }));
    expect(await screen.findByRole('dialog', { name: 'Settings' })).toBeInTheDocument();
  });

  it('links every tab and marks the current one', async () => {
    renderApp('/');
    const nav = screen.getByRole('navigation', { name: 'Sections' });
    const links = within(nav).getAllByRole('link');
    expect(links.map(a => [a.textContent, a.getAttribute('href')])).toEqual([
      ['Live', '/'],
      ['Review', '/review'],
      ['Journal', '/journal'],
      ['Known', '/people'],
      ['Calls', '/calls'],
      ['Inbox', '/inbox'],
      ['Reports', '/reports'],
      ['Ingest', '/ingest'],
    ]);
    expect(within(nav).getByRole('link', { name: 'Live' })).toHaveAttribute('aria-current', 'page');
  });

  it("keeps the Resource monitor open across tabs until it's closed (#2258)", async () => {
    server.use(
      http.get('/stats/history', () =>
        HttpResponse.json({ step: 5, window: 1800, cpu_count: 4, mem_total: 1, gpus: [], samples: [] }),
      ),
    );
    renderApp('/');
    const header = document.querySelector('header') as HTMLElement;
    await userEvent.click(within(header).getByRole('button', { name: 'Menu' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Resource monitor' }));
    const panel = await screen.findByRole('region', { name: 'Resource monitor' });
    await userEvent.click(screen.getByRole('link', { name: 'Review' }));
    expect(location.pathname).toBe('/review');
    expect(screen.getByRole('region', { name: 'Resource monitor' })).toBe(panel);
    await userEvent.click(within(panel).getByRole('button', { name: 'Close Resource monitor' }));
    expect(screen.queryByRole('region', { name: 'Resource monitor' })).toBeNull();
  });

  it('moves between tabs in place', async () => {
    renderApp('/');
    await userEvent.click(screen.getByRole('link', { name: 'Review' }));
    expect(location.pathname).toBe('/review');
    expect(screen.getByRole('link', { name: 'Review' })).toHaveAttribute('aria-current', 'page');
    expect(await screen.findByRole('slider', { name: /Timeline/ })).toBeInTheDocument();
    expect(document.title).toBe('Liberational: Review');
  });

  it('shows People, Pets, and Objects as tabs of Known, each at its own path (#1970)', async () => {
    renderApp('/objects'); // the old Objects route opens Known on its Objects tab
    const nav = screen.getByRole('navigation', { name: 'Sections' });
    expect(within(nav).getByRole('link', { name: 'Known' })).toHaveAttribute('aria-current', 'page');
    expect(await screen.findByRole('tab', { name: 'Objects' })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByRole('heading', { name: 'Objects', level: 1 })).toBeVisible();
    expect(document.title).toBe('Liberational: Known');
    await userEvent.click(screen.getByRole('tab', { name: 'Pets' }));
    expect(location.pathname).toBe('/pets');
    expect(screen.getByRole('tab', { name: 'Pets' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('heading', { name: 'Objects', level: 1, hidden: true })).not.toBeVisible(); // kept, hidden
    act(() => history.back());
    await waitFor(() => expect(location.pathname).toBe('/objects'));
    expect(screen.getByRole('tab', { name: 'Objects' })).toHaveAttribute('aria-selected', 'true');
    // Away and back through the nav: Known opens on the tab it was on.
    await userEvent.click(within(nav).getByRole('link', { name: 'Live' }));
    expect(within(nav).getByRole('link', { name: 'Known' })).toHaveAttribute('href', '/objects');
    await userEvent.click(within(nav).getByRole('link', { name: 'Known' }));
    expect(location.pathname).toBe('/objects');
    expect(screen.getByRole('navigation', { name: 'Sections' })).toBe(nav);
  });

  it("opens Devices as Known's tab, with the phones, watches, and tags card Settings had", async () => {
    server.use(
      http.get('/config', () =>
        HttpResponse.json({ ok: true, control: true, needs_setup: false, path: '', sections: [] }),
      ),
    );
    renderApp('/devices');
    expect(await screen.findByRole('tab', { name: 'Devices' })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByRole('heading', { name: 'Devices', level: 1 })).toBeVisible();
    expect(await screen.findByRole('region', { name: 'Phones, watches, and tags' })).toBeInTheDocument();
    expect(document.title).toBe('Liberational: Known');
  });

  it('opens Calls as its own tab and loads the people list for contacts', async () => {
    server.use(
      http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana', voices: [], number: '+14155550123' }] })),
    );
    renderApp('/calls');
    expect(await screen.findByRole('heading', { name: 'Calls', level: 1 })).toBeVisible();
    const nav = screen.getByRole('navigation', { name: 'Sections' });
    expect(within(nav).getByRole('link', { name: 'Calls' })).toHaveAttribute('aria-current', 'page');
    expect(await screen.findByRole('option', { name: 'Dana, +14155550123' })).toBeInTheDocument();
    expect(document.title).toBe('Liberational: Calls');
    await userEvent.click(within(nav).getByRole('link', { name: 'Known' }));
    expect(await screen.findByRole('heading', { name: 'People', level: 1 })).toBeVisible();
    expect(screen.queryByRole('heading', { name: 'Calls' })).not.toBeInTheDocument();
  });

  it('opens People again without asking for its fresh lists (#330)', async () => {
    const asked: string[] = [];
    const lists = ['/people', '/faces/unknown/groups', '/voices/unknown/groups'];
    const count = ({ request }: { request: Request }) => {
      const path = new URL(request.url).pathname;
      // Known's activity badge loads separately; this checks People's fresh lists.
      if (lists.includes(path)) asked.push(path);
    };
    server.events.on('request:start', count);
    try {
      renderApp('/people');
      expect(await screen.findByText('Nobody is waiting for a name.')).toBeVisible();
      expect(asked).toEqual(expect.arrayContaining(lists));
      await userEvent.click(screen.getByRole('link', { name: 'Live' }));
      const before = asked.length;
      await userEvent.click(screen.getByRole('link', { name: 'Known' }));
      await act(() => new Promise(r => setTimeout(r, 100)));
      expect(asked.slice(before)).toEqual([]);
    } finally {
      server.events.removeListener('request:start', count);
    }
  });

  it('keeps a visited tab as it was, hidden, and asks the server nothing again for fresh data (#330)', async () => {
    const asked: string[] = [];
    const count = ({ request }: { request: Request }) => void asked.push(new URL(request.url).pathname);
    server.events.on('request:start', count);
    try {
      renderApp('/review');
      const slider = await screen.findByRole('slider', { name: /Timeline/ });
      // Whole-day search and the debounced timeline range both use /review/summary. Wait for the range's
      // rendered result, not the first request starting: only completed data can be reused after hiding.
      await screen.findByText('No written summary for this stretch.');
      await userEvent.click(screen.getByRole('link', { name: 'Live' }));
      expect(slider).not.toBeVisible(); // hidden, not taken down
      const before = asked.length;
      await userEvent.click(screen.getByRole('link', { name: 'Review' }));
      expect(await screen.findByRole('slider', { name: /Timeline/ })).toBe(slider); // the same element: not rebuilt
      expect(slider).toBeVisible();
      await act(() => new Promise(r => setTimeout(r, 50)));
      expect(asked.slice(before)).toEqual([]);
    } finally {
      server.events.removeListener('request:start', count);
    }
  });
});

describe('status icons', () => {
  const icon = (name: RegExp) => screen.getByRole('img', { name });

  it('shows listening and every camera on', () => {
    renderApp('/', {
      ambient: { ts: NOW - 2 },
      scene: [
        { cam: 'cam0', ts: NOW - 3 },
        { cam: 'cam1', ts: NOW - 4 },
      ],
    });
    expect(icon(/^Listening$/)).toHaveAttribute('data-state', 'on');
    expect(icon(/^All 2 cameras on$/)).toHaveAttribute('data-state', 'on');
    expect(screen.queryByTestId('status-feed')).not.toBeInTheDocument();
  });

  it('updates active camera status immediately without removing recorded camera history', () => {
    renderApp('/', {
      scene: [
        { cam: 'cam0', ts: NOW - 3 },
        { cam: 'cam1', ts: NOW - 4 },
      ],
    });
    expect(icon(/^All 2 cameras on$/)).toBeInTheDocument();
    act(() => setActiveCameras(['cam1']));
    expect(icon(/^Camera on$/)).toBeInTheDocument();
    expect(shell().cams).toEqual(['cam0', 'cam1']);
  });

  it('shows a late microphone and some cameras, with a count', () => {
    renderApp('/', { ambient: { ts: NOW - 40 }, scene: [{ cam: 'cam0', ts: NOW - 3 }] });
    expect(icon(/^Microphone is late: last update 40 s ago$/)).toHaveAttribute('data-state', 'stale');
    const cams = icon(/^1 of 2 cameras on \(cam1 quiet\)$/);
    expect(cams).toHaveAttribute('data-state', 'some');
    expect(screen.getByTestId('status')).toHaveTextContent('1 of 2 cameras');
  });

  it('shows the microphone and cameras off when nothing has reported', () => {
    renderApp('/');
    expect(icon(/^Not listening$/)).toHaveAttribute('data-state', 'off');
    expect(icon(/^Cameras off$/)).toHaveAttribute('data-state', 'off');
  });

  it('turns stale as the clock moves on', () => {
    renderApp('/', { ambient: { ts: NOW - 2 } });
    expect(icon(/^Listening$/)).toBeInTheDocument();
    act(() => useLiveStatus.setState({ now: NOW + 30 }));
    expect(icon(/^Microphone is late/)).toHaveAttribute('data-state', 'stale');
  });

  it('adds a feed icon only while live updates are down', () => {
    renderApp('/');
    act(() => useLiveStatus.setState({ conn: { state: 'bad', text: 'polling' } }));
    expect(icon(/^Live updates off: checking every 3 s$/)).toHaveAttribute('data-state', 'bad');
  });

  it('is one control after Ask, and says its words in the menu', async () => {
    renderApp('/', { ambient: { ts: NOW - 2 } });
    await userEvent.tab(); // skip link
    await userEvent.tab(); // brand
    for (const _ of TABS.filter(t => t.id !== 'debug')) await userEvent.tab(); // tabs
    await userEvent.tab(); // Go to (the command palette, #2586)
    await userEvent.tab(); // Ask
    await userEvent.tab();
    const pill = screen.getByRole('button', { name: /Microphone: Listening\./ });
    expect(pill).toHaveFocus();
    await userEvent.keyboard('{Enter}');
    expect(await screen.findByRole('menu')).toHaveTextContent('Listening');
  });
});

describe('theme', () => {
  it('saves the choice where the old pages read it', async () => {
    renderApp('/');
    await userEvent.click(screen.getByRole('button', { name: 'Menu' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Dark' }));
    expect(JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}')).toMatchObject({ theme: 'dark' });
  });

  it("opens Live's display choices from the same menu, on any tab", async () => {
    renderApp('/review');
    await userEvent.click(screen.getByRole('button', { name: 'Menu' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Live display…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Live display' });
    expect(within(dialog).getByRole('radiogroup', { name: 'Transcript' })).toBeInTheDocument();
  });
});

it('shows first-run setup instead of the tab content and hides the Settings opener', async () => {
  // The server fills needsSetup into the page: the shell doesn't ask /config to know.
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Luca', cams: ['cam0', 'cam1'], hotkeys: false, needsSetup: true });
  server.use(
    http.get('/config', () =>
      HttpResponse.json({ ok: true, control: true, needs_setup: true, path: 'pack.toml', sections: [] }),
    ),
  );
  renderApp('/review');
  expect(await screen.findByRole('heading', { name: 'Set up Liberational' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /^Settings$/ })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Save and start' })).toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: /^Review$/ })).not.toBeInTheDocument();
});

it('opens the Settings route in a dialog over Live and closes a direct link to Live', async () => {
  renderApp('/settings');
  const dialog = await screen.findByRole('dialog', { name: 'Settings' });
  expect(await within(dialog).findByRole('navigation', { name: 'Settings sections' })).toBeInTheDocument();
  expect(screen.getByRole('main')).not.toContainElement(dialog);
  await userEvent.click(within(dialog).getByRole('button', { name: 'Close settings' }));
  await waitFor(() => expect(location.pathname).toBe('/'));
  expect(screen.queryByRole('dialog', { name: 'Settings' })).not.toBeInTheDocument();
});

it('opens Settings with , when single-key shortcuts are on, and not while typing', async () => {
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Alice', cams: ['cam0'], hotkeys: true });
  renderApp('/');
  const box = document.body.appendChild(document.createElement('input')); // any text box on the page
  await userEvent.type(box, ',');
  expect(location.pathname).toBe('/');
  box.remove();
  await userEvent.keyboard(',');
  expect(await screen.findByRole('dialog', { name: 'Settings' })).toBeInTheDocument();
  expect(location.pathname).toBe('/settings');
});

it("puts Settings' search in the drawer's header row and lands focus on its title, not the close button", async () => {
  renderApp('/');
  await openSettings();
  const dialog = await screen.findByRole('dialog', { name: 'Settings' });
  const search = await within(dialog).findByRole('searchbox', { name: 'Search settings' });
  expect(within(screen.getByRole('navigation', { name: 'Settings sections' })).queryByRole('searchbox')).toBeNull();
  const close = within(dialog).getByRole('button', { name: 'Close settings' });
  expect(search.closest('header')).toBe(close.closest('header'));
  const header = close.closest('header') as HTMLElement;
  await waitFor(() => expect(within(header).getByRole('heading', { name: 'Settings' })).toHaveFocus());
});

it('returns to the same Review view and restores its opener after Escape', async () => {
  renderApp('/review#day=2026-10-28');
  const timeline = await screen.findByRole('slider', { name: /Timeline/ });
  const opener = screen.getByRole('button', { name: 'Menu' });
  await openSettings();
  expect(await screen.findByRole('dialog', { name: 'Settings' })).toBeInTheDocument();
  await userEvent.keyboard('{Escape}');
  await waitFor(() => expect(location.pathname).toBe('/review'));
  expect(location.hash).toBe('#day=2026-10-28');
  expect(screen.getByRole('slider', { name: /Timeline/ })).toBe(timeline);
  await waitFor(() => expect(opener).toHaveFocus());
});

it('keeps an unsaved settings draft when the drawer closes and opens again', async () => {
  server.use(
    http.get('/config', () =>
      HttpResponse.json({
        ok: true,
        control: true,
        needs_setup: false,
        path: 'pack.toml',
        sections: [
          {
            name: 'owner',
            description: '',
            settings: [
              {
                key: 'owner.name',
                type: 'str',
                value: 'Alice',
                default: '',
                secret: false,
                description: '',
                label: 'Your name',
                section: 'you',
              },
            ],
          },
        ],
      }),
    ),
  );
  renderApp('/people');
  await openSettings();
  const name = await screen.findByRole('textbox', { name: 'Your name' });
  await userEvent.clear(name);
  await userEvent.type(name, 'Dana');
  await userEvent.click(screen.getByRole('button', { name: 'Close settings' }));
  await waitFor(() => expect(location.pathname).toBe('/people'));
  await openSettings();
  expect(await screen.findByRole('textbox', { name: 'Your name' })).toHaveValue('Dana');
  expect(screen.getByRole('region', { name: 'Unsaved changes' })).toBeInTheDocument();
});

it('closes a Settings child modal with Escape before closing the drawer', async () => {
  server.use(
    http.get('/config', () =>
      HttpResponse.json({
        ok: true,
        control: true,
        needs_setup: false,
        path: 'pack.toml',
        sections: [
          {
            name: 'owner',
            description: '',
            settings: [
              {
                key: 'owner.voice_ref',
                type: 'str',
                value: '',
                default: '',
                secret: false,
                description: '',
                label: 'Your voice',
                section: 'you',
              },
            ],
          },
        ],
      }),
    ),
  );
  renderApp('/settings#you');
  await userEvent.click(await screen.findByRole('button', { name: 'Record' }));
  expect(await screen.findByRole('dialog', { name: 'Record your voice' })).toBeInTheDocument();
  await userEvent.keyboard('{Escape}');
  await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Record your voice' })).not.toBeInTheDocument());
  expect(screen.getByRole('dialog', { name: 'Settings' })).toBeInTheDocument();
  expect(location.pathname).toBe('/settings');
});

describe('Ask', () => {
  it('opens Ask in a dialog from the top bar on Live', async () => {
    renderApp('/');
    await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
    const dialog = await screen.findByRole('dialog', { name: 'Ask' });
    // The dialog opens at once and its body loads inside it (#2239).
    expect(await within(dialog).findByRole('textbox', { name: /What would you like to ask/ })).toBeInTheDocument();
    // The dialog names it, so the widget's own heading is left out.
    expect(within(within(dialog).getByRole('region', { name: 'Ask' })).queryByRole('heading')).toBeNull();
  });

  it('names its question box on Live after Review was opened, with Review kept hidden (#330)', async () => {
    renderApp('/review');
    await screen.findByRole('slider', { name: /Timeline/ });
    await userEvent.click(screen.getByRole('link', { name: 'Live' }));
    await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
    const dialog = await screen.findByRole('dialog', { name: 'Ask' });
    expect(within(dialog).getByRole('textbox', { name: /What would you like to ask/ })).toBeVisible();
    const ids = [...document.querySelectorAll('[id]')].map(e => e.id);
    expect(ids.filter((id, i) => ids.indexOf(id) !== i)).toEqual([]); // a label or description would find the wrong one
  });

  it('is only on Live', () => {
    renderApp('/review');
    expect(screen.queryByRole('button', { name: 'Ask' })).not.toBeInTheDocument();
  });

  it('opens with / when hotkeys are on, but not while typing', async () => {
    vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Luca', cams: ['cam0', 'cam1'], hotkeys: true });
    renderApp('/');
    const field = document.createElement('input');
    document.body.append(field);
    await userEvent.type(field, '/');
    expect(screen.queryByRole('dialog', { name: 'Ask' })).not.toBeInTheDocument();
    field.remove();
    await userEvent.keyboard('/');
    expect(await screen.findByRole('dialog', { name: 'Ask' })).toBeInTheDocument();
  });

  it('ignores / when hotkeys are off', async () => {
    renderApp('/');
    await userEvent.keyboard('/');
    expect(screen.queryByRole('dialog', { name: 'Ask' })).not.toBeInTheDocument();
  });
});

it('uses the same default-weight Ask trigger in the same top-bar slot on Live and Review', async () => {
  renderApp('/');
  const header = document.querySelector('header') as HTMLElement;
  const live = within(header).getByRole('button', { name: 'Ask' });
  expect(live).toHaveAttribute('data-variant', 'default');
  expect(live).toHaveTextContent(/^Ask/);
  const appearance = {
    classes: live.className,
    style: live.getAttribute('style'),
    icon: live.querySelector('svg')?.outerHTML,
  };
  await userEvent.click(screen.getByRole('link', { name: 'Review' }));
  const review = await within(header).findByRole('button', { name: 'Ask about this day' });
  expect({
    classes: review.className,
    style: review.getAttribute('style'),
    icon: review.querySelector('svg')?.outerHTML,
  }).toEqual(appearance);
  expect(
    within(screen.getByRole('main')).queryByRole('button', { name: 'Ask about this day' }),
  ).not.toBeInTheDocument();
  await userEvent.click(review);
  expect(await screen.findByRole('dialog', { name: /^Ask about / })).toBeInTheDocument();
});

it('keeps the Review shortcut and removes its portaled trigger while another tab is active', async () => {
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Alice', cams: ['cam0'], hotkeys: true });
  renderApp('/review');
  const header = document.querySelector('header') as HTMLElement;
  await screen.findByRole('slider', { name: /Timeline/ });
  await userEvent.keyboard('/');
  expect(await screen.findByRole('dialog', { name: /^Ask about / })).toBeInTheDocument();
  await userEvent.keyboard('{Escape}');
  await userEvent.click(screen.getByRole('link', { name: 'Live' }));
  expect(within(header).getAllByRole('button', { name: /Ask/ })).toHaveLength(1);
  expect(header.querySelector('[data-review-ask-trigger]')).toBeNull();
  await userEvent.click(screen.getByRole('link', { name: 'Review' }));
  expect(await within(header).findByRole('button', { name: 'Ask about this day' })).toBeInTheDocument();
  expect(within(header).getAllByRole('button', { name: /Ask/ })).toHaveLength(1);
});

it('keeps Review’s top-bar Ask host connected when Settings closes over the same visible Review page', async () => {
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Alice', cams: ['cam0'], hotkeys: true });
  renderApp('/review');
  const header = document.querySelector('header') as HTMLElement;
  await within(header).findByRole('button', { name: 'Ask about this day' });
  await openSettings();
  await screen.findByRole('dialog', { name: 'Settings' });
  expect(within(header).queryByRole('button', { name: 'Ask about this day' })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Close settings' }));
  await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Settings' })).not.toBeInTheDocument());
  const trigger = await within(header).findByRole('button', { name: 'Ask about this day' });
  await userEvent.click(trigger);
  expect(await screen.findByRole('dialog', { name: /^Ask about / })).toBeInTheDocument();
  await userEvent.keyboard('{Escape}');
  await waitFor(() => expect(screen.queryByRole('dialog', { name: /^Ask about / })).not.toBeInTheDocument());
  await userEvent.keyboard('/');
  expect(await screen.findByRole('dialog', { name: /^Ask about / })).toBeInTheDocument();
});

it.each([
  { path: '/', trigger: 'Ask' },
  { path: '/review', trigger: 'Ask about this day' },
])('opens a nonblocking Ask panel from $path and toggles it closed from the header', async ({ path, trigger }) => {
  renderApp(path);
  const button = await screen.findByRole('button', { name: trigger });
  await userEvent.click(button);
  const panel = await screen.findByRole('dialog', { name: /Ask/ });
  expect(panel).toHaveAttribute('aria-modal', 'false');
  await userEvent.click(button);
  await waitFor(() => expect(screen.queryByRole('dialog', { name: /Ask/ })).not.toBeInTheDocument());
});

it('resumes an open Review Ask panel after Settings without treating it as a child modal', async () => {
  renderApp('/review');
  await userEvent.click(await screen.findByRole('button', { name: 'Ask about this day' }));
  await screen.findByRole('dialog', { name: /^Ask about / });
  await openSettings();
  await screen.findByRole('dialog', { name: 'Settings' });
  expect(screen.queryByRole('dialog', { name: /^Ask about / })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Close settings' }));
  expect(await screen.findByRole('dialog', { name: /^Ask about / })).toBeInTheDocument();
});

it('preserves Live’s draft and reconnects its pane after Settings and an Activity tab roundtrip', async () => {
  renderApp('/');
  await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
  await userEvent.type(await screen.findByRole('textbox', { name: /What would you like to ask/ }), 'A Live draft');
  await openSettings();
  await screen.findByRole('dialog', { name: 'Settings' });
  await userEvent.click(screen.getByRole('button', { name: 'Close settings' }));
  // The floating window (#1358) stays open under Settings, with its draft.
  expect(await screen.findByRole('textbox', { name: /What would you like to ask/ })).toHaveValue('A Live draft');
  await userEvent.click(screen.getByRole('link', { name: 'Review' }));
  await screen.findByRole('slider', { name: /Timeline/ });
  await userEvent.click(screen.getByRole('link', { name: 'Live' }));
  await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
  expect(await screen.findByRole('textbox', { name: /What would you like to ask/ })).toHaveValue('A Live draft');
});

it.each([
  { path: '/', trigger: 'Ask' },
  { path: '/review', trigger: 'Ask about this day' },
])('focuses an already-open panel with / on $path instead of toggling it closed', async ({ path, trigger }) => {
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Alice', cams: ['cam0'], hotkeys: true });
  renderApp(path);
  await userEvent.click(await screen.findByRole('button', { name: trigger }));
  const panel = await screen.findByRole('dialog', { name: /Ask/ });
  await userEvent.click(screen.getByRole('heading', { level: 1 }));
  await userEvent.keyboard('/');
  expect(panel).toBeVisible();
  expect(within(panel).getByRole('textbox')).toHaveFocus();
});
