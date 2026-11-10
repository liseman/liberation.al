import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { usePrefs } from '../lib/prefs';
import type { Stats } from '../sources/api';
import { stats } from '../sources/fixtures';
import { DebugTab } from './DebugTab';
import { RETRY_MS, STREAM_URL } from './stream';

/** Just enough EventSource for the tab: it opens one, and the test sends it events. */
class FakeEventSource {
  static all: FakeEventSource[] = [];
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  listeners: Record<string, ((e: MessageEvent<string>) => void)[]> = {};
  closed = false;
  readyState = 0; // CONNECTING; 2 (CLOSED) once the browser has given up on it
  readonly url: string;
  constructor(url: string) {
    this.url = url;
    FakeEventSource.all.push(this);
  }
  addEventListener(type: string, fn: (e: MessageEvent<string>) => void) {
    this.listeners[type] = [...(this.listeners[type] ?? []), fn];
  }
  close() {
    this.closed = true;
  }
  /** Deliver an event and let the query cache tell the tab (it notifies on the next tick). */
  async send(type: string, data: unknown) {
    await act(async () => {
      for (const fn of this.listeners[type] ?? []) fn(new MessageEvent(type, { data: JSON.stringify(data) }));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
  }
}

const stream = () => {
  const es = FakeEventSource.all.at(-1);
  if (!es) throw new Error('no stream opened');
  return es;
};
const tailLines = () => [...screen.getByRole('log', { name: 'Service logs' }).querySelectorAll('[data-line]')];
/** The tail's lines as "service: text", without the time and the level badge in front. */
const lineTexts = () =>
  tailLines().map(el => el.textContent?.replace(/^(earlier|\d\d:\d\d:\d\d)(DEBUG|INFO|WARN|ERROR)/, '') ?? '');
/** Each line's level badge, as shown. */
const badges = () => tailLines().map(el => el.querySelector('[data-level]')?.textContent);
const levelRadio = (name: string) => screen.getByRole('radio', { name: new RegExp(`^${name} `) });

const serveStats = (s: Stats) => server.use(http.get('/stats', () => HttpResponse.json(s)));

beforeEach(() => {
  FakeEventSource.all = [];
  vi.stubGlobal('EventSource', FakeEventSource);
  serveStats(stats());
  usePrefs.setState({}, true); // localStorage is cleared after each test; the store must forget too
});
afterEach(() => vi.unstubAllGlobals());

async function start() {
  const page = renderPage(<DebugTab />);
  await stream().send('hello', { ts: 1, services: ['ambient', 'vision', 'whisper'], missing: ['whisper'] });
  await stream().send('lines', [
    { s: 'ambient', l: 'Heard Dana', t: null },
    { s: 'vision', l: 'cam0 dark', t: 1_800_000_000 },
    { s: 'ambient', l: 'idle', t: 1_800_000_001 },
  ]);
  return page;
}

describe('the Debug tab', () => {
  it('streams /logs/stream and closes it when the tab goes away', async () => {
    const { unmount } = await start();
    expect(stream().url).toBe(STREAM_URL);
    unmount();
    expect(stream().closed).toBe(true);
  });

  it('opens a new stream when the browser gives up on one (an HTTP error closes it for good)', async () => {
    const { unmount } = await start();
    const first = stream();
    await act(async () => {
      first.readyState = 2;
      first.onerror?.();
    });
    expect(screen.getByText('Reconnecting…')).toBeInTheDocument();
    await waitFor(() => expect(FakeEventSource.all).toHaveLength(2), { timeout: RETRY_MS + 1000 });
    expect(first.closed).toBe(true);
    await stream().send('hello', { ts: 2, services: ['ambient'], missing: [] });
    expect(screen.getByText('Live')).toBeInTheDocument();
    unmount();
    expect(stream().closed).toBe(true);
  });

  it('leaves a stream the browser is reconnecting by itself alone', async () => {
    await start();
    await act(async () => stream().onerror?.());
    expect(screen.getByText('Reconnecting…')).toBeInTheDocument();
    await act(() => new Promise(resolve => setTimeout(resolve, RETRY_MS + 200)));
    expect(FakeEventSource.all).toHaveLength(1);
    expect(stream().closed).toBe(false);
  });

  it('holds the chips and the tail behind one loading line until the stream says which services it has', async () => {
    renderPage(<DebugTab />);
    expect(screen.getByLabelText('Loading the logs')).toBeInTheDocument();
    expect(screen.queryByRole('log', { name: 'Service logs' })).toBeNull();
    expect(screen.queryByRole('checkbox', { name: 'ambient' })).toBeNull();
    await stream().send('hello', { ts: 1, services: ['ambient'], missing: [] });
    expect(screen.queryByLabelText('Loading the logs')).toBeNull();
    expect(screen.getByRole('log', { name: 'Service logs' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'ambient' })).toBeInTheDocument();
  });

  it('prefixes each line with its service, and says which services have no log yet', async () => {
    await start();
    expect(lineTexts()).toEqual(['ambient: Heard Dana', 'vision: cam0 dark', 'ambient: idle']);
    expect(screen.getByText(/No log yet for whisper/)).toBeInTheDocument();
  });

  it('hides a service when its chip is turned off', async () => {
    await start();
    await userEvent.click(screen.getByRole('checkbox', { name: 'vision' }));
    expect(lineTexts()).toEqual(['ambient: Heard Dana', 'ambient: idle']);
  });

  it('sorts the service chips into sections, and a section title hides or shows all of its services', async () => {
    renderPage(<DebugTab />);
    await stream().send('hello', { ts: 1, services: ['listen', 'see-actions', 'see-decode'], missing: [] });
    await stream().send('lines', [
      { s: 'see-decode', l: 'cam0 decoding', t: 1_800_000_000 },
      { s: 'listen', l: 'Heard Dana', t: 1_800_000_001 },
      { s: 'see-actions', l: 'cam0 28 fps', t: 1_800_000_002 },
    ]);
    const vision = screen.getByLabelText('Vision');
    expect(
      within(vision)
        .getAllByText(/^see-/)
        .map(e => e.textContent),
    ).toEqual(['see-decode', 'see-actions']);
    await userEvent.click(screen.getByRole('button', { name: 'Hide all of Vision' }));
    expect(lineTexts()).toEqual(['listen: Heard Dana']);
    await userEvent.click(screen.getByRole('button', { name: 'Show all of Vision' }));
    expect(lineTexts()).toHaveLength(3);
  });

  it('filters by the search, or only highlights it', async () => {
    await start();
    await userEvent.type(screen.getByRole('textbox', { name: 'Search the logs' }), 'dana');
    expect(lineTexts()).toEqual(['ambient: Heard Dana']);
    await userEvent.click(screen.getByText('Highlight'));
    expect(lineTexts()).toHaveLength(3);
    expect(screen.getByText('Dana', { selector: 'mark' })).toBeInTheDocument();
  });

  it('holds new lines while paused, counts them, and shows them on resume', async () => {
    await start();
    await userEvent.click(screen.getByRole('button', { name: 'Pause' }));
    await stream().send('lines', [{ s: 'vision', l: 'cam1 moved', t: 1_800_000_002 }]);
    expect(lineTexts()).toHaveLength(3);
    await userEvent.click(screen.getByRole('button', { name: 'Resume (1 held)' }));
    expect(lineTexts().at(-1)).toBe('vision: cam1 moved');
  });

  it('opens the lines around one when its time is clicked', async () => {
    await start();
    await userEvent.click(screen.getAllByRole('button', { name: /^Lines around/ })[1] as HTMLElement);
    const dialog = await screen.findByRole('dialog', { name: 'Lines around it' });
    expect(within(dialog).getByText('cam0 dark')).toBeInTheDocument();
    expect(within(dialog).getByText('Heard Dana')).toBeInTheDocument();
  });
});

describe('the Ollama alert', () => {
  it('says Ollama is down and how to start it', async () => {
    serveStats(
      stats({
        ollama: { models: [], error: 'refused', up: false, missing: [], message: 'Ollama is not running: start it' },
      }),
    );
    renderPage(<DebugTab />);
    const alert = await screen.findByRole('alert');
    expect(within(alert).getByText('Ollama is down')).toBeInTheDocument();
    expect(within(alert).getByText('Ollama is not running: start it')).toBeInTheDocument();
  });

  it('says which model is missing', async () => {
    serveStats(
      stats({ ollama: { models: [], error: null, up: true, missing: ['emb:4b'], message: 'Model missing: emb:4b' } }),
    );
    renderPage(<DebugTab />);
    const alert = await screen.findByRole('alert');
    expect(within(alert).getByText('Ollama is missing a model')).toBeInTheDocument();
    expect(within(alert).getByText('Model missing: emb:4b')).toBeInTheDocument();
  });

  it('shows nothing while Ollama is fine', async () => {
    renderPage(<DebugTab />);
    await screen.findByRole('heading', { name: 'Debug' });
    await new Promise(resolve => setTimeout(resolve, 20));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

describe('log levels', () => {
  async function startLevels() {
    const page = renderPage(<DebugTab />);
    await stream().send('hello', { ts: 1, services: ['ambient', 'vision'], missing: [] });
    await stream().send('lines', [
      { s: 'ambient', l: '10:00:00 [ambient] DEBUG frame 3', t: null },
      { s: 'ambient', l: '10:00:01 [ambient] INFO Heard Dana', t: null },
      { s: 'vision', l: '10:00:02 [vision] WARN cam1 lost', t: null },
      { s: 'vision', l: '10:00:03 [vision] ERROR cam1 gone', t: null },
      { s: 'vision', l: 'ValueError: boom', t: null },
      { s: 'ambient', l: '10:00:04 [ambient] WARN Dana too quiet', t: null },
    ]);
    return page;
  }

  it('badges each line with its level, an inherited one too, and hides debug lines by default', async () => {
    await startLevels();
    expect(badges()).toEqual(['INFO', 'WARN', 'ERROR', 'ERROR', 'WARN']);
    expect(levelRadio('Info')).toBeChecked();
    const inherited = tailLines()[3]?.querySelector('[data-level]');
    expect(inherited).toHaveAttribute('data-inherited');
    expect(tailLines()[2]?.querySelector('[data-level]')).not.toHaveAttribute('data-inherited');
  });

  it('counts each level on the control', async () => {
    await startLevels();
    expect(levelRadio('Debug')).toHaveAccessibleName('Debug 1');
    expect(levelRadio('Info')).toHaveAccessibleName('Info 1');
    expect(levelRadio('Warn')).toHaveAccessibleName('Warn 2');
    expect(levelRadio('Error')).toHaveAccessibleName('Error 2');
  });

  it('shows only the chosen level and those above it', async () => {
    await startLevels();
    await userEvent.click(levelRadio('Warn'));
    expect(badges()).toEqual(['WARN', 'ERROR', 'ERROR', 'WARN']);
    await userEvent.click(levelRadio('Error'));
    expect(lineTexts()).toEqual(['vision: 10:00:03 [vision] ERROR cam1 gone', 'vision: ValueError: boom']);
    await userEvent.click(levelRadio('Debug'));
    expect(lineTexts()).toHaveLength(6);
  });

  it('combines with a service toggle and a search, and the counts follow them', async () => {
    await startLevels();
    await userEvent.click(levelRadio('Warn'));
    await userEvent.click(screen.getByRole('checkbox', { name: 'vision' }));
    expect(lineTexts()).toEqual(['ambient: 10:00:04 [ambient] WARN Dana too quiet']);
    expect(levelRadio('Error')).toHaveAccessibleName('Error 0');
    await userEvent.click(screen.getByRole('checkbox', { name: 'vision' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Search the logs' }), 'cam1');
    expect(lineTexts()).toEqual([
      'vision: 10:00:02 [vision] WARN cam1 lost',
      'vision: 10:00:03 [vision] ERROR cam1 gone',
    ]);
    expect(levelRadio('Warn')).toHaveAccessibleName('Warn 1');
  });

  it('remembers the chosen level in this browser', async () => {
    const { unmount } = await startLevels();
    await userEvent.click(levelRadio('Error'));
    unmount();
    await startLevels();
    expect(levelRadio('Error')).toBeChecked();
    expect(lineTexts()).toHaveLength(2);
  });
});
