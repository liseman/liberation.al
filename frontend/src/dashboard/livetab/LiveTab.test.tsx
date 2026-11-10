import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { newQueryClient, Providers } from '../../shared/Providers';
import { server } from '../../test/server';
import { setGatewayForTests } from '../camera/gateway';
import { fakeGateway } from '../camera/testFakes';
import { hm } from '../lib/format';
import { setPref } from '../lib/prefs';
import { liveKey } from '../live/feed';
import { reduce } from '../live/reduce';
import { useLiveStatus } from '../live/status';
import { setActiveCameras, setCameraNames } from '../shell';
import { LiveTab } from './LiveTab';

vi.mock('../camera/PreviewCam', () => ({
  PreviewCam: ({ cam }: { cam: string }) => <canvas className="preview-canvas" data-cam={cam} />,
}));

const NOW = 1_793_000_000;
const clock = (ts: number) =>
  new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });

/** An EventSource that never sends: the camera boxes subscribe to one. */
class QuietEventSource {
  static readonly CLOSED = 2;
  readyState = 0;
  onmessage = null;
  onerror = null;
  addEventListener() {}
  close() {}
}

const SERVICES = {
  ok: true,
  control: true,
  platform: 'linux',
  services: ['listen', 'continuum-broker', 'continuum-capture', 'see-decode'].map(name => ({
    name,
    description: '',
    state: name === 'continuum-broker' ? 'active' : 'inactive', // the broker runs already: not started again
    installed: name === 'continuum-broker',
    pids: [],
    selected: true,
  })),
};

function renderLive(
  live: { ambient?: unknown; scene?: unknown; lines?: unknown[]; vision?: unknown; unread?: boolean } = {},
) {
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false } });
  if (!live.unread) {
    // unread: the first status reads haven't answered yet
    client.setQueryData(liveKey('ambient'), live.ambient ?? {});
    client.setQueryData(liveKey('scene'), live.scene ?? []);
    client.setQueryData(liveKey('vision'), live.vision ?? {});
  }
  client.setQueryData(
    liveKey('transcript'),
    reduce('transcript', undefined, { entries: live.lines ?? [], reset: true }),
  );
  client.setQueryData(liveKey('speakers'), reduce('speakers', undefined, { speakers: [], merged: {} }));
  return render(
    <Providers client={client} test>
      <LiveTab />
    </Providers>,
  );
}

beforeEach(() => {
  setPref('liveTiles', false); // the sparse-preview overview these tests cover; live tiles have their own test
  history.replaceState(null, '', '/');
  vi.stubGlobal('EventSource', QuietEventSource);
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Luca', cams: ['cam0', 'cam1'], hotkeys: false });
  useLiveStatus.setState({ now: NOW });
  setGatewayForTests(fakeGateway().connect); // a gateway that never answers, unless a test answers for it
  server.use(
    // Getting started's reads: nothing set up, so it shows; tests about it set their own.
    http.get('/stats', () => HttpResponse.json({})),
    http.get('/devices', () => HttpResponse.json({ control: true, mics: [], cameras: [], notes: {} })),
    http.get('/config/voice_ref', () => HttpResponse.json({ exists: false })),
    http.get('/config', () =>
      HttpResponse.json({ ok: true, control: true, needs_setup: false, path: 'pack.toml', sections: [] }),
    ),
    http.get('/people', () =>
      HttpResponse.json({
        people: [
          { name: 'Dana', here_now: { cam: 'cam0', camera_name: 'front camera', since: NOW - 600 } },
          { name: 'Luca' },
          { name: 'Eve' },
        ],
      }),
    ),
    http.get('/live/happening', () =>
      HttpResponse.json({
        items: [
          {
            id: 'p:1',
            ts: NOW - 900,
            kind: 'arrival',
            text: 'Dana walked in',
            people: ['Dana'],
            cam: 'cam0',
            camera_name: null,
          },
          {
            id: 't:2',
            ts: NOW - 60,
            kind: 'talk',
            text: 'Dana asked Luca about dinner',
            people: ['Dana', 'Luca'],
            cam: null,
            camera_name: null,
          },
        ],
        updated: NOW - 60,
      }),
    ),
    http.get('/voices/unknown/groups', () => HttpResponse.json({ groups: [], count: 0, skipped: [] })),
    http.get('/faces/unknown/groups', () =>
      HttpResponse.json({
        groups: [
          {
            id: 4,
            label: 'Face 4',
            count: 9,
            last_seen: NOW - 3,
            shots: [],
            guess: null,
            score: 0,
            band: 'none',
            reason: 'A clear face, unlike anyone named',
          },
        ],
        count: 1,
      }),
    ),
    http.get('/services', () => HttpResponse.json(SERVICES)),
  );
  return () => vi.unstubAllGlobals();
});

const both = [
  { cam: 'cam0', ts: NOW - 2, caption: 'A kitchen', faces: [{ name: 'Dana', bbox: [0, 0, 40, 40] }] },
  { cam: 'cam1', ts: NOW - 3, caption: 'A hallway', faces: [{ name: 'Face 4', bbox: [0, 0, 40, 40] }] },
];

it.each(['__proto__', 'constructor', 'toString'])('shows the newest own caption for %s', cam => {
  vi.stubGlobal('NAMES', { cams: [cam] });
  renderLive({
    scene: [
      { cam, ts: NOW - 10, caption: 'An earlier caption' },
      { cam, ts: NOW - 1, caption: 'The current caption' },
    ],
  });
  const camera = screen.getByRole('figure', { name: `${cam}, large` });
  expect(within(camera).getByText('The current caption')).toBeInTheDocument();
  expect(within(camera).queryByText('An earlier caption')).toBeNull();
});

describe('Live', () => {
  it("marks a caption the view hasn't changed from, saying since when", () => {
    history.replaceState(null, '', '/#camera=cam0');
    const since = NOW - 600;
    renderLive({
      ambient: { ts: NOW - 1 },
      scene: [{ ...both[0], unchanged: true, since }, both[1]],
    });
    const large = screen.getByRole('figure', { name: 'cam0, large' });
    expect(within(large).getByText('A kitchen')).toBeInTheDocument();
    expect(within(large).getByRole('img', { name: `Unchanged since ${hm(since)}` })).toBeInTheDocument();
  });

  it('shows a fresh caption with no mark', () => {
    history.replaceState(null, '', '/#camera=cam0');
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    const large = screen.getByRole('figure', { name: 'cam0, large' });
    expect(within(large).queryByRole('img', { name: /Unchanged/ })).toBeNull();
  });

  it('shows six labeled previews, focuses an exact network-camera id, and returns to the overview', async () => {
    const cameras = ['cam0', 'east-door', 'portrait-side', 'upper-hall', 'rear-yard', '__proto__'];
    vi.stubGlobal('NAMES', {
      cams: cameras,
      views: JSON.parse('{"portrait-side":"Side view","__proto__":"Rear view"}'),
    });
    const net = fakeGateway();
    setGatewayForTests(net.connect);
    renderLive({ ambient: { ts: NOW }, scene: cameras.map(cam => ({ cam, ts: NOW })) });
    expect(screen.getByText('6 cameras')).toBeInTheDocument();
    expect(document.querySelectorAll('canvas.preview-canvas')).toHaveLength(6);
    expect(document.querySelectorAll('canvas.live-canvas')).toHaveLength(0);
    expect(net.conns.every(c => c.sent.every(r => r.type !== 'live'))).toBe(true);
    const sideTile = screen.getByRole('button', { name: 'Open Side view camera' }).closest('.cam-tile') as HTMLElement;
    const preview = sideTile.querySelector('canvas') as HTMLCanvasElement;
    preview.width = 720;
    preview.height = 1280;
    await waitFor(() => expect(sideTile).toHaveAttribute('data-orientation', 'portrait'));
    expect(sideTile.style.getPropertyValue('--camera-aspect')).toBe('0.5625');
    expect(within(sideTile).getByText('Side view').parentElement).toHaveClass('cam-tile-label');
    preview.width = 1920;
    preview.height = 1080;
    await waitFor(() => expect(sideTile).toHaveAttribute('data-orientation', 'landscape'));
    await userEvent.click(screen.getByRole('button', { name: 'Open Side view camera' }));
    expect(screen.getByRole('figure', { name: 'Side view, large' })).toBeInTheDocument();
    expect(location.hash).toBe('#camera=portrait-side');
    const portrait = document.querySelector('canvas.live-canvas') as HTMLCanvasElement;
    portrait.width = 720;
    portrait.height = 1280;
    await waitFor(() =>
      expect(screen.getByRole('figure', { name: 'Side view, large' }).style.getPropertyValue('--camera-aspect')).toBe(
        '0.5625',
      ),
    );
    expect(document.querySelectorAll('canvas.preview-canvas')).toHaveLength(0);
    expect(document.querySelectorAll('canvas.live-canvas')).toHaveLength(1);
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Choose camera' }), '__proto__');
    expect(screen.getByRole('figure', { name: 'Rear view, large' })).toBeInTheDocument();
    expect(location.hash).toBe('#camera=__proto__');
    await userEvent.click(screen.getByRole('button', { name: 'All cameras' }));
    expect(document.querySelectorAll('canvas.live-canvas')).toHaveLength(0);
    expect(document.querySelectorAll('canvas.preview-canvas')).toHaveLength(6);
  });

  it('plays every overview camera live when Live tiles is on, and goes back to previews when it is off', async () => {
    setPref('liveTiles', true);
    const net = fakeGateway();
    setGatewayForTests(net.connect);
    renderLive({
      ambient: { ts: NOW },
      scene: [
        { cam: 'cam0', ts: NOW },
        { cam: 'cam1', ts: NOW },
      ],
    });
    expect(document.querySelectorAll('.cam-tile canvas.live-canvas')).toHaveLength(2);
    expect(document.querySelectorAll('canvas.preview-canvas')).toHaveLength(0);
    await userEvent.click(screen.getByRole('switch', { name: 'Play every camera live in the overview' }));
    expect(document.querySelectorAll('canvas.preview-canvas')).toHaveLength(2);
    expect(document.querySelectorAll('canvas.live-canvas')).toHaveLength(0);
  });

  it('shows setup, with Start buttons, when nothing is running and nothing was said', async () => {
    renderLive();
    expect(screen.getByRole('heading', { name: "Liberational isn't listening yet" })).toBeInTheDocument();
    expect(await screen.findByRole('button', { name: 'Start listening' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start cameras' })).toBeInTheDocument();
  });

  it("starts the cameras' services that aren't running, each once", async () => {
    const started: unknown[] = [];
    server.use(
      http.post('/services', async ({ request }) => {
        started.push(await request.json());
        return HttpResponse.json({ ok: true });
      }),
    );
    renderLive();
    await userEvent.click(await screen.findByRole('button', { name: 'Start cameras' }));
    await vi.waitFor(() =>
      expect(started).toEqual([
        { name: 'continuum-capture', action: 'on' },
        { name: 'see-decode', action: 'on' },
      ]),
    );
  });

  it.each(['cam0', 'front-door', 'FaceCam_A', '7North', 'cam00', '0', '__proto__', 'constructor', 'toString'])(
    "shows %s with live video at Continuum's gateway, with nothing from v1",
    async cam => {
      vi.stubGlobal('NAMES', { cams: [cam] });
      const net = fakeGateway();
      setGatewayForTests(net.connect);
      renderLive();
      expect(screen.getByRole('heading', { name: "Liberational isn't listening yet" })).toBeInTheDocument();
      await vi.waitFor(() => expect(net.open().length).toBeGreaterThan(0));
      const c = net.open()[0];
      c?.hello();
      c?.handlers.onReply({
        type: 'streams',
        streams: [
          {
            id: 1,
            name: `${cam}.video`,
            live: true,
            format: 'passthrough',
            start_us: 0,
            end_us: Date.now() * 1000,
          },
        ],
      });
      expect(await screen.findByRole('figure', { name: `${cam}, large` })).toBeInTheDocument();
    },
  );

  it('says the microphone is off while the cameras run', async () => {
    renderLive({ scene: both });
    expect(screen.getByRole('heading', { name: 'Live' })).toBeInTheDocument();
    // A state of the conversation header's mic line, not a notice above the page.
    const mics = screen.getByRole('region', { name: 'Microphones' });
    expect(within(mics).getByRole('status')).toHaveTextContent('Not listening');
    expect(await within(mics).findByRole('button', { name: 'Start listening' })).toBeInTheDocument();
  });

  it("calls nothing off before the first status reads answer, and holds the cameras' place", () => {
    renderLive({ unread: true });
    expect(screen.queryByText("Liberational isn't listening yet")).toBeNull();
    expect(screen.queryByText('Not listening')).toBeNull();
    expect(screen.queryByText(/Cameras are off/)).toBeNull();
    expect(screen.getByRole('region', { name: 'Cameras' })).toHaveAttribute('aria-busy', 'true');
  });

  it('starts with previews and opens one camera without a popup', async () => {
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    expect(screen.queryByRole('figure')).not.toBeInTheDocument();
    expect(document.querySelectorAll('canvas.live-canvas')).toHaveLength(0);
    await userEvent.click(screen.getByRole('button', { name: 'Open cam1 camera' }));
    expect(screen.getByRole('figure', { name: 'cam1, large' })).toBeInTheDocument();
    expect(location.hash).toBe('#camera=cam1');
    await userEvent.click(screen.getByRole('button', { name: 'All cameras' }));
    expect(screen.getByRole('button', { name: 'Open cam0 camera' })).toBeInTheDocument();
    expect(location.hash).toBe('');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument(); // no full-size popup any more
  });

  it('heads the rig Cameras even with a stale room value, and names each camera', () => {
    vi.stubGlobal('NAMES', {
      assistant: 'Bob',
      owner: 'Luca',
      cams: ['cam0', 'cam1'],
      views: { cam0: 'front camera', cam1: 'rear camera' },
      room: 'living room',
      hotkeys: false,
    });
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    expect(screen.getByRole('heading', { name: 'Cameras' })).toBeInTheDocument();
    expect(screen.getByText('2 cameras')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open Front camera camera' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open Rear camera camera' })).toBeInTheDocument();
  });

  it('offers to name only the cameras with no name, linking to Settings', () => {
    vi.stubGlobal('NAMES', { cams: ['cam0', 'cam1'], views: { cam0: 'front camera' } });
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    expect(screen.getByRole('link', { name: 'Name cam1' })).toHaveAttribute('href', '/settings#device-cameras');
  });

  it('drops the link as soon as Settings saves the name, without a reload', () => {
    vi.stubGlobal('NAMES', { cams: ['cam0', 'cam1'], views: { cam0: 'front camera' } });
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    expect(screen.getByRole('link', { name: 'Name cam1' })).toBeInTheDocument();
    act(() => setCameraNames({ cam0: 'front camera', cam1: 'desk' }));
    expect(screen.queryByRole('link', { name: /^Name / })).not.toBeInTheDocument();
  });

  it('counts the cameras to name when more than one has none, and says nothing once all are named', () => {
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    expect(screen.getByRole('link', { name: 'Name 2 cameras' })).toBeInTheDocument();
    cleanup();
    vi.stubGlobal('NAMES', { cams: ['cam0', 'cam1'], views: { cam0: 'front camera', cam1: 'rear camera' } });
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    expect(screen.queryByRole('link', { name: /^Name / })).not.toBeInTheDocument();
  });

  it('shows a camera vision says is on, even before its first caption', async () => {
    renderLive({
      ambient: { ts: NOW - 1 },
      scene: both.filter(e => e.cam === 'cam0'),
      vision: { ts: NOW - 1, cameras: { cam0: { state: 'on' }, cam1: { state: 'on' } } },
    });
    await userEvent.click(screen.getByRole('button', { name: 'Open cam1 camera' }));
    const large = screen.getByRole('figure', { name: 'cam1, large' });
    expect(within(large).queryByText('Camera off')).not.toBeInTheDocument();
    expect(large.querySelector('canvas.live-canvas')).not.toBeNull();
  });

  it("puts vision's retrying ahead of a recent caption", () => {
    renderLive({
      ambient: { ts: NOW - 1 },
      scene: both,
      vision: { ts: NOW - 1, cameras: { cam0: { state: 'retrying' }, cam1: { state: 'on' } } },
    });
    const tile = screen.getByRole('button', { name: 'Open cam0 camera' });
    expect(within(tile).getByText('Camera not responding, retrying')).toBeInTheDocument();
  });

  it('keeps a camera that stopped giving frames in its tile, saying it is retrying', () => {
    renderLive({
      ambient: { ts: NOW - 1 },
      scene: both.filter(e => e.cam === 'cam0'),
      vision: { ts: NOW - 1, cameras: { cam0: { state: 'on' }, cam1: { state: 'retrying' } } },
    });
    const tile = screen.getByRole('button', { name: 'Open cam1 camera' });
    expect(within(tile).getByText('Camera not responding, retrying')).toBeInTheDocument();
  });

  it('says a camera and a microphone are paused for privacy instead of showing their last picture', async () => {
    server.use(
      http.get('/devices/privacy', () =>
        HttpResponse.json({
          control: true,
          ready: true,
          devices: [
            { name: 'cam0', kind: 'camera', paused: true, state: 'paused' },
            { name: 'cam1', kind: 'camera', paused: false, state: 'on' },
            { name: 'wave3', kind: 'microphone', paused: true, state: 'pausing' },
          ],
        }),
      ),
    );
    renderLive({
      ambient: { ts: NOW - 1 },
      scene: both,
      vision: { ts: NOW - 1, cameras: { cam0: { state: 'on' }, cam1: { state: 'on' } } },
    });
    const tile = screen.getByRole('button', { name: 'Open cam0 camera' });
    expect(await within(tile).findByText('Paused for privacy')).toBeInTheDocument();
    expect(tile.querySelector('img, canvas')).toBeNull();
    expect(
      within(screen.getByRole('button', { name: 'Open cam1 camera' })).queryByText('Paused for privacy'),
    ).toBeNull();
    expect(screen.getByText('wave3: Paused for privacy')).toBeInTheDocument();
  });

  it("shows what's happening, newest at the bottom, and who is here, each linking to their card", async () => {
    renderLive({
      ambient: { ts: NOW - 1 },
      scene: both,
      lines: [{ ts_start: NOW - 20, ts_end: NOW - 18, speaker: 'Luca', text: 'Hi' }],
    });
    const log = screen.getByRole('region', { name: "What's happening" });
    expect((await within(log).findAllByRole('listitem')).map(li => li.textContent)).toEqual([
      `${clock(NOW - 900)}Dana walked in`,
      `${clock(NOW - 60)}Dana asked Luca about dinner`,
    ]);
    const here = screen.getByRole('region', { name: 'Here now' });
    const dana = await within(here).findByRole('link', { name: /Dana/ });
    expect(dana).toHaveAttribute('href', '/people#card=Dana');
    expect(dana).toHaveTextContent(`Front camera, since ${clock(NOW - 600)}`);
    expect(within(here).getByRole('link', { name: /Luca/ })).toHaveTextContent('Heard just now');
    expect(within(here).queryByRole('link', { name: /Eve/ })).toBeNull();
  });

  it("leaves Who's this? to People and the day's summary to Review", async () => {
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    expect(screen.queryByRole('heading', { name: "Who's this?" })).toBeNull(); // removed from Live
    expect(screen.queryByText(/Planned the week|summary|minutes of speech/i)).toBeNull();
  });

  it('leaves the page name to the tab bar, Review today to Review, and Appearance to the top bar', () => {
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    expect(screen.getByRole('heading', { level: 1, name: 'Live' })).toBeInTheDocument(); // visually hidden
    expect(screen.queryByText(/as they happen/)).toBeNull();
    expect(screen.queryByRole('link', { name: 'Review today' })).toBeNull();
    expect(screen.queryByRole('button', { name: /Appearance/ })).toBeNull();
  });

  it('puts Here now under the cameras, and leaves the rail to the conversation', async () => {
    renderLive({ ambient: { ts: NOW - 1 }, scene: both });
    // One region a panel, named as its heading and its grip are: no "Who's here" or "Right now" around them
    const here = screen.getByRole('region', { name: 'Here now' });
    expect(screen.getAllByRole('region', { name: 'Here now' })).toHaveLength(1);
    expect(within(here).queryByRole('region', { name: 'Linked device evidence' })).toBeNull(); // cut
    const rail = screen.getByRole('region', { name: "What's happening" });
    expect(within(rail).queryByRole('region', { name: 'Here now' })).toBeNull();
    expect(screen.queryByRole('complementary', { name: 'Right now' })).toBeNull();
    // Narrow screens stack them in document order: cameras, the conversation, then who's here.
    const cameras = screen.getByRole('region', { name: 'Cameras' });
    const conversation = screen.getByRole('region', { name: 'Live conversation' });
    expect(cameras.compareDocumentPosition(conversation) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(conversation.compareDocumentPosition(here) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('puts the microphones under the Conversation heading, saying what they hear', () => {
    renderLive({ ambient: { ts: NOW - 1, devices: { wave: { name: 'Room mic', healthy: true } } }, scene: both });
    const head = screen.getByRole('heading', { name: 'Conversation' }).parentElement as HTMLElement;
    const mics = within(head).getByRole('region', { name: 'Microphones' });
    expect(mics).toHaveTextContent('Room mic');
    expect(mics).toHaveTextContent(/^Listening · (quiet|sound|level unavailable)/);
    expect(screen.getByRole('heading', { name: 'Cameras' }).parentElement).not.toContainElement(mics);
  });

  it('says the cameras are off when none is live', () => {
    renderLive({ ambient: { ts: NOW - 1 } });
    expect(screen.getByText("Cameras are off, so Liberational can't see who's here.")).toBeInTheDocument();
  });

  it.each([
    ['retrying', 'Camera not responding, retrying'],
    ['starting', 'Starting…'],
  ])('says a camera is %s instead of off', (state, text) => {
    renderLive({ vision: { ts: NOW - 1, cameras: { cam0: { state } } } });
    expect(screen.getByText(text)).toBeInTheDocument();
    expect(screen.queryByText("Cameras are off, so Liberational can't see who's here.")).not.toBeInTheDocument();
  });

  it("says the cameras are off when vision's status is old (vision isn't running)", () => {
    renderLive({ ambient: { ts: NOW - 1 }, vision: { ts: NOW - 600, cameras: { cam0: { state: 'retrying' } } } });
    expect(screen.getByText("Cameras are off, so Liberational can't see who's here.")).toBeInTheDocument();
  });
});

it('removes disabled cameras from Live immediately despite retained fresh scene observations', async () => {
  renderLive({
    scene: [
      { cam: 'cam0', ts: NOW - 1 },
      { cam: 'cam1', ts: NOW - 1 },
    ],
  });
  await waitFor(() => expect(document.querySelectorAll('.preview-canvas')).toHaveLength(2));
  act(() => setActiveCameras([]));
  await waitFor(() => expect(document.querySelectorAll('.preview-canvas, .live-canvas')).toHaveLength(0));
});
