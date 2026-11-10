import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { liveKey } from '../../live/feed';
import { useLiveStatus } from '../../live/status';
import { shell } from '../../shell';
import type { Devices as DeviceView } from '../../sources/api';
import { METER_MS } from '../../sources/levels';
import { DraftProvider, useDraft } from '../draft';
import { CONFIG_KEY, type ConfigView, type Setting } from '../model';
import { Devices } from './Devices';

const NOW = 1_800_000_000;
const row = (key: string, value: Setting['value']): Setting => ({
  key,
  value,
  default: value,
  type: Array.isArray(value) ? 'list' : 'str',
  secret: false,
  description: key,
  section: 'devices',
});
const config = (control = true, { cams = [0, 1] }: { cams?: number[] } = {}): ConfigView => ({
  ok: true,
  control,
  needs_setup: false,
  path: 'pack.toml',
  sections: [
    {
      name: 'vision',
      description: '',
      settings: [
        row('vision.room', 'living room'),
        row('vision.camera_names', ['cam0: front camera']),
        row('vision.cams', cams),
      ],
    },
    {
      name: 'audio',
      description: '',
      settings: [row('audio.sensitivity', 70), row('audio.auto_gain_target_dbfs', -20)],
    },
  ],
});
const WAVE = 'alsa_input.usb-Elgato_Systems_Elgato_Wave_3_A017A41910AAFN-00.mono-fallback';
const devices: DeviceView = {
  ok: true,
  control: true,
  notes: {},
  default_sensitivity: 70,
  mics: [
    {
      id: WAVE,
      name: WAVE,
      present: true,
      selected: true,
      stream: 'wave',
      in_use: true,
      level_dbfs: -60,
      sensitivity: 70,
      sensitivity_custom: false,
    },
    {
      id: 'Desk Mic',
      name: 'Desk Mic',
      present: true,
      selected: true,
      stream: 'desk',
      in_use: true,
      level_dbfs: null,
      sensitivity: 85,
      sensitivity_custom: true,
    },
  ],
  cameras: [
    { index: 0, name: 'Facecam', path: '/dev/video0', present: true, selected: true, stream: 'cam0', in_use: true },
    { index: 1, name: 'Room cam', path: '/dev/video2', present: true, selected: true, stream: 'cam1', in_use: false },
  ],
};

/** The draft's values, shown so a test can read what a control changed. */
function Values() {
  const d = useDraft();
  return (
    <>
      <button type="button" onClick={d.save}>
        Save the draft
      </button>
      <pre data-testid="draft">
        {JSON.stringify({
          room: d.value('vision.room'),
          places: d.value('vision.camera_names'),
          changedKeys: d.changedKeys,
        })}
      </pre>
    </>
  );
}
const INGEST = {
  items: [],
  jobs: [],
  control: true,
  sources: [{ name: 'wyze', label: 'Wyze cameras', ongoing: true, configured: false, message: 'Not set up', seen: 0 }],
  wyze: { cameras: [], rtsp_url: '', segment_s: 300, email: false, password: false, api_id: false, api_key: false },
};
const draft = () => JSON.parse(screen.getByTestId('draft').textContent ?? '{}');

it('renames a network camera through the display-name draft without replacing its ID', async () => {
  server.use(
    http.get('/devices', () => HttpResponse.json(devices)),
    http.get('/network-cameras', () =>
      HttpResponse.json({
        capture_ready: true,
        controls_ready: true,
        cameras: [{ id: 'entryway', name: 'entryway', source: 'rtsp', url_display: 'redacted', enabled: true }],
      }),
    ),
  );
  const page = renderDevices();
  act(() => page.client.setQueryData(CONFIG_KEY, { control: true }));
  const label = await screen.findByLabelText('Name (entryway)');
  await userEvent.type(label, 'Front door');
  expect(draft().places).toBe('cam0: front camera, entryway: Front door');
  expect(screen.getByRole('checkbox', { name: 'Use entryway' })).toBeChecked();
});

function renderDevices(control = true, settings: { cams?: number[] } = {}) {
  const view = config(control, settings);
  // The Network cameras card between Cameras and Microphones reads the ingest sources.
  server.use(http.get('/ingest/api/items', () => HttpResponse.json(INGEST)));
  const page = renderPage(
    <DraftProvider view={view}>
      <Devices />
      <Values />
    </DraftProvider>,
  );
  page.client.setQueryData(liveKey('vision'), {
    ts: NOW - 1,
    cameras: { cam0: { state: 'on', size: [1280, 720], fps: 30 }, cam1: { state: 'retrying' } },
  });
  page.client.setQueryData(liveKey('ambient'), {
    ts: NOW - 1,
    devices: {
      wave: {
        name: WAVE,
        level_dbfs: -45,
        threshold_dbfs: -75,
        healthy: true,
        gain: { auto: true, kind: 'hardware', db: 12.5 },
      },
    },
  });
  return page;
}

describe('Settings: Cameras and microphones (#384)', () => {
  beforeEach(() => {
    useLiveStatus.setState({ now: NOW });
    server.use(http.get('/devices', () => HttpResponse.json(devices)));
  });

  it.each(['__proto__', 'constructor', 'toString'])('does not invent a camera status for %s', async cam => {
    server.use(
      http.get('/devices', () => HttpResponse.json({ ...devices, cameras: [{ ...devices.cameras[0], stream: cam }] })),
    );
    renderDevices();
    expect(await screen.findByText('Off: capture is sending nothing from it')).toBeInTheDocument();
    expect(screen.queryByText('On')).toBeNull();
  });

  it('lists every card on the page in On this page, Network cameras and Bluetooth too (#2695)', async () => {
    renderDevices();
    const jump = (await screen.findByText(/^On this page:/)).closest('p') as HTMLElement;
    expect(
      within(jump)
        .getAllByRole('link')
        .map(a => [a.textContent, a.getAttribute('href')]),
    ).toEqual([
      ['Cameras', '#device-cameras'],
      ['Network cameras', '#network-cameras'],
      ['Microphones', '#device-microphones'],
      ['Bluetooth', '#settings-group-bluetooth'],
    ]);
  });

  it('names each camera without a room setting', async () => {
    renderDevices();
    expect(await screen.findByLabelText('Name (Facecam)')).toHaveValue('front camera');
    expect(screen.queryByLabelText('Room name')).not.toBeInTheDocument();
    expect(screen.queryByText('In this browser')).not.toBeInTheDocument();
    expect(await screen.findByText('On, 1280×720 at 30 fps')).toBeInTheDocument();
    expect(screen.getByText('Not responding, retrying')).toBeInTheDocument();
    const preview = screen.getByRole('img', { name: 'Preview of cam0' }).parentElement;
    expect(within(preview as HTMLElement).getByText('Front camera')).toBeInTheDocument();
  });

  it('asks for a name for the view while a camera has none, until one is typed (#2585)', async () => {
    renderDevices();
    const hint = 'No name yet, so pages call it cam1. Name the view, like front or desk.';
    expect(await screen.findByLabelText('Name (Room cam)')).toHaveAccessibleDescription(hint);
    expect(screen.getByLabelText('Name (Facecam)')).not.toHaveAccessibleDescription();
    await userEvent.type(screen.getByLabelText('Name (Room cam)'), 'rear');
    expect(screen.queryByText(hint)).not.toBeInTheDocument();
  });

  it('renames the camera on this page once the name is saved, without a reload (#2595)', async () => {
    vi.stubGlobal('NAMES', { cams: ['cam0', 'cam1'], views: { cam0: 'front camera' } });
    server.use(
      http.get('/health', () => HttpResponse.json({ boot: 'b1' })),
      http.post('/config', () => HttpResponse.json({ ok: true, restarted: [], restarting: false })),
    );
    renderDevices();
    await userEvent.type(await screen.findByLabelText('Name (Room cam)'), 'rear camera');
    expect(shell().views).toEqual({ cam0: 'front camera' }); // typed, not saved
    await userEvent.click(screen.getByRole('button', { name: 'Save the draft' }));
    await waitFor(() => expect(shell().views).toEqual({ cam0: 'front camera', cam1: 'rear camera' }));
  });

  it('puts a camera name into the draft', async () => {
    renderDevices();
    const name = await screen.findByLabelText('Name (Room cam)');
    await userEvent.type(name, 'rear camera');
    expect(draft()).toMatchObject({ places: 'cam0: front camera, cam1: rear camera' });
  });

  it('opens the zone editor for the camera whose Zones was pressed (#1316)', async () => {
    server.use(http.get('/zones', () => HttpResponse.json({ cameras: {} })));
    renderDevices();
    await screen.findByLabelText('Name (Room cam)');
    await userEvent.click(screen.getAllByRole('button', { name: 'Zones' })[1] as HTMLElement);
    expect(await screen.findByRole('dialog', { name: 'Zones on Room cam' })).toBeInTheDocument();
  });

  it('saves one upside-down camera through capture without changing the other camera', async () => {
    const sent: unknown[] = [];
    server.use(
      http.post('/devices/camera', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ...devices, cameras: devices.cameras.map(c => ({ ...c, flip: c.index === 0 })) });
      }),
    );
    renderDevices();
    const [front, rear] = await screen.findAllByRole('checkbox', { name: 'Upside down' });
    await userEvent.click(front as HTMLElement);
    await waitFor(() => expect(front).toBeChecked());
    expect(rear).not.toBeChecked();
    expect(sent).toEqual([{ index: 0, flip: true }]);
  });

  it('places microphone defaults inside Microphones and keeps the gain target in the draft', async () => {
    renderDevices();
    const heading = await screen.findByRole('heading', { name: 'Microphones' });
    const card = heading.closest('section');
    expect(card).toContainElement(screen.getByRole('slider', { name: 'Default sensitivity' }));
    const target = screen.getByRole('textbox', { name: 'Auto gain target dbfs' });
    expect(card).toContainElement(target);
    await userEvent.clear(target);
    await userEvent.type(target, '-24');
    expect(target).toHaveValue('-24');
  });

  it("stops using a camera at once, through capture's manifest", async () => {
    const sent: unknown[] = [];
    const [front, roomRow] = devices.cameras;
    server.use(
      http.post('/devices/camera', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ...devices, cameras: [front, { ...roomRow, selected: false, stream: null }] });
      }),
    );
    renderDevices();
    const [, roomCam] = await screen.findAllByRole('checkbox', { name: /^Use / });
    await userEvent.click(roomCam as HTMLElement);
    await waitFor(() => expect(roomCam).not.toBeChecked());
    expect(sent).toEqual([{ index: 1, on: false }]);
    // Off, it shows only its name and Use: its settings are hidden (SourceItem)
    const room = screen.getByLabelText('Room cam', { selector: 'div' });
    expect(within(room).getAllByRole('checkbox')).toHaveLength(1);
  });

  it('hides the settings of a camera that is not used, and shows them again as they were once it is (#1331)', async () => {
    const [front, roomRow] = devices.cameras;
    const video = { width: 1280, height: 720, fps: 30, format: 'MJPEG', auto_exposure: true, exposure_target: 110 };
    const camera = (selected: boolean) => ({ ...roomRow, selected, flip: true, video });
    server.use(
      http.get('/devices', () => HttpResponse.json({ ...devices, cameras: [front, camera(false)] })),
      http.post('/devices/camera', async ({ request }) => {
        const { on } = (await request.json()) as { on: boolean };
        return HttpResponse.json({ ...devices, cameras: [front, camera(on)] });
      }),
    );
    renderDevices();
    const room = await screen.findByLabelText('Room cam', { selector: 'div' });
    expect(within(room).getByText('Room cam')).toBeInTheDocument();
    expect(within(room).getByRole('checkbox', { name: /^Use/ })).not.toBeChecked();
    for (const gone of ['Mirror', 'Upside down'])
      expect(within(room).queryByRole('checkbox', { name: gone })).not.toBeInTheDocument();
    expect(within(room).queryByRole('button', { name: 'Zones' })).not.toBeInTheDocument();
    expect(within(room).queryByRole('textbox')).not.toBeInTheDocument(); // its name field
    expect(within(room).queryByText(/Resolution/)).not.toBeInTheDocument();
    expect(within(room).queryByRole('img')).not.toBeInTheDocument(); // no preview either
    await userEvent.click(within(room).getByRole('checkbox', { name: /^Use/ }));
    expect(await within(room).findByRole('checkbox', { name: 'Upside down' })).toBeChecked(); // kept
    expect(within(room).getByRole('button', { name: 'Zones' })).toBeInTheDocument();
    expect(within(room).getByRole('textbox')).toBeInTheDocument();
    expect(within(room).getAllByText(/Resolution/).length).toBeGreaterThan(0);
  });

  it('hides the settings of a microphone it does not listen with, and shows them again once it does (#1331)', async () => {
    const [wave, desk] = devices.mics;
    const mic = (selected: boolean) => ({ ...desk, selected, auto_gain: true });
    server.use(
      http.get('/devices', () => HttpResponse.json({ ...devices, mics: [wave, mic(false)] })),
      http.post('/devices/mic', async ({ request }) => {
        const { on } = (await request.json()) as { on: boolean };
        return HttpResponse.json({ ...devices, mics: [wave, mic(on)] });
      }),
    );
    renderDevices();
    const deskMic = await screen.findByLabelText('Desk Mic', { selector: 'div' });
    const listen = within(deskMic).getByRole('checkbox', { name: 'Listen with this' });
    expect(listen).not.toBeChecked();
    expect(within(deskMic).queryByRole('slider')).not.toBeInTheDocument();
    expect(within(deskMic).queryByRole('checkbox', { name: /^Auto gain/ })).not.toBeInTheDocument();
    expect(within(deskMic).queryByRole('meter')).not.toBeInTheDocument();
    await userEvent.click(listen);
    expect(await within(deskMic).findByRole('checkbox', { name: /^Auto gain/ })).toBeChecked(); // kept
    expect(within(deskMic).getByRole('slider')).toBeInTheDocument();
  });

  it('turns auto gain on for a microphone at once, and says which gain it moves (#410)', async () => {
    const posted: unknown[] = [];
    server.use(
      http.post('/devices/auto_gain', async ({ request }) => {
        posted.push(await request.json());
        const [wave, desk] = devices.mics;
        return HttpResponse.json({ ...devices, mics: [{ ...wave, auto_gain: true }, desk] });
      }),
    );
    renderDevices();
    const [waveBox] = await screen.findAllByRole('checkbox', { name: /^Auto gain/ });
    await userEvent.click(waveBox as HTMLElement);
    await waitFor(() => expect(posted).toEqual([{ mic: WAVE, on: true }]));
    expect(await screen.findByText('Hardware gain +12.5 dB')).toBeInTheDocument(); // from the live status
    expect(screen.getAllByText(/can change how voices are recognized/i)[0]).toBeInTheDocument();
    expect(waveBox).toBeChecked();
    expect(draft().changedKeys).toEqual([]); // applied directly: nothing for the save bar
  });

  it('keeps one camera and one microphone at least', async () => {
    const [frontRow, roomRow] = devices.cameras;
    const [waveRow, deskRow] = devices.mics;
    server.use(
      http.post('/devices/camera', () =>
        HttpResponse.json({ ...devices, cameras: [frontRow, { ...roomRow, selected: false, stream: null }] }),
      ),
      http.post('/devices/mic', () =>
        HttpResponse.json({ ...devices, mics: [waveRow, { ...deskRow, selected: false, stream: null }] }),
      ),
    );
    renderDevices();
    const [front, room] = await screen.findAllByRole('checkbox', { name: /^Use / });
    await userEvent.click(room as HTMLElement);
    await waitFor(() => expect(front).toBeDisabled()); // none would leave capture no camera
    const [wave, desk] = screen.getAllByRole('checkbox', { name: 'Listen with this' });
    await userEvent.click(desk as HTMLElement);
    await waitFor(() => expect(wave).toBeDisabled()); // none would leave capture no microphone
  });

  it("stops listening with a microphone at once, through capture's manifest", async () => {
    const sent: unknown[] = [];
    const [waveRow, deskRow] = devices.mics;
    server.use(
      http.post('/devices/mic', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ...devices, mics: [waveRow, { ...deskRow, selected: false, stream: null }] });
      }),
    );
    renderDevices();
    const boxes = await screen.findAllByRole('checkbox', { name: 'Listen with this' });
    expect(boxes.every(b => (b as HTMLInputElement).checked)).toBe(true);
    await userEvent.click(boxes[1] as HTMLElement);
    await waitFor(() => expect(boxes[1]).not.toBeChecked());
    expect(sent).toEqual([{ id: 'Desk Mic', on: false }]);
  });

  it('shows each microphone by a short name, with its level and the threshold mark', async () => {
    renderDevices();
    expect(await screen.findByRole('meter', { name: 'Elgato Wave 3 level' })).toHaveAttribute('value', '50');
    expect(screen.getByTestId('threshold').style.left).toBe('16.7%');
  });

  it("saves a microphone's sensitivity at once, outside the draft", async () => {
    const posted: unknown[] = [];
    server.use(
      http.post('/devices/sensitivity', async ({ request }) => {
        posted.push(await request.json());
        return HttpResponse.json(devices);
      }),
    );
    renderDevices();
    const slider = await screen.findByRole('slider', { name: 'Sensitivity for Desk Mic' });
    slider.focus();
    await userEvent.keyboard('{ArrowRight}');
    expect(await screen.findByText('Saved, in use now')).toBeInTheDocument();
    expect(posted).toEqual([{ mic: 'Desk Mic', sensitivity: 90 }]);
  });

  it('saves the default sensitivity at once, and a microphone without its own follows it', async () => {
    const posted: unknown[] = [];
    server.use(
      http.post('/devices/sensitivity', async ({ request }) => {
        posted.push(await request.json());
        const [wave, desk] = devices.mics;
        return HttpResponse.json({ ...devices, default_sensitivity: 75, mics: [{ ...wave, sensitivity: 75 }, desk] });
      }),
    );
    renderDevices();
    const slider = await screen.findByRole('slider', { name: 'Default sensitivity' });
    slider.focus();
    await userEvent.keyboard('{ArrowRight}');
    await waitFor(() => expect(posted).toEqual([{ default: 75 }]));
    expect(await screen.findByRole('slider', { name: 'Sensitivity for Elgato Wave 3' })).toHaveAttribute(
      'aria-valuenow',
      '75',
    );
    expect(screen.getByRole('slider', { name: 'Sensitivity for Desk Mic' })).toHaveAttribute('aria-valuenow', '85');
  });

  it('keeps one plugged-in camera at least, not counting one that is unplugged', async () => {
    const absent = {
      index: -1,
      name: 'cam4',
      path: null,
      present: false,
      selected: true,
      stream: 'cam4',
      in_use: false,
    };
    server.use(http.get('/devices', () => HttpResponse.json({ ...devices, cameras: [devices.cameras[0], absent] })));
    renderDevices(true, { cams: [0, 4] });
    const [front] = await screen.findAllByRole('checkbox', { name: /^Use / });
    expect(front).toBeDisabled();
  });

  it("puts a microphone with its own sensitivity back on the default, and a failed save shows what's saved", async () => {
    let fail = false;
    const posted: unknown[] = [];
    server.use(
      http.post('/devices/sensitivity', async ({ request }) => {
        if (fail) return HttpResponse.json({ ok: false, error: 'disk full' }, { status: 500 });
        posted.push(await request.json());
        const [wave, desk] = devices.mics;
        return HttpResponse.json({ ...devices, mics: [wave, { ...desk, sensitivity: 70, sensitivity_custom: false }] });
      }),
    );
    renderDevices();
    await userEvent.click(await screen.findByRole('button', { name: 'Use default' }));
    const slider = await screen.findByRole('slider', { name: 'Sensitivity for Desk Mic' });
    await waitFor(() => expect(slider).toHaveAttribute('aria-valuenow', '70'));
    expect(posted).toEqual([{ mic: 'Desk Mic', sensitivity: null }]);
    expect(screen.queryByRole('button', { name: 'Use default' })).not.toBeInTheDocument();
    fail = true;
    slider.focus();
    await userEvent.keyboard('{ArrowRight}');
    expect(await screen.findByText(/Not saved/)).toBeInTheDocument();
    expect(slider).toHaveAttribute('aria-valuenow', '70'); // the saved value, not the page's first
  });

  it('is read only from another machine', async () => {
    renderDevices(false);
    expect(await screen.findByLabelText('Name (Facecam)')).toBeDisabled();
    expect(screen.getAllByRole('checkbox', { name: 'Upside down' }).every(x => (x as HTMLInputElement).disabled)).toBe(
      true,
    );
    expect(screen.getByText(/can only be changed from the machine/)).toBeInTheDocument();
    const [mic] = await screen.findAllByRole('checkbox', { name: 'Listen with this' });
    expect(mic).toBeDisabled();
  });
});

function videoDevices() {
  return {
    ...devices,
    cameras: devices.cameras.map((camera, i) => ({
      ...camera,
      video: { width: 1280, height: 720, fps: 30, format: 'MJPEG', auto_exposure: true, exposure_target: 110 },
      modes: {
        status: 'ok',
        choices:
          i === 0
            ? [
                { width: 1280, height: 720, fps: [30] },
                { width: 640, height: 480, fps: [60] },
              ]
            : [{ width: 1280, height: 720, fps: [30] }],
        diagnostics: [],
      },
      exposure_supported: i === 0,
      exposure_note: i === 0 ? '' : 'Automatic exposure control is unavailable on this platform.',
    })),
  };
}

describe('camera video controls', () => {
  beforeEach(() => {
    useLiveStatus.setState({ now: NOW });
    server.use(http.get('/devices', () => HttpResponse.json(videoDevices())));
  });

  it('offers a resolution select only for multiple modes and saves the exact camera and rate', async () => {
    const posted: unknown[] = [];
    const view = videoDevices();
    server.use(
      http.post('/devices/camera', async ({ request }) => {
        posted.push(await request.json());
        return HttpResponse.json({
          ...view,
          cameras: view.cameras.map(c =>
            c.index === 0 ? { ...c, video: { ...c.video, width: 640, height: 480, fps: 60 } } : c,
          ),
          restart: { status: 'restarting', device: 'cam0', message: 'Camera restart scheduled' },
        });
      }),
    );
    renderDevices();
    const select = await screen.findByRole('combobox', { name: 'Resolution for Facecam' });
    expect(screen.queryByRole('combobox', { name: 'Resolution for Room cam' })).not.toBeInTheDocument();
    expect(screen.getByText('Resolution: 1280×720 at 30 fps')).toBeInTheDocument();
    await userEvent.selectOptions(select, '640x480@60');
    await waitFor(() => expect(posted).toEqual([{ index: 0, width: 640, height: 480, fps: 60 }]));
    expect(await screen.findByText(/restarting cam0/i)).toBeInTheDocument();
    expect(select).toHaveValue('640x480@60');
  });

  it('turns steering off for one camera and disables its brightness target', async () => {
    const posted: unknown[] = [];
    const view = videoDevices();
    server.use(
      http.post('/devices/camera', async ({ request }) => {
        posted.push(await request.json());
        return HttpResponse.json({
          ...view,
          cameras: view.cameras.map(c => (c.index === 0 ? { ...c, video: { ...c.video, auto_exposure: false } } : c)),
          restart: { status: 'restarting', device: 'cam0', message: 'Camera restart scheduled' },
        });
      }),
    );
    renderDevices();
    const exposure = await screen.findByRole('switch', { name: 'Automatic exposure for Facecam' });
    expect(screen.queryByRole('switch', { name: 'Automatic exposure for Room cam' })).not.toBeInTheDocument();
    expect(screen.getByText('Automatic exposure control is unavailable on this platform.')).toBeInTheDocument();
    await userEvent.click(exposure);
    await waitFor(() => expect(posted).toEqual([{ index: 0, auto_exposure: false }]));
    expect(exposure).not.toBeChecked();
    expect(screen.getByRole('slider', { name: 'Target brightness for Facecam' })).toHaveAttribute(
      'aria-disabled',
      'true',
    );
    expect(screen.getByText(/Off uses the camera/)).toBeInTheDocument();
  });

  it('saves target brightness when the slider is released', async () => {
    const posted: unknown[] = [];
    const view = videoDevices();
    server.use(
      http.post('/devices/camera', async ({ request }) => {
        posted.push(await request.json());
        return HttpResponse.json({
          ...view,
          cameras: view.cameras.map(c => (c.index === 0 ? { ...c, video: { ...c.video, exposure_target: 115 } } : c)),
          restart: { status: 'restarting', device: 'cam0', message: 'Camera restart scheduled' },
        });
      }),
    );
    renderDevices();
    const slider = await screen.findByRole('slider', { name: 'Target brightness for Facecam' });
    slider.focus();
    await userEvent.keyboard('{ArrowRight}');
    await waitFor(() => expect(posted).toEqual([{ index: 0, exposure_target: 115 }]));
  });

  it.each(['failed', 'unavailable'] as const)(
    'keeps saved settings after %s restart and requires an explicit global restart',
    async status => {
      let globalRestarts = 0;
      const view = videoDevices();
      const saved = {
        ...view,
        cameras: view.cameras.map(c =>
          c.index === 0 ? { ...c, video: { ...c.video, width: 640, height: 480, fps: 60 } } : c,
        ),
      };
      server.use(
        http.post('/devices/camera', () =>
          HttpResponse.json({
            ...saved,
            restart: { status, device: 'cam0', message: 'Control request could not finish' },
            restart_all_available: true,
          }),
        ),
        http.post('/devices/capture/restart', () => {
          globalRestarts++;
          return HttpResponse.json({
            ...saved,
            restart: { status: 'restarting', device: 'all', message: 'Restarting all capture' },
          });
        }),
      );
      renderDevices();
      const select = await screen.findByRole('combobox', { name: 'Resolution for Facecam' });
      await userEvent.selectOptions(select, '640x480@60');
      expect(await screen.findByText(/Couldn't restart cam0/)).toBeInTheDocument();
      expect(select).toHaveValue('640x480@60');
      expect(globalRestarts).toBe(0);
      await userEvent.click(screen.getByRole('button', { name: 'Restart all capture' }));
      await waitFor(() => expect(globalRestarts).toBe(1));
      await waitFor(() =>
        expect(screen.queryByRole('button', { name: 'Restart all capture' })).not.toBeInTheDocument(),
      );
    },
  );

  it('disables video changes for a read-only visitor', async () => {
    renderDevices(false);
    expect(await screen.findByRole('combobox', { name: 'Resolution for Facecam' })).toBeDisabled();
    expect(screen.getByRole('switch', { name: 'Automatic exposure for Facecam' })).toBeDisabled();
    expect(screen.getByRole('slider', { name: 'Target brightness for Facecam' })).toHaveAttribute(
      'aria-disabled',
      'true',
    );
  });

  it('shows the configured size as text when capability discovery is unavailable', async () => {
    const view = videoDevices();
    server.use(
      http.get('/devices', () =>
        HttpResponse.json({
          ...view,
          cameras: [
            {
              ...view.cameras[0],
              modes: { status: 'unavailable', choices: [], diagnostics: ['Camera modes unavailable'] },
            },
          ],
        }),
      ),
    );
    renderDevices();
    expect(await screen.findByText('Resolution: 1280×720 at 30 fps')).toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: 'Resolution for Facecam' })).not.toBeInTheDocument();
    expect(screen.getByText('Camera modes unavailable')).toBeInTheDocument();
  });

  it('retains an unadvertised configured mode as a disabled current value', async () => {
    const view = videoDevices();
    server.use(
      http.get('/devices', () =>
        HttpResponse.json({
          ...view,
          cameras: [{ ...view.cameras[0], video: { ...view.cameras[0]?.video, width: 1920, height: 1080 } }],
        }),
      ),
    );
    renderDevices();
    expect(await screen.findByRole('option', { name: '1920×1080 at 30 fps (current)' })).toBeDisabled();
    expect(screen.getByRole('combobox', { name: 'Resolution for Facecam' })).toHaveValue('1920x1080@30');
  });
});

/** Stands in for the browser's EventSource, which jsdom lacks. */
class FakeEventSource {
  static readonly CLOSED = 2;
  static all: FakeEventSource[] = [];
  readonly CLOSED = 2;
  readyState = 1;
  readonly url: string;
  onmessage: ((e: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(url: string) {
    this.url = url;
    FakeEventSource.all.push(this);
  }
  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
  send(data: unknown) {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }));
  }
}

describe('live levels on the microphone meters', () => {
  // Animation frames run only when a test steps them, at a time it picks.
  let frames: Map<number, FrameRequestCallback>;
  let nextId: number;
  const step = (now: number) =>
    act(() => {
      const due = [...frames.values()];
      frames.clear();
      for (const f of due) f(now);
    });

  beforeEach(() => {
    useLiveStatus.setState({ now: NOW });
    server.use(http.get('/devices', () => HttpResponse.json(devices)));
    FakeEventSource.all = [];
    frames = new Map();
    nextId = 0;
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('requestAnimationFrame', (f: FrameRequestCallback) => {
      frames.set(++nextId, f);
      return nextId;
    });
    vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id));
  });
  afterEach(() => vi.unstubAllGlobals());

  it('moves the meter to each level, holding the peak as it falls 20 dB a second', async () => {
    const { container, unmount } = renderDevices();
    await waitFor(() => expect(container.querySelector('[data-level-key="wave"]')).not.toBeNull());
    const meter = container.querySelector<HTMLElement>('[data-level-key="wave"]') as HTMLElement;
    const bar = meter.firstElementChild as HTMLElement;
    const peak = meter.lastElementChild as HTMLElement;
    const stream = FakeEventSource.all.find(s => s.url === '/audio/levels');
    expect(bar.style.getPropertyValue('--level')).toBe('50.0%'); // the status file's -45 dBFS until the stream speaks

    const t0 = performance.now();
    step(t0 + 40);
    expect(meter).not.toHaveAttribute('data-live'); // nothing streamed yet

    stream?.send({ ts: 1, levels: { wave: { rms: -60, peak: -18 } } });
    step(t0 + 80);
    expect(meter).toHaveAttribute('data-live');
    expect(bar.style.transform).toBe('scaleX(0.333)');
    expect(peak.style.transform).toBe('translateX(80%)');

    // Half a second later the -18 peak has fallen to -28, still above the new -60.
    stream?.send({ ts: 2, levels: { wave: { rms: -90, peak: -60 } } });
    step(t0 + 580);
    expect(bar.style.transform).toBe('scaleX(0)');
    expect(peak.style.transform).toBe('translateX(68.9%)');

    unmount();
    expect(stream?.readyState).toBe(FakeEventSource.CLOSED);
  });

  it('moves the meter at most every METER_MS, not every frame (#486)', async () => {
    const { container, unmount } = renderDevices();
    await waitFor(() => expect(container.querySelector('[data-level-key="wave"]')).not.toBeNull());
    const bar = container.querySelector<HTMLElement>('[data-level-key="wave"]')?.firstElementChild as HTMLElement;
    const stream = FakeEventSource.all.find(s => s.url === '/audio/levels');
    const t0 = performance.now();
    stream?.send({ ts: 1, levels: { wave: { rms: -60, peak: -60 } } });
    step(t0 + METER_MS);
    expect(bar.style.transform).toBe('scaleX(0.333)');
    stream?.send({ ts: 2, levels: { wave: { rms: -45, peak: -45 } } });
    step(t0 + METER_MS + 16); // the next frame, too soon
    expect(bar.style.transform).toBe('scaleX(0.333)');
    step(t0 + 2 * METER_MS);
    expect(bar.style.transform).toBe('scaleX(0.5)');
    unmount();
  });

  it('goes back to the status level when the stream gives up, instead of freezing', async () => {
    const { container, unmount } = renderDevices();
    await waitFor(() => expect(container.querySelector('[data-level-key="wave"]')).not.toBeNull());
    const meter = container.querySelector<HTMLElement>('[data-level-key="wave"]') as HTMLElement;
    const bar = meter.firstElementChild as HTMLElement;
    const stream = FakeEventSource.all.find(s => s.url === '/audio/levels');
    const t0 = performance.now();
    stream?.send({ ts: 1, levels: { wave: { rms: -60, peak: -60 } } });
    step(t0 + METER_MS);
    expect(meter).toHaveAttribute('data-live');
    if (stream) stream.readyState = FakeEventSource.CLOSED;
    stream?.onerror?.();
    step(t0 + 2 * METER_MS);
    expect(meter).not.toHaveAttribute('data-live');
    expect(bar.style.transform).toBe('');
    expect(bar.style.getPropertyValue('--level')).toBe('50.0%'); // the status file's level shows again
    unmount();
  });

  it('moves only the meters of microphones the microphone service reports', async () => {
    const { container } = renderDevices();
    await screen.findByRole('meter', { name: 'Desk Mic level' });
    await waitFor(() => expect(container.querySelector('[data-level-key="wave"]')).not.toBeNull());
    expect(container.querySelectorAll('[data-level-key]')).toHaveLength(1); // Desk Mic has no live status
  });
});
