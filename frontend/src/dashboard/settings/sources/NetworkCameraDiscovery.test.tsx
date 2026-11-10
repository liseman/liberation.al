import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { shell } from '../../shell';
import { DEVICES_KEY } from '../../sources/api';
import { CONFIG_KEY } from '../model';
import { NetworkCameraDiscovery } from './NetworkCameraDiscovery';
import { type CameraDiscovery, NETWORK_CAMERAS_KEY } from './networkCameras';

const discovery = (over: Partial<CameraDiscovery> = {}): CameraDiscovery => ({
  discovery_id: 'proposal-test-1',
  capture_ready: true,
  warnings: ['Some cameras may not have responded.'],
  cameras: [
    {
      id: 'camera-1',
      source: 'onvif',
      name: 'Workshop camera',
      model: 'Fixture model',
      ip: '192.168.40.8',
      requires_credentials: false,
      streams: [
        { id: 'stream-main', quality: 'Main', url_display: 'rtsp://192.168.40.8/[redacted]', ready: true },
        { id: 'stream-sub', quality: 'Sub', url_display: 'rtsp://192.168.40.8/[redacted-sub]', ready: true },
        { id: 'stream-unavailable', quality: 'Unavailable', url_display: 'Unavailable', ready: false },
      ],
    },
  ],
  ...over,
});

afterEach(() => vi.unstubAllGlobals());

function configured(ready = true) {
  server.use(http.get('/network-cameras', () => HttpResponse.json({ cameras: [], capture_ready: ready })));
}
function renderDiscovery(control = true) {
  const page = renderPage(<NetworkCameraDiscovery />);
  act(() => page.client.setQueryData(CONFIG_KEY, { control }));
  return page;
}
async function discoverClick() {
  await waitFor(() => expect(screen.getByRole('button', { name: 'Discover' })).toBeEnabled());
  await userEvent.click(screen.getByRole('button', { name: 'Discover' }));
}

// Each configured network camera's card reads capture's word on it, for its preview and status.
beforeEach(() => server.use(http.get('/vision', () => HttpResponse.json({}))));

describe('native network camera discovery', () => {
  it('turns a configured camera off and back on by immutable ID, refreshing Live without hiding history', async () => {
    vi.stubGlobal('NAMES', { cams: ['entryway', 'desk'], activeCams: ['entryway', 'desk'] });
    let enabled = true;
    const saved: unknown[] = [];
    server.use(
      http.get('/network-cameras', () =>
        HttpResponse.json({
          capture_ready: true,
          controls_ready: true,
          cameras: [
            {
              id: 'entryway',
              name: 'Front door',
              source: 'rtsp',
              url_display: 'rtsp://192.168.40.8/[redacted]',
              enabled,
            },
          ],
        }),
      ),
      http.post('/network-cameras/use', async ({ request }) => {
        const body = (await request.json()) as { id: string; enabled: boolean };
        saved.push(body);
        enabled = body.enabled;
        return HttpResponse.json({
          ok: true,
          restarted: ['continuum-capture'],
          active_cameras: enabled ? ['entryway', 'desk'] : ['desk'],
        });
      }),
    );
    renderDiscovery();
    const toggle = await screen.findByRole('checkbox', { name: 'Use Front door' });
    expect(toggle).toBeChecked();
    await userEvent.click(toggle);
    await waitFor(() => expect(toggle).not.toBeChecked());
    expect(saved).toEqual([{ id: 'entryway', enabled: false }]);
    expect(shell().activeCams).toEqual(['desk']);
    expect(shell().cams).toEqual(['entryway', 'desk']);
    expect(screen.getByText(/Capture is restarting/)).toBeInTheDocument();
    await userEvent.click(toggle);
    await waitFor(() => expect(toggle).toBeChecked());
    expect(saved).toEqual([
      { id: 'entryway', enabled: false },
      { id: 'entryway', enabled: true },
    ]);
    expect(shell().activeCams).toEqual(['entryway', 'desk']);
  });

  it.each([
    [false, true],
    [true, false],
  ])('cannot toggle unless owner=%s and native capability=%s', async (control, ready) => {
    server.use(
      http.get('/network-cameras', () =>
        HttpResponse.json({
          capture_ready: true,
          controls_ready: ready,
          cameras: [{ id: 'entryway', name: 'Front door', source: 'rtsp', url_display: 'redacted', enabled: true }],
        }),
      ),
    );
    renderDiscovery(control);
    expect(await screen.findByRole('checkbox', { name: 'Use Front door' })).toBeDisabled();
  });

  it('retains camera use and Live evidence when a save is unavailable', async () => {
    vi.stubGlobal('NAMES', { cams: ['entryway'], activeCams: ['entryway'] });
    server.use(
      http.get('/network-cameras', () =>
        HttpResponse.json({
          capture_ready: true,
          controls_ready: true,
          cameras: [{ id: 'entryway', name: 'Front door', source: 'rtsp', url_display: 'redacted', enabled: true }],
        }),
      ),
      http.post('/network-cameras/use', () =>
        HttpResponse.json({ error: 'Synthetic unavailable store' }, { status: 503 }),
      ),
    );
    renderDiscovery();
    const toggle = await screen.findByRole('checkbox', { name: 'Use Front door' });
    await userEvent.click(toggle);
    expect(await screen.findByText(/Could not change camera use: Synthetic unavailable store/)).toBeInTheDocument();
    expect(toggle).toBeChecked();
    expect(shell().activeCams).toEqual(['entryway']);
  });
  it('does not probe or add on mount, focus, or configured refresh', async () => {
    let probes = 0;
    let adds = 0;
    configured();
    server.use(
      http.post('/network-cameras/discover', () => {
        probes++;
        return HttpResponse.json(discovery());
      }),
      http.post('/network-cameras/add', () => {
        adds++;
        return HttpResponse.json({ ok: true });
      }),
    );
    const { client } = renderDiscovery();
    await waitFor(() => expect(client.getQueryData(NETWORK_CAMERAS_KEY)).toBeDefined());
    act(() => window.dispatchEvent(new Event('focus')));
    await act(() => client.invalidateQueries({ queryKey: NETWORK_CAMERAS_KEY }));
    expect(probes).toBe(0);
    expect(adds).toBe(0);
    expect(screen.queryByRole('list', { name: 'Discovered network cameras' })).not.toBeInTheDocument();
  });

  it('discovers the selected source and adds only opaque IDs and the chosen name', async () => {
    let sent: unknown;
    let added: unknown;
    configured();
    server.use(
      http.post('/network-cameras/discover', async ({ request }) => {
        sent = await request.json();
        return HttpResponse.json(discovery());
      }),
      http.post('/network-cameras/add', async ({ request }) => {
        added = await request.json();
        return HttpResponse.json({ ok: true });
      }),
    );
    const { client } = renderDiscovery();
    client.setQueryData(DEVICES_KEY, { stale: true });
    await waitFor(() => expect(screen.getByLabelText('Discovery source')).toBeEnabled());
    await userEvent.selectOptions(screen.getByLabelText('Discovery source'), 'onvif');
    await discoverClick();
    expect(await screen.findByText('Fixture model · 192.168.40.8')).toBeInTheDocument();
    expect(sent).toEqual({ source: 'onvif' });
    expect(screen.getByText('Some cameras may not have responded.')).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText('Stream for Workshop camera'), 'stream-sub');
    await userEvent.clear(screen.getByLabelText('Camera name for Workshop camera'));
    await userEvent.type(screen.getByLabelText('Camera name for Workshop camera'), '  test-garden  ');
    await userEvent.click(screen.getByRole('button', { name: 'Add Workshop camera' }));
    expect(await screen.findByText('Camera added.')).toBeInTheDocument();
    expect(added).toEqual({
      discovery_id: 'proposal-test-1',
      camera_id: 'camera-1',
      stream_id: 'stream-sub',
      name: 'test-garden',
    });
    expect(screen.getByRole('button', { name: 'Added' })).toBeDisabled();
    await waitFor(() => expect(client.getQueryState(DEVICES_KEY)?.isInvalidated).toBe(true));
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it.each([
    { getReady: false, postReady: true },
    { getReady: true, postReady: false },
  ])('keeps Add disabled when capture is unavailable ($getReady/$postReady)', async ({ getReady, postReady }) => {
    configured(getReady);
    server.use(
      http.post('/network-cameras/discover', () => HttpResponse.json(discovery({ capture_ready: postReady }))),
    );
    renderDiscovery();
    await discoverClick();
    const add = await screen.findByRole('button', { name: 'Add Workshop camera' });
    expect(add).toBeDisabled();
    expect(screen.getByText(/Network camera capture is not available/)).toBeInTheDocument();
    expect(screen.getByLabelText('Stream for Workshop camera')).toBeEnabled();
  });

  it('refuses unavailable profiles even when another profile is ready', async () => {
    configured();
    server.use(http.post('/network-cameras/discover', () => HttpResponse.json(discovery())));
    renderDiscovery();
    await discoverClick();
    await screen.findByRole('button', { name: 'Add Workshop camera' });
    await userEvent.selectOptions(screen.getByLabelText('Stream for Workshop camera'), 'stream-unavailable');
    expect(screen.getByRole('button', { name: 'Add Workshop camera' })).toBeDisabled();
  });

  it('shows credential requirements and missing streams without inventing a stream', async () => {
    configured();
    const result = discovery({
      cameras: [{ id: 'camera-1', source: 'onvif', name: 'Workshop camera', streams: [], requires_credentials: true }],
    });
    server.use(http.post('/network-cameras/discover', () => HttpResponse.json(result)));
    renderDiscovery();
    await discoverClick();
    expect(await screen.findByText(/Credentials are required/)).toBeInTheDocument();
    expect(screen.getByText('No usable streams were returned.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add Workshop camera' })).toBeDisabled();
  });

  it('uses global control permission even without an Ingest worker', async () => {
    configured();
    const { client } = renderDiscovery(false);
    expect(screen.getByRole('button', { name: 'Discover' })).toBeDisabled();
    act(() => client.setQueryData(CONFIG_KEY, { control: true }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Discover' })).toBeEnabled());
  });

  it('leaves failed discovery unknown and allows an explicit retry', async () => {
    configured();
    server.use(
      http.post('/network-cameras/discover', () =>
        HttpResponse.json({ error: 'Discovery unavailable' }, { status: 503 }),
      ),
    );
    renderDiscovery();
    await discoverClick();
    expect(await screen.findByText('Could not discover cameras: Discovery unavailable')).toBeInTheDocument();
    expect(screen.queryByText(/No cameras responded/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Discover' })).toBeEnabled();
  });

  it('reports empty results as unanswered coverage and preserves warnings', async () => {
    configured();
    server.use(http.post('/network-cameras/discover', () => HttpResponse.json(discovery({ cameras: [] }))));
    renderDiscovery();
    await discoverClick();
    expect(
      await screen.findByText('No cameras responded. Cameras that did not respond remain unknown.'),
    ).toBeInTheDocument();
    expect(screen.getByText('Some cameras may not have responded.')).toBeInTheDocument();
  });

  it('shows configured redacted cameras and keeps disabled camera details hidden', async () => {
    server.use(
      http.get('/network-cameras', () =>
        HttpResponse.json({
          capture_ready: true,
          cameras: [
            {
              id: 'native-1',
              name: 'Test porch',
              source: 'rtsp',
              url_display: 'rtsp://192.168.40.8/[redacted]',
              enabled: true,
            },
            {
              id: 'native-2',
              name: 'Test shed',
              source: 'rtsp',
              url_display: 'rtsp://192.168.40.9/[redacted]',
              enabled: false,
            },
          ],
        }),
      ),
    );
    renderDiscovery();
    const list = await screen.findByRole('list', { name: 'Configured network cameras' });
    expect(within(list).getByText('Test porch')).toBeInTheDocument();
    expect(within(list).getByText(/rtsp:\/\/192\.168\.40\.8\/\[redacted\]/)).toBeInTheDocument();
    expect(within(list).getByText('Test shed')).toBeInTheDocument();
    expect(within(list).queryByText(/rtsp:\/\/192\.168\.40\.9\//)).not.toBeInTheDocument();
  });

  it('retains the proposal after an Add refusal so a person can retry', async () => {
    configured();
    server.use(
      http.post('/network-cameras/discover', () => HttpResponse.json(discovery())),
      http.post('/network-cameras/add', () => HttpResponse.json({ error: 'Proposal expired' }, { status: 409 })),
    );
    renderDiscovery();
    await discoverClick();
    await userEvent.click(await screen.findByRole('button', { name: 'Add Workshop camera' }));
    expect(await screen.findByText('Could not add camera: Proposal expired')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add Workshop camera' })).toBeEnabled();
  });
});

it('disables duplicate discovery while a request is in progress', async () => {
  configured();
  let calls = 0;
  let finish: (() => void) | undefined;
  const gate = new Promise<void>(resolve => {
    finish = resolve;
  });
  server.use(
    http.post('/network-cameras/discover', async () => {
      calls++;
      await gate;
      return HttpResponse.json(discovery());
    }),
  );
  renderDiscovery();
  await discoverClick();
  await waitFor(() => expect(calls).toBe(1));
  expect(screen.getByRole('button', { name: 'Discover' })).toBeDisabled();
  await userEvent.click(screen.getByRole('button', { name: 'Discover' }));
  expect(calls).toBe(1);
  await act(async () => {
    finish?.();
    await gate;
  });
  expect(await screen.findByRole('button', { name: 'Add Workshop camera' })).toBeInTheDocument();
});

it('prefills safe manifest names and refuses invalid edited names', async () => {
  configured();
  server.use(http.post('/network-cameras/discover', () => HttpResponse.json(discovery())));
  renderDiscovery();
  await discoverClick();
  const input = await screen.findByLabelText('Camera name for Workshop camera');
  expect(input).toHaveValue('workshop-camera');
  expect(input).toHaveAttribute('maxlength', '64');
  await userEvent.clear(input);
  await userEvent.type(input, 'Front Door');
  expect(screen.getByText('Choose a valid camera name.')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Add Workshop camera' })).toBeDisabled();
  await userEvent.clear(input);
  await userEvent.type(input, 'front-door');
  expect(screen.getByRole('button', { name: 'Add Workshop camera' })).toBeEnabled();
});
