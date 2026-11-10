import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../test/render';
import { server } from '../test/server';
import type { PrivacyChange, PrivacyDevice, PrivacyView } from './privacy';
import { StatusPill } from './StatusIcons';

let devices: PrivacyDevice[];
let posts: PrivacyChange[];
let control: boolean;

const device = (name: string, kind: PrivacyDevice['kind'], paused = false): PrivacyDevice => ({
  name,
  kind,
  paused,
  state: paused ? 'paused' : 'on',
});

const view = (): PrivacyView => ({ control, ready: true, devices });

beforeEach(() => {
  control = true;
  posts = [];
  devices = [device('cam0', 'camera'), device('wave3', 'microphone'), device('patio', 'camera')];
  server.use(
    http.get('/ambient', () => HttpResponse.json({ ts: 0, devices: {} })),
    http.get('/scene', () => HttpResponse.json([])),
    http.get('/vision', () => HttpResponse.json({ ts: 0, cameras: {} })),
    http.get('/devices/privacy', () => HttpResponse.json(view())),
    http.post('/devices/privacy', async ({ request }) => {
      const change = (await request.json()) as PrivacyChange;
      posts.push(change);
      // As capture would see it saved: pausing until its worker has stopped.
      devices = devices.map(d =>
        d.kind === change.kind && (change.device === undefined || d.name === change.device)
          ? { ...d, paused: change.paused, state: change.paused ? 'pausing' : 'resuming' }
          : d,
      );
      return HttpResponse.json(view());
    }),
  );
});

afterEach(() => {
  server.resetHandlers();
  vi.unstubAllGlobals();
});

const pill = () => screen.findByRole('button', { name: /Microphone: / });

it('pauses the microphone from the menu, and resumes it from the next', async () => {
  const user = userEvent.setup();
  renderPage(<StatusPill />);
  await user.click(await pill());
  await user.click(await screen.findByRole('menuitem', { name: 'Pause the microphone' }));
  await waitFor(() => expect(screen.getByTestId('status-mic')).toHaveAttribute('data-state', 'paused'));
  expect(posts).toEqual([{ kind: 'microphone', paused: true }]);
  expect(await pill()).toHaveAccessibleName(/Microphone: Paused for privacy \(applying…\)\./);
  await user.click(await pill());
  await user.click(await screen.findByRole('menuitem', { name: 'Resume the microphone' }));
  await waitFor(() => expect(posts).toHaveLength(2));
  expect(posts[1]).toEqual({ kind: 'microphone', paused: false });
});

it('offers each camera on its own, and with only some paused the main item pauses the rest', async () => {
  const user = userEvent.setup();
  renderPage(<StatusPill />);
  await user.click(await pill());
  await user.click(await screen.findByRole('menuitem', { name: 'Pause patio' }));
  await waitFor(() => expect(posts).toEqual([{ kind: 'camera', paused: true, device: 'patio' }]));
  await waitFor(() => expect(screen.getByTestId('status-cams')).toHaveAttribute('data-state', 'paused'));
  await user.click(await pill());
  const menu = await screen.findByRole('menu');
  expect(within(menu).getByText(/patio paused for privacy/)).toBeInTheDocument();
  await user.click(within(menu).getByRole('menuitem', { name: 'Pause all cameras' }));
  await waitFor(() => expect(posts).toHaveLength(2));
  expect(posts[1]).toEqual({ kind: 'camera', paused: true });
});

it('shows a pause it cannot change from this browser, without offering to', async () => {
  const user = userEvent.setup();
  control = false;
  devices = [device('cam0', 'camera', true), device('wave3', 'microphone')];
  renderPage(<StatusPill />);
  await waitFor(() => expect(screen.getByTestId('status-cams')).toHaveAttribute('data-state', 'paused'));
  const button = await pill();
  expect(button).toHaveAccessibleName(/Cameras: Paused for privacy\.$/);
  await user.click(button);
  const menu = await screen.findByRole('menu');
  expect(within(menu).getByText('Pausing works only on the machine Liberational runs on.')).toBeInTheDocument();
  expect(within(menu).queryByRole('menuitem')).toBeNull();
});

it('counts a camera on by its frames, not its captions', async () => {
  vi.stubGlobal('NAMES', { cams: ['cam0', 'patio'] });
  control = false;
  // cam0 sends frames but has had no caption for a minute; patio sends nothing.
  server.use(
    http.get('/scene', () => HttpResponse.json([{ cam: 'cam0', ts: Date.now() / 1000 - 60 }])),
    http.get('/vision', () => HttpResponse.json({ ts: Date.now() / 1000, cameras: { cam0: { state: 'on' } } })),
  );
  renderPage(<StatusPill />);
  await waitFor(() => expect(screen.getByTestId('status-cams')).toHaveAttribute('data-state', 'some'));
  expect(await pill()).toHaveAccessibleName(/Cameras: 1 of 2 cameras on \(patio quiet\)\./);
  expect(screen.getByTestId('status')).toHaveTextContent('1 of 2 cameras');
});

it('says when a pause fails, on the pill and in the menu', async () => {
  const user = userEvent.setup();
  server.use(http.post('/devices/privacy', () => HttpResponse.json({ error: 'capture is busy' }, { status: 500 })));
  renderPage(<StatusPill />);
  await user.click(await pill());
  await user.click(await screen.findByRole('menuitem', { name: 'Pause the microphone' }));
  const failed = await screen.findByRole('button', { name: /^Couldn't change privacy: / });
  expect(screen.getByTestId('status')).toHaveAttribute('data-state', 'bad');
  await user.click(failed);
  expect(await screen.findByRole('alert')).toHaveTextContent(/capture is busy/);
});
