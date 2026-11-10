import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { CameraControls, type CameraControlsView } from './CameraControls';

const view: CameraControlsView = {
  camera: 'desk',
  available: true,
  can_control: true,
  auto_exposure: false,
  controls: [
    { id: 1, name: 'Brightness', type: 'integer', min: 0, max: 100, step: 1, default: 50, current: 25, disabled: true },
    { id: 2, name: 'Focus', type: 'boolean', min: 0, max: 1, step: 1, default: 1, current: 0, disabled: false },
    {
      id: 3,
      name: 'Frequency',
      type: 'menu',
      min: 0,
      max: 2,
      step: 1,
      default: 0,
      current: 0,
      disabled: false,
      options: [
        { value: 0, label: 'Off' },
        { value: 2, label: '60 Hz' },
      ],
    },
  ],
};
function setup(data = view) {
  const writes: unknown[] = [];
  server.use(
    http.get('/devices/camera-controls', () => HttpResponse.json(data)),
    http.post('/devices/camera-controls', async ({ request }) => {
      writes.push(await request.json());
      return HttpResponse.json(data);
    }),
  );
  const rendered = renderPage(<CameraControls camera="desk" name="Desk" />);
  return { ...rendered, writes };
}
async function open() {
  await userEvent.click(screen.getByRole('button', { name: 'Controls for Desk' }));
  await screen.findByRole('combobox', { name: 'Frequency' });
}
it('reads without writes and coalesces boolean and menu edits', async () => {
  const { writes } = setup();
  await open();
  expect(writes).toHaveLength(0);
  expect(screen.getByRole('slider')).toHaveAttribute('data-disabled');
  fireEvent.click(screen.getByRole('switch', { name: 'Focus' }));
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '2' } });
  expect(writes).toHaveLength(0);
  await waitFor(() => expect(writes).toEqual([{ camera: 'desk', values: { 2: true, 3: 2 } }]));
});
it('resets through the API', async () => {
  const { writes } = setup();
  await open();
  await userEvent.click(screen.getByRole('button', { name: 'Reset defaults' }));
  await waitFor(() => expect(writes).toEqual([{ camera: 'desk', reset: true }]));
});
it('disables mutations for read only clients', async () => {
  setup({ ...view, can_control: false });
  await open();
  expect(screen.getByRole('checkbox')).toBeDisabled();
  expect(screen.getByRole('switch')).toBeDisabled();
  expect(screen.getByRole('combobox')).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Reset defaults' })).toBeDisabled();
});
it('discards queued writes on unmount', async () => {
  const { writes, unmount } = setup();
  await open();
  fireEvent.click(screen.getByRole('switch'));
  unmount();
  await new Promise(resolve => setTimeout(resolve, 350));
  expect(writes).toHaveLength(0);
});
it('recovers after a failed write', async () => {
  const { writes } = setup();
  await open();
  server.use(http.post('/devices/camera-controls', () => HttpResponse.json({ error: 'Unavailable' }, { status: 503 })));
  fireEvent.click(screen.getByRole('switch'));
  await screen.findByRole('alert');
  server.use(
    http.post('/devices/camera-controls', async ({ request }) => {
      writes.push(await request.json());
      return HttpResponse.json(view);
    }),
  );
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '2' } });
  await waitFor(() => expect(writes).toHaveLength(1));
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});
it('serializes writes while preserving edits made during an in flight save', async () => {
  const { writes } = setup();
  let finish: (() => void) | undefined;
  server.use(
    http.post('/devices/camera-controls', async ({ request }) => {
      writes.push(await request.json());
      if (writes.length === 1)
        await new Promise<void>(resolve => {
          finish = resolve;
        });
      return HttpResponse.json(view);
    }),
  );
  await open();
  fireEvent.click(screen.getByRole('switch'));
  await waitFor(() => expect(writes).toHaveLength(1));
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '2' } });
  await new Promise(resolve => setTimeout(resolve, 350));
  expect(writes).toHaveLength(1);
  finish?.();
  await waitFor(() => expect(writes).toHaveLength(2));
  expect(writes[1]).toEqual({ camera: 'desk', values: { 3: 2 } });
});
it('polls pending hardware changes and replaces optimistic values after application', async () => {
  const { writes } = setup();
  await open();
  server.use(
    http.post('/devices/camera-controls', async ({ request }) => {
      writes.push(await request.json());
      return HttpResponse.json({ ...view, pending: true });
    }),
  );
  fireEvent.click(screen.getByRole('switch'));
  await screen.findByText('Saved. Applying camera controls…');
  expect(screen.getByRole('switch')).toBeChecked();
  await waitFor(() => expect(screen.queryByText('Saved. Applying camera controls…')).not.toBeInTheDocument(), {
    timeout: 3000,
  });
  expect(screen.getByRole('switch')).not.toBeChecked();
});
it('restarts bounded polling after a new edit following an application timeout', async () => {
  setup();
  await open();
  let reads = 0;
  let applied = false;
  server.use(
    http.get('/devices/camera-controls', () => {
      reads++;
      return HttpResponse.json({ ...view, pending: !applied });
    }),
    http.post('/devices/camera-controls', () => HttpResponse.json({ ...view, pending: true })),
  );
  fireEvent.click(screen.getByRole('switch'));
  await screen.findByText('Saved, but camera has not applied these values.', {}, { timeout: 10000 });
  expect(reads).toBe(6);
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '2' } });
  await waitFor(() =>
    expect(screen.queryByText('Saved, but camera has not applied these values.')).not.toBeInTheDocument(),
  );
  applied = true;
  await waitFor(() => expect(reads).toBe(7), { timeout: 3000 });
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
}, 15000);

it('sends reset separately from edits made within its debounce window', async () => {
  const { writes } = setup();
  await open();
  fireEvent.click(screen.getByRole('button', { name: 'Reset defaults' }));
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '2' } });
  await waitFor(() =>
    expect(writes).toEqual([
      { camera: 'desk', reset: true },
      { camera: 'desk', values: { 3: 2 } },
    ]),
  );
});
it('keeps a queued reset separate from later edits behind an in flight save', async () => {
  const { writes } = setup();
  let finish: (() => void) | undefined;
  server.use(
    http.post('/devices/camera-controls', async ({ request }) => {
      writes.push(await request.json());
      if (writes.length === 1)
        await new Promise<void>(resolve => {
          finish = resolve;
        });
      return HttpResponse.json(view);
    }),
  );
  await open();
  fireEvent.click(screen.getByRole('switch'));
  await waitFor(() => expect(writes).toHaveLength(1));
  fireEvent.click(screen.getByRole('button', { name: 'Reset defaults' }));
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '2' } });
  expect(writes).toEqual([{ camera: 'desk', values: { 2: true } }]);
  finish?.();
  await waitFor(() =>
    expect(writes).toEqual([
      { camera: 'desk', values: { 2: true } },
      { camera: 'desk', reset: true },
      { camera: 'desk', values: { 3: 2 } },
    ]),
  );
});
