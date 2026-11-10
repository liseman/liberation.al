import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { PlaceName } from './AccessPoint';

const AP = '1c:0b:8b:00:00:bc';

it('names an access point by its MAC, the box starting empty while the MAC is all it shows', async () => {
  let sent: unknown = null;
  server.use(
    http.post('/device-access-point', async ({ request }) => {
      sent = await request.json();
      return HttpResponse.json({ ok: true, access_points: { [AP]: 'Living room' } });
    }),
  );
  const user = userEvent.setup();
  renderPage(<PlaceName place={AP.toUpperCase()} placeId={AP} control />);
  await user.click(screen.getByRole('button', { name: `Rename access point ${AP.toUpperCase()}` }));
  const form = await screen.findByRole('form', { name: `Rename access point ${AP.toUpperCase()}` });
  const box = within(form).getByLabelText('Access point name');
  expect(box).toHaveValue('');
  await user.type(box, ' Living room ');
  await user.click(within(form).getByRole('button', { name: 'Save' }));
  await waitFor(() => expect(sent).toEqual({ mac: AP, name: 'Living room' }));
  await waitFor(() => expect(screen.queryByRole('form')).not.toBeInTheDocument());
});

it('starts from the shown name, and says why when the server refuses', async () => {
  server.use(
    http.post('/device-access-point', () => HttpResponse.json({ error: 'not a MAC address' }, { status: 400 })),
  );
  const user = userEvent.setup();
  renderPage(<PlaceName place="Upstairs" placeId={AP} control />);
  await user.click(screen.getByRole('button', { name: 'Rename access point Upstairs' }));
  const form = await screen.findByRole('form', { name: 'Rename access point Upstairs' });
  expect(within(form).getByLabelText('Access point name')).toHaveValue('Upstairs');
  await user.click(within(form).getByRole('button', { name: 'Save' }));
  expect(await within(form).findByRole('alert')).toHaveTextContent(/^Not renamed:/);
});

it.each([
  ['a Bluetooth sensor', { placeId: null, control: true }],
  ['away from this machine', { placeId: AP, control: false }],
])('offers no rename for %s', (_, props) => {
  renderPage(<PlaceName place="Desk" {...props} />);
  expect(screen.getByText('Desk')).toBeInTheDocument();
  expect(screen.queryByRole('button')).not.toBeInTheDocument();
});
