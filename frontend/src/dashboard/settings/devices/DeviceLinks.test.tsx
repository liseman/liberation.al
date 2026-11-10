import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, describe, expect, it } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { DeviceLinks, UNKNOWN_SHOWN } from './DeviceLinks';
import { DEVICE_LINKS_KEY, type DeviceLinksView, type SeenDevice } from './links';

const now = () => Date.now() / 1000;
const WATCH = 'c0:ff:ee:00:00:01';
const FINGERPRINT = 'ab'.repeat(32);

const device = (over: Partial<SeenDevice> = {}): SeenDevice => ({
  source: 'bluetooth.sighting',
  identifier: WATCH,
  name: null,
  place: 'Hall',
  rssi: -48,
  last_seen: now() - 120,
  sightings: 3,
  selectors: [
    { kind: 'ble_identifier', value: WATCH },
    { kind: 'ble_fingerprint', value: FINGERPRINT },
  ],
  linked: null,
  ...over,
});

const view = (over: Partial<DeviceLinksView> = {}): DeviceLinksView => {
  const devices = over.devices ?? [];
  return {
    devices,
    links: [],
    total: devices.length,
    linked: devices.filter(d => d.linked).length,
    unknown: devices.filter(d => !d.linked).length,
    hours: 24,
    ...over,
  };
};

const serve = (reply: DeviceLinksView) => server.use(http.get('/device-links', () => HttpResponse.json(reply)));

// Whose a device is comes from the people known by name.
const PEOPLE = http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }, { name: 'Sam' }] }));

describe('DeviceLinks', () => {
  beforeEach(() => server.use(PEOPLE));

  it('lists each device with its sensor, where, signal, and when, and counts them', async () => {
    serve(
      view({
        devices: [
          device({ name: 'Dana’s watch', linked: { id: 'l1', target: 'person', name: 'Dana' } }),
          device({
            source: 'network.device_observation',
            identifier: 'aa:bb:cc:dd:ee:01',
            place: 'Hall AP',
            rssi: null,
          }),
        ],
      }),
    );
    renderPage(<DeviceLinks control />);
    expect(await screen.findByText('2 devices seen in the last day: 1 linked, 1 unknown.')).toBeInTheDocument();
    expect(screen.getByTitle('Linked to Dana, a person')).toBeInTheDocument();
    expect(screen.getByText('Bluetooth · Hall · -48 dBm · 2 min ago')).toBeInTheDocument();
    expect(screen.getByText('Network · Hall AP · 2 min ago')).toBeInTheDocument();
  });

  it('names a Bluetooth card by what it is and who made it, and keeps its raw ids in its details', async () => {
    serve(
      view({
        devices: [
          device({ identifier: '66:00:00:00:00:95', place: 'AABBCCDDEE01', vendor: 'Apple', kind: 'AirPods or Beats' }),
          device({ identifier: 'E8:6E:82:00:00:6A', place: 'AABBCCDDEE01', name: 'Govee_H609D_006A', vendor: 'Govee' }),
          device({ identifier: 'DE:AD:BE:EF:00:2B', place: 'AABBCCDDEE01', vendor: 'Sonos' }),
        ],
      }),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    const list = await screen.findByRole('list', { name: 'Devices' });
    const [pods, govee, sonos] = within(list).getAllByRole('listitem');
    if (!pods || !govee || !sonos) throw new Error('three cards');
    expect(within(pods).getByText('AirPods or Beats')).toBeInTheDocument();
    expect(within(pods).getByText('Apple')).toBeInTheDocument();
    expect(within(govee).getByText('Govee')).toBeInTheDocument();
    expect(within(sonos).getByText('Sonos device')).toBeInTheDocument();
    // The sensor's chip id and the address are off the card face: only in its details.
    expect(within(pods).getByText('Bluetooth · -48 dBm · 2 min ago')).toBeInTheDocument();
    expect(list).not.toHaveTextContent('AABBCCDDEE01');
    expect(list).not.toHaveTextContent('66:00:00:00:00:95');
    await user.click(within(pods).getByRole('button', { name: 'Details for AirPods or Beats' }));
    const details = await screen.findByRole('dialog');
    expect(details).toHaveTextContent('66:00:00:00:00:95');
    expect(details).toHaveTextContent('AABBCCDDEE01');
  });

  it('links a device to a room from the picker, shows where it lives, and removes it in one click', async () => {
    let sent: unknown = null;
    let removed: unknown = null;
    const home = { id: 'p1', name: 'Home', rooms: ['Living room', 'Kitchen'] };
    const located = { id: 'loc1', place_id: 'p1', place: 'Home', room: 'Living room' };
    let reply = view({ devices: [device({ name: 'Bravia' })], places: [home] });
    server.use(
      http.get('/device-links', () => HttpResponse.json(reply)),
      http.post('/device-locations', async ({ request }) => {
        sent = await request.json();
        reply = view({ devices: [device({ name: 'Bravia', location: located })], places: [home] });
        return HttpResponse.json({ ok: true, location: located });
      }),
      http.post('/device-locations/remove', async ({ request }) => {
        removed = await request.json();
        reply = view({ devices: [device({ name: 'Bravia' })], places: [home] });
        return HttpResponse.json({ ok: true });
      }),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    const link = await screen.findByRole('button', { name: 'Link Bravia' });
    expect(link).toHaveAttribute('aria-keyshortcuts', 'L');
    await user.click(link);
    const form = await screen.findByRole('form', { name: 'Link Bravia' });
    await user.click(within(form).getByText('A place'));
    await user.selectOptions(within(form).getByLabelText('Where it lives'), 'Living room, Home');
    await user.click(within(form).getByRole('button', { name: 'Link' }));
    expect(sent).toEqual({ kind: 'ble_identifier', value: WATCH, place: 'p1', room: 'Living room' });
    const card = (await screen.findByText('Living room')).closest('li');
    if (!card) throw new Error('a card');
    await user.click(within(card).getByRole('button', { name: 'Remove where Bravia lives' }));
    await waitFor(() => expect(screen.queryByText('Living room')).not.toBeInTheDocument());
    expect(removed).toEqual({ id: 'loc1' });
  });

  it('opens the link picker with L on a focused card, and offers the whole place too', async () => {
    serve(view({ devices: [device({ name: 'Bravia' })], places: [{ id: 'p1', name: 'Home', rooms: ['Kitchen'] }] }));
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    await screen.findByText('Bravia');
    const card = screen.getByRole('listitem', { name: 'Bravia' });
    card.focus();
    await user.keyboard('l');
    const form = await screen.findByRole('form', { name: 'Link Bravia' });
    await user.click(within(form).getByText('A place'));
    const where = within(form).getByLabelText('Where it lives');
    expect([...where.querySelectorAll('option')].map(o => o.textContent)).toEqual(['Home (anywhere)', 'Kitchen, Home']);
  });

  it('shows a linked person as a chip with a quick unlink', async () => {
    let removed: unknown = null;
    server.use(
      http.get('/device-links', () =>
        HttpResponse.json(
          view({ devices: [device({ name: 'Pixel', linked: { id: 'l1', target: 'person', name: 'Dana' } })] }),
        ),
      ),
      http.post('/device-links/remove', async ({ request }) => {
        removed = await request.json();
        return HttpResponse.json({ ok: true });
      }),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    const chip = await screen.findByTitle('Linked to Dana, a person');
    await user.click(within(chip).getByRole('button', { name: 'Unlink Pixel' }));
    await waitFor(() => expect(removed).toEqual({ id: 'l1' }));
  });

  it('links an unknown device to a person by the selector picked', async () => {
    let sent: unknown = null;
    let reply = view({ devices: [device()] });
    server.use(
      http.get('/device-links', () => HttpResponse.json(reply)),
      http.post('/device-links', async ({ request }) => {
        sent = await request.json();
        reply = view({ devices: [device({ linked: { id: 'l1', target: 'person', name: 'Dana' } })] });
        return HttpResponse.json({ ok: true, link: { id: 'l1', kind: 'ble_fingerprint', value: FINGERPRINT } });
      }),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    await user.click(await screen.findByRole('button', { name: 'Link Unknown device' }));
    const form = screen.getByRole('form', { name: 'Link Unknown device' });
    await user.type(await within(form).findByRole('combobox', { name: 'Whose is it?' }), ' Dana ');
    await user.selectOptions(within(form).getByLabelText('Recognize it by'), '1');
    await user.click(within(form).getByRole('button', { name: 'Link' }));
    expect(await screen.findByTitle('Linked to Dana, a person')).toBeInTheDocument();
    expect(sent).toEqual({ kind: 'ble_fingerprint', value: FINGERPRINT, target: 'person', name: 'Dana' });
    expect(screen.queryByRole('form')).not.toBeInTheDocument();
  });

  it('offers the known people as you type and links the one picked', async () => {
    let sent: unknown = null;
    server.use(
      http.get('/device-links', () => HttpResponse.json(view({ devices: [device()] }))),
      http.post('/device-links', async ({ request }) => {
        sent = await request.json();
        return HttpResponse.json({ ok: true, link: { id: 'l1', kind: 'ble_identifier', value: WATCH } });
      }),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    await user.click(await screen.findByRole('button', { name: 'Link Unknown device' }));
    const form = screen.getByRole('form', { name: 'Link Unknown device' });
    await user.type(await within(form).findByRole('combobox', { name: 'Whose is it?' }), 'sa');
    expect(screen.queryByRole('option', { name: 'Dana' })).not.toBeInTheDocument();
    await user.click(await screen.findByRole('option', { name: 'Sam' }));
    expect(within(form).getByRole('combobox', { name: 'Whose is it?' })).toHaveValue('Sam');
    await user.click(within(form).getByRole('button', { name: 'Link' }));
    await waitFor(() => expect(sent).toEqual({ kind: 'ble_identifier', value: WATCH, target: 'person', name: 'Sam' }));
  });

  it('links a device only to someone known: a new name offers no Add and leaves Link off', async () => {
    let posted = false;
    server.use(
      http.get('/device-links', () => HttpResponse.json(view({ devices: [device()] }))),
      http.post('/device-links', () => {
        posted = true;
        return HttpResponse.json({ ok: true });
      }),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    await user.click(await screen.findByRole('button', { name: 'Link Unknown device' }));
    const form = screen.getByRole('form', { name: 'Link Unknown device' });
    const box = await within(form).findByRole('combobox', { name: 'Whose is it?' });
    await user.type(box, 'Danna{Enter}');
    expect(screen.queryByRole('option', { name: /Add/ })).not.toBeInTheDocument();
    expect(within(form).getByRole('button', { name: 'Link' })).toBeDisabled();
    // A known name typed in another case is that person, sent as they're known.
    await user.clear(box);
    await user.type(box, 'dana');
    expect(within(form).getByRole('button', { name: 'Link' })).toBeEnabled();
    expect(posted).toBe(false);
  });

  it('says so when nobody is known by name yet', async () => {
    server.use(
      http.get('/device-links', () => HttpResponse.json(view({ devices: [device()] }))),
      http.get('/people', () => HttpResponse.json({ people: [] })),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    await user.click(await screen.findByRole('button', { name: 'Link Unknown device' }));
    const form = screen.getByRole('form', { name: 'Link Unknown device' });
    expect(await within(form).findByText(/Nobody is known by name yet/)).toBeInTheDocument();
    expect(within(form).getByRole('button', { name: 'Link' })).toBeDisabled();
  });

  it('names a device as a thing, and says why when the server refuses', async () => {
    server.use(
      http.get('/device-links', () => HttpResponse.json(view({ devices: [device()] }))),
      http.post('/device-links', () =>
        HttpResponse.json({ ok: false, error: 'devices can only be linked on this machine' }, { status: 403 }),
      ),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    await user.click(await screen.findByRole('button', { name: 'Link Unknown device' }));
    const form = screen.getByRole('form', { name: 'Link Unknown device' });
    await user.click(within(form).getByText('A thing'));
    await user.type(within(form).getByLabelText('What is it?'), 'AirPods');
    await user.click(within(form).getByRole('button', { name: 'Link' }));
    expect(await within(form).findByRole('alert')).toHaveTextContent('devices can only be linked on this machine');
    expect(within(form).getByLabelText('What is it?')).toHaveValue('AirPods');
  });

  it('finds a named network phone beyond the initial limit and explicitly links its network address', async () => {
    const phone = device({
      source: 'network.device_observation',
      identifier: '02:00:00:00:00:31',
      name: 'Sample phone',
      rssi: null,
      selectors: [{ kind: 'net_mac', value: '02:00:00:00:00:31' }],
    });
    const earlier = Array.from({ length: 30 }, (_, i) =>
      device({ identifier: `02:00:00:00:01:${String(i).padStart(2, '0')}` }),
    );
    let reply = view({ devices: [...earlier, phone] });
    let sent: unknown = null;
    server.use(
      http.get('/device-links', () => HttpResponse.json(reply)),
      http.post('/device-links', async ({ request }) => {
        sent = await request.json();
        reply = view({
          devices: [...earlier, { ...phone, linked: { id: 'phone-link', target: 'person', name: 'Dana' } }],
        });
        return HttpResponse.json({
          ok: true,
          link: { id: 'phone-link', kind: 'net_mac', value: phone.identifier, target: 'person', name: 'Dana' },
        });
      }),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    expect(await screen.findByText('31 devices seen in the last day: 0 linked, 31 unknown.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Link Sample phone' })).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText('Device source'), 'network');
    expect(screen.getByRole('button', { name: 'Link Sample phone' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Show all/ })).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText('Device source'), 'all');
    await user.type(screen.getByLabelText('Search devices'), 'PHONE');
    await user.click(screen.getByRole('button', { name: 'Link Sample phone' }));
    const form = screen.getByRole('form', { name: 'Link Sample phone' });
    await user.type(await within(form).findByRole('combobox', { name: 'Whose is it?' }), ' Dana ');
    await user.click(within(form).getByRole('button', { name: 'Link' }));
    expect(await screen.findByTitle('Linked to Dana, a person')).toBeInTheDocument();
    expect(sent).toEqual({ kind: 'net_mac', value: phone.identifier, target: 'person', name: 'Dana' });
    expect(screen.getByText('31 devices seen in the last day: 1 linked, 30 unknown.')).toBeInTheDocument();
  });

  it('keeps the network-link draft and explicit person choice after a failed save', async () => {
    serve(
      view({
        devices: [
          device({
            source: 'network.device_observation',
            name: 'Sample phone',
            selectors: [{ kind: 'net_mac', value: WATCH }],
          }),
        ],
      }),
    );
    server.use(http.post('/device-links', () => HttpResponse.json({ error: 'Store unavailable' }, { status: 503 })));
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    await user.click(await screen.findByRole('button', { name: 'Link Sample phone' }));
    const form = screen.getByRole('form', { name: 'Link Sample phone' });
    await user.type(await within(form).findByRole('combobox', { name: 'Whose is it?' }), 'Dana');
    await user.click(within(form).getByRole('button', { name: 'Link' }));
    expect(await within(form).findByRole('alert')).toHaveTextContent('Store unavailable');
    expect(await within(form).findByRole('combobox', { name: 'Whose is it?' })).toHaveValue('Dana');
    expect(within(form).getByRole('button', { name: 'Link' })).toBeEnabled();
  });

  it('filters seen and unseen links without changing inventory totals or inventing unseen duplicates', async () => {
    serve(
      view({
        devices: [
          device({
            name: 'Desk phone',
            source: 'network.device_observation',
            linked: { id: 'seen', target: 'person', name: 'Dana' },
          }),
          device({ name: 'Yard tag', identifier: 'tag' }),
        ],
        links: [
          { id: 'seen', kind: 'net_mac', value: WATCH, target: 'person', name: 'Dana' },
          { id: 'unseen', kind: 'net_mac', value: '02:00:00:00:00:09', target: 'thing', name: 'Spare phone' },
        ],
      }),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    await screen.findByText('Desk phone');
    await user.type(screen.getByLabelText('Search devices'), 'Yard');
    expect(screen.queryByText(/Not seen in the last day/)).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText('Device source'), 'network');
    expect(screen.getByRole('status')).toHaveTextContent('No devices or saved links match these filters.');
    expect(screen.getByText('2 devices seen in the last day: 1 linked, 1 unknown.')).toBeInTheDocument();
    await user.clear(screen.getByLabelText('Search devices'));
    await user.type(screen.getByLabelText('Search devices'), 'Spare');
    expect(screen.getByTitle('Linked to Spare phone, a thing')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Unlink Spare phone' })).toBeInTheDocument();
  });

  it('renders an unrecognized inherited-property source as text', async () => {
    serve(view({ devices: [device({ source: '__proto__' })] }));
    renderPage(<DeviceLinks control />);
    expect(await screen.findByText('__proto__ · Hall · -48 dBm · 2 min ago')).toBeInTheDocument();
  });

  it('unlinks a link whose device was not seen in the window', async () => {
    let removed: unknown = null;
    let reply = view({ links: [{ id: 'l9', kind: 'ble_name', value: 'Keys', target: 'thing', name: 'Keys' }] });
    server.use(
      http.get('/device-links', () => HttpResponse.json(reply)),
      http.post('/device-links/remove', async ({ request }) => {
        removed = await request.json();
        reply = view();
        return HttpResponse.json({ ok: true });
      }),
    );
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    expect(await screen.findByText(/Not seen in the last day/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Unlink Keys' }));
    await waitFor(() => expect(screen.queryByTitle('Linked to Keys, a thing')).not.toBeInTheDocument());
    expect(removed).toEqual({ id: 'l9' });
  });

  it('shows the newest unknown devices first and the rest on request', async () => {
    const many = Array.from({ length: UNKNOWN_SHOWN + 5 }, (_, i) =>
      device({ identifier: `de:ad:be:ef:00:${String(i).padStart(2, '0')}` }),
    );
    serve(view({ devices: many }));
    const user = userEvent.setup();
    renderPage(<DeviceLinks control />);
    await user.click(await screen.findByRole('button', { name: `Show all ${UNKNOWN_SHOWN + 5} unknown devices` }));
    expect(screen.getAllByRole('button', { name: /^Link / })).toHaveLength(UNKNOWN_SHOWN + 5);
  });

  it("puts what came and went today first, and keeps what's on all day behind Show all", async () => {
    const net = (identifier: string, name: string) =>
      device({
        source: 'network.device_observation',
        identifier,
        name,
        selectors: [{ kind: 'net_mac', value: identifier }],
      });
    serve(view({ devices: [net('aa:00:00:00:00:01', 'plug'), net('aa:00:00:00:00:02', 'laptop')] }));
    const user = userEvent.setup();
    renderPage(<DeviceLinks control moving={new Set(['AA:00:00:00:00:02'])} steady={new Set(['AA:00:00:00:00:01'])} />);
    expect(await screen.findByRole('button', { name: 'Link laptop' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Link plug' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Show all 2 unknown devices' }));
    expect(screen.getAllByRole('button', { name: /^Link / }).map(b => b.textContent)).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Link plug' })).toBeInTheDocument();
  });

  it('offers no link or unlink away from this machine', async () => {
    serve(
      view({ devices: [device(), device({ identifier: 'x', linked: { id: 'l1', target: 'thing', name: 'Tag' } })] }),
    );
    renderPage(<DeviceLinks control={false} />);
    expect(await screen.findByTitle('Linked to Tag, a thing')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Link|Unlink/ })).not.toBeInTheDocument();
  });

  it('marks cached results as previous evidence when refreshing the inventory fails', async () => {
    serve(view({ devices: [device({ name: 'Sample phone' })] }));
    const { client } = renderPage(<DeviceLinks control />);
    await screen.findByText('Sample phone');
    server.use(http.get('/device-links', () => HttpResponse.json({ error: 'Store unavailable' }, { status: 503 })));
    await act(() => client.invalidateQueries({ queryKey: DEVICE_LINKS_KEY }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Showing the last list');
    expect(screen.getByText('Sample phone')).toBeInTheDocument();
    expect(screen.queryByText(/No devices seen/)).not.toBeInTheDocument();
  });

  it('says so when the devices cannot be read', async () => {
    server.use(
      http.get('/device-links', () =>
        HttpResponse.json({ error: 'Couldn’t reach the recordings store; try again in a minute.' }, { status: 503 }),
      ),
    );
    renderPage(<DeviceLinks control />);
    expect(await screen.findByText("Couldn't list devices")).toBeInTheDocument();
    expect(screen.queryByText(/No devices seen|0 devices/)).not.toBeInTheDocument();
  });
});
