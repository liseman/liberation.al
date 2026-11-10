import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { hm } from '../lib/format';
import { allDay, splitDay, staysText, Whereabouts, WhoIsHome, WifiToday } from './DeviceSections';
import type { Comings, WifiDevice } from './devicesApi';

// Whose a device is comes from the people known by name.
beforeEach(() => server.use(http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }] }))));

const MIDNIGHT = 1_800_000_000;
const NOW = MIDNIGHT + 12 * 3600;
const GAP = 600;

const device = (mac: string, name: string | null, stays: [number, number][]): WifiDevice => ({
  mac,
  name,
  place: 'Hall',
  linked: null,
  stays: stays.map(([start, end]) => ({ start: MIDNIGHT + start, end: MIDNIGHT + end })),
});

const today = (devices: WifiDevice[]): Comings => ({ at: NOW, start: MIDNIGHT, gap_seconds: GAP, rows: 9, devices });

it('says a stay that runs to now as "to now", and a short one as one time', () => {
  const stays = [
    { start: MIDNIGHT + 3600, end: MIDNIGHT + 7200 },
    { start: MIDNIGHT + 9000, end: MIDNIGHT + 9030 },
    { start: MIDNIGHT + 10_000, end: NOW - 60 },
  ];
  expect(staysText(stays, NOW, GAP)).toBe(
    `${hm(MIDNIGHT + 3600)} to ${hm(MIDNIGHT + 7200)}, ${hm(MIDNIGHT + 9000)}, ${hm(MIDNIGHT + 10_000)} to now`,
  );
  expect(staysText([{ start: MIDNIGHT, end: NOW - GAP - 1 }], NOW, GAP)).toBe(
    `${hm(MIDNIGHT)} to ${hm(NOW - GAP - 1)}`,
  );
});

it('lists what came and went, newest arrival first, and names what was on all day', () => {
  const plug = device('AA:00:00:00:00:01', 'plug', [[10, 12 * 3600 - 5]]);
  const morning = device('AA:00:00:00:00:02', 'iPhone', [[3600, 7200]]);
  const back = device('AA:00:00:00:00:03', 'iPhone', [
    [10, 3600],
    [9 * 3600, 12 * 3600 - 5],
  ]);
  const day = today([plug, morning, back]);
  expect(allDay(plug, day)).toBe(true);
  expect(allDay(back, day)).toBe(false); // left and came back
  const { moving, steady } = splitDay(day);
  expect(moving.map(d => d.mac)).toEqual([back.mac, morning.mac]);
  expect(steady).toEqual([plug]);
});

it('tells two devices of the same name apart, counting the link listing too, and asks whose each is', async () => {
  let sent: unknown = null;
  const phone = device('AA:00:00:00:00:02', 'iPhone', [[3600, 7200]]);
  server.use(
    http.get('/device-comings', () => HttpResponse.json(today([phone]))),
    // The only other iPhone was seen over Bluetooth, not on the Wi-Fi: the label still tells them apart.
    http.get('/device-links', () =>
      HttpResponse.json({
        devices: [
          {
            source: 'bluetooth.sighting',
            identifier: 'c0:ff:ee:00:00:09',
            name: 'iPhone',
            place: 'Hall',
            rssi: -50,
            last_seen: NOW,
            sightings: 1,
            selectors: [],
            linked: null,
          },
        ],
        links: [],
        total: 1,
        linked: 0,
        unknown: 1,
        hours: 24,
      }),
    ),
    http.post('/device-links', async ({ request }) => {
      sent = await request.json();
      return HttpResponse.json({ ok: true, link: { id: 'l1', kind: 'net_mac', value: phone.mac } });
    }),
  );
  const user = userEvent.setup();
  renderPage(<WifiToday control />);
  expect(await screen.findByText('iPhone (00:02)')).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Whose is iPhone (00:02)?' }));
  const form = screen.getByRole('form', { name: 'Link iPhone (00:02)' });
  await user.type(await within(form).findByRole('combobox', { name: 'Whose is it?' }), 'Dana');
  await user.click(within(form).getByRole('button', { name: 'Link' }));
  await waitFor(() =>
    expect(sent).toMatchObject({ kind: 'net_mac', value: phone.mac, target: 'person', name: 'Dana' }),
  );
});

it('asks no one whose a device is away from this machine', async () => {
  server.use(
    http.get('/device-comings', () =>
      HttpResponse.json(today([device('AA:00:00:00:00:02', 'iPhone', [[3600, 7200]])])),
    ),
  );
  renderPage(<WifiToday />);
  expect(await screen.findByText('iPhone')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /^Whose is/ })).not.toBeInTheDocument();
});

it('leads as the starter, asking whose phones these are', async () => {
  server.use(
    http.get('/device-comings', () =>
      HttpResponse.json(today([device('AA:00:00:00:00:02', 'iPhone', [[3600, 7200]])])),
    ),
  );
  renderPage(<WifiToday starter control />);
  expect(await screen.findByRole('heading', { name: 'Whose phones are these?' })).toBeInTheDocument();
  expect(await screen.findByRole('button', { name: 'Whose is iPhone?' })).toBeInTheDocument();
});

it("shows today's Wi-Fi comings and goings, and the all-day devices by name", async () => {
  const devices = [
    device('AA:00:00:00:00:03', 'iPhone', [[9 * 3600, 12 * 3600 - 5]]),
    ...Array.from({ length: 14 }, (_, n) =>
      device(`BB:00:00:00:00:${String(n).padStart(2, '0')}`, `plug-${n}`, [[10, 12 * 3600 - 5]]),
    ),
  ];
  server.use(http.get('/device-comings', () => HttpResponse.json(today(devices))));
  renderPage(<WifiToday />);
  const list = await screen.findByRole('list', { name: 'Came and went today' });
  expect(within(list).getByText('iPhone')).toBeInTheDocument();
  expect(
    within(list).getByRole('img', { name: `iPhone on the Wi-Fi: ${hm(MIDNIGHT + 9 * 3600)} to now` }),
  ).toBeInTheDocument();
  expect(screen.getByText('14 devices on all day')).toBeInTheDocument();
  expect(screen.queryByText(/plug-13/)).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'and 2 more' }));
  expect(screen.getByText(/plug-13/)).toBeInTheDocument();
});

it("says where each linked thing was last seen, leaving people to Who's home", async () => {
  server.use(
    http.get('/device-whereabouts', () =>
      HttpResponse.json({
        at: NOW,
        days: 30,
        caveat: '',
        links: [
          {
            target: 'thing',
            name: 'AirPods',
            links: 1,
            last_seen: NOW - 7200,
            place: 'Car',
            rssi: -70,
            source: 'bluetooth.sighting',
          },
          { target: 'thing', name: 'Keys', links: 1, last_seen: null, place: null, rssi: null, source: null },
          { target: 'person', name: 'Dana', links: 1, last_seen: NOW - 60, place: 'Hall', rssi: null, source: null },
        ],
      }),
    ),
  );
  renderPage(<Whereabouts />);
  const list = await screen.findByRole('list', { name: 'Your things' });
  expect(within(list).getByText(`Seen 2.0 h ago by Car, ${hm(NOW - 7200)}`)).toBeInTheDocument();
  expect(within(list).getByText('Not seen in 30 days')).toBeInTheDocument();
  expect(within(list).queryByText('Dana')).not.toBeInTheDocument();
});

it('points to linking when nothing is linked yet', async () => {
  server.use(http.get('/device-whereabouts', () => HttpResponse.json({ at: NOW, days: 30, caveat: '', links: [] })));
  renderPage(<Whereabouts />);
  expect(await screen.findByText(/^No things linked yet\./)).toBeInTheDocument();
});

it("says who's home by their linked devices", async () => {
  server.use(
    http.get('/device-presence', () =>
      HttpResponse.json({
        at: NOW,
        start: NOW - 600,
        gap_seconds: 600,
        caveat: 'A linked device was seen; that is not proof its owner is here.',
        people: [
          {
            name: 'Dana',
            status: 'recent',
            last_seen: NOW - 120,
            source: 'network.device_observation',
            place: 'Hall',
            links: 1,
          },
          { name: 'Sam', status: 'not_seen', last_seen: null, source: null, place: null, links: 1 },
          { name: 'Kara', status: 'unknown', last_seen: null, source: null, place: null, links: 2 },
        ],
      }),
    ),
  );
  renderPage(<WhoIsHome />);
  const list = await screen.findByRole('list', { name: "Who's home" });
  const row = (name: string) => within(list).getByText(name).closest('li') as HTMLElement;
  expect(within(row('Dana')).getByText('Here')).toBeInTheDocument();
  expect(within(row('Dana')).getByText(`Device seen 2 min ago by Hall, ${hm(NOW - 120)}`)).toBeInTheDocument();
  expect(within(row('Sam')).getByText('Not seen recently')).toBeInTheDocument();
  expect(within(row('Kara')).getByText("Can't tell")).toBeInTheDocument();
});

it("tells you to link a phone when no one's is linked", async () => {
  renderPage(<WhoIsHome />);
  expect(await screen.findByText(/^No phones or watches are linked to people yet\./)).toBeInTheDocument();
  expect(screen.getByText(/link it to them as a person/)).toBeInTheDocument();
});

it('opens the all-day devices as rows, each with a Link, and links one in place', async () => {
  let sent: unknown = null;
  const plug = device('BB:00:00:00:00:01', 'plug', [[10, 12 * 3600 - 5]]);
  server.use(
    http.get('/device-comings', () => HttpResponse.json(today([plug]))),
    http.post('/device-links', async ({ request }) => {
      sent = await request.json();
      return HttpResponse.json({ ok: true, link: { id: 'l1', kind: 'net_mac', value: plug.mac } });
    }),
  );
  const user = userEvent.setup();
  renderPage(<WifiToday control />);
  await user.click(await screen.findByRole('button', { name: 'Link one' }));
  const list = screen.getByRole('list', { name: 'On all day' });
  await user.click(within(list).getByRole('button', { name: 'Link plug' }));
  const form = screen.getByRole('form', { name: 'Link plug' });
  await user.click(within(form).getByText('A thing'));
  await user.type(within(form).getByLabelText('What is it?'), 'Lamp');
  await user.click(within(form).getByRole('button', { name: 'Link' }));
  await waitFor(() => expect(sent).toMatchObject({ kind: 'net_mac', value: plug.mac, target: 'thing', name: 'Lamp' }));
});

it('says who a Wi-Fi device is linked to, and unlinks it', async () => {
  let removed: unknown = null;
  const phone = {
    ...device('AA:00:00:00:00:02', 'iPhone', [[3600, 7200]]),
    linked: { id: 'l7', target: 'person' as const, name: 'Dana' },
  };
  server.use(
    http.get('/device-comings', () => HttpResponse.json(today([phone]))),
    http.post('/device-links/remove', async ({ request }) => {
      removed = await request.json();
      return HttpResponse.json({ ok: true });
    }),
  );
  const user = userEvent.setup();
  renderPage(<WifiToday control />);
  expect(await screen.findByTitle('Linked to Dana, a person')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /^Whose is/ })).not.toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Unlink iPhone' }));
  await waitFor(() => expect(removed).toEqual({ id: 'l7' }));
});

it('names a Wi-Fi device that gave no hostname by its maker, and says who made a named one', async () => {
  const devices = [
    { ...device('F0:9F:C2:00:00:01', null, [[9 * 3600, 9 * 3600 + 60]]), vendor: 'Ubiquiti' },
    { ...device('F0:9F:C2:00:00:02', 'kauf-plug', [[8 * 3600, 8 * 3600 + 60]]), vendor: 'Espressif' },
  ];
  server.use(http.get('/device-comings', () => HttpResponse.json(today(devices))));
  renderPage(<WifiToday />);
  const list = await screen.findByRole('list', { name: 'Came and went today' });
  const [ubiquiti, plug] = within(list).getAllByRole('listitem');
  if (!ubiquiti || !plug) throw new Error('two cards');
  expect(within(ubiquiti).getByText('Ubiquiti device')).toBeInTheDocument();
  expect(ubiquiti).not.toHaveTextContent('F0:9F:C2:00:00:01');
  expect(within(plug).getByText('kauf-plug')).toBeInTheDocument();
  expect(within(plug).getByText('Espressif')).toBeInTheDocument();
});
