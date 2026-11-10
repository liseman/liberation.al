import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { CountDay, Devices, PersonActivity, Walkbys } from './api';
import { DeviceReport, PersonActivityReport, WalkbyReport } from './HourlyReports';

const DAY = '2026-10-20';
const T = new Date(`${DAY}T09:23:00`).getTime() / 1000;
const progress = { first: DAY, last: DAY, pending: [], refreshing: [] };
const days: CountDay[] = [
  {
    day: DAY,
    n: 3,
    t: T,
    hours: Array.from({ length: 24 }, (_, h) => (h === 9 ? { n: 3, t: T } : h === 10 ? { n: 0, t: null } : null)),
  },
];

it('walk-by hours preserve camera coverage and link actual crossings rather than the hour boundary', async () => {
  const data: Walkbys = {
    ...progress,
    report: 'walkbys',
    covered_days: 1,
    cameras: [{ key: 'cam-a', name: 'Driveway', total: 3, days }],
  };
  server.use(
    http.get('/reports/api/report', ({ request }) => {
      expect(new URL(request.url).searchParams.get('arg')).toBe('cam-a');
      return HttpResponse.json(data);
    }),
  );
  renderPage(<WalkbyReport arg="cam-a" />);
  const chart = await screen.findByRole('figure', { name: 'Walk-bys, Driveway, by day and hour' });
  const links = within(chart).getAllByRole('link');
  expect(links).toHaveLength(1);
  expect(links[0]).toHaveAccessibleName(/9 am: 3 walk-bys/);
  expect(links[0]).toHaveAttribute('href', `/review#day=${DAY}&t=${T}`);
  expect(await screen.findByText('3 walk-bys on 1 checked day.')).toBeInTheDocument();
});

it('shows one row per activity with a human duration, one caveat, and the day grid behind By day', async () => {
  const data: PersonActivity = {
    ...progress,
    report: 'person_activity',
    covered_days: 1,
    people: [
      {
        id: 'p-alice',
        name: 'Alice',
        series: [
          {
            key: 'opening_package',
            name: 'Opening a package',
            total_min: 2.5,
            verified: false,
            days: [
              {
                day: DAY,
                minutes: 2.5,
                t: T,
                hours: Array.from({ length: 24 }, (_, h) => (h === 9 ? { minutes: 2.5, t: T } : null)),
              },
            ],
          },
        ],
      },
    ],
  };
  server.use(http.get('/reports/api/report', () => HttpResponse.json(data)));
  renderPage(<PersonActivityReport arg="p-alice" />);
  expect(await screen.findByText('Spans the model found. Not confirmed yet: Opening a package.')).toBeInTheDocument();
  const table = screen.getByRole('figure', { name: "Alice's activities by hour of day" });
  expect(within(table).getByText('Opening a package · 3 min')).toBeInTheDocument();
  expect(within(table).getByRole('link', { name: /Opening a package, 9 am: 3 min\. Open in Review/ })).toHaveAttribute(
    'href',
    `/review#day=${DAY}&t=${T}`,
  );
  expect(screen.queryByRole('heading', { name: 'Alice' })).not.toBeInTheDocument(); // the card already names her
  expect(screen.getAllByRole('figure')).toHaveLength(1);
  await userEvent.click(screen.getByRole('button', { name: 'By day' }));
  expect(screen.getByRole('figure', { name: 'Alice, Opening a package, by day and hour' })).toBeInTheDocument();
  expect(screen.getByText('Unconfirmed')).toBeInTheDocument();
});

it('counts device identifiers without asserting people or turning unknown hours into linked zeros', async () => {
  const data: Devices = {
    ...progress,
    report: 'devices',
    covered_days: 1,
    caveat: 'Identifiers may rotate; totals are device-hours.',
    sources: [{ key: 'bluetooth', name: 'Bluetooth', total: 3, days }],
  };
  server.use(http.get('/reports/api/report', () => HttpResponse.json(data)));
  renderPage(<DeviceReport arg="bluetooth" />);
  expect(await screen.findByText(data.caveat)).toBeInTheDocument();
  expect(screen.getAllByRole('link')).toHaveLength(1);
  expect(screen.getByRole('link')).toHaveAccessibleName(/3 device identifiers/);
});

it('opens Walk-bys by camera on its pinned camera, and shows every camera when asked, loading nothing again', async () => {
  const data: Walkbys = {
    ...progress,
    report: 'walkbys',
    covered_days: 1,
    cameras: [
      { key: 'cam-a', name: 'Driveway', total: 3, days },
      { key: 'cam-b', name: 'Patio', total: 1, days },
    ],
  };
  const reads: (string | null)[] = [];
  server.use(
    http.get('/reports/api/report', ({ request }) => {
      reads.push(new URL(request.url).searchParams.get('arg'));
      return HttpResponse.json(data);
    }),
  );
  renderPage(<WalkbyReport arg={null} pinned={['cam-b']} onPinned={() => {}} />);
  expect(await screen.findByText('1 walk-by on 1 checked day.')).toBeInTheDocument(); // Patio, pinned
  const show = screen.getByRole('radiogroup', { name: 'Show' });
  expect(within(show).getByRole('radio', { name: 'Patio' })).toBeChecked();
  expect(screen.queryByRole('figure', { name: /Driveway/ })).not.toBeInTheDocument();
  expect(screen.getByRole('combobox', { name: 'Another camera' })).toBeInTheDocument();
  await userEvent.click(within(show).getByText('Every camera'));
  expect(await screen.findByText('4 walk-bys on 1 checked day.')).toBeInTheDocument();
  expect(screen.getByRole('figure', { name: /Driveway/ })).toBeInTheDocument();
  expect(reads).toEqual([null]);
});

it('opens Activity by person on its pinned person, and shows everyone when asked', async () => {
  const series = (key: string) => ({ key, name: key, total_min: 1, verified: true, days: [] });
  const data: PersonActivity = {
    ...progress,
    report: 'person_activity',
    covered_days: 1,
    people: [
      { id: 'p-alice', name: 'Alice', series: [series('read')] },
      { id: 'p-bob', name: 'Bob', series: [series('eat')] },
    ],
  };
  server.use(http.get('/reports/api/report', () => HttpResponse.json(data)));
  renderPage(<PersonActivityReport arg={null} pinned={['p-bob']} onPinned={() => {}} />);
  expect(await screen.findByRole('figure', { name: "Bob's activities by hour of day" })).toBeInTheDocument();
  expect(screen.queryByRole('figure', { name: "Alice's activities by hour of day" })).not.toBeInTheDocument();
  await userEvent.click(within(screen.getByRole('radiogroup', { name: 'Show' })).getByText('Everyone'));
  expect(await screen.findByRole('figure', { name: "Alice's activities by hour of day" })).toBeInTheDocument();
  expect(screen.getByRole('figure', { name: "Bob's activities by hour of day" })).toBeInTheDocument();
});
