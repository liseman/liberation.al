import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { finding } from './Activity';
import type { Activity, ChosenReport, Conversations, Health, Suggestion, WhosAround } from './api';
import { dayTicks, step } from './charts';
import StatisticsTab, { pairsOf } from './StatisticsTab';

const DAY = '2026-10-20';
const T10 = new Date(`${DAY}T10:00:00`).getTime() / 1000;
const hours = (fill: (h: number) => WhosAround['days'][number]['hours'][number]) =>
  Array.from({ length: 24 }, (_, h) => fill(h));

const around = (over: Partial<WhosAround> = {}): WhosAround => ({
  report: 'whos_around',
  first: DAY,
  last: DAY,
  pending: [],
  refreshing: [],
  days: [
    {
      day: DAY,
      hours: hours(h =>
        h < 8 ? null : h === 10 ? { t: T10, who: [['p-alice', 30]] } : { t: T10 + (h - 10) * 3600, who: [] },
      ),
    },
  ],
  people: [{ id: 'p-alice', name: 'Alice', hours: 0.5 }],
  ...over,
});

const talk: Conversations = {
  report: 'conversations',
  first: DAY,
  last: DAY,
  pending: [],
  refreshing: [],
  people: [
    {
      id: 'p-dana',
      name: 'Dana',
      lines: 4,
      speech_min: 1,
      by_hour: Array.from({ length: 24 }, (_, h) =>
        h === 19 ? { n: 4, day: DAY, t: T10 + 9 * 3600 } : { n: 0, day: null, t: null },
      ),
    },
  ],
  daily: [{ day: DAY, lines: 4, t: T10 }],
  words: [],
  pairs: [],
};

const built: ChosenReport[] = [
  { id: 'whos_around', kind: 'whos_around', arg: null, title: "Who's around" },
  { id: 'conversations', kind: 'conversations', arg: null, title: 'Conversations' },
];

const activity: Activity = {
  report: 'activity',
  first: DAY,
  last: DAY,
  pending: [],
  refreshing: [],
  events: [
    {
      key: 'event:pushups',
      name: 'pushups',
      total: 6,
      typical: 3,
      days: [
        { day: '2026-10-19', n: null, t: null },
        { day: DAY, n: 6, t: T10 },
      ],
      by_hour: Array.from({ length: 24 }, () => ({ n: 0, day: null, t: null })),
    },
  ],
  zones: [],
  pets: [],
  covered_days: { events: 1, zones: 0, pets: 0 },
};

const health: Health = {
  report: 'health',
  first: DAY,
  last: DAY,
  pending: [],
  refreshing: [],
  sampled_from: T10,
  sample_s: 300,
  hours: [{ t: T10, mem_pct: 40, gpu_mem_pct: null, gpu_util: null }],
  services: [
    { name: 'listen', uptime_pct: 90, samples: 10, up_now: false },
    { name: 'dashboard', uptime_pct: 100, samples: 10, up_now: true },
  ],
  storage: {
    at: T10,
    days: [{ day: DAY, gb: 30, tier2_gb: 2, plan_gb: 18.75 }],
    media_gb: 30,
    plan_gb: 18.75,
    catalog_gb: 4,
    vectors_gb: 0.1,
  },
};

const ideas: Suggestion[] = Array.from({ length: 7 }, (_, i) => ({
  id: `activity:zone:z${i}`,
  kind: 'activity',
  arg: `zone:z${i}`,
  title: `Visits to zone ${i}`,
  why: `${i + 1} visits`,
}));

const KINDS = {
  whos_around: "Who's around",
  conversations: 'Conversations',
  activity: 'Activity and events',
  health: 'System health',
  devices: 'Device observations',
  walkbys: 'Walk-bys by hour',
  person_activity: 'Person activity',
};

function serve(reports = built, whos: WhosAround = around()) {
  const saved: ChosenReport[][] = [];
  server.use(
    http.get('/reports/api/layout', () => HttpResponse.json({ reports, kinds: KINDS })),
    http.post('/reports/api/layout', async ({ request }) => {
      const body = (await request.json()) as { reports: ChosenReport[] };
      saved.push(body.reports);
      return HttpResponse.json({
        reports: body.reports,
        kinds: { whos_around: "Who's around", conversations: 'Conversations' },
      });
    }),
    http.get('/reports/api/report', ({ request }) => {
      const kind = new URL(request.url).searchParams.get('kind');
      return HttpResponse.json(
        kind === 'whos_around' ? whos : kind === 'activity' ? activity : kind === 'health' ? health : talk,
      );
    }),
    http.get('/reports/api/suggestions', () => HttpResponse.json({ suggestions: ideas })),
  );
  return saved;
}

describe('StatisticsTab', () => {
  it('shows the chosen reports in order, each with a sentence stating what it found', async () => {
    serve();
    renderPage(<StatisticsTab />);
    const sections = await screen.findAllByRole('region');
    expect(sections.map(s => s.getAttribute('aria-label'))).toEqual([
      "Who's around",
      'Conversations',
      'Suggested reports',
    ]);
    expect(await screen.findByText(/Alice was around most, 30 min/)).toBeInTheDocument();
    expect(
      await screen.findByText(/Most talk happens around 7 pm, and Dana says the most \(4 lines\)/),
    ).toBeInTheDocument();
  });

  it('jumps to a report from its chip', async () => {
    serve();
    renderPage(<StatisticsTab />);
    const chips = await screen.findByRole('navigation', { name: 'Jump to a report' });
    expect(
      within(chips)
        .getAllByRole('link')
        .map(a => a.textContent),
    ).toEqual(["Who's around", 'Conversations']);
    await userEvent.click(within(chips).getByRole('link', { name: 'Conversations' }));
    expect(screen.getByRole('region', { name: 'Conversations' })).toHaveFocus();
  });

  it('links an hour someone was in to Review at that hour, and leaves gaps and empty hours unlinked', async () => {
    serve([built[0] as ChosenReport]);
    renderPage(<StatisticsTab />);
    const link = await screen.findByRole('link', { name: /Tue Oct 20, 10 am: Alice 30 min\. Open in Review/ });
    expect(link).toHaveAttribute('href', `/review#day=${DAY}&t=${T10}`);
    const chart = screen.getByRole('figure', { name: /Who was around/ });
    expect(within(chart).getAllByRole('link')).toHaveLength(1); // not the gap hours, not the empty ones
  });

  it('removes a report and saves the new set on the server', async () => {
    const saved = serve();
    renderPage(<StatisticsTab />);
    expect(await screen.findByRole('region', { name: "Who's around" })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: "Remove Who's around" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Arrange' }));
    await userEvent.click(screen.getByRole('button', { name: "Remove Who's around" }));
    await waitFor(() => expect(saved).toEqual([[built[1]]]));
    expect(screen.queryByRole('region', { name: "Who's around" })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: "Add Who's around" })).toBeInTheDocument(); // it can come back
  });

  it('removes a report from its ⋯ menu without Arrange', async () => {
    const saved = serve();
    renderPage(<StatisticsTab />);
    await userEvent.click(await screen.findByRole('button', { name: 'More for Conversations' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Remove from this page' }));
    await waitFor(() => expect(saved).toEqual([[built[0]]]));
    expect(screen.queryByRole('region', { name: 'Conversations' })).not.toBeInTheDocument();
  });

  it('pairs two smaller reports next to each other in order, and leaves a smaller one alone full width', () => {
    const r = (id: string, kind: string): ChosenReport => ({ id, kind, arg: null, title: id });
    const page = [
      r('a', 'devices'),
      r('b', 'walkbys'),
      r('c', 'conversations'),
      r('d', 'person_activity'),
      r('e', 'devices'),
      r('f', 'walkbys'),
    ];
    expect([...pairsOf(page)].sort()).toEqual(['a', 'b', 'd', 'e']); // f has no smaller neighbour left to pair with
    expect([...pairsOf([r('x', 'devices'), r('y', 'conversations'), r('z', 'walkbys')])]).toEqual([]);
  });

  it('shows pinned people beside Everyone, and Unpin saves the pins without them', async () => {
    const people = [
      { id: 'p-alice', name: 'Alice', hours: 0.5 },
      { id: 'p-bob', name: 'Bob', hours: 0.2 },
    ];
    const pinned = { ...(built[0] as ChosenReport), people: ['p-alice'] };
    const saved = serve([pinned], around({ people }));
    renderPage(<StatisticsTab />);
    const show = await screen.findByRole('radiogroup', { name: 'Show' });
    expect(within(show).getByRole('radio', { name: 'Everyone' })).toBeChecked();
    expect(screen.getByRole('combobox', { name: 'Someone else' })).toBeInTheDocument(); // Bob isn't pinned
    await userEvent.click(within(show).getByText('Alice'));
    expect(await screen.findByText(/Alice was around 30 min on 1 day/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Unpin Alice' }));
    await waitFor(() => expect(saved).toEqual([[{ ...pinned, people: [] }]]));
  });

  it('pins the person shown from the list when nobody is pinned yet, keeping focus on the button', async () => {
    const saved = serve([built[0] as ChosenReport]);
    renderPage(<StatisticsTab />);
    await userEvent.click(await screen.findByRole('combobox', { name: 'Someone else' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Alice' }));
    await userEvent.click(screen.getByRole('button', { name: 'Pin Alice' }));
    await waitFor(() => expect(saved).toEqual([[{ ...built[0], people: ['p-alice'] }]]));
    expect(await screen.findByRole('button', { name: 'Unpin Alice' })).toHaveFocus();
  });

  it('offers to unpin someone pinned but not seen in this time', async () => {
    const pinned = { ...(built[0] as ChosenReport), people: ['p-gone', 'p-alice'] };
    const saved = serve([pinned]);
    renderPage(<StatisticsTab />);
    expect(await screen.findByText("1 pinned person wasn't seen in this time.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Unpin them' }));
    await waitFor(() => expect(saved).toEqual([[{ ...pinned, people: ['p-alice'] }]]));
  });

  it('keeps the filter, and focus, while the next person loads', async () => {
    const pinned = { ...(built[0] as ChosenReport), people: ['p-alice'] };
    serve([pinned]);
    renderPage(<StatisticsTab />);
    const everyone = await screen.findByRole('radio', { name: 'Everyone' });
    everyone.focus();
    await userEvent.keyboard('{ArrowRight}');
    expect(await screen.findByText(/Alice was around 30 min on 1 day/)).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Alice' })).toHaveFocus();
  });

  it('moves a report down', async () => {
    const saved = serve();
    renderPage(<StatisticsTab />);
    await userEvent.click(await screen.findByRole('button', { name: 'Arrange' }));
    await userEvent.click(screen.getByRole('button', { name: "Move Who's around down" }));
    await waitFor(() => expect(saved).toEqual([[built[1], built[0]]]));
  });

  it('lists the words that come up while there is one week, leaving out filler', async () => {
    const week = (n: number) => [{ week: '2026-11-02', n, day: DAY }];
    serve([built[1] as ChosenReport]);
    server.use(
      http.get('/reports/api/report', () =>
        HttpResponse.json({
          ...talk,
          words: [
            { word: 'garden', total: 9, weeks: week(9) },
            { word: 'stuff', total: 7, weeks: week(7) },
            { word: 'tomatoes', total: 1, weeks: week(1) },
          ],
        }),
      ),
    );
    renderPage(<StatisticsTab />);
    const list = await screen.findByRole('list', { name: 'Words said most' });
    expect(
      within(list)
        .getAllByRole('listitem')
        .map(li => li.textContent),
    ).toEqual(['garden9 times', 'tomatoes1 time']);
    expect(screen.queryByRole('figure', { name: /each top word/ })).not.toBeInTheDocument();
  });

  it('charts the words week by week once there are two weeks', async () => {
    const weeks = [
      { week: '2026-10-26', n: 3, day: DAY },
      { week: '2026-11-02', n: 9, day: DAY },
    ];
    serve([built[1] as ChosenReport]);
    server.use(
      http.get('/reports/api/report', () =>
        HttpResponse.json({ ...talk, words: [{ word: 'garden', total: 12, weeks }] }),
      ),
    );
    renderPage(<StatisticsTab />);
    expect(await screen.findByRole('figure', { name: /each top word/ })).toBeInTheDocument();
    expect(screen.queryByRole('list', { name: 'Words said most' })).not.toBeInTheDocument();
  });

  it("shows health's measures as small lines with where they are now, the hourly charts folded", async () => {
    const hours = [
      { t: T10, mem_pct: 40, gpu_mem_pct: 70, gpu_util: 5 },
      { t: T10 + 3600, mem_pct: null, gpu_mem_pct: null, gpu_util: null },
      { t: T10 + 7200, mem_pct: 62, gpu_mem_pct: 70, gpu_util: 30 },
    ];
    serve([{ id: 'health', kind: 'health', arg: null, title: 'System health' }]);
    server.use(http.get('/reports/api/report', () => HttpResponse.json({ ...health, hours })));
    renderPage(<StatisticsTab />);
    expect(await screen.findByRole('img', { name: /^Memory in use: 62% now, from 40 to 62%/ })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /^GPU memory in use \(peak\): 70% now, steady/ })).toBeInTheDocument();
    expect(screen.queryByRole('figure', { name: /Memory in use, each hour/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Hour by hour' }));
    expect(await screen.findByRole('figure', { name: /Memory in use, each hour/ })).toBeInTheDocument();
  });

  it('says how many days are still being read rather than showing them as empty', async () => {
    serve([built[0] as ChosenReport], around({ pending: ['2026-10-19', '2026-10-18'] }));
    renderPage(<StatisticsTab />);
    expect(await screen.findByText(/Still reading 2 days/)).toBeInTheDocument();
  });

  it('keeps the card at its reserved height while the server is still reading the newest day', async () => {
    const today = around();
    serve([built[0] as ChosenReport], around({ pending: [today.last] }));
    renderPage(<StatisticsTab />);
    const status = await screen.findByText(/Reading 1 day of records/);
    expect(status.closest('[data-kind="whos_around"]')?.querySelector('[data-report-loading]')).not.toBeNull();
  });

  it('moves between linked marks with the arrow keys', async () => {
    const two = around();
    const first = two.days[0];
    if (first) first.hours[11] = { t: T10 + 3600, who: [['p-alice', 10]] };
    serve([built[0] as ChosenReport], two);
    renderPage(<StatisticsTab />);
    const ten = await screen.findByRole('link', { name: /10 am: Alice/ });
    expect(ten).toHaveAttribute('tabindex', '0');
    ten.focus();
    await userEvent.keyboard('{ArrowDown}');
    expect(screen.getByRole('link', { name: /11 am: Alice/ })).toHaveFocus();
  });
});

describe('suggested reports', () => {
  it('does not offer opt-in report kinds as removed built-ins before they have data', async () => {
    serve();
    renderPage(<StatisticsTab />);
    await screen.findByRole('region', { name: 'Suggested reports' });
    expect(screen.queryByRole('button', { name: 'Add Device observations' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Add Walk-bys by hour' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Add Person activity' })).not.toBeInTheDocument();
  });
  it('shows five at a time, More shows the next, and Add puts one on the page', async () => {
    const saved = serve();
    renderPage(<StatisticsTab />);
    const panel = await screen.findByRole('region', { name: 'Suggested reports' });
    // Activity and System health were removed (not in this layout), so they're offered first.
    await waitFor(() => expect(within(panel).getAllByRole('button', { name: /^Add / })).toHaveLength(5));
    expect(within(panel).getAllByRole('button', { name: /^Add / })[0]).toHaveAccessibleName('Add Activity and events');
    await userEvent.click(within(panel).getByRole('button', { name: 'More (4 left)' }));
    expect(within(panel).getAllByRole('button', { name: /^Add / })).toHaveLength(9);
    await userEvent.click(within(panel).getByRole('button', { name: 'Add Visits to zone 6' }));
    await waitFor(() =>
      expect(saved.at(-1)?.at(-1)).toEqual({
        id: 'activity:zone:z6',
        kind: 'activity',
        arg: 'zone:z6',
        title: 'Visits to zone 6',
      }),
    );
  });
});

describe('Activity and System health', () => {
  it('states the most common event against its typical day, and a day not looked at is unlinked', async () => {
    serve([{ id: 'activity', kind: 'activity', arg: null, title: 'Activity and events' }]);
    renderPage(<StatisticsTab />);
    expect(await screen.findByText(/^Pushups is the most common event, 3 on a typical day/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Tue Oct 20: pushups 6 times\. Open in Review/ })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /Mon Oct 19/ })).not.toBeInTheDocument();
  });

  it('names an unnamed crossing event in plain words, keeping its series key', () => {
    const crossing = { ...activity.events[0], key: 'event:crossing', name: 'crossing' } as Activity['events'][number];
    expect(finding({ ...activity, events: [crossing] })).toMatch(/^Walked past is the most common event/);
  });

  it('preserves an explicit model event label in Statistics', () => {
    const action = {
      ...activity.events[0],
      key: 'event:opening_package',
      name: 'Opening a package',
    } as Activity['events'][number];
    expect(finding({ ...activity, events: [action] })).toMatch(/^Opening a package is the most common event/);
  });

  it('leads with what is down, in plain names and words not only color, then storage against the plan', async () => {
    serve([{ id: 'health', kind: 'health', arg: null, title: 'System health' }]);
    renderPage(<StatisticsTab />);
    expect(await screen.findByText('1 service is down: Listening')).toBeInTheDocument();
    expect(
      screen.getByText('Recordings take 30 GB, over the 18.75 GB the plan allows for this day. Listening is down.'),
    ).toBeInTheDocument();
    expect(screen.getByText('30 of 18.75 GB planned, over the plan')).toBeInTheDocument();
    const services = screen.getByRole('list', { name: 'Services, last 30 days' });
    expect(within(services).getAllByRole('listitem')[0]).toHaveTextContent(/^Listeningdown/); // down comes first
  });
});

describe('chart helpers', () => {
  it.each([
    [0, 10, 0],
    [0.1, 10, 0],
    [10, 10, 5],
    [20, 10, 5],
    [5, 0, 0],
  ])('step(%s, %s) is %s', (value, max, expected) => {
    expect(step(value, max)).toBe(expected);
  });

  it('ticks the first of each month over a long range', () => {
    const days = Array.from({ length: 90 }, (_, i) => new Date(Date.UTC(2026, 0, 1 + i)).toISOString().slice(0, 10));
    expect([...dayTicks(days)]).toEqual([0, 31, 59]);
  });
});
