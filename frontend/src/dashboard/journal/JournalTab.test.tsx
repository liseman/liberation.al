import { fireEvent, screen, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { Column, Evidence } from './api';
import { todayWait } from './api';
import { JournalTab } from './JournalTab';

const HOUR = 1762502400; // a made-up hour start
const REPLIES: Record<string, Column | Evidence> = {
  '': {
    items: [
      { key: '2026-11-01', label: 'November 2026', page: false },
      { key: '2026-10-01', label: 'October 2026', page: true },
      { key: '2026-09-01', label: 'September 2026', page: false },
    ],
    page: null,
  },
  'month=2026-11-01': { items: [{ key: '2026-11-02', label: 'Nov 2 to 8', page: false }], page: null },
  'month=2026-10-01': {
    items: [{ key: '2026-10-26', label: 'Oct 26 to Nov 1', page: true }],
    page: { level: 'month', period: '2026-10-01', md: '# October\n\n## Notable events\n- Dana visited' },
  },
  'month=2026-09-01': { items: [], page: null },
  'week=2026-11-02': {
    items: [{ key: '2026-11-05', label: 'Thu Nov 5', page: true, hours: 2 }],
    page: null,
    why: 'rewriting',
  },
  'week=2026-10-26': { items: [], page: null },
  'day=2026-11-05': {
    items: [
      {
        key: String(HOUR),
        id: 'h1',
        label: '09:00',
        start: HOUR,
        end: HOUR + 3600,
        text: 'Dana and I made coffee.',
        sources: 3,
      },
    ],
    page: { level: 'day', period: '2026-11-05', md: '# Thursday\n\n| What | Today |\n| --- | ---: |\n| Barks | 14 |' },
  },
  [`entry=h1&start=${HOUR}`]: {
    status: 'verified',
    sources: [
      {
        stage: 'audio',
        caption: 'want coffee?',
        people: ['Dana'],
        start: HOUR + 60,
        end: HOUR + 64,
        review: `/review#t=${HOUR + 60}.0`,
        audio: '/audio/range?start=1&end=2',
      },
      {
        stage: 'clip',
        caption: 'pouring coffee',
        camera: 'kitchen',
        people: [],
        start: HOUR + 70,
        end: HOUR + 75,
        review: `/review#t=${HOUR + 70}.0`,
        url: '/journal/evidence/2026-11-05/kitchen-1.mp4',
        still: '/journal/evidence/2026-11-05/kitchen-1.jpg',
      },
      {
        stage: 'expired',
        note: 'footage expired',
        caption: 'two people in the kitchen',
        camera: 'kitchen',
        people: [],
        start: HOUR + 90,
        end: HOUR + 95,
        review: `/review#t=${HOUR + 90}.0`,
      },
    ],
  },
  [`hour=${HOUR}`]: {
    items: [{ key: 'e1', id: 'e1', label: '09:01', start: HOUR + 60, end: HOUR + 240, text: 'Coffee.', sources: 0 }],
    page: null,
  },
};

/** Every /journal query the page made, in order. */
let asked: string[] = [];

function serveJournal() {
  asked = [];
  server.use(
    http.get('/journal', ({ request }) => {
      const q = new URL(request.url).searchParams.toString();
      asked.push(q);
      const column = REPLIES[q];
      return column ? HttpResponse.json(column) : HttpResponse.json({ error: 'no' }, { status: 400 });
    }),
  );
}

const column = (name: string) => screen.findByRole('listbox', { name });
/** A column's row, once the column has loaded. */
const option = (list: HTMLElement, name: RegExp) => within(list).findByRole('option', { name });

it("opens on the newest month, with its weeks, and says when a month's page isn't written yet", async () => {
  serveJournal();
  renderPage(<JournalTab />);
  const weeks = await column('Weeks');
  expect(await within(weeks).findByText('Nov 2 to 8')).toBeTruthy();
  expect((await option(await column('Months'), /November 2026/)).getAttribute('aria-selected')).toBe('true');
  expect(await screen.findByText(/This month's page isn't written yet/)).toBeTruthy();
});

it('says why a finished week shows no page: being written again after a transcript correction', async () => {
  serveJournal();
  renderPage(<JournalTab />);
  fireEvent.click(await option(await column('Weeks'), /Nov 2 to 8/));
  expect(await screen.findByText(/This week's page is being written again after a transcript correction/)).toBeTruthy();
  expect(screen.queryByText(/isn't written yet/)).toBeNull();
});

it('drills from a month to an hour, showing each page and then the hour with what it cites', async () => {
  serveJournal();
  renderPage(<JournalTab />);
  fireEvent.click(await option(await column('Months'), /October 2026/));
  expect(await screen.findByText('Dana visited')).toBeTruthy(); // October's page
  fireEvent.click(await option(await column('Weeks'), /Oct 26/));
  expect(await screen.findByText('No days in this week yet.')).toBeTruthy();

  fireEvent.click(await option(await column('Months'), /November 2026/));
  fireEvent.click(await option(await column('Weeks'), /Nov 2 to 8/));
  fireEvent.click(await option(await column('Days'), /Thu Nov 5/));
  expect(await screen.findByRole('cell', { name: '14' })).toBeTruthy(); // the day page's Counts table
  fireEvent.click(await option(await column('Hours'), /09:00/));
  expect(await within(await column('Events')).findByText('Coffee.')).toBeTruthy();
  expect(screen.getByRole('link', { name: 'Where, on the plan' }).getAttribute('href')).toBe(`/places?t=${HOUR}`);
  expect(await screen.findByText('want coffee?')).toBeTruthy(); // the hour's evidence, read when it's shown
  expect(document.querySelector('audio')?.getAttribute('src')).toBe('/audio/range?start=1&end=2');
  expect(document.querySelector('video')?.getAttribute('src')).toBe('/journal/evidence/2026-11-05/kitchen-1.mp4');
  expect(screen.getByText(/^Footage expired\. What was seen and heard is still kept\.$/)).toBeTruthy();
  expect(screen.getAllByRole('link', { name: 'Open in Review' })[2]?.getAttribute('href')).toBe(
    `/review#t=${HOUR + 90}.0`,
  );
});

it('moves like Finder: up and down within a column, right into the next, left back out', async () => {
  serveJournal();
  renderPage(<JournalTab />);
  const months = await column('Months');
  await column('Weeks');
  months.focus();
  fireEvent.keyDown(months, { key: 'ArrowDown' });
  expect((await option(months, /October 2026/)).getAttribute('aria-selected')).toBe('true');
  const weeks = await screen.findByRole('listbox', { name: 'Weeks' });
  await within(weeks).findByText('Oct 26 to Nov 1');
  fireEvent.keyDown(months, { key: 'ArrowRight' });
  expect(document.activeElement).toBe(weeks);
  expect((await option(weeks, /Oct 26/)).getAttribute('aria-selected')).toBe('true');
  fireEvent.keyDown(weeks, { key: 'ArrowLeft' });
  expect(document.activeElement).toBe(months);
});

it('shows one column at a time on a phone, with Back', async () => {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    query =>
      ({
        matches: query.includes('max-width'),
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList,
  );
  serveJournal();
  renderPage(<JournalTab />);
  fireEvent.click(await option(await column('Months'), /November 2026/));
  const weeks = await column('Weeks');
  expect(screen.queryByRole('listbox', { name: 'Months' })).toBeNull();
  expect(await within(weeks).findByText('Nov 2 to 8')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Back' }));
  expect(await column('Months')).toBeTruthy();
});

it("shows today's page so far, and says while it's being brought up to date", async () => {
  const now = new Date();
  const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  let asked = 0;
  server.use(
    http.get('/journal', ({ request }) => {
      const q = new URL(request.url).searchParams;
      if (!q.toString()) return HttpResponse.json({ items: [{ key: 'm', label: 'This month' }], page: null });
      if (q.get('month')) return HttpResponse.json({ items: [{ key: 'w', label: 'This week' }], page: null });
      if (q.get('week')) return HttpResponse.json({ items: [{ key: today, label: 'Today', hours: 3 }], page: null });
      return HttpResponse.json({ items: [], page: null });
    }),
    http.get('/journal/today', ({ request }) => {
      expect(new URL(request.url).searchParams.get('day')).toBe(today);
      asked += 1;
      return asked === 1
        ? HttpResponse.json({ day: today, status: 'pending', md: '# Earlier so far' }, { status: 202 })
        : HttpResponse.json({ day: today, status: 'ready', md: '# Today so far\n\n- Coffee [h1](#t=1)' });
    }),
  );
  renderPage(<JournalTab />);
  fireEvent.click(await option(await column('Weeks'), /This week/));
  fireEvent.click(await option(await column('Days'), /Today/));
  expect(await screen.findByText('Bringing this up to date…')).toBeTruthy();
  expect(screen.getByText('Earlier so far')).toBeTruthy();
  expect(await screen.findByText('Today so far', {}, { timeout: 15_000 })).toBeTruthy(); // asked again while pending
  expect(screen.getByRole('link', { name: 'h1' }).getAttribute('href')).toBe('/review#t=1');
});

it('reads only the column an arrow key stops on, not every row it passes', async () => {
  serveJournal();
  renderPage(<JournalTab />);
  const months = await column('Months');
  await within(await column('Weeks')).findByText('Nov 2 to 8');
  months.focus();
  fireEvent.keyDown(months, { key: 'ArrowDown' });
  fireEvent.keyDown(months, { key: 'ArrowDown' });
  expect(await within(await column('Weeks')).findByText('No weeks in this month yet.')).toBeTruthy();
  expect(asked.filter(q => q.startsWith('month='))).toEqual(['month=2026-11-01', 'month=2026-09-01']);
});

it('moves the keys to the column a mouse picks in', async () => {
  serveJournal();
  renderPage(<JournalTab />);
  const weeks = await column('Weeks');
  fireEvent.click(await option(weeks, /Nov 2 to 8/));
  expect(document.activeElement).toBe(weeks);
});

it("backs off while today's page builds, to a minute at most", () => {
  expect([1, 2, 3].map(todayWait)).toEqual([5000, 7500, 11250]);
  expect(todayWait(20)).toBe(60_000);
});

it("keeps asking after today's page failed, since the server tries again", async () => {
  const now = new Date();
  const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  let tries = 0;
  server.use(
    http.get('/journal', ({ request }) => {
      const q = new URL(request.url).searchParams;
      if (!q.toString()) return HttpResponse.json({ items: [{ key: 'm', label: 'This month' }], page: null });
      if (q.get('month')) return HttpResponse.json({ items: [{ key: 'w', label: 'This week' }], page: null });
      if (q.get('week')) return HttpResponse.json({ items: [{ key: today, label: 'Today', hours: 3 }], page: null });
      return HttpResponse.json({ items: [], page: null });
    }),
    http.get('/journal/today', () => {
      tries += 1;
      return tries === 1
        ? HttpResponse.json({ day: today, status: 'error', md: '' })
        : HttpResponse.json({ day: today, status: 'ready', md: '# Today so far' });
    }),
  );
  renderPage(<JournalTab />);
  fireEvent.click(await option(await column('Weeks'), /This week/));
  fireEvent.click(await option(await column('Days'), /Today/));
  expect(await screen.findByText(/couldn't be built/)).toBeTruthy();
  expect(await screen.findByText('Today so far', {}, { timeout: 15_000 })).toBeTruthy();
});

it("says today's page couldn't be read when the dashboard fails, not that it isn't written", async () => {
  const now = new Date();
  const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  server.use(
    http.get('/journal', ({ request }) => {
      const q = new URL(request.url).searchParams;
      if (!q.toString()) return HttpResponse.json({ items: [{ key: 'm', label: 'This month' }], page: null });
      if (q.get('month')) return HttpResponse.json({ items: [{ key: 'w', label: 'This week' }], page: null });
      if (q.get('week')) return HttpResponse.json({ items: [{ key: today, label: 'Today', hours: 3 }], page: null });
      return HttpResponse.json({ items: [], page: null });
    }),
    http.get('/journal/today', () => HttpResponse.json({ error: 'the journal store is locked' }, { status: 500 })),
  );
  renderPage(<JournalTab />);
  fireEvent.click(await option(await column('Weeks'), /This week/));
  fireEvent.click(await option(await column('Days'), /Today/));
  expect(await screen.findByText(/Couldn't read the day so far: the journal store is locked/)).toBeTruthy();
  expect(screen.queryByText(/isn't written yet/)).toBeNull();
});

it("says a day the today route doesn't build (404) isn't written yet", async () => {
  const now = new Date();
  const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  server.use(
    http.get('/journal', ({ request }) => {
      const q = new URL(request.url).searchParams;
      if (!q.toString()) return HttpResponse.json({ items: [{ key: 'm', label: 'This month' }], page: null });
      if (q.get('month')) return HttpResponse.json({ items: [{ key: 'w', label: 'This week' }], page: null });
      if (q.get('week')) return HttpResponse.json({ items: [{ key: today, label: 'Today', hours: 3 }], page: null });
      return HttpResponse.json({ items: [], page: null });
    }),
    http.get('/journal/today', () => HttpResponse.json({ error: 'only today' }, { status: 404 })),
  );
  renderPage(<JournalTab />);
  fireEvent.click(await option(await column('Weeks'), /This week/));
  fireEvent.click(await option(await column('Days'), /Today/));
  expect(await screen.findByText(/This day's page isn't written yet/)).toBeTruthy();
});
