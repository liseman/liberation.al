import { fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import type { Series, WhosAround } from './api';
import { ConversationChord, EventStream, PresenceClock } from './visualizations';

const DAY = '2026-10-20';
const T = new Date(`${DAY}T10:13:00`).getTime() / 1000;
const around: WhosAround = {
  report: 'whos_around',
  first: DAY,
  last: DAY,
  pending: [],
  refreshing: [],
  people: [
    { id: 'p-a', name: 'Alice', hours: 1 },
    { id: 'p-b', name: 'Dana', hours: 0.5 },
  ],
  days: [
    {
      day: DAY,
      hours: Array.from({ length: 24 }, (_, h) =>
        h < 8
          ? null
          : {
              t: T + (h - 10) * 3600,
              who:
                h === 10
                  ? [
                      ['p-a', 60],
                      ['p-b', 30],
                    ]
                  : [],
            },
      ),
    },
  ],
};

describe('presence clock', () => {
  it('preserves gaps and recorded empty hours, and links a populated sector to the server timestamp', async () => {
    renderPage(<PresenceClock data={around} person={null} />);
    const chart = screen.getByRole('figure');
    expect(within(chart).getAllByRole('link')).toHaveLength(1);
    const mark = screen.getByRole('link', { name: /10 am: Alice 60 min, Dana 30 min/ });
    expect(mark).toHaveAttribute('href', `/review#day=${DAY}&t=${T}`);
    expect(mark.querySelector('path')).toHaveAttribute('d', expect.stringContaining('A'));
    expect(within(chart).getByText(/7 am: nothing recorded/)).toBeInTheDocument();
    expect(within(chart).getByText(/8 am: nobody seen or heard/)).toBeInTheDocument();
    await userEvent.tab();
    expect(mark).toHaveFocus();
    expect(screen.getByText(/10 am: Alice/, { selector: 'div' }).parentElement).toBe(document.body);
    fireEvent.scroll(window);
    expect(screen.queryByText(/10 am: Alice/, { selector: 'div' })).not.toBeInTheDocument();
  });

  it('keeps older days reachable without squeezing a year into rings', async () => {
    const days = Array.from({ length: 30 }, (_, i) => ({
      day: `2026-10-${String(i + 1).padStart(2, '0')}`,
      hours: around.days[0]?.hours ?? [],
    }));
    renderPage(<PresenceClock data={{ ...around, days }} person={null} />);
    expect(screen.getByRole('figure')).toHaveAccessibleName(/Oct 17 to Fri Oct 30/);
    await userEvent.click(screen.getByRole('button', { name: 'Older days' }));
    expect(screen.getByRole('figure')).toHaveAccessibleName(/Oct 3 to Fri Oct 16/);
    await userEvent.click(screen.getByRole('button', { name: 'Newer days' }));
    expect(screen.getByRole('figure')).toHaveAccessibleName(/Oct 17 to Fri Oct 30/);
  });

  it('offers no paging when every day fits, and says what the rings are', () => {
    renderPage(<PresenceClock data={around} person={null} />);
    expect(screen.queryByRole('button', { name: /days$/ })).not.toBeInTheDocument();
    expect(screen.getByText(/^Each ring is a day: .+ in the middle, .+ outside$/)).toBeInTheDocument();
  });
});

describe('exchange chord', () => {
  it('draws one ribbon per pair with the undoubled count and exact latest Review link', async () => {
    renderPage(
      <ConversationChord
        pairs={[
          { a: 'p-a', b: 'p-b', a_name: 'Alice', b_name: 'Dana', n: 7, day: DAY, t: T },
          { a: 'p-b', b: 'p-c', a_name: 'Dana', b_name: 'Pat', n: 2, day: DAY, t: T + 60 },
        ]}
      />,
    );
    const chart = screen.getByRole('figure');
    expect(within(chart).getAllByRole('link')).toHaveLength(2);
    const first = screen.getByRole('link', { name: /Alice and Dana: 7 exchanges/ });
    expect(first).toHaveAttribute('href', `/review#day=${DAY}&t=${T}`);
    expect(first.querySelector('path')).toHaveAttribute('d', expect.stringContaining('Q'));
    first.focus();
    await userEvent.keyboard('{ArrowRight}');
    expect(screen.getByRole('link', { name: /Dana and Pat: 2 exchanges/ })).toHaveFocus();
  });
});

describe('event stream', () => {
  const series: Series[] = [
    {
      key: 'event:crossing',
      name: 'crossing',
      total: 5,
      typical: 2.5,
      days: [
        { day: '2026-10-19', n: null, t: null },
        { day: DAY, n: 5, t: T },
        { day: '2026-10-21', n: 0, t: null },
      ],
      by_hour: [],
    },
  ];

  it('moves focus between quoted owner-named events and keeps the destination Review timestamp', async () => {
    const nextTime = T + 86400;
    renderPage(
      <EventStream
        series={[
          {
            key: 'event:Saying "hello"',
            name: 'Saying "hello"',
            total: 5,
            typical: 2.5,
            by_hour: [],
            days: [
              { day: DAY, n: 2, t: T },
              { day: '2026-10-21', n: 3, t: nextTime },
            ],
          },
        ]}
      />,
    );
    const first = screen.getByRole('link', { name: /Tue Oct 20: Saying "hello" 2 times/ });
    const next = screen.getByRole('link', { name: /Wed Oct 21: Saying "hello" 3 times/ });
    await userEvent.tab();
    expect(first).toHaveFocus();
    await userEvent.keyboard('{ArrowRight}');
    expect(next).toHaveFocus();
    expect(next).toHaveAttribute('href', `/review#day=2026-10-21&t=${nextTime}`);
  });

  it('hatches unchecked days and only links positive counts, including a single measured day', () => {
    renderPage(<EventStream series={series} />);
    const chart = screen.getByRole('figure');
    expect(within(chart).getAllByRole('link')).toHaveLength(1);
    expect(screen.getByRole('link', { name: /Tue Oct 20: Walked past 5 times/i })).toHaveAttribute(
      'href',
      `/review#day=${DAY}&t=${T}`,
    );
    expect(within(chart).getByText('Mon Oct 19: not looked for')).toBeInTheDocument();
    expect(within(chart).getByText(/Wed Oct 21: Walked past 0 times/i)).toBeInTheDocument();
    expect(chart.querySelector('rect[fill^="url"]')).toBeInTheDocument();
  });

  it('keeps the stream near its drawn size on a wide card, so its axis text stays the size of the other charts', () => {
    renderPage(<EventStream series={series} />);
    // 202 units tall, allowed 10% larger: a 1,240 px card no longer scales its 12 px labels to about 25 px.
    expect(screen.getByRole('figure').querySelector('svg')).toHaveStyle({ maxHeight: '222px' });
  });

  it('does not turn an unmeasured series into zero when another series covers that day', () => {
    renderPage(
      <EventStream
        series={[
          ...series,
          { ...(series[0] as Series), key: 'event:other', name: 'other', days: [{ day: DAY, n: null, t: null }] },
        ]}
      />,
    );
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText('Tue Oct 20: not looked for')).toBeInTheDocument();
  });
});
