// Review's day: Previous day and Next day, which skip days with nothing recorded, and a calendar that marks the days
// with video, room audio, or transcript lines. Any day can still be picked from the calendar.

import { ActionIcon, Group, Indicator, Tooltip } from '@mantine/core';
import { DatePickerInput } from '@mantine/dates';
import { IconChevronLeft, IconChevronRight } from '@tabler/icons-react';
import dayjs from 'dayjs';
import { kinds, type RecordedDay, stepDay } from './days';

/** How the picker shows its day ("Fri 6 Nov 2026"), and so how its accessible name says it. */
const DAY_FORMAT = 'ddd D MMM YYYY';

const longDay = (day: string) =>
  new Date(`${day}T12:00:00`).toLocaleDateString(undefined, {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });

export function DayPicker({
  day,
  days,
  onPick,
}: {
  day: string | null;
  days: readonly RecordedDay[] | undefined; // undefined while unknown: the arrows step one calendar day
  onPick: (day: string) => void;
}) {
  const byDay = new Map((days ?? []).map(d => [d.day, d]));
  const list = days?.map(d => d.day);
  const prev = day ? stepDay(list, day, -1) : null;
  const next = day ? stepDay(list, day, 1) : null;
  const arrow = (to: string | null, dir: 'Previous' | 'Next') => {
    const label = `${dir} day`;
    // A disabled button gets no hover or focus, so only an arrow that goes somewhere has a tip.
    return (
      <Tooltip label={days && to ? `${label} with recordings: ${longDay(to)}` : label} disabled={!to} withinPortal>
        <ActionIcon variant="subtle" size="md" aria-label={label} disabled={!to} onClick={() => to && onPick(to)}>
          {dir === 'Previous' ? <IconChevronLeft size={16} /> : <IconChevronRight size={16} />}
        </ActionIcon>
      </Tooltip>
    );
  };
  return (
    <Group gap={2} wrap="nowrap">
      {arrow(prev, 'Previous')}
      <DatePickerInput
        size="xs"
        aria-label={day ? `Day, ${dayjs(day).format(DAY_FORMAT)}` : 'Day'} // the date it shows, not only "Day"
        value={day}
        onChange={d => d && onPick(d)}
        valueFormat={DAY_FORMAT}
        popoverProps={{ withinPortal: true }}
        getDayAriaLabel={d => {
          const r = byDay.get(d);
          return r ? `${longDay(d)}, ${kinds(r)}` : longDay(d);
        }}
        getDayProps={d => (byDay.has(d) ? { 'data-recorded': true } : {})}
        renderDay={d => (
          <Indicator size={5} offset={-3} position="bottom-center" disabled={!byDay.has(d)}>
            <div>{Number(d.slice(8))}</div>
          </Indicator>
        )}
      />
      {arrow(next, 'Next')}
    </Group>
  );
}
