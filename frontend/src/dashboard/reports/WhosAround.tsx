// Who's around: a calendar heatmap of who was home or in view, by day (across) and hour (down), from face visits and
// heard voices. Everyone at once shows how many people were there; one person shows their minutes. The owner's pinned
// people (#2582, which folded the old "When X is around" cards in) sit one click away beside Everyone.

import { Group, SegmentedControl, Stack, Text } from '@mantine/core';
import { useRef, useState } from 'react';
import { minutesText } from '../lib/format';
import type { WhosAround as Data } from './api';
import { useReport } from './api';
import { dayName, dayTicks, Heatmap, hourName, Pending, RampLegend, reviewAt, shortDayName } from './charts';
import { PinFilter } from './PinFilter';
import classes from './reports.module.css';
import { ReportState } from './state';
import { PresenceClock } from './visualizations';

const HOURS = Array.from({ length: 24 }, (_, h) => hourName(h));
const HOUR_TICKS = new Set([0, 3, 6, 9, 12, 15, 18, 21]);

/** The sentence over the chart: who was around most, or when the one person shown usually is. */
export function finding(data: Data, person: string | null): string {
  const named = person ? data.people.find(p => p.id === person) : null;
  if (!data.people.length) return 'Nobody was seen or heard in this time.';
  const byHour = new Array<number>(24).fill(0);
  for (const d of data.days)
    d.hours.forEach((cell, h) => {
      for (const [, minutes] of cell?.who ?? []) byHour[h] = (byHour[h] ?? 0) + minutes;
    });
  const peak = byHour.indexOf(Math.max(...byHour));
  const days = data.days.filter(d => d.hours.some(c => c?.who.length)).length;
  if (named)
    return `${named.name} was around ${minutesText(named.hours * 60)} on ${days} ${days === 1 ? 'day' : 'days'}, most often around ${hourName(peak)}.`;
  const top = data.people[0];
  return `${top?.name} was around most, ${minutesText((top?.hours ?? 0) * 60)}. The house is busiest around ${hourName(peak)}.`;
}

export function WhosAround({
  arg,
  pinned = [],
  onPinned,
}: {
  arg: string | null;
  /** The owner's pinned people, by id, saved with the page's layout. */
  pinned?: string[];
  onPinned?: (people: string[]) => void;
}) {
  const [person, setPerson] = useState<string | null>(arg);
  const [view, setView] = useState('clock');
  // Keep showing the last figures while another person's load (#2591): swapping the card for a skeleton on each step
  // would take the filter, and keyboard focus with it, away mid-arrow. The chart dims until the new figures arrive.
  const query = useReport<Data>('whos_around', person, { keep: true });
  const last = useRef(person);
  if (!query.isPlaceholderData) last.current = person;
  const shown = last.current;
  return (
    <ReportState query={query}>
      {data => {
        const days = data.days.map(d => d.day);
        const ticks = dayTicks(days);
        return (
          <Stack gap="sm">
            <Text className={classes.findings}>{finding(data, shown)}</Text>
            <Group justify="space-between" align="flex-end" wrap="wrap">
              <PinFilter
                choices={data.people}
                pinned={pinned}
                value={person}
                onChange={setPerson}
                onPinned={onPinned}
              />
              <RampLegend low={shown ? 'Fewer minutes' : 'Fewer people'} high="More" />
            </Group>
            <SegmentedControl
              aria-label="Presence chart"
              size="xs"
              value={view}
              onChange={setView}
              data={[
                { value: 'clock', label: 'Daily clock' },
                { value: 'calendar', label: 'Day × hour' },
              ]}
            />
            <div
              className={classes.stale}
              data-stale={query.isPlaceholderData || undefined}
              aria-busy={query.isPlaceholderData}
            >
              {view === 'clock' ? (
                <PresenceClock data={data} person={shown} />
              ) : (
                <Heatmap
                  label={`Who was around, by day and hour, ${dayName(data.first)} to ${dayName(data.last)}`}
                  rows={HOURS}
                  rowTicks={HOUR_TICKS}
                  cols={days.map(shortDayName)}
                  colTicks={ticks}
                  xTitle="Day"
                  yTitle="Hour"
                  cell={(h, c) => {
                    const day = data.days[c];
                    const cell = day?.hours[h];
                    const when = `${dayName(day?.day ?? '')}, ${hourName(h)}`;
                    if (!day || cell == null) return { value: null, label: `${when}: nothing recorded` };
                    const names = new Map(data.people.map(p => [p.id, p.name]));
                    const who = cell.who.map(([id, m]) => `${names.get(id) ?? 'Someone'} ${Math.round(m)} min`);
                    return {
                      value: shown ? (cell.who[0]?.[1] ?? 0) : cell.who.length,
                      label: who.length ? `${when}: ${who.join(', ')}` : `${when}: nobody seen or heard`,
                      href: who.length ? reviewAt(day.day, cell.t) : null,
                    };
                  }}
                />
              )}
            </div>
            <Pending pending={data.pending} />
          </Stack>
        );
      }}
    </ReportState>
  );
}
