// Activity and events: detected events, zone visits, and pet activity, each by day against a typical day (the
// median of the days it was looked for) and, for animals, by hour of day. Days the worker never looked at are gaps.

import { Group, Stack, Text, Title } from '@mantine/core';
import { activityHint, eventName } from '../lib/activity';
import { capitalize } from '../lib/format';
import type { Activity as Data, Series } from './api';
import { useReport } from './api';
import {
  Bars,
  dayName,
  dayTicks,
  GapKey,
  Heatmap,
  hourName,
  LineKey,
  Pending,
  RampLegend,
  reviewAt,
  shortDayName,
} from './charts';
import classes from './reports.module.css';
import { ReportState } from './state';
import { EventStream } from './visualizations';

const HOURS = Array.from({ length: 24 }, (_, h) => hourName(h));
const HOUR_TICKS = new Set([0, 3, 6, 9, 12, 15, 18, 21]);
const times = (n: number) => `${n} ${n === 1 ? 'time' : 'times'}`;
const typical = (s: Series) => (s.typical === null ? 'not looked for yet' : `${s.typical} on a typical day`);

export function finding(data: Data): string {
  const parts: string[] = [];
  const event = data.events.find(s => s.name !== 'Other events');
  if (event) parts.push(`${eventName(event.name)} is the most common event, ${typical(event)}.`);
  const zone = data.zones[0];
  if (zone) parts.push(`${zone.name} gets the most visits, ${typical(zone)}.`);
  const pet = data.pets[0];
  if (pet) {
    const peak = pet.by_hour.reduce((best, c, h, all) => (c.n > (all[best]?.n ?? 0) ? h : best), 0);
    parts.push(`The ${pet.name} is most active around ${hourName(peak)}.`);
  }
  if (parts.length) return parts.map(capitalize).join(' ');
  if (!data.covered_days.events && !data.covered_days.zones)
    return 'Nothing has been checked for events or zone visits yet.';
  return 'No events or zone visits were found in the time that was checked.';
}

function SeriesChart({ s, unit }: { s: Series; unit: string }) {
  const days = s.days.map(d => d.day);
  const name = eventName(s.name);
  return (
    <Stack gap={4}>
      <Text size="sm" fw={600} title={activityHint(s.name) ?? undefined}>
        {name}{' '}
        <Text span size="xs" c="dimmed" fw={400}>
          {s.total} in all, {typical(s)}
        </Text>
      </Text>
      <Bars
        label={`${name}, ${unit} each day`}
        points={s.days.map(d => ({
          key: d.day,
          value: d.n,
          label: d.n === null ? `${dayName(d.day)}: not looked for` : `${dayName(d.day)}: ${name} ${times(d.n)}`,
          href: d.n ? reviewAt(d.day, d.t) : null,
        }))}
        ticks={dayTicks(days)}
        tickLabel={i => shortDayName(days[i] ?? '')}
        xTitle="Day"
        yTitle="Times"
        height={90}
        plan={s.typical}
      />
    </Stack>
  );
}

function SeriesGroup({
  title,
  series,
  unit,
  empty,
  stream = false,
}: {
  title: string;
  series: Series[];
  unit: string;
  empty: string;
  stream?: boolean;
}) {
  return (
    <Stack gap="xs">
      <Group justify="space-between" wrap="wrap">
        <Title order={3}>{title}</Title>
        {series.length > 0 && (
          <Group gap="md">
            {!stream && <LineKey>Typical day</LineKey>}
            <GapKey>Not looked for</GapKey>
          </Group>
        )}
      </Group>
      {series.length ? (
        stream ? (
          <>
            <EventStream series={series} />
            {series.map(s => (
              <Text key={s.key} size="xs" c="dimmed">
                {eventName(s.name)}: {s.total} in all, {typical(s)}
              </Text>
            ))}
          </>
        ) : (
          series.map(s => <SeriesChart key={s.key} s={s} unit={unit} />)
        )
      ) : (
        <Text size="sm" c="dimmed">
          {empty}
        </Text>
      )}
    </Stack>
  );
}

export function Activity({ arg }: { arg: string | null }) {
  const query = useReport<Data>('activity', arg);
  return (
    <ReportState query={query}>
      {data => (
        <Stack gap="lg">
          <Text className={classes.findings}>{finding(data)}</Text>
          {(!arg || arg.startsWith('event:')) && (
            <SeriesGroup
              title="Events"
              stream
              series={data.events}
              unit="times"
              empty={
                data.covered_days.events ? 'No events were detected.' : 'No video has been checked for events yet.'
              }
            />
          )}
          {(!arg || arg.startsWith('zone:')) && (
            <SeriesGroup
              title="Zone visits"
              series={data.zones}
              unit="visits"
              empty={
                data.covered_days.zones
                  ? 'Nothing stayed in a zone long enough to count.'
                  : 'No zones are drawn yet, or none have been checked. Draw them in Settings, on a camera.'
              }
            />
          )}
          {(!arg || arg.startsWith('pet:')) && (
            <Stack gap="xs">
              <Group justify="space-between" wrap="wrap">
                <Title order={3}>Pet activity by hour of day</Title>
                {data.pets.length > 0 && <RampLegend low="Fewer visits" high="More" />}
              </Group>
              {data.pets.length ? (
                <>
                  <Heatmap
                    label="Zone visits by each animal in each hour of the day"
                    rows={data.pets.map(p => p.name)}
                    cols={HOURS}
                    colTicks={HOUR_TICKS}
                    cell={(r, h) => {
                      const p = data.pets[r];
                      const c = p?.by_hour[h];
                      return {
                        value: c?.n ?? 0,
                        label: `The ${p?.name}, ${hourName(h)}: ${c?.n ?? 0} zone ${c?.n === 1 ? 'visit' : 'visits'}`,
                        href: c?.n && c.day ? reviewAt(c.day, c.t) : null,
                      };
                    }}
                    xTitle="Hour of day (links open the latest day)"
                    yTitle=""
                    cellHeight={22}
                    maxCellWidth={36}
                    minCellWidth={11}
                    rowLabelWidth={80}
                  />
                  {arg?.startsWith('pet:') && data.pets.map(s => <SeriesChart key={s.key} s={s} unit="visits" />)}
                </>
              ) : (
                <Text size="sm" c="dimmed">
                  {data.covered_days.pets
                    ? 'No animal has visited a zone yet.'
                    : 'No zones have been checked for animals yet. Draw zones in Settings, on a camera.'}
                </Text>
              )}
            </Stack>
          )}
          <Pending pending={data.pending} />
        </Stack>
      )}
    </ReportState>
  );
}
