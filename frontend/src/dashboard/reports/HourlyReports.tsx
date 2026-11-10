// Optional reports share one day-by-hour view. Each source owns its coverage, so an unchecked hour stays null.
import { Badge, Stack, Text, Title } from '@mantine/core';
import { useState } from 'react';
import { activityName } from '../lib/activity';
import { Disclosure } from '../lib/Disclosure';
import { minutesText } from '../lib/format';
import type { CountDay, Devices, PersonActivity, Walkbys } from './api';
import { useReport } from './api';
import { dayName, Heatmap, hourName, Pending, RampLegend, reviewAt, shortDayName } from './charts';
import { CAMERAS, PEOPLE, PinFilter } from './PinFilter';
import classes from './reports.module.css';
import { ReportState } from './state';

const HOURS = Array.from({ length: 24 }, (_, h) => hourName(h));
const TICKS = new Set([0, 3, 6, 9, 12, 15, 18, 21]);

function Hours({
  name,
  days,
  unit,
  amount = n => `${n} ${unit}`,
  legend = true,
}: {
  name: string;
  days: CountDay[];
  unit: string;
  amount?: (n: number) => string;
  legend?: boolean;
}) {
  return (
    <Stack gap={4}>
      {legend && (
        <RampLegend low={`Fewer ${unit}`} high="More" zero="Checked, no observations" gap="Unknown or not checked" />
      )}
      <Heatmap
        label={`${name}, by day and hour`}
        rows={days.map(d => shortDayName(d.day))}
        cols={HOURS}
        colTicks={TICKS}
        cellHeight={22}
        minCellWidth={7}
        maxCellWidth={36}
        rowLabelWidth={62}
        xTitle="Hour of day"
        yTitle="Day"
        cell={(r, h) => {
          const day = days[r],
            cell = day?.hours[h];
          const when = `${dayName(day?.day ?? '')}, ${hourName(h)}`;
          return {
            value: cell?.n ?? null,
            label: cell == null ? `${when}: not checked` : `${when}: ${amount(cell.n)}`,
            href: cell?.n && cell.t != null && day ? reviewAt(day.day, cell.t) : null,
          };
        }}
      />
    </Stack>
  );
}

/** The first pinned choice that's here, else the first choice, else none (show all). */
const firstShown = (pinned: readonly string[], ids: readonly string[]): string | null =>
  pinned.find(p => ids.includes(p)) ?? ids[0] ?? null;

const walkBys = (n: number) => `${n} ${n === 1 ? 'walk-by' : 'walk-bys'}`;

/** Walk-bys by camera: every camera, or one picked in its filter. The filter works on the
 *  figures already here, so picking a camera loads nothing. A card saved for one camera (arg) shows only that one. */
export function WalkbyReport({
  arg,
  pinned = [],
  onPinned,
}: {
  arg: string | null;
  pinned?: string[];
  onPinned?: (cameras: string[]) => void;
}) {
  const query = useReport<Walkbys>('walkbys', arg);
  // Until one is picked, the card shows its first pinned camera (else the first camera): one at a time fits the card's
  // cell, and every camera is one click away. undefined: not picked yet; null: every camera.
  const [picked, setCamera] = useState<string | null | undefined>(undefined);
  return (
    <ReportState query={query}>
      {data => {
        const camera =
          picked === undefined && !arg
            ? firstShown(
                pinned,
                data.cameras.map(c => c.key),
              )
            : (picked ?? null);
        const cameras = camera ? data.cameras.filter(c => c.key === camera) : data.cameras;
        return (
          <Stack gap="lg">
            <Text className={classes.findings}>
              {cameras.length
                ? `${walkBys(cameras.reduce((n, c) => n + c.total, 0))} on ${data.covered_days} checked ${data.covered_days === 1 ? 'day' : 'days'}.`
                : 'No walk-bys found in the days read so far.'}
            </Text>
            {!arg && data.cameras.length > 1 && (
              <PinFilter
                choices={data.cameras.map(c => ({ id: c.key, name: c.name }))}
                pinned={pinned}
                value={camera}
                onChange={setCamera}
                onPinned={onPinned}
                words={CAMERAS}
              />
            )}
            {cameras.map(c => (
              <Stack key={c.key} gap="xs">
                <Title order={3}>{c.name}</Title>
                <Hours name={`Walk-bys, ${c.name}`} days={c.days} unit="walk-bys" />
              </Stack>
            ))}
            <Pending pending={data.pending} />
          </Stack>
        );
      }}
    </ReportState>
  );
}

type PersonSeries = PersonActivity['people'][number]['series'][number];

/** One activity's minutes in each hour of the day, summed over the days read; null where no day was checked. The
 * link opens the latest day that had any. */
function byHour(s: PersonSeries, h: number): { minutes: number | null; day: string | null; t: number | null } {
  let minutes: number | null = null;
  let day: string | null = null;
  let t: number | null = null;
  for (const d of s.days) {
    const cell = d.hours[h];
    if (!cell) continue;
    minutes = (minutes ?? 0) + cell.minutes;
    if (cell.minutes > 0 && (day === null || d.day > day)) {
      day = d.day;
      t = cell.t;
    }
  }
  return { minutes, day, t };
}

/** "Not confirmed yet: Reading and Eating." for the kinds the owner hasn't confirmed. */
function unconfirmedLine(names: string[]): string {
  const list =
    names.length < 3 ? names.join(' and ') : `${names.slice(0, -1).join(', ')}, and ${names[names.length - 1]}`;
  return `Spans the model found. Not confirmed yet: ${list}.`;
}

/** One table per person: a row per activity across the hours of the day, with its time in all, one legend
 * and one caveat for the whole report, and each activity's day-by-day grid behind "By day". */
export function PersonActivityReport({
  arg,
  pinned = [],
  onPinned,
}: {
  arg: string | null;
  pinned?: string[];
  onPinned?: (people: string[]) => void;
}) {
  const query = useReport<PersonActivity>('person_activity', arg);
  // Activity by person: one person at a time, from the figures already here, starting on the
  // first pinned one (else the first); everyone is one click away. undefined: not picked yet; null: everyone.
  const [picked, setPerson] = useState<string | null | undefined>(undefined);
  return (
    <ReportState query={query}>
      {loaded => {
        const person =
          picked === undefined && !arg
            ? firstShown(
                pinned,
                loaded.people.map(p => p.id),
              )
            : (picked ?? null);
        const data = person ? { ...loaded, people: loaded.people.filter(p => p.id === person) } : loaded;
        const unconfirmed = [
          ...new Set(data.people.flatMap(p => p.series.filter(s => !s.verified).map(s => activityName(s.name)))),
        ];
        return (
          <Stack gap="lg">
            {!data.people.length && <Text c="dimmed">No person has model activity spans in the days read so far.</Text>}
            {!arg && loaded.people.length > 1 && (
              <PinFilter
                choices={loaded.people.map(p => ({ id: p.id, name: p.name }))}
                pinned={pinned}
                value={person}
                onChange={setPerson}
                onPinned={onPinned}
                words={PEOPLE}
              />
            )}
            {data.people.length > 0 && (
              <Stack gap={4}>
                <RampLegend low="Less time" high="More" zero="Checked, none" gap="Not checked" />
                {unconfirmed.length > 0 && (
                  <Text size="sm" c="dimmed">
                    {unconfirmedLine(unconfirmed)}
                  </Text>
                )}
              </Stack>
            )}
            {data.people.map(p => (
              <Stack key={p.id} gap="xs">
                {/* The card already names a single person ("Dana's activity"); a name only tells several apart. */}
                {data.people.length > 1 && <Title order={3}>{p.name}</Title>}
                <Heatmap
                  label={`${p.name}'s activities by hour of day`}
                  rows={p.series.map(s => `${activityName(s.name)} · ${minutesText(s.total_min)}`)}
                  cols={HOURS}
                  colTicks={TICKS}
                  cellHeight={22}
                  minCellWidth={7}
                  maxCellWidth={36}
                  rowLabelWidth={170}
                  xTitle="Hour of day (links open the latest day)"
                  yTitle=""
                  cell={(r, h) => {
                    const s = p.series[r];
                    const c = s ? byHour(s, h) : { minutes: null, day: null, t: null };
                    const what = `${activityName(s?.name ?? '')}, ${hourName(h)}`;
                    return {
                      value: c.minutes,
                      label: c.minutes === null ? `${what}: not checked` : `${what}: ${minutesText(c.minutes)}`,
                      href: c.minutes && c.day ? reviewAt(c.day, c.t) : null,
                    };
                  }}
                />
                <Disclosure summary="By day" className={classes.byDay} lazy>
                  <Stack gap="md" pt="xs">
                    {p.series.map(s => (
                      <Stack key={s.key} gap={4}>
                        <Text size="sm" fw={600}>
                          {activityName(s.name)} · {minutesText(s.total_min)}{' '}
                          {!s.verified && (
                            <Badge size="xs" variant="light" color="gray">
                              Unconfirmed
                            </Badge>
                          )}
                        </Text>
                        <Hours
                          name={`${p.name}, ${activityName(s.name)}`}
                          unit="minutes"
                          amount={minutesText}
                          legend={false}
                          days={s.days.map(d => ({
                            day: d.day,
                            n: d.minutes,
                            t: d.t,
                            hours: d.hours.map(h => (h === null ? null : { n: h.minutes, t: h.t })),
                          }))}
                        />
                      </Stack>
                    ))}
                  </Stack>
                </Disclosure>
              </Stack>
            ))}
            <Pending pending={data.pending} />
          </Stack>
        );
      }}
    </ReportState>
  );
}

export function DeviceReport({ arg }: { arg: string | null }) {
  const query = useReport<Devices>('devices', arg);
  return (
    <ReportState query={query}>
      {data => (
        <Stack gap="lg">
          <Text size="sm" c="dimmed">
            {data.caveat}
          </Text>
          {!data.sources.length && <Text c="dimmed">No devices were observed in the days read so far.</Text>}
          {data.sources.map(s => (
            <Stack key={s.key} gap="xs">
              <Title order={3}>{s.name}</Title>
              <Hours name={s.name} days={s.days} unit="device identifiers" />
            </Stack>
          ))}
          <Pending pending={data.pending} />
        </Stack>
      )}
    </ReportState>
  );
}
