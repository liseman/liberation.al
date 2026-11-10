// System health: storage held per recorded day against the plan (docs/STORAGE.md), memory and GPU use, and service
// uptime. These marks stand for the machine, not for moments, so they don't link to Review.
// Everything but storage comes from samples the dashboard takes while it runs; hours without one are gaps.
import { Group, Progress, SimpleGrid, Stack, Text, Title } from '@mantine/core';
import { IconCircleCheck, IconCircleX } from '@tabler/icons-react';
import { timeFormat } from 'd3';
import { Disclosure } from '../lib/Disclosure';
import type { Health as Data, HealthHour } from './api';
import { useReport } from './api';
import { Bars, dayName, dayTicks, GapKey, LineKey, midnightTicks, shortDayName } from './charts';
import classes from './reports.module.css';
import { ReportState } from './state';

const hourLabel = timeFormat('%a %b %-d, %-I %p');
const tickLabel = timeFormat('%b %-d');
const when = (t: number) => hourLabel(new Date(t * 1000));

export function finding(data: Data): string {
  const parts: string[] = [];
  const s = data.storage;
  if (s?.days.length) {
    const over = s.media_gb > s.plan_gb;
    parts.push(
      `Recordings take ${s.media_gb} GB, ${over ? 'over' : 'within'} the ${s.plan_gb} GB the plan allows for ${s.days.length === 1 ? 'this day' : `these ${s.days.length} days`}.`,
    );
  }
  const down = data.services.filter(x => !x.up_now);
  if (data.services.length)
    parts.push(
      down.length
        ? `${down.map(x => serviceName(x.name)).join(', ')} ${down.length === 1 ? 'is' : 'are'} down.`
        : 'Every service is up.',
    );
  return parts.join(' ') || 'No measurements yet. The dashboard takes one every few minutes.';
}

/** A measure as one small line: where it is now, its range, and its shape, breaking where an hour wasn't measured. */
export function Spark({ data, pick, title }: { data: Data; pick: (h: HealthHour) => number | null; title: string }) {
  const values = data.hours.map(pick);
  const known = values.filter((v): v is number => v !== null);
  const now = [...values].reverse().find((v): v is number => v !== null);
  if (now === undefined) return null;
  const low = Math.min(...known);
  const high = Math.max(...known);
  const W = 160;
  const H = 32;
  const x = (i: number) => (values.length < 2 ? W : (i / (values.length - 1)) * W);
  const y = (v: number) => H - 2 - (v / 100) * (H - 4); // percentages, on 0 to 100 so tiles compare
  const runs: string[] = [];
  let run: string[] = [];
  values.forEach((v, i) => {
    if (v === null) {
      if (run.length) runs.push(run.join(' '));
      run = [];
    } else run.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  });
  if (run.length) runs.push(run.join(' '));
  return (
    <div className={classes.spark}>
      <Title order={3}>{title}</Title>
      <Group gap={8} align="baseline" wrap="nowrap">
        <Text className={classes.sparkNow}>{now}%</Text>
        <Text size="xs" c="dimmed">
          {low === high ? 'steady' : `${low} to ${high}%`}
        </Text>
      </Group>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className={classes.sparkLine}
        role="img"
        aria-label={`${title}: ${now}% now, ${low === high ? 'steady' : `from ${low} to ${high}%`} since ${when(data.hours[0]?.t ?? 0)}`}
      >
        {runs.map(points => (
          <polyline key={points.slice(0, 24)} points={points} />
        ))}
      </svg>
    </div>
  );
}

function Hourly({
  data,
  pick,
  title,
  unit,
}: {
  data: Data;
  pick: (h: HealthHour) => number | null;
  title: string;
  unit: string;
}) {
  const ts = data.hours.map(h => h.t);
  return (
    <Stack gap={4}>
      <Title order={3}>{title}</Title>
      <Bars
        label={`${title}, each hour of the last two weeks`}
        points={data.hours.map(h => {
          const v = pick(h);
          return {
            key: String(h.t),
            value: v,
            label: v === null ? `${when(h.t)}: not measured` : `${when(h.t)}: ${v}${unit}`,
          };
        })}
        ticks={midnightTicks(ts)}
        tickLabel={i => tickLabel(new Date((ts[i] ?? 0) * 1000))}
        xTitle="Hour"
        yTitle={unit.trim() || 'Count'}
        height={100}
      />
    </Stack>
  );
}

/** Services by the job they do; one this doesn't know keeps its own name. */
const PLAIN: Record<string, string> = {
  'see-decode': 'Video decoding',
  'see-actions': 'Action detection',
  'see-zones': 'Zones',
  'see-objects': 'Object detection',
  'see-captions': 'Camera captions',
  'continuum-tier-worker': 'Storage tiers',
  'continuum-vector-worker': 'Search vectors',
  'continuum-broker': 'Recording broker',
  'continuum-capture': 'Camera capture',
  'continuum-gateway': 'Video gateway',
  'ask-ollama': 'Ask (local model)',
  'deep-transcribe': 'Careful transcription',
  identity: 'Face and voice matching',
  ingest: 'Imports',
  'ingest-analyze': 'Import analysis',
  journal: 'Journal',
  listen: 'Listening',
  'network-devices': 'Network devices',
  research: 'Research',
  'video-window': 'Recent video',
  voice: 'Voice',
  dashboard: 'Dashboard',
};
export const serviceName = (name: string): string => PLAIN[name] ?? name;

/** How many services show before the rest fold into one line. */
const SHOWN = 4;

/** Every service in one aligned list: down first, then the least up; the rest of the healthy ones folded. */
function Services({ services }: { services: Data['services'] }) {
  const sorted = [...services].sort((a, b) => Number(a.up_now) - Number(b.up_now) || a.uptime_pct - b.uptime_pct);
  const down = sorted.filter(x => !x.up_now).length;
  const head = sorted.slice(0, Math.max(SHOWN, down));
  const rest = sorted.slice(head.length);
  const row = (x: Data['services'][number]) => (
    <li key={x.name} className={classes.serviceRow} data-down={x.up_now ? undefined : ''}>
      <span className={classes.serviceName}>
        {x.up_now ? (
          <IconCircleCheck size={14} color="var(--lib-ok)" aria-hidden="true" />
        ) : (
          <IconCircleX size={14} color="var(--lib-bad)" aria-hidden="true" />
        )}
        <span title={x.name}>{serviceName(x.name)}</span>
        {!x.up_now && <span className={classes.downTag}>down</span>}
      </span>
      <span className={classes.servicePct}>{x.uptime_pct}%</span>
    </li>
  );
  const low = rest.length ? Math.min(...rest.map(x => x.uptime_pct)) : 0;
  return (
    <>
      <ul className={classes.services} aria-label="Services, last 30 days">
        {head.map(row)}
      </ul>
      {rest.length > 0 && (
        <Disclosure
          summary={`${rest.length} more, all up, at ${low === 100 ? '100%' : `${low}% to 100%`}`}
          className={classes.moreServices}
        >
          <ul className={classes.services} aria-label="More services">
            {rest.map(row)}
          </ul>
        </Disclosure>
      )}
    </>
  );
}

/** Storage as one meter against what the plan allows for the days held. */
function StorageMeter({ storage }: { storage: NonNullable<Data['storage']> }) {
  const pct = storage.plan_gb > 0 ? (100 * storage.media_gb) / storage.plan_gb : 0;
  const over = storage.media_gb > storage.plan_gb;
  return (
    <Stack gap={4}>
      <Group justify="space-between" wrap="wrap" gap="xs">
        <Text size="sm" fw={600}>
          Recordings
        </Text>
        <Text size="sm" c={over ? 'var(--lib-bad)' : 'dimmed'}>
          {storage.media_gb} of {storage.plan_gb} GB planned{over ? ', over the plan' : ''}
        </Text>
      </Group>
      <Progress
        value={Math.min(100, pct)}
        color={over ? 'red' : undefined}
        aria-label={`Recordings use ${Math.round(pct)}% of the planned storage`}
      />
    </Stack>
  );
}

export function Health() {
  const query = useReport<Data>('health', null);
  return (
    <ReportState query={query}>
      {data => {
        const s = data.storage;
        const days = s?.days.map(d => d.day) ?? [];
        // The hourly charts start at the first measurement, not 30 days back, so a new install isn't all hatching.
        const first = data.hours.findIndex(h => h.mem_pct !== null || h.gpu_mem_pct !== null || h.gpu_util !== null);
        const measured = { ...data, hours: first < 0 ? [] : data.hours.slice(first) };
        const gpu = measured.hours.some(h => h.gpu_mem_pct !== null);
        const down = data.services.filter(x => !x.up_now);
        return (
          <Stack gap="lg">
            {/* Problems first: what is down, in words and colour, or that everything is up. */}
            {data.services.length > 0 &&
              (down.length ? (
                <Text className={classes.healthBad} role="status">
                  {down.length === 1 ? '1 service is down' : `${down.length} services are down`}:{' '}
                  {down.map(x => serviceName(x.name)).join(', ')}
                </Text>
              ) : (
                <Text className={classes.healthOk}>All {data.services.length} services up</Text>
              ))}
            <Text className={classes.findings}>{finding(data)}</Text>
            {s?.days.length ? (
              <Stack gap="xs">
                <StorageMeter storage={s} />
                <Disclosure summary="Storage held for each recorded day" className={classes.moreServices} lazy>
                  <Stack gap="xs" pt="xs">
                    <LineKey>The plan: full quality for 14 days, then aged</LineKey>
                    <Bars
                      label="Gigabytes of video and audio held for each recorded day, against the plan"
                      points={s.days.map(d => ({
                        key: d.day,
                        value: d.gb,
                        label: `${dayName(d.day)}: ${d.gb} GB held (${d.tier2_gb} GB recompressed), plan ${d.plan_gb} GB`,
                      }))}
                      plan={s.days.map(d => d.plan_gb)}
                      ticks={dayTicks(days)}
                      tickLabel={i => shortDayName(days[i] ?? '')}
                      xTitle="Day recorded"
                      yTitle="GB"
                    />
                    <Text size="xs" c="dimmed">
                      Also the catalog, {s.catalog_gb} GB, and search vectors, {s.vectors_gb} GB. Measured {when(s.at)}.
                    </Text>
                  </Stack>
                </Disclosure>
              </Stack>
            ) : (
              <Text size="sm" c="dimmed">
                Storage isn't measured yet. The dashboard measures it every half hour.
              </Text>
            )}
            {/* One row of small lines with where each is now (#2621 board, idea 7): steady charts don't need a page
                each. The hour-by-hour charts are a click away. */}
            {measured.hours.length > 0 && (
              <Stack gap="xs">
                <SimpleGrid cols={{ base: 1, xs: gpu ? 3 : 1 }} spacing="md">
                  <Spark data={measured} pick={h => h.mem_pct} title="Memory in use" />
                  {gpu && <Spark data={measured} pick={h => h.gpu_mem_pct} title="GPU memory in use (peak)" />}
                  {gpu && <Spark data={measured} pick={h => h.gpu_util} title="GPU busy" />}
                </SimpleGrid>
                <Disclosure summary="Hour by hour" className={classes.moreServices} lazy>
                  <Stack gap="lg" pt="xs">
                    <Group justify="flex-end">
                      <GapKey />
                    </Group>
                    <Hourly data={measured} pick={h => h.mem_pct} title="Memory in use" unit="%" />
                    {gpu && (
                      <Hourly data={measured} pick={h => h.gpu_mem_pct} title="GPU memory in use (peak)" unit="%" />
                    )}
                    {gpu && <Hourly data={measured} pick={h => h.gpu_util} title="GPU busy" unit="%" />}
                  </Stack>
                </Disclosure>
              </Stack>
            )}
            <Stack gap={6}>
              <Title order={3}>Services, last 30 days</Title>
              {data.services.length ? (
                <Services services={data.services} />
              ) : (
                <Text size="sm" c="dimmed">
                  No samples yet.
                </Text>
              )}
              {data.sampled_from && (
                <Text size="xs" c="dimmed">
                  Sampled every {Math.round(data.sample_s / 60)} minutes while the dashboard runs, since{' '}
                  {when(data.sampled_from)}.
                </Text>
              )}
            </Stack>
          </Stack>
        );
      }}
    </ReportState>
  );
}
