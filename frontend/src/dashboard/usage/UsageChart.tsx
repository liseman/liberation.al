// The Resource monitor's chart (in UsagePanel.tsx): a stacked area over the last half hour, one band per busy service
// and one for everything else, for the processor, memory, or one GPU's memory. SVG drawn by React with D3's scales and
// stack, like Statistics. The hover (or arrow keys on the chart) shows every service's value at that moment; a table under
// it gives the newest values, so nothing depends on telling colours apart.
import { SegmentedControl, Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { useQuery } from '@tanstack/react-query';
import { area, bisector, scaleLinear, scaleTime, stack, timeFormat } from 'd3';
import { type KeyboardEvent, type PointerEvent, type ReactNode, useId, useRef, useState } from 'react';
import { messageOf } from '../../shared/http';
import { PageError, PageLoading } from '../lib/page';
import { getUsageHistory, type UsageSample } from '../sources/api';
import classes from './usageChart.module.css';
import { bands, breakdown, merge, nameOf, OTHER, type Resource, type Row, resources, rows, utilOf } from './usageModel';

export const USAGE_MS = 5000; // the server keeps a sample every 5 s
const HEIGHT = 200;
const M = { top: 8, right: 8, bottom: 22, left: 56 };
const clock = timeFormat('%H:%M');
const moment = timeFormat('%H:%M:%S');
const SLOTS: (string | undefined)[] = [classes.c0, classes.c1, classes.c2, classes.c3, classes.c4, classes.c5];

function useUsage() {
  const held = useRef<UsageSample[]>([]);
  return useQuery({
    queryKey: ['stats-history'],
    queryFn: async ({ signal }) => {
      const since = held.current.at(-1)?.ts ?? 0;
      const h = await getUsageHistory(since, { signal });
      held.current = merge(held.current, h.samples, h.window);
      return { ...h, samples: held.current };
    },
    refetchInterval: USAGE_MS,
  });
}

function Swatch({ slot }: { slot: string }) {
  return <span className={`${classes.swatch} ${slot}`} aria-hidden />;
}

function Chart({
  data,
  resource,
  names,
  slotOf,
  step,
  window,
}: {
  data: Row[];
  resource: Resource;
  names: string[];
  slotOf: (name: string) => string;
  step: number;
  window: number;
}) {
  const { ref, width } = useElementSize();
  const [at, setAt] = useState<number | null>(null); // the hovered row's index
  const tip = useId();
  const w = Math.max(240, width);
  const newest = data.at(-1)?.ts ?? Date.now() / 1000;
  const x = scaleTime()
    .domain([new Date((newest - window) * 1000), new Date(newest * 1000)])
    .range([M.left, w - M.right]);
  const y = scaleLinear()
    .domain([0, resource.total])
    .range([HEIGHT - M.bottom, M.top]);
  const keys = [...names, OTHER];
  const layers = stack<Row>()
    .keys(keys)
    .value((d, k) => d.values[k] ?? 0)(data);
  const shape = area<[number, number] & { data: Row }>()
    .defined(d => !d.data.gap)
    .x(d => x(new Date(d.data.ts * 1000)))
    .y0(d => y(d[0]))
    .y1(d => y(d[1]));
  const real = data.flatMap((r, i) => (r.gap ? [] : [i]));
  const pick = (px: number) => {
    const t = x.invert(px).getTime() / 1000;
    const i = bisector((r: Row) => r.ts).center(data, t);
    const near = real.reduce((best, j) => (Math.abs(j - i) < Math.abs(best - i) ? j : best), real[0] ?? -1);
    const found = data[near];
    setAt(found && Math.abs(found.ts - t) <= step * 2 ? near : null);
  };
  const onPointer = (e: PointerEvent<HTMLDivElement>) => pick(e.clientX - e.currentTarget.getBoundingClientRect().left);
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!real.length) return;
    const k = at == null ? real.length - 1 : real.indexOf(at);
    if (e.key === 'ArrowLeft') setAt(real[Math.max(0, k - 1)] ?? null);
    else if (e.key === 'ArrowRight') setAt(real[Math.min(real.length - 1, k + 1)] ?? null);
    else if (e.key === 'Home') setAt(real[0] ?? null);
    else if (e.key === 'End') setAt(real.at(-1) ?? null);
    else if (e.key === 'Escape') setAt(null);
    else return;
    e.preventDefault();
  };
  const row = at == null ? null : (data[at] ?? null);
  const px = row ? x(new Date(row.ts * 1000)) : 0;
  const ticks = scaleLinear()
    .domain([0, resource.total / resource.unit])
    .ticks(4)
    .map(t => t * resource.unit);
  const util = row?.sample ? utilOf(row.sample, resource.key) : null;
  return (
    <div ref={ref} className={classes.frame}>
      <div
        className={classes.plot}
        role="slider"
        aria-label={`${resource.title} over the last ${Math.round(window / 60)} minutes, stacked by service`}
        aria-orientation="horizontal"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, real.length - 1)}
        aria-valuenow={Math.max(0, at == null ? real.length - 1 : real.indexOf(at))}
        aria-valuetext={
          row?.sample
            ? `${moment(new Date(row.ts * 1000))}: ${breakdown(row.sample, resource.key)
                .slice(0, 3)
                .map(b => `${nameOf(b.name)} ${resource.format(b.value)}`)
                .join(', ')}`
            : 'Use the arrow keys to read each moment'
        }
        aria-describedby={row ? tip : undefined}
        tabIndex={0}
        onPointerMove={onPointer}
        onPointerLeave={() => setAt(null)}
        onKeyDown={onKey}
        onBlur={() => setAt(null)}
      >
        <svg className={classes.svg} width={w} height={HEIGHT} aria-hidden>
          {ticks.map(t => (
            <g key={t}>
              <line className={classes.grid} x1={M.left} x2={w - M.right} y1={y(t)} y2={y(t)} />
              <text className={classes.tick} x={M.left - 6} y={y(t)} dy="0.32em" textAnchor="end">
                {resource.format(t)}
              </text>
            </g>
          ))}
          {layers.map(layer => (
            <path
              key={layer.key}
              className={`${classes.band} ${slotOf(layer.key)}`}
              d={shape(layer as unknown as ([number, number] & { data: Row })[]) ?? undefined}
            />
          ))}
          {x
            .ticks(Math.max(2, Math.floor(w / 110)))
            .filter(t => x(t) > M.left + 16 && x(t) < w - M.right - 16) // a label that fits inside the plot
            .map(t => (
              <text key={t.getTime()} className={classes.tick} x={x(t)} y={HEIGHT - 6} textAnchor="middle">
                {clock(t)}
              </text>
            ))}
          {row && <line className={classes.cross} x1={px} x2={px} y1={M.top} y2={HEIGHT - M.bottom} />}
        </svg>
      </div>
      {row?.sample && (
        <div
          id={tip}
          role="tooltip"
          className={classes.tip}
          style={px > w / 2 ? { right: w - px + 12 } : { left: px + 12 }}
        >
          <b>{moment(new Date(row.ts * 1000))}</b>
          {util != null && <div className={classes.dim}>GPU busy {util}%</div>}
          <ul className={classes.tipList}>
            {breakdown(row.sample, resource.key).map(({ name, value }) => (
              <li key={name}>
                <Swatch slot={slotOf(name)} />
                <span className={classes.tipName}>{nameOf(name)}</span>
                <span className={classes.value}>{resource.format(value)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** The chart and its toggle, legend, and newest values; it polls /stats/history only while it's mounted. */
export function UsageChart() {
  const usage = useUsage();
  const [choice, setChoice] = useState('cpu');
  const h = usage.data;
  let body: ReactNode;
  if (!h)
    body = usage.error ? (
      <PageError
        title="Couldn't load resource use"
        detail={messageOf(usage.error)}
        onRetry={() => void usage.refetch()}
        busy={usage.isFetching}
      />
    ) : (
      <PageLoading label="resource use" lines={4} />
    );
  else {
    const all = resources(h);
    const resource = all.find(r => r.key === choice) ?? all[0];
    const names = bands(h.samples, all);
    const slotOf = (name: string): string =>
      (name === OTHER ? classes.other : SLOTS[names.indexOf(name)]) ?? classes.other ?? '';
    const newest = h.samples.at(-1);
    body = (
      <div className={classes.root}>
        {all.length > 1 && (
          <SegmentedControl
            size="xs"
            className={classes.toggle}
            aria-label="What to show"
            value={resource.key}
            onChange={setChoice}
            data={all.map(r => ({ value: r.key, label: r.label }))}
          />
        )}
        <ul className={classes.legend} aria-label="Services">
          {[...names, OTHER].map(n => (
            <li key={n}>
              <Swatch slot={slotOf(n)} />
              {nameOf(n)}
            </li>
          ))}
        </ul>
        {h.samples.length < 2 ? (
          <Text size="sm" c="dimmed">
            Gathering a few samples. The chart fills in every {h.step} seconds while this panel is open.
          </Text>
        ) : (
          <Chart
            data={rows(h.samples, resource.key, names, h.step)}
            resource={resource}
            names={names}
            slotOf={slotOf}
            step={h.step}
            window={h.window}
          />
        )}
        {newest && (
          <table className={classes.now}>
            <caption>{resource.title}, now</caption>
            <tbody>
              {breakdown(newest, resource.key).map(({ name, value }) => (
                <tr key={name}>
                  <th scope="row">
                    <Swatch slot={slotOf(name)} />
                    {nameOf(name)}
                  </th>
                  <td>{resource.format(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    );
  }
  return body;
}
