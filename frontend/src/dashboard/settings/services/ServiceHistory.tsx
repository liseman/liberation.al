// Service uptime: D3 time scales, recorded spans, explicit unknowns, and one accessible detail target per service.

import { Badge, Popover, SegmentedControl, Text, UnstyledButton } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { scaleTime, timeFormat } from 'd3';
import { useId, useState } from 'react';
import { messageOf } from '../../../shared/http';
import { secondsText } from '../../lib/format';
import { PageError, PageLoading } from '../../lib/page';
import { SettingsSection } from '../layout';
import { serviceHistoryQuery } from './api';
import {
  type HistoryView,
  historyOrder,
  isHistoryFailure,
  observedUptime,
  type Service,
  type ServiceHistory,
} from './model';
import classes from './serviceHistory.module.css';

const clock = timeFormat('%a %-d %b, %H:%M'),
  tickClock = timeFormat('%H:%M'),
  tickDay = timeFormat('%a %-d');
const when = (ts: number) => clock(new Date(ts * 1000));
const STATE = { up: 'Running', stopped: 'Stopped', unknown: 'No recorded history' };
const EVENT = {
  started: 'Started',
  stopped: 'Stopped',
  restart: 'Restart scheduled',
  crashed: 'Crashed',
  'start-failed': 'Start failed',
};
export const duration = secondsText;
interface Detail {
  at: number;
  title: string;
  when: string;
  body: string;
  run?: string;
}
export function historyDetails(row: ServiceHistory): Detail[] {
  return [
    ...row.spans.map(s => ({
      at: s.start,
      title: STATE[s.state],
      when: `${when(s.start)} – ${when(s.end)}`,
      body: `${duration(s.end - s.start)}${s.state === 'unknown' ? '; availability was not recorded' : ''}`,
    })),
    ...row.events.map(e => {
      const span = row.spans.find(
        s => s.state === 'up' && (e.kind === 'crashed' ? s.end === e.at : e.kind === 'started' && s.start === e.at),
      );
      return {
        at: e.at,
        title: EVENT[e.kind],
        when: when(e.at),
        run: span ? `Observed run: ${duration(span.end - span.start)}` : undefined,
        body:
          e.reason ||
          (e.kind === 'crashed'
            ? 'Exit reason was not recorded.'
            : e.kind === 'start-failed'
              ? 'Start failure reason was not recorded.'
              : 'Recorded service transition.'),
      };
    }),
  ].sort((a, b) => a.at - b.at);
}
function Timeline({ row, data }: { row: ServiceHistory; data: HistoryView }) {
  const id = useId().replaceAll(':', ''),
    details = historyDetails(row);
  const [index, setIndex] = useState(Math.max(0, details.length - 1)),
    [opened, setOpened] = useState(false),
    [pinned, setPinned] = useState(false);
  const picked = details[Math.min(index, details.length - 1)];
  const x = scaleTime()
    .domain([new Date(data.start * 1000), new Date(data.end * 1000)])
    .range([0, 1000])
    .clamp(true);
  const at = (ts: number) => x(new Date(ts * 1000));
  function point(clientX: number, el: HTMLElement) {
    const rect = el.getBoundingClientRect(),
      time =
        data.start +
        Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width))) * (data.end - data.start);
    const event = row.events.reduce<(typeof row.events)[number] | null>(
      (best, e) => (Math.abs(e.at - time) < Math.abs((best?.at ?? Infinity) - time) ? e : best),
      null,
    );
    const near = event && Math.abs(event.at - time) < ((data.end - data.start) * 8) / Math.max(1, rect.width),
      span = row.spans.find(s => s.start <= time && time <= s.end);
    const next = near
      ? details.findIndex(d => d.at === event.at && d.title === EVENT[event.kind])
      : details.findIndex(d => span && d.at === span.start && d.title === STATE[span.state]);
    if (next >= 0) setIndex(next);
  }
  return (
    <Popover opened={opened} onChange={setOpened} position="top" withArrow shadow="sm" width={280} withinPortal>
      <Popover.Target>
        <UnstyledButton
          className={classes.plot}
          aria-label={`${row.name} uptime history`}
          onMouseEnter={() => setOpened(true)}
          onMouseMove={e => {
            if (!pinned) point(e.clientX, e.currentTarget);
          }}
          onMouseLeave={() => {
            if (!pinned) setOpened(false);
          }}
          onFocus={() => setOpened(true)}
          onBlur={() => {
            setPinned(false);
            setOpened(false);
          }}
          onClick={e => {
            if (e.detail > 0) point(e.clientX, e.currentTarget);
            setPinned(!pinned);
            setOpened(!pinned);
          }}
          onKeyDown={e => {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
              e.preventDefault();
              setIndex(i => Math.max(0, Math.min(details.length - 1, i + (e.key === 'ArrowLeft' ? -1 : 1))));
              setOpened(true);
            }
            if (e.key === 'Escape') {
              setPinned(false);
              setOpened(false);
            }
          }}
        >
          <svg viewBox="0 0 1000 44" preserveAspectRatio="none" aria-hidden>
            <defs>
              <pattern id={id} width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">
                <rect width="8" height="8" fill="var(--lib-track)" />
                <path d="M 0 0 V 8" stroke="var(--mantine-color-dimmed)" opacity=".32" strokeWidth="2" />
              </pattern>
            </defs>
            <rect x="0" y="13" width="1000" height="18" rx="4" fill="var(--lib-track)" />
            {row.spans.map(s => (
              <rect
                key={`${s.start}:${s.state}`}
                x={at(s.start)}
                y="13"
                width={Math.max(0, at(s.end) - at(s.start))}
                height="18"
                fill={
                  s.state === 'up'
                    ? 'var(--lib-ok)'
                    : s.state === 'unknown'
                      ? `url(#${id})`
                      : 'var(--mantine-color-body)'
                }
                opacity={s.state === 'up' ? '.86' : '1'}
              />
            ))}
            {row.events
              .filter(e => e.kind !== 'stopped')
              .map(e =>
                isHistoryFailure(e) ? (
                  <path
                    key={`${e.at}:${e.kind}:${e.reason ?? ''}`}
                    d={`M${at(e.at)},7 l6,7 -6,7 -6,-7 Z`}
                    fill="var(--lib-bad)"
                  />
                ) : (
                  <line
                    key={`${e.at}:${e.kind}:${e.reason ?? ''}`}
                    x1={at(e.at)}
                    x2={at(e.at)}
                    y1="8"
                    y2="36"
                    stroke="var(--lib-warn)"
                    strokeWidth="2"
                  />
                ),
              )}
            {opened && picked && (
              <line
                x1={at(picked.at)}
                x2={at(picked.at)}
                y1="2"
                y2="42"
                stroke="var(--mantine-primary-color-filled)"
                strokeWidth="2"
                strokeDasharray="3 3"
              />
            )}
          </svg>
        </UnstyledButton>
      </Popover.Target>
      <Popover.Dropdown className={classes.detail}>
        {picked ? (
          <>
            <Text fw={600} size="sm">
              {row.name}: {picked.title}
            </Text>
            <Text size="xs" c="dimmed">
              {picked.when}
            </Text>
            <Text size="sm">{picked.body}</Text>
            {picked.run && (
              <Text size="xs" c="dimmed">
                {picked.run}
              </Text>
            )}
          </>
        ) : (
          <Text size="sm">No recorded history.</Text>
        )}
      </Popover.Dropdown>
    </Popover>
  );
}
export function ServiceHistoryPanel({ services }: { services: readonly Service[] }) {
  const [hours, setHours] = useState<24 | 168>(24),
    query = useQuery(serviceHistoryQuery(hours, services.length > 0)),
    data = query.data;
  if (!services.length) return null;
  const x =
      data &&
      scaleTime()
        .domain([new Date(data.start * 1000), new Date(data.end * 1000)])
        .range([0, 100]),
    ticks = x?.ticks(4) ?? [];
  return (
    <SettingsSection
      title="Service uptime"
      description="Running time, restarts, and failures. Trouble first."
      trailing={
        <SegmentedControl
          size="xs"
          value={String(hours)}
          onChange={v => setHours(v === '168' ? 168 : 24)}
          data={[
            { label: '24 hours', value: '24' },
            { label: '7 days', value: '168' },
          ]}
          aria-label="Service history window"
        />
      }
    >
      <div className={classes.body}>
        {query.isPending && <PageLoading label="service history" lines={4} />}
        {query.isError && (
          <PageError
            title="Couldn't read service history"
            detail={messageOf(query.error)}
            onRetry={() => void query.refetch()}
            busy={query.isFetching}
          />
        )}
        {data && (
          <>
            <div className={classes.key}>
              <span>
                <i className={classes.up} />
                Running
              </span>
              <span>
                <i className={classes.restart} />
                Start/restart
              </span>
              <span>
                <i className={classes.failure} />
                Failure
              </span>
              <span>
                <i className={classes.stopped} />
                Stopped
              </span>
              <span>
                <i className={classes.unknown} />
                Unknown
              </span>
              <Badge variant="light" color="gray" size="sm">
                {data.source === 'journal'
                  ? 'Recorded events'
                  : data.source === 'sampled'
                    ? 'Five-minute samples'
                    : 'History unavailable'}
              </Badge>
            </div>
            {data.note && (
              <Text size="xs" c={data.available ? 'dimmed' : 'var(--lib-warn)'} role="status">
                {data.note}
              </Text>
            )}
            <div className={classes.axis} aria-hidden>
              {ticks.map(t => (
                <span key={t.toISOString()} style={{ left: `${x?.(t) ?? 0}%` }}>
                  {hours === 24 ? tickClock(t) : tickDay(t)}
                </span>
              ))}
            </div>
            <ul className={classes.rows} aria-label="Service uptime history">
              {historyOrder(data.services, services).map(row => {
                const pct = observedUptime(row),
                  failures = row.events.filter(isHistoryFailure).length,
                  restarts = row.events.filter(e => e.kind === 'restart').length;
                return (
                  <li key={row.name} className={classes.row} aria-label={`${row.name} history`}>
                    <div className={classes.name}>
                      <b>{row.name}</b>
                      <span>
                        {pct === null ? 'Not observed' : `${Math.round(pct)}% of observed time`}
                        {failures > 0 && ` · ${failures} ${failures === 1 ? 'failure' : 'failures'}`}
                        {restarts > 0 &&
                          ` · ${restarts} ${restarts === 1 ? 'automatic restart' : 'automatic restarts'}`}
                      </span>
                    </div>
                    <Timeline row={row} data={data} />
                  </li>
                );
              })}
            </ul>
            <Text size="xs" c="dimmed" className={classes.hint}>
              Hover, focus, or tap for details. Arrow keys move through recorded intervals.
            </Text>
          </>
        )}
      </div>
    </SettingsSection>
  );
}
