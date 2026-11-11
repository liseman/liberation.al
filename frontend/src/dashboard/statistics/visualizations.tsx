// D3 computes the clock arcs, exchange chords, and centered activity bands; React owns their SVG and links.
// Reuse the Statistics tooltip and roving focus so a shape works with a pointer, touch, or the keyboard.
import { Button, Group, Stack, Text } from '@mantine/core';
import {
  arc,
  area,
  type Chord,
  type ChordSubgroup,
  chord,
  ribbon,
  scaleLinear,
  stack,
  stackOffsetSilhouette,
} from 'd3';
import { type ReactNode, useState } from 'react';
import { eventName } from '../lib/activity';
import type { Conversations, Series, WhosAround } from './api';
import {
  dayName,
  dayTicks,
  follow,
  GapPattern,
  hourName,
  reviewAt,
  shortDayName,
  spaced,
  step,
  Tooltip,
  usePatternId,
  useRoving,
  useTip,
} from './charts';
import classes from './reports.module.css';

const RAMP = [classes.s0, classes.s1, classes.s2, classes.s3, classes.s4, classes.s5];

/** Sort by identity rather than counts; neighboring categories do not accidentally hash to the same color. */
function category(key: string, keys: string[]): string {
  return `var(--rep-cat-${[...keys].sort().indexOf(key) % 6})`;
}

type Mark = (key: string, label: string, href: string | null, shape: ReactNode) => ReactNode;

function Plot({
  label,
  height,
  keys,
  children,
  maxScale,
}: {
  label: string;
  height: number;
  keys: string[][];
  children: (mark: Mark, gap: string) => ReactNode;
  /** How far a wide card may enlarge the drawing: a flat chart (the event stream) otherwise grows with the card until
   *  its axis text is about twice the other charts'. */
  maxScale?: number;
}) {
  const gap = usePatternId();
  const [tip, setTip] = useTip();
  const focus = useRoving(keys);
  const show = (e: { clientX: number; clientY: number }, text: string) => setTip({ x: e.clientX, y: e.clientY, text });
  const mark: Mark = (key, text, href, shape) =>
    href ? (
      <a
        key={key}
        href={href}
        data-key={key}
        className={classes.mark}
        tabIndex={focus.current === key ? 0 : -1}
        aria-label={`${text}. Open in Review`}
        onClick={e => follow(e, href)}
        onKeyDown={focus.onKeyDown}
        onMouseMove={e => show(e, text)}
        onMouseLeave={() => setTip(null)}
        onBlur={() => setTip(null)}
        onFocus={e => {
          focus.setActive(key);
          const box = e.currentTarget.getBoundingClientRect();
          show({ clientX: box.right, clientY: box.bottom }, text);
        }}
      >
        {shape}
      </a>
    ) : (
      // biome-ignore lint/a11y/noStaticElementInteractions: unlinked zero/gap details, also described in the chart caption
      <g key={key} onMouseMove={e => show(e, text)} onMouseLeave={() => setTip(null)}>
        <title>{text}</title>
        {shape}
      </g>
    );
  return (
    <figure className={classes.fluidChart} aria-label={label}>
      <svg
        ref={focus.root}
        viewBox={`0 0 360 ${height}`}
        style={maxScale ? { maxHeight: Math.round(height * maxScale) } : undefined}
      >
        <title>{label}</title>
        <GapPattern id={gap} />
        {children(mark, gap)}
      </svg>
      <Tooltip tip={tip} />
    </figure>
  );
}

const CLOCK_DAYS = 14;

/** Days are concentric rings, oldest inside, each split into the same 24 local-clock slots as the calendar. */
export function PresenceClock({ data, person }: { data: WhosAround; person: string | null }) {
  const [offset, setOffset] = useState(0);
  const end = Math.max(1, data.days.length - Math.min(offset, Math.max(0, data.days.length - 1)));
  const days = data.days.slice(Math.max(0, end - CLOCK_DAYS), end);
  const names = new Map(data.people.map(p => [p.id, p.name]));
  const value = (cell: WhosAround['days'][number]['hours'][number]) =>
    person ? (cell?.who.find(([id]) => id === person)?.[1] ?? 0) : (cell?.who.length ?? 0);
  const maximum = Math.max(1, ...days.flatMap(d => d.hours.map(value)));
  const ring = 92 / Math.max(1, days.length);
  const outer = 42 + days.length * ring;
  const keys = Array.from({ length: 24 }, (_, h) =>
    days.flatMap((d, i) => (d.hours[h]?.who.length ? [`${h}:${i}`] : [])),
  );
  const shape = arc();
  const paging = data.days.length > CLOCK_DAYS;
  return (
    <Stack gap={4}>
      <Plot
        label={`Who was around, by day and hour, ${dayName(days[0]?.day ?? data.first)} to ${dayName(days.at(-1)?.day ?? data.last)}`}
        height={330}
        keys={keys}
      >
        {(mark, gap) => (
          <g transform="translate(180 165)">
            {days.flatMap((d, i) =>
              Array.from({ length: 24 }, (_, h) => {
                const cell = d.hours[h] ?? null;
                const when = `${dayName(d.day)}, ${hourName(h)}`;
                const who =
                  cell?.who.map(([id, minutes]) => `${names.get(id) ?? 'Someone'} ${Math.round(minutes)} min`) ?? [];
                const label = `${when}: ${cell === null ? 'nothing recorded' : who.length ? who.join(', ') : 'nobody seen or heard'}`;
                return mark(
                  `${h}:${i}`,
                  label,
                  who.length && cell ? reviewAt(d.day, cell.t) : null,
                  <path
                    d={
                      shape({
                        innerRadius: 42 + i * ring,
                        outerRadius: 42 + (i + 1) * ring - 1,
                        startAngle: (h * Math.PI) / 12,
                        endAngle: ((h + 1) * Math.PI) / 12,
                        padAngle: 0.012,
                      }) ?? ''
                    }
                    className={
                      cell === null ? undefined : value(cell) > 0 ? RAMP[step(value(cell), maximum)] : classes.zero
                    }
                    fill={cell === null ? `url(#${gap})` : undefined}
                  />,
                );
              }),
            )}
            {[0, 3, 6, 9, 12, 15, 18, 21].map(h => {
              const a = (h * Math.PI) / 12;
              return (
                <text
                  key={h}
                  x={Math.sin(a) * (outer + 20)}
                  y={-Math.cos(a) * (outer + 20)}
                  className={classes.axis}
                  textAnchor="middle"
                  dominantBaseline="central"
                >
                  {hourName(h)}
                </text>
              );
            })}
            <text className={classes.clockCenter} textAnchor="middle" y={-3}>
              24 hours
            </text>
            <text className={classes.axis} textAnchor="middle" y={14}>
              {days.length} {days.length === 1 ? 'day' : 'days'}
            </text>
          </g>
        )}
      </Plot>
      {/* Paging only when the report has more days than the clock shows; the one that can't go
          further keeps its place, hidden, so the caption doesn't move. */}
      <Group justify={paging ? 'space-between' : 'center'} gap="xs" wrap="nowrap">
        {paging && (
          <Button
            size="compact-xs"
            variant="default"
            style={{ visibility: end <= CLOCK_DAYS ? 'hidden' : undefined, flexShrink: 0 }}
            onClick={() => setOffset(offset + CLOCK_DAYS)}
          >
            Older days
          </Button>
        )}
        <Text size="xs" c="dimmed" ta="center">
          Each ring is a day: {shortDayName(days[0]?.day ?? data.first)} in the middle,{' '}
          {shortDayName(days.at(-1)?.day ?? data.last)} outside
        </Text>
        {paging && (
          <Button
            size="compact-xs"
            variant="default"
            style={{ visibility: offset ? undefined : 'hidden', flexShrink: 0 }}
            onClick={() => setOffset(Math.max(0, offset - CLOCK_DAYS))}
          >
            Newer days
          </Button>
        )}
      </Group>
    </Stack>
  );
}

/** An undirected exchange contributes once to each endpoint; ribbons carry the original pair count. */
export function ConversationChord({ pairs }: { pairs: Conversations['pairs'] }) {
  const names = new Map<string, string>();
  for (const p of pairs) {
    names.set(p.a, p.a_name);
    names.set(p.b, p.b_name);
  }
  const ids = [...names.keys()].sort();
  const index = new Map(ids.map((id, i) => [id, i]));
  const matrix = ids.map(() => ids.map(() => 0));
  const byPair = new Map(pairs.map(p => [[p.a, p.b].sort().join('|'), p]));
  for (const p of pairs) {
    const a = index.get(p.a) ?? 0,
      b = index.get(p.b) ?? 0;
    const rowA = matrix[a],
      rowB = matrix[b];
    if (rowA && rowB) {
      rowA[b] = p.n;
      rowB[a] = p.n;
    }
  }
  const chords = chord().padAngle(0.08)(matrix);
  const outline = arc().innerRadius(113).outerRadius(125);
  const band = ribbon<Chord, ChordSubgroup>().radius(110);
  return (
    <Stack gap={6}>
      <Plot
        label="Conversation exchanges between people"
        height={290}
        keys={[chords.map(c => `${c.source.index}:${c.target.index}`)]}
      >
        {mark => (
          <g transform="translate(180 145)" className={classes.chord}>
            {chords.groups.map(g => (
              <path
                key={g.index}
                d={outline({ ...g, innerRadius: 113, outerRadius: 125 }) ?? ''}
                fill={category(ids[g.index] ?? '', ids)}
              >
                <title>
                  {names.get(ids[g.index] ?? '')}: {g.value} exchanges
                </title>
              </path>
            ))}
            {chords.map(c => {
              const a = ids[c.source.index] ?? '',
                b = ids[c.target.index] ?? '';
              const p = byPair.get([a, b].sort().join('|'));
              if (!p) return null;
              return mark(
                `${c.source.index}:${c.target.index}`,
                `${p.a_name} and ${p.b_name}: ${p.n} ${p.n === 1 ? 'exchange' : 'exchanges'}, latest ${dayName(p.day)}`,
                reviewAt(p.day, p.t),
                <path d={band(c) ?? ''} fill={category(a, ids)} className={classes.ribbon} />,
              );
            })}
          </g>
        )}
      </Plot>
      <Group gap="md" justify="center">
        {ids.map(id => (
          <Text key={id} size="xs">
            <span className={classes.swatch} style={{ background: category(id, ids) }} /> {names.get(id)}
          </Text>
        ))}
      </Group>
      <Text size="xs" c="dimmed">
        An exchange is two people's lines within a minute. Ribbon width shows exchanges; links open the latest day.
      </Text>
    </Stack>
  );
}

/** Centered daily totals, joined between checked days. A gap breaks the bands instead of inserting zeros. */
export function EventStream({ series }: { series: Series[] }) {
  const days = [...new Set(series.flatMap(s => s.days.map(d => d.day)))].sort();
  const colorKeys = series.map(s => s.key);
  const lookup = series.map(s => new Map(s.days.map(d => [d.day, d])));
  const rows = days.map(day => Object.fromEntries(series.map((s, i) => [s.key, lookup[i]?.get(day)?.n ?? 0])));
  const covered = days.map(day => lookup.every(m => m.get(day)?.n != null));
  const layers = stack<Record<string, number>>()
    .keys(series.map(s => s.key))
    .offset(stackOffsetSilhouette)(rows);
  const top = Math.max(1, ...rows.map(r => Object.values(r).reduce((a, b) => a + b, 0))) / 2;
  const x = scaleLinear()
    .domain([0, Math.max(1, days.length)])
    .range([30, 352]);
  const y = scaleLinear().domain([-top, top]).range([164, 12]);
  const ticks = spaced(dayTicks(days), i => x(i), 60);
  const keys = series.map((_, i) => days.flatMap((d, j) => (covered[j] && lookup[i]?.get(d)?.n ? [`${i}:${j}`] : [])));
  const path = area<{ x: number; lo: number; hi: number }>()
    .x(p => p.x)
    .y0(p => y(p.lo))
    .y1(p => y(p.hi));
  return (
    <Stack gap="xs">
      <Plot label="Event counts over time, centered bands by kind" height={202} keys={keys} maxScale={1.1}>
        {(mark, gap) => (
          <>
            <line x1={30} x2={352} y1={y(0)} y2={y(0)} className={classes.gridline} />
            {days.map((d, j) =>
              !covered[j]
                ? mark(
                    `gap:${j}`,
                    `${dayName(d)}: not looked for`,
                    null,
                    <rect x={x(j)} y={12} width={x(j + 1) - x(j)} height={152} fill={`url(#${gap})`} />,
                  )
                : null,
            )}
            {layers.flatMap((layer, i) =>
              layer.map((p, j) => {
                if (!covered[j]) return null;
                const d = days[j] ?? '',
                  s = series[i],
                  cell = lookup[i]?.get(d);
                if (!s || !cell) return null;
                return mark(
                  `${i}:${j}`,
                  `${dayName(d)}: ${eventName(s.name)} ${cell.n} times`,
                  cell.n ? reviewAt(d, cell.t) : null,
                  <path
                    d={
                      path([
                        {
                          x: x(j),
                          lo: (p[0] + (covered[j - 1] ? (layer[j - 1]?.[0] ?? p[0]) : p[0])) / 2,
                          hi: (p[1] + (covered[j - 1] ? (layer[j - 1]?.[1] ?? p[1]) : p[1])) / 2,
                        },
                        { x: x(j + 0.5), lo: p[0], hi: p[1] },
                        {
                          x: x(j + 1),
                          lo: (p[0] + (covered[j + 1] ? (layer[j + 1]?.[0] ?? p[0]) : p[0])) / 2,
                          hi: (p[1] + (covered[j + 1] ? (layer[j + 1]?.[1] ?? p[1]) : p[1])) / 2,
                        },
                      ]) ?? ''
                    }
                    fill={category(s.key, colorKeys)}
                    className={classes.streamBand}
                  />,
                );
              }),
            )}
            {days.map((d, i) =>
              ticks.has(i) ? (
                <text key={d} x={x(i)} y={184} className={classes.axis}>
                  {shortDayName(d)}
                </text>
              ) : null,
            )}
            <text x={30} y={199} className={classes.axisTitle}>
              Each band's width is that day's count
            </text>
          </>
        )}
      </Plot>
      <Group gap="md">
        {series.map(s => (
          <Text key={s.key} size="xs">
            <span className={classes.swatch} style={{ background: category(s.key, colorKeys) }} /> {eventName(s.name)}
          </Text>
        ))}
      </Group>
    </Stack>
  );
}
