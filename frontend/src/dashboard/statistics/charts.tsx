// The Statistics tab's chart pieces, drawn as SVG by React with D3's scales: a heatmap grid and a bar series. One style
// for both: magnitude is one blue ramp in six steps, a recorded stretch with nothing in it is the track color, and a
// stretch nothing was recorded in is hatched, so a gap never reads as a zero. Every mark that stands for moments is a
// link to Review there. The marks are one tab stop: arrow keys move between them, Enter follows, and the tooltip a
// pointer gets is also each link's accessible name.
import { Group, Text } from '@mantine/core';
import { useElementSize, useMergedRef } from '@mantine/hooks';
import { curveStepAfter, max as d3max, line, scaleBand, scaleLinear, scaleQuantize, timeFormat } from 'd3';
import {
  type KeyboardEvent,
  type MouseEvent,
  memo,
  type ReactNode,
  type RefObject,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { createPortal } from 'react-dom';
import { reviewURL } from '../review/url';
import { navigate } from '../router';
import classes from './reports.module.css';

const STEPS = [classes.s0, classes.s1, classes.s2, classes.s3, classes.s4, classes.s5];

/** The ramp step of a value from 0 to max: 0 for anything above zero but small, 5 at the top. */
export function step(value: number, max: number): number {
  if (!(max > 0) || value <= 0) return 0;
  return scaleQuantize<number>().domain([0, max]).range([0, 1, 2, 3, 4, 5])(Math.min(value, max));
}

/** Review at a moment: the day, and the time when there is one. */
export function reviewAt(day: string, t: number | null | undefined): string {
  return reviewURL({ day, t: t ?? undefined });
}

const dayFormat = timeFormat('%a %b %-d');
const shortDay = timeFormat('%b %-d');
const parseDay = (day: string) => new Date(`${day}T12:00:00`);
export const dayName = (day: string) => dayFormat(parseDay(day));
export const shortDayName = (day: string) => shortDay(parseDay(day));
export const hourName = (h: number) => `${h % 12 === 0 ? 12 : h % 12} ${h < 12 ? 'am' : 'pm'}`;

/** Which of n days get a tick: about eight, or the first of each month when there are many. */
export function dayTicks(days: string[]): Set<number> {
  if (days.length > 62) return new Set(days.flatMap((d, i) => (d.endsWith('-01') || i === 0 ? [i] : [])));
  const every = Math.max(1, Math.ceil(days.length / 8));
  return new Set(days.flatMap((_, i) => (i % every === 0 ? [i] : [])));
}

/** The ticks far enough apart to read: each at least `gap` pixels after the last one kept. */
export function spaced(ticks: Set<number>, at: (i: number) => number, gap = 56): Set<number> {
  const out = new Set<number>();
  let last = Number.NEGATIVE_INFINITY;
  for (const i of [...ticks].sort((a, b) => a - b))
    if (at(i) - last >= gap) {
      out.add(i);
      last = at(i);
    }
  return out;
}

/** A chart wider than its frame scrolls, newest at the right: show the newest end first, as on a phone, where two
 *  weeks of hours don't fit. Only when the drawing's width changes, so it never fights the reader's own scrolling. */
export function useNewestInView(ref: RefObject<HTMLElement | null>, drawn: number): void {
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new width is the trigger, though the effect reads the DOM
  useLayoutEffect(() => {
    const el = ref.current;
    if (el && el.scrollWidth > el.clientWidth) el.scrollLeft = el.scrollWidth - el.clientWidth;
  }, [ref, drawn]);
}

/** A chart's left axis, kept in view while the plot scrolls under it: its own SVG, stuck to the frame's left
 *  edge, overlapping the plot's blank left margin so the plot's coordinates don't move. */
function YAxis({
  title,
  width,
  height,
  children,
}: {
  title: string;
  width: number;
  height: number;
  children: ReactNode;
}) {
  return (
    <svg className={classes.yAxis} width={width} height={height} style={{ marginRight: -width }}>
      <title>{title}</title>
      {children}
    </svg>
  );
}

/** A chart's x-axis caption, in the page rather than in the scrolling plot: the plot scrolls to its newest column,
 *  which used to put the caption's start behind the sticky label column. It sticks to the frame's left edge,
 *  starts where the plot does, and wraps at the frame's width. */
function XCaption({ left, children }: { left: number; children: ReactNode }) {
  return (
    <figcaption className={classes.xCaption} style={{ paddingLeft: left }}>
      {children}
    </figcaption>
  );
}

export function follow(e: MouseEvent<Element>, to: string): void {
  if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  navigate(to);
}

/** A tooltip's anchor: the pointer, or the focused mark's corner, in viewport coordinates. */
interface Tip {
  x: number;
  y: number;
  text: string;
}

const TIP_OFFSET = 12; // from the pointer, so the tooltip never sits under it
const TIP_MARGIN = 8; // the least it keeps from the window's edges

/** Where a `width` x `height` tooltip goes for a pointer at (x, y) in a `viewWidth` x `viewHeight` window: below and to
 *  the right of it, flipped to its other side where that would run off the window, and never past the margin. */
export function placeTip(
  x: number,
  y: number,
  width: number,
  height: number,
  viewWidth: number,
  viewHeight: number,
): { left: number; top: number } {
  const side = (at: number, size: number, view: number) => {
    const after = at + TIP_OFFSET;
    const before = at - TIP_OFFSET - size;
    const chosen = after + size <= view - TIP_MARGIN || before < TIP_MARGIN ? after : before;
    return Math.max(TIP_MARGIN, Math.min(chosen, view - TIP_MARGIN - size));
  };
  return { left: side(x, width, viewWidth), top: side(y, height, viewHeight) };
}

/** A chart's tooltip state. A scroll anywhere takes it down: it floats at fixed viewport coordinates, so it would
 *  otherwise stay put while its mark moved away under it. The next pointer move or focus puts it back. */
export function useTip(): [Tip | null, (tip: Tip | null) => void] {
  const [tip, setTip] = useState<Tip | null>(null);
  const open = tip !== null;
  useEffect(() => {
    if (!open) return;
    const hide = () => setTip(() => null);
    window.addEventListener('scroll', hide, { capture: true, passive: true });
    return () => window.removeEventListener('scroll', hide, { capture: true });
  }, [open]);
  return [tip, setTip];
}

/** A chart's pointer callbacks, stable for as long as its tip's setter is, so the drawn plot can be memoized: a
 *  pointer move then re-renders the tooltip alone. */
function useTipShow(setTip: (tip: Tip | null) => void) {
  return {
    show: useCallback(
      (e: { clientX: number; clientY: number }, text: string) => setTip({ x: e.clientX, y: e.clientY, text }),
      [setTip],
    ),
    hide: useCallback(() => setTip(null), [setTip]),
  };
}

/** The tooltip floats over the page: in a portal on the document's body with fixed positioning, so the chart's
 *  card, which scrolls sideways and clips, never cuts it off or grows around it. It's placed before it's painted. */
export function Tooltip({ tip }: { tip: Tip | null }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!tip || !el) return;
    const { width, height } = el.getBoundingClientRect();
    const view = document.documentElement;
    const { left, top } = placeTip(tip.x, tip.y, width, height, view.clientWidth, view.clientHeight);
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;
  }, [tip]);
  if (!tip) return null;
  return createPortal(
    <div
      ref={ref}
      className={classes.tooltip}
      style={{ left: tip.x + TIP_OFFSET, top: tip.y + TIP_OFFSET }}
      aria-hidden="true"
    >
      {tip.text}
    </div>,
    document.body,
  );
}

/** An id for a pattern a fill refers to: useId's, kept to characters url(#...) takes as they are. */
export function usePatternId(): string {
  return `gap-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
}

export function GapPattern({ id }: { id: string }) {
  return (
    <defs>
      <pattern id={id} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="5" className={classes.gapStroke} />
      </pattern>
    </defs>
  );
}

/** Arrow-key movement between a chart's links: one of them is the tab stop, and the keys move it. */
export function useRoving(keys: string[][]) {
  const [active, setActive] = useState<string | null>(null);
  const root = useRef<SVGSVGElement>(null);
  const flat = keys.flat();
  const current = active && flat.includes(active) ? active : (flat[0] ?? null);
  const at = new Map<string, [number, number]>();
  keys.forEach((row, r) => {
    row.forEach((k, c) => {
      at.set(k, [r, c]);
    });
  });
  const move = (to: string | undefined) => {
    if (!to) return;
    setActive(to);
    root.current?.querySelector<SVGAElement>(`[data-key="${to}"]`)?.focus();
  };
  const onKeyDown = (e: KeyboardEvent<Element>) => {
    if (!current) return;
    const [r, c] = at.get(current) ?? [0, 0];
    const row = keys[r] ?? [];
    const near = (other: string[]) => {
      // The link in another row closest to this one's place in its row, by position.
      const here = c / Math.max(1, row.length - 1);
      return other.reduce<string | undefined>((best, k, i) => {
        const pos = i / Math.max(1, other.length - 1);
        const bestPos = best ? (at.get(best)?.[1] ?? 0) / Math.max(1, other.length - 1) : Number.POSITIVE_INFINITY;
        return Math.abs(pos - here) < Math.abs(bestPos - here) ? k : best;
      }, undefined);
    };
    const nextRow = (dir: 1 | -1) => {
      for (let i = r + dir; i >= 0 && i < keys.length; i += dir) if (keys[i]?.length) return near(keys[i] ?? []);
      return undefined;
    };
    const target: Record<string, string | undefined> = {
      ArrowRight: row[c + 1],
      ArrowLeft: row[c - 1],
      ArrowDown: nextRow(1),
      ArrowUp: nextRow(-1),
      Home: row[0],
      End: row[row.length - 1],
    };
    if (!(e.key in target)) return;
    e.preventDefault();
    move(target[e.key]);
  };
  return { root, current, setActive, onKeyDown };
}

export interface HeatCell {
  value: number | null; // null: nothing recorded
  label: string; // what the cell says, for the tooltip and a screen reader
  href?: string | null; // Review at the moment, when the cell has any
}

interface HeatmapProps {
  label: string;
  rows: string[];
  cols: string[];
  colTicks: Set<number>;
  rowTicks?: Set<number>;
  cell: (row: number, col: number) => HeatCell;
  xTitle: string;
  yTitle: string;
  cellHeight?: number;
  maxCellWidth?: number;
  minCellWidth?: number;
  rowLabelWidth?: number;
}

/** A grid of cells, rows by columns, colored by value on the ramp. The tooltip's state stays in this frame, so a
 *  pointer move re-renders the tooltip and not HeatPlot's cells, scales, and tab stops. */
export function Heatmap({
  label,
  rows,
  cols,
  colTicks,
  rowTicks,
  cell,
  xTitle,
  yTitle,
  cellHeight = 12,
  maxCellWidth = 28,
  minCellWidth = 3,
  rowLabelWidth = 60,
}: HeatmapProps) {
  const { ref: sized, width } = useElementSize();
  const ref = useRef<HTMLElement>(null);
  const figure = useMergedRef(sized, ref);
  const [tip, setTip] = useTip();
  const { show, hide } = useTipShow(setTip);
  return (
    <figure ref={figure} className={classes.chart} aria-label={label}>
      <HeatPlot
        label={label}
        rows={rows}
        cols={cols}
        colTicks={colTicks}
        rowTicks={rowTicks}
        cell={cell}
        xTitle={xTitle}
        yTitle={yTitle}
        cellHeight={cellHeight}
        maxCellWidth={maxCellWidth}
        minCellWidth={minCellWidth}
        rowLabelWidth={rowLabelWidth}
        width={width}
        frame={ref}
        show={show}
        hide={hide}
      />
      <Tooltip tip={tip} />
    </figure>
  );
}

interface HeatPlotProps {
  label: string;
  rows: string[];
  cols: string[];
  colTicks: Set<number>;
  rowTicks?: Set<number>;
  cell: (row: number, col: number) => HeatCell;
  xTitle: string;
  yTitle: string;
  cellHeight: number;
  maxCellWidth: number;
  minCellWidth: number;
  rowLabelWidth: number;
  width: number;
  frame: RefObject<HTMLElement | null>;
  show: (e: { clientX: number; clientY: number }, text: string) => void;
  hide: () => void;
}

/** The heatmap's drawing: everything a pointer move would otherwise rebuild. Nothing in the grid changes when the
 *  pointer moves, so the tip's state is the frame's alone and this re-renders only when its data or width does. */
const HeatPlot = memo(function HeatPlot({
  label,
  rows,
  cols,
  colTicks,
  rowTicks,
  cell,
  xTitle,
  yTitle,
  cellHeight,
  maxCellWidth,
  minCellWidth,
  rowLabelWidth,
  width,
  frame,
  show,
  hide,
}: HeatPlotProps) {
  const gap = usePatternId();
  const cells = rows.map((_, r) => cols.map((_, c) => cell(r, c)));
  const top = d3max(cells.flat(), d => d.value ?? 0) ?? 0;
  const left = rowLabelWidth;
  const cw = Math.max(minCellWidth, Math.min(maxCellWidth, Math.floor(((width || 640) - left - 8) / cols.length)));
  const x = scaleBand<number>()
    .domain(cols.map((_, i) => i))
    .range([left, left + cw * cols.length]);
  const y = scaleBand<number>()
    .domain(rows.map((_, i) => i))
    .range([4, 4 + cellHeight * rows.length]);
  const plotBottom = 4 + cellHeight * rows.length;
  const shownCols = spaced(colTicks, c => x(c) ?? 0);
  const w = left + cw * cols.length + 8;
  const h = plotBottom + 20; // the tick labels; the caption is below, in the page
  useNewestInView(frame, w);
  const keys = cells.map((row, r) => row.flatMap((d, c) => (d.href ? [`${r}:${c}`] : [])));
  const roving = useRoving(keys);
  const inset = cw > 6 ? 1 : 0;
  return (
    <>
      <div className={classes.track}>
        <YAxis title={yTitle || 'Rows'} width={left} height={h}>
          {rows.map((name, r) =>
            !rowTicks || rowTicks.has(r) ? (
              <text
                key={name}
                className={classes.axis}
                x={left - 6}
                y={(y(r) ?? 0) + cellHeight / 2}
                textAnchor="end"
                dominantBaseline="central"
              >
                {name}
              </text>
            ) : null,
          )}
          <text
            className={classes.axisTitle}
            transform={`translate(8 ${plotBottom / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            {yTitle}
          </text>
        </YAxis>
        <svg ref={roving.root} width={w} height={h}>
          <title>{label}</title>
          <GapPattern id={gap} />
          {cols.map((name, c) =>
            shownCols.has(c) ? (
              <text key={name} className={classes.axis} x={x(c) ?? 0} y={plotBottom + 13}>
                {name}
              </text>
            ) : null,
          )}
          {cells.map((row, r) =>
            row.map((d, c) => {
              const rect = (
                <rect
                  x={(x(c) ?? 0) + inset}
                  y={(y(r) ?? 0) + inset}
                  width={cw - 2 * inset}
                  height={cellHeight - 2 * inset}
                  rx={cw > 6 ? 2 : 0}
                  className={d.value === null ? undefined : d.value > 0 ? STEPS[step(d.value, top)] : classes.zero}
                  fill={d.value === null ? `url(#${gap})` : undefined}
                />
              );
              const key = `${r}:${c}`;
              if (!d.href)
                return (
                  // biome-ignore lint/a11y/noStaticElementInteractions: hover only; the same facts are in the summary
                  <g key={key} onMouseMove={e => show(e, d.label)} onMouseLeave={hide}>
                    {rect}
                  </g>
                );
              return (
                <a
                  key={key}
                  href={d.href}
                  data-key={key}
                  className={classes.mark}
                  tabIndex={roving.current === key ? 0 : -1}
                  aria-label={`${d.label}. Open in Review`}
                  onClick={e => follow(e, d.href ?? '')}
                  onKeyDown={roving.onKeyDown}
                  onBlur={hide}
                  onMouseMove={e => show(e, d.label)}
                  onMouseLeave={hide}
                  onFocus={e => {
                    roving.setActive(key);
                    const box = e.currentTarget.getBoundingClientRect();
                    show({ clientX: box.right, clientY: box.bottom }, d.label);
                  }}
                >
                  {rect}
                </a>
              );
            }),
          )}
        </svg>
      </div>
      <XCaption left={left}>{xTitle}</XCaption>
    </>
  );
});

export interface BarPoint {
  key: string;
  value: number | null; // null: nothing recorded
  label: string;
  href?: string | null;
}

interface BarsProps {
  label: string;
  points: BarPoint[];
  ticks: Set<number>;
  tickLabel: (i: number) => string;
  xTitle: string;
  yTitle: string;
  height?: number;
  plan?: number | null | (number | null)[]; // what the bars are measured against: one level, or one per bar (a dashed line)
}

/** Bars over time from a zero baseline; a gap is a hatched column the height of the plot. The tooltip's state stays
 *  in this frame, so a pointer move re-renders the tooltip and not BarsPlot's scales, marks, and plan line. */
export function Bars({ label, points, ticks, tickLabel, xTitle, yTitle, height = 160, plan }: BarsProps) {
  const { ref: sized, width } = useElementSize();
  const ref = useRef<HTMLElement>(null);
  const figure = useMergedRef(sized, ref);
  const [tip, setTip] = useTip();
  const { show, hide } = useTipShow(setTip);
  return (
    <figure ref={figure} className={classes.chart} aria-label={label}>
      <BarsPlot
        label={label}
        points={points}
        ticks={ticks}
        tickLabel={tickLabel}
        xTitle={xTitle}
        yTitle={yTitle}
        height={height}
        plan={plan}
        width={width}
        frame={ref}
        show={show}
        hide={hide}
      />
      <Tooltip tip={tip} />
    </figure>
  );
}

interface BarsPlotProps {
  label: string;
  points: BarPoint[];
  ticks: Set<number>;
  tickLabel: (i: number) => string;
  xTitle: string;
  yTitle: string;
  height: number;
  plan?: number | null | (number | null)[];
  width: number;
  frame: RefObject<HTMLElement | null>;
  show: (e: { clientX: number; clientY: number }, text: string) => void;
  hide: () => void;
}

/** The bars' drawing, memoized on everything but the tooltip, as HeatPlot is. */
const BarsPlot = memo(function BarsPlot({
  label,
  points,
  ticks,
  tickLabel,
  xTitle,
  yTitle,
  height,
  plan,
  width,
  frame,
  show,
  hide,
}: BarsPlotProps) {
  const gap = usePatternId();
  const left = 44;
  const levels = Array.isArray(plan) ? plan : points.map(() => plan ?? null);
  const top = Math.max(d3max(points, p => p.value ?? 0) ?? 0, d3max(levels, v => v ?? 0) ?? 0);
  const plotW = Math.max(points.length * 3, (width || 640) - left - 8);
  const x = scaleBand<number>()
    .domain(points.map((_, i) => i))
    .range([left, left + plotW])
    .paddingInner(points.length > 60 ? 0.1 : 0.2);
  const y = scaleLinear()
    .domain([0, top || 1])
    .nice()
    .range([height, 6]);
  const keys = [points.flatMap(p => (p.href ? [p.key] : []))];
  const roving = useRoving(keys);
  const bw = x.bandwidth();
  const shownTicks = spaced(ticks, i => x(i) ?? 0);
  useNewestInView(frame, plotW);
  return (
    <>
      <div className={classes.track}>
        <YAxis title={yTitle} width={left} height={height + 20}>
          {y.ticks(4).map(v => (
            <text key={v} className={classes.axis} x={left - 6} y={y(v)} textAnchor="end" dominantBaseline="central">
              {v}
            </text>
          ))}
          <text className={classes.axisTitle} transform={`translate(10 ${height / 2}) rotate(-90)`} textAnchor="middle">
            {yTitle}
          </text>
        </YAxis>
        <svg ref={roving.root} width={left + plotW + 8} height={height + 20}>
          <title>{label}</title>
          <GapPattern id={gap} />
          {y.ticks(4).map(v => (
            <line key={v} className={classes.gridline} x1={left} x2={left + plotW} y1={y(v)} y2={y(v)} />
          ))}
          {points.map((p, i) =>
            shownTicks.has(i) ? (
              <text key={p.key} className={classes.axis} x={x(i) ?? 0} y={height + 14}>
                {tickLabel(i)}
              </text>
            ) : null,
          )}
          {points.map((p, i) => {
            const x0 = x(i) ?? 0;
            if (p.value === null)
              return (
                // biome-ignore lint/a11y/noStaticElementInteractions: hover only; the title says what a gap is
                <g key={p.key} onMouseMove={e => show(e, p.label)} onMouseLeave={hide}>
                  <rect x={x0} y={6} width={bw} height={height - 6} fill={`url(#${gap})`} />
                </g>
              );
            const y0 = y(p.value);
            const bar = (
              <rect
                x={x0}
                y={y0}
                width={bw}
                height={Math.max(0, height - y0)}
                rx={Math.min(2, bw / 2)}
                className={classes.bar}
              />
            );
            // A transparent hit area the full height, so a short bar is easy to point at.
            const hit = <rect x={x0} y={6} width={bw} height={height - 6} fill="transparent" />;
            if (!p.href)
              return (
                // biome-ignore lint/a11y/noStaticElementInteractions: hover only
                <g key={p.key} onMouseMove={e => show(e, p.label)} onMouseLeave={hide}>
                  {hit}
                  {bar}
                </g>
              );
            return (
              <a
                key={p.key}
                href={p.href}
                data-key={p.key}
                className={classes.mark}
                tabIndex={roving.current === p.key ? 0 : -1}
                aria-label={`${p.label}. Open in Review`}
                onClick={e => follow(e, p.href ?? '')}
                onKeyDown={roving.onKeyDown}
                onBlur={hide}
                onMouseMove={e => show(e, p.label)}
                onMouseLeave={hide}
                onFocus={e => {
                  roving.setActive(p.key);
                  const box = e.currentTarget.getBoundingClientRect();
                  show({ clientX: box.right, clientY: box.top }, p.label);
                }}
              >
                {hit}
                {bar}
              </a>
            );
          })}
          {plan != null && (
            <path
              className={classes.plan}
              d={
                line<[number, number]>()
                  .curve(curveStepAfter)
                  .defined(([, v]) => Number.isFinite(v))(
                  [...levels, levels[levels.length - 1] ?? null].map((v, i) => [
                    i < points.length ? (x(i) ?? 0) : left + plotW,
                    v == null ? Number.NaN : y(v),
                  ]),
                ) ?? ''
              }
            />
          )}
          <line className={classes.gridline} x1={left} x2={left + plotW} y1={height} y2={height} />
        </svg>
      </div>
      <XCaption left={left}>{xTitle}</XCaption>
    </>
  );
});

/** The key under a ramp chart: what the steps, an empty recorded stretch, and a gap look like. */
export function RampLegend({
  low,
  high,
  unit,
  zero = 'Recorded, nothing there',
  gap = 'Nothing recorded',
}: {
  low: string;
  high: string;
  unit?: ReactNode;
  zero?: string;
  gap?: string;
}) {
  return (
    <Group gap="md" className={classes.legend} wrap="wrap">
      <Group gap={4} wrap="nowrap">
        <span>{low}</span>
        {STEPS.map(s => (
          <svg key={s} width="12" height="12" aria-hidden="true">
            <rect width="12" height="12" rx="2" className={s} />
          </svg>
        ))}
        <span>{high}</span>
        {unit && <span>{unit}</span>}
      </Group>
      <Group gap={4} wrap="nowrap">
        <svg width="12" height="12" aria-hidden="true">
          <rect width="12" height="12" rx="2" className={classes.zero} />
        </svg>
        <span>{zero}</span>
      </Group>
      <GapKey>{gap}</GapKey>
    </Group>
  );
}

export function GapKey({ children = 'Nothing recorded' }: { children?: ReactNode }) {
  const id = usePatternId();
  return (
    <Group gap={4} wrap="nowrap" className={classes.legend}>
      <svg width="12" height="12" aria-hidden="true">
        <GapPattern id={id} />
        <rect width="12" height="12" rx="2" fill={`url(#${id})`} />
      </svg>
      <span>{children}</span>
    </Group>
  );
}

/** Days still being read: the chart shows the rest, and says so rather than drawing them as empty. */
export function Pending({ pending }: { pending: string[] }) {
  if (!pending.length) return null;
  return (
    <Text size="xs" c="dimmed" role="status">
      Still reading {pending.length} {pending.length === 1 ? 'day' : 'days'}. They appear here as they're done.
    </Text>
  );
}

/** The key for a dashed reference line (a plan, or a typical day). */
export function LineKey({ children }: { children: ReactNode }) {
  return (
    <Group gap={4} wrap="nowrap" className={classes.legend}>
      <svg width="18" height="12" aria-hidden="true">
        <line x1="0" x2="18" y1="6" y2="6" className={classes.plan} />
      </svg>
      <span>{children}</span>
    </Group>
  );
}

/** Which of a run of hourly points get a tick: each midnight. */
export function midnightTicks(times: number[]): Set<number> {
  return new Set(times.flatMap((t, i) => (new Date(t * 1000).getHours() === 0 ? [i] : [])));
}
