// Paths on a place's plan, Strava style: where each person and pet went over a window of time, in their own
// colour, each a smooth curve with an arrowhead for the way they went, fading with age, with a dot where they
// were at the scrubber's time and their name on hover. Someone known only by the room they were in (a zone visit) lights
// that room up instead of drawing a point. The window is the last hour, day, or week up to now, or around the moment a
// link asked for (/places?place=&t=, as the Journal's "where" links).
//
// The plan is the library's SVG string; the paths are a second SVG laid over it with the same viewBox, placed through
// the library's projection, so scrubbing redraws only the paths.
import { Button, Group, SegmentedControl, Slider, Text } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';
import type { FloorplanResult, RenderOptions } from '../../../vendor/floorplan/index.js';
import { projection } from '../../../vendor/floorplan/index.js';
import { hm, plural } from '../../lib/format';
import { type Point, type Positions, positionsQuery } from '../api';
import classes from '../places.module.css';
import { examplePositions } from './fixture';
import {
  arrowHead,
  colours,
  fade,
  gapFor,
  label,
  latest,
  PLAIN_SCALE,
  type PlainFrame,
  pathData,
  plainFrame,
  plainToScreen,
  planRoom,
  roomsAt,
  segments,
  spline,
  UNKNOWN_COLOUR,
} from './paths';

/** `hidden` with `ids` hidden (`hide`) or shown. */
const toggle = (hidden: Set<string>, ids: string[], hide: boolean): Set<string> => {
  const next = new Set(hidden);
  for (const id of ids) {
    if (hide) next.add(id);
    else next.delete(id);
  }
  return next;
};

/** While the window ends at now, it moves on this often (a fresh read of the window); otherwise it stays put, so
 *  scrubbing and toggling people never read again. */
export const FOLLOW_NOW_MS = 5 * 60 * 1000;

/** The windows the scrubber can span, in seconds. */
export const WINDOWS = { hour: 3600, day: 86400, week: 7 * 86400 } as const;
export type Window = keyof typeof WINDOWS;

/** The window `span` long that shows `t`: ending now, or centred on `t` when that's earlier than half a span ago. */
export function windowAround(t: number, span: number, now: number): { start: number; end: number } {
  const end = Math.min(now, t + span / 2);
  return { start: end - span, end };
}

/** A time on the scrubber: "14:05", with the day for a window longer than one. */
function when(ts: number, span: number): string {
  if (span < 86400) return hm(ts); // a day or longer names the day, or its two ends read the same
  return `${new Date(ts * 1000).toLocaleDateString([], { weekday: 'short' })} ${hm(ts)}`;
}

/** Where the paths are drawn: floor metres to the drawing's pixels, and the plan's rooms (none on a plain frame). */
interface Surface {
  toScreen: (p: [number, number]) => [number, number];
  rooms: { id: string; name: string; poly: [number, number][] }[];
}

/** A place with no plan yet: plain floor, a line every metre, and a metre's scale bar, under its paths. */
function PlainGrid({ frame }: { frame: PlainFrame }) {
  const w = frame.width * PLAIN_SCALE;
  const h = frame.height * PLAIN_SCALE;
  const xs: number[] = [];
  for (let x = Math.ceil(frame.minX); x < frame.minX + frame.width; x++) xs.push((x - frame.minX) * PLAIN_SCALE);
  const ys: number[] = [];
  for (let y = Math.ceil(frame.minY); y < frame.minY + frame.height; y++) ys.push((y - frame.minY) * PLAIN_SCALE);
  const line = 'var(--mantine-color-default-border)';
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Floor, a line every metre">
      <rect x={0.5} y={0.5} width={w - 1} height={h - 1} fill="var(--lib-card)" stroke={line} />
      {xs.map(x => (
        <line key={`x${x}`} x1={x} y1={0} x2={x} y2={h} stroke={line} strokeDasharray="2 4" />
      ))}
      {ys.map(y => (
        <line key={`y${y}`} x1={0} y1={y} x2={w} y2={y} stroke={line} strokeDasharray="2 4" />
      ))}
      <line x1={8} y1={h - 10} x2={8 + PLAIN_SCALE} y2={h - 10} stroke="var(--mantine-color-dimmed)" strokeWidth={2} />
      <text x={12 + PLAIN_SCALE} y={h - 6} fontSize={11} fill="var(--mantine-color-dimmed)">
        1 m
      </text>
    </svg>
  );
}

function Overlay({
  surface,
  viewBox,
  data,
  at,
  span,
  hidden,
}: {
  surface: Surface;
  viewBox: string;
  data: Positions;
  at: number;
  span: number;
  hidden: Set<string>;
}) {
  const { toScreen, rooms } = surface;
  const colour = colours(data.entities);
  const name = new Map(data.entities.map(e => [e.id, label(e)]));
  const unknown = new Set(data.entities.filter(e => e.kind === 'unknown').map(e => e.id));
  const xy = (x: number, y: number) =>
    toScreen([x, y])
      .map(v => v.toFixed(1))
      .join(',');
  return (
    <svg className={classes.paths} viewBox={viewBox} aria-hidden>
      {roomsAt(data.rooms, at)
        .filter(r => !hidden.has(r.entity))
        .map(r => {
          const room = rooms.find(x => x.id === planRoom(r, rooms));
          if (!room) return null;
          return (
            <polygon
              key={`${r.entity}-${r.start}`}
              points={room.poly.map(([x, y]) => xy(x, y)).join(' ')}
              fill={colour.get(r.entity)}
              fillOpacity={0.18}
              stroke={colour.get(r.entity)}
              strokeDasharray="6 4"
              strokeWidth={1.5}
            >
              <title>{`${name.get(r.entity)} in the ${r.name.toLowerCase()}, ${hm(r.start)} to ${hm(r.end)}`}</title>
            </polygon>
          );
        })}
      {[...data.tracks]
        // the not yet identified underneath, so the people and pets the house knows read over them
        .sort((a, b) => Number(!unknown.has(a.entity)) - Number(!unknown.has(b.entity)))
        .filter(t => !hidden.has(t.entity))
        .map(t => {
          const c = colour.get(t.entity);
          const now = latest(t.points, at, gapFor(span));
          const stranger = unknown.has(t.entity);
          const parts = segments(t.points, at, gapFor(span));
          const first = parts[0]?.[0];
          const last = parts.at(-1)?.at(-1);
          return (
            <g key={t.track ?? t.entity}>
              {first && last && <title>{`${name.get(t.entity)}, ${hm(first.t)} to ${hm(last.t)}`}</title>}
              {parts.map(s => {
                const curve = spline(s.map(p => toScreen([p.x, p.y])));
                const d = pathData(curve);
                const opacity = fade(at - (s.at(-1) as Point).t, span);
                const width = stranger ? 1.5 : 2.5;
                return (
                  <g key={s[0]?.t} opacity={opacity}>
                    {!stranger && (
                      // a casing in the card's colour, so a path reads over the plan and over other paths
                      <path
                        d={d}
                        fill="none"
                        stroke="var(--lib-card)"
                        strokeWidth={width + 2.5}
                        strokeLinecap="round"
                      />
                    )}
                    <path
                      d={d}
                      fill="none"
                      stroke={c}
                      strokeWidth={width}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <polygon points={arrowHead(curve, stranger ? 5 : 9)} fill={c} />
                    {/* a wider invisible stroke to hover for the name: the drawn one is a couple of pixels wide */}
                    <path d={d} fill="none" stroke="transparent" strokeWidth={10} pointerEvents="stroke" />
                  </g>
                );
              })}
              {now && (
                <circle
                  className={classes.here}
                  cx={toScreen([now.x, now.y])[0]}
                  cy={toScreen([now.x, now.y])[1]}
                  r={stranger ? 3.5 : 5.5}
                  fill={c}
                >
                  <title>{`${name.get(t.entity)} at ${hm(now.t)}`}</title>
                </circle>
              )}
            </g>
          );
        })}
    </svg>
  );
}

/** A place's plan with its paths, the scrubber, and who's on it. `svg` is the plan as drawn (drawPlan), and `result`
 *  the plan it drew; a place with neither (no plan yet) draws its paths on plain floor fitted to them. */
export function PlacePaths({
  place,
  example,
  svg,
  result,
  options,
  linked,
}: {
  place: string;
  example: boolean;
  svg?: string;
  result?: FloorplanResult;
  options: RenderOptions;
  /** A moment a link asked for (epoch seconds), or null for now. */
  linked: number | null;
}) {
  const [win, setWin] = useState<Window>('day');
  const span = WINDOWS[win];
  // The window is pinned: the moment it shows (a link's, or now) and what "now" was when it was set, both in state, so
  // a render (a scrub, a toggle) never moves it. Only Now, a new window length, or the slow tick below moves it.
  const [pinned, setPinned] = useState(() => {
    const now = Date.now() / 1000;
    return { anchor: linked ?? now, now, following: linked === null };
  });
  useEffect(() => {
    if (!pinned.following) return;
    const id = setInterval(() => {
      const now = Date.now() / 1000;
      setPinned({ anchor: now, now, following: true });
    }, FOLLOW_NOW_MS);
    return () => clearInterval(id);
  }, [pinned.following]);
  const range = windowAround(pinned.anchor, span, Math.max(pinned.anchor, pinned.now));
  const start = Math.floor(range.start);
  const end = Math.floor(range.end);
  const [at, setAt] = useState<number | null>(linked);
  const cursor = Math.min(end, Math.max(start, at ?? end));
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const query = useQuery({ ...positionsQuery(place, start, end), enabled: !example });
  const data = useMemo(
    () => (example ? examplePositions(place, start, end) : query.data),
    [example, place, start, end, query.data],
  );
  const planned = useMemo(
    (): Surface | null =>
      result
        ? {
            toScreen: projection(result.model, options).toScreen,
            rooms: result.model.rooms.map(r => ({
              id: r.room.id,
              name: r.room.name,
              poly: r.room.poly as [number, number][],
            })),
          }
        : null,
    [result, options],
  );
  const plain = useMemo(() => (svg || !data ? null : plainFrame(data.tracks)), [svg, data]);
  const surface: Surface | null = planned ?? (plain ? { toScreen: p => plainToScreen(plain, p), rooms: [] } : null);
  const viewBox = svg
    ? (svg.match(/viewBox="([^"]+)"/)?.[1] ?? '0 0 100 100')
    : plain
      ? `0 0 ${plain.width * PLAIN_SCALE} ${plain.height * PLAIN_SCALE}`
      : '0 0 100 100';
  const colour = data ? colours(data.entities) : new Map<string, string>();
  // Everyone not identified yet is one legend entry: a day can hold hundreds of their short camera tracks.
  const strangers = data ? data.entities.filter(e => e.kind === 'unknown').map(e => e.id) : [];
  const strangersOff = strangers.length > 0 && strangers.every(id => hidden.has(id));
  const nobody = data && data.tracks.length === 0 && data.rooms.length === 0;
  return (
    <>
      <div className={classes.drawingWrap}>
        {svg ? (
          <div
            className={classes.drawing}
            // biome-ignore lint/security/noDangerouslySetInnerHtml: the renderer and the overlay escape every label
            dangerouslySetInnerHTML={{ __html: svg }}
          />
        ) : (
          plain && (
            <div className={classes.drawing}>
              <PlainGrid frame={plain} />
            </div>
          )
        )}
        {data && surface && (
          <Overlay surface={surface} viewBox={viewBox} data={data} at={cursor} span={span} hidden={hidden} />
        )}
      </div>
      <div className={classes.scrubber}>
        <Group justify="space-between" gap="xs" wrap="wrap">
          <SegmentedControl
            size="xs"
            value={win}
            onChange={v => setWin(v as Window)}
            data={[
              { value: 'hour', label: 'Hour' },
              { value: 'day', label: 'Day' },
              { value: 'week', label: 'Week' },
            ]}
            aria-label="How far back"
          />
          <Text size="sm" fw={500} role="status">
            {when(cursor, span)}
          </Text>
          <Button
            size="compact-xs"
            variant="subtle"
            onClick={() => {
              const now = Date.now() / 1000;
              setPinned({ anchor: now, now, following: true });
              setAt(null);
            }}
          >
            Now
          </Button>
        </Group>
        <Slider
          min={start}
          max={end}
          step={Math.max(1, Math.round(span / 600))}
          value={cursor}
          onChange={v => setAt(v)}
          label={v => when(v, span)}
          thumbLabel="Time"
          mt="xs"
        />
        <Group justify="space-between" mt={4}>
          <Text size="xs" c="dimmed">
            {when(start, span)}
          </Text>
          <Text size="xs" c="dimmed">
            {when(end, span)}
          </Text>
        </Group>
      </div>
      {query.isError && !example ? (
        <Text size="sm" c="red">
          Couldn't load where people were.
        </Text>
      ) : nobody ? (
        <Text size="sm" c="dimmed">
          Nobody was placed here in this window.
        </Text>
      ) : (
        data && (
          <ul className={classes.who} aria-label="Who">
            {data.entities
              .filter(e => e.kind !== 'unknown')
              .map(e => {
                const off = hidden.has(e.id);
                const zone = data.rooms.find(r => r.entity === e.id && !data.tracks.some(t => t.entity === e.id));
                return (
                  <li key={e.id}>
                    <button
                      type="button"
                      className={classes.whoButton}
                      aria-pressed={!off}
                      onClick={() => setHidden(toggle(hidden, [e.id], !off))}
                    >
                      <span className={classes.whoSwatch} style={{ background: colour.get(e.id) }} aria-hidden />
                      {label(e)}
                      {zone && <span className={classes.whoNote}>{` · in the ${zone.name.toLowerCase()}`}</span>}
                    </button>
                  </li>
                );
              })}
            {strangers.length > 0 && (
              <li>
                <button
                  type="button"
                  className={classes.whoButton}
                  aria-pressed={!strangersOff}
                  onClick={() => setHidden(toggle(hidden, strangers, !strangersOff))}
                >
                  <span className={classes.whoSwatch} style={{ background: UNKNOWN_COLOUR }} aria-hidden />
                  Not identified yet
                  <span className={classes.whoNote}>{` · ${plural(strangers.length, 'track', 'tracks')}`}</span>
                </button>
              </li>
            )}
          </ul>
        )
      )}
    </>
  );
}
