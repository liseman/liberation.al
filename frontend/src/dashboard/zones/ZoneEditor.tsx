// Drawing named zones on one camera's live picture: click corners to outline a zone, name it, and save the
// camera's whole set. The picture is never mirrored here, since zones are in the camera's own frame coordinates, and
// the box takes the frame's shape, so a corner lands where it was clicked.
import { Button, Checkbox, Group, Modal, Stack, Text, TextInput, Timeline } from '@mantine/core';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type KeyboardEvent, type MouseEvent, useEffect, useRef, useState } from 'react';
import { messageOf } from '../../shared/http';
import { LiveCam } from '../camera/LiveCam';
import { PageError, PageLoading } from '../lib/page';
import { type Definition, saveZones, ZONES_KEY, type Zone, zoneHistoryQuery, zonesQuery } from './api';
import { centre, type Point, pointAt, problem } from './geometry';
import classes from './zones.module.css';

const DEFAULT_RATIO = 16 / 9;

/** The camera canvas's frame shape (width / height), read from the canvas once frames arrive. */
function useFrameRatio(box: React.RefObject<HTMLDivElement | null>): number {
  const [ratio, setRatio] = useState(DEFAULT_RATIO);
  useEffect(() => {
    const read = () => {
      const canvas = box.current?.querySelector('canvas');
      if (canvas && canvas.width > 0 && canvas.height > 0) setRatio(canvas.width / canvas.height);
    };
    read();
    const timer = setInterval(read, 500);
    return () => clearInterval(timer);
  }, [box]);
  return ratio;
}

const same = (a: Zone[], b: Zone[]) => JSON.stringify(a) === JSON.stringify(b);

/** Why the set can't be saved yet, or null. */
export function setProblem(zones: Zone[]): string | null {
  if (zones.some(z => !z.name.trim())) return 'Every zone needs a name.';
  const names = zones.map(z => z.name.trim().toLowerCase());
  if (new Set(names).size !== names.length) return 'Two zones share a name.';
  return null;
}

function Versions({ cam }: { cam: string }) {
  const q = useQuery(zoneHistoryQuery(cam));
  if (q.isPending) return <PageLoading label="earlier versions" lines={3} />;
  if (q.isError)
    return (
      <PageError
        title="Couldn't load earlier versions"
        detail={messageOf(q.error)}
        onRetry={() => void q.refetch()}
        busy={q.isFetching}
      />
    );
  return (
    <Timeline role="list" className={classes.versions} bulletSize={12} lineWidth={2}>
      {q.data.versions.map(v => (
        <Timeline.Item key={v.id} role="listitem">
          {new Date(v.at * 1000).toLocaleString()}: {v.zones.length ? v.zones.map(z => z.name).join(', ') : 'no zones'}
        </Timeline.Item>
      ))}
    </Timeline>
  );
}

export function ZoneEditor({ cam, label, onClose }: { cam: string; label: string; onClose: () => void }) {
  const client = useQueryClient();
  const q = useQuery(zonesQuery());
  const saved: Definition | undefined = q.data && Object.hasOwn(q.data.cameras, cam) ? q.data.cameras[cam] : undefined;
  const [zones, setZones] = useState<Zone[] | null>(null);
  // A new zone drawn from "Draw the TV" starts as a screen named TV.
  const [drawing, setDrawing] = useState<{ corners: Point[]; replaces: number | null; tv?: boolean } | null>(null);
  const [note, setNote] = useState('');
  const [history, setHistory] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const ratio = useFrameRatio(box);
  // The editor starts from the saved set once it has loaded, and keeps its own edits after that.
  useEffect(() => {
    if (q.data && zones === null) setZones(saved?.zones ?? []);
  }, [q.data, saved, zones]);
  const current = zones ?? [];
  const save = useMutation({
    mutationFn: () => saveZones(cam, current, saved?.id ?? null),
    onSuccess: async reply => {
      setZones(reply.definition.zones);
      setNote('Saved.');
      await client.invalidateQueries({ queryKey: ZONES_KEY });
    },
  });

  const add = (corner: Point) => drawing && setDrawing({ ...drawing, corners: [...drawing.corners, corner] });
  const click = (e: MouseEvent<SVGSVGElement>) =>
    add(pointAt(e.currentTarget.getBoundingClientRect(), e.clientX, e.clientY));
  // Without a mouse: the arrow keys move a crosshair (Shift for bigger steps), and Enter or Space puts a corner there.
  const [cursor, setCursor] = useState<Point>([0.5, 0.5]);
  const key = (e: KeyboardEvent<SVGSVGElement>) => {
    if (!drawing) return;
    const step = e.shiftKey ? 0.05 : 0.01;
    const moves: Record<string, Point> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const move = moves[e.key];
    if (move) {
      e.preventDefault();
      const clamp = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 1e5) / 1e5;
      setCursor([clamp(cursor[0] + move[0]), clamp(cursor[1] + move[1])]);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      add(cursor);
    }
  };
  const finish = () => {
    if (!drawing) return;
    const { corners, replaces, tv } = drawing;
    setZones(
      replaces === null
        ? [...current, tv ? { name: 'TV', polygon: corners, kind: 'screen' as const } : { name: '', polygon: corners }]
        : current.map((z, i) => (i === replaces ? { ...z, polygon: corners } : z)),
    );
    setDrawing(null);
    setNote('');
  };
  const edit = (next: Zone[]) => {
    setZones(next);
    setNote('');
  };
  const drawn = drawing ? problem(drawing.corners) : null;
  const blocked = setProblem(current);
  const changed = !same(current, saved?.zones ?? []);

  return (
    <Modal opened onClose={onClose} title={`Zones on ${label}`} size="xl">
      <Stack gap="sm">
        <Text size="sm">
          Outline the places you want counted, like a litter box, a doorway, or a food bowl. Visits are counted for
          anyone or any animal whose position is inside a zone.
        </Text>
        <div ref={box} className={classes.picture} style={{ aspectRatio: String(ratio) }}>
          <LiveCam cam={cam} alt={`Picture from ${label}`} />
          <svg
            className={drawing ? `${classes.overlay} ${classes.drawing}` : classes.overlay}
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
            onClick={click}
            onKeyDown={key}
            tabIndex={drawing ? 0 : -1}
            role={drawing ? 'application' : 'img'}
            aria-label={
              drawing ? 'Click to add a corner, or move with the arrow keys and press Enter' : `Zones on ${label}`
            }
          >
            {current.map((z, i) =>
              drawing?.replaces === i ? null : (
                <polygon key={z.id ?? `new-${i}`} className={classes.zone} points={z.polygon.join(' ')} />
              ),
            )}
            {drawing && drawing.corners.length > 0 && (
              <polyline className={classes.outline} points={drawing.corners.join(' ')} />
            )}
            {drawing && <circle className={classes.cursor} cx={cursor[0]} cy={cursor[1]} r={0.008} />}
          </svg>
          {current.map((z, i) => {
            const [x, y] = centre(z.polygon);
            return drawing?.replaces === i ? null : (
              <span
                key={z.id ?? `new-${i}`}
                className={classes.label}
                style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
              >
                {z.name || 'Unnamed'}
              </span>
            );
          })}
        </div>
        {drawing ? (
          <Group gap="xs">
            <Text size="sm">
              {drawing.corners.length} corner{drawing.corners.length === 1 ? '' : 's'}. {drawn ?? 'Ready.'}
            </Text>
            <Button
              variant="default"
              size="xs"
              disabled={!drawing.corners.length}
              onClick={() => setDrawing({ ...drawing, corners: drawing.corners.slice(0, -1) })}
            >
              Undo corner
            </Button>
            <Button variant="default" size="xs" onClick={() => setDrawing(null)}>
              Cancel
            </Button>
            <Button size="xs" variant="default" disabled={drawn !== null} onClick={finish}>
              Done
            </Button>
          </Group>
        ) : (
          <Stack gap={4}>
            <Group gap="xs">
              <Button size="xs" variant="default" onClick={() => setDrawing({ corners: [], replaces: null })}>
                Add a zone
              </Button>
              <Button size="xs" variant="default" onClick={() => setDrawing({ corners: [], replaces: null, tv: true })}>
                Draw the TV
              </Button>
            </Group>
            <Text size="xs" c="dimmed">
              Draw the TV around its screen: while its picture moves, what's heard can be marked as TV by ear, and
              nobody it shows is counted.
            </Text>
          </Stack>
        )}
        {q.isError && (
          <PageError
            title="Couldn't load this camera's zones"
            detail={messageOf(q.error)}
            onRetry={() => void q.refetch()}
            busy={q.isFetching}
          />
        )}
        {current.map((z, i) => (
          <Group key={z.id ?? `new-${i}`} gap="xs" align="flex-end">
            <TextInput
              label={`Name of zone ${i + 1}`}
              placeholder="Litter box"
              value={z.name}
              maxLength={60}
              onChange={e => edit(current.map((y, j) => (j === i ? { ...y, name: e.currentTarget.value } : y)))}
              className={classes.name}
            />
            <Checkbox
              label="A screen"
              description="A TV: nobody it shows is counted, and while its picture moves, what's heard can be marked as TV. A monitor drawn as a screen would count as a TV too."
              checked={z.kind === 'screen'}
              onChange={e => {
                const kind = e.currentTarget.checked ? ('screen' as const) : undefined;
                edit(current.map((y, j) => (j === i ? { ...y, kind } : y)));
              }}
            />
            <Button
              variant="default"
              size="xs"
              disabled={drawing !== null}
              onClick={() => setDrawing({ corners: [], replaces: i })}
            >
              Redraw
            </Button>
            <Button
              variant="subtle"
              color="red"
              size="xs"
              disabled={drawing !== null}
              onClick={() => edit(current.filter((_, j) => j !== i))}
            >
              Remove
            </Button>
          </Group>
        ))}
        {save.isError && (
          <PageError title="The zones weren't saved" detail={messageOf(save.error)}>
            Your edits are still here. Try saving again.
          </PageError>
        )}
        <Group justify="space-between">
          <Button variant="subtle" size="xs" onClick={() => setHistory(!history)}>
            {history ? 'Hide earlier versions' : 'Earlier versions'}
          </Button>
          <Group gap="xs">
            {note && <Text size="sm">{note}</Text>}
            {changed && blocked && <Text size="sm">{blocked}</Text>}
            <Button
              onClick={() => save.mutate()}
              disabled={!changed || blocked !== null || drawing !== null || zones === null}
              loading={save.isPending}
            >
              Save zones
            </Button>
          </Group>
        </Group>
        {history && <Versions cam={cam} />}
      </Stack>
    </Modal>
  );
}
