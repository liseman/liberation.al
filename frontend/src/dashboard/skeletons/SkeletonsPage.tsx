// Skeleton playback, a test page at /skeletons that's not in the nav: pick a recent action or walk-by span and
// watch its track's poses as a stick figure at real speed, with LEAD_S either side. Pose rows outlive the 30-day video
// window, so this plays spans whose video is gone.
import {
  Alert,
  Button,
  Checkbox,
  Container,
  Group,
  Loader,
  ScrollArea,
  Slider,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useRef, useState } from 'react';
import { PageError } from '../lib/page';
import { wireframe } from '../pose/skeleton';
import { type Frame, fitView, frameAt, getPoses, getSpans, LEAD_S, type Span } from './playback';

const ASPECT = 16 / 9;
const clock = (s: number) => new Date(s * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'medium' });

/** Draws `frame`'s skeleton, showing the part of the frame `view` names ([x, y, side], as `fitView` gives). */
function draw(canvas: HTMLCanvasElement, frame: Frame | null, inSpan: boolean, view: [number, number, number]): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { width: w, height: h } = canvas;
  const [vx, vy, side] = view;
  const px = (x: number) => ((x - vx) / side) * w;
  const py = (y: number) => ((y - vy) / side) * h;
  const style = getComputedStyle(canvas);
  ctx.fillStyle = style.getPropertyValue('--mantine-color-body') || '#fff';
  ctx.fillRect(0, 0, w, h);
  if (!frame) return;
  const { bones, joints } = wireframe(frame);
  ctx.strokeStyle = ctx.fillStyle = inSpan
    ? style.getPropertyValue('--mantine-color-blue-filled') || '#228be6'
    : style.getPropertyValue('--mantine-color-dimmed') || '#868e96';
  ctx.lineWidth = Math.max(2, w / 300);
  ctx.lineCap = 'round';
  for (const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] of bones) {
    ctx.beginPath();
    ctx.moveTo(px(x1), py(y1));
    ctx.lineTo(px(x2), py(y2));
    ctx.stroke();
  }
  for (const [x = 0, y = 0] of joints) {
    ctx.beginPath();
    ctx.arc(px(x), py(y), ctx.lineWidth * 1.5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function Player({ span }: { span: Span }) {
  const poses = useQuery({ queryKey: ['skeletons', 'poses', span.id], queryFn: () => getPoses(span) });
  const from = span.start - LEAD_S;
  const to = span.end + LEAD_S;
  const [t, setTime] = useState(from);
  const at = useRef(from); // the clock the animation advances, so a frame's step never waits on a render
  const [playing, setPlaying] = useState(true);
  const canvas = useRef<HTMLCanvasElement>(null);
  const frames = poses.data?.frames ?? [];
  const [fitted, setFitted] = useState(true);
  const around = useMemo(() => fitView(poses.data?.frames ?? []), [poses.data]);
  const view: [number, number, number] = fitted ? around : [0, 0, 1];
  const setT = (v: number) => {
    at.current = v;
    setTime(v);
  };

  useEffect(() => {
    if (!playing || !poses.data) return;
    let last = performance.now();
    let id = requestAnimationFrame(function tick(now) {
      at.current = Math.min(at.current + (now - last) / 1000, to);
      last = now;
      setTime(at.current);
      if (at.current >= to) setPlaying(false);
      else id = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(id);
  }, [playing, poses.data, to]);
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const width = c.clientWidth * devicePixelRatio;
    if (c.width !== Math.round(width)) {
      c.width = Math.round(width);
      c.height = Math.round(width / ASPECT);
    }
    draw(c, frameAt(frames, t), t >= span.start && t < span.end, view);
  });

  const where = t < span.start ? 'lead-in' : t >= span.end ? 'lead-out' : span.kind;
  return (
    <Stack gap="xs">
      <Text fw={600}>
        {span.kind} on {span.cam}
        {span.person ? `, ${span.person}` : ''} (track {span.track_id})
      </Text>
      <canvas
        ref={canvas}
        aria-label={`Skeleton of track ${span.track_id}`}
        style={{ width: '100%', aspectRatio: String(ASPECT), border: '1px solid var(--mantine-color-default-border)' }}
      />
      {poses.isPending && <Loader size="sm" />}
      {poses.error && <PageError title="Couldn't load the poses" detail={poses.error.message} />}
      {poses.data && frames.length === 0 && <Alert color="yellow">No pose rows for this track in the span.</Alert>}
      <Group gap="sm" wrap="nowrap">
        <Button
          size="xs"
          w={72}
          onClick={() => {
            if (!playing && t >= to) setT(from);
            setPlaying(!playing);
          }}
        >
          {playing ? 'Pause' : 'Play'}
        </Button>
        <Slider
          style={{ flex: 1 }}
          min={from}
          max={to}
          step={0.01}
          value={t}
          label={null}
          marks={[{ value: span.start }, { value: span.end }]}
          onChange={v => {
            setPlaying(false);
            setT(v);
          }}
          aria-label="Playback position"
        />
        <Checkbox
          size="xs"
          label="Fit to the person"
          checked={fitted}
          onChange={e => setFitted(e.currentTarget.checked)}
        />
        <Text size="sm" c="dimmed" w={150} ta="right">
          {(t - span.start).toFixed(1)} s, {where}
        </Text>
      </Group>
    </Stack>
  );
}

export function SkeletonsPage() {
  const spans = useQuery({ queryKey: ['skeletons', 'spans'], queryFn: () => getSpans() });
  const [picked, setPicked] = useState<Span | null>(null);
  return (
    <Container size="xl" py="md">
      <Title order={2} mb="sm">
        Skeleton playback
      </Title>
      <Group align="flex-start" wrap="wrap" gap="lg">
        <ScrollArea h="75vh" style={{ flex: '1 1 340px', maxWidth: 520 }}>
          {spans.isPending && <Loader size="sm" />}
          {spans.error && <PageError title="Couldn't load movement spans" detail={spans.error.message} />}
          {spans.data?.length === 0 && <Text c="dimmed">No movement spans in the last 24 hours.</Text>}
          {!!spans.data?.length && (
            <Table highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>When</Table.Th>
                  <Table.Th>Camera</Table.Th>
                  <Table.Th>Kind</Table.Th>
                  <Table.Th>Person</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {spans.data.map(s => (
                  <Table.Tr
                    key={s.id}
                    onClick={() => setPicked(s)}
                    style={{ cursor: 'pointer' }}
                    bg={picked?.id === s.id ? 'var(--mantine-primary-color-light)' : undefined}
                  >
                    <Table.Td>{clock(s.start)}</Table.Td>
                    <Table.Td>{s.cam}</Table.Td>
                    <Table.Td>{s.kind}</Table.Td>
                    <Table.Td>{s.person ?? ''}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          )}
        </ScrollArea>
        <div style={{ flex: '2 1 480px', minWidth: 0 }}>
          {picked ? <Player key={picked.id} span={picked} /> : <Text c="dimmed">Pick a span to play it.</Text>}
        </div>
      </Group>
    </Container>
  );
}
