// What "Learn surroundings" is doing right now, on the Places page (#283): a "Learning" badge on a place while the
// switch is on, moving only while a map step runs, and a panel that says in plain words what's going on (the
// cameras sampled, the frames kept and waiting, what the running step is doing, or since when the map is up to
// date), with a bar: measured while the step says how far it is (the splat's training), unmeasured otherwise. For
// a sampled camera none of whose recent frames was kept, it says why. GET /surroundings/status
// (dashboard/surroundings.py) is the map step's own summary: a few small files a read, so it's asked often.
import { Badge, Loader, Paper, Progress, Stack, Text } from '@mantine/core';
import { queryOptions, useQuery } from '@tanstack/react-query';
import { request } from '../../shared/http';
import { hm, plural } from '../lib/format';
import { viewName } from '../review/views';

/** What a running map step says it's doing; each part null while it hasn't said. */
export interface LearningStep {
  phase: 'pairing' | 'depth' | 'training' | 'writing' | null;
  /** The waiting frames the step took. */
  frames: number | null;
  /** The place being trained: null for one being founded, and before training. */
  place: string | null;
  /** The new views going into it. */
  views: number | null;
  /** The training's step, of `steps`. */
  step: number | null;
  steps: number | null;
}

/** A sampled camera's last looks, counted by what each came to, and when a frame of it was last kept. */
export interface LearningCamera {
  recent: Record<string, number>;
  last_kept: number | null;
}

export interface LearningStatus {
  on: boolean;
  /** The cameras sampled now; none while off. */
  sampling: string[];
  /** Frames kept in all, and how many of them aren't in a place yet. */
  kept: number;
  waiting: number;
  running: boolean;
  step: LearningStep | null;
  /** When the last step ended (epoch seconds), or null. */
  last_step: number | null;
  cameras: Record<string, LearningCamera>;
}

export const LEARNING_RUNNING_MS = 3_000;
export const LEARNING_ON_MS = 10_000;
export const LEARNING_OFF_MS = 60_000;

/** How long until the next read: often while a step runs (its bar moves), less so while learning is only on, and
 *  seldom while it's off, to notice that it was turned on. */
export const learningEvery = (status: LearningStatus | undefined): number =>
  status?.running ? LEARNING_RUNNING_MS : status?.on ? LEARNING_ON_MS : LEARNING_OFF_MS;

export const learningQuery = () =>
  queryOptions({
    queryKey: ['surroundings', 'status'],
    queryFn: ({ signal }) => request<LearningStatus>('/surroundings/status', undefined, signal),
    refetchInterval: q => learningEvery(q.state.data),
    // a page nobody is looking at asks nothing
    refetchIntervalInBackground: false,
  });

const count = (n: number) => n.toLocaleString('en-US');

/** "2 new views", or "new views" when the step hasn't said how many. */
const newViews = (n: number | null) => (n == null ? 'new views' : `${count(n)} new ${n === 1 ? 'view' : 'views'}`);

/** What's being done, in a sentence: the running step's work, else the frames waiting, else that it's up to date. */
export function activityText(status: LearningStatus): string {
  const step = status.step;
  if (status.running) {
    switch (step?.phase) {
      case 'pairing':
        return `Placing ${newViews(step.frames)}: matching them against the map.`;
      case 'depth':
        return `Placing ${newViews(step.frames)}: measuring their depth.`;
      case 'training': {
        const far = step.step != null && step.steps ? ` (step ${count(step.step)} of ${count(step.steps)})` : '';
        return `Training the splat with ${newViews(step.views)}${far}.`;
      }
      case 'writing':
        return 'Writing the new map and splat.';
      default:
        return 'Working on the map.';
    }
  }
  if (status.waiting > 0)
    return `${plural(status.waiting, 'kept frame', 'kept frames')} waiting to be placed. The next step starts by itself.`;
  if (status.last_step != null) return `Up to date since ${hm(status.last_step)}.`;
  return status.kept > 0 ? 'Up to date.' : 'Watching for views worth keeping. Nothing kept yet.';
}

/** How far the running step is, 0 to 100, or null when it doesn't say (or nothing runs). */
export function measured(status: LearningStatus): number | null {
  const step = status.step;
  if (!status.running || step?.phase !== 'training' || step.step == null || !step.steps) return null;
  return Math.max(0, Math.min(100, (step.step / step.steps) * 100));
}

/** "Sampling 2 cameras · 37 frames kept, 3 waiting." */
export function samplingText(status: LearningStatus): string {
  const cams = status.sampling.length;
  const sampled = cams ? `Sampling ${plural(cams, 'camera', 'cameras')}` : 'No camera is chosen to learn from';
  const waiting = status.waiting > 0 ? `, ${count(status.waiting)} waiting` : '';
  return `${sampled} · ${plural(status.kept, 'frame', 'frames')} kept${waiting}.`;
}

/** The outcomes of a look that kept its frame. */
const KEPT = ['view', 'changed', 'sharper'];
/** Why a look didn't keep its frame, in words; several outcomes are one reason. */
const WHY: Record<string, string> = {
  same: 'nothing new in view',
  copy: 'nothing new in view',
  person: 'a person or an animal is in view',
  dark: "it's too dark",
  blurred: 'the picture is blurred',
};

/** For each sampled camera none of whose recent frames was kept: why, by what most of its recent looks came to.
 *  A camera that kept one lately, or that hasn't been looked at yet, isn't listed. */
export function cameraReasons(status: LearningStatus): { camera: string; why: string }[] {
  const out: { camera: string; why: string }[] = [];
  for (const camera of status.sampling) {
    const recent = status.cameras[camera]?.recent ?? {};
    if (KEPT.some(k => (recent[k] ?? 0) > 0)) continue;
    const byReason = new Map<string, number>();
    for (const [outcome, n] of Object.entries(recent)) {
      const why = WHY[outcome];
      if (why && n > 0) byReason.set(why, (byReason.get(why) ?? 0) + n);
    }
    const top = [...byReason.entries()].sort((a, b) => b[1] - a[1])[0];
    if (top) out.push({ camera, why: top[0] });
  }
  return out;
}

/** Whether the running step is (or may be) working on `place`: before training it hasn't said which place, and a
 *  place being founded has no id yet. */
export const workingOn = (status: LearningStatus | undefined, place: string): boolean =>
  !!status?.running && (status.step?.place == null || status.step.place === place);

/** The status, asked for while the Places page is open. */
export function useLearning(): LearningStatus | undefined {
  return useQuery(learningQuery()).data;
}

/** "Learning", on a place while the switch is on: with a spinner only while a map step is working on it. */
export function LearningBadge({ status, place }: { status: LearningStatus | undefined; place: string }) {
  if (!status?.on) return null;
  const working = workingOn(status, place);
  return (
    <Badge
      size="xs"
      variant="light"
      color={working ? 'blue' : 'gray'}
      tt="none"
      fw={500}
      leftSection={working ? <Loader size={9} color="blue" aria-hidden /> : undefined}
      title={working ? 'A map step is running now' : 'Learn surroundings is on'}
      data-working={working}
    >
      Learning
    </Badge>
  );
}

/** What learning is doing, while the switch is on: nothing is drawn while it's off. */
export function LearningPanel({ status }: { status: LearningStatus | undefined }) {
  if (!status?.on) return null;
  const done = measured(status);
  const reasons = cameraReasons(status);
  return (
    <Paper withBorder radius="md" p="sm" mb="sm" role="status" aria-label="Learning surroundings">
      <Stack gap={6}>
        <Text size="sm" fw={500}>
          {activityText(status)}
        </Text>
        {status.running &&
          (done == null ? (
            // unmeasured: a full striped bar that moves, since the step doesn't say how far it is
            <Progress value={100} size="sm" striped animated aria-label="A map step is running" />
          ) : (
            <Progress value={done} size="sm" aria-label="How far the splat's training is" />
          ))}
        <Text size="xs" c="dimmed">
          {samplingText(status)}
        </Text>
        {reasons.map(r => (
          <Text size="xs" c="dimmed" key={r.camera}>
            {viewName(r.camera)}: no frames kept lately, {r.why}.
          </Text>
        ))}
      </Stack>
    </Paper>
  );
}
