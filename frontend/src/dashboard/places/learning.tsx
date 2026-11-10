// What "Learn surroundings" is doing right now, on the Places page (#283, #294): a "Learning" badge on a place while
// the switch is on, moving only while a map step runs, and a panel beside the place list. The panel leads with one
// state (up to date, frames waiting, placing views, training the splat, or off) and what happens next, with a bar
// only while a step runs: measured while the step says how far it is (the splat's training), unmeasured otherwise.
// Under it are the numbers (views in the map, frames kept, waiting, and parked, and when the map last improved) and
// a row a camera: its name, how it's doing in a few words (a parked frame and why, keeping frames, or why none was
// kept lately), and its newest kept frame as a small picture, which the server paints over wherever someone was in
// view and withholds when it can't. GET /surroundings/status (dashboard/surroundings.py) is the map step's own
// summary: a few small files a read, so it's asked often.
import { Badge, Loader, Progress, Tooltip } from '@mantine/core';
import { IconCamera } from '@tabler/icons-react';
import { queryOptions, useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { request } from '../../shared/http';
import { ago, capitalize, momentWords, plural } from '../lib/format';
import { viewName } from '../review/views';
import classes from './learning.module.css';

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
  /** Kept frames no step is due for: nothing places them until the map grows or their camera's spot changes. */
  parked?: number;
  /** For each camera with a parked frame, why (a word of the map step's PARKED). */
  parked_cameras?: Record<string, string>;
  running: boolean;
  step: LearningStep | null;
  /** When the last step ended (epoch seconds), or null. */
  last_step: number | null;
  cameras: Record<string, LearningCamera>;
  /** Each learned place's map: the views in it, and when a step last added one (epoch seconds, or null). */
  places?: Record<string, { views: number; improved: number | null }>;
  /** Each camera's newest kept frame, by id, when a picture of it may be shown (GET /surroundings/thumb/<id>.jpg). */
  frames?: Record<string, string>;
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

/** How a state or a camera reads at a glance: fine, wants the owner's attention, or nothing to say. */
export type Tone = 'ok' | 'warn' | 'none';

/** The one state the panel leads with: what it is in a few words, and what's being done or happens next. */
export interface LearningState {
  kind: 'off' | 'placing' | 'training' | 'writing' | 'working' | 'unchosen' | 'waiting' | 'watching' | 'current';
  tone: Tone;
  title: string;
  next: string;
}

/** The views in every learned place's map, and when a step last added one to any of them (or null). */
export function mapTotals(status: LearningStatus): { views: number; improved: number | null } {
  const places = Object.values(status.places ?? {});
  const improved = places.map(p => p.improved).filter((t): t is number => typeof t === 'number' && t > 0);
  return {
    views: places.reduce((n, p) => n + (p.views || 0), 0),
    improved: improved.length ? Math.max(...improved) : null,
  };
}

/** The state to lead with: off, else the running step's work, else no camera chosen, the frames waiting, or that
 *  it's up to date. */
export function learningState(status: LearningStatus): LearningState {
  if (!status.on && !status.running)
    return {
      kind: 'off',
      tone: 'none',
      title: 'Learning is off',
      next: 'Turn on Learn surroundings on the Live tab to carry on.',
    };
  const step = status.step;
  if (status.running) {
    switch (step?.phase) {
      case 'pairing':
        return {
          kind: 'placing',
          tone: 'ok',
          title: `Placing ${newViews(step.frames)}`,
          next: 'Matching them against the map. Training comes next.',
        };
      case 'depth':
        return {
          kind: 'placing',
          tone: 'ok',
          title: `Placing ${newViews(step.frames)}`,
          next: 'Measuring their depth. Training comes next.',
        };
      case 'training': {
        const views = newViews(step.views);
        const adding = `adding ${views}`;
        if (step.step != null && step.steps) {
          const pct = Math.round((step.step / step.steps) * 100);
          return { kind: 'training', tone: 'ok', title: 'Building the 3D map', next: `${pct}% done, ${adding}.` };
        }
        return { kind: 'training', tone: 'ok', title: 'Building the 3D map', next: `${capitalize(adding)}.` };
      }
      case 'writing':
        return { kind: 'writing', tone: 'ok', title: 'Saving the map', next: 'Writing the improved map.' };
      default:
        return { kind: 'working', tone: 'ok', title: 'Working on the map', next: 'A step is running.' };
    }
  }
  if (!status.sampling.length)
    return {
      kind: 'unchosen',
      tone: 'warn',
      title: 'No camera chosen',
      next: 'Choose the cameras to learn from beside Learn surroundings on the Live tab.',
    };
  if (status.waiting > 0)
    return {
      kind: 'waiting',
      tone: 'ok',
      title: `${plural(status.waiting, 'picture', 'pictures')} waiting`,
      next: 'The next step starts by itself.',
    };
  const watching = `Watching ${plural(status.sampling.length, 'camera', 'cameras')}`;
  if (status.kept === 0 && mapTotals(status).views === 0)
    return {
      kind: 'watching',
      tone: 'none',
      title: 'Nothing kept yet',
      next: `${watching} for views worth keeping.`,
    };
  return { kind: 'current', tone: 'ok', title: 'Up to date', next: `${watching} for new views.` };
}

/** How far the running step is, 0 to 100, or null when it doesn't say (or nothing runs). */
export function measured(status: LearningStatus): number | null {
  const step = status.step;
  if (!status.running || step?.phase !== 'training' || step.step == null || !step.steps) return null;
  return Math.max(0, Math.min(100, (step.step / step.steps) * 100));
}

/** The numbers, in the order shown, each with a short label and what it counts in full. Waiting and parked are
 *  the kept frames a step is, and isn't, due for. */
export function learningStats(status: LearningStatus): { key: string; label: string; full: string; value: number }[] {
  return [
    { key: 'views', label: 'In the map', full: "Views in this place's 3D map", value: mapTotals(status).views },
    { key: 'kept', label: 'Pictures kept', full: 'Pictures kept to learn from', value: status.kept },
    {
      key: 'waiting',
      label: 'Waiting',
      full: 'Pictures kept, waiting to be placed in the map',
      value: status.waiting,
    },
    {
      key: 'parked',
      label: "Can't be placed",
      full: "Pictures that can't be placed until the map grows or the camera's spot changes",
      value: status.parked ?? 0,
    },
  ];
}

/** Why a camera's kept frame isn't in the map, by the map step's word for it: in a few words, and in full. */
const PARKED: Record<string, [label: string, full: string]> = {
  overlap: [
    'No spot on the plan',
    'Its view overlaps nothing in the map yet, and it has no spot on the plan to place it by.',
  ],
  facing: ['Needs a facing', 'Its spot on the plan has no facing. Give it one in the plan editor.'],
  height: ['Needs a height', 'Its spot on the plan has no height. Give it one in the plan editor.'],
  lens: ['Lens not known yet', "Its lens isn't known yet."],
  tilt: ['Working out its tilt', 'Too few people have been seen walking past it to work out its tilt.'],
  check: ["Setup doesn't fit", "Its height, lens, or tilt doesn't fit the people it sees."],
  frame: ["Plan can't hold a map", "The plan's frame can't hold a map."],
  floor: ["Floor doesn't match", "The floor it sees doesn't sit on the plan's floor. Check its height on the plan."],
};

/** The outcomes of a look that kept its frame. */
const KEPT = ['view', 'changed', 'sharper'];
/** Why a look didn't keep its frame, in a few words; several outcomes are one reason. */
const WHY: Record<string, string> = {
  same: 'Nothing new in view',
  copy: 'Nothing new in view',
  person: 'Person or animal in view',
  dark: 'Too dark',
  blurred: 'Picture is blurred',
};

/** One camera, as its row shows it: how it's doing in a few words, more for a pointer that rests on it, and its
 *  newest kept frame when a picture of it may be shown. */
export interface CameraRow {
  camera: string;
  tone: Tone;
  label: string;
  detail: string | null;
  frame: string | null;
}

/** A row for each sampled camera, in the order sampled, then for any other camera with a parked frame. A parked
 *  frame comes first (it's what the owner can act on), then that frames are being kept, then why none was lately,
 *  by what most of the camera's recent looks came to. */
export function cameraRows(status: LearningStatus, now = Date.now() / 1000): CameraRow[] {
  const parked = status.parked_cameras ?? {};
  const names = [
    ...status.sampling,
    ...Object.keys(parked)
      .filter(c => !status.sampling.includes(c))
      .sort(),
  ];
  return names.map(camera => {
    const seen = status.cameras[camera];
    const frame = status.frames?.[camera] ?? null;
    const kept = seen?.last_kept ? `Last kept a picture ${ago(seen.last_kept, now)}.` : null;
    const why = parked[camera];
    if (why != null) {
      const [label, full] = PARKED[why] ?? ["Can't be placed yet", "Its kept frame can't be placed in the map yet."];
      return { camera, tone: 'warn', label, detail: full, frame };
    }
    const recent = seen?.recent ?? {};
    if (KEPT.some(k => (recent[k] ?? 0) > 0))
      return { camera, tone: 'ok', label: 'Keeping pictures', detail: kept, frame };
    const byReason = new Map<string, number>();
    for (const [outcome, n] of Object.entries(recent)) {
      const reason = WHY[outcome];
      if (reason && n > 0) byReason.set(reason, (byReason.get(reason) ?? 0) + n);
    }
    const top = [...byReason.entries()].sort((a, b) => b[1] - a[1])[0];
    return { camera, tone: 'none', label: top ? top[0] : 'Not looked at yet', detail: kept, frame };
  });
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
      size="sm"
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

/** A camera's newest kept frame, small, or a plain tile when there's none to show (or it didn't load). */
function Thumb({ row }: { row: CameraRow }) {
  const [failed, setFailed] = useState<string | null>(null);
  if (!row.frame || failed === row.frame)
    return (
      <span className={classes.thumb} data-empty aria-hidden>
        <IconCamera size={16} stroke={1.5} />
      </span>
    );
  return (
    <img
      className={classes.thumb}
      src={`/surroundings/thumb/${row.frame}.jpg`}
      alt={`The newest view kept from ${viewName(row.camera)}`}
      width={64}
      height={36}
      loading="lazy"
      decoding="async"
      onError={() => setFailed(row.frame)}
    />
  );
}

/** What learning is doing, beside the place list: one state with what happens next, the numbers, and a row a
 *  camera. Nothing is drawn before the first answer, or while it's off with nothing ever learned. */
export function LearningPanel({ status }: { status: LearningStatus | undefined }) {
  if (!status) return null;
  const totals = mapTotals(status);
  if (!status.on && !status.running && status.kept === 0 && totals.views === 0) return null;
  const state = learningState(status);
  const done = measured(status);
  const rows = status.on ? cameraRows(status) : [];
  return (
    <section className={classes.panel} aria-labelledby="learning-h">
      <h3 id="learning-h" className={classes.title}>
        Learning surroundings
      </h3>
      <div className={classes.state} role="status" data-state={state.kind}>
        {status.running ? (
          <Loader size={14} className={classes.spinner} aria-hidden />
        ) : (
          <span className={classes.dot} data-tone={state.tone} data-halo aria-hidden />
        )}
        <div className={classes.stateText}>
          <p className={classes.headline}>{state.title}</p>
          <p className={classes.next}>{state.next}</p>
        </div>
      </div>
      {status.running &&
        (done == null ? (
          // unmeasured: a full striped bar that moves, since the step doesn't say how far it is
          <Progress
            className={classes.bar}
            value={100}
            size={8}
            radius="xl"
            striped
            animated
            aria-label="A map step is running"
          />
        ) : (
          <div className={classes.measured}>
            <Progress
              className={classes.bar}
              value={done}
              size={8}
              radius="xl"
              bg="var(--lib-track)"
              aria-label="How far along the 3D map build is"
            />
            <span className={classes.percent}>{`${Math.round(done)}%`}</span>
          </div>
        ))}
      <dl className={classes.stats}>
        {learningStats(status).map(stat => (
          <Tooltip
            key={stat.key}
            label={stat.full}
            events={{ hover: true, focus: true, touch: true }}
            multiline
            maw={280}
          >
            <div className={classes.stat} data-zero={stat.value === 0 || undefined}>
              <dt>{stat.label}</dt>
              <dd>{count(stat.value)}</dd>
            </div>
          </Tooltip>
        ))}
      </dl>
      {totals.improved != null && (
        <p className={classes.improved}>
          Map last improved{' '}
          <time dateTime={new Date(totals.improved * 1000).toISOString()} title={momentWords(totals.improved)}>
            {ago(totals.improved)}
          </time>
        </p>
      )}
      {rows.length > 0 && (
        <ul className={classes.cameras} aria-label="Cameras learned from">
          {rows.map(row => (
            <li key={row.camera} className={classes.camera}>
              <Thumb row={row} />
              <div className={classes.cameraText}>
                <span className={classes.cameraName}>{viewName(row.camera)}</span>
                <span className={classes.cameraStatus}>
                  <span className={classes.dot} data-tone={row.tone} aria-hidden />
                  {row.label}
                </span>
                {row.tone === 'warn' && row.detail && <span className={classes.cameraProblem}>{row.detail}</span>}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
