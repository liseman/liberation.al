// What "Learn surroundings" is doing right now, on the Places page (#283, #294): a "Learning" badge on a place while
// the switch is on, and a panel beside the place list. Learning collects pictures from the chosen cameras, for the 3D
// map, which is being rebuilt with a new method: nothing is built from them at present. The panel says whether it's
// collecting, how many pictures are kept, and has a row a camera: its name, how it's doing in a few words (keeping
// pictures, or why none was kept lately), and its newest kept picture, which the server paints over wherever someone
// was in view and withholds when it can't. GET /surroundings/status (dashboard/surroundings.py) is the sampler's own
// status: two small files a read, so it's asked often.
import { Badge, Tooltip } from '@mantine/core';
import { IconCamera } from '@tabler/icons-react';
import { queryOptions, useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { request } from '../../shared/http';
import { ago, plural } from '../lib/format';
import { viewName } from '../review/views';
import classes from './learning.module.css';

/** A sampled camera's last looks, counted by what each came to, and when a frame of it was last kept. */
export interface LearningCamera {
  recent: Record<string, number>;
  last_kept: number | null;
}

export interface LearningStatus {
  on: boolean;
  /** The cameras sampled now; none while off. */
  sampling: string[];
  /** Pictures kept in all. */
  kept: number;
  cameras: Record<string, LearningCamera>;
  /** Each camera's newest kept frame, by id, when a picture of it may be shown (GET /surroundings/thumb/<id>.jpg). */
  frames?: Record<string, string>;
}

export const LEARNING_ON_MS = 10_000;
export const LEARNING_OFF_MS = 60_000;

/** How long until the next read: less often while it's off, to notice that it was turned on. */
export const learningEvery = (status: LearningStatus | undefined): number =>
  status?.on ? LEARNING_ON_MS : LEARNING_OFF_MS;

export const learningQuery = () =>
  queryOptions({
    queryKey: ['surroundings', 'status'],
    queryFn: ({ signal }) => request<LearningStatus>('/surroundings/status', undefined, signal),
    refetchInterval: q => learningEvery(q.state.data),
    // a page nobody is looking at asks nothing
    refetchIntervalInBackground: false,
  });

const count = (n: number) => n.toLocaleString('en-US');

/** How a state or a camera reads at a glance: fine, wants the owner's attention, or nothing to say. */
export type Tone = 'ok' | 'warn' | 'none';

/** The one state the panel leads with: what it is in a few words, and what happens next. */
export interface LearningState {
  kind: 'off' | 'unchosen' | 'collecting';
  tone: Tone;
  title: string;
  next: string;
}

/** What the 3D map is doing, said wherever learning is: nothing is built from the pictures at present. */
export const REBUILDING = 'The 3D map is being rebuilt with a new method, which will use these pictures.';

/** The state to lead with: off, no camera chosen, or collecting. */
export function learningState(status: LearningStatus): LearningState {
  if (!status.on)
    return {
      kind: 'off',
      tone: 'none',
      title: 'Not collecting pictures',
      next: 'Turn on Learn surroundings on the Live tab to collect them.',
    };
  if (!status.sampling.length)
    return {
      kind: 'unchosen',
      tone: 'warn',
      title: 'No camera chosen',
      next: 'Choose the cameras to collect pictures from beside Learn surroundings on the Live tab.',
    };
  return {
    kind: 'collecting',
    tone: 'ok',
    title: `Collecting pictures from ${plural(status.sampling.length, 'camera', 'cameras')}`,
    next: REBUILDING,
  };
}

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

/** A row for each sampled camera, in the order sampled: that pictures are being kept, else why none was lately, by
 *  what most of the camera's recent looks came to. */
export function cameraRows(status: LearningStatus, now = Date.now() / 1000): CameraRow[] {
  return status.sampling.map(camera => {
    const seen = status.cameras[camera];
    const frame = status.frames?.[camera] ?? null;
    const kept = seen?.last_kept ? `Last kept a picture ${ago(seen.last_kept, now)}.` : null;
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

/** The status, asked for while the Places page is open. */
export function useLearning(): LearningStatus | undefined {
  return useQuery(learningQuery()).data;
}

/** "Learning", on a place while the switch is on. */
export function LearningBadge({ status }: { status: LearningStatus | undefined }) {
  if (!status?.on) return null;
  return (
    <Badge
      size="sm"
      variant="light"
      color="gray"
      tt="none"
      fw={500}
      title="Learn surroundings is on: collecting pictures"
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

/** What learning is doing, beside the place list: one state with what happens next, the pictures kept, and a row a
 *  camera. Nothing is drawn before the first answer, or while it's off with nothing kept. */
export function LearningPanel({ status }: { status: LearningStatus | undefined }) {
  if (!status) return null;
  if (!status.on && status.kept === 0) return null;
  const state = learningState(status);
  const rows = status.on ? cameraRows(status) : [];
  return (
    <section className={classes.panel} aria-labelledby="learning-h">
      <h3 id="learning-h" className={classes.title}>
        Learning surroundings
      </h3>
      <div className={classes.state} role="status" data-state={state.kind}>
        <span className={classes.dot} data-tone={state.tone} data-halo aria-hidden />
        <div className={classes.stateText}>
          <p className={classes.headline}>{state.title}</p>
          <p className={classes.next}>{state.next}</p>
        </div>
      </div>
      <dl className={classes.stats}>
        <Tooltip
          label="Pictures kept for the 3D map"
          events={{ hover: true, focus: true, touch: true }}
          multiline
          maw={280}
        >
          <div className={classes.stat} data-zero={status.kept === 0 || undefined}>
            <dt>Pictures kept</dt>
            <dd>{count(status.kept)}</dd>
          </div>
        </Tooltip>
      </dl>
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
