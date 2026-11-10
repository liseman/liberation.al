import { activityHint, activityName } from '../lib/activity';
import { viewName } from '../review/views';
import classes from './ask.module.css';
import { AppLink, Stamp } from './bits';
import { clock, isRec, list, minutes, pct, plain, type Rec, records } from './format';
import { Lazy, Paged } from './Paged';
import { TextBlock } from './prose';

/** Per-bin bars. Heights are relative to the busiest bin; the title carries the real count. */
function Histogram({ bins: value, label }: { bins: unknown; label: string }) {
  const bins = records(value);
  const first = bins[0];
  const last = bins.at(-1);
  if (!first || !last) return null;
  const count = (b: Rec) => (typeof b.count === 'number' ? b.count : 0);
  const top = Math.max(1, ...bins.map(count));
  return (
    <>
      <div className={classes.hist} role="img" aria-label={label}>
        {bins.map((b, i) => (
          <span
            // biome-ignore lint/suspicious/noArrayIndexKey: bins are in time order and never move
            key={i}
            title={`${clock(b.start)} to ${clock(b.end)}: ${plain(b.count)}`}
            style={{ height: `${Math.max(2, (count(b) / top) * 100)}%` }}
          />
        ))}
      </div>
      <p className={classes.muted}>
        <span>{clock(first.start)}</span> to <span>{clock(last.end)}</span>
      </p>
    </>
  );
}

function CrossingCamera({ name, cam }: { name: string; cam: Rec }) {
  const counts = isRec(cam.crossings) ? cam.crossings : {};
  const directions = Object.keys(counts).filter(k => k !== 'total');
  const examples = records(cam.examples);
  const rates = list(cam.sample_rates_hz).map(plain).join(', ') || 'unknown';
  const motion = isRec(cam.camera_motion) ? cam.camera_motion : null;
  const gaps = list(cam.unrecorded_or_unscanned_intervals);
  return (
    <article className={classes.card}>
      <header className={classes.cardHeader}>
        <h4>{viewName(name)}</h4>
        <span className={classes.plainBadge}>{`Walked past ${plain(counts.total ?? 0)} times (estimate)`}</span>
      </header>
      <p>{directions.map(d => `${d.replace(/_/g, ' ')}: ${plain(counts[d])}`).join(' · ')}</p>
      <p className={classes.muted}>Line: {plain(cam.line_description)}</p>
      <p className={classes.muted}>
        {`Scanned ${minutes(cam.scanned_seconds)} of the requested window (${pct(cam.coverage_of_requested)}), sampled at ${rates} frames per second.`}
      </p>
      {!!motion?.uncertain && (
        <p className={classes.error}>
          {`Camera motion was detected in ${list(motion.flagged_files).length} recordings, so a fixed line is unreliable there. Treat this count with extra caution.`}
        </p>
      )}
      <Histogram bins={cam.histogram} label="Times walked past, over the window" />
      {examples.length > 0 && (
        <Lazy summary={`Times walked past (${examples.length}${cam.examples_truncated ? ', first shown' : ''})`}>
          {() => (
            <ul>
              <Paged
                items={examples}
                list
                render={e => (
                  <li>
                    <AppLink to={e.link}>
                      <Stamp value={e.ts} />
                    </AppLink>{' '}
                    {plain(e.direction).replace(/_/g, ' ')}
                  </li>
                )}
              />
            </ul>
          )}
        </Lazy>
      )}
      {gaps.length > 0 && (
        <Lazy summary={`Unscanned or unrecorded intervals (${gaps.length})`}>
          {() => (
            <ul>
              {gaps.slice(0, 60).map((g, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: intervals are in time order and never move
                <li key={i}>
                  {clock(list(g)[0])} to {clock(list(g)[1])}
                </li>
              ))}
            </ul>
          )}
        </Lazy>
      )}
    </article>
  );
}

/** Walked past: see-actions' walk-bys for the window, the only walk-by source. */
function CrossingResult({ b }: { b: Rec }) {
  const result = isRec(b.result) ? b.result : null;
  const cameras = result && isRec(result.cameras) ? Object.entries(result.cameras) : [];
  if (!result) return null;
  return (
    <section className={classes.answer} data-path="B">
      <h3 title={activityHint('crossing') ?? undefined}>{activityName('crossing')}</h3>
      <p className={classes.muted}>
        Counted from see-actions' tracking of each camera: walked past means a tracked person crossed the camera's
        counting line.
      </p>
      <TextBlock value={typeof b.answer === 'string' ? b.answer : ''} />
      {!!result.partial && (
        <p className={classes.muted}>
          Partial: see-actions wasn't watching for some of the requested window, so these counts cover only what it saw.
        </p>
      )}
      <div className={classes.grid}>
        {cameras.map(([n, c]) => isRec(c) && <CrossingCamera key={n} name={n} cam={c} />)}
      </div>
      <p className={classes.muted}>
        {plain(result.cross_camera)} {plain(result.notice)}
      </p>
    </section>
  );
}

export function Crossings({ crossings }: { crossings: Rec }) {
  const b = isRec(crossings.path_b) ? crossings.path_b : null;
  return (
    <>
      {b && isRec(b.result) && <CrossingResult b={b} />}
      {b && !isRec(b.result) && <p className={classes.muted}>Walked-past count unavailable: {plain(b.reason)}</p>}
    </>
  );
}
