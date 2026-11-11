// A finished (or partial) Ask result: the answer, warnings, the search rounds, history coverage, and the recorded
// evidence behind it.
import type { ReactNode } from 'react';
import { Disclosure } from '../lib/Disclosure';
import { shownName } from '../lib/speakers';
import { viewName } from '../review/views';
import classes from './ask.module.css';
import { AppLink, type NewSetting, NewTag, Stamp, StatusBadge } from './bits';
import { Crossings } from './Crossings';
import { isRec, label, list, noteText, plain, records, roundText } from './format';
import { Paged } from './Paged';
import { inline, TextBlock } from './prose';

function Notes({ title, items, collapsed = false }: { title: string; items: unknown; collapsed?: boolean }) {
  if (!Array.isArray(items) || !items.length) return null;
  const body = (
    <ul>
      <Paged
        items={items}
        list
        render={x => {
          const text = noteText(x);
          return text && <li>{inline(text)}</li>;
        }}
      />
    </ul>
  );
  return collapsed ? (
    <Disclosure className={classes.details} summary={title}>
      {body}
    </Disclosure>
  ) : (
    <section className={classes.notes}>
      <h3>{title}</h3>
      {body}
    </section>
  );
}

/** Any JSON value as nested definition lists, bounded in depth and breadth. */
export function Facts({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null || value === undefined) return <span>Not available</span>;
  if (typeof value !== 'object')
    return <span>{typeof value === 'boolean' ? (value ? 'Yes' : 'No') : plain(value)}</span>;
  if (depth > 3) return <span>Additional details omitted</span>;
  if (Array.isArray(value))
    return (
      <ul>
        <Paged
          items={value}
          list
          render={v => (
            <li>
              <Facts value={v} depth={depth + 1} />
            </li>
          )}
        />
      </ul>
    );
  return (
    <dl className={classes.facts}>
      {Object.entries(value)
        .slice(0, 40)
        .map(([k, v]) => (
          <div key={k} className={classes.fact}>
            <dt>{label(k)}</dt>
            <dd>
              {(k === 'first' || k === 'last') && typeof v === 'number' ? (
                <Stamp value={v} />
              ) : (
                <Facts value={v} depth={depth + 1} />
              )}
            </dd>
          </div>
        ))}
    </dl>
  );
}

const EVIDENCE_PATHS: Record<string, NewSetting> = { journal: 'journal.enabled', facts: 'journal.facts' };

/** One piece of recorded evidence: when (linking into Review), who or which camera, and what it says. */
export function Evidence({ item }: { item: unknown }) {
  if (!isRec(item)) return null;
  const stamp = <Stamp value={item.timestamp} />;
  const meta = [
    item.speaker !== undefined ? `Speaker: ${shownName(plain(item.speaker))}` : '',
    item.camera !== undefined ? `Camera: ${viewName(plain(item.camera))}` : '',
    item.id !== undefined ? `ID: ${plain(item.id)}` : '',
  ]
    .filter(Boolean)
    .join(' · ');
  const source = plain(item.source);
  const tag = Object.hasOwn(EVIDENCE_PATHS, source) ? EVIDENCE_PATHS[source] : undefined;
  return (
    <article className={classes.evidence}>
      <div>
        <AppLink to={item.href}>{stamp}</AppLink>
        <span>{meta ? ` · ${meta}` : ''}</span>
        {tag && (
          <>
            {' '}
            <NewTag setting={tag} />
          </>
        )}
      </div>
      {item.text !== undefined && <TextBlock value={item.text} />}
    </article>
  );
}

/** The answer with each citation of recorded evidence ("C6", "J3") linked to that item's moment in Review, as the
 *  evidence list links it: a code no evidence item has stays text, and code spans and links already in the answer
 *  are left alone. */
export function citeLinks(text: string, evidence: readonly unknown[]): string {
  const to = new Map<string, string>();
  for (const e of evidence) {
    if (!isRec(e)) continue;
    const target = typeof e.href === 'string' ? e.href : typeof e.link === 'string' ? e.link : '';
    if (!target || /[\s()]/.test(target)) continue; // a Markdown link can't hold it
    for (const code of [e.id, e.label]) if (typeof code === 'string' && code) to.set(code, target);
  }
  if (!to.size) return text;
  return text
    .split(/(`[^`\n]+`|\[[^\]\n]+\]\([^\s)]+\))/)
    .map((part, i) =>
      i % 2
        ? part
        : // "[T1]" (the brackets go) or a bare "C6 (13:10)"
          part.replace(
            /\[([A-Z]{1,2}\d{1,4})\](?!\()|(?<![\w[])([A-Z]{1,2}\d{1,4})(?![\w\]])/g,
            (whole, inBrackets, bare) => {
              const code = inBrackets ?? bare;
              const target = to.get(code);
              return target ? `[${code}](${target})` : whole;
            },
          ),
    )
    .join('');
}

/** answer={false} leaves out the answer itself, for a page that shows it on its own (Review's Ask, R4); folded leaves
 * out the warnings and the evidence, for a page that folds them its own way (the Inbox's report reader). */
export function Result({
  result,
  history = false,
  answer = true,
  folded = false,
  bare = false,
  after,
}: {
  result: unknown;
  history?: boolean;
  answer?: boolean;
  folded?: boolean;
  /** No "Answer" heading or status badge, for a page that frames the answer itself (the Inbox reader) */
  bare?: boolean;
  /** Shown in the answer's box under its text: the Inbox reader's cited moments */
  after?: ReactNode;
}) {
  if (!isRec(result)) return null;
  // A general result has no recorded evidence: never show coverage or evidence panels for it, even if a stray field is present.
  const recorded = history && result.scope !== 'general' && result.is_history !== false;
  const evidence = recorded ? list(result.evidence) : [];
  const rounds = recorded ? records(result.rounds) : [];
  const warnings = Array.isArray(result.warnings)
    ? result.warnings
    : typeof result.warnings === 'string'
      ? [result.warnings]
      : [];
  const coverage = recorded && isRec(result.coverage) && Object.keys(result.coverage).length ? result.coverage : null;
  const paths = recorded ? list(result.paths) : [];
  // The range it read ("today", "since 2026-10-03", "all your recordings"), so the person can narrow it.
  const window = recorded && isRec(result.query) && typeof result.query.window === 'string' ? result.query.window : '';
  // The forgotten-person reply is not a history answer (no evidence), but it is a new answer path with a switch.
  const forgotten = list(result.paths).includes('forgotten');
  return (
    <>
      {answer && result.answer !== undefined && (
        <section className={classes.answer} data-bare={bare || undefined}>
          {!bare && <h3>Answer</h3>}
          {!bare && !!result.status && result.status !== 'ok' && <StatusBadge status={plain(result.status)} />}
          <TextBlock value={evidence.length ? citeLinks(plain(result.answer), evidence) : result.answer} />
          {after}
          {window && <p className={classes.muted}>Looked through {window}.</p>}
          {paths.includes('timeline') && (
            <p className={classes.muted}>
              From the timeline: who was seen and heard. <NewTag setting="timeline.enabled" />
            </p>
          )}
          {paths.includes('detections') && (
            <p className={classes.muted}>
              From live face detections. <NewTag setting="detections.last_seen" />
            </p>
          )}
          {forgotten && (
            <p className={classes.muted}>
              A person deleted in People. <NewTag setting="assistant.forgotten_reply" />
            </p>
          )}
        </section>
      )}
      {recorded && isRec(result.crossings) && <Crossings crossings={result.crossings} />}
      {!folded && <Notes title="Warnings" items={warnings} />}
      {rounds.length > 0 && (
        <div className={classes.path}>
          <Disclosure className={classes.details} summary={`Search rounds (${rounds.length})`}>
            <ol>
              {rounds.map((r, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: rounds are in order and never move
                <li key={i}>{roundText(r)}</li>
              ))}
            </ol>
          </Disclosure>
          <NewTag setting="recall.enabled" />
        </div>
      )}
      {coverage && (
        <>
          {!!(coverage.partial || coverage.truncated) && (
            <p className={classes.muted}>This answer is based on partial recorded evidence.</p>
          )}
          <Disclosure className={classes.details} summary="History coverage">
            <Facts value={coverage} />
          </Disclosure>
        </>
      )}
      {!folded && evidence.length > 0 && (
        <>
          <h3>{`Evidence (${evidence.length})`}</h3>
          <Paged items={evidence} render={e => <Evidence item={e} />} />
        </>
      )}
    </>
  );
}
