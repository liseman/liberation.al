// The Ask widget, ported from static/lib/ask.js. With a day (Review) it asks about that day's recordings, with a
// single local Ask. Without one, local searches the recordings for the time the question names, and friends and
// everyone are general questions for the LLMs (#2307); Live opts into the scope choice (showModes), shown as one
// segmented control under the question box with a single Ask button (#2452).
import {
  Button,
  CloseButton,
  Group,
  Paper,
  SegmentedControl,
  Text,
  Textarea,
  Title,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type KeyboardEvent, useEffect, useId, useMemo, useState } from 'react';
import { messageOf } from '../../shared/http';
import { researchQuery, startResearch } from '../inbox/api';
import { PageError } from '../lib/page';
import { href, linkClick } from '../router';
import classes from './ask.module.css';
import { Stamp } from './bits';
import { isRec } from './format';
import { History } from './History';
import { isMode, MODE_NAMES, MODES, type Mode, storageKey, useAskJob } from './job';
import { MarkAnswer } from './MarkAnswer';
import { citedMoments } from './moments';
import { type AskContext, useAskOpener } from './openAsk';
import { Result } from './Result';

const SCOPES = (Object.keys(MODES) as Mode[]).map(m => ({ value: m, label: MODE_NAMES[m] }));

function help(day: string | null): string {
  return day
    ? 'Enter asks · Shift+Enter adds a line.'
    : 'Searches your recordings for the time the question names. Enter asks · Shift+Enter adds a line.';
}

/** Research (#1336): the question taken slowly in the background, its report in the Inbox. */
function ResearchButton({ question, day, busy }: { question: string; day: string; busy: boolean }) {
  const client = useQueryClient();
  const jobs = useQuery(researchQuery());
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);
  const reason = jobs.data && !jobs.data.available ? jobs.data.reason : null;
  const reasonId = useId();
  const start = async () => {
    try {
      await startResearch(client, question.trim(), day);
      setNote({ ok: true, text: 'Research started. The report will be in the Inbox.' });
    } catch (e) {
      setNote({ ok: false, text: messageOf(e) });
    }
  };
  return (
    <>
      <Tooltip
        label={reason ?? 'Look into this in the background, more thoroughly; the report goes to the Inbox.'}
        events={{ hover: true, focus: true, touch: false }}
      >
        <span>
          <Button
            variant="default"
            radius="xl"
            disabled={busy || !question.trim() || !!reason}
            onClick={() => void start()}
            className={classes.action}
            aria-describedby={reason ? reasonId : undefined}
          >
            Research
          </Button>
        </span>
      </Tooltip>
      {/* A disabled button can't take focus, so its reason is on the page, not only in the hover tooltip (#2271). */}
      {reason && (
        <Text id={reasonId} size="xs" c="dimmed">
          {reason}
        </Text>
      )}
      {note && (
        <Text size="sm" role={note.ok ? 'status' : 'alert'} c={note.ok ? undefined : 'red'}>
          {note.text}{' '}
          {note.ok && (
            <a href={href('/inbox')} onClick={linkClick}>
              Open the Inbox
            </a>
          )}
        </Text>
      )}
    </>
  );
}

/** A finished answer to a question about recordings, which the measurement set can ask again. */
const markable = (r: unknown): boolean =>
  isRec(r) &&
  typeof r.answer === 'string' &&
  (r.status === 'ok' || r.status === 'partial') &&
  r.is_history !== false &&
  r.scope !== 'general';

/** What Review adds to its Ask: the day as the box names it ("today", "Tue 27 Oct"), whether that is today (for the
 *  suggested questions), playing a cited moment, and marking the newest answer's moments on the timeline. */
interface ReviewHooks {
  dayLabel?: string;
  today?: boolean;
  onSeek?: (t: number) => void;
  onMarks?: (times: number[]) => void;
}

/** A few questions to start from, for a day that's today or another day. */
export function suggestions(today: boolean): string[] {
  return today
    ? ['Who has been here today?', 'What did we talk about today?', 'What happened this morning?']
    : ['Who was here that day?', 'What did we talk about?', 'What happened in the morning?'];
}

/** Live's starters: questions about your own recordings, so a starter always asks My recordings. */
export const LIVE_STARTERS = [
  'What was I doing this afternoon?',
  'Who came by today?',
  'What did we talk about today?',
];

/** The moments an answer cites, as time chips that play from there (Review). */
function Moments({ result, onSeek }: { result: unknown; onSeek: (t: number) => void }) {
  const chips = citedMoments(result);
  if (!chips.length) return null;
  return (
    <ul className={classes.moments} aria-label="Moments it cites">
      {chips.map(c => (
        <li key={c.label}>
          <UnstyledButton
            type="button"
            className={classes.moment}
            title={`Play from ${c.label}`}
            onClick={() => onSeek(c.t)}
          >
            {c.label}
          </UnstyledButton>
        </li>
      ))}
    </ul>
  );
}

function AskCard({
  day,
  showModes,
  bare,
  compact,
  dayLabel,
  today,
  onSeek,
  onMarks,
}: {
  day: string | null;
  showModes: boolean;
  bare: boolean;
  compact: boolean;
} & ReviewHooks) {
  const ask = useAskJob(day, showModes);
  // Live's scope choice: what Enter and Ask send. It starts on My recordings (or the scope of a job picked back up),
  // so a question only leaves this machine after Friends or Everyone is chosen.
  const [pick, setPick] = useState<Mode>(ask.mode);
  const id = useId(); // Review and the Live panel can both be in the page (a hidden tab stays mounted, #330)
  // A research report this Ask follows up on (openAsk, from the Inbox): shown as a chip over the box, and sent with
  // each question until it's closed. A follow-up asks My recordings, since a report is about your own recordings.
  const [context, setContext] = useState<AskContext | null>(null);
  const requested = useAskOpener(s => s.request);
  const { change, input } = ask;
  useEffect(() => {
    if (!showModes || !requested) return;
    const taken = useAskOpener.getState().take();
    if (!taken) return;
    setContext(taken.context ?? null);
    if (taken.context) setPick('local');
    change(taken.draft ?? '');
    requestAnimationFrame(() => input.current?.focus());
  }, [showModes, requested, change, input]);
  const submit = (m: Mode) =>
    context ? ask.submit('local', { followup: { report: context.id }, day: context.day }) : ask.submit(m);
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void submit(showModes ? pick : 'local');
    }
  };
  // Review marks the newest answer's moments on the timeline (or a saved answer's, once one is opened from History),
  // and none once the day changes (this unmounts with it).
  const [savedTimes, setSavedTimes] = useState<number[] | null>(null);
  const times = useMemo(
    () => (onMarks ? (savedTimes ?? citedMoments(ask.result).map(c => c.t)) : []),
    [onMarks, ask.result, savedTimes],
  );
  useEffect(() => onMarks?.(times), [onMarks, times]);
  useEffect(() => () => onMarks?.([]), [onMarks]);
  const send = (question: string) => {
    if (showModes) setPick('local');
    ask.change(question);
    void ask.submit('local');
  };
  const nothingYet = !ask.result && !ask.busy && !ask.question.trim();
  // A few questions to start from, under the box (#2594 follow-up): Review's are about the day it shows, Live's about
  // your own recordings. They go once there's a question or an answer, and there are none while following up on a
  // report, where the next question is the owner's own.
  const starters = day ? (onSeek ? suggestions(!!today) : null) : showModes && !context ? LIVE_STARTERS : null;
  const answer = (
    <>
      {starters && nothingYet && (
        <ul className={classes.suggest} aria-label="Suggested questions">
          {starters.map(q => (
            <li key={q}>
              <Button variant="default" size="compact-sm" onClick={() => send(q)}>
                {q}
              </Button>
            </li>
          ))}
        </ul>
      )}
      {ask.busy && ask.partial && (
        <div className={classes.partial} role="status" aria-label="Answer so far">
          {ask.partial}
        </div>
      )}
      <Result result={ask.result} history={ask.scope === 'history'} />
      {onSeek && !ask.busy && <Moments result={ask.result} onSeek={onSeek} />}
      {ask.scope === 'history' && !ask.busy && ask.asked && markable(ask.result) && (
        <MarkAnswer key={ask.asked.at} result={ask.result} asked={ask.asked} />
      )}
    </>
  );
  const history = (
    <History
      key={day ?? 'general'}
      day={day}
      refresh={!ask.busy ? ask.result : null}
      onSeek={onSeek}
      onMoments={onMarks ? setSavedTimes : undefined}
    />
  );
  // Every Ask puts the answer right under the box, then the past questions (#2452).
  const output = (
    <div className={classes.output}>
      {answer}
      {history}
    </div>
  );
  // Live's scope choice and Ask sit between the box and the sentence that says what the scope shares, which is the
  // box's description, so a screen reader hears it with the question.
  const scopeRow = (
    <Group gap="xs" mt="sm" wrap="nowrap" aria-busy={ask.busy} className={classes.askRow}>
      <SegmentedControl
        size="xs"
        aria-label="Who to ask"
        data={SCOPES}
        value={pick}
        onChange={v => isMode(v) && setPick(v)}
        disabled={ask.busy || !!context}
        className={classes.scope}
      />
      <Button
        radius="xl"
        data-mode={pick}
        disabled={ask.busy}
        onClick={() => void submit(pick)}
        className={classes.askButton}
      >
        Ask
      </Button>
    </Group>
  );
  const title = day ? 'Ask about video & conversations' : 'Ask';
  return (
    <Paper
      component="section"
      id={`${id}ask`}
      className={`${classes.root}${bare ? ` ${classes.bare}` : ''}${compact ? ` ${classes.compact}` : ''}`}
      withBorder={!bare}
      radius="lg"
      aria-label={day ? 'Ask about recordings' : 'Ask'}
    >
      {!bare && (
        <Title order={2} className={classes.heading}>
          {title}
        </Title>
      )}
      <div className={classes.body}>
        <Textarea
          id={`${id}question`}
          data-ask-question={day ? 'review' : 'general'}
          autoFocus={bare && compact}
          ref={ask.input}
          rows={compact ? 2 : 3}
          resize={compact ? 'none' : 'vertical'}
          label={
            dayLabel ? `Ask about ${dayLabel}` : day ? 'What would you like to know?' : 'What would you like to ask?'
          }
          inputContainer={
            showModes
              ? box => (
                  <>
                    {context && (
                      <div className={classes.context}>
                        <Text span size="sm" className={classes.contextText}>
                          <Text span size="sm" c="dimmed">
                            Following up on:{' '}
                          </Text>
                          {context.question}
                        </Text>
                        <CloseButton
                          size="sm"
                          aria-label="Stop following up on this report"
                          onClick={() => {
                            setContext(null);
                            input.current?.focus();
                          }}
                        />
                      </div>
                    )}
                    {box}
                    {scopeRow}
                  </>
                )
              : undefined
          }
          description={
            showModes ? (
              MODES[pick]
            ) : compact ? (
              'Enter asks, which searches your recordings.'
            ) : (
              <>
                {day && (
                  <>
                    <span>
                      Searches recordings and raw transcripts for {day}. An explicit time such as "this week" overrides
                      the selected day.
                    </span>{' '}
                  </>
                )}
                {help(day)}
              </>
            )
          }
          inputWrapperOrder={['label', 'input', 'description']}
          placeholder={
            dayLabel
              ? `Ask about ${dayLabel}`
              : day
                ? 'What happened in the kitchen this morning? How long were lights on? What did we plan for this week?'
                : 'Ask about your recordings, or anyone you share with'
          }
          value={ask.question}
          onChange={e => ask.change(e.currentTarget.value)}
          onKeyDown={onKeyDown}
          classNames={{
            input: classes.input,
            label: showModes ? classes.quietLabel : classes.label,
            description: showModes ? classes.scopeHelp : classes.help,
          }}
        />
        {!showModes && (
          <Group gap="xs" mt="sm" aria-busy={ask.busy}>
            <Button
              variant="default"
              radius="xl"
              data-mode="local"
              disabled={ask.busy}
              onClick={() => void ask.submit('local')}
              className={classes.action}
            >
              Ask
            </Button>
            {day && <ResearchButton question={ask.question} day={day} busy={ask.busy} />}
          </Group>
        )}
        <p className={classes.status} role="status" aria-live="polite">
          {ask.status.text}
          {!!ask.status.finished && (
            <>
              {' · '}
              <Stamp value={ask.status.finished} />
            </>
          )}
        </p>
        <Text component="span" className={`${classes.muted} ${classes.elapsed}`}>
          {ask.elapsed}
        </Text>
        {ask.error && <PageError title={ask.error}>{null}</PageError>}
        {output}
      </div>
    </Paper>
  );
}

/**
 * Ask a question and follow its answer. A day (YYYY-MM-DD) makes it a question about that day's recordings; without
 * one it searches the recordings for the time the question names, and showModes adds Live's scope choice: My
 * recordings, Friends, or Everyone. bare drops the heading and the card's border, for a panel that already names it.
 */
export function PackAsk({
  day = null,
  showModes = false,
  bare = false,
  compact = false,
  ...review
}: {
  day?: string | null;
  showModes?: boolean;
  bare?: boolean;
  compact?: boolean;
} & ReviewHooks) {
  // Each saved-job key is its own widget: another day starts clean and never adopts this day's job.
  return (
    <AskCard
      key={storageKey(day, showModes)}
      day={day || null}
      showModes={showModes}
      bare={bare}
      compact={compact}
      {...review}
    />
  );
}
