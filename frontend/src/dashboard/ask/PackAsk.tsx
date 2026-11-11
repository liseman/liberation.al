// The one Ask's box: a question about your recordings, answered quickly right there with its evidence, or, with Deep
// research on, looked into slowly in the background with its report delivered to the Inbox. It lives in Ask's floating
// window (Ask.tsx) and is the same wherever Ask was opened from: a page can only hand it a starting question, Deep
// research switched on, or a hint (the day Review shows, or a report to follow up on), which shows as a chip over the
// box and goes when the person closes it.
//
// Deep research is off again after each question it sends: it takes minutes of the GPU, so a question is quick unless
// the person asks for more that time. The Mantine conventions have nothing on it either way; the Inbox's "Research a
// question" switches it on for the one question that follows. Its limits (rounds, time, model) are settings, not words
// here: the toggle says one plain line, the same on or off, so the box doesn't jump when it flips.
import {
  Anchor,
  Badge,
  Button,
  CloseButton,
  Group,
  Paper,
  Switch,
  Text,
  Textarea,
  UnstyledButton,
} from '@mantine/core';
import { IconCircleCheck } from '@tabler/icons-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type KeyboardEvent, useEffect, useId, useMemo, useState } from 'react';
import { messageOf } from '../../shared/http';
import { researchQuery, startResearch } from '../inbox/api';
import { PageError } from '../lib/page';
import { href, linkClick, useRoute } from '../router';
import classes from './ask.module.css';
import { isRec } from './format';
import { History } from './History';
import { useAskJob } from './job';
import { MarkAnswer } from './MarkAnswer';
import { citedMoments } from './moments';
import { type AskHint, useAsk, useAskPage } from './openAsk';
import { Result } from './Result';

/** A few questions to start from, for a day that's today or another day (Review's day as the hint). */
export function suggestions(today: boolean): string[] {
  return today
    ? ['Who has been here today?', 'What did we talk about today?', 'What happened this morning?']
    : ['Who was here that day?', 'What did we talk about?', 'What happened in the morning?'];
}

/** Questions to start from without a day. */
export const STARTERS = ['What was I doing this afternoon?', 'Who came by today?', 'What did we talk about today?'];

/** A finished answer to a question about recordings, which the measurement set can ask again. */
const markable = (r: unknown): boolean =>
  isRec(r) && typeof r.answer === 'string' && (r.status === 'ok' || r.status === 'partial') && r.is_history !== false;

/** How a quick answer falls short, or null when it stands: Ask then offers deep research. `none` is no answer at
 *  all (an error, or the server's `found` false: nothing in the recordings), which shows only the offer; `thin` keeps
 *  the answer above it. A recorded count is an answer even without an evidence list, so `found` decides, not the list. */
export function thin(result: unknown): { kind: 'none' | 'thin'; why: string } | null {
  if (!isRec(result) || result.is_history === false) return null;
  if (result.status === 'error') return { kind: 'none', why: "Ask couldn't answer this quickly." };
  if (result.found === false) return { kind: 'none', why: 'Nothing in your recordings answered this quickly.' };
  if (result.status === 'partial')
    return { kind: 'thin', why: 'This quick answer rests on only part of the evidence.' };
  return null;
}

export const DEEP_HELP = 'Takes longer and looks deeper. The report arrives in the Inbox.';

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** When research is usually ready: the median of this machine's recent reports, else the usual half hour. */
export function readyWords(typical: number | null | undefined): string {
  return typical
    ? `Usually ready in about ${plural(Math.max(1, Math.round(typical / 60)), 'minute')}.`
    : 'Usually ready within half an hour.';
}

/** The moments an answer cites, as time chips that play from there (Ask opened on Review's day). */
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

/** The hint over the box, as a chip with its own remove cross. */
function HintChip({ hint, onClose }: { hint: AskHint; onClose: () => void }) {
  const [text, close] =
    hint.kind === 'report'
      ? [`Following up: ${hint.question}`, 'Stop following up on this report']
      : [hint.label, `Stop asking about ${hint.label}`];
  return (
    <div className={classes.context} data-ask-hint={hint.kind}>
      {/* Mantine's Pill hides its remove button from the keyboard (it is meant for a pills input, removed with
          Backspace), so this is a Badge with a labelled CloseButton inside it. */}
      <Badge
        size="lg"
        variant="light"
        color="gray"
        tt="none"
        fw={500}
        className={classes.hint}
        title={text}
        rightSection={<CloseButton size="xs" aria-label={close} onClick={onClose} />}
      >
        {text}
      </Badge>
    </div>
  );
}

/** Research sent: accepted at once, where it lands, and roughly how long it takes. */
interface Sent {
  question: string;
  typical: number | null;
}

export function PackAsk() {
  const ask = useAskJob();
  const client = useQueryClient();
  const research = useQuery(researchQuery());
  const id = useId();
  const [hint, setHint] = useState<AskHint | null>(null);
  const [deep, setDeep] = useState(false);
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState<Sent | null>(null);
  const [deepError, setDeepError] = useState('');
  // The question the shown answer is for, so the offer under a thin answer researches that one, not a new draft.
  const [answered, setAnswered] = useState<{ question: string; day?: string } | null>(null);

  // A page opened Ask with something to start from: take it once.
  const requested = useAsk(s => s.request);
  const { change, input } = ask;
  useEffect(() => {
    if (!requested) return;
    const taken = useAsk.getState().take();
    if (!taken) return;
    if (taken.hint !== undefined || taken.draft !== undefined || taken.deep !== undefined) {
      setHint(taken.hint ?? null);
      setDeep(!!taken.deep);
      setSent(null);
      setDeepError('');
    }
    if (taken.draft !== undefined) change(taken.draft);
    requestAnimationFrame(() => input.current?.focus());
  }, [requested, change, input]);

  // Review's day as a hint: its moments play in place and are marked on its timeline while Review shows that day.
  const page = useAskPage();
  const onReview = useRoute(s => s.tab === 'review');
  const seek = onReview && hint?.kind === 'day' && page.day?.day === hint.day ? page.seek : null;
  const times = useMemo(() => (seek ? citedMoments(ask.result).map(c => c.t) : []), [seek, ask.result]);
  const mark = page.mark;
  useEffect(() => mark?.(times), [mark, times]);

  const day = hint?.day;
  const reason = research.data && !research.data.available ? research.data.reason : null;
  const followup = hint?.kind === 'report';

  const sendDeep = async (question: string, asDay: string | undefined) => {
    const q = question.trim();
    if (!q) {
      setDeepError('Please enter a question first.');
      input.current?.focus();
      return;
    }
    setSending(true);
    setDeepError('');
    try {
      await startResearch(client, q, asDay ?? null);
      setSent({ question: q, typical: research.data?.typical_s ?? null });
      setDeep(false);
      ask.clear();
      setAnswered(null);
      if (q === ask.question.trim()) change('');
    } catch (e) {
      setDeepError(messageOf(e));
    } finally {
      setSending(false);
    }
  };
  const submit = (question: string = ask.question) => {
    if (deep) {
      void sendDeep(question, day);
      return;
    }
    setSent(null);
    setDeepError('');
    setAnswered({ question: question.trim(), ...(day ? { day } : {}) });
    void ask.submit({ ...(day ? { day } : {}), ...(hint?.kind === 'report' ? { followup: { report: hint.id } } : {}) });
  };
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };
  const start = (question: string) => {
    change(question);
    submit(question);
  };

  const busy = ask.busy || sending;
  const nothingYet = !ask.result && !ask.busy && !ask.question.trim() && !sent;
  const starters = followup ? null : hint?.kind === 'day' ? suggestions(hint.today) : STARTERS;
  const short = !ask.busy && answered ? thin(ask.result) : null;

  const deepRow = (
    <div className={classes.deep} aria-busy={busy}>
      <Switch
        label="Deep research"
        className={classes.deepSwitch}
        checked={deep}
        onChange={e => setDeep(e.currentTarget.checked)}
        disabled={busy || !!reason || followup}
        description={reason ?? (followup ? 'A follow-up on a report gets a quick answer.' : DEEP_HELP)}
        classNames={{ description: classes.deepHelp }}
      />
      <Button
        radius="xl"
        onClick={() => submit()}
        disabled={busy}
        loading={sending}
        className={classes.askButton}
        data-deep={deep || undefined}
      >
        {deep ? 'Research' : 'Ask'}
      </Button>
    </div>
  );

  return (
    <Paper component="section" className={`${classes.root} ${classes.bare} ${classes.compact}`} aria-label="Ask">
      <div className={classes.body}>
        <Textarea
          id={`${id}question`}
          data-ask-question
          autoFocus
          ref={ask.input}
          rows={2}
          resize="none"
          enterKeyHint="send"
          label="Your question"
          inputContainer={box => (
            <>
              {hint && (
                <HintChip
                  hint={hint}
                  onClose={() => {
                    setHint(null);
                    input.current?.focus();
                  }}
                />
              )}
              {box}
              {deepRow}
            </>
          )}
          description={
            hint?.kind === 'day'
              ? `Searches your recordings from ${hint.label}, unless the question names another time. Nothing leaves this machine.`
              : 'Searches your recordings for the time the question names. Nothing leaves this machine.'
          }
          inputWrapperOrder={['label', 'input', 'description']}
          placeholder={hint?.kind === 'day' ? `Ask about ${hint.label}` : 'Ask about your recordings'}
          value={ask.question}
          onChange={e => change(e.currentTarget.value)}
          onKeyDown={onKeyDown}
          classNames={{ input: classes.input, label: classes.quietLabel, description: classes.scopeHelp }}
        />
        {/* Progress while an answer is coming, in plain words; nothing at rest or once it's there */}
        <p className={classes.status} role="status" aria-live="polite">
          {ask.busy ? `Looking through your recordings${ask.elapsed ? ` · ${ask.elapsed}` : ''}` : ''}
        </p>
        {ask.error && <PageError title={ask.error}>{null}</PageError>}
        {deepError && <PageError title={deepError}>{null}</PageError>}
        <div className={classes.output}>
          {sent && (
            <div className={classes.sent} role="status" data-ask-sent>
              <Group gap={6} wrap="nowrap">
                <IconCircleCheck size={18} color="var(--lib-ok)" aria-hidden />
                <Text size="sm" fw={600}>
                  Research started
                </Text>
              </Group>
              <Text size="sm">
                “{sent.question}” is being looked into. The report will be in the{' '}
                <Anchor href={href('/inbox')} onClick={linkClick} inherit>
                  Inbox
                </Anchor>
                . {readyWords(sent.typical)}
              </Text>
            </div>
          )}
          {starters && nothingYet && (
            <ul className={classes.suggest} aria-label="Suggested questions">
              {starters.map(q => (
                <li key={q}>
                  <Button variant="default" size="compact-sm" onClick={() => start(q)}>
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
          {short && answered && (
            <Paper className={classes.offer} data-ask-offer>
              <Text size="sm">{short.why}</Text>
              <Button
                size="xs"
                variant="default"
                disabled={!!reason || sending}
                onClick={() => void sendDeep(answered.question, answered.day)}
              >
                Try Deep research
              </Button>
              {reason && (
                <Text size="xs" c="dimmed">
                  {reason}
                </Text>
              )}
            </Paper>
          )}
          {/* No answer at all: only the line and the offer above, not the same words again or a rating */}
          {short?.kind !== 'none' && (
            <>
              <Result result={ask.result} history />
              {seek && !ask.busy && <Moments result={ask.result} onSeek={seek} />}
              {!ask.busy && ask.asked && markable(ask.result) && (
                <MarkAnswer key={ask.asked.at} result={ask.result} asked={ask.asked} />
              )}
            </>
          )}
          <History refresh={!ask.busy ? ask.result : null} onSeek={seek ?? undefined} />
        </div>
      </div>
    </Paper>
  );
}
