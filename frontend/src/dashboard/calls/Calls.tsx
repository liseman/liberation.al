// The Calls tab: calls the assistant answered or placed, each with its recording and transcript,
// and a form to have it call someone with a number. Ported from Calls and CallRow in static/dashboard/tab-people.js.
import {
  Accordion,
  ActionIcon,
  Anchor,
  Box,
  Button,
  Group,
  NativeSelect,
  SegmentedControl,
  Slider,
  Stack,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core';
import {
  IconCornerDownRight,
  IconPhone,
  IconPhoneIncoming,
  IconPhoneOutgoing,
  IconPlayerPauseFilled,
  IconPlayerPlayFilled,
} from '@tabler/icons-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, type JSX, useEffect, useId, useRef, useState } from 'react';
import { HttpError, messageOf } from '../../shared/http';
import { confirm } from '../lib/confirm';
import { clipLength, plural } from '../lib/format';
import { type Message, StatusLine } from '../lib/message';
import { PageEmpty, PageError, PageFrame, PageHeader, PageLoading } from '../lib/page';
import { play, seek, toggle, usePlayer } from '../lib/player';
import {
  CALLS_KEY,
  type Call,
  callAudio,
  callDetail,
  getCalls,
  getLiveCall,
  getPeople,
  hangupCall,
  LIVE_CALL_KEY,
  PEOPLE_KEY,
  type Person,
  placeCall,
} from '../people/api';
import { href, linkClick } from '../router';
import { shell } from '../shell';
import classes from './calls.module.css';
import { callDay, callDirection, callMeta, callTitle, callWho, dayHeading } from './callWords';

export function callNumber(value: string): string | null {
  if (/\(\s*0\s*\)/.test(value)) return null;
  const number = value.replace(/[\s().-]/g, '');
  return /^\+[1-9][0-9]{7,14}$/.test(number) ? number : null;
}

const muted = { size: 'sm', c: 'dimmed' } as const;

/** The opened row's player: the shared audio player (so the room microphone isn't left transcribing it), with where
 *  it is and a bar to move through the call. */
function CallPlayer({ c }: { c: Call }) {
  const key = `call:${c.dir}`;
  const mine = usePlayer(s => s.key === key);
  const paused = usePlayer(s => s.paused);
  const position = usePlayer(s => (s.key === key ? s.position : 0));
  const known = usePlayer(s => (s.key === key ? s.duration : null));
  const length = known ?? c.durationSeconds ?? 0;
  const playing = mine && !paused;
  return (
    <Group gap="sm" wrap="nowrap" className={classes.player}>
      <ActionIcon
        size={32}
        radius="xl"
        aria-label={playing ? 'Pause the call' : 'Play the call'}
        onClick={() => (mine ? toggle() : play(callAudio(c), key))}
      >
        {playing ? <IconPlayerPauseFilled size={14} aria-hidden /> : <IconPlayerPlayFilled size={14} aria-hidden />}
      </ActionIcon>
      <Slider
        className={classes.position}
        thumbLabel="Where in the call"
        min={0}
        max={Math.max(1, length)}
        step={1}
        value={Math.min(position, Math.max(1, length))}
        onChange={v => seek(key, v)}
        disabled={!mine}
        label={v => clipLength(v)}
        size="sm"
      />
      <Text component="span" size="sm" c="dimmed" className={classes.clock}>
        {clipLength(position)} / {clipLength(length)}
      </Text>
    </Group>
  );
}

/** One call: which way it went and with whom, when and how long, what the assistant was asked and what came of it.
 *  The whole row opens it, to its player and its transcript. */
function CallRow({ c, open, people }: { c: Call; open: boolean; people: readonly Person[] }) {
  const detail = useQuery({
    queryKey: [...CALLS_KEY, 'detail', c.day, c.dir],
    queryFn: ({ signal }) => callDetail(c, { signal }),
    enabled: open,
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
  const title = callTitle(c, people);
  const way = callDirection(c);
  const Way = way === 'out' ? IconPhoneOutgoing : way === 'in' ? IconPhoneIncoming : IconPhone;
  const asked = c.instructions?.trim();
  const came = asked ? c.outcome?.trim() : null; // a call with instructions says what came of them
  return (
    <Accordion.Item value={`${c.day}/${c.dir}`}>
      <Accordion.Control aria-label={`${title}, ${callMeta(c, title)}`}>
        <Group component="span" gap="sm" wrap="nowrap" align="flex-start" miw={0}>
          <Way size={18} aria-hidden className={classes.way} />
          <Stack component="span" gap={2} miw={0}>
            <Text component="span" fw={600}>
              {title}
            </Text>
            <Text component="span" {...muted}>
              {callMeta(c, title)}
            </Text>
            {asked && (
              <Text component="span" {...muted} fs="italic" lineClamp={1}>
                Asked: {asked}
              </Text>
            )}
            {came && (
              <Text component="span" size="sm" lineClamp={1} className={classes.outcome}>
                <IconCornerDownRight size={14} aria-hidden /> {came}
              </Text>
            )}
          </Stack>
        </Group>
      </Accordion.Control>
      <Accordion.Panel>
        <CallPlayer c={c} />
        {detail.isError ? (
          <PageError title="Couldn't load this call's transcript" detail={messageOf(detail.error)}>
            The recording above still plays.
          </PageError>
        ) : !detail.data ? (
          <PageLoading label="the transcript" lines={3} />
        ) : (
          (detail.data.transcript ?? []).map((t, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: transcript lines have no id and never reorder
            <Text key={i} my={6} style={{ font: '15.5px/1.5 var(--lib-speech)' }}>
              <Text component="span" {...muted}>
                {t.who || 'Unknown'} · {clipLength(t.t || 0)}
              </Text>
              <br />
              {t.text || ''}
            </Text>
          ))
        )}
      </Accordion.Panel>
    </Accordion.Item>
  );
}

/** The log, a section per day ("Today", "Yesterday", "Mon 3 Nov"), newest first as the server sends them. */
function CallLog({ calls, people }: { calls: readonly Call[]; people: readonly Person[] }) {
  const [open, setOpen] = useState<string[]>([]);
  const days: { day: string; calls: Call[] }[] = [];
  for (const c of calls.slice(0, 20)) {
    const day = callDay(c);
    const last = days.at(-1);
    if (last?.day === day) last.calls.push(c);
    else days.push({ day, calls: [c] });
  }
  return (
    <Stack gap="md">
      {days.map(({ day, calls: theirs }) => {
        const values = theirs.map(c => `${c.day}/${c.dir}`);
        return (
          <section key={day} aria-label={dayHeading(day)}>
            <Text component="h3" size="sm" fw={600} c="dimmed" className={classes.dayHeading}>
              {dayHeading(day)}
            </Text>
            <Accordion
              multiple
              value={open.filter(v => values.includes(v))}
              onChange={next => setOpen(was => [...was.filter(v => !values.includes(v)), ...next])}
              keepMounted={false}
              transitionDuration={0}
              classNames={{ control: classes.callControl, label: classes.callLabel, content: classes.callContent }}
            >
              {theirs.map(c => (
                <CallRow key={`${c.day}/${c.dir}`} c={c} open={open.includes(`${c.day}/${c.dir}`)} people={people} />
              ))}
            </Accordion>
          </section>
        );
      })}
    </Stack>
  );
}

const LIVE_POLL_MS = 3000; // how often Calls looks for a call in progress while it's open
const SAVED_AGAIN_MS = 5000; // when a call ends, the log is read again now and once more after this, as it's saved

/** How long a call has been on, ticking each second: only this re-renders as the time goes. */
function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(tick);
  }, []);
  return <time className={classes.liveClock}>{clipLength(Math.max(0, now - since))}</time>;
}

/** A call in progress, pinned at the top while it's on ("On a call with Dana · 2:14"), from GET /calls/live.
 *  It takes its place only while there's a call. When the call ends, the log is read again, where it's now the newest
 *  row. Polling stops when Calls is hidden (React Query pauses in a hidden tab; a hidden page drops its queries). */
function LiveCallRow({ people }: { people: readonly Person[] }) {
  const client = useQueryClient();
  const live = useQuery({ queryKey: LIVE_CALL_KEY, queryFn: getLiveCall, refetchInterval: LIVE_POLL_MS, retry: false });
  const call = live.data?.live ?? null;
  const pending = useRef(false);
  const [ending, setEnding] = useState(false);
  const [result, setResult] = useState<Message | null>(null);
  const endCall = async () => {
    if (!call || pending.current) return;
    // Snapshot the clicked row, never whatever a later poll says is live.
    const target = { day: call.day, dir: call.dir };
    pending.current = true;
    setEnding(true);
    setResult(null);
    try {
      const reply = await hangupCall(target);
      if (!reply.ok) throw new Error(reply.detail || 'The call could not be ended.');
      setResult({ text: reply.detail });
      await Promise.all([
        client.invalidateQueries({ queryKey: LIVE_CALL_KEY, exact: true }),
        client.invalidateQueries({ queryKey: CALLS_KEY, exact: true }),
      ]);
    } catch (err) {
      setResult({ err: `Could not hang up: ${messageOf(err)}` });
    } finally {
      pending.current = false;
      setEnding(false);
    }
  };
  const was = useRef<string | null>(null);
  const again = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => {
    const key = call ? `${call.day}/${call.dir}` : null;
    if (was.current && was.current !== key) {
      // the call that was on ended: it's in the log now, or will be once the bridge has saved it
      const reread = () => void client.invalidateQueries({ queryKey: CALLS_KEY, exact: true });
      reread();
      clearTimeout(again.current);
      again.current = setTimeout(reread, SAVED_AGAIN_MS);
    }
    was.current = key;
  }, [call, client]);
  useEffect(() => () => clearTimeout(again.current), []);
  if (!call) return <StatusLine msg={result} />;
  const who = call.name?.trim() || callWho(call.number, people) || 'someone';
  const Way = call.direction === 'out' ? IconPhoneOutgoing : call.direction === 'in' ? IconPhoneIncoming : IconPhone;
  return (
    <>
      <div className={classes.liveRow}>
        <span className={classes.liveDot} aria-hidden />
        <Way size={18} aria-hidden className={classes.way} />
        {/* Said once when the call starts; the ticking time isn't announced each second. */}
        <Text component="span" fw={600} role="status">
          On a call with {who}
        </Text>
        <Text component="span" c="dimmed" aria-hidden>
          ·
        </Text>
        <Elapsed since={call.since} />
        <Button color="red" size="xs" loading={ending} disabled={ending} onClick={() => void endCall()}>
          {ending ? 'Hanging up…' : 'Hang up'}
        </Button>
      </div>
      <StatusLine msg={result} />
    </>
  );
}

/** Calls shares People's contact query, so navigating between the tabs reuses fresh data. */
export function CallsTab(): JSX.Element {
  const contacts = useQuery({ queryKey: PEOPLE_KEY, queryFn: getPeople, refetchInterval: 30000 });
  return (
    <PageFrame width="narrow">
      {contacts.isError && !contacts.data && (
        <PageError
          title="Couldn't load your contacts"
          detail={messageOf(contacts.error)}
          onRetry={() => void contacts.refetch()}
          busy={contacts.isFetching}
        >
          You can still call by typing a number.
        </PageError>
      )}
      {/* undefined while the contacts load; a failed load counts as nobody to pick, so typing a number is offered */}
      <Calls people={contacts.data?.people ?? (contacts.isError ? [] : undefined)} />
    </PageFrame>
  );
}

/** The Calls section. People with a number are who the assistant can be asked to call; `people` is undefined while
 *  they load. */
export function Calls({ people }: { people: Person[] | undefined }): JSX.Element {
  const client = useQueryClient();
  const { assistant } = shell();
  const headingId = useId();
  const formId = useId();
  const query = useQuery({ queryKey: CALLS_KEY, queryFn: getCalls, refetchInterval: 30000 });
  const calls = query.data;
  const logFailed = query.isError && !query.data;
  // Once there are calls, the form folds to "Call someone" beside the title, so the log is what the page shows.
  const [composing, setComposing] = useState(false);
  const folded = !!calls?.length && !composing;
  const [who, setWho] = useState('');
  // Person or Number: the owner's pick, else Number while nobody has one to pick, else Person
  const [chosen, setMode] = useState<string | null>(null);
  const [number, setNumber] = useState('');
  const [left, setLeft] = useState(false); // the number field was left: its error may show
  const [msg, setMsg] = useState('');
  const [st, setSt] = useState<Message | null>(null);
  const later = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (later.current) clearTimeout(later.current);
    },
    [],
  );
  const callable = (people ?? []).filter(p => p.number);
  const mode = chosen ?? (people && !callable.length ? 'number' : 'contact');
  // The button says who it calls: "Call Carly", "Call +14155550123", or "Call" until someone is picked.
  const typed = mode === 'number' ? callNumber(number) : null;
  const callLabel = mode === 'contact' && who ? `Call ${who}` : typed ? `Call ${typed}` : 'Call';
  // Call waits for someone to call: a person with a number picked, or a whole number typed
  const ready = mode === 'contact' ? callable.some(p => p.name === who) : !!typed;

  const place = async (e: FormEvent) => {
    e.preventDefault();
    const contact = callable.find(p => p.name === who);
    if (mode === 'contact' && !contact) return setSt({ err: 'Pick someone to call.' });
    const target = callNumber(mode === 'number' ? number : contact?.number || '');
    if (!target) return setSt({ err: 'Enter a valid international number with + and country code.' });
    const label = mode === 'number' ? target : `${who} (${target})`;
    const m = msg.trim();
    const sure = await confirm({
      title: 'Place a call',
      body: `Have ${assistant} call ${label} now${m ? ` with instructions: “${m}”` : ''}?`,
      confirm: 'Call',
    });
    if (!sure) return;
    setSt({ text: 'Placing the call' });
    try {
      const j = (await placeCall(mode === 'contact' ? who : '', m, target)) as {
        ok: boolean;
        detail?: string;
        error?: string;
      };
      setSt(
        j.ok
          ? { text: j.detail || `Calling ${who}.` }
          : { err: j.error || j.detail || 'The call could not be placed.' },
      );
      if (j.ok) {
        if (later.current) clearTimeout(later.current);
        later.current = setTimeout(() => void client.invalidateQueries({ queryKey: CALLS_KEY }), 20000);
      }
    } catch (err) {
      setSt({
        err:
          err instanceof HttpError
            ? err.reason || 'The call could not be placed.'
            : `The call could not be placed: ${messageOf(err)}`,
      });
    }
  };

  const heading = (
    <PageHeader
      title="Calls"
      titleId={headingId}
      meta={calls?.length ? plural(calls.length, 'call', 'calls') : undefined}
      lede={`${assistant} calls on your behalf, follows your instructions as the call goes, and records and transcribes it.`}
      actions={
        calls?.length ? (
          <Button
            variant={composing ? 'default' : 'filled'}
            leftSection={<IconPhoneOutgoing size={16} aria-hidden />}
            aria-expanded={composing}
            aria-controls={formId}
            onClick={() => setComposing(was => !was)}
          >
            {composing ? 'Close' : 'Call someone'}
          </Button>
        ) : undefined
      }
    />
  );
  return (
    <Box component="section" aria-labelledby={headingId}>
      {heading}
      <LiveCallRow people={people ?? []} />
      {/* Folded by style, not `hidden`: Stack's own display would show a hidden form anyway. */}
      <Stack
        component="form"
        id={formId}
        gap="xs"
        mt="md"
        onSubmit={place}
        hidden={folded}
        style={folded ? { display: 'none' } : undefined}
      >
        <Title order={2}>Have {assistant} call someone</Title>
        {/* One choice, Person or Number, then only the field that applies. */}
        <SegmentedControl
          aria-label="Call a person or a number"
          w="fit-content"
          value={mode}
          onChange={value => {
            setMode(value);
            setWho('');
            setNumber('');
            setLeft(false);
            setSt(null);
          }}
          data={[
            { value: 'contact', label: 'Person' },
            { value: 'number', label: 'Number' },
          ]}
        />
        {mode === 'number' ? (
          <TextInput
            label="Phone number with country code"
            type="tel"
            placeholder="+1 415 555 0123"
            maw={320}
            value={number}
            onChange={e => setNumber(e.currentTarget.value)}
            onBlur={() => setLeft(true)}
            // Call stays off until the number is whole: once the field is left, say what's missing (not while typing)
            error={
              left && number.trim() && !typed
                ? 'Enter a whole international number: + and the country code.'
                : undefined
            }
          />
        ) : (
          <NativeSelect
            label="Who to call"
            maw={320}
            value={who}
            onChange={e => setWho(e.currentTarget.value)}
            disabled={!callable.length}
            description={
              people && !callable.length ? (
                <>
                  Nobody has a number yet. Add one on their card in{' '}
                  <Anchor href={href('/people')} onClick={linkClick} inherit>
                    People
                  </Anchor>
                  , or call a number.
                </>
              ) : undefined
            }
            inputWrapperOrder={['label', 'input', 'description']}
            data={[
              {
                value: '',
                label: !people ? 'Loading people…' : callable.length ? 'Pick someone' : 'Nobody has a number yet',
              },
              ...callable.map(p => ({ value: p.name, label: `${p.name}, ${p.number}` })),
            ]}
          />
        )}
        <Textarea
          label={`What should ${assistant} do?`}
          description="Optional"
          placeholder="For example: ask whether they're free for dinner on Friday."
          rows={2}
          resize="vertical"
          value={msg}
          onChange={e => setMsg(e.currentTarget.value)}
        />
        <Group justify="space-between" gap="xs" wrap="wrap">
          <Text size="sm" c="dimmed">
            {assistant} records and transcribes the call.
          </Text>
          <Button type="submit" disabled={!ready}>
            {callLabel}
          </Button>
        </Group>
        <StatusLine msg={st} />
      </Stack>
      <Title order={2} mt="lg">
        Call log
      </Title>
      {logFailed ? (
        <PageError
          title="Couldn't load the call log"
          detail={messageOf(query.error)}
          onRetry={() => void query.refetch()}
          busy={query.isFetching}
        >
          You can still place a call.
        </PageError>
      ) : calls == null ? (
        <PageLoading label="the call log" shape="rows" lines={3} />
      ) : calls.length ? (
        <CallLog calls={calls} people={people ?? []} />
      ) : (
        <PageEmpty title="No calls yet" compact>
          Calls {assistant} answers or places show up here, with a recording and a transcript. Use the form above to
          start one.
        </PageEmpty>
      )}
    </Box>
  );
}
