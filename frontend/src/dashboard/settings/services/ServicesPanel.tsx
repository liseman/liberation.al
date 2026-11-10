// Settings' Services: a summary card with the counts and the bulk actions, then a card
// per group (Continuum core, Workers, and so on), with an uptime history card. Each row has its state, a switch to
// start or stop it now, and a menu with the log and Disable; a failed or looping row opens a box with the error and
// Start again. The ones stopped and not enabled here are listed under Turned off, with Enable. This machine's
// load is its own section (Machine.tsx). It replaces the top bar's Services drawer, whose bulk actions it keeps.
import {
  ActionIcon,
  Anchor,
  Badge,
  Button,
  Code,
  ColorSwatch,
  Group,
  Loader,
  Menu,
  Modal,
  Switch,
  Text,
  Tooltip,
} from '@mantine/core';
import { IconChevronDown, IconChevronRight, IconDots } from '@tabler/icons-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type ReactNode, useId, useState } from 'react';
import { HttpError, messageOf, request } from '../../../shared/http';
import { Disclosure } from '../../lib/Disclosure';
import { href, linkClick } from '../../router';
import { SettingsRow, SettingsSection } from '../layout';
import layout from '../layout.module.css';
import { CONFIG_KEY } from '../model';
import { SECTION_POLL_MS, servicesQuery, useServiceIssues } from './api';
import {
  ACTIONS,
  ASK,
  type BulkAction,
  type Busy,
  describe,
  dismissible,
  errorTime,
  groupCount,
  grouped,
  groupLabel,
  isUp,
  isView,
  locked,
  looping,
  overall,
  rowStatus,
  SERVICES_KEY,
  type Service,
  type ServiceAction,
  type ServicesPost,
  type ServicesReply,
  type ServicesView,
  startable,
  summary,
  swept,
  TONE_COLOR,
  TONE_TEXT,
  TONE_TINT,
  type Tone,
  toggleAction,
  troubled,
  turnedOff,
  unreadable,
} from './model';
import { ServiceHistoryPanel } from './ServiceHistory';
import classes from './services.module.css';

const CONFIRM: Record<BulkAction, { title: string; button: string }> = {
  repair: { title: 'Repair services?', button: 'Repair' },
  off: { title: 'Turn every service off?', button: 'Turn off' },
};

/** Phone width: Turned off starts folded, and the rows stack. */
const PHONE = '(max-width: 600px)';

/** The element id of a service's row, which the summary's "needs attention" scrolls to. */
export const rowId = (name: string) => `service-${name}`;

const times = (n: number) => (n === 1 ? 'once' : `${n} times`);

function Dot({ tone }: { tone: Tone }) {
  return (
    <ColorSwatch
      component="span"
      aria-hidden
      className={classes.dot}
      color={TONE_COLOR[tone]}
      size={8}
      withShadow={false}
    />
  );
}

function LogLink() {
  return (
    <Anchor href={href('/debug')} onClick={linkClick} size="sm">
      See the log
    </Anchor>
  );
}

/** A message under the summary: an error (an alert) or a note from the server. */
interface Message {
  text: string;
  error: boolean;
}

type OnAct = (s: Service, action: ServiceAction) => void;

/** When a row last failed; uptime is drawn in the history card. */
function since(s: Service, busy: Busy | null): string {
  if (busy) return '';
  if (troubled(s)) return s.last_error && errorTime(s.last_error) ? `at ${errorTime(s.last_error)}` : '';
  return '';
}

/** What happened to a failed or looping service, in a sentence. */
function whatHappened(s: Service): string {
  const at = s.last_error ? errorTime(s.last_error) : '';
  const n = s.restarts ?? 0;
  if (describe(s).label === 'failed')
    return `Stopped${at ? ` at ${at}` : ''} after an error.${n > 0 ? ` The service manager restarted it ${times(n)}, then stopped trying.` : ''}`;
  if (s.state === 'active' || s.state === 'running')
    return `It is running after ${n} automatic restarts. Dismiss this issue if it has recovered; a new error or restart brings it back.`;
  return `It keeps stopping after an error. The service manager has restarted it ${times(n)}.`;
}

function Problem({
  s,
  control,
  held,
  onAct,
  onDismiss,
}: {
  s: Service;
  control: boolean;
  held: boolean;
  onAct: OnAct;
  onDismiss: (s: Service) => void;
}) {
  const failed = describe(s).label === 'failed';
  return (
    <div className={classes.problem}>
      <p>{whatHappened(s)}</p>
      {s.last_error && <Code className={classes.code}>{s.last_error.text}</Code>}
      <div className={classes.problemActions}>
        {control && failed && s.selected && (
          <Tooltip label={ACTIONS.start.does} events={{ hover: true, focus: true, touch: false }}>
            <span style={{ display: 'inline-flex' }}>
              <Button size="xs" variant="default" onClick={() => onAct(s, 'start')} disabled={held}>
                Start again
              </Button>
            </span>
          </Tooltip>
        )}
        {dismissible(s) && (
          <Button size="xs" variant="default" onClick={() => onDismiss(s)}>
            Dismiss issue
          </Button>
        )}
        <LogLink />
      </div>
    </div>
  );
}

function ServiceRow({
  s,
  control,
  busy,
  held,
  onAct,
  onDismiss,
}: {
  s: Service;
  control: boolean;
  busy: Busy | null;
  held: boolean; // some action is in flight, here or on another row: one at a time
  onAct: OnAct;
  onDismiss: (s: Service) => void;
}) {
  const status = rowStatus(s, busy);
  const why = locked(s, control);
  const on = busy ? busy.on : isUp(s);
  const byHand = !s.installed && s.pids.length > 0; // started outside the service manager, so not switched here
  const trouble = !busy && troubled(s);
  const sub = since(s, busy);
  const at = s.last_error ? errorTime(s.last_error) : '';
  return (
    <SettingsRow
      component="li"
      id={rowId(s.name)}
      tabIndex={-1}
      className={classes.row}
      data-trouble={trouble || undefined}
      aria-label={s.name}
    >
      <div className={classes.line}>
        <Dot tone={status.tone} />
        <Text component="p" m={0} className={`${layout.label} ${classes.name}`}>
          {s.name}
        </Text>
        <Text component="p" m={0} size="sm" c="dimmed" className={classes.desc}>
          {s.description}
        </Text>
        <div className={classes.status} title={s.pids.length ? `pid ${s.pids.join(', ')}` : undefined}>
          <span className={classes.label} style={status.tone === 'bad' ? { color: 'var(--lib-bad)' } : undefined}>
            {status.label}
          </span>
          {sub && (
            <>
              <span className={classes.sep}>{sub.startsWith('up') ? ', ' : ' '}</span>
              <span className={classes.sub}>{sub}</span>
            </>
          )}
        </div>
        {/* Its switch or Enable is off for this, and a disabled control can't take focus: so it's said here. */}
        {control && byHand && (
          <Text component="p" m={0} size="xs" c="dimmed" id={handId(s.name)}>
            {BY_HAND}
          </Text>
        )}
        {control && (
          <div className={classes.controls}>
            {!s.selected ? (
              <EnableButton s={s} held={held} onAct={onAct} />
            ) : s.name === 'dashboard' ? (
              <span className={classes.always} title={why ?? undefined}>
                Always on
              </span>
            ) : (
              <>
                <Tooltip
                  label={why ?? (byHand ? BY_HAND : ACTIONS[toggleAction(s, !on)].does)}
                  events={{ hover: true, focus: true, touch: false }}
                >
                  <span className={classes.switchBox}>
                    <Switch
                      checked={on}
                      onChange={e => onAct(s, toggleAction(s, e.currentTarget.checked))}
                      disabled={!!why || held || byHand}
                      aria-label={`Run ${s.name}`}
                      aria-describedby={byHand ? handId(s.name) : undefined}
                      classNames={{ body: classes.switchBody, track: classes.switchTrack }}
                    />
                  </span>
                </Tooltip>
                <Menu position="bottom-end" width={272} shadow="md">
                  <Menu.Target>
                    <ActionIcon
                      variant="subtle"
                      color="gray"
                      aria-label={`More for ${s.name}`}
                      className={classes.more}
                    >
                      <IconDots size={18} aria-hidden />
                    </ActionIcon>
                  </Menu.Target>
                  <Menu.Dropdown>
                    <Menu.Item component="a" href={href('/debug')} onClick={linkClick}>
                      See the log
                    </Menu.Item>
                    {!s.required && (
                      <>
                        <Menu.Divider />
                        <Menu.Item color="red" onClick={() => onAct(s, 'disable')} disabled={!!why || held}>
                          <Text component="span" fw={600} size="sm">
                            Disable
                          </Text>
                          <span className={classes.menuSub}>{ACTIONS.disable.does}</span>
                        </Menu.Item>
                      </>
                    )}
                  </Menu.Dropdown>
                </Menu>
              </>
            )}
          </div>
        )}
      </div>
      {trouble ? (
        <Problem s={s} control={control} held={held} onAct={onAct} onDismiss={onDismiss} />
      ) : (
        s.last_error && (
          <p className={classes.lastError}>
            {`Recent error${at ? ` at ${at}` : ''}: `}
            <Code className={classes.code}>{s.last_error.text}</Code> <LogLink />{' '}
            {dismissible(s) && (
              <Button size="compact-xs" variant="default" onClick={() => onDismiss(s)}>
                Dismiss issue
              </Button>
            )}
          </p>
        )
      )}
    </SettingsRow>
  );
}

/** Why a service started outside the service manager has no working switch or Enable here. */
const BY_HAND = 'Started outside the service manager, so it can only be stopped where it was started.';
const handId = (name: string) => `${rowId(name)}-by-hand`;

function EnableButton({ s, held, onAct }: { s: Service; held: boolean; onAct: OnAct }) {
  const byHand = !s.installed && s.pids.length > 0;
  return (
    <Tooltip label={byHand ? BY_HAND : ACTIONS.enable.does} events={{ hover: true, focus: true, touch: false }}>
      <span style={{ display: 'inline-flex' }}>
        <Button
          size="xs"
          variant="default"
          onClick={() => onAct(s, 'enable')}
          disabled={held || byHand}
          aria-label={`Enable ${s.name}`}
          aria-describedby={byHand ? handId(s.name) : undefined}
        >
          Enable
        </Button>
      </span>
    </Tooltip>
  );
}

/** The services stopped and not enabled for this machine: open on a desktop, folded on a phone. */
function TurnedOff({
  list,
  control,
  busy,
  held,
  onAct,
}: {
  list: Service[];
  control: boolean;
  busy: Busy | null;
  held: boolean;
  onAct: OnAct;
}) {
  const [open, setOpen] = useState(() => !window.matchMedia?.(PHONE).matches);
  const Chevron = open ? IconChevronDown : IconChevronRight;
  return (
    <SettingsSection aria-label="Turned off">
      <Disclosure
        buttonClassName={classes.offHead}
        opened={open}
        onChange={setOpen}
        keepMounted={false}
        withChevron={false}
        summary={
          <>
            <Chevron size={16} aria-hidden className={classes.chevron} />
            <span className={classes.offTitle}>Turned off</span>
            <Badge className={classes.pill} variant="light" size="sm" tt="none" fw={400}>
              {list.length}
            </Badge>
            <span className={classes.offHint}>Not set to run on this machine</span>
            <span className={classes.showHide} aria-hidden>
              {open ? 'Hide' : 'Show'}
            </span>
          </>
        }
      >
        <p className={classes.offNote}>Not set to run on this machine.</p>
        <ul className={classes.list}>
          {list.map(s => (
            <SettingsRow component="li" key={s.name} className={classes.offRow} aria-label={s.name}>
              <span aria-hidden className={classes.hollow} />
              <div>
                <p className={`${layout.label} ${classes.offName}`} style={{ margin: 0 }}>
                  {s.name}
                </p>
                <Text component="p" m={0} size="sm" c="dimmed">
                  {s.description}
                </Text>
              </div>
              <Badge className={classes.pill} variant="light" size="sm" tt="none" fw={400}>
                {groupLabel(s)}
              </Badge>
              <div className={classes.offAct}>
                {busy?.name === s.name ? (
                  <Text size="sm">starting</Text>
                ) : (
                  control && <EnableButton s={s} held={held} onAct={onAct} />
                )}
              </div>
            </SettingsRow>
          ))}
        </ul>
      </Disclosure>
    </SettingsSection>
  );
}

/** Whether a group has something to look at: a service not running, a crash loop, or an error not yet dismissed. */
export function needsAttention(services: readonly Service[]): boolean {
  return groupCount(services).tone !== 'ok' || services.some(s => looping(s) || !!s.last_error);
}

/** A group of services: one that's all running and quiet starts folded to its
 *  heading and count, so the page leads with what needs attention; one with a problem starts open, and a problem
 *  appearing opens it. Show or Hide is the owner's. */
function GroupCard({ group, children }: { group: ReturnType<typeof grouped>[number]; children: ReactNode }) {
  const count = groupCount(group.services);
  const attention = needsAttention(group.services);
  // Open when it first shows with a problem, or when one appears; it stays open when the problem clears (folding the
  // rows away under the owner's Dismiss or Start again would move what they were looking at), until they hide it.
  const [open, setOpen] = useState(attention);
  const [was, setWas] = useState(attention);
  if (attention !== was) {
    setWas(attention);
    if (attention) setOpen(true);
  }
  const list = useId();
  return (
    <SettingsSection
      title={group.label}
      description={group.blurb}
      trailing={
        <span className={classes.groupTrail}>
          <span style={{ color: TONE_TEXT[count.tone] }}>{count.text}</span>
          <Button
            size="compact-xs"
            variant="subtle"
            aria-expanded={open}
            aria-controls={list}
            aria-label={`${open ? 'Hide' : 'Show'} ${group.label}`}
            onClick={() => setOpen(!open)}
          >
            {open ? 'Hide' : 'Show'}
          </Button>
        </span>
      }
    >
      <ul id={list} className={classes.list} hidden={!open}>
        {children}
      </ul>
    </SettingsSection>
  );
}

/** Scroll to a service's row and put the focus on it. */
function goTo(name: string) {
  const el = document.getElementById(rowId(name));
  el?.scrollIntoView?.({ block: 'center', behavior: 'smooth' });
  el?.focus({ preventScroll: true });
}

function Headline({ view }: { view: ServicesView }) {
  if (!view.ok)
    return (
      <>
        <p className={classes.headline}>Status unknown</p>
        <Text size="sm" c="var(--lib-bad)" role="alert" mt={3}>
          {view.error}
        </Text>
      </>
    );
  const n = summary(view);
  const first = n.attention[0];
  const attention = n.attention.length;
  return (
    <>
      <p className={classes.headline}>
        {n.running || attention || n.stopped ? `${n.running} running` : 'Nothing running'}
        {first && (
          <>
            {', '}
            <Anchor component="button" type="button" className={classes.attention} onClick={() => goTo(first.name)}>
              {`${attention} ${attention === 1 ? 'needs' : 'need'} attention`}
            </Anchor>
          </>
        )}
        {n.stopped > 0 && `, ${n.stopped} stopped`}
      </p>
      <Text size="sm" c="dimmed" mt={3}>
        {`${n.enabled} set to run here.${n.off ? ` ${n.off} turned off.` : ''}`}
      </Text>
    </>
  );
}

function Panel({ view: rawView }: { view: ServicesView | undefined }) {
  const { view, dismiss } = useServiceIssues(rawView);
  const [asking, setAsking] = useState<BulkAction | null>(null); // Repair or All off, while its confirm is open
  const client = useQueryClient();
  const post = useMutation({
    mutationFn: (body: ServicesPost) => request<ServicesReply>('/services', body),
    // Enable and Disable write pack.toml's [services] lists: read the settings again, so the draft's Also run and
    // Don't run show them and a later save can't write the old lists back. Even a failure may have written them.
    onSettled: (_r, _e, body) => {
      if ('action' in body && (body.action === 'enable' || body.action === 'disable'))
        return client.invalidateQueries({ queryKey: CONFIG_KEY });
    },
    onSuccess: async r => {
      if (!isView(r)) return;
      await client.cancelQueries({ queryKey: SERVICES_KEY }); // a poll sent before this reply is older than it
      client.setQueryData(SERVICES_KEY, r);
    },
    // A refused request's body may still carry the view, but HttpError drops it: read it again.
    onError: () => client.invalidateQueries({ queryKey: SERVICES_KEY }),
  });
  const [busy, setBusy] = useState<Busy | null>(null); // the service being switched, until its request ends
  const [startingAll, setStartingAll] = useState(false);
  const [bulk, setBulk] = useState<BulkAction | null>(null); // Repair or All off while that request runs
  const [message, setMessage] = useState<Message | null>(null);

  /** One service's action; resolves to an error message, or null when it worked. */
  const act = async (name: string, action: ServiceAction): Promise<string | null> => {
    const failed = `${ACTIONS[action].label} ${name} failed`;
    try {
      const r = await post.mutateAsync({ name, action });
      return r.ok ? null : r.error || failed;
    } catch (e) {
      return e instanceof HttpError ? e.reason || failed : `${failed}: ${messageOf(e)}`;
    }
  };
  const onAct = async (s: Service, action: ServiceAction) => {
    setBusy({ name: s.name, on: action === 'on' || action === 'start' || action === 'enable' });
    const e = await act(s.name, action);
    setMessage(e ? { text: e, error: true } : null);
    setBusy(null);
  };

  const list = view?.services ?? [];
  const toStart = startable(list);
  const startAll = async () => {
    setStartingAll(true);
    const errs: string[] = [];
    for (const s of toStart) {
      setBusy({ name: s.name, on: true });
      const e = await act(s.name, s.installed ? 'start' : 'on'); // a stopped one starts; one not installed goes on
      if (e) errs.push(e);
    }
    setBusy(null);
    setStartingAll(false);
    setMessage(errs.length ? { text: errs.join('\n'), error: true } : null);
  };

  const runAll = async (action: BulkAction) => {
    setAsking(null);
    setBulk(action);
    try {
      const r = await post.mutateAsync({ all: action });
      setMessage(
        r.ok ? (r.note ? { text: r.note, error: false } : null) : { text: r.error || 'That failed', error: true },
      );
    } catch (e) {
      const text = e instanceof HttpError ? e.reason || 'That failed' : `That failed: ${messageOf(e)}`;
      setMessage({ text, error: true });
    }
    setBulk(null);
  };

  const idle = !busy && !bulk;
  const control = !!view?.control;
  const off = list.filter(turnedOff);
  const tone = overall(view);

  return (
    <div className={classes.stack}>
      {view && !view.control && (
        <p className={classes.viewOnly} role="status">
          View only. Services can be switched only on the machine Liberational runs on.
        </p>
      )}
      <SettingsSection className={classes.summary} aria-label="Health">
        <div className={classes.verdict}>
          <ColorSwatch
            component="span"
            aria-hidden
            className={classes.bigDot}
            color={TONE_COLOR[tone]}
            size={12}
            withShadow={false}
            style={{ boxShadow: `0 0 0 5px ${TONE_TINT[tone]}` }}
          />
          <div>
            {view ? (
              <Headline view={view} />
            ) : (
              <Group gap="xs" role="status" aria-live="polite">
                <Loader size="xs" />
                <Text size="sm" c="dimmed">
                  Loading...
                </Text>
              </Group>
            )}
          </div>
        </div>
        {control && (
          <div className={classes.bulk}>
            {(toStart.length > 0 || startingAll) && (
              <Tooltip
                label={`Start ${toStart.map(s => s.name).join(', ')}`}
                events={{ hover: true, focus: true, touch: false }}
              >
                <span className={classes.startAll} style={{ display: 'inline-flex' }}>
                  <Button variant="default" disabled={!idle} onClick={startAll} w="100%">
                    {startingAll ? `Starting ${busy?.name ?? ''}` : `Start ${toStart.length} stopped`}
                  </Button>
                </span>
              </Tooltip>
            )}
            <Tooltip
              label="Remove and reinstall every enabled service except the dashboard"
              events={{ hover: true, focus: true, touch: false }}
            >
              <span style={{ display: 'inline-flex' }}>
                <Button variant="default" disabled={!idle} onClick={() => setAsking('repair')} w="100%">
                  {bulk === 'repair' ? 'Repairing...' : 'Repair'}
                </Button>
              </span>
            </Tooltip>
            <Tooltip
              label="Turn off every service except the dashboard"
              events={{ hover: true, focus: true, touch: false }}
            >
              <span style={{ display: 'inline-flex' }}>
                <Button
                  variant="subtle"
                  className={classes.allOff}
                  disabled={!idle}
                  onClick={() => setAsking('off')}
                  w="100%"
                >
                  {bulk === 'off' ? 'Turning off...' : 'Turn all off'}
                </Button>
              </span>
            </Tooltip>
          </div>
        )}
      </SettingsSection>
      {message && (
        <Text
          size="sm"
          c={message.error ? 'var(--lib-bad)' : 'dimmed'}
          role={message.error ? 'alert' : 'status'}
          className={classes.message}
        >
          {message.text}
        </Text>
      )}
      <ServiceHistoryPanel services={list} />
      {grouped(list).map(g => (
        <GroupCard key={g.key} group={g}>
          {g.services.map(s => (
            <ServiceRow
              key={s.name}
              s={s}
              control={control}
              busy={busy?.name === s.name ? busy : swept(s, bulk)}
              held={!idle}
              onAct={onAct}
              onDismiss={dismiss}
            />
          ))}
        </GroupCard>
      ))}
      {off.length > 0 && <TurnedOff list={off} control={control} busy={busy} held={!idle} onAct={onAct} />}
      {control && (
        <p className={classes.footnote}>
          The switch starts or stops a service now. A stopped service starts again when the computer restarts. To keep
          one off, choose Disable from its menu.
        </p>
      )}
      <Modal
        opened={asking !== null}
        onClose={() => setAsking(null)}
        title={asking ? CONFIRM[asking].title : ''}
        centered
      >
        <Text size="sm">{asking && ASK[asking]}</Text>
        <Group justify="flex-end" gap="xs" mt="md">
          <Button variant="default" onClick={() => setAsking(null)}>
            Cancel
          </Button>
          <Button color={asking === 'off' ? 'red' : undefined} onClick={() => asking && runAll(asking)}>
            {asking ? CONFIRM[asking].button : ''}
          </Button>
        </Group>
      </Modal>
    </div>
  );
}

/** The Services section's panel. */
export function Services() {
  const query = useQuery(servicesQuery(SECTION_POLL_MS));
  return <Panel view={query.isError ? unreadable(messageOf(query.error)) : query.data} />;
}
