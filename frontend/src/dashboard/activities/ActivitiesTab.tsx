// Known › Activities: teach the house which repeated movements it detects are real, and what to call them, so
// Ask can count them. Laid out like Pets: the kinds on the left (counted in Ask, or how far each has to go), and on the
// right the clips of the last week still waiting for an answer, newest first, each with its own small player.
// Answers and names use Review's routes and stores (/activity/feedback, /activity/name), so Ask's counting reads them
// as before; "Not sure" only sets a clip aside for now, since the stores take right or wrong. The kinds double as the
// clips filter: pick one to see only its clips, remembered per browser. On a phone the clips come first (Known's
// design pass, P5): the kinds are a row of chips that scrolls sideways under the clips' heading, and the chosen kind's
// progress is one line under it. The list of activities is the owner's: + adds one, and each row's bin deletes
// it after asking, with an Undo.
import {
  ActionIcon,
  Alert,
  Autocomplete,
  Badge,
  Button,
  Group,
  Modal,
  Progress,
  Text,
  Textarea,
  TextInput,
  Tooltip,
  UnstyledButton,
  VisuallyHidden,
} from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { IconPlus, IconTrash } from '@tabler/icons-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, type ReactNode, useState } from 'react';
import { messageOf, request } from '../../shared/http';
import { confirm } from '../lib/confirm';
import { hm } from '../lib/format';
import { CountSkeleton } from '../lib/loading';
import type { Message } from '../lib/message';
import { LoadError, PageEmpty, PageHeader, PageLoading, PHONE_WIDTH } from '../lib/page';
import { Toast } from '../lib/Toast';
import { type ActivitiesResponse, type ActivityPromotion, type DetectedActivity, reviewKey } from '../review/api';
import { activityKind, activitySubject, promotionText } from '../review/lanes';
import classes from './activities.module.css';
import { ClipPlayer } from './ClipPlayer';
import {
  type ActivityKind,
  addKind,
  classesOf,
  deleteKind,
  kindsQuery,
  lookingFor,
  refreshKinds,
  restoreKind,
} from './kinds';
import {
  type KindRow,
  kindKey,
  kindRows,
  loadKindFilter,
  progress,
  RECENT_DAYS,
  SPOT_EVERY,
  saveKindFilter,
  toCheck,
  unanswered,
  useRecentActivities,
} from './recent';

const PAGE = 20; // clips shown at a time, newest first: a week can hold hundreds

const dayWords = (day: string): string => {
  const d = new Date(`${day}T12:00:00`);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
};

/** Every activity list is cached per day; an answer can change any kind's counting, so all of them refresh. */
const refreshAll = (client: ReturnType<typeof useQueryClient>) =>
  client.invalidateQueries({
    predicate: ({ queryKey }) =>
      queryKey.length === 4 && queryKey[0] === 'review' && queryKey[1] === 'day' && queryKey[3] === 'activities',
  });

function ClipCard({
  day,
  item,
  names,
  label,
  onAside,
}: {
  day: string;
  item: DetectedActivity;
  names: readonly string[]; // the owner's activities, offered as names: a clip called by one teaches it
  label?: string; // its activity's own name, when the owner defined it
  onAside: () => void;
}) {
  const client = useQueryClient();
  const [name, setName] = useState(item.name ?? '');
  // The name field opens on demand, or at once for a clip already named.
  const [naming, setNaming] = useState(!!item.name);
  const kind = item.name ?? label ?? activityKind(item);
  const answer = useMutation({
    mutationKey: ['review', 'activity-feedback', item.id],
    mutationFn: (verdict: 'right' | 'wrong') =>
      request<{ ok: boolean; verdict: 'right' | 'wrong' }>('/activity/feedback', {
        activity: item.id,
        start: item.start,
        verdict,
      }),
    onSuccess: async data => {
      client.setQueryData<ActivitiesResponse>(
        reviewKey.activities(day),
        old =>
          old && { ...old, items: old.items.map(it => (it.id === item.id ? { ...it, feedback: data.verdict } : it)) },
      );
      await refreshAll(client);
    },
  });
  const rename = useMutation({
    mutationFn: (next: string) => request('/activity/name', { activity: item.id, start: item.start, name: next }),
    onSuccess: () => refreshAll(client),
  });
  const saveName = (value = name) => {
    const next = value.trim();
    if (next && next !== (item.name ?? '').trim()) rename.mutate(next);
  };
  const who = activitySubject(item);
  const facts = [
    `${dayWords(day)} ${hm(item.start)}`,
    item.camera,
    `${Math.max(1, Math.round(item.end - item.start))} s`,
    who,
    item.count == null ? null : `${item.count} repeats`,
  ].filter(Boolean);
  return (
    <li className={classes.card} aria-label={`${kind} at ${hm(item.start)}`}>
      <div className={classes.clip}>
        <ClipPlayer cam={item.camera} start={item.start} end={item.end} label={kind} />
        <Text size="xs" c="dimmed">
          {facts.join(' · ')}
          {item.promoted ? '' : ' · unconfirmed'}
        </Text>
      </div>
      <div className={classes.ask}>
        <Text fw={600}>Is this “{capital(kind)}”?</Text>
        {/* Three answers at the same weight: checking a model, No counts as much as Yes. */}
        <Button.Group className={classes.answers} aria-label="Answer">
          <Button
            variant="default"
            onClick={() => answer.mutate('right')}
            loading={answer.isPending && answer.variables === 'right'}
          >
            Yes
          </Button>
          <Button
            variant="default"
            onClick={() => answer.mutate('wrong')}
            loading={answer.isPending && answer.variables === 'wrong'}
          >
            No
          </Button>
          <Button variant="default" onClick={onAside} disabled={answer.isPending}>
            Not sure
          </Button>
        </Button.Group>
        {answer.isError && (
          <Text size="sm" c="red">
            Couldn't save that answer: {messageOf(answer.error)}
          </Text>
        )}
        {!naming ? (
          <Button variant="subtle" size="compact-sm" className={classes.callIt} onClick={() => setNaming(true)}>
            Call it something else
          </Button>
        ) : (
          <Autocomplete
            // Opened by its own button, so focus goes where the owner is about to type.
            autoFocus={!item.name}
            label="Call it"
            placeholder={activityKind(item).toLocaleLowerCase()}
            data={[...names]}
            value={name}
            onChange={setName}
            onOptionSubmit={value => {
              setName(value);
              saveName(value);
            }}
            onBlur={() => saveName()}
            onKeyDown={e => {
              if (e.key === 'Enter') saveName();
            }}
            maw={300}
            error={rename.isError ? `Couldn't save the name: ${messageOf(rename.error)}` : undefined}
            description={rename.isSuccess ? 'Saved' : undefined}
          />
        )}
      </div>
    </li>
  );
}

/** How far a learning kind is from being counted, in one wording: "3 more answers to count". */
export function learningText(p: ActivityPromotion | undefined): string {
  if (p?.status === 'needs_ratings') {
    const n = p.remaining_ratings;
    if (typeof n === 'number' && Number.isInteger(n) && n > 0)
      return `${n} more ${n === 1 ? 'answer' : 'answers'} to count`;
  }
  if (p?.status === 'needs_correct_ratings') return 'More Yes answers to count';
  return promotionText(p);
}

const capital = (text: string): string => text.charAt(0).toLocaleUpperCase() + text.slice(1);

/** Names in a sentence: "A", "A and B", or "A, B, and C". */
const listed = (names: readonly string[]): string =>
  names.length < 3 ? names.join(' and ') : `${names.slice(0, -1).join(', ')}, and ${names.at(-1)}`;

/** Why the list holds only some of the counted kinds' clips: "Walked past is counted in Ask already, so 1 in 10 of its
 *  clips are here to check." */
export function thinnedText(kinds: readonly string[]): string {
  const names = listed(kinds.map(capital));
  return kinds.length === 1
    ? `${names} is counted in Ask already, so 1 in ${SPOT_EVERY} of its clips are here to check.`
    : `${names} are counted in Ask already, so 1 in ${SPOT_EVERY} of their clips are here to check.`;
}

const clipsWords = (n: number): string => (n === 1 ? '1 clip to check' : `${n} clips to check`);
// After a count shown on its own ("52"), read as "52 clips to check": the button's name keeps its visible words (axe)
// (the space is its own text: a hidden span's own leading space is trimmed from the name)
const afterCount = (n: number) => (
  <>
    {' '}
    <VisuallyHidden>{n === 1 ? 'clip to check' : 'clips to check'}</VisuallyHidden>
  </>
);

/** One roster row, which is also the clips filter: pressed when it's the kind shown. */
function KindButton({
  active,
  waiting,
  counting = false,
  onClick,
  onDelete,
  name,
  children,
}: {
  active: boolean;
  waiting: number;
  counting?: boolean; // its count is still loading: a badge-sized skeleton where the number goes
  onClick: () => void;
  onDelete?: () => void; // the owner's activity: it can be deleted
  name?: string;
  children: ReactNode;
}) {
  return (
    <li className={classes.kindRow}>
      <UnstyledButton
        className={classes.kind}
        data-active={active || undefined}
        aria-pressed={active}
        onClick={onClick}
      >
        <div className={classes.kindBody}>{children}</div>{' '}
        {counting ? (
          <CountSkeleton chars={3} />
        ) : (
          waiting > 0 && (
            <>
              <Badge variant={active ? 'filled' : 'light'}>{waiting}</Badge>
              {afterCount(waiting)}
            </>
          )
        )}
      </UnstyledButton>
      {onDelete && <DeleteKind name={name ?? ''} onDelete={onDelete} />}
    </li>
  );
}

/** A row's bin: quiet until it's meant, as Ingest's, and it asks first. */
function DeleteKind({ name, onDelete }: { name: string; onDelete: () => void }) {
  return (
    <Tooltip label="Delete this activity">
      <ActionIcon
        variant="subtle"
        color="gray"
        className={classes.delete}
        aria-label={`Delete ${name}`}
        onClick={onDelete}
      >
        <IconTrash size={16} aria-hidden />
      </ActionIcon>
    </Tooltip>
  );
}

function AddButton({ onClick }: { onClick: () => void }) {
  return (
    <Tooltip label="Add an activity to look for">
      <ActionIcon variant="default" size="lg" aria-label="Add an activity" onClick={onClick}>
        <IconPlus size={18} aria-hidden />
      </ActionIcon>
    </Tooltip>
  );
}

/** Add an activity: its name, and a few words on what it looks like, which help find the classes to look for. */
function AddActivity({
  opened,
  onClose,
  onAdded,
}: {
  opened: boolean;
  onClose: () => void;
  onAdded: (made: ActivityKind) => void;
}) {
  const client = useQueryClient();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const add = useMutation({
    mutationFn: () => addKind(name.trim(), description.trim()),
    onSuccess: async ({ kind }) => {
      setName('');
      setDescription('');
      onAdded(kind);
      await refreshKinds(client);
    },
  });
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (name.trim()) add.mutate();
  };
  return (
    <Modal opened={opened} onClose={onClose} title="Add an activity" centered>
      <form onSubmit={submit} className={classes.addForm}>
        <TextInput
          data-autofocus
          label="Name"
          placeholder="Squats"
          required
          maxLength={60}
          value={name}
          onChange={e => setName(e.currentTarget.value)}
        />
        <Textarea
          label="What it looks like"
          description="Optional. A few words help it find what to look for."
          placeholder="Bending at the knees and standing back up"
          rows={2}
          maxLength={300}
          value={description}
          onChange={e => setDescription(e.currentTarget.value)}
        />
        {add.isError && (
          <Text size="sm" c="var(--lib-bad)">
            Couldn't add it: {messageOf(add.error)}
          </Text>
        )}
        <Group justify="flex-end" gap="xs">
          <Button variant="default" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" loading={add.isPending} disabled={!name.trim()}>
            Add
          </Button>
        </Group>
      </form>
    </Modal>
  );
}

/** Adding, deleting (after asking), and undoing a delete, with what each said for the toast. */
function useKindEdits() {
  const client = useQueryClient();
  const [msg, setMsg] = useState<Message<string>>(null);
  const [adding, setAdding] = useState(false);
  const restore = useMutation({
    mutationFn: restoreKind,
    onSuccess: async ({ kind }) => {
      setMsg({ text: `${kind.name} is back` });
      await refreshKinds(client);
    },
    onError: e => setMsg({ err: `Couldn't bring it back: ${messageOf(e)}` }),
  });
  const remove = async (k: ActivityKind) => {
    const kept = !k.answers
      ? 'It has no answers yet; Undo (or adding it again) brings it back.'
      : k.answers === 1
        ? 'Its 1 answer is kept, so Undo (or adding it again) brings it back.'
        : `Its ${k.answers} answers are kept, so Undo (or adding it again) brings them back.`;
    const yes = await confirm({
      title: `Delete “${k.name}”?`,
      body: `The cameras stop looking for it, and its clips leave this page and Ask's counts. ${kept}`,
      confirm: 'Delete',
      danger: true,
      id: 'delete-activity',
    });
    if (!yes) return;
    try {
      await deleteKind(k.kind);
      setMsg({ text: `Deleted ${k.name}`, undo: k.kind });
    } catch (e) {
      setMsg({ err: `Couldn't delete ${k.name}: ${messageOf(e)}` });
    }
    await refreshKinds(client);
  };
  const added = (k: ActivityKind) => {
    setAdding(false);
    setMsg({ text: `Added ${k.name}. ${lookingFor(k)}` });
  };
  return { msg, setMsg, adding, setAdding, restore, remove, added };
}

/** The kinds as a sideways row of chips over the clips, a phone's roster: All, then each kind with how many wait. It
 *  keeps its height while the kinds load, so the clips under it don't move when they arrive. */
function KindChips({
  rows,
  selected,
  all,
  counting,
  waitingOf,
  onChoose,
}: {
  rows: KindRow[];
  selected: KindRow | undefined;
  all: number;
  counting: boolean; // the days still load: All's count is a skeleton until they have
  waitingOf: ReadonlyMap<string, number>;
  onChoose: (key: string | null) => void;
}) {
  const chip = (key: string | null, label: string, n: number, active: boolean, loading = false) => (
    <li key={key ?? 'all'}>
      <UnstyledButton
        className={classes.chip}
        data-active={active || undefined}
        aria-pressed={active}
        onClick={() => onChoose(key)}
      >
        {label}{' '}
        {loading ? (
          <CountSkeleton chars={2} />
        ) : (
          n > 0 && (
            <>
              <span className={classes.chipCount}>{n}</span>
              {afterCount(n)}
            </>
          )
        )}
      </UnstyledButton>
    </li>
  );
  return (
    <ul className={classes.chips} aria-label="Show clips of">
      {chip(null, 'All', all, !selected, counting)}
      {rows.map(r => chip(kindKey(r.kind), capital(r.label), waitingOf.get(kindKey(r.kind)) ?? 0, selected === r))}
    </ul>
  );
}

export function ActivitiesTab() {
  // Read at once, not after the first render: a phone first laid out as a desktop and then switched moved every clip.
  const phone = useMediaQuery(PHONE_WIDTH, undefined, { getInitialValueInEffect: false }) ?? false;
  const recent = useRecentActivities();
  const defined = useQuery(kindsQuery()).data?.kinds;
  const edits = useKindEdits();
  const [aside, setAside] = useState<ReadonlySet<string>>(() => new Set());
  const [shown, setShown] = useState(PAGE);
  const [chosen, setChosen] = useState<string | null>(loadKindFilter);
  const all = recent.items.map(r => r.item);
  const rows = kindRows(all, recent.promotion, defined);
  const names = (defined ?? []).map(k => k.name);
  const open = recent.items.filter(r => unanswered(r.item) && !aside.has(r.item.id));
  const waitingOf = new Map<string, number>();
  for (const { item } of open) waitingOf.set(kindKey(item.kind), (waitingOf.get(kindKey(item.kind)) ?? 0) + 1);
  // A remembered kind that's no longer listed (its clips aged out of the week) shows every kind instead.
  const selected = rows.find(r => chosen !== null && kindKey(r.kind) === chosen);
  // A kind counted in Ask shows only a spot-check sample of its clips among all activities; picking it
  // shows every one.
  const sampled = toCheck(open, recent.promotion);
  const thinned = !selected && sampled.length < open.length;
  const waiting = selected ? open.filter(r => kindKey(r.item.kind) === chosen) : sampled;
  const choose = (key: string | null) => {
    setChosen(key);
    saveKindFilter(key);
    setShown(PAGE);
  };
  // The counts wait for every day of the week: a count read off the days in so far rose as each came in. A failed day
  // shows its error instead.
  const counting = recent.pending && !recent.error;
  const counted = rows.filter(r => r.counted);
  const learning = rows.filter(r => !r.counted);
  const row = (r: KindRow, body: ReactNode) => (
    <KindButton
      key={kindKey(r.kind)}
      active={selected === r}
      waiting={waitingOf.get(kindKey(r.kind)) ?? 0}
      onClick={() => choose(kindKey(r.kind))}
      onDelete={r.defined ? () => r.defined && void edits.remove(r.defined) : undefined}
      name={r.label}
    >
      {body}
    </KindButton>
  );
  const addButton = <AddButton onClick={() => edits.setAdding(true)} />;
  return (
    <div className={classes.page}>
      {/* A phone has no roster, and so no "Activities" heading: this one is for screen readers (page-has-heading-one) */}
      {phone && <VisuallyHidden component="h1">Activities</VisuallyHidden>}
      {!phone && (
        <aside className={classes.roster} aria-label="Activity kinds">
          <PageHeader
            title="Activities"
            meta={
              counting ? (
                <CountSkeleton chars={7} label="Loading how many kinds" />
              ) : rows.length === 1 ? (
                '1 kind'
              ) : (
                `${rows.length} kinds`
              )
            }
            actions={addButton}
          />
          {/* Held while the days load, so the kinds under it don't drop when it fills in. */}
          {counting ? (
            <div className={classes.callout}>
              <CountSkeleton chars={16} label="Loading how many clips to check" />
            </div>
          ) : sampled.length > 0 ? (
            <a className={classes.callout} href="#clips">
              {clipsWords(sampled.length)} ›
            </a>
          ) : null}
          <ul className={classes.kinds} aria-label="Show clips of">
            <KindButton active={!selected} waiting={sampled.length} counting={counting} onClick={() => choose(null)}>
              <b>All activities</b>
            </KindButton>
          </ul>
          {/* The kinds wait for every day: listed as the days came in, they were inserted and moved between groups
              while on screen (the page's last load shift). Until then, a skeleton holds their place. */}
          {recent.pending ? (
            <PageLoading label="kinds" shape="rows" lines={4} />
          ) : (
            <>
              {counted.length > 0 && <h2 className={classes.group}>Counted in Ask</h2>}
              <ul className={classes.kinds}>
                {counted.map(r =>
                  row(
                    r,
                    <>
                      <b>{r.label}</b>{' '}
                      <Text size="sm" c="dimmed">
                        {r.promotion ? `${r.promotion.right} confirmed` : promotionText(r.promotion)}
                      </Text>
                    </>,
                  ),
                )}
              </ul>
              {learning.length > 0 && <h2 className={classes.group}>Learning</h2>}
              <ul className={classes.kinds}>
                {learning.map(r =>
                  row(
                    r,
                    <>
                      <b>{r.label}</b>{' '}
                      <Text size="sm" c="dimmed">
                        {r.defined && !r.defined.rule && !classesOf(r.defined).length
                          ? lookingFor(r.defined)
                          : learningText(r.promotion)}
                      </Text>
                      <Progress
                        value={progress(r.promotion) * 100}
                        size="sm"
                        aria-label={`${r.label}: ${promotionText(r.promotion)}`}
                      />
                    </>,
                  ),
                )}
              </ul>
            </>
          )}
          {/* Under the kinds, so it waits for them: shown while they still fill in, it was pushed down by each. */}
          {!recent.pending && (
            <Text size="xs" c="dimmed">
              A kind is counted in Ask once enough of its clips are answered Yes. Clips answered No are left out of
              counts, and Yes answers teach it what to look for. Pick a kind to see only its clips.
            </Text>
          )}
        </aside>
      )}
      <section className={classes.main} aria-labelledby="clips">
        <Group gap="xs" align="center">
          <h2 id="clips" className={classes.mainTitle}>
            {selected ? `${capital(selected.label)}: clips to check` : 'Clips to check'}
          </h2>
          {counting ? (
            <CountSkeleton chars={3} label="Loading how many clips to check" />
          ) : (
            !recent.pending && (
              <Badge variant="light" size="lg" aria-label={clipsWords(waiting.length)}>
                {waiting.length}
              </Badge>
            )
          )}
          {phone && <span className={classes.phoneAdd}>{addButton}</span>}
        </Group>
        {phone ? (
          <>
            <KindChips
              rows={rows}
              selected={selected}
              all={sampled.length}
              counting={counting}
              waitingOf={waitingOf}
              onChoose={choose}
            />
            {selected && (
              <Group gap="xs" wrap="nowrap" justify="space-between">
                <Text size="sm" c="dimmed">
                  {selected.counted && selected.promotion
                    ? `Counted in Ask · ${selected.promotion.right} confirmed`
                    : learningText(selected.promotion)}
                </Text>
                {selected.defined && (
                  <DeleteKind
                    name={selected.label}
                    onDelete={() => selected.defined && void edits.remove(selected.defined)}
                  />
                )}
              </Group>
            )}
          </>
        ) : (
          <Text c="dimmed">
            Movements the cameras saw over the last {RECENT_DAYS} days. Answer each, and name it in your own words if
            you like.
          </Text>
        )}
        {/* Its line is held while the days still load: whether a counted kind thins the list is known only once they
            have, and the line turning up then pushed every clip down (a load shift). */}
        {(thinned || recent.pending) && (
          <Text size="sm" c="dimmed" aria-hidden={!thinned || undefined}>
            {thinned ? thinnedText(counted.filter(r => waitingOf.has(kindKey(r.kind))).map(r => r.label)) : '\u00a0'}
          </Text>
        )}
        {/* The clips wait for the newest day, which says which kinds are counted and so sampled: shown before it,
            most of them left again when it came. */}
        {!recent.ranked && <PageLoading label="activities" shape="rows" lines={3} />}
        {recent.error ? (
          <LoadError
            message="Couldn't load some days of activities"
            detail={messageOf(recent.error)}
            onRetry={recent.retry}
          />
        ) : null}
        {!recent.pending && !waiting.length && !recent.error && (
          <PageEmpty
            title={
              selected
                ? `No ${selected.label.toLocaleLowerCase()} clips left to check`
                : aside.size
                  ? 'Nothing else to check for now'
                  : 'Nothing to check'
            }
            compact
            action={
              selected && open.length > 0 ? (
                <Button variant="subtle" size="compact-sm" onClick={() => choose(null)}>
                  Show all activities
                </Button>
              ) : undefined
            }
          >
            {selected || aside.size ? undefined : 'New clips show up here as the cameras see them.'}
          </PageEmpty>
        )}
        <ul className={classes.cards}>
          {(recent.ranked ? waiting.slice(0, shown) : []).map(({ day, item }) => (
            <ClipCard
              key={item.id}
              day={day}
              item={item}
              names={names}
              label={defined?.find(k => k.kind === item.kind)?.name}
              onAside={() => setAside(previous => new Set(previous).add(item.id))}
            />
          ))}
        </ul>
        {waiting.length > shown && (
          <Button variant="default" onClick={() => setShown(n => n + PAGE)} style={{ alignSelf: 'flex-start' }}>
            Show {Math.min(PAGE, waiting.length - shown)} more of {waiting.length - shown}
          </Button>
        )}
        {recent.truncated.length > 0 && (
          <Alert color="gray">
            Some days have more activities than one listing holds; the newest are shown, so older clips may be missing.
          </Alert>
        )}
      </section>
      <AddActivity opened={edits.adding} onClose={() => edits.setAdding(false)} onAdded={edits.added} />
      <Toast
        message={edits.msg}
        onUndo={kind => edits.restore.mutate(kind)}
        busy={edits.restore.isPending}
        label="Activities"
        onClose={() => edits.setMsg(null)}
      />
    </div>
  );
}
