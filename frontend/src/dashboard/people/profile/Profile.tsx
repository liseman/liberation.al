// A person's contact card, beside the People list: their main photo, name, where they are, and what's known; then
// their clearest photos, recent video, recent voice, About (with the public lookup), and their voice clone. Rename
// sits in the header, and Role, Phone number, Same person as, Pick what's theirs, and Delete in its "More" menu.
// Each photo, video, and line has its own menu to correct it (#309): "Not Dana…", Use as main photo, and Hide.
// A new person's card opens on "pick what's theirs" (Pickers.tsx) instead.
import {
  ActionIcon,
  Badge,
  Button,
  Group,
  Menu,
  NativeSelect,
  Skeleton,
  Stack,
  Text,
  TextInput,
  Timeline,
  Title,
} from '@mantine/core';
import { modals } from '@mantine/modals';
import { IconDots, IconMicrophone, IconUserMinus } from '@tabler/icons-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, type ReactNode, useEffect, useId, useRef, useState } from 'react';
import { HttpError, RequestTimeout } from '../../../shared/http';
import { ProcessingImages } from '../../../shared/ProcessingImages';
import { confirm } from '../../lib/confirm';
import { isoDay, plural, seenStatus } from '../../lib/format';
import { BlockSkeleton, SlowLoader, useSlow } from '../../lib/loading';
import { LoadError } from '../../lib/page';
import { play, stop, usePlayer } from '../../lib/player';
import { Toast as FloatingToast } from '../../lib/Toast';
import { reviewURL } from '../../review/url';
import { linkClick } from '../../router';
import { shell } from '../../shell';
import {
  addPerson,
  type Card,
  CLONES_KEY,
  type Clone,
  type Correction,
  type CorrectKind,
  cardKey,
  cloneAudio,
  correct,
  deletePerson,
  describe,
  getCard,
  getClones,
  getProfile,
  type Lookup,
  lookUp,
  mergeIntoOwner,
  type Person,
  type Profile as ProfileData,
  profileFromRow,
  profileKey,
  rename,
  sayClone,
  undoChange,
  type ViewReply,
  type WhyTarget,
} from '../api';
import { Recorder } from '../clones/Recorder';
import { build, cloneOf, remove, useClones } from '../clones/store';
import { MainPhoto, PhotoGrid } from '../entity';
import { References } from '../References';
import { WhoIsThis } from '../WhoIsThis';
import { WhyName } from '../WhyName';
import { PlayIcon, StopIcon } from './icons';
import { PickClips } from './PickClips';
import { Pickers } from './Pickers';
import { Empty, ItemActions, type MenuAction, PlayLine, Section, type Status, StatusLine, type Toast } from './parts';
import classes from './profile.module.css';
import { dayTime, failure, refusal, removedText, roleLine, whereabouts } from './role';
import { VideoTile } from './VideoTile';

const LOOKUP_FAILED = 'The lookup failed. Check that the agent backend is reachable, then try again.';
const DIDNT_WORK = 'That didn’t work';
const PHONE = /^\+[1-9]\d{6,14}$/; // a quick check before sending; the server's (people/contacts.py) is the one that counts

type Mode = 'rename' | 'merge' | 'describe' | 'phone' | null;

export interface ProfileProps {
  name: string;
  person: Person | undefined; // the list entry: its photo shows while the card loads
  people: Person[]; // everyone, for Same person as and the merge's phone check
  onChanged: (name?: string, reply?: ViewReply) => void;
  onDeleted: (reply: ViewReply, text: string) => void;
  onBack: () => void; // Back to people (narrow windows only)
  isNew?: boolean; // just added: the card opens on "pick what's theirs"
  onPickDone?: () => void; // picking ended (added, or skipped)
}

/** A photo, video, or line on the card that can be corrected, and how to say it. */
interface Target {
  kind: CorrectKind;
  id: number | string;
  label: string; // "this photo", "this video", or "this line"
}
type Fix = (t: Target, action: 'not' | 'main' | 'hide', from?: HTMLElement | null) => void;

const secs = (n: number | undefined) => `${Math.round(n ?? 0)} s`;

/** "34 s heard, 6 s usable for cloning, needs 12 s." (#388) */
export function cloneSpeech(c: Clone): string {
  return `${secs(c.clean_s)} heard, ${secs(c.usable_s)} usable for cloning, needs ${secs(c.needs_s ?? 12)}.`;
}

/** Why a voice can't be cloned yet, and what would help. */
export function cloneShort(c: Clone, name: string): string {
  const heard = c.clean_s ?? 0;
  if (!heard)
    return `${name}'s voice can be cloned once Liberational has heard about 12 seconds of clear speech from them.`;
  const rejected = c.rejected_s ?? 0;
  const why = rejected >= 1 ? ` ${secs(rejected)} matched to their voice didn't sound enough like it.` : '';
  return `${why.trim()} Name more of ${name}'s lines (in Live, Review, or here), pick clips, or record a reference.`.trim();
}

/** What explains a photo's or a line's name (#288): its sighting, or its line key. Videos have no one name. */
function whyOf(t: Target): WhyTarget | null {
  if (t.kind === 'shot' && typeof t.id === 'string') return { shot: t.id }; // a detection's payload id (#549)
  if (t.kind === 'voice' && typeof t.id === 'string') return { line: t.id };
  return null;
}

/** The ⋯ menu on one photo, video, or line: Why this is <name> (photos and lines, #288), Not <name>…, Use as main
 *  photo (photos only), and Hide from the card. Why opens beside the ⋯, with Not <name>… in it too. */
function ItemMenu({ name, t, onFix, className }: { name: string; t: Target; onFix: Fix; className?: string }) {
  const button = useRef<HTMLButtonElement>(null); // where focus goes back to after Who is this?
  const [why, setWhy] = useState(false);
  const target = whyOf(t);
  const actions: MenuAction[] = [
    ...(target ? [{ label: `Why this is ${name}`, onClick: () => setWhy(true) }] : []),
    {
      label: `Not ${name}…`,
      icon: <IconUserMinus size={14} aria-hidden />,
      onClick: () => onFix(t, 'not', button.current),
    },
    ...(t.kind === 'shot' ? [{ label: 'Use as main photo', onClick: () => onFix(t, 'main') }] : []),
    { label: 'Hide from the card', onClick: () => onFix(t, 'hide') },
  ];
  const menu = <ItemActions ref={button} label={t.label} actions={actions} className={className} />;
  if (!target) return menu;
  return (
    <WhyName
      target={target}
      name={name}
      opened={why}
      onClose={() => {
        setWhy(false);
        button.current?.focus();
      }}
      actions={
        <Button
          size="compact-xs"
          variant="subtle"
          leftSection={<IconUserMinus size={14} aria-hidden />}
          onClick={() => {
            setWhy(false);
            onFix(t, 'not', button.current);
          }}
        >
          {`Not ${name}…`}
        </Button>
      }
    >
      <span className={classes.whyAnchor}>{menu}</span>
    </WhyName>
  );
}

/** A link into Review that stays in the app. */
function ReviewLink({ to, children, label }: { to: string; children: ReactNode; label?: string }) {
  return (
    <a href={to} onClick={linkClick} className={classes.link} aria-label={label}>
      {children}
    </a>
  );
}

function Photos({ card, onFix }: { card: Card; onFix: Fix }) {
  const shots = card.shots.slice(0, 6);
  return (
    <Section title="Photos">
      {shots.length ? (
        <>
          <Text size="sm" c="dimmed" mb={8}>
            The clearest front-facing shots
          </Text>
          <PhotoGrid
            photos={shots.map(s => ({
              key: `${s.id ?? 'saved'}-${s.ts}`,
              src: s.url,
              alt: `${card.name}, ${dayTime(s.ts)}`,
              title: dayTime(s.ts),
              menu: s.id != null && (
                <ItemMenu
                  name={card.name}
                  t={{ kind: 'shot', id: s.id, label: `the photo from ${dayTime(s.ts)}` }}
                  onFix={onFix}
                  className={classes.tileMenu}
                />
              ),
            }))}
          />
        </>
      ) : card.shots_pending || card.shots_unavailable ? null : (
        <Empty>
          {`No clear photo of ${card.name} yet. The best shots show up here once a camera has seen their face.`}
        </Empty>
      )}
      <ProcessingImages count={card.shots_pending} unavailable={card.shots_unavailable} />
    </Section>
  );
}

function Videos({ card, onFix }: { card: Card; onFix: Fix }) {
  const list = card.videos.filter(v => v.thumb).slice(0, 4);
  return (
    <Section
      title="Recent video"
      aside={
        list[0] ? (
          // On the day of their newest video, which Review's Show only can pick them on (#392)
          <ReviewLink to={reviewURL({ person: card.name, day: isoDay(new Date(list[0].start * 1000)) })}>
            See all in Review
          </ReviewLink>
        ) : null
      }
    >
      {list.length ? (
        <ul className={classes.videos}>
          {list.map(v => (
            <VideoTile
              key={`${v.cam}-${v.start}`}
              v={v}
              name={card.name}
              menu={
                v.id && (
                  <ItemMenu
                    name={card.name}
                    t={{ kind: 'video', id: v.id, label: `the video from ${dayTime(v.start)}` }}
                    onFix={onFix}
                  />
                )
              }
            />
          ))}
        </ul>
      ) : card.videos_pending || card.videos_unavailable ? null : (
        <Empty>{`No recorded video of ${card.name} yet.`}</Empty>
      )}
      <ProcessingImages kind="video" count={card.videos_pending} unavailable={card.videos_unavailable} />
    </Section>
  );
}

function Voices({ card, onFix }: { card: Card; onFix: Fix }) {
  const list = card.voices.slice(0, 4);
  return (
    <Section
      title="Recent voice"
      aside={
        list[0] ? (
          // On the day of the newest line they said (#392)
          <ReviewLink to={reviewURL({ person: card.name, day: isoDay(new Date(list[0].ts_start * 1000)) })}>
            Hear more in Review
          </ReviewLink>
        ) : null
      }
    >
      {list.length ? (
        <Timeline role="list" bulletSize={18} lineWidth={2} className={classes.saidLine}>
          {list.map(v => (
            <Timeline.Item key={v.line} role="listitem" bullet={<IconMicrophone size={11} aria-hidden />}>
              <PlayLine v={v} />
              <span className={classes.saidText}>
                <span>{v.text}</span>
                <Text span size="xs" c="dimmed" className={classes.saidWhen}>
                  {[dayTime(v.ts_start), v.mic].filter(Boolean).join(', ')}
                </Text>
              </span>
              <ItemMenu
                name={card.name}
                t={{ kind: 'voice', id: v.line, label: `the line “${v.text}”` }}
                onFix={onFix}
                className={classes.lineMenu}
              />
            </Timeline.Item>
          ))}
        </Timeline>
      ) : (
        <Empty>{`Nothing heard from ${card.name} yet.`}</Empty>
      )}
    </Section>
  );
}

function PublicInfo({ name }: { name: string }) {
  const [look, setLook] = useState<(Lookup & { busy?: boolean; error?: string }) | null>(null);
  const lookup = () => {
    setLook({ busy: true });
    lookUp(name).then(
      r => setLook('error' in r && r.error ? { error: refusal(r, LOOKUP_FAILED) } : r),
      e => setLook({ error: e instanceof HttpError && e.reason ? e.reason : LOOKUP_FAILED }),
    );
  };
  if (look?.result != null)
    return (
      <>
        <p className={classes.public}>{look.result}</p>
        <Text size="sm" c="dimmed">
          {`${look.note ?? ''}${look.looked_up ? ` Looked up ${look.looked_up}.` : ''}`}
        </Text>
      </>
    );
  return (
    <>
      <Text size="sm" c="dimmed" mt="xs">
        {`Public information: searches the web for facts about ${name}, with sources. Nothing is sent until you ask.`}
      </Text>
      <Button variant="default" size="xs" mt={6} disabled={look?.busy} onClick={lookup}>
        {look?.busy ? 'Looking up' : `Look up ${name}`}
      </Button>
      {look?.error && (
        <Text size="sm" c="var(--lib-bad)" role="alert">
          {look.error}
        </Text>
      )}
    </>
  );
}

function About({ d }: { d: ProfileData }) {
  const facts = [
    roleLine(d),
    d.number ? `Phone ${d.number}` : '',
    d.aliases?.length ? `Also called ${d.aliases.join(', ')}` : '',
    d.lines
      ? `${plural(d.lines, 'line', 'lines')} heard, ${d.room_lines ?? 0} in the room and ${d.call_lines ?? 0} on calls`
      : '',
  ].filter(Boolean);
  return (
    <Section title="About">
      {facts.length > 0 && (
        <ul className={classes.facts}>
          {facts.map(f => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      )}
      {(d.memory?.length ?? 0) > 0 && (
        <>
          <Title order={4} size="h6" mt="sm">
            Remembered
          </Title>
          <ul className={classes.facts}>
            {d.memory?.map(m => (
              <li key={m}>{m}</li>
            ))}
          </ul>
        </>
      )}
      <PublicInfo name={d.name} />
      {d.note && (
        <Text size="sm" c="dimmed" mt="sm">
          {d.note}
        </Text>
      )}
    </Section>
  );
}

/** Type something for their cloned voice to say. */
function SayForm({ id, name }: { id: number; name: string }) {
  const [text, setText] = useState('');
  const [st, setSt] = useState<Status>(null);
  const say = async (e: FormEvent) => {
    e.preventDefault();
    const words = text.trim();
    if (!words) return;
    setSt({ text: 'Speaking. The first time, building the reference takes 5 to 20 seconds.' });
    try {
      const j = await sayClone(id, words);
      if (!j.ok) return setSt({ err: refusal(j, 'Speaking failed.') });
      setSt(null);
      play(j.url, `say:${Date.now()}`);
    } catch (err) {
      setSt({ err: failure(err, 'Speaking failed.') });
    }
  };
  return (
    <form onSubmit={say}>
      <Group gap="xs" wrap="nowrap" mt="xs">
        <TextInput
          flex={1}
          miw={160}
          aria-label={`Something for ${name}'s voice to say`}
          placeholder={`Something to say in ${name}'s voice`}
          value={text}
          onChange={e => setText(e.currentTarget.value)}
        />
        <Button type="submit" size="sm" disabled={!text.trim()}>
          Say it
        </Button>
      </Group>
      <StatusLine msg={st} />
    </form>
  );
}

function VoiceClone({ name, voices, owner = false }: { name: string; voices: number[]; owner?: boolean }) {
  const client = useQueryClient();
  const q = useQuery({ queryKey: CLONES_KEY, queryFn: getClones });
  const busy = useClones(s => s.busy);
  const [rec, setRec] = useState(false);
  const [picking, setPicking] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const clone = q.data ? cloneOf({ voices }, q.data.clones ?? []) : null;
  const key = clone ? `ref:${clone.id}` : '';
  // Pick clips works for any of their voices, even one with no clear speech yet (its weak matches can be picked).
  // Not for the owner, whose clone is their recorded reference.
  const pickId = owner ? null : (clone?.id ?? voices[0] ?? null);
  const playing = usePlayer(s => !!key && s.key === key);
  const make = async () => {
    if (!clone) return;
    setErr(null);
    const r = await build(client, clone.id);
    if (!r.ok) setErr(r.error);
  };
  // People's Voice clones sidebar could remove a clone; with it gone from People (#2455) the card does.
  const drop = async () => {
    if (!clone?.built) return;
    const sure = await confirm({
      title: 'Remove clone',
      body: `Remove the cloned voice for ${name}? Their lines stay; the clone can be rebuilt later.`,
      confirm: 'Remove',
      danger: true,
    });
    if (!sure) return;
    setErr(null);
    const r = await remove(client, clone.id);
    if (!r.ok) setErr(r.error);
  };
  let body: ReactNode = null;
  if (q.isPending) body = null;
  else if (!clone) body = <Empty>{`Liberational doesn't know ${name}'s voice yet.`}</Empty>;
  else if (clone.built)
    body = (
      <>
        <Button
          variant="default"
          size="xs"
          leftSection={playing ? <StopIcon /> : <PlayIcon />}
          onClick={() => (playing ? stop() : play(cloneAudio(clone.id), key))}
        >
          {playing ? 'Stop' : 'Play the sample'}
        </Button>
        <SayForm id={clone.id} name={name} />
      </>
    );
  else if (clone.ready)
    body = (
      <>
        {!!clone.clean_s && (
          <Text size="sm" c="dimmed" mb={6}>
            {cloneSpeech(clone)}
          </Text>
        )}
        <Button variant="default" size="xs" disabled={busy != null} onClick={() => void make()}>
          {busy === clone.id ? 'Building the voice' : `Clone ${name}'s voice`}
        </Button>
      </>
    );
  else
    body = (
      <Empty>{clone.clean_s ? `${cloneSpeech(clone)} ${cloneShort(clone, name)}` : `${cloneShort(clone, name)}`}</Empty>
    );
  return (
    <Section title="Voice clone">
      {body}
      {err && (
        <Text size="sm" c="var(--lib-bad)" role="alert">
          {err}
        </Text>
      )}
      <Group gap="xs" mt="sm">
        <Button variant="subtle" size="xs" onClick={() => setRec(true)}>
          Record a new reference
        </Button>
        {pickId != null && (
          <Button variant="subtle" size="xs" onClick={() => setPicking(true)}>
            Pick clips
          </Button>
        )}
        {clone?.built && (
          <Button variant="subtle" size="xs" color="red" disabled={busy != null} onClick={() => void drop()}>
            Remove clone
          </Button>
        )}
      </Group>
      <Recorder opened={rec} onClose={() => setRec(false)} name={name} />
      {pickId != null && (
        <PickClips opened={picking} onClose={() => setPicking(false)} id={pickId} name={name} needs={clone?.needs_s} />
      )}
    </Section>
  );
}

function Back({ onBack }: { onBack: () => void }) {
  return (
    <div className={classes.back}>
      <Button variant="subtle" size="xs" onClick={onBack}>
        Back to people
      </Button>
    </div>
  );
}

type Where = Parameters<typeof whereabouts>[0];

/** Where they are now or were last seen: from their list row when there is one (/people refreshes every 30 s, the
 * card doesn't), else the card. */
function whereOf(person: Person | undefined, card: Card | undefined, d: ProfileData): Where {
  if (person) return person;
  if (card)
    return {
      here_now: card.here_now,
      last_seen: card.last_seen?.ts,
      last_seen_cam: card.last_seen?.cam,
      last_seen_place: card.last_seen?.place,
    };
  return d;
}

/** The phrase for where. While identity is still rematching faces, say so instead of a stale time. */
function whereText(w: Where, d: ProfileData): string {
  if (!w.here_now && (d.last_seen_error || d.last_seen_pending)) return seenStatus(d);
  return whereabouts(w);
}

export function Profile(props: ProfileProps) {
  // Everything below starts over for another person.
  return <Body key={props.name} {...props} />;
}

/** What the toast says after a correction. */
function fixedText(c: Correction, name: string, to: string | undefined): string {
  if (c.action === 'move') return `Moved to ${to ?? c.to}`;
  if (c.action === 'remove') return `Removed from ${name}`;
  if (c.action === 'main') return 'Now the main photo';
  return 'Hidden from the card';
}

function Body({ name, person, people, onChanged, onDeleted, onBack, isNew = false, onPickDone }: ProfileProps) {
  const client = useQueryClient();
  // The page shows at once from their row of the people list; GET /people/<name> fills in the rest (#2397).
  const q = useQuery({
    queryKey: profileKey(name),
    queryFn: ({ signal }) => getProfile(name, { signal }),
    placeholderData: person ? profileFromRow(person) : undefined,
  });
  // A card can already spend the full request deadline gathering retained media. Show the first failure and
  // let the person retry, rather than silently keeping the skeleton through another full deadline.
  const c = useQuery({
    queryKey: cardKey(name),
    queryFn: ({ signal }) => getCard(name, { signal }),
    retry: false,
    // Poll while crops are being made, or while the server reads again an older card it answered (#2415).
    refetchInterval: query =>
      query.state.status !== 'error' &&
      (query.state.data?.shots_pending || query.state.data?.videos_pending || query.state.data?.refreshing)
        ? 2000
        : false,
  });
  const cardSlow = useSlow(c.isFetching);
  const [mode, setMode] = useState<Mode>(null);
  const ownerMergeBusy = useRef(false);
  const deleteId = useId();
  const deleteBusy = useRef(false);
  const deleteOpen = useRef(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      modals.close(deleteId);
    };
  }, [deleteId]);
  const [val, setVal] = useState('');
  const [role, setRole] = useState({ role: '', organization: '' });
  const [st, setSt] = useState<Status>(null);
  const [picking, setPicking] = useState(isNew);
  const [target, setTarget] = useState<Target | null>(null); // the item "Not <name>…" was picked on
  const [toast, setToast] = useState<Toast>(null);
  const [undoing, setUndoing] = useState(false);
  const [fixing, setFixing] = useState(false); // one correction at a time, so no Undo is lost
  const opener = useRef<HTMLElement | null>(null); // the ⋯ button Who is this? was opened from

  if (q.isPending)
    return (
      <div className={classes.profile}>
        <Back onBack={onBack} />
        <Title order={2} className={classes.name}>
          {name}
        </Title>
        <SlowLoader active label={`Loading ${name}`} />
        <BlockSkeleton heading={false} lines={6} label={`Loading ${name}`} />
      </div>
    );
  if (q.isError && !q.data)
    return (
      <div className={classes.profile}>
        <Back onBack={onBack} />
        <Title order={2} className={classes.name}>
          {name}
        </Title>
        <LoadError
          message={q.error instanceof HttpError && q.error.reason ? q.error.reason : 'Could not load this profile.'}
          onRetry={() => void q.refetch()}
          busy={q.isFetching}
        />
      </div>
    );

  const d = q.data;
  const card = c.data;
  const cardFailure = `Could not load ${d.name}'s photos, video, and voice.`;
  const cardError =
    c.error instanceof HttpError && c.error.reason
      ? c.error.reason
      : c.error instanceof RequestTimeout
        ? `${cardFailure} ${c.error.message}`
        : cardFailure;
  const voices = d.voices ?? [];
  const others = people.filter(p => p.name !== d.name);
  const roleText = roleLine(d);
  const whereabout = whereOf(person, card, d);
  const where = whereText(whereabout, d);
  const chips = [
    voices.length ? 'Voice known' : '',
    d.faces || card?.photo ? 'Face known' : '',
    d.number ? 'Can be called' : '',
  ].filter(Boolean);

  const open = (m: Exclude<Mode, null>) => {
    setMode(mode === m ? null : m);
    setSt(null);
  };
  /** A person has one number, and a merge keeps the survivor's, so say which number goes before it happens. */
  const lost = (to: string) => {
    const t = people.find(p => p.name.toLowerCase() === to.toLowerCase());
    return t && t.name !== d.name && t.number && d.number && t.number !== d.number
      ? ` ${t.name} keeps the number ${t.number}, and ${d.name}'s number ${d.number} is not kept.`
      : '';
  };
  /** Run a change: show busy, then the reply's refusal or the failure if there is one. The reply when it worked. */
  const run = async <T extends object>(busy: string, f: () => Promise<T>): Promise<T | null> => {
    setSt({ text: busy });
    try {
      const j = await f();
      if ('ok' in j && !j.ok) {
        setSt({ err: refusal(j, DIDNT_WORK) });
        return null;
      }
      setSt(null);
      setMode(null);
      return j;
    } catch (e) {
      setSt({ err: failure(e, DIDNT_WORK) });
      return null;
    }
  };
  const doThisIsMe = async () => {
    if (ownerMergeBusy.current) return;
    ownerMergeBusy.current = true;
    setSt({ text: 'Saving' });
    try {
      let j = await mergeIntoOwner(d.name);
      if (j.confirmation && j.voices) {
        const count = j.voices.length;
        setSt(null);
        if (
          !(await confirm({
            title: `Merge ${d.name} into you?`,
            body: `${d.name} also has ${count} ${count === 1 ? 'voice' : 'voices'}. They will become yours and be recognized as you.${lost(shell().owner)}`,
            confirm: 'This is me',
          }))
        )
          return;
        j = await mergeIntoOwner(d.name, j.voices);
      }
      if (!j.ok) {
        setSt({
          err: j.confirmation
            ? 'Their voices changed. Choose This is me again to confirm the current voices.'
            : refusal(j, DIDNT_WORK),
        });
        return;
      }
      setSt(null);
      setMode(null);
      onChanged(j.name, j);
    } catch (e) {
      setSt({ err: failure(e, DIDNT_WORK) });
    } finally {
      ownerMergeBusy.current = false;
    }
  };
  const doRename = async (to: string, question: string) => {
    if (!to || to === d.name) return;
    if (
      to.toLocaleLowerCase() === shell().owner.toLocaleLowerCase() ||
      people.some(p => p.owner && p.name.toLocaleLowerCase() === to.toLocaleLowerCase())
    )
      return doThisIsMe();
    if (question) {
      // The question is the title and what follows its first "? " the body, as the old window.confirm read.
      const at = question.indexOf('? ');
      const title = at < 0 ? question : question.slice(0, at + 1);
      const body = at < 0 ? undefined : question.slice(at + 2);
      if (!(await confirm({ title, body, confirm: 'Merge' }))) return;
    }
    const j = await run('Saving', () => rename(d.name, to));
    if (j) onChanged(j.name, j);
  };
  const doDescribe = async () => {
    const j = await run('Saving', () => describe(d.name, role.role.trim(), role.organization.trim()));
    if (!j) return;
    client.setQueryData<ProfileData>(profileKey(name), x => x && { ...x, role: j.role, organization: j.organization });
    onChanged(j.name);
  };
  const doPhone = async () => {
    const num = val.trim();
    if (!PHONE.test(num)) return setSt({ err: 'Type the number with its country code, like +14155550123.' });
    if (!(await run('Saving', () => addPerson(d.name, num)))) return;
    client.setQueryData<ProfileData>(profileKey(name), x => x && { ...x, number: num });
    onChanged();
  };
  const doDelete = async () => {
    if (deleteBusy.current || !mounted.current) return;
    deleteBusy.current = true;
    try {
      const j = await run(`Forgetting ${d.name}`, () => deletePerson(d.name));
      if (j && mounted.current)
        onDeleted(j, removedText(d.name, j.removed ?? {}) + (j.notes?.length ? ` ${j.notes.join(' ')}` : ''));
    } finally {
      deleteBusy.current = false;
    }
  };
  const confirmDelete = () => {
    if (deleteOpen.current || deleteBusy.current) return;
    deleteOpen.current = true;
    setMode(null);
    setSt(null);
    void confirm({
      id: deleteId,
      title: `Delete ${d.name}`,
      body: `Delete ${d.name}? This forgets their face and voice. Recordings stay.`,
      confirm: 'Delete',
      danger: true,
    }).then(yes => {
      deleteOpen.current = false;
      if (yes) void doDelete();
    });
  };

  /** A correction from an item's menu; the reply carries the fresh card (the next best shot already in). */
  /** Close Who is this? and give focus back to the ⋯ button it came from, while that is still on the page. */
  const closeWho = () => {
    setTarget(null);
    const el = opener.current;
    opener.current = null;
    if (el) setTimeout(() => el.isConnected && el.focus(), 0);
  };
  const doCorrect = async (c: Correction) => {
    closeWho();
    if (fixing) return;
    setFixing(true);
    setToast({ text: 'Saving', sticky: true });
    try {
      const j = await correct(c);
      if (!j.ok) return setToast({ err: refusal(j, DIDNT_WORK) });
      if (j.card) client.setQueryData(cardKey(name), j.card);
      else void client.invalidateQueries({ queryKey: cardKey(name) });
      setToast({ text: fixedText(c, d.name, j.to), undo: j.undo });
      onChanged(); // the list's photo and whereabouts may have changed, and a move may have made someone new
    } catch (e) {
      setToast({ err: failure(e, DIDNT_WORK) });
    } finally {
      setFixing(false);
    }
  };
  const fix: Fix = (t, action, from) => {
    if (action === 'not') {
      opener.current = from ?? null;
      return setTarget(t);
    }
    void doCorrect({ name: d.name, kind: t.kind, id: t.id, action });
  };
  const doUndo = async (token: string) => {
    setUndoing(true);
    try {
      const j = await undoChange(token);
      if (j.card) client.setQueryData(cardKey(name), j.card);
      else void client.invalidateQueries({ queryKey: cardKey(name) });
      void client.invalidateQueries({ queryKey: profileKey(name) });
      onChanged(undefined, j);
      // Part of it may not have gone back: what did is already applied above, so say what didn't.
      if (!j.ok)
        return setToast({
          err: j.errors?.length ? `Partly undone: ${j.errors.join(' ')}` : refusal(j, 'Undo failed.'),
        });
      setToast({ text: 'Undone' });
    } catch (e) {
      setToast({ err: failure(e, 'Undo failed.') });
    } finally {
      setUndoing(false);
    }
  };
  const endPicking = () => {
    setPicking(false);
    onPickDone?.();
  };

  const renameTo = val.trim();
  const renameError = mode === 'rename' && st && 'err' in st ? st.err : undefined;
  const mergeQuestion = lost(val)
    ? `Merge ${d.name} into ${val}? Their voices, faces, and lines become ${val}'s.${lost(val)}`
    : `Merge ${d.name} into ${val}? Their voices, faces, phone, and lines become ${val}'s.`;
  return (
    <article className={classes.profile} aria-label={`Profile of ${d.name}`}>
      <Back onBack={onBack} />
      <header className={classes.head}>
        <MainPhoto url={card?.photo?.url ?? person?.photo} name={d.name} />
        <div className={classes.headText}>
          <Title order={2} className={classes.name}>
            {d.name}
            {d.owner && (
              <Badge variant="light" size="sm" ml={8} style={{ verticalAlign: 'middle' }}>
                You
              </Badge>
            )}
          </Title>
          {roleText && <Text c="dimmed">{roleText}</Text>}
          {where && (
            <Text size="sm" className={whereabout.here_now ? classes.here : undefined}>
              {where}
            </Text>
          )}
          {chips.length > 0 && (
            <Group gap={6} mt={6}>
              {chips.map(t => (
                <Badge key={t} variant="default" radius="sm" tt="none" fw={500}>
                  {t}
                </Badge>
              ))}
            </Group>
          )}
          {/* Hidden while picking: a rename mid-pick would start the card over and lose the picks. */}
          {!picking && (
            <Group gap="xs" mt={10}>
              <Button
                variant="default"
                size="xs"
                onClick={() => {
                  open('rename');
                  setVal(d.name);
                }}
              >
                Rename
              </Button>
              <Menu position="bottom-start" withinPortal>
                <Menu.Target>
                  <ActionIcon variant="default" size={30} aria-label={`More for ${d.name}`}>
                    <IconDots size={16} />
                  </ActionIcon>
                </Menu.Target>
                <Menu.Dropdown>
                  <Menu.Item
                    onClick={() => {
                      open('describe');
                      setRole({ role: d.role || '', organization: d.organization || '' });
                    }}
                  >
                    Role and organization
                  </Menu.Item>
                  <Menu.Item
                    onClick={() => {
                      open('phone');
                      setVal(d.number || '');
                    }}
                  >
                    Phone number
                  </Menu.Item>
                  <Menu.Item
                    onClick={() => {
                      setMode(null);
                      setToast(null);
                      setPicking(true);
                    }}
                  >
                    Pick what's theirs
                  </Menu.Item>
                  {!d.owner && <Menu.Item onClick={() => void doThisIsMe()}>This is me</Menu.Item>}
                  {!d.owner && others.length > 0 && (
                    <Menu.Item
                      onClick={() => {
                        open('merge');
                        setVal('');
                      }}
                    >
                      Same person as
                    </Menu.Item>
                  )}
                  {!d.owner && (
                    <Menu.Item color="red" onClick={confirmDelete}>
                      Delete
                    </Menu.Item>
                  )}
                </Menu.Dropdown>
              </Menu>
            </Group>
          )}
        </div>
      </header>

      {mode === 'rename' && (
        <form
          onSubmit={e => {
            e.preventDefault();
            void doRename(renameTo, lost(renameTo) && `Merge ${d.name} into ${renameTo}?${lost(renameTo)}`);
          }}
        >
          <Group gap="xs" align="flex-start" className={classes.form}>
            <TextInput
              aria-label="New name"
              value={val}
              error={renameError}
              onChange={e => setVal(e.currentTarget.value)}
            />
            <Button type="submit" size="xs" disabled={!renameTo}>
              Save
            </Button>
            <Button variant="subtle" size="xs" onClick={() => setMode(null)}>
              Cancel
            </Button>
          </Group>
        </form>
      )}
      {mode === 'describe' && (
        <form
          onSubmit={e => {
            e.preventDefault();
            void doDescribe();
          }}
        >
          <Group gap="xs" className={classes.form}>
            <TextInput
              aria-label="Role"
              placeholder="Role, like CEO"
              maxLength={80}
              value={role.role}
              onChange={e => setRole({ ...role, role: e.currentTarget.value })}
            />
            <TextInput
              aria-label="Organization"
              placeholder="Organization, like Acme"
              maxLength={80}
              flex={1}
              miw={160}
              value={role.organization}
              onChange={e => setRole({ ...role, organization: e.currentTarget.value })}
            />
            <Button type="submit" size="xs">
              Save
            </Button>
            <Button variant="subtle" size="xs" onClick={() => setMode(null)}>
              Cancel
            </Button>
          </Group>
        </form>
      )}
      {mode === 'phone' && (
        <form
          onSubmit={e => {
            e.preventDefault();
            void doPhone();
          }}
        >
          <Group gap="xs" className={classes.form}>
            <TextInput
              aria-label="Phone number"
              placeholder="+14155550123"
              value={val}
              onChange={e => setVal(e.currentTarget.value)}
            />
            <Button type="submit" size="xs" disabled={!val.trim()}>
              Save
            </Button>
            <Button variant="subtle" size="xs" onClick={() => setMode(null)}>
              Cancel
            </Button>
          </Group>
        </form>
      )}
      {mode === 'merge' && (
        <form
          onSubmit={e => {
            e.preventDefault();
            void doRename(val, mergeQuestion);
          }}
        >
          <Group gap="xs" className={classes.form}>
            <NativeSelect
              aria-label="Same person as"
              value={val}
              onChange={e => setVal(e.currentTarget.value)}
              data={[
                { value: '', label: `Pick who ${d.name} really is` },
                ...others.map(p => ({ value: p.name, label: p.name })),
              ]}
            />
            <Button type="submit" size="xs" disabled={!val}>
              Merge
            </Button>
            <Button variant="subtle" size="xs" onClick={() => setMode(null)}>
              Cancel
            </Button>
          </Group>
        </form>
      )}
      <StatusLine msg={renameError ? null : st} />
      <References key={d.name} name={d.name} />
      <FloatingToast message={toast} onUndo={t => void doUndo(t)} onClose={() => setToast(null)} busy={undoing} />

      {picking ? (
        <>
          <Text className={classes.pickIntro}>
            {`${isNew ? 'New. ' : ''}Pick the photos, video, and voice that are ${d.name}'s, and the card fills in.`}
          </Text>
          <Pickers
            name={d.name}
            onSkip={endPicking}
            onAssigned={j => {
              const a = j.assigned ?? { faces: 0, voices: 0, videos: 0 };
              const n = a.faces + a.voices + a.videos;
              const missed = j.failed?.length
                ? ` ${plural(j.failed.length, 'item', 'items')} didn't stick: ${j.failed.map(f => f.error).join('; ')}.`
                : '';
              setToast({ text: `Added ${n} to ${d.name}.${missed}`, undo: j.undo });
              endPicking();
              onChanged(undefined, j);
            }}
          />
        </>
      ) : (
        <>
          {c.isError && <LoadError message={cardError} onRetry={() => void c.refetch()} busy={c.isFetching} />}
          {/* One line that's always there (#2485), empty when nothing loads: coming and going above the sections, it
              moved them a line down and back up on every read. */}
          <Text aria-live="polite" size="sm" c="dimmed" className={classes.loadLine}>
            {c.isFetching
              ? cardSlow
                ? `Still loading ${d.name}'s photos, video, and voice. You can keep using this profile.`
                : `Loading ${d.name}'s photos, video, and voice…`
              : ''}
          </Text>
          {card ? (
            <>
              <Photos card={card} onFix={fix} />
              <Videos card={card} onFix={fix} />
              <Voices card={card} onFix={fix} />
            </>
          ) : !c.isError ? (
            // Shaped like the three sections coming (#2485), so About and Voice clone under them stay about where
            // they'll be, instead of starting high and dropping most of a screen when the sections land.
            <Stack gap="lg" aria-busy="true" aria-label={`Loading ${d.name}'s photos, video, and voice`}>
              <Skeleton height={372} radius="md" />
              <Skeleton height={201} radius="md" />
              <Skeleton height={289} radius="md" />
            </Stack>
          ) : null}
        </>
      )}
      {!picking && (
        <>
          <About d={d} />
          <VoiceClone name={d.name} voices={voices} owner={!!d.owner} />
        </>
      )}
      <WhoIsThis
        name={d.name}
        label={target?.label ?? null}
        people={people}
        onClose={closeWho}
        onMove={to => target && void doCorrect({ name: d.name, kind: target.kind, id: target.id, action: 'move', to })}
        onRemove={() => target && void doCorrect({ name: d.name, kind: target.kind, id: target.id, action: 'remove' })}
      />
    </article>
  );
}
