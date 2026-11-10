// "Who's this?": every face group and voice nobody has named yet, one row each, best guess first. Live
// shows it in its rail (compact), and People in its right pane (wide, with the recent decisions under it). Each row
// has the face's newest picture or the voice's most useful clip to play, how often and when it was seen or heard, the
// best guess with its score ("Probably Kara 86%"), a bar, and a one-line reason.
//
// A face's card has two buttons, by its guess. With one: "<Name>", which names its pictures as that person, and
// "Not <Name>", which records that they aren't (the guess is never offered for them again) and leaves them waiting.
// With none: the name box, "Ignore" (these pictures, from now on), and "Block" (faces very like them are no longer
// shown). A card with several pictures has no buttons until it is selected, which opens every picture, all selected:
// clicking one deselects it, and that leaves it out of the answer. Deselected pictures stay unnamed and wait again as
// a group of their own. In People's pane, "Show ignored and blocked" lists what was hidden, each with a way back.
//
// A voice's card keeps "Yes, <name>", the name box, Ignore, "Multi" (Multiple speakers), which labels the clip's line
// as several people, undoably (/speakers/refine), "Noise", which drops just that clip, and "TV", which marks the
// whole voice as a TV or radio. The guesses and reasons come from the server (vision/face_guess.py,
// people/voice_guess.py); a score is how sure the guess is (likeness, held down by a close runner-up), not a
// probability. Face rows need a saved thumbnail; live faces can still be named directly on the camera stage.
import { Anchor, Button, Group, Skeleton, Stack, Text } from '@mantine/core';
import { IconDeviceTv, IconMessageX, IconUsers } from '@tabler/icons-react';
import { type QueryClient, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useId, useRef, useState } from 'react';
import { messageOf } from '../../shared/http';
import { ProcessingImages } from '../../shared/ProcessingImages';
import { capitalize, plural, seenAt } from '../lib/format';
import { useSlow } from '../lib/loading';
import { PlayClip } from '../lib/PlayToggle';
import { LoadError, PAGE_SLOW_MS } from '../lib/page';
import { play } from '../lib/player';
import { Toast } from '../lib/Toast';
import { type LineRef, oneLine, someLines } from '../live/types';
import { claim, markTvVoice, refreshNames, undoRejected } from '../livetab/api';
import { faceGroup, type SceneFace, unknownFaces, where } from '../livetab/model';
import { useKnownPeople } from '../livetab/people';
import type { LiveScene } from '../livetab/Rail';
import { Bulk, type BulkTarget } from '../naming/Bulk';
import { type HiddenItem, HiddenItems } from '../naming/HiddenItems';
import { NamingInbox, useLeaving } from '../naming/NamingInbox';
import { type CardChoice, NamingRow } from '../naming/NamingRow';
import { toggled } from '../naming/Pictures';
import { assignMany, MULTI, refine, skipItem, undoChange, undoLabels } from '../people/api';
import {
  type Band,
  dropClip,
  dropWaiting,
  HIDDEN_FACES_KEY,
  type HiddenKind,
  hiddenFaces,
  hideShots,
  ignore,
  markNotFace,
  restoreHidden,
  splitShots,
  type Together,
  UNKNOWN_GROUPS_KEY,
  type UnknownGroup,
  type UnknownVoice,
  unknownGroupsQuery,
  unknownVoicesQuery,
  waitingTotal,
} from './api';
import { FacePictures } from './FacePictures';
import { History } from './History';
import { showLabel, snip, WHOS_THIS } from './labels';
import { NameSelect } from './NameSelect';
import './whosthis.css';
import playClasses from './whosthis.module.css';

export { WHOS_THIS };
// The most face groups By person names in one save: the server keeps the newest 50 undo records
// (people/card_prefs.py UNDO_KEEP), and one save's Undo must cover all of it.
export const FACE_BATCH = 50;

const RANK: Record<Band, number> = { probable: 0, maybe: 1, none: 2 };

type Row = ({ kind: 'face' } & UnknownGroup) | ({ kind: 'voice' } & UnknownGroup & UnknownVoice);

/** What a row's suggestion from who was seen or heard with it says, or ''. */
export function togetherText(kind: 'face' | 'voice', t: Together | null | undefined, now = Date.now() / 1000): string {
  if (!t) return '';
  const times = `${t.moments} ${t.moments === 1 ? 'time' : 'times'}, last ${seenAt(t.last, now)}`;
  return kind === 'face'
    ? `${t.name} was heard while this face was in view: ${times}.`
    : `${t.name} was in view while this voice spoke: ${times}.`;
}

/** "Probably Kara 86%", "Maybe Luca 58%", "Maybe Dana, together 3 times", or "No good guess". */
export function guessText(g: Pick<UnknownGroup, 'guess' | 'band' | 'score' | 'together'>): string {
  if ((!g.guess || g.band === 'none') && g.together)
    return `Maybe ${g.together.name}, together ${g.together.moments} ${g.together.moments === 1 ? 'time' : 'times'}`;
  if (!g.guess || g.band === 'none') return 'No good guess';
  return `${g.band === 'probable' ? 'Probably' : 'Maybe'} ${g.guess.name} ${g.score}%`;
}

interface InView extends SceneFace {
  cam: string;
  ts: number;
}

/** A face shot at the picked size: the 80 px thumbnail for the small size, else the saved crop. The server's URLs may
 *  already ask for the thumbnail (`?size=80`), so the parameter is set or removed, never appended. */
export function shotSrc(url: string, small: boolean): string {
  const u = new URL(url, location.origin);
  if (small) u.searchParams.set('size', '80');
  else u.searchParams.delete('size');
  return `${u.pathname}${u.search}`;
}

/** What requests about a clip's line send: its key, or its time for a clip from before keys. */
const ref = (clip: { ts: number; line?: string }): LineRef => clip.line ?? clip.ts;

/** Best guess first: the probable ones by score, then the maybes, then the rest by when they were last seen. */
/** A row's band for sorting: one with no guess but a suggestion from who was with it sorts with the maybes, as
 *  the server sorts it. */
const sortBand = (g: Row): Band => (g.band === 'none' && g.together ? 'maybe' : g.band);
const scoreKey = (g: Row): number => (g.band !== 'none' ? -g.score : 0);
const byGuess = (a: Row, b: Row) =>
  RANK[sortBand(a)] - RANK[sortBand(b)] || scoreKey(a) - scoreKey(b) || (b.last_seen ?? 0) - (a.last_seen ?? 0);

/** What a row's decision says, and the line it labeled when it can be undone. */
type ClaimUndo = { token: string; row: string };
type NotUndo = { name: string; group: number };
type HiddenUndo = { kind: HiddenKind; keys: string[]; row: string }; // Ignore or Block, and the row it took pictures of
type Said = ({ text: string; undo?: LineRef } | { err: string }) & {
  claimUndo?: ClaimUndo;
  notUndo?: NotUndo;
  hiddenUndo?: HiddenUndo;
  tvUndo?: { token: string; row: string }; // "TV or radio": the mark's Undo, and the row it took away
  bulkUndo?: () => Promise<void>; // By person's save, both halves
};

type RowState = { mode: 'idle' | 'saving'; err?: string } | { mode: 'done'; text: string };

function WhoRow({
  g,
  face,
  names,
  onSaid,
  onGone,
  onPending,
  hidden,
  onMissingPicture,
  thumb,
}: {
  g: Row;
  thumb: boolean; // the small size: the newest picture's thumbnail; else the picture itself
  face: InView | undefined;
  names: readonly string[]; // the people Liberational knows, for the name box
  onSaid: (said: Said) => void;
  hidden: boolean;
  onPending: (pending: boolean) => void;
  onGone: () => void; // the row was decided: take it off the list
  onMissingPicture: (url: string) => void;
}) {
  const client = useQueryClient();
  const leaving = useLeaving();
  const [state, setState] = useState<RowState>({ mode: 'idle' });
  const [name, setName] = useState('');
  // The pictures deselected on the open card: left out of its answer. Every other one is selected, loaded or not.
  const [out, setOut] = useState<ReadonlySet<string>>(() => new Set());
  const [counted, setCounted] = useState<number | null>(null); // how many pictures it has, once they are listed
  const submitting = useRef(false);
  const begin = (terminal = false) => {
    if (submitting.current) return false;
    submitting.current = true;
    setState({ mode: 'saving' });
    if (terminal) {
      leaving(`${g.kind}:${g.id}`); // focus moves on first, without scrolling
      onPending(true);
    }
    return true;
  };
  const busy = state.mode === 'saving';
  const clip = g.kind === 'voice' ? g.clips[0] : undefined;
  // who was seen or heard with it, when that's not the guess already
  const suggested =
    g.together && g.together.name.toLocaleLowerCase() !== g.guess?.name.toLocaleLowerCase() ? g.together.name : null;
  const done = (text: string, undo?: LineRef, claimUndo?: ClaimUndo) => {
    setState({ mode: 'done', text });
    onSaid({ text, undo, claimUndo }); // the confirmation floats as the window's toast
    onGone();
  };
  const failed = (err: string, attemptedName = name) => {
    setName(attemptedName); // the Select clears its search on a pick or a blur: the name comes back to retry
    submitting.current = false;
    setState({ mode: 'idle', err });
    onPending(false);
  };
  /** A decision about one clip: the row stays (the voice still waits, with its next clip once the list is read again). */
  const kept = (text: string, undo?: LineRef) => {
    submitting.current = false;
    setState({ mode: 'idle' });
    refreshNames(client);
    onSaid({ text, undo });
  };

  /** The deselected pictures leave the group before its answer, to wait as a group of their own. Why that
   *  failed, or null when the answer can go on. */
  const leaveOut = async (): Promise<string | null> => {
    if (g.kind !== 'face' || !out.size) return null;
    try {
      const r = await splitShots(g.id, [...out]);
      if (!r.ok) return r.error || 'The server refused that.';
      setOut(new Set());
      return null;
    } catch (e) {
      return messageOf(e);
    }
  };
  const tag = async (who: string, editorName = name) => {
    if (!begin(true)) return;
    const left = g.kind === 'face' ? out.size : 0;
    const refused = await leaveOut();
    if (refused) return failed(`Could not leave those pictures out: ${refused}`, editorName);
    const j = await claim(client, g.kind, g.id, who);
    if (j.ok)
      done(
        `${g.label} tagged as ${j.name || who}. Liberational will know them from now on.${
          left
            ? ` The ${plural(left, 'picture', 'pictures')} left out ${left === 1 ? 'waits' : 'wait'} for a name.`
            : ''
        }`,
        undefined,
        typeof j.undo === 'string' ? { token: j.undo, row: `${g.kind}:${g.id}` } : undefined,
      );
    else failed(j.error || 'Tagging failed', editorName);
  };
  const aside = async () => {
    if (!begin(true)) return;
    const j = await ignore(client, g.kind, g.id);
    if (j.ok) {
      refreshNames(client); // People's counts and its recent decisions too, wherever this was answered
      done(`${g.label} ignored. It won't be asked about again.`);
    } else failed(j.error || 'Could not put this aside');
  };
  /** Not the guessed person: the group stays, guessed afresh without them; Undo takes it back. */
  const notThem = async (who: string) => {
    if (g.kind !== 'face' || !begin()) return;
    const refused = await leaveOut();
    if (refused) return failed(`Could not leave those pictures out: ${refused}`);
    const j = await markNotFace(client, who, [g.id]);
    if (!j.ok) return failed(j.error || 'That did not work');
    submitting.current = false;
    setState({ mode: 'idle' });
    refreshNames(client);
    onSaid({ text: `${g.label} is not ${who}. It waits for another name.`, notUndo: { name: who, group: g.id } });
  };
  /** Ignore or Block the selected pictures: they leave the card, which goes with them when none is left. Kept as
   *  taught state on the server; Undo (and "Show ignored and blocked") brings them back. */
  const hide = async (action: 'ignore' | 'block') => {
    if (g.kind !== 'face') return;
    const whole = out.size === 0;
    if (!begin(whole)) return;
    try {
      const r = await hideShots(g.id, action, [...out]);
      if (!r.ok) throw new Error(r.error || 'The server refused that.');
      const n = plural(r.hidden, 'picture', 'pictures');
      const text =
        action === 'block'
          ? `Blocked ${n} of ${g.label}. Faces very like ${r.hidden === 1 ? 'it' : 'them'} won't be shown.`
          : `Ignored ${n} of ${g.label}. ${r.hidden === 1 ? "It won't" : "They won't"} be shown again.`;
      const said = { text, hiddenUndo: { kind: r.kind, keys: r.keys, row: `${g.kind}:${g.id}` } };
      const emptied = whole || r.left === 0;
      // Every count of what waits drops now, not after the list is read again (which the drop's cancel precedes).
      if (emptied) dropWaiting(client, 'face', g.id);
      refreshNames(client);
      void client.invalidateQueries({ queryKey: UNKNOWN_GROUPS_KEY });
      void client.invalidateQueries({ queryKey: HIDDEN_FACES_KEY });
      if (emptied) {
        setState({ mode: 'done', text });
        onSaid(said);
        onGone();
        return;
      }
      setOut(new Set()); // what was left out is what the card has now, all selected
      submitting.current = false;
      setState({ mode: 'idle' });
      onSaid(said);
    } catch (e) {
      failed(`Could not ${action} ${out.size ? 'those pictures' : 'it'}: ${messageOf(e)}`);
    }
  };
  /** Several people spoke in the clip: its line is labeled so, and can be put back with Undo. */
  const several = async () => {
    if (!clip || !begin()) return;
    try {
      const d = await refine({ action: 'multi', ...oneLine(ref(clip)) });
      if (!d.ok) throw new Error((d as { error?: string }).error || 'The server refused that.');
      dropClip(client, g.id, clip); // off the list at once, not after the list is worked out again
      kept(`Labeled “${snip(clip.text)}” as ${showLabel(d.entry?.speaker ?? MULTI)}.`, ref(clip));
    } catch (e) {
      failed(`Saving failed: ${messageOf(e)}`);
    }
  };
  /** A TV or radio: every line of the voice is marked as one, kept in the transcript but nobody's; the row
   *  leaves, and Undo brings it back. */
  const tv = async () => {
    if (g.kind !== 'voice' || !begin(true)) return;
    const j = await markTvVoice(g.id);
    if (!j.ok || typeof j.undo !== 'string') return failed(j.error || 'Could not mark it as TV');
    refreshNames(client);
    const text = `${g.label} marked as a TV or radio. Its lines are left out of people and the journal.`;
    setState({ mode: 'done', text });
    onSaid({ text, tvUndo: { token: j.undo, row: `${g.kind}:${g.id}` } });
    onGone();
  };
  /** Noise, not a voice (a door, a kettle; a TV has its own answer): that one clip leaves the questions for good; the
   *  voice stays. */
  const noise = async () => {
    if (!clip || !begin()) return;
    try {
      const j = await skipItem({ kind: 'line', id: ref(clip) });
      if (!j.ok) throw new Error((j as { error?: string }).error || 'unknown error');
      dropClip(client, g.id, clip);
      kept('Removed that clip from the questions.');
    } catch (e) {
      failed(`Could not remove that clip: ${messageOf(e)}`);
    }
  };
  const when = g.last_seen != null ? seenAt(g.last_seen) : '';
  const seen =
    g.kind === 'voice'
      ? [`Heard ${plural(g.count, 'line', 'lines')}`, when].filter(Boolean).join(', ')
      : g.count
        ? [`Seen ${g.count} ${g.count === 1 ? 'time' : 'times'}`, when].filter(Boolean).join(', ')
        : 'Just seen';
  const spot = face ? where(face.bearing_deg) : '';
  const shots = g.kind === 'face' ? g.shots.slice(0, 6) : [];
  const guess =
    g.guess && g.band !== 'none'
      ? { id: g.guess.id ?? g.guess.name, name: g.guess.name, score: g.score, band: g.band }
      : null;
  const missing = (url: string) => () => onMissingPicture(url);
  // A face's pictures: how many it has, and its two buttons, by whether it has a guess.
  const pictures = g.kind === 'face' ? (counted ?? g.shot_count ?? g.shots.length) : 0;
  const these = Math.max(1, pictures - out.size) === 1 ? 'This picture is' : 'These pictures are';
  const choices: CardChoice[] | undefined =
    g.kind !== 'face'
      ? undefined
      : guess
        ? [
            {
              label: guess.name,
              hint: `${these} ${guess.name}. Liberational learns from them and knows ${guess.name} from now on. Undo takes it back.`,
              shortcut: 'Y',
              onClick: () => void tag(guess.name, guess.name),
            },
            {
              label: `Not ${guess.name}`,
              hint: `${these} not ${guess.name}. ${guess.name} is never the guess for them again, and they wait for another name. Undo takes it back.`,
              shortcut: 'N',
              onClick: () => void notThem(guess.name),
            },
          ]
        : [
            {
              label: 'Ignore',
              hint: 'Ignore the selected pictures from now on. New pictures of this face are still asked about.',
              shortcut: 'I',
              onClick: () => void hide('ignore'),
            },
            {
              label: 'Block',
              hint: 'Stop showing this face: the selected pictures go, and so does any face very like them from now on.',
              shortcut: 'B',
              onClick: () => void hide('block'),
            },
          ];
  return (
    <NamingRow
      rowKey={`${g.kind}:${g.id}`}
      kind={g.kind}
      label={g.label}
      text={guessText(g)}
      guess={guess}
      hidden={hidden}
      busy={busy}
      error={'err' in state ? state.err : undefined}
      done={state.mode === 'done' ? state.text : undefined}
      lead={
        g.kind === 'face' && shots[0] ? { src: shotSrc(shots[0].url, thumb), onMissing: missing(shots[0].url) } : null
      }
      media={
        clip && (
          <PlayClip
            size="lg"
            className={playClasses.play}
            url={clip.url}
            playKey={`who:${ref(clip)}`}
            ts={clip.ts}
            label={`${g.label}: ${clip.text}`}
          />
        )
      }
      quote={clip && <span className="who-quote">“{clip.text}”</span>}
      choices={choices}
      choicesDisabled={pictures > 0 && out.size >= pictures}
      foldable={g.kind === 'face' && pictures > 1}
      count={pictures}
      note={
        g.kind === 'face' && !guess
          ? 'Ignore skips the selected pictures. Block also hides any face very like them from now on.'
          : undefined
      }
      meta={
        <Text size="xs" c="dimmed">
          {seen}
          {face && <span className="who-now"> · in view now{spot ? `, ${spot}` : ''}</span>}
          {pictures > 1 && ` · ${pictures} pictures`}
        </Text>
      }
      reason={
        <>
          <Text size="xs" c="dimmed">
            {g.reason}
          </Text>
          {g.together && (
            <Text size="xs" className="who-together">
              {togetherText(g.kind, g.together)}
            </Text>
          )}
        </>
      }
      onYes={picked => void tag(picked.name, picked.name)}
      onIgnore={g.kind === 'voice' ? () => void aside() : undefined}
      ignoreHint="Set this whole voice aside. It leaves the list and you won't be asked about it again."
      picker={
        (g.kind === 'voice' || !guess) && (
          <NameSelect
            known={names}
            suggestions={[
              ...(g.guess ? [{ name: g.guess.name, label: `${g.guess.name} (guess, ${g.score}%)` }] : []),
              ...(suggested
                ? [{ name: suggested, label: `${suggested} (guess, together ${g.together?.moments} times)` }]
                : []),
            ]}
            typed={name}
            disabled={busy}
            onType={setName}
            onPick={picked => {
              setName(picked);
              void tag(picked, picked);
            }}
            onEscape={() => {
              setName('');
              setState({ mode: 'idle' });
            }}
          />
        )
      }
      answers={
        clip
          ? [
              {
                label: 'Multiple speakers',
                caption: 'Multi',
                hint: 'Several people talk over each other in this clip. The clip is labeled Multiple speakers and leaves the questions; the voice stays.',
                icon: <IconUsers size={16} aria-hidden />,
                shortcut: 'M',
                onClick: () => void several(),
              },
              {
                label: 'Noise: not a voice, drop just this clip',
                caption: 'Noise',
                hint: 'Not a voice at all (a door, a kettle). Drops just this clip from the questions; the voice stays.',
                icon: <IconMessageX size={16} aria-hidden />,
                shortcut: 'N',
                onClick: () => void noise(),
              },
              {
                label: 'TV or radio',
                caption: 'TV',
                hint: "This voice is a TV or radio. All its lines stay in the transcript as nobody's, and the voice leaves the list. Undo brings it back.",
                icon: <IconDeviceTv size={16} aria-hidden />,
                shortcut: 'T',
                onClick: () => void tv(),
              },
            ]
          : []
      }
      onSpace={clip ? () => play(clip.url, `who:${ref(clip)}`, clip.ts) : undefined}
    >
      {expanded =>
        g.kind === 'face' &&
        expanded &&
        pictures > 1 && (
          <FacePictures
            id={g.id}
            label={g.label}
            total={g.shot_count ?? g.shots.length}
            out={out}
            onToggle={shot => setOut(have => toggled(have, shot))}
            onCount={setCounted}
            disabled={busy}
          />
        )
      }
    </NamingRow>
  );
}

export { focusAfter } from '../naming/focus';

import { focusRestored } from '../naming/focus';

/** Undo labels: particular lines, or everything since a time. True when any were put back. */
async function undoing(
  client: QueryClient,
  body: { lines: string[] } | { since: number },
  say: (said: Said) => void,
  onUndone?: () => void,
): Promise<boolean> {
  try {
    const d = await undoLabels(body);
    const undone = d.undone ?? [];
    if (!undone.length) {
      say({ err: d.errors?.join(', ') || d.error || 'Nothing to undo.' });
      return false;
    }
    say({ text: `Undid ${plural(undone.length, 'label', 'labels')}.` });
    onUndone?.();
    refreshNames(client);
    return true;
  } catch (e) {
    say({ err: `Undo failed: ${messageOf(e)}` });
    return false;
  }
}

/** "Who's this?". last and live (Live's scenes) say which faces are in a camera's view now; wide is People's pane,
 *  with the recent decisions under the list. onChanged runs after every decision (People reloads its list). */
/** Row-sized placeholders while the lists load, tall enough that the rows replacing them move nothing on screen. */
function RowPlaceholders({ rows, height }: { rows: number; height: number }) {
  return (
    <Stack gap="sm" role="group" aria-busy="true" aria-label="Loading faces and voices" data-testid="row-placeholders">
      {Array.from({ length: rows }, (_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: identical placeholders with no identity of their own
        <Skeleton key={i} height={height} radius="md" />
      ))}
    </Stack>
  );
}

export function WhosThis({
  last = {},
  live = [],
  wide = false,
  onChanged,
  onBack,
}: {
  last?: Record<string, LiveScene>;
  live?: string[];
  wide?: boolean;
  onChanged?: () => void;
  onBack?: () => void; // People's "Back to people", shown on narrow windows where this pane is the whole page
}) {
  const client = useQueryClient();
  const faces = useQuery(unknownGroupsQuery());
  const voices = useQuery(unknownVoicesQuery());
  const known = useKnownPeople();
  const uid = useId(); // the heading's id
  const [said, setSaid] = useState<Said | null>(null);
  const [undoBusy, setUndoBusy] = useState(false);
  const undoPending = useRef(false);
  // After an Undo, focus goes to the row it brought back (or kept), else the heading (focusRestored), once drawn
  // (tried on every render until the row is drawn, which can wait for its list to be read again; the heading after 3 s)
  const [refocus, setRefocus] = useState<{ row: string | null; at?: number } | null>(null);
  useEffect(() => {
    if (!refocus) return;
    const at = refocus.at ?? Date.now();
    if (focusRestored(refocus.row, `${uid}-h`)) return setRefocus(null);
    if (refocus.at === undefined) setRefocus({ ...refocus, at });
    const t = setTimeout(
      () => {
        focusRestored(refocus.row, `${uid}-h`, true);
        setRefocus(null);
      },
      3000 - (Date.now() - at),
    );
    return () => clearTimeout(t);
  });
  // Rows tagged or ignored here leave the list at once, before the server's list is read again.
  const [gone, setGone] = useState<ReadonlySet<string>>(() => new Set());
  // Retain pending/failed rows across polls, including their draft and local error state.
  const [retained, setRetained] = useState<ReadonlyMap<string, Row>>(() => new Map());
  const [pending, setPending] = useState<ReadonlySet<string>>(() => new Set());
  const [missingPictures, setMissingPictures] = useState<ReadonlySet<string>>(() => new Set());
  useEffect(() => {
    // A later successful list read revalidates saved files, so a transient image error can recover on that refresh.
    if (!faces.dataUpdatedAt) return;
    setMissingPictures(previous => (previous.size ? new Set() : previous));
  }, [faces.dataUpdatedAt]);
  const inView = new Map<number, InView>();
  for (const f of unknownFaces(
    live.flatMap(c => (last[c]?.faces ?? []).map(x => ({ ...x, cam: c, ts: last[c]?.ts ?? 0 }))),
  ))
    inView.set(faceGroup(f.name) as number, f);
  const faceGroups = (faces.data?.groups ?? []).map(g => ({
    ...g,
    shots: g.shots.filter(s => typeof s.url === 'string' && s.url.trim() && !missingPictures.has(s.url)),
  }));
  const rows: Row[] = [
    ...[
      ...faceGroups.filter(g => g.shots.length).map(g => ({ kind: 'face' as const, ...g })),
      ...(voices.data?.groups ?? []).map(g => ({ kind: 'voice' as const, shots: [], ...g })),
    ].sort(byGuess),
  ].filter(g => !gone.has(`${g.kind}:${g.id}`));
  for (const [id, g] of retained) {
    if (!gone.has(id) && !rows.some(row => row.kind === g.kind && row.id === g.id)) rows.push(g);
  }
  const visible = rows.filter(g => !pending.has(`${g.kind}:${g.id}`));
  const say = (s: Said) => {
    setSaid(s);
    onChanged?.();
  };
  const undoSaid = async () => {
    if (!said || undoPending.current) return;
    undoPending.current = true;
    setUndoBusy(true);
    try {
      if (said.bulkUndo) {
        await said.bulkUndo();
        setRefocus({ row: null });
      } else if (said.notUndo) {
        const { name, group } = said.notUndo;
        const result = await markNotFace(client, name, [group], false);
        say(
          result.ok ? { text: `Undid it: ${name} can be the guess again.` } : { err: `Undo failed: ${result.error}` },
        );
        if (result.ok) setRefocus({ row: `face:${group}` });
      } else if (said.hiddenUndo) {
        const { kind, keys, row } = said.hiddenUndo;
        const result = await restoreHidden(kind, keys);
        if (!result.ok) {
          say({ err: `Undo failed: ${result.error || 'try again'}`, hiddenUndo: said.hiddenUndo });
          return;
        }
        setGone(previous => {
          const next = new Set(previous);
          next.delete(row);
          return next;
        });
        refreshNames(client);
        void client.invalidateQueries({ queryKey: UNKNOWN_GROUPS_KEY });
        void client.invalidateQueries({ queryKey: HIDDEN_FACES_KEY });
        say({ text: `Undid it: the pictures are ${kind === 'blocked' ? 'unblocked' : 'back'}.` });
        setRefocus({ row });
      } else if (said.tvUndo) {
        const { token, row } = said.tvUndo;
        const result = await undoRejected(token);
        if (!result.ok) {
          say({ err: `Undo failed: ${result.error || 'try again'}`, tvUndo: said.tvUndo });
          return;
        }
        setGone(previous => {
          const next = new Set(previous);
          next.delete(row);
          return next;
        });
        refreshNames(client);
        say({ text: 'Undid it. The voice is waiting for a name again.' });
        setRefocus({ row });
      } else if (said.claimUndo) {
        const { token, row } = said.claimUndo;
        const result = await undoChange(token);
        if (!result.ok) {
          // The endpoint consumes an undo record even if part of the operation could not be restored.
          refreshNames(client);
          say({ err: `Undo failed: ${result.errors?.join(', ') || result.error || 'The name could not be undone.'}` });
          return;
        }
        setGone(previous => {
          const next = new Set(previous);
          next.delete(row);
          return next;
        });
        refreshNames(client);
        say({ text: 'Undid the name. The group is waiting for a name again.' });
        setRefocus({ row });
      } else if ('undo' in said && said.undo != null) {
        if (await undoing(client, someLines([said.undo]), say, () => setGone(new Set()))) setRefocus({ row: null });
      }
    } catch (error) {
      say({ err: `Undo failed: ${messageOf(error)}`, claimUndo: said.claimUndo });
    } finally {
      undoPending.current = false;
      setUndoBusy(false);
    }
  };
  const facesWaiting = faces.isPending || (!!faces.data?.pending && !faces.isError);
  const voicesWaiting = voices.isPending || (!!voices.data?.pending && !voices.isError);
  const loading = facesWaiting || voicesWaiting;
  const slow = useSlow(loading);
  // Rows wait for both first reads (faces and voices) before they show, under row-sized placeholders: one list drawn
  // first and the other sorted in among it moved every row the owner was reading (layout shift). A read still out at
  // PAGE_SLOW_MS no longer holds the other back, and neither does a list the server says it's still working out
  // (pending, which can take minutes): its rows join when they come.
  const reading = faces.isPending || voices.isPending;
  const graceOver = useSlow(reading, PAGE_SLOW_MS);
  const holding = reading && !graceOver;
  // Only a first load says so: a refetch behind a list (or behind an empty answer) is silent.
  const waiting = [facesWaiting && 'faces', voicesWaiting && 'voices'].filter(Boolean).join(' and ');
  const failed = faces.isError || voices.isError;
  // A link to /#whos-this lands here once the rows are in, and again on each such link while Live stays open (in-app
  // links fire hashchange; a visited tab isn't mounted afresh). Only the first load lands: a list read again
  // after a decision (a cold answer flips loading) mustn't pull the page back up while the owner works down it.
  const landed = useRef(false);
  useEffect(() => {
    if (loading) return;
    const land = () => {
      if (location.hash === `#${WHOS_THIS}`) document.getElementById(WHOS_THIS)?.scrollIntoView?.({ block: 'start' });
    };
    if (!landed.current) land();
    landed.current = true;
    window.addEventListener('hashchange', land);
    return () => window.removeEventListener('hashchange', land);
  }, [loading]);
  const bring = async () => {
    try {
      const j = await skipItem({ kind: 'clear' });
      if (!j.ok) throw new Error((j as { error?: string }).error || 'unknown error');
      refreshNames(client);
      say({ text: 'Everything you ignored is back.' });
    } catch (e) {
      say({ err: `That didn't work: ${messageOf(e)}` });
    }
  };
  /** "Show ignored and blocked": take one back. The pictures return to their group, so the list is read again. */
  const unhide = async (kind: HiddenKind, item: HiddenItem) => {
    const result = await restoreHidden(kind, [item.key]);
    if (!result.ok) throw new Error(result.error || 'The server refused that.');
    setGone(new Set());
    refreshNames(client);
    void client.invalidateQueries({ queryKey: UNKNOWN_GROUPS_KEY });
    const what = item.name ?? item.label;
    say({ text: kind === 'blocked' ? `Unblocked ${what}.` : `${capitalize(what)} is no longer ignored.` });
  };
  // By person: face groups guessed as each person, as tiles. Saving claims the checked ones and records the
  // unchecked ones as not that person, with one Undo for both.
  const byPerson = (thumb: boolean): BulkTarget[] => {
    const found = new Map<string, Row[]>();
    for (const g of rows)
      if (g.kind === 'face' && g.guess && g.band !== 'none' && g.shots.length && !gone.has(`face:${g.id}`))
        found.set(g.guess.name, [...(found.get(g.guess.name) ?? []), g]);
    return [...found]
      .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
      .map(([name, groups]) => ({
        id: name,
        name,
        tiles: [...groups]
          .sort((a, b) => b.score - a.score)
          .map(g => ({
            key: String(g.id),
            src: g.shots[0] ? shotSrc(g.shots[0].url, thumb) : '',
            label: `${g.label}, seen ${g.count} times, match ${g.score}%`,
            caption: `${g.label} · ${g.score}%`,
          })),
      }));
  };
  const saveByPerson = async (name: string, checked: number[], unchecked: number[]) => {
    const rowOf = (id: number) => `face:${id}`;
    setGone(previous => new Set([...previous, ...checked.map(rowOf)]));
    const tokens: string[] = [];
    const named: number[] = [];
    const failures: string[] = [];
    for (const id of checked) {
      const j = await claim(client, 'face', id, name);
      if (j.ok) {
        named.push(id);
        if (typeof j.undo === 'string') tokens.push(j.undo);
      } else failures.push(j.error || `Face ${id} could not be named`);
    }
    // A claim that failed is still waiting: it comes back to the list (and to By person) at once.
    const failed = checked.filter(id => !named.includes(id));
    if (failed.length)
      setGone(previous => {
        const next = new Set(previous);
        for (const id of failed) next.delete(rowOf(id));
        return next;
      });
    const marked = unchecked.length ? await markNotFace(client, name, unchecked) : { ok: true };
    if (!marked.ok) failures.push(marked.error || 'The others could not be marked');
    refreshNames(client);
    const done = [
      named.length ? `Named ${plural(named.length, 'face group', 'face groups')} as ${name}.` : '',
      unchecked.length && marked.ok ? `Marked ${unchecked.length} as not ${name}.` : '',
    ].filter(Boolean);
    // Undo what did happen: the claims newest first, carrying on past one the server no longer keeps, then the marks.
    const bulkUndo = async () => {
      const missed: string[] = [];
      for (const token of [...tokens].reverse()) {
        try {
          const r = await undoChange(token);
          if (!r.ok) missed.push(r.error || 'one name could not be undone');
        } catch (e) {
          missed.push(messageOf(e)); // a 404: the server no longer keeps it; the rest still go back
        }
      }
      if (unchecked.length && marked.ok) await markNotFace(client, name, unchecked, false);
      setGone(previous => {
        const next = new Set(previous);
        for (const id of named) next.delete(rowOf(id));
        return next;
      });
      refreshNames(client);
      say(
        missed.length
          ? { err: `Undid the rest, but ${missed.join('; ')}.` }
          : { text: 'Undid it: those faces are waiting again.' },
      );
    };
    const undoable = named.length > 0 || (unchecked.length > 0 && marked.ok);
    say(
      failures.length
        ? { err: [...done, ...failures].join(' '), ...(undoable ? { bulkUndo } : {}) }
        : { text: done.join(' '), bulkUndo },
    );
  };
  // Yes to all (Known design pass): every row whose guess is a Probably, named in one call with one Undo. They leave
  // the list at once and come back on a failure or Undo.
  const sure = visible.filter(g => g.band === 'probable' && g.guess?.name);
  const yesToAll = async () => {
    const keys = sure.map(g => `${g.kind}:${g.id}`);
    const byName = new Map<string, { name: string; faces: number[]; voices: number[]; videos: string[] }>();
    for (const g of sure) {
      const name = g.guess?.name ?? '';
      const item = byName.get(name) ?? { name, faces: [], voices: [], videos: [] };
      (g.kind === 'face' ? item.faces : item.voices).push(Number(g.id));
      byName.set(name, item);
    }
    const back = (gone: readonly string[]) =>
      setGone(previous => {
        const next = new Set(previous);
        for (const key of gone) next.delete(key);
        return next;
      });
    setGone(previous => new Set([...previous, ...keys]));
    try {
      const j = await assignMany([...byName.values()]);
      const missed = (j.failed ?? []).filter(f => f.kind && f.id != null).map(f => `${f.kind}:${f.id}`);
      back(missed);
      refreshNames(client);
      const named = (j.assigned ?? []).map(a => `${a.faces + a.voices + a.videos} as ${a.name}`);
      const count = keys.length - missed.length;
      const kinds = new Set(sure.filter(g => !missed.includes(`${g.kind}:${g.id}`)).map(g => g.kind));
      const noun =
        kinds.size > 1
          ? plural(count, 'face or voice', 'faces and voices')
          : kinds.has('voice')
            ? plural(count, 'voice', 'voices')
            : plural(count, 'face', 'faces');
      const text = `Named ${noun}${named.length ? `: ${named.join(', ')}` : ''}.`;
      const token = j.undo;
      const bulkUndo = token
        ? async () => {
            const r = await undoChange(token);
            back(keys);
            refreshNames(client);
            say(
              r.ok
                ? { text: 'Undid it: they are waiting for a name again.' }
                : { err: `Undo failed: ${r.errors?.join(', ') || r.error || 'try again'}` },
            );
          }
        : undefined;
      say(missed.length ? { err: `${text} ${missed.length} couldn't be named.`, bulkUndo } : { text, bulkUndo });
    } catch (e) {
      back(keys);
      say({ err: `Couldn't name them: ${messageOf(e)}` });
    }
  };
  // No good guess folds behind its count (Known design pass): there's nothing to say yes to on those, only to type.
  const [showNone, setShowNone] = useState(false);
  const noGuess = visible.filter(g => g.band === 'none').length;
  // Newest first as they come, so "Newest" is that order and the sure sorts break ties by it, as the server does.
  const ordered = [...rows].sort((a, b) => (b.last_seen ?? 0) - (a.last_seen ?? 0));
  // Folded rows stay in the list (and its count), out of sight, as a row held while it saves is.
  const folded = (g: Row) => wide && !showNone && g.band === 'none';
  return (
    <NamingInbox
      prefs="face"
      title="Who's this?"
      total={waitingTotal(faces, voices)}
      counting={waitingTotal(faces, voices) == null && !failed}
      headingId={`${uid}-h`}
      sectionId={wide ? undefined : WHOS_THIS}
      className={wide ? 'whos-this who-wide' : 'rail-sec whos-this'}
      compact={!wide}
      defaultSize="M"
      sortLabel="Sort"
      bulkLabel={wide ? 'By person' : undefined}
      bulk={
        wide
          ? ({ thumb }) => (
              <Bulk
                label="By person"
                which="Which person"
                noun="faces"
                storageKey="whosthis.byPerson"
                targets={byPerson(thumb)}
                max={FACE_BATCH}
                onSave={(target, checked, unchecked) =>
                  void saveByPerson(target.id, checked.map(Number), unchecked.map(Number))
                }
              />
            )
          : undefined
      }
      onBack={wide ? onBack : undefined}
      backLabel="Back to people"
      intro="Faces and voices nobody has named yet, best guess first. Say who each one is, and Liberational learns from every answer."
      toast={
        <Toast
          text={said && ('err' in said ? said.err : said.text)}
          error={!!said && 'err' in said}
          label="Decisions"
          sticky={undoBusy}
          onClose={() => setSaid(null)}
          action={
            said &&
            (said.claimUndo ||
              said.notUndo ||
              said.hiddenUndo ||
              said.tvUndo ||
              said.bulkUndo ||
              ('undo' in said && said.undo != null)) && (
              <Button
                size="compact-xs"
                variant="subtle"
                flex="0 0 auto"
                disabled={undoBusy}
                onClick={() => void undoSaid()}
              >
                Undo
              </Button>
            )
          }
        />
      }
      above={
        <>
          {wide && !holding && sure.length > 1 && (
            <Group className="who-all" gap="sm" mb="sm">
              <Text size="sm">
                <b>{plural(sure.length, 'is', 'are')} probably right.</b>
              </Text>
              <Button size="compact-sm" variant="default" onClick={() => void yesToAll()}>
                Say yes to all {sure.length}
              </Button>
            </Group>
          )}
          {faces.isError && (
            <LoadError
              message="Couldn't load the faces waiting for a name"
              detail={messageOf(faces.error)}
              onRetry={() => void faces.refetch()}
              busy={faces.isFetching}
            />
          )}
          {holding && <RowPlaceholders rows={wide ? 5 : 3} height={wide ? 140 : 72} />}
          {voices.isError && (
            <LoadError
              message="Couldn't load the voices waiting for a name"
              detail={messageOf(voices.error)}
              onRetry={() => void voices.refetch()}
              busy={voices.isFetching}
            />
          )}
        </>
      }
      items={(holding ? [] : ordered).map(g => ({
        key: `${g.kind}:${g.id}`,
        guess: {
          id: g.guess?.name ?? '',
          name: g.guess?.name ?? '',
          band: sortBand(g),
          score: g.band !== 'none' ? g.score : 0,
        },
        render: ({ thumb }) => (
          <WhoRow
            g={g}
            thumb={thumb}
            face={g.kind === 'face' ? inView.get(g.id) : undefined}
            names={known}
            onSaid={say}
            hidden={pending.has(`${g.kind}:${g.id}`) || folded(g)}
            onPending={hiding => {
              const id = `${g.kind}:${g.id}`;
              if (hiding) setRetained(prev => new Map(prev).set(id, g));
              setPending(prev => {
                const next = new Set(prev);
                if (hiding) next.add(id);
                else next.delete(id);
                return next;
              });
            }}
            onGone={() => {
              const id = `${g.kind}:${g.id}`;
              setGone(prev => new Set(prev).add(id));
              setPending(prev => {
                const next = new Set(prev);
                next.delete(id);
                return next;
              });
              setRetained(prev => {
                const next = new Map(prev);
                next.delete(id);
                return next;
              });
            }}
            onMissingPicture={url => setMissingPictures(previous => new Set(previous).add(url))}
          />
        ),
      }))}
      footer={
        <>
          {wide && noGuess > 0 && (
            <Anchor component="button" type="button" size="sm" mt="xs" onClick={() => setShowNone(v => !v)}>
              {showNone ? 'Hide the ones with no good guess' : `${noGuess} more with no good guess`}
            </Anchor>
          )}
          <ProcessingImages count={faces.data?.shots_pending} unavailable={faces.data?.shots_unavailable} />
          {/* One box for loading, saving, and empty, always in the page and a line tall, so moving between them
          doesn't move what's below. */}
          <p
            className="who-state"
            role="status"
            aria-live="polite"
            aria-label={loading ? 'Loading unnamed lists' : undefined}
          >
            {!visible.length && pending.size > 0
              ? `Saving ${pending.size} decision(s)…`
              : loading
                ? `${slow ? 'Still loading' : 'Loading'} ${waiting}…`
                : !rows.length && !failed && !faces.data?.shots_pending && !faces.data?.shots_unavailable
                  ? 'Nobody is waiting for a name.'
                  : null}
          </p>
          {wide && (
            <HiddenItems
              queryKey={HIDDEN_FACES_KEY}
              load={async signal => {
                const found = await hiddenFaces(signal);
                // An ignored picture is named by when it was ignored: "A face" says nothing the picture doesn't.
                return {
                  ignored: found.ignored.map(i => ({
                    key: i.id,
                    label: `Ignored ${seenAt(i.at)}`,
                    name: `the picture ignored ${seenAt(i.at)}`,
                    pictures: [i.url],
                  })),
                  ignoredCount: found.ignored_count,
                  blocked: found.blocked.map(b => ({
                    key: b.id,
                    label: `${b.group == null ? `A face blocked ${seenAt(b.at)}` : `Face ${b.group}`}, ${plural(b.count, 'picture', 'pictures')}`,
                    pictures: b.shots.map(shot => shot.url),
                  })),
                  blockedCount: found.blocked_count,
                };
              }}
              onRestore={unhide}
            />
          )}
          {wide && (
            <History
              onUndo={body => undoing(client, body, say, () => setGone(new Set()))}
              onUnskip={() => void bring()}
            />
          )}
        </>
      }
    />
  );
}
