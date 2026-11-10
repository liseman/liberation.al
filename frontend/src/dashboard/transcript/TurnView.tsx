// One turn of a transcript, on Live and Review alike: the speaker's avatar and name, the time, the turn's actions,
// then its lines. A voice is named from here (an unmatched voice's name, or Rename in Why this name); a name that matches
// nobody asks "Add <name> as a new person?" first, and Add creates them with just the name. A named
// speaker's name opens Why this name. A turn put down to the wrong person opens Who is this?. A voice no
// person was matched to shows as Unmatched with "?" for its avatar, never a placeholder like "Speaker 261".
import { Button, Mark, Text } from '@mantine/core';
import { IconUserMinus, IconX } from '@tabler/icons-react';
import { useQueryClient } from '@tanstack/react-query';
import { type CSSProperties, memo, type ReactNode, useEffect, useState } from 'react';
import { messageOf } from '../../shared/http';
import { confirm } from '../lib/confirm';
import { hm, hms } from '../lib/format';
import { lineKey as playerKey, usePlayer } from '../lib/player';
import { isNamed, speakerName, spkColor, UNMATCHED } from '../lib/speakers';
import type { Speakers, TranscriptEntry } from '../live/types';
import { useNameVoice } from '../livetab/naming';
import { useKnownPeople } from '../livetab/people';
import { addPerson, PEOPLE_KEY } from '../people/api';
import { shell } from '../shell';
import type { Not } from './corrections';
import { initials, type LineHit, lineKey, named, type Range, sameTurn, type Turn, turnKey } from './model';
import { IconButton, InlineName, NameButton, StatusLine, TurnActions } from './parts';
import classes from './transcript.module.css';

type Mode = null | 'edit' | 'saving' | 'tagging';

const iso = (ts: number) => new Date(ts * 1000).toISOString();

export interface TurnHandlers {
  /** A line's words were clicked: play it (Live) or seek there (Review). */
  onLine: (e: TranscriptEntry, i: number) => void;
  onNot: (n: Not, from: HTMLElement | null) => void;
  onTrash: (lines: readonly TranscriptEntry[]) => void;
  /** "This is TV": the turn's lines not marked as a TV's yet. */
  onTv: (lines: readonly TranscriptEntry[]) => void;
  /** "Not TV": take the TV marks off these lines. */
  onNotTv: (lines: readonly TranscriptEntry[]) => void;
  /** A voice was named or tagged: read the lines again. */
  onNamed: (tagged: boolean) => void;
  /** Naming or asking: keep this turn mounted though scrolled away. */
  onKeep: (key: string, on: boolean) => void;
}

/** Words with a search's matches marked; just the words with no search. The ranges are where Review's search matched
 *  (model.ts lineHits), so a match it found by ignoring accents or a typo is marked too. */
const marked = (text: string, ranges: readonly Range[] | undefined): ReactNode => {
  if (!ranges?.length) return text;
  const out: ReactNode[] = [];
  let at = 0;
  for (const [a, b] of [...ranges].sort((x, y) => x[0] - y[0])) {
    if (a < at || b > text.length) continue;
    if (a > at) out.push(text.slice(at, a));
    out.push(<Mark key={a}>{text.slice(a, b)}</Mark>);
    at = b;
  }
  out.push(text.slice(at));
  return out;
};

/** A TV line's tag: the audio tagger's mark says it was by ear. */
function TvTag({ e, onNot, disabled }: { e: TranscriptEntry; onNot?: () => void; disabled?: boolean }) {
  const ear = e.media_by === 'classifier';
  const said = ear ? 'Marked as TV by ear' : 'Marked as TV';
  if (!onNot)
    return (
      <span className={classes.tvTag} title={said}>
        {ear ? 'TV by ear' : 'TV'}
      </span>
    );
  // Not TV is the tag itself, at the start of the line, away from Not <name> at its end.
  return (
    <button
      type="button"
      className={classes.tvTag}
      data-action
      title={`${said}. Click if it isn't`}
      aria-label={`Not TV: the line at ${hms(e.ts_start)}`}
      disabled={disabled}
      onClick={onNot}
    >
      {ear ? 'TV by ear' : 'TV'}
      <IconX size={10} aria-hidden />
    </button>
  );
}

/** The key of the line playing in this turn, so a play elsewhere doesn't re-render it. */
function usePlayingIn(t: Turn): string | null {
  return usePlayer(s => (s.key != null && t.lines.some(({ e }) => playerKey(e) === s.key) ? s.key : null));
}

// Memoized, with stable handlers from the list: scrolling re-renders the list, never the turns already mounted, and a
// new line re-renders only the turn it joins: turns are rebuilt from the lines, so they compare by their lines.
export const TurnView = memo(
  function TurnView({
    t,
    speakers,
    saving,
    other,
    missed,
    hits,
    more,
    h,
  }: {
    t: Turn;
    speakers: Speakers;
    saving: boolean;
    other?: boolean; // someone else's, shown dimmed beside the person Review is for
    missed?: boolean; // nothing in it matches the search: dimmed, but kept in place
    hits?: ReadonlyMap<string, LineHit>; // a search's matches in this turn's lines, by line key
    more?: ReactNode; // actions a page adds to every turn
    h: TurnHandlers;
  }) {
    const client = useQueryClient();
    const nameVoice = useNameVoice();
    const known = useKnownPeople();
    const playing = usePlayingIn(t);
    const [mode, setMode] = useState<Mode>(null);
    const [adding, setAdding] = useState<string | null>(null); // a name that matches nobody, asking to add them
    const [draft, setDraft] = useState<{ text: string; n: number } | null>(null); // the box again after Cancel
    const [status, setStatus] = useState<{ text: string; err?: boolean } | null>(null);
    const [why, setWhy] = useState(false);
    const [tagLines, setTagLines] = useState<readonly (string | number)[]>([]); // lines being tagged, until confirmed
    const [taggingAs, setTaggingAs] = useState<string | null>(null); // the name they're being tagged with, shown at once
    const first = t.lines[0]?.e;
    const bot = t.id === 'assistant' || t.id === 'status';
    const name = bot ? shell().assistant : first ? speakerName(speakers, first) : '';
    const unnamed = !bot && !isNamed(name);
    // What the turn shows: the name it's being tagged with, until its lines come back with it, then theirs.
    const shown = taggingAs ?? name;
    const unmatched = !bot && !isNamed(shown);
    // A turn that loses its name closes Why this name, so it can't open by itself once a later read names it again.
    useEffect(() => {
      if (unnamed) setWhy(false);
    }, [unnamed]);
    // A tag reaches the lines a moment later: the turn stays "Tagging as" until the looked-at lines arrive confirmed. A
    // new line appended to the turn meanwhile wasn't part of that request.
    useEffect(() => {
      if (!tagLines.length) return;
      const tagged = t.lines.filter(({ e }) => tagLines.includes(e.line ?? e.ts_start));
      if (tagged.length && tagged.every(({ e }) => e.tagged)) {
        setTagLines([]);
        setTaggingAs(null);
        setStatus(s => (s && !s.err ? null : s));
      }
    }, [t.lines, tagLines]);
    const active = mode != null || adding != null || why;
    // biome-ignore lint/correctness/useExhaustiveDependencies: reports a change of active, not of the callback
    useEffect(() => {
      h.onKeep(turnKey(t), active);
    }, [active]);
    if (!first) return null;
    if (t.id === 'status') return <StatusLine t={t} name={name} />;

    const style = { '--who': spkColor(bot ? null : t.id) } as CSSProperties;
    const time = (
      <time dateTime={iso(first.ts_start)} title={hms(first.ts_start)}>
        {hm(first.ts_start)}
      </time>
    );
    if (t.id === 'assistant')
      return (
        <article className={classes.turn} data-bot data-missed={missed || undefined} style={style} data-transcript-turn>
          <span className={classes.avatar} aria-hidden>
            {initials(name)}
          </span>
          <div className={classes.body}>
            <header className={classes.head}>
              <span className={classes.speaker} title={`${name} said`}>
                {name}
              </span>
              {time}
            </header>
            {t.lines.map(({ e }) => (
              <div className={classes.lineRow} key={`a${e.ts_start}`}>
                <p
                  className={classes.line}
                  data-said
                  data-new={e._new || undefined}
                  data-media={e.media || undefined}
                  title={e.route ? `Route: ${e.route}${e.error ? `, error: ${e.error}` : ''}` : undefined}
                >
                  <span className={classes.lineText}>
                    {e.media && <TvTag e={e} />}
                    {marked(String(e.text ?? ''), hits?.get(lineKey(e))?.text)}
                  </span>
                </p>
              </div>
            ))}
          </div>
        </article>
      );

    const who = unmatched ? UNMATCHED : shown;
    const mine = name.toLowerCase() === shell().owner.toLowerCase(); // the owner's turns sit on the right as balloons
    const rename = async (typed: string | null, added = false) => {
      setStatus(null);
      // An exact match (any case) is that person, spelled as they are. Naming a voice nobody has named with a name
      // that matches nobody asks to add them first; renaming a named voice relabels the voice.
      const match = typed ? known.find(k => k.toLowerCase() === typed.toLowerCase()) : undefined;
      if (typed && unnamed && !added && !match && typed !== name) {
        setAdding(typed);
        // The shared confirm: Add is focused (the new person is easily removed); Cancel (or closing) gives the name box back with what was typed.
        void confirm({
          title: 'New person',
          body: `Add ${typed} as a new person?`,
          confirm: 'Add',
          focusConfirm: true,
        }).then(yes => (yes ? void addThem(typed) : keepTyping(typed)));
        return;
      }
      setDraft(null);
      const n = match ?? typed;
      const r = await nameVoice(
        {
          voice: t.id as number | string | null,
          current: name,
          lines: t.lines.map(({ e }) => e.line ?? e.ts_start),
          added,
        },
        n,
        (how, as) => {
          setMode(how);
          if (how === 'tagging') {
            setStatus({ text: `Tagging as ${as}` });
            setTagLines(t.lines.map(({ e }) => e.line ?? e.ts_start));
            setTaggingAs(as);
          }
        },
      );
      setMode(null);
      if ('error' in r) {
        setTaggingAs(null);
        return setStatus({ text: r.error, err: true });
      }
      if (r.done === 'tagged') setStatus({ text: `Tagging as ${r.as}` });
      if (r.done !== 'nothing') h.onNamed(r.done === 'tagged');
    };
    const addThem = async (typed: string) => {
      setAdding(null);
      if (!typed) return;
      setMode('saving');
      let as: string;
      try {
        as = (await addPerson(typed, '')).name;
      } catch (e) {
        setMode(null);
        return setStatus({ text: `Couldn't add ${typed}: ${messageOf(e)}`, err: true });
      }
      void client.invalidateQueries({ queryKey: PEOPLE_KEY });
      await rename(as, true);
    };
    const keepTyping = (typed: string) => {
      setAdding(null);
      setDraft(d => ({ text: typed, n: (d?.n ?? 0) + 1 })); // a fresh box, holding what was typed
    };
    const busy = saving || mode === 'saving' || mode === 'tagging';
    const canCorrect = (t.id != null || !unnamed) && t.lines.every(({ e }) => !!e.line);
    const lines = () => t.lines.map(({ e }) => e);
    // Every line a TV's: nobody said them, so there's no voice to name, take them off, or say why.
    const allTv = t.lines.every(({ e }) => e.media);
    const tvLines = () => lines().filter(e => e.media && e.line);
    // Not <name>, for the turn or one line, matched or not: Who is this? moves the lines to someone or just
    // takes them off the voice, with one Undo, and retries what failed. An unmatched voice goes by its label there.
    const notLines = (ls: readonly TranscriptEntry[], shown: string) => {
      setWhy(false);
      h.onNot(
        { name: shown, lines: ls.map(e => ({ key: e.line ?? '', name: speakerName(speakers, e) })) },
        document.activeElement instanceof HTMLElement ? document.activeElement : null,
      );
    };
    const notThisTurn = () => notLines(lines(), unnamed ? 'the unmatched voice' : name);
    const edit = () => {
      setWhy(false);
      setMode('edit');
    };
    return (
      <article
        className={classes.turn}
        data-transcript-turn
        data-other={other || undefined}
        data-missed={missed || undefined}
        data-unmatched={unmatched || undefined}
        data-mine={mine || undefined}
        data-tagging={mode === 'tagging' || tagLines.length > 0 || undefined}
        style={style}
      >
        <span className={classes.avatar} aria-hidden>
          {initials(shown)}
        </span>
        <div className={classes.body}>
          <header className={classes.head}>
            {mode === 'edit' ? (
              <InlineName
                key={draft?.n ?? 0}
                value={draft?.text ?? (unnamed ? '' : name)}
                onDone={n => void rename(n)}
              />
            ) : allTv ? (
              <span className={classes.speaker} title="Heard from a TV or radio">
                TV
              </span>
            ) : (
              <NameButton
                unnamed={unnamed}
                name={name}
                target={first.line ?? first.ts_start} // the line the turn's name comes from
                disabled={busy}
                title={unnamed ? 'No person matched this voice. Click to say who this is' : `${name}. Click to see why`}
                label={unnamed ? `${who}: say who this is` : `Why ${name}`}
                opened={why && !unnamed}
                onOpen={() => (unnamed ? setMode('edit') : setWhy(true))}
                onClose={() => setWhy(false)}
                actions={
                  <>
                    {canCorrect && (
                      <Button
                        size="compact-xs"
                        variant="subtle"
                        leftSection={<IconUserMinus size={14} aria-hidden />}
                        onClick={notThisTurn}
                      >
                        {`Not ${name}`}
                      </Button>
                    )}
                    <Button size="compact-xs" variant="subtle" onClick={edit}>
                      {t.id != null ? 'Rename this voice…' : 'Say who this is…'}
                    </Button>
                  </>
                }
              >
                {busy ? '...' : marked(who, hits && [...hits.values()].find(hit => hit.name.length)?.name)}
              </NameButton>
            )}
            {time}
            <TurnActions
              name={unnamed ? 'this voice' : name}
              canUnlabel={canCorrect}
              busy={busy}
              onNot={notThisTurn}
              onTrash={() => h.onTrash(lines())}
              onTv={allTv ? undefined : () => h.onTv(lines().filter(e => !e.media))}
              onNotTv={tvLines().length ? () => h.onNotTv(tvLines()) : undefined}
              nobodys={allTv}
              more={more}
            />
            {status && (
              <Text
                span
                className={classes.status}
                c={status.err ? 'var(--lib-bad)' : 'dimmed'}
                role={status.err ? 'alert' : 'status'}
              >
                {status.text}
              </Text>
            )}
          </header>
          {t.lines.map(({ e, i }) => {
            // A label removal is about the voice currently displayed on this exact line.
            const said = speakerName(speakers, e);
            return (
              <div key={e.line ?? e.ts_start} className={classes.lineRow}>
                {e.media && e.line && <TvTag e={e} onNot={() => h.onNotTv([e])} disabled={busy} />}
                <button
                  type="button"
                  className={classes.line}
                  data-i={i}
                  data-key={e.line ?? e.ts_start}
                  data-review-keys // Review's shortcuts still work from a focused line
                  data-new={e._new || undefined}
                  data-media={e.media || undefined}
                  data-playing={playing === playerKey(e) || undefined}
                  onClick={() => h.onLine(e, i)}
                >
                  <span className={classes.lineText}>
                    {e.media && !e.line && <TvTag e={e} />}
                    {marked(String(e.text ?? ''), hits?.get(lineKey(e))?.text)}
                  </span>
                  <time className={classes.lineTime} dateTime={iso(e.ts_start)}>
                    {hms(e.ts_start)}
                  </time>
                </button>
                {e.line && !e.media && (t.id != null || named(said)) && (
                  <IconButton
                    className={classes.lineUnlabel}
                    title={`Not ${named(said) ? said : 'this voice'}`}
                    label={`Not ${named(said) ? said : 'this voice'} in the line at ${hms(e.ts_start)}`}
                    disabled={busy}
                    onClick={() => notLines([e], named(said) ? said : 'the unmatched voice')}
                  >
                    <IconUserMinus size={13} />
                  </IconButton>
                )}
              </div>
            );
          })}
        </div>
      </article>
    );
  },
  (a, b) =>
    sameTurn(a.t, b.t) &&
    a.speakers === b.speakers &&
    a.saving === b.saving &&
    a.other === b.other &&
    a.missed === b.missed &&
    a.hits === b.hits &&
    a.more === b.more &&
    a.h === b.h,
);
