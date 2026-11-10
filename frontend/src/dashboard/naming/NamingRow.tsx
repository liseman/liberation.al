// One card of the naming inbox (#2366, direction A; laid out to Joe's annotated mockup, #3263): the lead picture (or a
// voice's play button), then the guess ("Maybe Mia 67%") with "Yes, Mia" right beside it, a bar, why, a voice's
// words, and when it was seen or heard. The name box sits at the card's top right, and the kind's answers in one row
// of buttons under it, each an icon over a caption and its key ("Not N", a voice's "Multi M", "Ignore I"), with a
// popover on hover and on keyboard focus that says what it does. With the card focused (not its name box), Y says
// yes, N says not, I ignores, and each answer's own key runs it; `hotkeys` adds keys with no button (a picture's A
// for Show all). The arrows move the selection to the card beside, above, or below on screen, and Space runs the
// kind's own Space answer (a voice's play, a picture's Show all), #3241. A card that leaves the list (Yes, a name,
// Ignore) first moves focus to the next one, without scrolling (#2329). Anything below the card (every picture, to
// pick the wrong ones out) is its children.
import { Button, Kbd, Progress, Text, Tooltip, UnstyledButton } from '@mantine/core';
import { IconUserMinus, IconX } from '@tabler/icons-react';
import type { KeyboardEvent, MouseEvent, ReactNode } from 'react';
import { SafeImage } from '../../shared/SafeImage';
import { type Direction, neighbour } from './focus';
import { type Band, type InboxGuess, useCursor, useLeaving } from './NamingInbox';
import classes from './naming.module.css';

const COLOR: Record<Band, string> = { probable: 'var(--lib-ok)', maybe: 'var(--lib-warn)', none: 'var(--lib-dim)' };
// The guess's words in the same colors, a shade darker so they read at 4.5:1 (axe)
const INK: Record<Band, string> = {
  probable: 'var(--lib-ok-text)',
  maybe: 'var(--lib-warn-text)',
  none: 'var(--mantine-color-dimmed)',
};

/** What a row's guess says: "Probably Mia", "Maybe Mia", or "No guess yet". */
export const guessWords = (guess: InboxGuess | null) =>
  guess && guess.band !== 'none' ? `${guess.band === 'probable' ? 'Probably' : 'Maybe'} ${guess.name}` : 'No guess yet';

/** A picture in a row: its source, and what to do when it can't be shown (a missing file drops it, #2049). */
export interface RowPicture {
  src: string;
  alt?: string;
  onMissing?: () => void;
}

/** One of the card's answer buttons: a voice's Multiple speakers, or a picture's Not this picture. */
export interface CardAnswer {
  label: string; // its accessible name ("Multiple speakers")
  caption: string; // the short word under its icon ("Multi")
  hint: string; // what it does, in its popover
  icon: ReactNode;
  shortcut: string; // its one key, run on the focused card
  onClick: () => void;
}

const ARROWS: Record<string, Direction> = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
};

/** One answer button: its icon over its caption and key, with a popover saying what it does on hover and on focus. */
function AnswerButton({ answer, disabled }: { answer: CardAnswer; disabled: boolean }) {
  const key = answer.shortcut.toUpperCase();
  return (
    <Tooltip
      label={`${answer.caption} (${key}): ${answer.hint}`}
      multiline
      w={260}
      withArrow
      openDelay={150}
      events={{ hover: true, focus: true, touch: false }}
    >
      <UnstyledButton
        className={classes.answer}
        disabled={disabled}
        aria-label={answer.label}
        aria-keyshortcuts={key}
        onClick={answer.onClick}
      >
        <span className={classes.icon}>{answer.icon}</span>
        <span className={classes.caption}>
          {answer.caption}
          <Kbd className={classes.kbd}>{key}</Kbd>
        </span>
      </UnstyledButton>
    </Tooltip>
  );
}

export function NamingRow({
  rowKey,
  label,
  lead,
  media,
  meta,
  guess,
  text: said,
  reason,
  error,
  done,
  picker,
  answers = [],
  pictureAnswers = [],
  hotkeys = {},
  onYes,
  onNot,
  notHint,
  onIgnore,
  ignoreHint,
  busy = false,
  hidden = false,
  kind,
  quote,
  onSpace,
  children,
}: {
  rowKey: string;
  label: string; // what it is ("3 cat sightings"): the row's name starts with it
  lead: RowPicture | null; // the newest picture
  media?: ReactNode; // or something else in the picture's place (a voice's clip)
  text?: string; // what the guess says, when the kind words it itself ("Probably Kara 86%")
  error?: string; // why the last answer failed
  done?: string; // the answer was taken: this replaces the guess and the answers
  busy?: boolean;
  hidden?: boolean; // held while its answer saves (#2329): out of sight, still in the list
  kind?: string; // `data-kind`, for a list of several kinds
  answers?: CardAnswer[]; // the kind's own answers, after Not and before Ignore (a voice's Multi, Noise, and TV)
  pictureAnswers?: CardAnswer[]; // the lead picture's own answers, after Ignore
  hotkeys?: Record<string, () => void>; // keys with no button of their own (a picture's A, Show all)
  quote?: ReactNode; // what a voice said, under its guess (its play button is the picture)
  meta: ReactNode; // when and where it was seen, and any links under the picture
  guess: InboxGuess | null;
  reason?: ReactNode;
  picker: ReactNode;
  onYes?: (guess: InboxGuess) => void;
  onNot?: (guess: InboxGuess) => void;
  notHint?: (name: string) => string; // what Not <name> does, in its popover
  onIgnore: () => void;
  ignoreHint: string; // what Ignore does, in its popover
  onSpace?: () => void; // Space on the focused card: a voice plays or stops its clip, a picture shows them all
  children?: ReactNode;
}) {
  const leaving = useLeaving();
  const { cursor, select } = useCursor();
  const guessed = guess && guess.band !== 'none' ? guess : null;
  const text = said ?? guessWords(guessed);
  const yes =
    guessed && onYes
      ? () => {
          leaving(rowKey);
          onYes(guessed);
        }
      : undefined;
  const not = guessed && onNot ? () => onNot(guessed) : undefined;
  const ignore = () => {
    leaving(rowKey);
    onIgnore();
  };
  const buttons: CardAnswer[] = [
    ...(not && guessed
      ? [
          {
            label: `Not ${guessed.name}`,
            caption: 'Not',
            hint: notHint?.(guessed.name) ?? `This isn't ${guessed.name}.`,
            icon: <IconUserMinus size={16} aria-hidden />,
            shortcut: 'N',
            onClick: not,
          },
        ]
      : []),
    ...answers,
    {
      label: 'Ignore',
      caption: 'Ignore',
      hint: ignoreHint,
      icon: <IconX size={16} aria-hidden />,
      shortcut: 'I',
      onClick: ignore,
    },
    ...pictureAnswers,
  ];
  const shortcuts = new Map<string, () => void>(Object.entries(hotkeys).map(([k, run]) => [k.toLowerCase(), run]));
  if (yes) shortcuts.set('y', yes);
  for (const b of buttons) shortcuts.set(b.shortcut.toLowerCase(), b.onClick);
  const keys = (e: KeyboardEvent<HTMLLIElement>) => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.nativeEvent.isComposing) return;
    const target = e.target as HTMLElement;
    if (target.closest('input, textarea, select, [contenteditable], [role="combobox"], [role="menu"]')) return;
    // The arrows and Space act on the card itself, not from the pictures under it (Show all's grid has its own).
    const own = target === e.currentTarget || !!target.closest(`.${classes.pic}, .${classes.info}, .${classes.side}`);
    const dir = ARROWS[e.key];
    if (dir) {
      if (!own || e.shiftKey) return;
      const list = e.currentTarget.parentElement;
      const cards = [...(list?.querySelectorAll<HTMLElement>(':scope > li[data-row]') ?? [])];
      const next = neighbour(cards, e.currentTarget, dir);
      if (!next) return; // at an edge the page scrolls as usual
      e.preventDefault();
      next.focus({ preventScroll: true });
      next.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      return;
    }
    if (e.key === ' ') {
      // A focused button takes Space as its own click.
      if (!onSpace || !own || target.closest('button, a, [role="button"]')) return;
      e.preventDefault();
      onSpace();
      return;
    }
    const run = shortcuts.get(e.key.toLowerCase());
    if (!run) return;
    e.preventDefault();
    run();
  };
  // A click off the answers (the picture, the guess, the words) selects the row and gives it focus, so its keys work
  // at once; the answers, the name box and its list of names, and the pictures below keep their own clicks. React
  // bubbles a click in a portal (a menu or a tooltip, outside the row on the page) up to here too: that isn't the row's.
  const click = (e: MouseEvent<HTMLLIElement>) => {
    const target = e.target as HTMLElement;
    if (!e.currentTarget.contains(target)) return;
    const own = 'button, a, input, textarea, select, label, [role="combobox"], [role="listbox"], [role="menu"]';
    if (target.closest(own)) return;
    e.currentTarget.focus({ preventScroll: true });
  };
  return (
    <li
      style={hidden ? { display: 'none' } : undefined}
      className={`who-row who-group ${classes.row}`}
      data-row={rowKey}
      data-kind={kind}
      data-media={!lead && media ? '' : undefined}
      tabIndex={-1}
      aria-label={`${label}: ${text}`}
      aria-keyshortcuts={`ArrowUp ArrowDown ArrowLeft ArrowRight${onSpace ? ' Space' : ''}`}
      aria-current={cursor === rowKey ? 'true' : undefined}
      onKeyDown={keys}
      onClick={click}
      onFocus={() => select(rowKey)}
    >
      <div className={classes.card}>
        <div className={`who-what ${classes.pic}`}>
          {lead ? (
            <div className={classes.lead}>
              <SafeImage
                className={`who-shot ${classes.photo}`}
                src={lead.src}
                alt={lead.alt ?? ''}
                loading="lazy"
                retries={2}
                fallback={lead.onMissing ? null : undefined}
                onError={lead.onMissing}
              />
            </div>
          ) : (
            media
          )}
        </div>
        {done ? (
          <Text size="sm" className={`who-body ${classes.info}`}>
            {done}
          </Text>
        ) : (
          <>
            <div className={`who-body ${classes.info}`}>
              <div className={classes.guessLine}>
                <Text
                  size="sm"
                  fw={600}
                  className="who-guess"
                  style={{ color: INK[guessed?.band ?? 'none'] }}
                  data-band={guessed?.band ?? 'none'}
                >
                  {text}
                </Text>
                {yes && guessed && (
                  <Button
                    // A Probably's Yes is a Default button and a Maybe's a Subtle one, so it asks for a look (#2361).
                    // Neither is filled: a list of rows would otherwise fill the page with Primaries (#2594).
                    variant={guessed.band === 'probable' ? 'default' : 'subtle'}
                    size="compact-sm"
                    aria-keyshortcuts="Y"
                    disabled={busy}
                    onClick={yes}
                  >
                    Yes, {guessed.name}
                  </Button>
                )}
              </div>
              {guessed && (
                <Progress
                  value={guessed.score}
                  size="sm"
                  color={COLOR[guessed.band]}
                  aria-label={`Match score ${guessed.score}%`}
                  className="who-bar"
                />
              )}
              {reason}
              {quote && (
                <Text size="sm" className={classes.quote}>
                  {quote}
                </Text>
              )}
              {meta}
              {error && (
                <Text size="xs" c="var(--lib-bad)" role="alert">
                  {error}
                </Text>
              )}
            </div>
            <div className={classes.side}>
              <div className={classes.name}>{picker}</div>
              <div className={classes.answers}>
                {buttons.map(b => (
                  <AnswerButton key={`${b.shortcut}-${b.label}`} answer={b} disabled={busy} />
                ))}
              </div>
            </div>
          </>
        )}
        {children}
      </div>
    </li>
  );
}
