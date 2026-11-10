// The transcript's small parts: the name box, a turn's name, its actions, a quiet icon button, and the assistant's
// folded log lines.
import { UnstyledButton } from '@mantine/core';
import { IconDeviceTv, IconDeviceTvOff, IconMessageX, IconUserMinus } from '@tabler/icons-react';
import { type ReactNode, useRef, useState } from 'react';
import { Disclosure } from '../lib/Disclosure';
import { hms } from '../lib/format';
import { LazyPersonNameInput } from '../lib/LazyPersonNameInput';
import type { LineRef } from '../live/types';
import { useKnownPeople } from '../livetab/people';
import { WhyName } from '../people/WhyName';
import type { Turn } from './model';
import classes from './transcript.module.css';

/** A turn's name box: a suggestion pick, Enter, or leaving it saves; Escape cancels (onDone gets null). */
export function InlineName({ value, onDone }: { value: string; onDone: (name: string | null) => void }) {
  const known = useKnownPeople();
  const [typed, setTyped] = useState(value);
  const done = useRef(false);
  const finish = (name: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(name);
  };
  return (
    <LazyPersonNameInput
      known={known}
      value={typed}
      onChange={setTyped}
      onPick={name => finish(name)}
      onCancel={() => finish(null)}
      commitOnBlur
      autoFocus
      ariaLabel="Speaker name"
      inputClassName={classes.nameEdit}
      w="14em"
      maxLength={40}
    />
  );
}

/** A turn's name: for a named voice it opens Why this name (#288); for an unnamed one, the name box. */
export function NameButton({
  unnamed,
  name,
  target,
  disabled,
  title,
  label,
  opened,
  onOpen,
  onClose,
  actions,
  children,
}: {
  unnamed: boolean;
  name: string;
  target: LineRef;
  disabled: boolean;
  title: string;
  label: string;
  opened: boolean;
  onOpen: () => void;
  onClose: () => void;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const button = (
    <UnstyledButton
      type="button"
      className={classes.speaker}
      disabled={disabled}
      title={title}
      aria-label={label}
      aria-haspopup={unnamed ? undefined : 'dialog'}
      aria-expanded={unnamed ? undefined : opened}
      onClick={onOpen}
    >
      {children}
    </UnstyledButton>
  );
  if (unnamed) return button;
  return (
    <WhyName target={{ line: target }} name={name} opened={opened} onClose={onClose} actions={actions}>
      {button}
    </WhyName>
  );
}

/** A small quiet icon button, as Mantine's subtle gray ActionIcon looks, as a plain button: transcript rows mount
 *  by the dozen as they scroll into view, and each ActionIcon cost several Mantine components (#2412). */
export function IconButton({
  label,
  title,
  disabled,
  onClick,
  className,
  children,
}: {
  label: string;
  title: string;
  disabled?: boolean;
  onClick: () => void;
  className?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      className={className ? `${classes.iconBtn} ${className}` : classes.iconBtn}
      title={title}
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

/** Quiet, keyboard-reachable actions for a whole turn. Plain titles, not Tooltips: a turn header mounts with every
 *  turn scrolled into view, and three Tooltips each made it cost several times more (#2412). `more` holds actions a
 *  page adds after these. A voice is named from its name, so there's no separate button for that (#2580). */
export function TurnActions({
  name,
  canUnlabel,
  busy,
  onNot,
  onTrash,
  onTv,
  onNotTv,
  more,
  nobodys = false,
}: {
  name: string;
  canUnlabel: boolean;
  busy: boolean;
  onNot: () => void;
  onTrash: () => void;
  onTv?: () => void; // "This is TV" (#2468), on a turn with a line not marked as a TV's yet
  onNotTv?: () => void; // "Not TV", on a turn with a line marked as a TV's
  more?: ReactNode;
  nobodys?: boolean; // every line is a TV's: there is no voice to take it off
}) {
  return (
    <span className={classes.acts}>
      {canUnlabel && !nobodys && (
        <IconButton label={`Not ${name}`} title={`Not ${name}`} disabled={busy} onClick={onNot}>
          <IconUserMinus size={15} />
        </IconButton>
      )}
      <IconButton label="Not transcribed correctly" title="Not transcribed correctly" disabled={busy} onClick={onTrash}>
        <IconMessageX size={15} />
      </IconButton>
      {onNotTv && (
        <IconButton
          label="Not TV"
          title="Not TV: someone here said it, so it counts again"
          disabled={busy}
          onClick={onNotTv}
        >
          <IconDeviceTvOff size={15} />
        </IconButton>
      )}
      {onTv && (
        <IconButton
          label="This is TV"
          title="This is TV: leave it out of people and the journal"
          disabled={busy}
          onClick={onTv}
        >
          <IconDeviceTv size={15} />
        </IconButton>
      )}
      {more}
    </span>
  );
}

/** The assistant's log lines that weren't spoken, folded into one line that expands. */
export function StatusLine({ t, name }: { t: Turn; name: string }) {
  const [open, setOpen] = useState(false);
  const latest = t.lines.at(-1)?.e;
  const more = t.lines.length - 1;
  return (
    <Disclosure
      className={classes.statusLine}
      buttonClassName={classes.statusToggle}
      opened={open}
      onChange={setOpen}
      keepMounted={false}
      withChevron={false}
      summary={
        <>
          <span aria-hidden>{open ? '▾' : '▸'}</span>
          <span className={classes.statusWho}>{name}</span>
          <span className={classes.statusText}>{latest?.text}</span>
          {more > 0 && <span className={classes.statusMore}>{more} more</span>}
        </>
      }
    >
      <ol className={classes.statusLog}>
        {t.lines.map(({ e }) => (
          <li key={e.ts_start}>
            <time>{hms(e.ts_start)}</time> {e.text}
          </li>
        ))}
      </ol>
    </Disclosure>
  );
}
