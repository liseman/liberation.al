// Small pieces the contact card and its pickers share: a section with a heading, a dimmed empty line, a status line,
// the toast's message type, and the ⋯ menu on one photo, video, or line (People's and a pet's card).
import { ActionIcon, Menu, Text, Title } from '@mantine/core';
import { IconDots } from '@tabler/icons-react';
import type { ReactNode, Ref } from 'react';
import { type Message, StatusLine } from '../../lib/message';
import { PlayClip } from '../../lib/PlayToggle';
import { type CardVoice, lineClip } from '../api';
import classes from './profile.module.css';

/** The card's status line and its toast take the shared Message (lib/message.tsx); a toast's Undo carries the
 *  server's undo token. */
export type Status = Message;
export type Toast = Message<string>;
export { StatusLine };

/** One of the card's sections: a heading, an optional link beside it, and its body. */
export function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  const id = `card-${title.toLowerCase().replace(/\W+/g, '-')}`;
  return (
    <section className={classes.section} aria-labelledby={id}>
      <div className={classes.sectionHead}>
        <Title order={3} size="h4" id={id}>
          {title}
        </Title>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <Text size="sm" c="dimmed">
      {children}
    </Text>
  );
}

/** Play one of their lines (the card's Recent voice, and a split's groups). */
export function PlayLine({ v }: { v: CardVoice }) {
  return (
    <PlayClip
      className={classes.play}
      url={lineClip(v.line)}
      playKey={`said:${v.line}`}
      ts={v.ts_start}
      label={`“${v.text}”`}
    />
  );
}

/** One action in an item's ⋯ menu. */
export interface MenuAction {
  label: string;
  onClick: () => void;
  color?: string;
  icon?: ReactNode; // "Not <name>" has the person-minus icon, everywhere
}

/** The ⋯ menu on one photo, video, or line of a card: People's (Not <name>…, Use as main photo, Hide from the card)
 *  and a pet's use the same one. `label` names the item for the button ("More for <item>"); `ref` is the
 *  button, for focus to go back to. */
export function ItemActions({
  label,
  actions,
  className,
  ref,
}: {
  label: string;
  actions: MenuAction[];
  className?: string;
  ref?: Ref<HTMLButtonElement>;
}) {
  return (
    <Menu position="bottom-end" withinPortal>
      <Menu.Target>
        <ActionIcon
          ref={ref}
          variant="default"
          size={24}
          radius="xl"
          className={className}
          aria-label={`More for ${label}`}
        >
          <IconDots size={14} />
        </ActionIcon>
      </Menu.Target>
      <Menu.Dropdown>
        {actions.map(a => (
          <Menu.Item key={a.label} color={a.color} leftSection={a.icon} onClick={a.onClick}>
            {a.label}
          </Menu.Item>
        ))}
      </Menu.Dropdown>
    </Menu>
  );
}
