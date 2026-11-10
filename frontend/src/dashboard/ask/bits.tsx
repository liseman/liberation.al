// Small pieces the Ask results share: dashboard links, timestamps, status badges, and the "New" tag.
import { Badge } from '@mantine/core';
import type { ReactNode } from 'react';
import { href, linkClick } from '../router';
import classes from './ask.module.css';
import { dateOf, plain, statusLabel } from './format';
import { safeHref, tabPath } from './links';

/**
 * A link from an answer or evidence into the dashboard. The value goes through safeHref; an unsafe one is plain text.
 * A tab of this app (Review, People) opens in place, a hash link just moves the hash, and the rest are normal links.
 */
export function AppLink({ to, children }: { to: unknown; children: ReactNode }) {
  const safe = safeHref(to);
  if (!safe) return <span>{children}</span>;
  if (safe.startsWith('#')) return <a href={safe}>{children}</a>;
  const tab = tabPath(safe);
  return tab ? (
    <a href={href(tab)} onClick={linkClick}>
      {children}
    </a>
  ) : (
    <a href={safe}>{children}</a>
  );
}

/** A server timestamp (epoch seconds or ms, or a date string) in local time. */
export function Stamp({ value }: { value: unknown }) {
  const d = dateOf(value);
  const raw = value === null || value === undefined ? '' : plain(value) || String(value);
  return (
    <time dateTime={d ? d.toISOString() : undefined} title={raw}>
      {d ? d.toLocaleString() : raw || 'Timestamp unavailable'}
    </time>
  );
}

const BAD = new Set(['error', 'timeout', 'synthesis_failed']);

export function StatusBadge({ status }: { status: string }) {
  return (
    <Badge
      variant="outline"
      tt="none"
      radius="xl"
      color={status === 'ok' ? 'var(--lib-ok)' : BAD.has(status) ? 'var(--lib-bad)' : 'gray'}
      data-status={status}
      className={classes.badge}
    >
      {statusLabel(status)}
    </Badge>
  );
}

// A "New" tag on what a new answer path produced, linked to its switch in Settings' New features panel (whose list
// is the "New feature:" keys in config/defaults.toml). Search by meaning has no tag: its hits aren't marked apart.
export const NEW_PATHS = {
  'recall.enabled': "Ask's search loop",
  'journal.enabled': 'the journal',
  'journal.facts': 'facts about people',
  'timeline.enabled': 'who was here, seen and heard',
  'detections.last_seen': 'last seen from live face detections',
  'assistant.forgotten_reply': 'the reply for a person deleted in People',
} as const;

export type NewSetting = keyof typeof NEW_PATHS;

export function NewTag({ setting }: { setting: NewSetting }) {
  const what = `New: ${NEW_PATHS[setting]}. Its switch is in Settings.`;
  return (
    <Badge
      component="a"
      href={`${href('/settings')}#set-${setting}`}
      onClick={linkClick}
      variant="outline"
      tt="none"
      radius="xl"
      className={classes.newTag}
      aria-label={what}
      title={what}
    >
      New
    </Badge>
  );
}
