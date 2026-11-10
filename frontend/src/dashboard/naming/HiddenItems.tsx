// "Show ignored and blocked" under a naming inbox: a switch, off by default, that lists what the owner ignored
// (pictures, one by one) and blocked (a face or an animal, and everything very like it), each with a way back. The
// list is read only once the switch is on.
import { Button, Switch, Text } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf } from '../../shared/http';
import { SafeImage } from '../../shared/SafeImage';
import { LoadError } from '../lib/page';
import classes from './hidden.module.css';

/** One ignored or blocked item: its key (what taking it back sends), what it is, and its pictures. */
export interface HiddenItem {
  key: string;
  label: string; // "Face 3", "Cat on the hall camera"
  pictures: string[]; // image URLs; a picture no longer kept shows as a blank tile
}

export interface HiddenList {
  ignored: HiddenItem[];
  ignoredCount: number;
  blocked: HiddenItem[];
  blockedCount: number;
}

function Section({
  title,
  items,
  total,
  action,
  busy,
  onRestore,
}: {
  title: string;
  items: HiddenItem[];
  total: number;
  action: string; // "Un-ignore" or "Unblock"
  busy: string | null;
  onRestore: (item: HiddenItem) => void;
}) {
  if (!items.length) return null;
  return (
    <section aria-label={title}>
      <Text size="sm" fw={600}>
        {title} ({total}){total > items.length ? `, newest ${items.length} shown` : ''}
      </Text>
      <ul className={classes.list}>
        {items.map(item => (
          <li key={item.key} className={classes.item} aria-label={item.label}>
            <div className={classes.pictures}>
              {item.pictures.slice(0, 3).map(src => (
                <SafeImage key={src} className={classes.picture} src={src} alt="" loading="lazy" />
              ))}
            </div>
            <Text size="xs" c="dimmed" className={classes.label}>
              {item.label}
            </Text>
            <Button
              size="compact-xs"
              variant="default"
              disabled={busy === item.key}
              aria-label={`${action} ${item.label}`}
              onClick={() => onRestore(item)}
            >
              {action}
            </Button>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function HiddenItems({
  queryKey,
  load,
  onRestore,
}: {
  queryKey: readonly string[];
  load: (signal: AbortSignal) => Promise<HiddenList>;
  // Take one back; resolves once the server has, and rejects with why it couldn't.
  onRestore: (kind: 'ignored' | 'blocked', item: HiddenItem) => Promise<void>;
}) {
  const client = useQueryClient();
  const [on, setOn] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const hidden = useQuery({ queryKey, queryFn: ({ signal }) => load(signal), enabled: on });
  const restore = (kind: 'ignored' | 'blocked') => async (item: HiddenItem) => {
    setBusy(item.key);
    setError(null);
    try {
      await onRestore(kind, item);
      await client.invalidateQueries({ queryKey });
    } catch (e) {
      setError(`That didn't work: ${messageOf(e)}`);
    } finally {
      setBusy(null);
    }
  };
  const data = hidden.data;
  return (
    <div className={classes.hidden}>
      <Switch size="sm" label="Show ignored and blocked" checked={on} onChange={e => setOn(e.currentTarget.checked)} />
      {on && hidden.isError && (
        <LoadError
          message="Couldn't load what's ignored and blocked"
          detail={messageOf(hidden.error)}
          onRetry={() => void hidden.refetch()}
          busy={hidden.isFetching}
        />
      )}
      {on && data && !data.ignored.length && !data.blocked.length && (
        <Text size="sm" c="dimmed">
          Nothing is ignored or blocked.
        </Text>
      )}
      {on && data && (
        <>
          <Section
            title="Ignored"
            items={data.ignored}
            total={data.ignoredCount}
            action="Un-ignore"
            busy={busy}
            onRestore={item => void restore('ignored')(item)}
          />
          <Section
            title="Blocked"
            items={data.blocked}
            total={data.blockedCount}
            action="Unblock"
            busy={busy}
            onRestore={item => void restore('blocked')(item)}
          />
        </>
      )}
      {error && (
        <Text size="xs" c="var(--lib-bad)" role="alert">
          {error}
        </Text>
      )}
    </div>
  );
}
