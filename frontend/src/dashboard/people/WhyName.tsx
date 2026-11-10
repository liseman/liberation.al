// "Why this name" (#288): a popover that says what decided a line's or a face's name (the owner's word, an
// introduction, a voice's name, a match), when, and how closely it matched; asks whether the name is right (#224's
// marks, which change nothing); and holds the fix ("Not Dana…", and Live's "Rename this voice…") beside it.
import { Button, Group, Loader, Popover, Stack, Text } from '@mantine/core';
import { useMutation, useQuery } from '@tanstack/react-query';
import { type ReactElement, type ReactNode, useId } from 'react';
import { messageOf } from '../../shared/http';
import { PageError } from '../lib/page';
import { getWhy, markName, type Why, type WhyTarget, whyKey } from './api';

/** When an assertion was made: "27 Oct, 14:05". */
const when = (ts: number) =>
  new Date(ts * 1000).toLocaleString(undefined, {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });

/** What decided the name, in a sentence, and how closely it matched, or null when there was no match score. */
export function explain(w: Why): { said: string; match: string | null } {
  const n = w.name;
  const at = w.asserted_at ? ` on ${when(w.asserted_at)}` : '';
  const thing = w.subject === 'shot' ? 'this photo' : 'this line';
  let said: string;
  if (w.via === 'line' || w.via === 'shot') {
    said =
      w.basis === 'correction'
        ? `You corrected ${thing} to ${n}${at}.`
        : w.basis === 'owner_tag'
          ? `You named ${thing} ${n}${at}.`
          : w.basis === 'introduction'
            ? `${n} was named from an introduction${at}.`
            : w.basis === 'name_badge'
              ? `${n} was named from a name badge${at}.`
              : `${thing[0]?.toUpperCase()}${thing.slice(1)} was matched to ${n}${at}.`;
  } else if (w.via === 'voice') {
    said =
      w.basis === 'owner_tag'
        ? `Its voice was named ${n}${at}.`
        : w.basis === 'correction'
          ? `Its voice was corrected to ${n}${at}.`
          : w.basis === 'introduction'
            ? `Its voice was named ${n} from an introduction${at}.`
            : w.basis === 'cluster_merge'
              ? `Its voice was merged into ${n}'s${at}.`
              : `Its voice was matched to ${n}'s${at}.`;
  } else if (w.via === 'face') said = `The face was recognized as ${n}'s when it was seen.`;
  else if (w.via === 'detached') said = `This line was taken off its voice${at}.`;
  else if (w.via === 'hand') said = `${n} was written on this line by hand, before names were kept in the log.`;
  else if (w.via === 'heard')
    said =
      w.voice?.label && w.voice.label !== n
        ? `It was heard as ${n}. Its voice is ${w.voice.label}'s now, which doesn't rename lines already heard.`
        : `It was heard as ${n}; nothing has named this line or its voice since.`;
  else said = 'Nobody has named its voice yet.';
  const kind = w.subject === 'shot' ? 'Face' : 'Voice';
  const match =
    w.similarity == null ? null : `${kind} match ${w.similarity.toFixed(2)} (a match needs ${w.threshold.toFixed(2)}).`;
  return { said, match };
}

function Body({ target, actions }: { target: WhyTarget; actions?: ReactNode }) {
  const why = useQuery({ queryKey: whyKey(target), queryFn: ({ signal }) => getWhy(target, { signal }), staleTime: 0 });
  const mark = useMutation({
    mutationFn: async (right: boolean) => {
      if (!why.data) throw new Error('nothing to mark');
      const j = await markName(why.data, right);
      if (!j.ok) throw new Error(j.error || 'not saved');
      return right;
    },
  });
  if (why.isPending)
    return (
      <Group gap="xs" role="status" aria-live="polite">
        <Loader size="xs" />
        <Text size="sm" c="dimmed">
          Looking it up…
        </Text>
      </Group>
    );
  if (why.isError)
    return (
      <PageError
        title="Couldn't say why"
        detail={messageOf(why.error)}
        onRetry={() => void why.refetch()}
        busy={why.isFetching}
      >
        {null}
      </PageError>
    );
  const w = why.data;
  const { said, match } = explain(w);
  return (
    <Stack gap={8}>
      <Text size="sm">{said}</Text>
      {match && (
        <Text size="sm" c="dimmed">
          {match}
        </Text>
      )}
      <Group gap={6} aria-label="Mark this name">
        <Text size="xs" c="dimmed">
          {`Is this ${w.name}?`}
        </Text>
        <Button
          size="compact-xs"
          variant={mark.data === true ? 'light' : 'subtle'}
          disabled={mark.isPending}
          onClick={() => mark.mutate(true)}
        >
          Yes
        </Button>
        <Button
          size="compact-xs"
          variant={mark.data === false ? 'light' : 'subtle'}
          disabled={mark.isPending}
          onClick={() => mark.mutate(false)}
        >
          No
        </Button>
      </Group>
      {mark.isSuccess && (
        <Text size="xs" c="dimmed" role="status">
          {`Marked ${mark.data ? 'right' : 'wrong'}, to check later names against. It changes nothing.`}
        </Text>
      )}
      {mark.isError && (
        <PageError title="That wasn't saved" detail={messageOf(mark.error)}>
          Try again.
        </PageError>
      )}
      {actions && <Group gap={6}>{actions}</Group>}
    </Stack>
  );
}

/** The popover, opened and closed by its owner, around the element it points at (a button that takes a ref). The
 *  explanation is asked for only while it's open. */
export function WhyName({
  target,
  name,
  opened,
  onClose,
  actions,
  children,
}: {
  target: WhyTarget;
  name: string;
  opened: boolean;
  onClose: () => void;
  actions?: ReactNode; // the fixes: "Not Dana…", "Rename this voice…"
  children: ReactElement;
}) {
  const title = useId();
  return (
    <Popover
      opened={opened}
      onChange={o => !o && onClose()}
      position="bottom-start"
      width={320}
      withinPortal
      trapFocus
      returnFocus
      closeOnEscape
      withRoles={false}
    >
      {/* the target sets its own aria state (Live's name button); the card's is only a wrapper around the ⋯ menu */}
      <Popover.Target>{children}</Popover.Target>
      {/* named by its heading: Mantine would name it by its target, which on the card is only the ⋯ menu */}
      <Popover.Dropdown role="dialog" aria-labelledby={title}>
        {/* where focus lands on opening, even while the explanation loads, so Escape is always caught inside */}
        <Text id={title} size="sm" fw={600} mb={6} tabIndex={-1} data-autofocus>
          {`Why ${name}`}
        </Text>
        {opened && <Body target={target} actions={actions} />}
      </Popover.Dropdown>
    </Popover>
  );
}
