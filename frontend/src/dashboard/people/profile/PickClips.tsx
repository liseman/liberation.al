// "Pick clips" on the contact card's Voice clone section: hear the person's recent lines, tick the ones that
// are clearly them, and make their clone's reference from those. The picked lines count as theirs from then on.
import { Button, Checkbox, Group, Modal, Stack, Text } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf } from '../../../shared/http';
import { PlayClip } from '../../lib/PlayToggle';
import { PageError, PageLoading } from '../../lib/page';
import { CLONES_KEY, getPickable, lineClip, type PickableLine, pickClips } from '../api';
import { useClones } from '../clones/store';

const secs = (n: number) => `${Math.round(n)} s`;

function Row({ l, on, toggle }: { l: PickableLine; on: boolean; toggle: () => void }) {
  return (
    <Group gap="xs" wrap="nowrap" align="flex-start">
      <Checkbox checked={on} onChange={toggle} aria-label={`Use "${l.text}"`} mt={4} />
      <PlayClip size="xs" url={lineClip(l.line)} playKey={`pick:${l.line}`} ts={l.ts_start} label={`“${l.text}”`} />
      <Text size="sm" style={{ flex: 1 }}>
        {l.text}{' '}
        <Text span size="xs" c="dimmed">
          {secs(l.dur_s)}
        </Text>
      </Text>
    </Group>
  );
}

function Body({ id, name, needs, onDone }: { id: number; name: string; needs: number; onDone: () => void }) {
  const client = useQueryClient();
  const q = useQuery({ queryKey: ['clone-lines', id], queryFn: ({ signal }) => getPickable(id, { signal }) });
  const [picked, setPicked] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const building = useClones(s => s.busy != null); // a build writes the reference too: one at a time
  const [err, setErr] = useState<string | null>(null);
  const lines = q.data?.lines ?? [];
  const total = lines.filter(l => picked.includes(l.line)).reduce((t, l) => t + l.dur_s, 0);
  const toggle = (line: string) => setPicked(p => (p.includes(line) ? p.filter(x => x !== line) : [...p, line]));
  const use = async () => {
    setBusy(true);
    setErr(null);
    try {
      await pickClips(id, picked);
      await client.invalidateQueries({ queryKey: CLONES_KEY });
      onDone();
    } catch (e) {
      setErr(messageOf(e));
    } finally {
      setBusy(false);
    }
  };
  if (q.isPending) return <PageLoading label={`${name}'s lines`} shape="rows" lines={3} />;
  if (q.isError)
    return (
      <PageError
        title={`Couldn't load ${name}'s lines`}
        detail={messageOf(q.error)}
        onRetry={() => void q.refetch()}
        busy={q.isFetching}
      />
    );
  if (!lines.length) return <Text size="sm">No lines of {name}'s are long and clear enough to use yet.</Text>;
  return (
    <Stack gap="xs">
      <Text size="sm" c="dimmed">
        Play each one and tick the lines where only {name} is speaking. The reference is made from them, in the order
        you tick them.
      </Text>
      {lines.map(l => (
        <Row key={l.line} l={l} on={picked.includes(l.line)} toggle={() => toggle(l.line)} />
      ))}
      <Group justify="space-between" mt="sm">
        <Text size="sm" role="status">
          {secs(total)} picked, needs {secs(needs)}
        </Text>
        <Button size="xs" disabled={busy || building || total < needs} onClick={() => void use()}>
          {busy ? 'Making the reference' : building ? 'A voice is being built' : 'Use these clips'}
        </Button>
      </Group>
      {err && <PageError title={err}>{null}</PageError>}
    </Stack>
  );
}

export function PickClips({
  opened,
  onClose,
  id,
  name,
  needs = 12,
}: {
  opened: boolean;
  onClose: () => void;
  id: number;
  name: string;
  needs?: number;
}) {
  return (
    <Modal opened={opened} onClose={onClose} title={`Pick clips of ${name}'s voice`} size={560}>
      {opened && <Body id={id} name={name} needs={needs} onDone={onClose} />}
    </Modal>
  );
}
