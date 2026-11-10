// Label a window's entries and exits: every time someone came into or went out of one camera's view in a span
// of the timeline, and the stretches that could not be told. Saved in entries.jsonl (POST /measurement/mark, kind
// "entries", dashboard/marks.py) for tools/measure.py --part entries. A side drawer with no overlay, so the player
// and the timeline stay usable: scrub to each one and add it at the playhead.
import { ActionIcon, Button, Checkbox, Drawer, Group, NativeSelect, Stack, Text, TextInput } from '@mantine/core';
import { IconX } from '@tabler/icons-react';
import { type FormEvent, useState } from 'react';
import { HttpError, messageOf, request } from '../../shared/http';
import { hms } from '../lib/format';
import { useEngine } from './context';
import { parseClock } from './EventForm';
import type { EventSpan } from './types';
import { viewName } from './views';

type Kind = 'entry' | 'exit';
interface Row {
  kind: Kind;
  at: string;
}
interface Unsure {
  from: string;
  to: string;
  reason: string;
}

const CONDITIONS = [
  ['camera_moving', 'Camera moving'],
  ['low_light', 'Low light'],
  ['occlusion', 'Something in the way'],
  ['crowded', 'Crowded'],
] as const;

export function EntriesForm({ span, onClose }: { span: EventSpan | null; onClose: () => void }) {
  return (
    <Drawer
      opened={!!span}
      onClose={onClose}
      title="Entries and exits"
      position="right"
      size="md"
      withOverlay={false}
      lockScroll={false}
      trapFocus={false}
      closeOnClickOutside={false}
    >
      {span && <Fields key={`${span.cam}-${span.start}-${span.end}`} span={span} onClose={onClose} />}
    </Drawer>
  );
}

function Fields({ span, onClose }: { span: EventSpan; onClose: () => void }) {
  const engine = useEngine();
  const [cam, setCam] = useState(span.cam);
  const [from, setFrom] = useState(hms(span.start));
  const [to, setTo] = useState(hms(span.end));
  const [rows, setRows] = useState<Row[]>([]);
  const [unsure, setUnsure] = useState<Unsure[]>([]);
  const [conditions, setConditions] = useState<string[]>([]);
  const [noneSeen, setNoneSeen] = useState(false);
  const [note, setNote] = useState('');
  const [overlap, setOverlap] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState('');
  const [error, setError] = useState('');

  // The playhead, else the window's start: where an added entry, exit, or unsure stretch begins.
  const now = () => hms(engine.t != null && Number.isFinite(engine.t) ? engine.t : span.start);
  const add = (kind: Kind) => setRows(r => [...r, { kind, at: now() }]);
  const setRow = (i: number, change: Partial<Row>) =>
    setRows(r => r.map((x, j) => (j === i ? { ...x, ...change } : x)));
  const setSpan = (i: number, change: Partial<Unsure>) =>
    setUnsure(u => u.map((x, j) => (j === i ? { ...x, ...change } : x)));

  const submit = async (e: FormEvent, replace = false) => {
    e.preventDefault();
    setError('');
    setSaved('');
    setOverlap(false);
    const start = parseClock(from, span.start);
    const end = parseClock(to, span.start);
    if (start == null || end == null) return setError('Give the times as hours, minutes, and seconds, like 18:00:05.');
    if (end <= start) return setError('The window ends after it starts.');
    const events = [];
    for (const r of rows) {
      const at = parseClock(r.at, span.start);
      if (at == null) return setError(`"${r.at}" is not a time like 18:00:05.`);
      events.push({ kind: r.kind, at });
    }
    const stretches = [];
    for (const u of unsure) {
      const a = parseClock(u.from, span.start);
      const b = parseClock(u.to, span.start);
      if (a == null || b == null || b <= a) return setError('Each unsure stretch needs a start and a later end.');
      if (!u.reason.trim()) return setError('Say why each unsure stretch could not be told.');
      stretches.push({ start: a, end: b, reason: u.reason.trim() });
    }
    if (!events.length && !stretches.length && !noneSeen)
      return setError('Add each entry and exit, or tick that nobody came or went.');
    setSaving(true);
    try {
      const j = await request<{ ok: boolean; replaced: number }>('/measurement/mark', {
        kind: 'entries',
        camera: cam,
        start,
        end,
        events,
        unsure: stretches,
        conditions,
        none_seen: !events.length && !stretches.length,
        ...(note.trim() ? { note: note.trim() } : {}),
        ...(replace ? { replace: true } : {}),
      });
      const n = events.length;
      setSaved(
        `Saved: ${n ? `${n} ${n === 1 ? 'entry or exit' : 'entries and exits'}` : 'nobody came or went'}` +
          `${j.replaced ? `, in place of ${j.replaced} overlapping ${j.replaced === 1 ? 'window' : 'windows'}` : ''}.`,
      );
    } catch (ex) {
      if (ex instanceof HttpError && ex.status === 409) setOverlap(true);
      setError(`Not saved: ${messageOf(ex)}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit} aria-label="Entries and exits">
      <Stack gap="sm">
        <Text size="sm" c="dimmed">
          Every time someone came into or went out of this camera's view in the window, by watching. Scrub to each one
          and add it at the playhead. A label to score the movement tracks against; it changes nothing in Review.
        </Text>
        <Group grow gap="sm">
          <TextInput label="From" value={from} onChange={e => setFrom(e.currentTarget.value)} required />
          <TextInput label="To" value={to} onChange={e => setTo(e.currentTarget.value)} required />
        </Group>
        <NativeSelect
          label="Camera"
          value={cam}
          onChange={e => setCam(e.currentTarget.value)}
          data={engine.cams.map(c => ({ value: c, label: viewName(c) }))}
        />
        <Group gap="xs">
          <Button size="compact-sm" variant="default" onClick={() => add('entry')}>
            Entry at playhead
          </Button>
          <Button size="compact-sm" variant="default" onClick={() => add('exit')}>
            Exit at playhead
          </Button>
        </Group>
        {rows.map((r, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: rows are edited in place, never reordered
          <Group key={i} gap={6} wrap="nowrap">
            <NativeSelect
              aria-label="Entry or exit"
              value={r.kind}
              onChange={e => setRow(i, { kind: e.currentTarget.value as Kind })}
              data={[
                { value: 'entry', label: 'Entry' },
                { value: 'exit', label: 'Exit' },
              ]}
            />
            <TextInput aria-label="At" value={r.at} onChange={e => setRow(i, { at: e.currentTarget.value })} w={110} />
            <ActionIcon
              variant="subtle"
              aria-label={`Remove this ${r.kind}`}
              onClick={() => setRows(x => x.filter((_, j) => j !== i))}
            >
              <IconX size={14} />
            </ActionIcon>
          </Group>
        ))}
        <Stack gap={4}>
          <Text size="sm" fw={500}>
            Could not tell (optional)
          </Text>
          <Text size="xs" c="dimmed">
            A stretch where the edge of the view was hidden, dark, or missing. Leave out anything you would be guessing.
          </Text>
          {unsure.map((u, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: rows are edited in place, never reordered
            <Group key={i} gap={6} wrap="nowrap">
              <TextInput
                aria-label="Unsure from"
                value={u.from}
                onChange={e => setSpan(i, { from: e.currentTarget.value })}
                w={90}
              />
              <TextInput
                aria-label="Unsure to"
                value={u.to}
                onChange={e => setSpan(i, { to: e.currentTarget.value })}
                w={90}
              />
              <TextInput
                aria-label="Why"
                placeholder="door hides the edge"
                value={u.reason}
                maxLength={200}
                onChange={e => setSpan(i, { reason: e.currentTarget.value })}
              />
              <ActionIcon
                variant="subtle"
                aria-label="Remove this unsure stretch"
                onClick={() => setUnsure(x => x.filter((_, j) => j !== i))}
              >
                <IconX size={14} />
              </ActionIcon>
            </Group>
          ))}
          <Button
            variant="subtle"
            size="compact-sm"
            style={{ alignSelf: 'flex-start' }}
            onClick={() => setUnsure(u => [...u, { from: now(), to: now(), reason: '' }])}
          >
            Add an unsure stretch
          </Button>
        </Stack>
        <Checkbox.Group label="Conditions (optional)" value={conditions} onChange={setConditions}>
          <Group gap="sm" mt={4}>
            {CONDITIONS.map(([value, label]) => (
              <Checkbox key={value} value={value} label={label} size="xs" />
            ))}
          </Group>
        </Checkbox.Group>
        {!rows.length && !unsure.length && (
          <Checkbox
            label="I watched the whole window: nobody came or went"
            checked={noneSeen}
            onChange={e => setNoneSeen(e.currentTarget.checked)}
          />
        )}
        <TextInput
          label="Note (optional)"
          value={note}
          maxLength={200}
          onChange={e => setNote(e.currentTarget.value)}
        />
        {error && (
          <Text size="sm" c="var(--lib-bad)" role="alert">
            {error}
          </Text>
        )}
        {saved && (
          <Text size="sm" role="status">
            {saved}
          </Text>
        )}
        <Group justify="flex-end" gap="xs">
          <Button variant="default" onClick={onClose}>
            {saved ? 'Done' : 'Cancel'}
          </Button>
          {overlap && (
            <Button variant="subtle" c="var(--lib-bad)" loading={saving} onClick={e => void submit(e, true)}>
              Replace the overlapping window
            </Button>
          )}
          <Button type="submit" loading={saving}>
            {saved ? 'Save again' : 'Save'}
          </Button>
        </Group>
      </Stack>
    </form>
  );
}
