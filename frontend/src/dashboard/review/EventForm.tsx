// Mark a span of one camera as an event (#287, #518): what happened in the owner's own words, how many times it
// happened by watching, and optionally who it involved, what was measured (a value and a unit), and notes. It is
// saved as a label in events.jsonl (POST /measurement/mark, dashboard/marks.py), for tools/measure.py --part events.
// Marking the same span again replaces its label.
import {
  ActionIcon,
  Button,
  Group,
  Modal,
  NativeSelect,
  NumberInput,
  Stack,
  Text,
  Textarea,
  TextInput,
} from '@mantine/core';
import { IconX } from '@tabler/icons-react';
import { type FormEvent, useState } from 'react';
import { messageOf, request } from '../../shared/http';
import { hms } from '../lib/format';
import { PersonNameInput } from '../lib/PersonNameInput';
import { useKnownPeople } from '../livetab/people';
import { useEngine, useReview } from './context';
import type { EventSpan } from './types';
import { viewName } from './views';

/** A time typed as HH:MM:SS (or HH:MM) on the same day as `near`, as Unix seconds; null when it isn't one. */
export function parseClock(text: string, near: number): number | null {
  const m = /^\s*(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?\s*$/.exec(text);
  if (!m) return null;
  const [h, min, sec] = [Number(m[1]), Number(m[2]), Number(m[3] ?? 0)];
  if (h > 23 || min > 59 || sec >= 60) return null;
  const d = new Date(near * 1000);
  d.setHours(h, min, 0, 0);
  return d.getTime() / 1000 + sec;
}

/** Notes, one per line; a line that starts with a number and a colon ("2: only half in view") is about that time
 *  it happened. */
export function parseNotes(text: string): { note: string; nth?: number }[] {
  return text
    .split('\n')
    .map(l => l.trim())
    .filter(Boolean)
    .map(l => {
      const m = /^(\d+)\s*:\s*(?!\d)(.+)$/.exec(l); // "18:05 came back" is a time, not the 18th
      return m ? { nth: Number(m[1]), note: (m[2] ?? '').trim() } : { note: l };
    });
}

interface Measured {
  name: string;
  value: number | string;
  unit: string;
}

export function EventForm({
  span,
  onClose,
  onEntries,
}: {
  span: EventSpan | null;
  onClose: () => void;
  /** Label this span's entries and exits instead (EntriesForm), when Review offers it. */
  onEntries?: (span: EventSpan) => void;
}) {
  return (
    <Modal opened={!!span} onClose={onClose} title="Mark an event" size="md">
      {span && (
        <Fields key={`${span.cam}-${span.start}-${span.end}`} span={span} onClose={onClose} onEntries={onEntries} />
      )}
    </Modal>
  );
}

function Fields({
  span,
  onClose,
  onEntries,
}: {
  span: EventSpan;
  onClose: () => void;
  onEntries?: (span: EventSpan) => void;
}) {
  const engine = useEngine();
  const filtered = useReview(s => s.person); // Review opened for one person: offered first, never filled in
  const known = useKnownPeople();
  const [person, setPerson] = useState('');
  const [cam, setCam] = useState(span.cam);
  const [from, setFrom] = useState(hms(span.start));
  const [to, setTo] = useState(hms(span.end));
  const [what, setWhat] = useState('');
  const [count, setCount] = useState<number | string>(1);
  const [measured, setMeasured] = useState<Measured[]>([]);
  const [notes, setNotes] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState('');
  const [error, setError] = useState('');

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setSaved('');
    const start = parseClock(from, span.start);
    const end = parseClock(to, span.start);
    if (start == null || end == null) return setError('Give the times as hours, minutes, and seconds, like 18:00:05.');
    if (end <= start) return setError('The span ends after it starts.');
    if (typeof count !== 'number' || !Number.isInteger(count) || count < 0)
      return setError('Count how many times it happened by watching: a whole number.');
    const attributes: Record<string, { value: number; unit: string }> = {};
    for (const m of measured) {
      if (!m.name.trim() && m.value === '' && !m.unit.trim()) continue; // a row left empty
      if (!m.name.trim() || typeof m.value !== 'number' || !m.unit.trim())
        return setError('Each measurement needs a name, a number, and a unit, like distance 3 m.');
      attributes[m.name.trim()] = { value: m.value, unit: m.unit.trim() };
    }
    setSaving(true);
    try {
      const j = await request<{ ok: boolean; error?: string }>('/measurement/mark', {
        kind: 'event',
        ...(person.trim() ? { person: person.trim() } : {}),
        camera: cam,
        what: what.trim(),
        start,
        end,
        count,
        attributes,
        notes: parseNotes(notes),
      });
      if (!j.ok) throw new Error(j.error || 'not saved');
      engine.clearEventMarks();
      setSaved(
        `Saved: ${[person.trim(), what.trim()].filter(Boolean).join(', ')}, ${count} ${count === 1 ? 'time' : 'times'}.`,
      );
    } catch (ex) {
      setError(`Not saved: ${messageOf(ex)}`);
    } finally {
      setSaving(false);
    }
  };

  const setRow = (i: number, change: Partial<Measured>) =>
    setMeasured(rows => rows.map((r, j) => (j === i ? { ...r, ...change } : r)));
  return (
    <form onSubmit={submit} aria-label="Mark an event">
      <Stack gap="sm">
        <Text size="sm" c="dimmed">
          A label to score finding and counting events against. It changes nothing in Review.
        </Text>
        {onEntries && (
          <Button
            variant="subtle"
            size="compact-sm"
            style={{ alignSelf: 'flex-start' }}
            onClick={() => onEntries(span)}
          >
            Label entries and exits instead
          </Button>
        )}
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
        <PersonNameInput
          label="Who (optional)"
          description="Who it involved, if anyone"
          known={[...new Set([filtered, ...known].filter(Boolean))]}
          value={person}
          onChange={setPerson}
          onPick={setPerson}
          placeholder=""
          size="sm"
        />
        <TextInput
          label="What happened"
          description="In your own words, the same words each time: sat down, opened the fridge, dog on the couch"
          value={what}
          onChange={e => setWhat(e.currentTarget.value)}
          required
          maxLength={80}
          data-autofocus
        />
        <NumberInput
          label="How many times"
          description="It happened in this span, counted by watching"
          value={count}
          onChange={setCount}
          min={0}
          allowDecimal={false}
          allowNegative={false}
          required
        />
        <Stack gap={4}>
          <Text size="sm" fw={500}>
            Measured (optional)
          </Text>
          {measured.map((m, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: rows are edited in place, never reordered
            <Group key={i} gap={6} wrap="nowrap">
              <TextInput
                aria-label="Measurement"
                placeholder="distance"
                value={m.name}
                onChange={e => setRow(i, { name: e.currentTarget.value })}
              />
              <NumberInput
                aria-label="Value"
                placeholder="3"
                value={m.value}
                onChange={v => setRow(i, { value: v })}
                w={90}
              />
              <TextInput
                aria-label="Unit"
                placeholder="m"
                value={m.unit}
                onChange={e => setRow(i, { unit: e.currentTarget.value })}
                w={80}
              />
              <ActionIcon
                variant="subtle"
                aria-label="Remove this measurement"
                onClick={() => setMeasured(rows => rows.filter((_, j) => j !== i))}
              >
                <IconX size={14} />
              </ActionIcon>
            </Group>
          ))}
          <Button
            variant="subtle"
            size="compact-sm"
            onClick={() => setMeasured(rows => [...rows, { name: '', value: '', unit: '' }])}
            style={{ alignSelf: 'flex-start' }}
          >
            Add a measurement
          </Button>
        </Stack>
        <Textarea
          label="Notes (optional)"
          description='One per line. Start a line with which time it is about, like "2: only half in view".'
          value={notes}
          onChange={e => setNotes(e.currentTarget.value)}
          autosize={false}
          rows={3}
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
          <Button type="submit" loading={saving}>
            {saved ? 'Save again' : 'Save'}
          </Button>
        </Group>
      </Stack>
    </form>
  );
}
