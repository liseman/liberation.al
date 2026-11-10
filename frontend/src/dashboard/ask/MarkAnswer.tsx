// "Was this answer right?" under a Review Ask answer. A mark is saved as a question in the measurement set
// (POST /measurement/mark, dashboard/marks.py), so tools/measure.py can ask it again and check the answer. Right keeps
// the records the answer cited; Wrong keeps the owner's correction. It changes no answer.
import { Button, Checkbox, Group, Text, TextInput } from '@mantine/core';
import { IconCheck, IconX } from '@tabler/icons-react';
import { type FormEvent, useContext, useState } from 'react';
import { useStore } from 'zustand';
import { messageOf, request } from '../../shared/http';
import { EngineContext } from '../review/context';
import type { ReviewEngine } from '../review/engine';
import { isRec, list, plain, type Rec } from './format';
import type { Asked } from './job';

// The evidence sources tools/measure.py checks (marks.SOURCES); another source is saved without one.
const SOURCES = new Set(['transcript', 'camera', 'call', 'journal']);
const MAX_SAYS = 200;

export interface Cited {
  ts: number;
  source?: string;
}

/** The evidence records an answer cites by id, such as [C1] or [T2, J3], with a time, in the order listed. */
export function citedEvidence(result: unknown): (Rec & { ts: number })[] {
  if (!isRec(result)) return [];
  const ids = new Set<string>();
  for (const [, inside] of plain(result.answer).matchAll(/\[([^\]]+)\]/g))
    for (const id of (inside ?? '').split(/[\s,;]+/)) {
      if (/^[CJT]\d+$/.test(id)) ids.add(id);
      // A range such as T1-T3 (a hyphen or an en dash): every id in it, up to a sane length.
      const range = id.match(/^([CJT])(\d+)[-\u2013]\1?(\d+)$/);
      if (range) {
        const [, kind, a, b] = range;
        for (let n = Number(a); n <= Number(b) && n - Number(a) < 200; n++) ids.add(`${kind}${n}`);
      }
    }
  return list(result.evidence).filter(
    (e): e is Rec & { ts: number } =>
      isRec(e) && ids.has(plain(e.id)) && typeof e.ts === 'number' && Number.isFinite(e.ts),
  );
}

/** The records an answer cites, each as its time and source, for a mark. */
export function citedRecords(result: unknown): Cited[] {
  const out: Cited[] = [];
  const seen = new Set<string>(); // one record can come back under two ids
  for (const e of citedEvidence(result)) {
    const source = e.kind === 'journal' ? 'journal' : plain(e.source);
    const record: Cited = SOURCES.has(source) ? { ts: e.ts, source } : { ts: e.ts };
    const key = `${record.ts} ${record.source ?? ''}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(record);
  }
  return out;
}

type Open = 'right' | 'wrong' | null;
type Saved = { right: boolean; what: string };

const records = (n: number) => (n === 1 ? 'the record it cited' : `the ${n} records it cited`);

/** "a", "a and b", or "a, b, and c". */
const listed = (items: string[]) =>
  items.length < 3 ? items.join(' and ') : `${items.slice(0, -1).join(', ')}, and ${items[items.length - 1]}`;

/** "Cite where Review is now": the playhead, read when the box is ticked so the label follows it. */
function MomentBox({
  engine,
  checked,
  onChange,
}: {
  engine: ReviewEngine;
  checked: number | null;
  onChange: (t: number | null) => void;
}) {
  const t = useStore(engine.ui, s => s.t);
  const at = checked ?? t;
  return (
    <Checkbox
      size="xs"
      mt="xs"
      disabled={at === null}
      checked={checked !== null}
      onChange={e => onChange(e.currentTarget.checked ? t : null)}
      label={
        at === null
          ? 'It should cite a moment (move Review to it first)'
          : `It should cite ${new Date(at * 1000).toLocaleTimeString()}, ${checked === null ? 'where Review is now' : 'the moment you picked'}`
      }
    />
  );
}

export function MarkAnswer({ result, asked }: { result: unknown; asked: Asked }) {
  const engine = useContext(EngineContext);
  const cited = citedRecords(result);
  const [open, setOpen] = useState<Open>(null);
  const [says, setSays] = useState('');
  const [notSays, setNotSays] = useState('');
  const [moment, setMoment] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<Saved | null>(null);
  const [error, setError] = useState('');

  const save = async (right: boolean, extra: Record<string, unknown>, what: string) => {
    setSaving(true);
    setError('');
    try {
      await request('/measurement/mark', {
        kind: 'answer',
        question: asked.question,
        asked_at: asked.at,
        ...(asked.day ? { day: asked.day } : {}),
        right,
        ...extra,
      });
      setSaved({ right, what });
      setOpen(null);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setSaving(false);
    }
  };

  // Yes saves at once: with the records the answer cited when there are any, else as a plain thumbs-up. It never
  // opens a form; "No, correct it" is where details go.
  const right = () => {
    setError('');
    if (cited.length) void save(true, { cited }, records(cited.length));
    else void save(true, {}, 'that it was right');
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const s = says.trim();
    const n = notSays.trim();
    if (open === 'right') {
      if (s) void save(true, { says: s }, 'what a right answer says');
      else setError('Say what a right answer says: this one cited no records to check.');
      return;
    }
    if (!s && !n && moment === null) {
      setError("Say what a right answer says, what it shouldn't say, or pick the moment it should cite.");
      return;
    }
    const what = listed(
      [
        s && 'what a right answer says',
        n && "what it shouldn't say",
        moment !== null && 'the moment it should cite',
      ].filter((x): x is string => !!x),
    );
    void save(
      false,
      { ...(s ? { says: s } : {}), ...(n ? { not_says: n } : {}), ...(moment !== null ? { moment } : {}) },
      what,
    );
  };

  return (
    <section
      aria-label="Mark this answer"
      title="Saved as a measurement question, to check later answers against. It doesn't change this answer."
    >
      <Group gap={4} wrap="wrap" mt="sm">
        <Text span size="xs" c="dimmed">
          Was this right?
        </Text>
        <Button
          variant={saved?.right ? 'light' : 'subtle'}
          size="compact-xs"
          leftSection={<IconCheck size={12} />}
          disabled={saving}
          aria-pressed={saved?.right === true}
          onClick={right}
        >
          Yes
        </Button>
        <Button
          variant={saved && !saved.right ? 'light' : 'subtle'}
          size="compact-xs"
          leftSection={<IconX size={12} />}
          disabled={saving}
          aria-pressed={saved?.right === false}
          aria-expanded={open === 'wrong'}
          onClick={() => {
            setError('');
            setOpen(open === 'wrong' ? null : 'wrong');
          }}
        >
          No, correct it
        </Button>
      </Group>
      {open && (
        <form onSubmit={submit} aria-label={open === 'right' ? 'What a right answer says' : 'Correct this answer'}>
          <TextInput
            size="xs"
            mt="xs"
            label="A right answer says"
            description={open === 'right' ? 'It cited no records, so say what it must say.' : undefined}
            maxLength={MAX_SAYS}
            value={says}
            onChange={e => setSays(e.currentTarget.value)}
          />
          {open === 'wrong' && (
            <>
              <TextInput
                size="xs"
                mt="xs"
                label="It shouldn't say"
                maxLength={MAX_SAYS}
                value={notSays}
                onChange={e => setNotSays(e.currentTarget.value)}
              />
              {engine && <MomentBox engine={engine} checked={moment} onChange={setMoment} />}
            </>
          )}
          <Group gap="xs" mt="xs">
            <Button type="submit" size="compact-sm" loading={saving}>
              Save
            </Button>
            <Button variant="subtle" size="compact-sm" disabled={saving} onClick={() => setOpen(null)}>
              Cancel
            </Button>
          </Group>
        </form>
      )}
      {saved && !open && (
        <Text size="xs" c="dimmed" role="status" mt={4}>
          {`Marked ${saved.right ? 'right' : 'wrong'}: saved the question with ${saved.what}.`}
        </Text>
      )}
      {error && (
        <Text size="xs" c="var(--lib-bad)" role="alert" mt={4}>
          {error}
        </Text>
      )}
    </section>
  );
}
