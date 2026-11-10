import { Button, Group, Text, TextInput } from '@mantine/core';
import { useCallback, useEffect, useRef, useState } from 'react';
import { messageOf, request } from '../../shared/http';

type Candidate = { name: string; detail: string; url: string; source_title?: string; provider?: string };
/** Text-only lookup. No media is sent, and choosing a result never saves a person. */
export function PersonSearch({ name, onSelect }: { name: string; onSelect: (name: string) => void }) {
  const [context, setContext] = useState('');
  const [rows, setRows] = useState<Candidate[] | null>(null);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [note, setNote] = useState('');
  const epoch = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const invalidate = useCallback(() => {
    epoch.current++;
    controller.current?.abort();
    setBusy(false);
    setRows(null);
    setSelected(null);
    setError('');
    setNote('');
  }, []);
  useEffect(() => {
    if (name !== undefined) invalidate();
    return () => {
      epoch.current++;
      controller.current?.abort();
    };
  }, [name, invalidate]);
  const search = async () => {
    const id = ++epoch.current;
    controller.current?.abort();
    controller.current = new AbortController();
    setRows(null);
    setBusy(true);
    setError('');
    setSelected(null);
    try {
      const reply = await request<{ candidates?: Candidate[]; error?: string; note?: string }>(
        '/people/search',
        { name, context },
        controller.current.signal,
      );
      if (id !== epoch.current) return;
      if (reply.error) throw new Error(reply.error);
      setRows(reply.candidates ?? []);
      setNote(reply.note ?? '');
    } catch (e) {
      if (id === epoch.current) setError(messageOf(e));
    } finally {
      if (id === epoch.current) setBusy(false);
    }
  };
  return (
    <section aria-label="Find the right person">
      <Text size="sm" mt="sm">
        Optional: search the general web by name, role, company, or location. Only the text you enter is sent. No
        photos, audio, or stored notes are sent. Results are pages to review, not verified identities.
      </Text>
      <TextInput
        label="Affiliation, location, or other public context"
        value={context}
        maxLength={400}
        onKeyDown={e => {
          if (e.key === 'Enter') {
            e.preventDefault();
            if (!busy && name.trim()) void search();
          }
        }}
        onChange={e => {
          invalidate();
          setContext(e.currentTarget.value);
        }}
      />
      <Group mt="xs">
        <Button type="button" variant="default" disabled={!name.trim() || busy} onClick={() => void search()}>
          {busy ? 'Searching…' : 'Search / refine'}
        </Button>
        {busy && (
          <Button type="button" variant="subtle" onClick={invalidate}>
            Cancel search
          </Button>
        )}
      </Group>
      {error && (
        <Text role="alert" c="red">
          {error} Refine and retry, or add manually.
        </Text>
      )}
      {note && (
        <Text size="sm" c="dimmed">
          {note}
        </Text>
      )}
      <Text size="sm" mt="xs">
        Supplementary:{' '}
        <a
          href={`https://www.wikidata.org/w/index.php?search=${encodeURIComponent(name)}`}
          target="_blank"
          rel="noreferrer"
        >
          Wikidata
        </a>
      </Text>
      {rows?.length === 0 && <Text>No candidates found. Refine the context or add manually.</Text>}
      {!selected &&
        rows?.map(r => (
          <div key={`${r.url}:${r.name}`}>
            <Text fw={600}>{r.source_title ?? r.name}</Text>
            {r.provider && (
              <Text size="xs" c="dimmed">
                {r.provider}
              </Text>
            )}
            <Text size="sm">{r.detail}</Text>
            <a href={r.url} target="_blank" rel="noreferrer">
              Source
            </a>{' '}
            <Button type="button" size="xs" variant="subtle" onClick={() => setSelected(r)}>
              Preview {r.source_title ?? r.name}
            </Button>
          </div>
        ))}
      {selected && (
        <section aria-label="Selected candidate preview">
          <Text fw={600}>{selected.source_title ?? selected.name}</Text>
          <Text>Name to use: {selected.name}</Text>
          <Text>{selected.detail}</Text>
          <a href={selected.url} target="_blank" rel="noreferrer">
            Check source
          </a>
          <Text size="sm">
            Web results may refer to someone else. Confirm the name, then use Add to create the person. No identity
            training occurs.
          </Text>
          <Group>
            <Button type="button" variant="default" onClick={() => setSelected(null)}>
              Back to candidates
            </Button>
            <Button type="button" onClick={() => onSelect(selected.name)}>
              Use this name
            </Button>
          </Group>
        </section>
      )}
    </section>
  );
}
