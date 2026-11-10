// "Who is this?" (#309): after "Not <name>…" on something put down to someone (a photo, a video, a line), pick who it
// really is, someone new, or just take it off them. The contact card opens it from an item's ⋯ menu, and Review's
// transcript from a line's (#300); each posts the correction it picks (/people/correct).
import { Button, Group, Modal, Text, TextInput } from '@mantine/core';
import { useEffect, useState } from 'react';
import classes from './whoIsThis.module.css';

/** "Who is this?" after "Not Dana…": pick who it really is (search, or someone new), or just take it off Dana. Open
 *  while label is set: what it's about, "this photo" or "this line". */
export function WhoIsThis({
  name,
  label,
  people,
  onMove,
  onRemove,
  onClose,
}: {
  name: string;
  label: string | null;
  people: readonly { name: string }[];
  onMove: (to: string) => void;
  onRemove: () => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState('');
  const [fresh, setFresh] = useState<string | null>(null); // "Someone new…": the name being typed
  // Each opening starts clean. It stays mounted between openings, so Mantine gives focus back to the ⋯ button.
  useEffect(() => {
    if (label != null) {
      setQuery('');
      setFresh(null);
    }
  }, [label]);
  const q = query.trim().toLowerCase();
  const others = people.filter(p => p.name !== name && (!q || p.name.toLowerCase().includes(q)));
  const close = onClose;
  const newName = fresh?.trim() ?? '';
  return (
    <Modal opened={label != null} onClose={close} title="Who is this?" size={420} returnFocus={false}>
      {label != null && (
        <Text size="sm" c="dimmed" mb="sm">
          {`It's not ${name} in ${label}. Who is it?`}
        </Text>
      )}
      {fresh == null ? (
        <>
          <TextInput
            aria-label="Search people"
            placeholder="Search people"
            data-autofocus
            value={query}
            onChange={e => setQuery(e.currentTarget.value)}
            mb="xs"
          />
          <ul className={classes.whoList}>
            {others.map(p => (
              <li key={p.name}>
                <button type="button" className={classes.whoRow} onClick={() => onMove(p.name)}>
                  {p.name}
                </button>
              </li>
            ))}
            {!others.length && (
              <li>
                <Text size="sm" c="dimmed" px={10} py={6}>
                  {q ? `Nobody matches “${query.trim()}”.` : 'Nobody else is known yet.'}
                </Text>
              </li>
            )}
            <li>
              <button type="button" className={classes.whoRow} onClick={() => setFresh(query.trim())}>
                Someone new…
              </button>
            </li>
          </ul>
        </>
      ) : (
        <form
          onSubmit={e => {
            e.preventDefault();
            if (newName) onMove(newName);
          }}
        >
          <TextInput
            label="New person's name"
            data-autofocus
            value={fresh}
            onChange={e => setFresh(e.currentTarget.value)}
          />
          <Group justify="flex-end" gap="xs" mt="sm">
            <Button variant="default" onClick={() => setFresh(null)}>
              Back
            </Button>
            <Button type="submit" disabled={!newName}>
              {newName ? `Move to ${newName}` : 'Move'}
            </Button>
          </Group>
        </form>
      )}
      <Button variant="subtle" fullWidth mt="md" onClick={onRemove}>
        {`Not sure, just remove it from ${name}`}
      </Button>
    </Modal>
  );
}
