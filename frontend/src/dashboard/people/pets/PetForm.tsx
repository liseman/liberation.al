// The "New pet" and "Edit" dialog, as People's "New person" is: a name, the species, and optionally the sex, then Save.
import { Button, Group, Modal, NativeSelect, Text, TextInput } from '@mantine/core';
import { type FormEvent, useState } from 'react';
import { messageOf } from '../../../shared/http';
import { type Pet, savePet } from './api';

const SEXES = [
  { value: '', label: 'Not set' },
  { value: 'female', label: 'Female' },
  { value: 'male', label: 'Male' },
];

export function PetForm({
  opened,
  pet,
  options,
  species: fixed,
  onClose,
  onSaved,
}: {
  opened: boolean;
  pet?: Pet | null; // editing this pet; none adds one
  options: string[]; // the species a pet can be
  species?: string; // a new pet's species to start with (the picker's level, naming a sighting group)
  onClose: () => void;
  onSaved: (pets: Pet[], saved: Pet) => void;
}) {
  const [name, setName] = useState(pet?.name ?? '');
  const [kind, setKind] = useState(pet?.species ?? fixed ?? options[0] ?? 'cat');
  const [sex, setSex] = useState<string>(pet?.sex ?? '');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const nm = name.trim();
  const save = async (e: FormEvent) => {
    e.preventDefault();
    if (!nm) return;
    setBusy(true);
    setErr(null);
    try {
      const reply = await savePet({ id: pet?.id ?? null, name: nm, species: kind, sex: sex || null });
      const saved = reply.pets.find(p => (pet ? p.id === pet.id : p.name === nm && p.species === kind));
      if (saved) onSaved(reply.pets, saved);
      onClose();
    } catch (e) {
      setErr(messageOf(e) || 'Saving failed. Try again.');
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal opened={opened} onClose={onClose} title={pet ? `Edit ${pet.name}` : 'New pet'} size={440}>
      <form onSubmit={save}>
        <TextInput
          label="Pet name"
          maxLength={80}
          data-autofocus
          value={name}
          onChange={e => setName(e.currentTarget.value)}
        />
        <NativeSelect
          label="Species"
          mt="xs"
          data={options.map(value => ({ value, label: value }))}
          value={kind}
          onChange={e => setKind(e.currentTarget.value)}
        />
        <NativeSelect
          label="Sex (optional)"
          description="Lets Ask know who “she” or “he” is"
          mt="xs"
          data={SEXES}
          value={sex}
          onChange={e => setSex(e.currentTarget.value)}
        />
        <Text size="sm" c="dimmed" mt={8}>
          With one named pet of a species, a lone animal can be assumed to be that pet. Several named pets of the same
          species stay unnamed until their sightings show they can be told apart.
        </Text>
        {err && (
          <Text size="sm" c="var(--lib-bad)" role="alert" mt={8}>
            {err}
          </Text>
        )}
        <Group justify="flex-end" gap="xs" mt="lg">
          <Button variant="default" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" disabled={!nm || busy || !options.includes(kind)}>
            {pet ? 'Save pet' : nm ? `Add ${nm}` : 'Add'}
          </Button>
        </Group>
      </form>
    </Modal>
  );
}
