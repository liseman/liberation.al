// A face's name, in both face pickers: the shared person name box (lib/PersonNameInput.tsx). Picking an existing name
// or Add is required; typing alone never claims a face.
import { useState } from 'react';
import { LazyPersonNameInput } from '../lib/LazyPersonNameInput';

export function FaceNameInput({
  people,
  disabled,
  pending,
  onSelect,
  onAdd,
}: {
  people: string[];
  disabled: boolean;
  pending: boolean;
  onSelect: (name: string | null) => void;
  onAdd: (name: string) => void;
}) {
  const [search, setSearch] = useState('');
  return (
    <LazyPersonNameInput
      known={people}
      value={search}
      onChange={typed => {
        setSearch(typed);
        onSelect(null);
      }}
      onPick={name => {
        if (disabled || pending) return;
        const person = people.find(p => p.toLocaleLowerCase() === name.toLocaleLowerCase());
        if (person) {
          setSearch(person);
          onSelect(person);
        } else onAdd(name);
      }}
      allowNew={!pending}
      label="Person"
      placeholder="Pick someone or type a new name"
      emptyText={pending ? 'Loading people…' : 'Type a name'}
      disabled={disabled}
      size="sm"
      modalFocus
    />
  );
}
