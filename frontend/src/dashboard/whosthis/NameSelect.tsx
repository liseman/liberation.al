// Who's this?'s name box: the shared person name box (lib/PersonNameInput.tsx), searchable over the people
// Liberational knows, its guesses first, with "Add <name>" when what's typed is nobody's name yet. Enter takes what was
// typed: a known name it matches exactly, else the new name; the arrow keys reach the other matches.
import { type NameSuggestion, PersonNameInput } from '../lib/PersonNameInput';

export { nameOptions } from '../lib/PersonNameInput';

export function NameSelect({
  known,
  suggestions = [],
  typed,
  onType,
  onPick,
  onEscape,
  disabled,
}: {
  known: readonly string[];
  suggestions?: readonly NameSuggestion[];
  typed: string;
  onType: (typed: string) => void;
  onPick: (name: string) => void;
  onEscape: () => void;
  disabled?: boolean;
}) {
  return (
    <PersonNameInput
      known={known}
      suggestions={suggestions}
      value={typed}
      onChange={onType}
      onPick={onPick}
      onCancel={onEscape}
      disabled={disabled}
      ariaLabel="This person's name"
      flex={1}
      miw={140}
    />
  );
}
