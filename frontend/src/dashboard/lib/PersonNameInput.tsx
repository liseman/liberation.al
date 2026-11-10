// One name box for people (#2450 duplicates item 7): the transcript's name box, Who's this?, a face's name, and an
// event's "Who". It searches the people Liberational knows, with any guesses first; what's typed comes first as an
// exact match or, where a new name is allowed, "Add “<name>”" (#2047's creatable pattern), so Enter keeps the typed
// name while the arrow keys reach the other matches. Escape cancels; a box that saves as it's left can commit on blur.
import { Combobox, TextInput, useCombobox } from '@mantine/core';
import { type KeyboardEvent, type ReactNode, useEffect, useRef } from 'react';

const ADD = '\u0000add:'; // an option's value that adds the typed name, which no real name can start with

export interface NameSuggestion {
  name: string;
  label: string;
}

/** The options for what's typed: an exact match or Add first, then the guesses and known names that contain it. At
 *  rest the guesses come first. */
export function nameOptions(
  known: readonly string[],
  typed: string,
  suggestions: readonly NameSuggestion[] = [],
  allowNew = true,
): { value: string; label: string }[] {
  const names = [...new Set([...suggestions.map(s => s.name), ...known])];
  const q = typed.trim();
  const fold = q.toLocaleLowerCase();
  const exact = names.find(n => n.toLocaleLowerCase() === fold);
  const option = (name: string) => ({ value: name, label: suggestions.find(s => s.name === name)?.label ?? name });
  const first = exact ? [option(exact)] : q && allowNew ? [{ value: `${ADD}${q}`, label: `Add “${q}”` }] : [];
  const rest = names.filter(n => n !== exact && n.toLocaleLowerCase().includes(fold)).map(option);
  return [...first, ...rest];
}

export function PersonNameInput({
  known,
  suggestions = [],
  value,
  onChange,
  onPick,
  onCancel,
  allowNew = true,
  commitOnBlur = false,
  autoFocus = false,
  modalFocus = false,
  label,
  description,
  ariaLabel,
  placeholder = "Who's this?",
  emptyText = 'Type a name',
  disabled,
  size = 'xs',
  maxLength = 80,
  inputClassName,
  w,
  miw,
  flex,
}: {
  known: readonly string[];
  suggestions?: readonly NameSuggestion[];
  /** What's typed. */
  value: string;
  onChange: (typed: string) => void;
  /** A name was chosen: a known one, or a new one where allowed. */
  onPick: (name: string) => void;
  /** Escape. */
  onCancel?: () => void;
  /** Whether a name nobody has can be picked ("Add “…”"). */
  allowNew?: boolean;
  /** Leaving the box picks what's typed (the transcript's name box). */
  commitOnBlur?: boolean;
  /** Focus the box and select its text as it mounts. */
  autoFocus?: boolean;
  /** The box a Modal focuses as it opens (Mantine's data-autofocus). */
  modalFocus?: boolean;
  label?: ReactNode;
  description?: ReactNode;
  ariaLabel?: string;
  placeholder?: string;
  /** What the open list says with nothing to offer. */
  emptyText?: string;
  disabled?: boolean;
  size?: 'xs' | 'sm' | 'md';
  maxLength?: number;
  inputClassName?: string;
  w?: string | number;
  miw?: string | number;
  flex?: number;
}) {
  const input = useRef<HTMLInputElement>(null);
  // The first option is the one Enter takes: what's typed (an exact match, or Add) while typing, else the top guess.
  const lead = () => !!value.trim() || suggestions.length > 0;
  const box = useCombobox({ onDropdownClose: () => box.resetSelectedOption() });
  const options = nameOptions(known, value, suggestions, allowNew);
  const optionsKey = options.map(o => o.value).join('\n');
  // Options render only while the list is open (below), so opening highlights the first once they're there.
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-highlight when the options change or the list opens
  useEffect(() => {
    if (box.dropdownOpened && lead()) box.selectFirstOption();
  }, [optionsKey, box.dropdownOpened]);
  useEffect(() => {
    if (!autoFocus) return;
    input.current?.focus();
    input.current?.select();
  }, [autoFocus]);
  const pick = (v: string) => {
    box.closeDropdown();
    const name = v.startsWith(ADD) ? v.slice(ADD.length) : v;
    onPick(name);
  };
  const key = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === 'Escape') {
      e.stopPropagation();
      box.closeDropdown();
      onCancel?.();
    } else if (e.key === 'Enter' && box.getSelectedOptionIndex() < 0) {
      // Nothing highlighted: Enter takes what's typed, when a new name is allowed; Mantine submits a highlighted one.
      e.preventDefault();
      e.stopPropagation();
      const typed = value.trim();
      if (typed && (allowNew || known.some(k => k.toLocaleLowerCase() === typed.toLocaleLowerCase()))) pick(typed);
    }
  };
  return (
    <Combobox store={box} onOptionSubmit={pick} size={size} disabled={disabled}>
      {/* aria-expanded closed as well as open: a combobox needs it (axe) */}
      <Combobox.Target withExpandedAttribute>
        <TextInput
          ref={input}
          role="combobox"
          size={size}
          label={label}
          description={description}
          aria-label={ariaLabel}
          placeholder={placeholder}
          value={value}
          maxLength={maxLength}
          disabled={disabled}
          classNames={inputClassName ? { input: inputClassName } : undefined}
          w={w}
          miw={miw}
          flex={flex}
          autoComplete="off"
          data-autofocus={modalFocus || undefined}
          onChange={e => {
            onChange(e.currentTarget.value);
            box.openDropdown();
          }}
          onClick={() => box.openDropdown()}
          onFocus={() => box.openDropdown()}
          onBlur={() => {
            box.closeDropdown();
            if (commitOnBlur) onPick(value.trim());
          }}
          onKeyDown={key}
        />
      </Combobox.Target>
      <Combobox.Dropdown hidden={!options.length && !value.trim()}>
        {/* Mantine keeps a closed Combobox's dropdown mounted; rendering its options only while open keeps a page of
            name boxes (Who's this? has one per row) from holding every known name once per box (#3214). */}
        <Combobox.Options>
          {!box.dropdownOpened ? null : options.length ? (
            options.map(o => (
              <Combobox.Option key={o.value} value={o.value}>
                {o.label}
              </Combobox.Option>
            ))
          ) : (
            <Combobox.Empty>{emptyText}</Combobox.Empty>
          )}
        </Combobox.Options>
      </Combobox.Dropdown>
    </Combobox>
  );
}
