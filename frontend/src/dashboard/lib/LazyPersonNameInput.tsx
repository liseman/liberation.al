// The person name box (PersonNameInput, Mantine's Combobox), loaded the first time one opens (#2529): Live shows it
// only when a turn or a face is being named, so its code stays out of Live's first load. Until it's there, a plain
// box of the same size and label holds its place, so nothing moves when it arrives.
import { TextInput } from '@mantine/core';
import { type ComponentProps, lazy, Suspense } from 'react';

const Loaded = lazy(() => import('./PersonNameInput').then(m => ({ default: m.PersonNameInput })));

type Props = ComponentProps<typeof Loaded>;

/** PersonNameInput, loaded on first use. */
export function LazyPersonNameInput(props: Props) {
  // The loaded box's own defaults (size xs, "Who's this?"), so the stand-in is the same size and says the same.
  const {
    label,
    description,
    ariaLabel,
    placeholder = "Who's this?",
    value,
    size = 'xs',
    w,
    miw,
    flex,
    inputClassName,
    disabled,
  } = props;
  return (
    <Suspense
      fallback={
        <TextInput
          label={label}
          description={description}
          aria-label={ariaLabel}
          placeholder={placeholder}
          value={value}
          size={size}
          w={w}
          miw={miw}
          flex={flex}
          classNames={inputClassName ? { input: inputClassName } : undefined}
          disabled={disabled}
          readOnly
        />
      }
    >
      <Loaded {...props} />
    </Suspense>
  );
}
