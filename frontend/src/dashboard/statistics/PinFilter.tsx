// The filter on a report that shows one person or one camera at a time (Who's around, Activity by person, Walk-bys
// by camera): all of them or a pinned one in one row, any other from a list, and Pin or Unpin for the one shown. The
// pins are saved with the page's layout (dashboard/reports.py FILTERED).
import { Button, Group, SegmentedControl, Select, Stack, Text } from '@mantine/core';
import { useId } from 'react';
import classes from './reports.module.css';

/** Most choices a page can pin, as the layout API allows (dashboard/reports.py MAX_PEOPLE). */
const MAX_PINS = 20;
const ALL = 'all';

/** What the filter calls its choices: "Everyone" or "Every camera", "Someone else" or "Another camera", and one and
 *  several ("person", "people"), for the line about pins not seen in this time. */
export interface PinWords {
  all: string;
  other: string;
  one: string;
  many: string;
}

export const PEOPLE: PinWords = { all: 'Everyone', other: 'Someone else', one: 'person', many: 'people' };
export const CAMERAS: PinWords = { all: 'Every camera', other: 'Another camera', one: 'camera', many: 'cameras' };

/**
 * Which one the report shows. The parts keep their places whatever is pinned, so the button just pressed keeps
 * keyboard focus. A pin not among the choices now (merged, deleted, or not seen in this time) gets a line of
 * its own to clear it.
 */
export function PinFilter({
  choices,
  pinned,
  value,
  onChange,
  onPinned,
  words = PEOPLE,
}: {
  choices: readonly { id: string; name: string }[];
  pinned: readonly string[];
  value: string | null;
  onChange: (value: string | null) => void;
  onPinned?: (pins: string[]) => void;
  words?: PinWords;
}) {
  const id = useId();
  const names = new Map(choices.map(c => [c.id, c.name]));
  const saved = [...new Set(pinned)]; // a repeat would be two choices with one key
  const pins = saved.filter(p => names.has(p));
  const away = saved.filter(p => !names.has(p));
  const others = choices.filter(c => !pins.includes(c.id));
  const shown = value ? names.get(value) : undefined;
  const isPinned = !!value && pins.includes(value);
  const canPin = isPinned || saved.length < MAX_PINS;
  return (
    <Stack gap={4} miw={0} maw="100%">
      <Text id={id} size="xs" fw={500}>
        Show
      </Text>
      <Group gap="xs" align="center" wrap="wrap">
        <div className={classes.pins}>
          <SegmentedControl
            aria-labelledby={id}
            size="xs"
            value={value ?? ALL}
            onChange={v => onChange(v === ALL ? null : v)}
            data={[{ value: ALL, label: words.all }, ...pins.map(p => ({ value: p, label: names.get(p) ?? p }))]}
          />
        </div>
        {others.length > 0 && (
          <Select
            aria-label={words.other}
            placeholder={words.other}
            size="xs"
            w={160}
            clearable
            value={value && !isPinned ? value : null}
            onChange={v => onChange(v || null)}
            data={others.map(c => ({ value: c.id, label: c.name }))}
          />
        )}
        {onPinned && value && shown && canPin && (
          <Button
            variant="subtle"
            size="xs"
            onClick={() => onPinned(isPinned ? saved.filter(p => p !== value) : [...saved, value])}
          >
            {isPinned ? `Unpin ${shown}` : `Pin ${shown}`}
          </Button>
        )}
      </Group>
      {onPinned && away.length > 0 && (
        <Group gap={4}>
          <Text size="xs" c="dimmed">
            {away.length === 1 ? `1 pinned ${words.one} wasn't` : `${away.length} pinned ${words.many} weren't`} seen in
            this time.
          </Text>
          <Button variant="subtle" size="compact-xs" onClick={() => onPinned(saved.filter(p => names.has(p)))}>
            Unpin {away.length === 1 ? 'them' : 'all of them'}
          </Button>
        </Group>
      )}
    </Stack>
  );
}
