// How Live looks in this browser: a modal of choices that apply as you pick, opened from the top bar's menu
// (ShellMenu.tsx), which also holds the theme.
import { Button, Group, Modal, SegmentedControl, Stack, Text } from '@mantine/core';
import { type LivePrefs, setPref, usePrefs } from '../lib/prefs';
import { shell } from '../shell';

type Look = { key: keyof LivePrefs; label: string; def: string | boolean; opts: [string | boolean, string][] };

const looks = (): Look[] => [
  {
    key: 'transcript',
    label: 'Transcript',
    def: 'lines',
    opts: [
      ['lines', 'Lines'],
      ['balloons', 'Word balloons'],
    ],
  },
  {
    key: 'waveform',
    label: 'Live waveform',
    def: false,
    opts: [
      [true, 'On'],
      [false, 'Off'],
    ],
  },
  {
    key: 'density',
    label: 'Density',
    def: 'comfortable',
    opts: [
      ['comfortable', 'Comfortable'],
      ['compact', 'Compact'],
    ],
  },
  {
    key: 'timestamps',
    label: 'Timestamps',
    def: 'hover',
    opts: [
      ['hover', 'On hover'],
      ['always', 'Always'],
    ],
  },
  {
    key: 'statusLine',
    label: `${shell().assistant}'s status line`,
    def: true,
    opts: [
      [true, 'Show'],
      [false, 'Hide'],
    ],
  },
];

export function LiveDisplay({ opened, onClose }: { opened: boolean; onClose: () => void }) {
  const prefs = usePrefs();
  return (
    <Modal opened={opened} onClose={onClose} title="Live display" size={460} returnFocus>
      <Text size="sm" c="dimmed" mb="sm">
        How Live looks in this browser. Changes apply as you pick them.
      </Text>
      <Stack gap="sm">
        {looks().map(l => {
          const current = prefs[l.key] === undefined ? l.def : prefs[l.key];
          return (
            <Group key={l.key} justify="space-between" wrap="wrap" gap="xs">
              <Text fw={500} id={`look-${l.key}`}>
                {l.label}
              </Text>
              <SegmentedControl
                size="xs"
                radius="xl"
                aria-labelledby={`look-${l.key}`}
                value={String(current)}
                onChange={v => setPref(l.key, (l.opts.find(([o]) => String(o) === v)?.[0] ?? l.def) as never)}
                data={l.opts.map(([v, label]) => ({ value: String(v), label }))}
              />
            </Group>
          );
        })}
      </Stack>
      <Group justify="flex-end" mt="md">
        <Button variant="default" size="compact-sm" onClick={onClose}>
          Close
        </Button>
      </Group>
    </Modal>
  );
}
