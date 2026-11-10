// Controls for settings a text box would serve badly: the assistant's voice, the time zone, and the owner's
// voice reference. Row.tsx draws one of these in a setting's section instead of its plain control; Advanced still
// shows each setting's raw value.
import { ActionIcon, Button, Group, Modal, Select, Text, TextInput } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { type ComponentType, useState } from 'react';
import { request } from '../../shared/http';
import { play, stop, usePlayer } from '../lib/player';
import { useDraft } from './draft';
import type { Setting } from './model';
import { ServicesPick } from './services/ServicesPick';
import classes from './settings.module.css';
import { VOICE_REF_KEY, VoiceRef } from './VoiceRef';

export interface ControlProps {
  row: Setting;
  id: string; // what a link to the setting focuses
  labelled: string; // the label's id
  disabled: boolean;
  error?: string; // why a save refused it
  inputClass?: string; // the changed mark, as plain controls get it
}

/** A setting's value as free text: where a list to pick from can't be had (an old browser, no voices file). */
function Free({ row, id, labelled, disabled, error, inputClass }: ControlProps) {
  const d = useDraft();
  return (
    <TextInput
      id={id}
      aria-labelledby={labelled}
      value={String(d.value(row.key) ?? '')}
      onChange={e => d.set(row.key, e.currentTarget.value)}
      disabled={disabled}
      error={error}
      classNames={{ input: inputClass }}
    />
  );
}

// ---- the assistant's voice ----
export const VOICES_KEY = ['config', 'voices'] as const;
interface Voices {
  clone: boolean; // speaks in the owner's cloned voice on this machine
  kokoro: string[]; // Kokoro's voices, for a machine that can't clone one
}

const ACCENTS: Record<string, string> = {
  a: 'American English',
  b: 'British English',
  e: 'Spanish',
  f: 'French',
  h: 'Hindi',
  i: 'Italian',
  j: 'Japanese',
  p: 'Brazilian Portuguese',
  z: 'Mandarin',
};

/** A Kokoro voice's name in words: "af_heart" is "Heart (American English, female)". */
export function voiceName(id: string): string {
  const m = /^([a-z])([fm])_(.+)$/.exec(id);
  if (!m) return id;
  const [, accent = '', sex, name = ''] = m;
  const words = [ACCENTS[accent], sex === 'f' ? 'female' : 'male'].filter(Boolean).join(', ');
  return `${name.charAt(0).toUpperCase()}${name.slice(1).replace(/_/g, ' ')} (${words})`;
}

function AssistantVoice(props: ControlProps) {
  const { row, id, labelled, disabled, error, inputClass } = props;
  const d = useDraft();
  const voices = useQuery({
    queryKey: VOICES_KEY,
    queryFn: ({ signal }) => request<Voices>('/config/voices', undefined, signal),
  });
  // With a cloned voice it speaks in that, and this is the voice for when it can't: the picker still shows, with
  // that said under it. It used to show "Your cloned voice" in the Fallback voice row, which isn't the fallback.
  const fallbackOnly = !!voices.data?.clone;
  const value = String(d.value(row.key) ?? '');
  const ids = voices.data?.kokoro ?? [];
  if (!voices.isPending && !ids.length) return <Free {...props} />; // no voices file here, or it couldn't be read
  const data = (ids.includes(value) || !value ? ids : [value, ...ids]).map(v => ({ value: v, label: voiceName(v) }));
  return (
    <Select
      id={id}
      aria-labelledby={labelled}
      data={data}
      value={value}
      onChange={v => v && d.set(row.key, v)}
      searchable
      allowDeselect={false}
      error={error}
      classNames={{ input: inputClass }}
      disabled={disabled || voices.isPending}
      placeholder={voices.isPending ? 'Loading voices' : 'Pick a voice'}
      nothingFoundMessage="No voice by that name"
      // Its line is held while the voices load, so the note arriving moves nothing (a shift of 0.0012 before)
      description={
        fallbackOnly
          ? 'It speaks in your cloned voice. This one is for when it can’t.'
          : voices.isPending
            ? '\u00a0'
            : undefined
      }
      inputWrapperOrder={['input', 'description', 'error']}
    />
  );
}

// ---- the time zone ----
/** Every IANA zone the browser knows, with UTC; [] on a browser that can't list them. */
export const zones = (): string[] => {
  try {
    const all = Intl.supportedValuesOf('timeZone');
    return all.includes('UTC') ? all : ['UTC', ...all];
  } catch {
    return [];
  }
};

// Zones the browser lists by an older name (both work, the server reads either): the current one, to find them by.
const NOW_CALLED: Record<string, string> = {
  'Asia/Calcutta': 'Kolkata',
  'Europe/Kiev': 'Kyiv',
  'Asia/Saigon': 'Ho Chi Minh',
  'Asia/Katmandu': 'Kathmandu',
  'Asia/Rangoon': 'Yangon',
  'Atlantic/Faeroe': 'Faroe',
};

/** A zone in words: underscores as spaces, and its current name beside an older one. */
export const zoneLabel = (z: string): string => `${z.replace(/_/g, ' ')}${NOW_CALLED[z] ? ` (${NOW_CALLED[z]})` : ''}`;

function TimeZone(props: ControlProps) {
  const { row, id, labelled, disabled, error, inputClass } = props;
  const d = useDraft();
  const value = String(d.value(row.key) ?? '');
  const here = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const all = zones();
  if (!all.length) return <Free {...props} />;
  const data = [
    { value: '', label: `This machine's (${here})` },
    ...(value && !all.includes(value) ? [value] : []).map(v => ({ value: v, label: v })),
    ...all.map(z => ({ value: z, label: zoneLabel(z) })),
  ];
  return (
    <Select
      id={id}
      aria-labelledby={labelled}
      data={data}
      value={value}
      onChange={v => v != null && d.set(row.key, v)}
      searchable
      allowDeselect={false}
      error={error}
      classNames={{ input: inputClass }}
      disabled={disabled}
      nothingFoundMessage="No time zone by that name"
    />
  );
}

// ---- the owner's voice reference ----
interface Reference {
  exists: boolean;
  seconds: number;
  recorded?: number;
  problem?: 'missing' | 'unreadable' | null; // the voice can't clone from it
}
const SAVED_KEY = 'voice-ref';

/** "Recorded Oct 23, 15 s", or that there is none yet. */
export function recordedLine(ref: Reference | undefined, now = new Date(), failed = false): string {
  if (!ref) return failed ? "Couldn't load your voice reference" : 'Loading';
  if (ref.problem === 'unreadable') return "The saved reference can't be read. Record it again";
  if (!ref.exists) return 'Not recorded yet';
  const when = ref.recorded ? new Date(ref.recorded * 1000) : null;
  const day = when?.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    ...(when.getFullYear() !== now.getFullYear() ? { year: 'numeric' as const } : {}),
  });
  return `${day ? `Recorded ${day}, ` : ''}${Math.round(ref.seconds)} s`;
}

function VoiceReference({ id, labelled, disabled }: ControlProps) {
  const reference = useQuery({
    queryKey: VOICE_REF_KEY,
    queryFn: ({ signal }) => request<Reference>('/config/voice_ref', undefined, signal),
  });
  const [open, setOpen] = useState(false);
  const playing = usePlayer(s => s.key === SAVED_KEY);
  const exists = !!reference.data?.exists;
  return (
    // Not a field: a link to it (Getting started's "Record now") scrolls to and focuses it
    <Group id={id} tabIndex={-1} gap="xs" className={classes.control}>
      <ActionIcon
        size="lg"
        radius="xl"
        variant="light"
        disabled={!exists}
        aria-label={playing ? 'Stop your voice reference' : 'Play your voice reference'}
        onClick={() => (playing ? stop() : play(`/config/voice_ref?audio=1&t=${Date.now()}`, SAVED_KEY))}
      >
        {playing ? '■' : '▶'}
      </ActionIcon>
      <Text size="sm" c="dimmed" style={{ flexGrow: 1 }}>
        {recordedLine(reference.data, undefined, reference.isError)}
      </Text>
      {reference.isError && (
        <Button variant="subtle" size="xs" onClick={() => void reference.refetch()}>
          Try again
        </Button>
      )}
      <Button variant="default" size="xs" disabled={disabled} onClick={() => setOpen(true)} aria-describedby={labelled}>
        {exists ? 'Record again' : 'Record'}
      </Button>
      <Modal opened={open} onClose={() => setOpen(false)} title="Record your voice" size="lg">
        <VoiceRef readOnly={disabled} bare />
      </Modal>
    </Group>
  );
}

/** The controls Row.tsx draws in place of a setting's plain one, by its key, in the section that shows it. */
export const CONTROLS: Record<string, ComponentType<ControlProps>> = {
  'tts.voice': AssistantVoice,
  'time.timezone': TimeZone,
  'owner.voice_ref': VoiceReference,
  'services.enable': ServicesPick,
  'services.disable': ServicesPick,
};
