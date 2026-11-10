// Recording a new voice from the browser's microphone: at least 10 s of speech, saved under a name so Liberational
// learns the voice and can clone it. Ported from Recorder in static/dashboard/voice-tools.js.
import { Button, Group, Modal, Text, TextInput } from '@mantine/core';
import { useQueryClient } from '@tanstack/react-query';
import { type FormEvent, useState } from 'react';
import { messageOf, request } from '../../../shared/http';
import { useMicRecorder } from '../../lib/mic';
import { liveKey } from '../../live/feed';
import { CLONES_KEY } from '../api';
import classes from './clones.module.css';

const MIN_S = 10;
const MAX_S = 30;
const MIN_VOICED_S = 3; // the server refuses less, as it is not enough to learn a voice from

interface Saved {
  ok: boolean;
  name: string;
  ref_text?: string;
  error?: string;
}

function Body({ name: whose = '' }: { name?: string }) {
  const client = useQueryClient();
  const mic = useMicRecorder(MAX_S);
  const [name, setName] = useState(whose);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState('');
  const done = mic.phase === 'done';
  const short = done && mic.seconds < MIN_S;
  const silent = done && !short && mic.voiced < MIN_VOICED_S;
  const wav = done && !short && !silent ? mic.wav : null;
  const recording = mic.phase === 'rec';
  const status =
    saved ||
    mic.error ||
    (recording
      ? 'Recording. Press Stop when you are done.'
      : short
        ? `That was ${mic.seconds.toFixed(1)} s. Record at least ${MIN_S} seconds.`
        : silent
          ? `The microphone picked up almost nothing (${mic.voiced.toFixed(1)} s of sound). Check the input level, or speak closer, and record again.`
          : done
            ? `${mic.seconds.toFixed(1)} s recorded. Name the voice and save it.`
            : '');

  const start = () => {
    setSaved('');
    void mic.start();
  };

  const save = async (e: FormEvent) => {
    e.preventDefault();
    const nm = name.trim();
    if (!nm || !wav) return;
    setBusy(true);
    setSaved('Saving');
    try {
      const j = await request<Saved>(`/clones/record?name=${encodeURIComponent(nm)}`, wav);
      if (!j.ok) {
        setBusy(false);
        setSaved(`Saving failed: ${j.error || 'unknown error'}`);
        return;
      }
      setSaved(`Saved as ${j.name}${j.ref_text ? `. Heard: “${j.ref_text.slice(0, 120)}”` : '.'}`);
      void client.invalidateQueries({ queryKey: CLONES_KEY });
      void client.invalidateQueries({ queryKey: liveKey('speakers') });
    } catch (err) {
      setBusy(false);
      setSaved(`Saving failed: ${messageOf(err)}`);
    }
  };

  return (
    <>
      <Text c="dimmed">
        Speak for at least {MIN_S} seconds in a quiet room (it stops by itself at {MAX_S}). Liberational learns the
        voice from this recording and can clone it.
      </Text>
      <Group gap={14} my={14}>
        <Button
          variant="default"
          c={recording ? 'var(--lib-live)' : undefined}
          aria-pressed={recording}
          onClick={recording ? mic.stop : start}
        >
          {recording ? 'Stop' : done ? 'Record again' : 'Start recording'}
        </Button>
        {recording && (
          <span className={classes.time} aria-live="off">
            {mic.secs.toFixed(0)} s
          </span>
        )}
      </Group>
      {status && (
        <Text size="sm" role="status" mb="sm">
          {status}
        </Text>
      )}
      <Group component="form" gap="xs" onSubmit={save}>
        <TextInput
          aria-label="Name for this voice"
          placeholder="Whose voice is this?"
          style={{ flex: 1 }}
          miw={160}
          value={name}
          onChange={e => setName(e.currentTarget.value)}
        />
        <Button type="submit" disabled={busy || !wav || !name.trim()}>
          Save voice
        </Button>
      </Group>
    </>
  );
}

/** The "Record a new voice" dialog. Closing it lets go of the microphone (the body unmounts). With a name (from a
 * person's card), the recording is saved as theirs unless the name is changed. */
export function Recorder({ opened, onClose, name }: { opened: boolean; onClose: () => void; name?: string }) {
  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={name ? `Record a new reference for ${name}` : 'Record a new voice'}
      size={520}
    >
      <Body name={name} />
    </Modal>
  );
}
