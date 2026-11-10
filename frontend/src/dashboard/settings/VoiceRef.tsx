// Owner voice-reference capture. Audio stays a local draft until the viewer explicitly saves it.
import { Blockquote, Button, Code, Group, Stack, Text } from '@mantine/core';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { messageOf, request } from '../../shared/http';
import { useMicRecorder } from '../lib/mic';
import { PageError, PageLoading } from '../lib/page';
import { play, stop, usePlayer } from '../lib/player';
import { SettingsRow, SettingsSection } from './layout';
import { VOICE_REF_KEY } from './model';

export { VOICE_REF_KEY };

interface Reference {
  path: string;
  exists: boolean;
  seconds: number;
  prompt: string;
  problem?: 'missing' | 'unreadable' | null; // the voice can't clone from it (#301)
}
interface Saved {
  ok: boolean;
  seconds?: number;
  note?: string;
  error?: string;
}
const DRAFT_KEY = 'voice-ref-take';
const SAVED_KEY = 'voice-ref';
const ownPlayback = () => [DRAFT_KEY, SAVED_KEY].includes(usePlayer.getState().key ?? '');

/** bare: in Settings' recording dialog (#383), where the row above already says what is saved and plays it. */
export function VoiceRef({ readOnly, bare = false }: { readOnly: boolean; bare?: boolean }) {
  const client = useQueryClient();
  const reference = useQuery({
    queryKey: VOICE_REF_KEY,
    queryFn: ({ signal }) => request<Reference>('/config/voice_ref', undefined, signal),
  });
  const mic = useMicRecorder(40);
  const playerKey = usePlayer(s => s.key);
  const [draftUrl, setDraftUrl] = useState<string | null>(null);
  const save = useMutation({
    mutationFn: async (wav: Blob) => {
      const response = await request<Saved>('/config/voice_ref', wav);
      if (!response.ok) throw new Error(response.error || 'Saving failed');
      return response;
    },
    // A new reference is what the cloned voice is made from: the voice choice reads it too (controls.tsx)
    onSuccess: () => client.invalidateQueries({ queryKey: ['config'] }),
  });
  useEffect(() => {
    if (!mic.wav) {
      setDraftUrl(null);
      return;
    }
    const url = URL.createObjectURL(mic.wav);
    setDraftUrl(url);
    return () => {
      if (usePlayer.getState().key === DRAFT_KEY) stop();
      URL.revokeObjectURL(url);
    };
  }, [mic.wav]);
  useEffect(
    () => () => {
      if (ownPlayback()) stop();
    },
    [],
  );

  const recording = mic.phase === 'rec';
  const short = mic.phase === 'done' && mic.seconds < 8;
  const take = mic.phase === 'done' && !short ? mic.wav : null;
  const record = () => {
    save.reset();
    if (ownPlayback()) stop();
    void mic.start();
  };
  return (
    <SettingsSection aria-label="Voice reference">
      <SettingsRow>
        <Stack gap="sm">
          {reference.isPending && <PageLoading label="the voice reference" lines={3} />}
          {reference.isError && (
            <PageError
              title="Couldn't load the voice reference"
              detail={messageOf(reference.error)}
              onRetry={() => void reference.refetch()}
              busy={reference.isFetching}
            />
          )}
          {reference.data && (
            <>
              {!bare && (
                <Group gap="xs">
                  <Text size="sm">
                    {reference.data.exists ? (
                      <>
                        A {reference.data.seconds} s reference is saved at <Code>{reference.data.path}</Code>.
                      </>
                    ) : reference.data.problem === 'unreadable' ? (
                      <>
                        The reference at <Code>{reference.data.path}</Code> can't be read. Record it again.
                      </>
                    ) : (
                      'No reference recorded yet.'
                    )}
                  </Text>
                  {reference.data.exists && (
                    <Button
                      type="button"
                      variant="subtle"
                      size="compact-sm"
                      onClick={() => play(`/config/voice_ref?audio=1&t=${Date.now()}`, SAVED_KEY)}
                    >
                      {playerKey === SAVED_KEY ? 'Stop reference' : 'Play reference'}
                    </Button>
                  )}
                </Group>
              )}
              {!readOnly && (
                <>
                  <Text size="sm" c="dimmed">
                    To record a new one, press Record and read this aloud at your normal pace, in a quiet room:
                  </Text>
                  <Blockquote>{reference.data.prompt}</Blockquote>
                  <Group>
                    <Button
                      type="button"
                      variant="default"
                      color={recording ? 'red' : undefined}
                      aria-pressed={recording}
                      disabled={save.isPending}
                      onClick={recording ? mic.stop : record}
                    >
                      {recording ? `Stop (${mic.secs.toFixed(0)} s)` : mic.phase === 'done' ? 'Record again' : 'Record'}
                    </Button>
                    {take && (
                      <>
                        <Button
                          type="button"
                          variant="default"
                          disabled={!draftUrl}
                          onClick={() => {
                            if (draftUrl) play(draftUrl, DRAFT_KEY);
                          }}
                        >
                          {playerKey === DRAFT_KEY ? 'Stop preview' : 'Listen back'}
                        </Button>
                        <Button type="button" loading={save.isPending} onClick={() => save.mutate(take)}>
                          Save as my voice
                        </Button>
                      </>
                    )}
                  </Group>
                  {mic.error && <PageError title={mic.error}>{null}</PageError>}
                  {short && (
                    <PageError title={`That was ${mic.seconds.toFixed(1)} s`}>
                      Read the whole passage (at least 8 seconds).
                    </PageError>
                  )}
                  {save.isError && (
                    <PageError title="The recording wasn't saved" detail={messageOf(save.error)}>
                      Try again.
                    </PageError>
                  )}
                  {save.isSuccess && (
                    <Text role="status" size="sm">
                      Saved {save.data.seconds} s. {save.data.note}
                    </Text>
                  )}
                </>
              )}
            </>
          )}
        </Stack>
      </SettingsRow>
    </SettingsSection>
  );
}
