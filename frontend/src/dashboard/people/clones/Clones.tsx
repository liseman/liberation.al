// Settings' Voice clones panel (#2455; People's sidebar before): pick any voice heard, named or not, hear or build its
// reference, have it say something, or record a new voice. A person's own clone is on their card. The status line keeps its height so building and speaking don't move the
// page. Ported from Clones in static/dashboard/voice-tools.js.
import { Box, Button, Group, Radio, Stack, Text, Textarea, Tooltip } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, type JSX, useId, useState } from 'react';
import { HttpError, messageOf } from '../../../shared/http';
import { confirm } from '../../lib/confirm';
import { plural } from '../../lib/format';
import { isErr, type Message, messageWords } from '../../lib/message';
import { LoadError, PageLoading } from '../../lib/page';
import { play } from '../../lib/player';
import { spkColor } from '../../lib/speakers';
import { CLONES_KEY, cloneAudio, getClones, sayClone } from '../api';
import classes from './clones.module.css';
import { Recorder } from './Recorder';
import { build, CLONE_TEXT_ID, remove, useClones } from './store';

export function Clones(): JSX.Element {
  const client = useQueryClient();
  const groupName = useId(); // the radio cards' shared name
  const query = useQuery({ queryKey: CLONES_KEY, queryFn: getClones, refetchInterval: 120000 });
  const { sel, busy: working, pick } = useClones();
  const [text, setText] = useState('');
  const [st, setSt] = useState<Message | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [rec, setRec] = useState(false);
  const list = query.data?.clones ?? [];
  const error = query.data?.error || (query.isError ? messageOf(query.error) : null);
  const cur = list.find(c => c.id === sel);
  const busy = speaking || working != null;
  const ready = list.filter(c => c.ready);

  const reference = async () => {
    if (sel == null) return;
    if (cur?.built) return play(cloneAudio(sel), `ref:${sel}`);
    setSt({ text: 'Building the reference clip' });
    const j = await build(client, sel);
    setSt(
      j.ok
        ? { text: `Reference built from ${plural(j.utterances ?? 0, 'line', 'lines')}, ${j.seconds ?? 0} s` }
        : { err: j.error },
    );
  };

  const say = async (e: FormEvent) => {
    e.preventDefault();
    const t = text.trim();
    if (!t || sel == null) return;
    setSpeaking(true);
    const t0 = Date.now();
    setSt({ text: cur?.built ? 'Speaking' : 'Speaking (the first time takes 5 to 20 s)' });
    try {
      const j = (await sayClone(sel, t)) as { ok: boolean; url: string; error?: string };
      if (!j.ok) setSt({ err: j.error || 'Speaking failed.' });
      else {
        setSt({ text: `Spoken in ${((Date.now() - t0) / 1000).toFixed(1)} s` });
        play(j.url, `say:${Date.now()}`);
        void client.invalidateQueries({ queryKey: CLONES_KEY });
      }
    } catch (err) {
      setSt({
        err: err instanceof HttpError && err.reason ? err.reason : `Speaking failed: ${messageOf(err)}`,
      });
    }
    setSpeaking(false);
  };

  const removeClone = async () => {
    if (sel == null || !cur?.built) return;
    const sure = await confirm({
      title: 'Remove clone',
      body: `Remove the cloned voice for ${cur.label}? The speaker and their lines stay; the clone can be rebuilt later.`,
      confirm: 'Remove',
      danger: true,
    });
    if (!sure) return;
    const j = await remove(client, sel);
    setSt(j.ok ? { text: `Clone removed (${plural((j.removed ?? []).length, 'file', 'files')})` } : { err: j.error });
  };

  // The first list and its controls arrive together; cached controls stay mounted during refreshes.
  if (!query.data) {
    if (query.isError)
      return (
        <LoadError
          message="Couldn't load voice clones"
          detail={messageOf(query.error)}
          onRetry={() => void query.refetch()}
          busy={query.isFetching}
        />
      );
    return <PageLoading label="voice clones" shape="rows" lines={3} />;
  }

  const line: Message = st ?? (error ? { err: error } : null);
  const placeholder =
    sel == null ? (ready.length ? 'Pick a voice first' : 'Text to say') : `What should ${cur?.label} say?`;
  return (
    <Box className={classes.clones}>
      {list.length ? (
        <div className={classes.pick} role="radiogroup" aria-label="Voice to clone">
          {list.map(c => (
            <Tooltip
              key={c.id}
              label={
                c.built
                  ? 'Reference clip ready'
                  : `${Math.round(c.clean_s ?? 0)} s heard, ${Math.round(c.usable_s ?? 0)} s usable for cloning, needs ${c.needs_s ?? 12} s`
              }
              events={{ hover: true, focus: true, touch: false }}
            >
              <span style={{ display: 'inline-flex' }}>
                <Radio.Card
                  name={groupName}
                  checked={c.id === sel}
                  disabled={!c.ready}
                  className={classes.clone}
                  style={{ '--who': spkColor(c.id) }}
                  onClick={() => pick(c.id)}
                >
                  {c.label}
                  <span>
                    {working === c.id
                      ? 'building'
                      : c.built
                        ? 'cloned'
                        : c.ready
                          ? `${Math.round(c.usable_s ?? 0)} of ${Math.round(c.clean_s ?? 0)} s usable`
                          : 'not enough usable speech yet'}
                  </span>
                </Radio.Card>
              </span>
            </Tooltip>
          ))}
        </div>
      ) : (
        <Text size="sm" mb={12}>
          No voices heard yet. Talk near the microphone for a while, or record a voice.
        </Text>
      )}
      <Stack component="form" gap={6} onSubmit={say}>
        <Textarea
          id={CLONE_TEXT_ID}
          aria-label="Text to say"
          placeholder={placeholder}
          classNames={{ input: classes.text }}
          rows={3}
          resize="vertical"
          mb={2}
          value={text}
          onChange={e => setText(e.currentTarget.value)}
        />
        <Group gap={6}>
          <Button type="submit" size="xs" disabled={sel == null || busy || !text.trim()}>
            Say it
          </Button>
          <Button type="button" variant="default" size="xs" disabled={sel == null || busy} onClick={reference}>
            {cur?.built ? 'Hear the reference' : 'Build the reference'}
          </Button>
          {cur?.built && (
            <Tooltip
              label="Delete this voice's clone reference and renders"
              events={{ hover: true, focus: true, touch: false }}
            >
              <span style={{ display: 'inline-flex' }}>
                <Button
                  type="button"
                  variant="subtle"
                  c="var(--lib-bad)"
                  size="xs"
                  disabled={busy}
                  onClick={removeClone}
                >
                  Remove clone
                </Button>
              </span>
            </Tooltip>
          )}
        </Group>
        <Group gap={6}>
          <Button type="button" variant="default" size="xs" onClick={() => setRec(true)}>
            Record a new voice
          </Button>
        </Group>
      </Stack>
      <Text size="sm" className={classes.status} c={isErr(line) ? 'var(--lib-bad)' : 'dimmed'} role="status">
        {messageWords(line) ?? ''}
      </Text>
      <Recorder opened={rec} onClose={() => setRec(false)} />
    </Box>
  );
}
