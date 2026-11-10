// Review for one person: their visits of the day (their lines, the camera views that saw them, and the
// actions put down to them, merged when close), and the summary blocks that mention them. A visit or a block seeks
// there, on the camera that saw them most.
import { Button, Group, Stack, Text, Timeline, Title, UnstyledButton } from '@mantine/core';
import { IconEye, IconMessage, IconNotes } from '@tabler/icons-react';
import { useMemo } from 'react';
import { useStore } from 'zustand';
import { hm, hms } from '../lib/format';
import { EMPTY_SPEAKERS, speakerName } from '../lib/speakers';
import { useLive } from '../live/hooks';
import type { TranscriptEntry } from '../live/types';
import type { Block, DetectedActivity } from './api';
import { duration } from './chipStyle';
import { useEngine, useReview } from './context';
import {
  LEAD_S,
  type Moment,
  mentioning,
  mentions,
  momentsOf,
  playbackOf,
  type Visit,
  visitsOf,
} from './personMomentsModel';
import type { PersonPlayback } from './personPlayback';
import classes from './review.module.css';
import type { Chip } from './types';
import { viewName } from './views';

function momentLabel(m: Moment): string {
  if (m.kind === 'line') return m.label || 'Speech';
  return `${m.kind === 'seen' ? 'Seen' : m.label}${m.cam ? ` · ${viewName(m.cam)}` : ''}`;
}

function said(v: Visit): string {
  const parts = [];
  if (v.lines) parts.push(`said ${v.lines} ${v.lines === 1 ? 'line' : 'lines'}`);
  if (v.seen_s > 0) parts.push(`seen ${duration(v.seen_s)}${v.cams.length ? ` on ${v.cams.join(', ')}` : ''}`);
  if (v.actions.length) parts.push(v.actions.join(', '));
  const text = parts.join(' · ');
  return text.charAt(0).toLocaleUpperCase() + text.slice(1);
}

/** A block's text with the person's name marked wherever it's said. */
function Marked({ text, person }: { text: string; person: string }) {
  const found = mentions(text, person);
  if (!found.length) return <>{text}</>;
  const out = [];
  let at = 0;
  for (const [a, b] of found) {
    out.push(text.slice(at, a), <mark key={a}>{text.slice(a, b)}</mark>);
    at = b;
  }
  out.push(text.slice(at));
  return <>{out}</>;
}

export function PersonMoments({
  person,
  entries,
  chips,
  activities,
  blocks,
  playback,
  summaryState,
  onRetrySummary,
}: {
  person: string;
  entries: readonly TranscriptEntry[];
  chips: readonly Chip[];
  activities: readonly DetectedActivity[];
  blocks: readonly Block[];
  playback?: PersonPlayback;
  summaryState: 'loading' | 'error' | 'partial' | 'ready';
  onRetrySummary?: () => void;
}) {
  const engine = useEngine();
  const t = useReview(s => s.t);
  const speakers = useLive('speakers') ?? EMPTY_SPEAKERS;
  const moments = useMemo(
    () => momentsOf(person, entries, e => speakerName(speakers, e), chips, activities),
    [person, entries, speakers, chips, activities],
  );
  const visits = useMemo(() => visitsOf(moments), [moments]);
  const chipId = chips.find(c => c.kind === 'person' && c.label.toLocaleLowerCase() === person.toLocaleLowerCase())?.id;
  const said_ = useMemo(() => mentioning(blocks, person, chipId ?? null), [blocks, person, chipId]);
  const active =
    t == null
      ? -1
      : visits.findLastIndex(v => {
          // Point actions have a short video tail. Use the same bounded window as playback without extending the visit's evidence.
          const end = playbackOf(v.moments, 'video', engine.dayStart).reduce(
            (end, moment) => Math.max(end, moment.end),
            v.end,
          );
          return v.start - LEAD_S <= t && t < end;
        });
  const go = (v: Visit) => engine.seekActivity(Math.max(0, v.start - LEAD_S), v.cams[0] ?? '');

  return (
    <section className={classes.person} aria-labelledby="person-moments">
      <Title order={2} id="person-moments">
        {person}'s day
      </Title>
      {playback && (
        <PlaybackControls person={person} moments={moments} dayStart={engine.dayStart} playback={playback} />
      )}
      {visits.length ? (
        <Timeline role="list" active={active} bulletSize={16} lineWidth={2} aria-label={`${person}'s visits`}>
          {visits.map((v, i) => (
            <Timeline.Item
              key={v.start}
              role="listitem"
              bullet={v.lines && !v.seen_s ? <IconMessage size={11} aria-hidden /> : <IconEye size={11} aria-hidden />}
              title={
                <Button
                  variant="subtle"
                  size="compact-sm"
                  onClick={() => go(v)}
                  aria-current={i === active || undefined}
                >
                  {v.end - v.start >= 60 ? `${hm(v.start)} to ${hm(v.end)}` : hm(v.start)}
                </Button>
              }
            >
              <Text size="sm" c="dimmed">
                {said(v)}
              </Text>
              <Stack gap={2} mt={4}>
                {v.moments.map((m, j) => (
                  <UnstyledButton
                    // biome-ignore lint/suspicious/noArrayIndexKey: observations can share timestamps and have no unique ID
                    key={`${m.kind}:${m.start}:${m.cam}:${j}`}
                    className={classes.personMoment}
                    aria-label={`Show ${hms(m.start)}: ${momentLabel(m)}`}
                    onClick={() => engine.seekActivity(m.start, m.cam ?? '')}
                  >
                    <Text component="span" display="block" size="xs" c="dimmed">
                      {hms(m.start)}
                    </Text>
                    <Text component="span" display="block" size="sm">
                      {momentLabel(m)}
                    </Text>
                  </UnstyledButton>
                ))}
              </Stack>
            </Timeline.Item>
          ))}
        </Timeline>
      ) : (
        <Text size="sm" c="dimmed">
          Nothing of {person} this day: no lines, no camera views, and no actions.
        </Text>
      )}
      <Title order={3} mt="md">
        Mentions in the summary
      </Title>
      {summaryState === 'loading' && (
        <Text size="sm" c="dimmed" role="status">
          Loading summary mentions…
        </Text>
      )}
      {summaryState === 'error' && (
        <div>
          <Text size="sm" c="red" role="status">
            Summary unavailable. Mentions could not be checked.
          </Text>
          {onRetrySummary && (
            <Button variant="default" size="compact-sm" onClick={onRetrySummary}>
              Retry summary
            </Button>
          )}
        </div>
      )}
      {summaryState === 'partial' && (
        <Text size="sm" c="dimmed" role="status">
          The summary is still being written; mentions may be missing.
        </Text>
      )}
      {said_.length > 0 ? (
        <Timeline role="list" bulletSize={14} lineWidth={2} aria-label={`Summary mentions of ${person}`}>
          {said_.map(b => (
            <Timeline.Item key={b.start} role="listitem" bullet={<IconNotes size={11} aria-hidden />}>
              <Button variant="subtle" size="compact-xs" onClick={() => engine.seek(b.start)}>
                {hm(b.start)}
              </Button>
              <Text size="sm">
                <Marked text={b.text} person={person} />
              </Text>
            </Timeline.Item>
          ))}
        </Timeline>
      ) : summaryState === 'ready' ? (
        <Text size="sm" c="dimmed">
          The summary doesn't mention {person} this day.
        </Text>
      ) : null}
    </section>
  );
}

/** Controls use the queue snapshot, so a list refresh cannot change a sequence already playing. */
function PlaybackControls({
  person,
  moments,
  dayStart,
  playback,
}: {
  person: string;
  moments: readonly Moment[];
  dayStart: number;
  playback: PersonPlayback;
}) {
  const run = useStore(playback.ui, state => state.run);
  const audio = useMemo(() => playbackOf(moments, 'audio', dayStart), [moments, dayStart]);
  const video = useMemo(() => playbackOf(moments, 'video', dayStart), [moments, dayStart]);
  const active = run?.person === person ? run : null;
  return (
    <>
      <Group gap="xs" my="sm">
        <Button
          variant="default"
          disabled={!audio.length}
          aria-pressed={active?.mode === 'audio'}
          onClick={() => playback.start(person, 'audio', audio)}
        >
          Play {person}'s audio
        </Button>
        <Button
          variant="default"
          disabled={!video.length}
          aria-pressed={active?.mode === 'video'}
          onClick={() => playback.start(person, 'video', video)}
        >
          Play {person}'s video
        </Button>
        {active && (
          <Button variant="subtle" onClick={() => playback.stop()}>
            Stop person playback
          </Button>
        )}
      </Group>
      {active && (
        <Text size="sm" role="status">
          Playing {person}'s {active.mode}, moment {active.index + 1} of {active.moments.length}.
        </Text>
      )}
    </>
  );
}
