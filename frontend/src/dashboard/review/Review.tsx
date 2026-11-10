// Review: video and timeline beside transcript/summary tabs, with day-scoped Ask in a nonblocking side panel.
// The engine owns time and media while this page fetches and arranges the day.
import { Button, Tabs, Text, VisuallyHidden } from '@mantine/core';
import { IconX } from '@tabler/icons-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useLayoutEffect, useState } from 'react';
import { useStore } from 'zustand';
import { messageOf } from '../../shared/http';
import { useReviewAsk } from '../AskBar';
import { AskPanel } from '../ask/AskPanel';
import { PackAsk } from '../ask/PackAsk';
import { daysWords, isoDay } from '../lib/format';
import { useSlow } from '../lib/loading';
import { LoadError, PAGE_SLOW_MS } from '../lib/page';
import { rememberedHeight, useRememberHeight } from '../lib/rememberedHeight';
import { useRoute } from '../router';
import { shell } from '../shell';
import {
  activitiesQuery,
  chipsQuery,
  dayBlocksQuery,
  daysQuery,
  dayWorking,
  isDetectorStorage,
  isDetectorStoragePending,
  isStillComputing,
  notesQuery,
  reviewKey,
  segmentsQuery,
  spansQuery,
  transcriptQuery,
} from './api';
import { EngineContext } from './context';
import { DayPicker } from './DayPicker';
import { EntriesForm } from './EntriesForm';
import { EventForm } from './EventForm';
import { ReviewEngine } from './engine';
import { ChipsSkeleton } from './Loading';
import { PersonMoments } from './PersonMoments';
import { Player, SkipQuiet, VideoMarkers } from './Player';
import { ReviewSearch } from './ReviewSearch';
import classes from './review.module.css';
import { ShowOnly } from './ShowOnly';
import { SummaryBlocks } from './SummaryBlocks';
import { Timeline } from './Timeline';
import { Transcript } from './Transcript';
import type { EventSpan } from './types';
import { useReviewWindow } from './window';

const NOTHING = 'nothing';

/** How long the truncated-markers note must hold before it shows (the landing passes through the day before). */
const NOTE_HOLD_MS = 1500;

/** About a usual day's Show only height, held while the chips load on a first visit (#2485). */
const TIMELINE_FIRST_H = 'clamp(180px, 30vh, 270px)';

export function Review() {
  const client = useQueryClient();
  const [engine] = useState(
    () =>
      new ReviewEngine({
        cams: shell().cams,
        fetchNotes: b => client.fetchQuery(notesQuery(b)),
        reload: d => void client.invalidateQueries({ queryKey: reviewKey.day(d) }),
      }),
  );
  // A layout effect: hidden behind another tab (#330), its cleanup runs before React lets go of the <video>s and the
  // <audio>, so stop() can still pause them and release their sources.
  useLayoutEffect(() => {
    engine.start();
    return () => engine.stop();
  }, [engine]);
  // One ref callback for the page's life: an inline one is a new function each render, so React detached and
  // attached the <audio> on every render, and each attach forgot the room audio it held (loadedRoom), which stopped
  // keeping the sound in step with the picture until the next stretch.
  const audioRef = useCallback((el: HTMLAudioElement | null) => engine.setAudio(el), [engine]);

  const day = useStore(engine.ui, s => s.day);
  useStore(engine.ui, s => s.dataVersion); // aggregate chips change when any loaded day refreshes
  useStore(engine.ui, s => s.sel); // a chunk's chips publish after its query render
  const chunks = useStore(engine.ui, s => s.chunkStatus);
  const waiting = useStore(engine.ui, s => s.waiting);
  const recordingWindow = useReviewWindow(engine);
  const key = day ?? NOTHING;
  const on = { enabled: !!day };
  // The window hook owns polling; these observers supply only the playhead day's detail panels.
  const transcript = useQuery({ ...transcriptQuery(key), ...on });
  const summary = useQuery({ ...dayBlocksQuery(day ?? isoDay()), ...on });
  const [query, setQuery] = useState(''); // Review's search: its results above, the transcript dimmed to it (#2593)
  const searching = !!query;
  const segments = useQuery({ ...segmentsQuery(key), ...on });
  const spans = useQuery({ ...spansQuery(key), ...on });
  const sources = [transcript, segments, spans];
  // Today's video is listed again every few seconds (#806): only its first listing holds the day back.
  const settled = !!day && sources.every(s => !s.isPending && (!s.isFetching || s.data !== undefined));
  const chips = useQuery({ ...chipsQuery(key, false), refetchInterval: false, enabled: !!day });
  const activities = useQuery({ ...activitiesQuery(key), enabled: !!day });
  const days = useQuery(daysQuery());
  useEffect(() => {
    if (days.data)
      engine.recordedDays(
        days.data.days.map(d => d.day),
        days.data.today,
      );
  }, [engine, days.data]);

  // While the server works a day out again (#435), the chips Review had for it stay: they were right a moment ago.
  const working = dayWorking(chips.error);
  const notice = useStore(engine.ui, s => s.notice);
  // Chunk updates preserve neighboring chips and filters while the playhead crosses a date marker.
  const visibleChips = engine.chips;
  // Review opened for one person (#person=Dana, from their card's "See all in Review"): a removable filter, and a word
  // when this day has nothing of them.
  const person = useStore(engine.ui, s => s.person);
  // Opened for a person, the details open on their day (#2057); without one, on the transcript.
  const [tabPick, setTab] = useState<string | null>(null);
  const tab = tabPick === 'person' && !person ? 'transcript' : (tabPick ?? (person ? 'person' : 'transcript'));
  // biome-ignore lint/correctness/useExhaustiveDependencies: a newly picked person opens on their tab
  useEffect(() => setTab(null), [person]);
  const absent =
    !!person &&
    !!chips.data &&
    !chips.data.some(c => c.kind === 'person' && c.label.toLocaleLowerCase() === person.toLocaleLowerCase());
  const coveredBySettings = useRoute(s => s.tab === 'settings');
  const askOpen = useReviewAsk(s => s.open);
  const setAskOpen = useReviewAsk(s => s.setOpen);
  // A span picked on the timeline to mark as an event (#287, #518), while its form is open.
  const marking = useStore(engine.ui, s => s.eventSpan);
  // A span whose entries and exits are being labeled (#1472), in a drawer that leaves the player usable.
  const [entryWindow, setEntryWindow] = useState<EventSpan | null>(null);
  // The moments the newest answer cites, marked on the timeline.
  const [marks, setMarks] = useState<readonly number[]>([]);
  const today = !!day && day === isoDay();
  const dayName = today
    ? 'today'
    : day
      ? new Date(`${day}T12:00:00`).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })
      : '';

  const pick = (d: string) => {
    // Picking the day that's showing reloads it (Today, while today keeps growing).
    if (d === day) client.invalidateQueries({ queryKey: reviewKey.day(d) });
    engine.pickDay(d);
  };

  // The day's lines, video, and audio: loading on the first open and on every new day; then each either loaded or
  // failed. The chips load on their own (they can take a minute), and nothing else waits for them.
  const loading = !!day && !recordingWindow.initialized;
  const slowDay = useSlow(loading, PAGE_SLOW_MS);
  // The view passes through the day before while it lands on this one; a note about that day's markers would flash
  // for a moment and move everything under the player. It shows once it holds.
  const truncatedShown = useSlow(recordingWindow.truncatedDays.length > 0, NOTE_HOLD_MS);
  const loadingDays = Object.entries(chunks)
    .filter(([, status]) => status === 'loading')
    .map(([d]) => d);
  const failedDays = Object.entries(chunks)
    .filter(([, status]) => status === 'error')
    .map(([d]) => d);
  const failed = [
    transcript.isError && 'transcript',
    segments.isError && 'video list',
    spans.isError && 'audio list',
  ].filter(Boolean);
  // Retry the recording sources in place; detector analysis and summaries have their own recovery actions.
  const retryCore = (d: string) => {
    for (const queryKey of [reviewKey.transcript(d), reviewKey.segments(d), reviewKey.spans(d)]) {
      void client.invalidateQueries({ queryKey });
    }
  };
  const retry = () => {
    if (day) retryCore(day);
  };
  const allFailed = settled && failed.length === sources.length;
  // One line says the day is loading, under the skeletons; it says more only once the wait runs long (#2450).
  let status = slowDay ? 'Still loading the day. This can take a minute after a restart.' : 'Loading the day…';
  if (allFailed)
    status = ''; // the alert below says it
  // A loaded day says nothing here (#2577): the video and the transcript show what it holds.
  else if (settled) status = failed.length ? `Could not load the ${failed.join(' or ')}.` : '';

  // "Working through Oct 28" shows in the chips' place; here only when chips from before stand in for them
  if (chips.isError && (!working || chips.data)) status = `${status} ${messageOf(chips.error)}`.trim();
  if (absent && settled) status = `${status} Nothing of ${person} on ${day}.`.trim();

  // What stands under the player: Show only, its skeleton, or its load error. The video's boxes and Skip quiet time
  // live in Show only, so until it shows (or when the chips fail) they sit in a row of their own there (#2486).
  const chipsView: 'showOnly' | 'loading' | 'error' =
    chips.data && !isDetectorStorage(chips.error) && !(working && !chips.data.length)
      ? 'showOnly'
      : chips.isLoading ||
          working ||
          isDetectorStoragePending(chips.error) ||
          (isStillComputing(chips.error) && chips.isFetching)
        ? 'loading'
        : chips.isError
          ? 'error'
          : 'showOnly';
  const timelineRef = useRememberHeight<HTMLDivElement>('review-timeline', chipsView === 'loading');
  // What's happening right now first: on a phone the line is cut at two lines.
  const statusLine = [
    waiting && 'Waiting for recordings at the playhead…',
    !loading && loadingDays.length > 0 && `Loading recordings for ${daysWords(loadingDays)}.`,
    notice ?? status,
  ]
    .filter(Boolean)
    .join(' · ');
  return (
    <EngineContext value={engine}>
      <div className={classes.review}>
        {/* The tab bar already says this is Review; the heading is for screen readers and focus. */}
        <VisuallyHidden component="h1">Review</VisuallyHidden>
        <div className={classes.cols}>
          <div className={classes.left}>
            {/* The day header sits over the video, so the reading panel runs up to the app's header (#2577). */}
            <header className={classes.head}>
              <DayPicker day={day} days={days.data?.days} onPick={pick} />
              <Button variant="subtle" size="compact-sm" onClick={() => pick(isoDay())}>
                Today
              </Button>
              {person && (
                <Button
                  variant="light"
                  size="compact-sm"
                  radius="xl"
                  rightSection={<IconX size={14} aria-hidden />}
                  aria-label={`Stop showing only ${person}`}
                  onClick={() => engine.pickPerson('')}
                >
                  {person}
                </Button>
              )}
              {/* One line whatever it holds (#2485): the passing notes join the status sentence instead of wrapping
                  in beside it, which pushed the whole page down a line and back. */}
              {/* Its title holds the whole line where it's cut short; phones give it two lines. */}
              <span className={classes.status} role="status" title={statusLine || undefined}>
                {statusLine}
              </span>
              {failedDays.length > 0 && (
                <>
                  <Text size="xs" c="red" role="status">
                    Some recording sources could not load for {daysWords(failedDays)}. Loaded recordings remain
                    available.
                  </Text>
                  <Button
                    size="compact-sm"
                    variant="default"
                    onClick={() => {
                      for (const d of failedDays) retryCore(d);
                    }}
                  >
                    Retry recordings
                  </Button>
                </>
              )}
              {settled && failed.length > 0 && failed.length < sources.length && (
                <Button variant="default" size="compact-sm" onClick={retry}>
                  Retry
                </Button>
              )}
            </header>
            {allFailed && (
              <LoadError
                message={`Could not load ${day}: the transcript, the video list, and the audio list all failed.`}
                onRetry={retry}
              />
            )}
            <div className={classes.grid}>
              <Player
                loading={loading}
                inShowOnly
                empty={
                  settled && segments.data && !segments.data.length
                    ? `Nothing was recorded ${today ? 'today' : `on ${dayName}`}.`
                    : null
                }
                timeline={
                  <Timeline
                    loading={loading}
                    marks={marks}
                    onSpan={span => engine.markEventSpan(span)}
                    notes={
                      (truncatedShown || recordingWindow.activityErrorDays.length > 0) && (
                        <>
                          {truncatedShown &&
                            `Activity markers show only the latest 200 for ${daysWords(recordingWindow.truncatedDays)}.`}
                          {truncatedShown && recordingWindow.activityErrorDays.length > 0 && ' '}
                          {recordingWindow.activityErrorDays.length > 0 && (
                            <Text span inherit c="red">
                              Activity markers unavailable for {daysWords(recordingWindow.activityErrorDays)}.
                            </Text>
                          )}
                        </>
                      )
                    }
                  />
                }
              />
            </div>
            <AskPanel
              opened={askOpen && !coveredBySettings}
              onClose={() => setAskOpen(false)}
              title={`Ask about ${dayName}`}
              kind="review"
              className={classes.askPanel}
            >
              {day && (
                <PackAsk
                  key={day}
                  day={day}
                  bare
                  compact
                  dayLabel={dayName}
                  today={today}
                  onSeek={t => engine.seekPlay(t)}
                  onMarks={setMarks}
                />
              )}
            </AskPanel>
            <div
              className={classes.timeline}
              ref={timelineRef}
              // While the day's chips load, their place is held at the height they had last time here, or about a
              // usual day's the first time (#2485).
              style={
                chipsView === 'loading'
                  ? { minHeight: rememberedHeight('review-timeline') ?? TIMELINE_FIRST_H }
                  : undefined
              }
            >
              {chipsView === 'showOnly' ? (
                <ShowOnly chips={visibleChips} video />
              ) : chipsView === 'loading' ? (
                <ChipsSkeleton working={working} />
              ) : (
                <LoadError
                  message="Could not load the day's objects and events."
                  onRetry={() => void chips.refetch()}
                  busy={chips.isFetching}
                />
              )}
              {chipsView !== 'showOnly' && (
                // Show only holds these once it's there; until then (or when the chips fail) they wait here, under
                // the player, rather than on its bar, so the player keeps one height while the chips load (#2486).
                <div className={classes.videoTools}>
                  <SkipQuiet />
                  <VideoMarkers />
                </div>
              )}
            </div>
          </div>
          <aside className={classes.panel} aria-label="Review details">
            <Tabs
              value={tab}
              onChange={v => v && setTab(v)}
              keepMountedMode="display-none"
              classNames={{ root: classes.tabs, panel: classes.tabPanel }}
            >
              <Tabs.List aria-label="Review details">
                {person && <Tabs.Tab value="person">{person}</Tabs.Tab>}
                <Tabs.Tab value="transcript">Transcript</Tabs.Tab>
                <Tabs.Tab value="summary">Summary</Tabs.Tab>
              </Tabs.List>
              <ReviewSearch
                key={key}
                day={day}
                entries={transcript.data ?? []}
                blocks={summary.data?.blocks ?? []}
                loading={transcript.isPending || summary.isPending}
                summary={
                  summary.isError
                    ? 'error'
                    : !summary.data
                      ? 'loading'
                      : summary.data.status === 'ready'
                        ? 'ready'
                        : 'partial'
                }
                onSeek={ts => engine.seek(ts)}
                onSearching={setQuery}
                beside={tab === 'transcript'}
                onRetrySummary={() => void summary.refetch()}
              />
              {person && (
                <Tabs.Panel value="person" pt="sm" style={searching ? { display: 'none' } : undefined}>
                  <PersonMoments
                    playback={engine.personPlayback}
                    person={person}
                    entries={transcript.data ?? []}
                    chips={chips.data ?? []}
                    activities={activities.data?.items ?? []}
                    blocks={!summary.isPlaceholderData ? (summary.data?.blocks ?? []) : []}
                    summaryState={
                      summary.isError
                        ? 'error'
                        : summary.isPending || summary.isPlaceholderData
                          ? 'loading'
                          : summary.data?.status === 'ready'
                            ? 'ready'
                            : 'partial'
                    }
                    onRetrySummary={() => void summary.refetch()}
                  />
                </Tabs.Panel>
              )}
              <Tabs.Panel value="transcript" pt="sm">
                <Transcript entries={transcript.data ?? (transcript.isError ? [] : null)} query={query} />
              </Tabs.Panel>
              <Tabs.Panel value="summary" pt="sm" style={searching ? { display: 'none' } : undefined}>
                {day && (
                  <div className={classes.summaryDetails}>
                    <SummaryBlocks day={day} />
                  </div>
                )}
              </Tabs.Panel>
            </Tabs>
          </aside>
        </div>
        <EventForm
          span={marking}
          onClose={() => engine.closeEvent()}
          onEntries={span => {
            engine.closeEvent();
            setEntryWindow(span);
          }}
        />
        <EntriesForm span={entryWindow} onClose={() => setEntryWindow(null)} />
        {/* biome-ignore lint/a11y/useMediaCaption: the room's sound under the video; the transcript is its text */}
        <audio ref={audioRef} preload="none" hidden onEnded={() => engine.audioEnded()} />
      </div>
    </EngineContext>
  );
}
