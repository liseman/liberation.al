// How one event (or poll result) of each kind changes what the cache holds. Pure: the feed handles timers.
import type {
  Ambient,
  LiveData,
  LiveKind,
  SceneEntry,
  Speakers,
  SpeakersEvent,
  Transcript,
  TranscriptEntry,
  TranscriptEvent,
  VisionStatus,
} from './types';

export const MAX_LINES = 300; // live transcript lines kept (LIVE_LINES in dashboard/server.py)

export const EMPTY_TRANSCRIPT: Transcript = { entries: [], resets: 0 };

export function reduceTranscript(prev: Transcript | undefined, event: TranscriptEvent | null): Transcript {
  const state = prev ?? EMPTY_TRANSCRIPT;
  const incoming = event?.entries ?? [];
  if (event?.reset) return { entries: incoming.slice(-MAX_LINES), resets: state.resets + 1 };
  if (event?.replace) {
    // Lines read differently since they were sent (a label in the event log): swap the ones shown, add none.
    // By key (#247): two microphones' lines in one millisecond are two lines. A line with no key goes by its time.
    const id = (e: TranscriptEntry) => e.line ?? e.ts_start;
    const byLine = new Map(incoming.map(e => [id(e), e]));
    const entries = state.entries.map(e => {
      const fresh = byLine.get(id(e));
      return !e.assistant && fresh ? { ...fresh, _new: e._new } : e;
    });
    return { ...state, entries };
  }
  if (!incoming.length) return state;
  // Sorted: the assistant's replies arrive on their own schedule.
  const entries = [...state.entries, ...incoming.map(e => ({ ...e, _new: true }))]
    .sort((a, b) => a.ts_start - b.ts_start)
    .slice(-MAX_LINES);
  return { ...state, entries };
}

/** Clear the arrival highlight on every line. */
export const settle = (t: Transcript): Transcript =>
  t.entries.some(e => e._new) ? { ...t, entries: t.entries.map(e => (e._new ? { ...e, _new: false } : e)) } : t;

export function reduceSpeakers(event: SpeakersEvent | null): Speakers {
  const list = event?.speakers ?? [];
  return { list, merged: event?.merged ?? {}, byId: Object.fromEntries(list.map(s => [String(s.id), s])) };
}

/** The new cached value for one event of a kind, given the old one. */
export function reduce<K extends LiveKind>(kind: K, prev: LiveData[K] | undefined, data: unknown): LiveData[K] {
  const out: { [Kind in LiveKind]: () => LiveData[Kind] } = {
    transcript: () => reduceTranscript(prev as Transcript | undefined, data as TranscriptEvent | null),
    speakers: () => reduceSpeakers(data as SpeakersEvent | null),
    scene: () => (Array.isArray(data) ? (data as SceneEntry[]) : []),
    ambient: () => (data as Ambient) || {},
    vision: () => (data as VisionStatus) || {},
  };
  return out[kind]() as LiveData[K];
}

/** A speaker id after merges (a merged voice answers to the one it joined). */
export function resolveId(speakers: Speakers, id: number | string | null | undefined): number | string | null {
  if (id == null) return null;
  return speakers.merged[String(id)] ?? id;
}
