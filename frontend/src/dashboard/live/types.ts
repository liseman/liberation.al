// The live data the dashboard server streams on GET /events, and polls on the same shapes when the stream is down.
// Lines and scene entries carry more fields than these; the tabs that read them type what they use.

/** A transcript line as a request names it: its key (`line`, "<device slug>@<ts_start>"), which no other
 *  microphone's line shares, or its ts_start as before keys, which the server refuses when two lines share it. */
export type LineRef = string | number;

/** The body that names lines: { line } or { ts_start } for one, { lines } or { ts_starts } for several. */
export const oneLine = (ref: LineRef) => (typeof ref === 'string' ? { line: ref } : { ts_start: ref });
export const someLines = (refs: readonly LineRef[]) => ({
  lines: refs.map(r => (typeof r === 'string' ? r : r.toFixed(3))), // a time is the key of a line with no device
});

export interface TranscriptEntry {
  ts_start: number;
  line?: string; // its key (registry.key_of): what requests about this one line send
  ts_end?: number;
  text?: string;
  speaker?: string;
  speaker_id?: number | string | null;
  assistant?: boolean;
  media?: boolean; // heard from a TV or radio: shown, but nobody's line
  media_by?: 'owner' | 'classifier'; // who marked it a TV's: the owner, or the audio tagger by ear
  _new?: boolean; // arrived in the last few seconds (the Live tab highlights it)
  [field: string]: unknown;
}

/** One transcript event or poll result: a reset replaces the lines, replace swaps lines already shown. */
export interface TranscriptEvent {
  entries?: TranscriptEntry[];
  reset?: boolean;
  replace?: boolean;
}

export interface Transcript {
  entries: TranscriptEntry[]; // oldest first, at most MAX_LINES
  resets: number; // bumps when the lines are replaced wholesale (the Live tab scrolls to the bottom)
}

export interface SceneEntry {
  cam: string;
  ts: number;
  /** The view hadn't changed, so its caption is the last real one, from `since` (epoch seconds). */
  unchanged?: boolean;
  since?: number | null;
  [field: string]: unknown;
}

export interface Speaker {
  id: number | string;
  label: string;
  last_seen?: number;
  [field: string]: unknown;
}

export interface SpeakersEvent {
  speakers?: Speaker[];
  merged?: Record<string, number | string>;
}

export interface Speakers {
  list: Speaker[];
  byId: Record<string, Speaker>;
  merged: Record<string, number | string>;
}

export interface AmbientDevice {
  name: string;
  level_dbfs?: number;
  /** The level a frame has to pass for the VAD to hear speech now (dBFS). */
  threshold_dbfs?: number;
  sensitivity?: number | null;
  /** Its gain now (auto gain). */
  gain?: { auto: boolean; kind: 'hardware' | 'software' | null; db: number | null };
  [field: string]: unknown;
}

/** One camera as vision reports it: "starting" until its first frame, "on" while frames come, "retrying" while it
 *  is being reopened after frames stopped. */
export interface VisionCamera {
  state: 'starting' | 'on' | 'retrying';
  since?: number;
  attempts?: number;
}

/** The cameras capture is sending (GET /vision, from their <name>.video streams on the broker), sent every 5 s. */
export interface VisionStatus {
  ts?: number;
  cameras?: Record<string, VisionCamera>;
}

/** The microphones (GET /ambient, the dashboard's mic_status): each one healthy while the listen worker runs and
 *  its stream on the broker has fresh audio; ts while any is. */
export interface Ambient {
  ts?: number;
  devices?: Record<string, AmbientDevice>;
  [field: string]: unknown;
}

/** Each kind of live data, as the cache holds it. */
export interface LiveData {
  transcript: Transcript;
  scene: SceneEntry[];
  speakers: Speakers;
  ambient: Ambient;
  vision: VisionStatus;
}

export type LiveKind = keyof LiveData;
