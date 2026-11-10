// Review data types: server records (dashboard/server.py and presence.py), and local event-label ranges.

import type { TranscriptEntry } from '../live/types';
import type { DetectedActivity } from './api';

/** A camera and time range selected for the event-label form. */
export interface EventSpan {
  cam: string;
  start: number;
  end: number;
}

/** A start and end time, in Unix seconds. */
export type Span = [number, number];

/** A stretch of a camera's recorded video, from the gateway's spans (gatewaySegments.ts). */
export interface RawSegment {
  cam: string;
  start: number;
  end: number;
  /** What GatewayVideo plays (`gateway:<stream>:<start_us>:<end_us>`). */
  url: string;
  /** Video that aged out past the history window (#1748): shown on the timeline, never played. */
  expired?: true;
}

/** A video segment as Review keeps it: end filled in, the file's URL, and why it can't play once a load fails. */
export interface Segment {
  cam: string;
  start: number;
  end: number;
  url: string;
  bytes: number;
  broken?: 'recording' | 'cut' | 'expired';
}

/** Kept room audio: one span per archived chunk (GET /audio/spans). */
export interface AudioSpan {
  start: number;
  end: number;
  device?: string;
}

/** A person, object, or notable event of the day, with the time it was around (GET /review/presence). */
export interface Chip {
  id: string;
  kind: string; // 'person' | 'object' | 'event'
  label: string;
  spans: Span[];
  total_s: number;
  count: number;
  notes?: string[];
  activities?: DetectedActivity[];
  camera?: string; // per-camera presence and movement event chips
  /** A person chip's spans on each camera view that saw them (#328). */
  cams?: Record<string, Span[]>;
  /** A named person's camera-less speech spans, kept independently of face visits when filtering cameras. */
  speech?: Span[];
}

export interface Face {
  name?: string | null;
}

/** What one camera saw nearest a moment (GET /history/at). faces is null when face ID was off. */
export interface CamNote {
  ts: number;
  dt: number;
  caption: string;
  faces: Face[] | null;
}

/** The camera notes for a moment, or forbidden when camera history isn't shown to this viewer. */
export interface MomentNotes {
  cams: Record<string, CamNote>;
  forbidden?: boolean;
}

/** Everything Review draws for one day. */
export interface DayData {
  entries: TranscriptEntry[];
  segments: RawSegment[];
  spans: AudioSpan[];
}

export type ChunkStatus = 'loading' | 'ready' | 'error';
