// Review's engine: a continuous window of video, speech, and camera notes you can scrub, zoom, and play.
//
// It lives outside React and owns time and media. It works out which video segment covers a moment from the day's
// segment list (no request per seek), double-buffers each camera (the next segment preloads in a hidden <video>, so
// playback rolls over without a stall), coalesces seeks while scrubbing (one in flight per camera, at most every
// SEEK_MS, the latest target wins, an exact seek on release), and runs one animation-frame loop that moves the playhead
// and clock directly. The recordings carry no sound, so the room's audio plays under the video from one <audio>, kept
// in step; where there's no video, the transcript's lines play one after another. React re-renders only from its small
// store (ui), which changes when something visible does: which cameras have video, the camera notes for the moment,
// the selected chips, the play state, or the playhead's time to the half second.
import { createStore, type StoreApi } from 'zustand/vanilla';
import { HttpError } from '../../shared/http';
import { readPrefs, writePrefs } from '../../shared/prefs';
import { hms, isoDay } from '../lib/format';
import { playbackKeyBlocked, stop as stopClip, toggle as toggleClip } from '../lib/player';
import type { TranscriptEntry } from '../live/types';
import { setHash, tabOf } from '../router';
import { BUCKET_S } from './api';
import { chipColors } from './chipStyle';
import { calendarStep, landingDay } from './days';
import { MAX_DAY_SEGMENTS, normalizeSegments } from './gatewaySegments';
import { cameraChips } from './lanes';
import type { PlaybackMoment } from './personMomentsModel';
import { PersonPlayback, type PersonPlaybackMode } from './personPlayback';
import { type ReviewKey, shortcut } from './reviewKeys';
import { facetedSpans, nextSpanStart, presentFrom, presentSpans, prevSpanStart } from './spans';
import {
  clampView,
  DAY_S,
  dataEdges,
  dayBounds,
  latestVideo,
  lineAt,
  lineEnd,
  nextMedia,
  segAt,
  type View,
} from './time';
import type { AudioSpan, Chip, ChunkStatus, DayData, EventSpan, MomentNotes, RawSegment, Segment, Span } from './types';
import { DAY_RE, readHash, reviewHash } from './url';

const AHEAD_S = 30; // this long before a stretch of room audio ends, the next one is fetched (#344)
const AHEAD_KEEP = 3; // windows of room audio kept fetched (about 11 MB each at 48 kHz)
const PRELOAD_MS = 300; // paused this long at a spot before its room audio is fetched, so clicking around fetches once
type Span2 = { start: number; end: number };
type Ahead = { p: Promise<Blob>; ctl: AbortController; done: boolean };
const roomKey = (r: Span2) => `${r.start}-${r.end}`;
/** Room audio comes in windows of this long, on whole multiples of it, cut to the segment it plays under: a gateway
 *  segment runs for as long as the camera recorded (often most of an hour), while the server sends at most 330 s and
 *  a window of it is about 11 MB at the microphone's 48 kHz, so a whole segment's was cut short and slow (#917). */
const ROOM_WINDOW_S = 120;
/** The room audio window of `seg` holding `ts`. */
export function roomWindow(seg: Span2, ts: number, len = ROOM_WINDOW_S): Span2 {
  const at = Math.floor(ts / len) * len; // the window's own start, so windows never overlap
  return { start: Math.max(seg.start, at), end: Math.min(seg.end, at + len) };
}
const roomUrl = (r: Span2) => `/audio/range?start=${r.start}&end=${r.end}`;
const SEEK_MS = 33; // at most one seek per camera this often while scrubbing (a scrub seek is one frame, #835)
const UI_MS = 100; // how often the moment and transcript line follow the playhead
/** The link follows a seek or playback at most this often (#2391). A field so tests can turn the wait off. */
export const hashTiming = { everyMs: 1500 };
/** A Review hash without its time, to tell a seek from a change of place. */
const withoutTime = (hash: string) => hash.replace(/(^#|&)t=[^&]*/, '');
const ZOOM_MS = 350; // how long the timeline takes to zoom (to the selected chips, or Day, 1 h, and 10 min)
const reducedMotion = (): boolean =>
  typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

/** The view a fraction e (0 to 1, eased) of the way from a to b: the centre moves evenly and the span changes by
 *  the same ratio each step, so a day to 10 minutes reads as one steady zoom rather than a lurch at the end. */
export function glideView(
  a: { t0: number; t1: number },
  b: { t0: number; t1: number },
  e: number,
): [t0: number, t1: number] {
  const sa = a.t1 - a.t0;
  const sb = b.t1 - b.t0;
  const span = sa > 0 && sb > 0 ? sa * (sb / sa) ** e : sa + (sb - sa) * e;
  const c = (a.t0 + a.t1) / 2 + ((b.t0 + b.t1) / 2 - (a.t0 + a.t1) / 2) * e;
  return [c - span / 2, c + span / 2];
}
const SAFETY_MS = 1200; // a seek that never reports back is given up on after this long
const SYNC_MS = 500; // how often the other cameras and the room audio are nudged back in step with the lead camera
const DRIFT_S = 0.35; // how far out of step they may get first; the room audio is snapped back past this
// Smaller offsets of the room audio are eased away by its playback rate instead, so there is no audible jump: the rate
// moves by NUDGE_GAIN per second of offset, at most NUDGE_MAX, and goes back to 1 within NUDGE_MIN_S (#244).
const NUDGE_MIN_S = 0.03;
const NUDGE_GAIN = 0.5;
const NUDGE_MAX = 0.05;
const FOLLOW_MS = 4000; // after a hand moves the timeline, it isn't pulled back to the playhead for this long
const END_HOLD_S = 0.05; // where an embedded clip rests when it has played to its end, before that end

/** Where a clip that played to its end rests: just before its last segment's end, which segments exclude, and never
 *  before that segment's start (one shorter than the hold, #2550). */
export const endHold = (t: number, seg: { start: number; end: number }): number =>
  Math.max(seg.start, Math.min(t, seg.end - END_HOLD_S));
const CLIP_LEAD_S = 0.4; // a line's clip starts this long before the line
const ROOM_GAP_S = 5; // a gap between videos at least this long is filled with the kept room audio (#196)
const ROOM_JOIN_S = 1; // kept stretches this close together play as one (chunk starts can be a hair apart)
export const BLOCKED = 'The browser blocked the sound; click Play again.';
const namedPerson = (name: string): boolean =>
  !!name.trim() && !/^(?:(?:speaker|face)\s+\d+|unknown(?: faces)?|multiple speakers)$/i.test(name.trim());
// A moment of silence, played inside the click that starts playback: a browser that only lets sound start in a user
// gesture (Safari, and Chrome once the gesture has expired) then allows the room audio that arrives later (#213).
const SILENCE = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=';

/** What React shows. Everything else the engine keeps to itself. */
export interface ReviewUi {
  day: string | null;
  gen: number; // bumps each time a day is picked, so the page hands the engine that day's data again
  person: string;
  people: string[]; // confirmed person filters in selection order; the last name owns the person tab
  cameras: string[]; // empty means every camera, independent of the large tile
  camera?: string;
  cameraIntent?: number; // changes for each external camera link, even when it names the same camera
  t: number | null; // the playhead, to the half second (the clock and playhead themselves move every frame)
  avail: string[]; // cameras with video at the playhead
  gap: 'recording' | 'cut' | 'expired' | null; // why no camera can show the playhead, when a segment covers it
  unplayable: string[]; // cameras whose video at the playhead is in a codec this browser can't decode (HEVC, AV1)
  unplayableCodecs: Record<string, string>; // which codec, by camera, when the player could tell (`av1`, #1479)
  notes: MomentNotes | null; // what the cameras saw at the playhead
  sel: ReadonlySet<string>; // the selected chips
  filterStatus: 'unknown' | 'empty' | null;
  colors: Record<string, string>; // and their colours
  skip: boolean; // whether playback skips the time none of the selected chips is present
  playing: boolean;
  eventMarks: { cam: string; in: number | null; out: number | null } | null;
  eventSpan: EventSpan | null; // the form stays open after Save, while its timeline marks clear
  markingTool: boolean; // the bar's Mark is on: the next drag on the timeline picks a span instead of scrubbing
  notice: string | null; // a message for the status line, until the next day loads
  roomOnly: boolean; // playing the room's kept audio where there's no video (and no line playing) (#196)
  speed: number; // how fast playback goes: 1, 1.5, or 2 (#327)
  muted: boolean; // the room's sound is muted; the video plays on (#2041)
  /** The timeline's visible range, once zoom or pan settles (#414): null until the day's first load. */
  range: View | null;
  chunkStatus: Record<string, ChunkStatus>;
  dataVersion: number;
  waiting: boolean; // playback is waiting for an adjacent chunk, rather than treating it as an empty interval
  rangeIntent: number; // changes for a deliberate viewport pan/zoom, not cursor-follow navigation
}

export const RANGE_SETTLE_MS = 250; // how long the view holds still before range follows it

/** The parts of a <video> the engine uses (tests pass fakes). */
export interface VideoLike {
  src: string;
  preload: string;
  currentTime: number;
  playbackRate?: number;
  defaultPlaybackRate?: number;
  readonly readyState: number;
  readonly duration: number;
  readonly error: MediaError | null;
  readonly seeking: boolean;
  readonly paused: boolean;
  readonly ended: boolean;
  /** The picture can't be shown in this browser (its codec isn't supported); a player that can't tell leaves it out. */
  readonly unplayable?: boolean;
  /** Which codec it is, when the player knows. */
  readonly unplayableCodec?: string | null;
  /** The segment it plays grew to this many seconds long (a stretch still being recorded); a <video> can't. */
  extend?: (duration: number) => void;
  fastSeek?: (time: number) => void;
  play(): Promise<void>;
  pause(): void;
  load(): void;
  getAttribute(name: string): string | null;
  removeAttribute(name: string): void;
  toggleAttribute(name: string, force?: boolean): boolean;
  addEventListener(type: string, listener: () => void, options?: AddEventListenerOptions): void;
  removeEventListener(type: string, listener: () => void): void;
}

/** The parts of the <audio> the engine uses for room audio and lines (tests pass fakes). */
export interface AudioLike {
  muted?: boolean;
  src: string;
  currentTime: number;
  readonly readyState: number;
  readonly paused: boolean;
  readonly ended?: boolean;
  playbackRate: number;
  play(): Promise<void>;
  pause(): void;
  load(): void;
  addEventListener(type: string, listener: () => void, options?: AddEventListenerOptions): void;
  removeEventListener?: (type: string, listener: () => void) => void;
}

export interface TimelineHooks {
  frame(): void; // every animation frame: move the playhead, and redraw when marked dirty
  redraw(): void; // the data or the view changed
}

export interface TranscriptHooks {
  /** Mark line i as current; bring it into view, or centre it when centre is true (after a jump). */
  setCur(i: number, centre: boolean): void;
}

export interface EngineDeps {
  /** A local player owns no page URL or global keyboard shortcuts. */
  embedded?: boolean;
  cams: string[];
  fetchNotes: (bucket: number) => Promise<MomentNotes>;
  now?: () => number; // wall clock, Unix seconds
  /** Fetch a clip or a stretch of room audio whole. Streamed responses left Chromium's <audio> stalled at readyState
   *  0 (2026-10-25), so the engine plays a blob instead. */
  fetchBlob?: (url: string, signal?: AbortSignal) => Promise<Blob>;
  /** A day opened afresh by a link shown again (#330): its cached data is out of date, so load it again. */
  reload?: (day: string) => void;
  /** Seconds of room audio fetched at a time (ROOM_WINDOW_S). */
  roomWindowS?: number;
}

async function fetchBlob(url: string, signal?: AbortSignal): Promise<Blob> {
  const r = await fetch(url, { cache: 'no-store', signal });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.blob();
}

/** Play blob in audio, letting go of the blob it played before. */
function loadBlob(audio: AudioLike, blob: Blob): void {
  const old = audio.src;
  audio.src = URL.createObjectURL(blob);
  if (old.startsWith('blob:')) URL.revokeObjectURL(old);
}

interface Want {
  ts: number;
  exact: boolean;
}

/** One camera: two <video>s, which is showing, the segment each holds, and its seek queue. */
interface Cam {
  v: [VideoLike | null, VideoLike | null];
  seg: [Segment | null, Segment | null];
  act: 0 | 1;
  busy: boolean;
  epoch: number; // invalidates pending seeks when a source/listing/day is replaced
  want: Want | null;
  last: number;
  timer: ReturnType<typeof setTimeout> | undefined;
  safety: ReturnType<typeof setTimeout> | undefined;
  playing: [boolean, boolean]; // each <video> is showing frames (its playing event came, and no pause or stall since)
}

const other = (i: 0 | 1): 0 | 1 => (i === 0 ? 1 : 0);
const sameSet = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((x, i) => x === b[i]);

export class ReviewEngine {
  readonly personPlayback = new PersonPlayback({
    play: (moment, mode) => this.playPersonMoment(moment, mode),
    stop: () => this.setPlaying(false),
  });
  private personJump = false;
  private personAudioError: (() => void) | null = null;
  readonly ui: StoreApi<ReviewUi>;
  readonly cams: string[];

  // the day
  private rangeTimer: ReturnType<typeof setTimeout> | undefined; // publishes range once the view settles
  dayStart = 0;
  entries: TranscriptEntry[] = [];
  segs: Record<string, Segment[]> = Object.create(null);
  spans: AudioSpan[] = [];
  chips: Chip[] = [];
  private loaded: number | null = null; // the gen whose data is in
  private chipsGen: number | null = null; // the gen whose chips are in
  private readonly chunks = new Map<string, DayData>();
  private readonly chunkChips = new Map<string, Chip[]>();
  private windowDays: string[] | null = null;
  private dayLanding: string | null = null;
  private awaitingMedia: number | null = null;
  private awaitingTarget = false;
  navigationVersion = 0; // invalidates queued pointer/key seeks after a newer navigation intent
  private viewIntent = 0;

  // time and the view
  t: number | null = null;
  view: View = { t0: 0, t1: DAY_S };
  scrubbing = false;
  playing = false;
  private wasPlaying = false; // playing when a scrub started
  private lastUser = -1e9; // when a hand last moved the view
  private anim: { a: View; b: View; at: number } | null = null;
  private home: View | null = null; // where clearing the chips zooms back to

  // chips
  marks: { color: string; spans: Span[] }[] = [];
  present: Span[] | null = [];
  private pendingChips: Chip[] | null = null; // chips that arrived mid-scrub

  // media
  private displayedCamera: string | null = null; // the Player's effective large tile, while it is mounted
  private avail: string[] = []; // cameras with video at the playhead
  private lead: string | null = null; // the camera whose video sets the playhead while playing
  private audio: AudioLike | null = null;
  private audioEntry: number | null = null; // the line playing where there's no video
  private room: { start: number; end: number } | null = null; // the room audio playing (or loading) under the video
  private loadedRoom: { start: number; end: number } | null = null; // the room audio the <audio> holds, kept across pauses
  private ahead = new Map<string, Ahead>(); // room audio fetched or fetching, by stretch (#344)
  private aheadFailed = new Set<string>(); // stretches whose room audio failed today: not asked for again
  private unlocked = false; // the <audio> has played inside a click (#213, #344)
  private preloadTimer: ReturnType<typeof setTimeout> | undefined;
  private audioFetch: AbortController | null = null; // the clip or room audio on its way
  private audioToken = 0;
  private roomFailed = new Set<number>(); // starts of kept stretches whose audio wouldn't load this day // bumps with each clip or room audio asked for, so a late arrival is ignored
  private soundNotice: string | null = null; // the status-line notice playAudio set about the sound failing
  private lastSync = 0;

  private readonly P: Record<string, Cam> = Object.create(null);
  private readonly deps: EngineDeps;
  private tl: TimelineHooks | null = null;
  private tr: TranscriptHooks | null = null;
  private clock: { textContent: string | null } | null = null;
  private raf = 0;
  private lastUi = 0;
  private lastHash = 0; // when writeHash last replaced the entry, for its throttle
  private hashTimer: ReturnType<typeof setTimeout> | undefined; // a throttled write still to come
  private following = false; // reading a link (a load, Back, or another tab's link): its writes never push
  private seen: { cur: number; avail: string | null; t: number | null } = { cur: -2, avail: null, t: null };
  private notes = new Map<number, MomentNotes>();
  private noteBucket: number | null = null;
  private noteTimer: ReturnType<typeof setTimeout> | undefined;
  private notesForbidden = false;
  private started = false;
  private begun = false; // started once: a later start resumes where it was (the tab was hidden, #330)
  private wrote = ''; // the hash writeHash last wrote, so a start can tell a new link from its own
  private playOnLoad = false;
  private embeddedStart: number | null = null;
  private embeddedEnded = false;
  private playFromStartOnLoad = false;
  private landing = false; // opened on today by default, and no day picked since: recordedDays may move it

  constructor(deps: EngineDeps) {
    this.deps = deps;
    this.cams = deps.cams;
    this.ui = createStore<ReviewUi>(() => ({
      day: null,
      gen: 0,
      person: '',
      people: [],
      filterStatus: null,
      cameras: [],
      t: null,
      avail: [],
      gap: null,
      unplayable: [],
      unplayableCodecs: {},
      notes: null,
      sel: new Set(),
      colors: {},
      skip: !!readPrefs().reviewSkip,
      speed: 1,
      muted: false,
      range: null,
      playing: false,
      eventMarks: null,
      eventSpan: null,
      markingTool: false,
      notice: null,
      roomOnly: false,
      chunkStatus: {},
      dataVersion: 0,
      waiting: false,
      rangeIntent: 0,
    }));
  }

  get visibleCams(): string[] {
    return this.S.cameras.length ? this.cams.filter(c => this.S.cameras.includes(c)) : this.cams;
  }
  get filteredChips(): Chip[] {
    return cameraChips(this.chips, this.S.cameras);
  }
  pickCameras(cameras: readonly string[]): void {
    if (this.personPlayback.run) this.pause();
    this.set({ cameras: this.cams.filter(c => cameras.includes(c)) });
    this.setSel(this.S.sel, false);
    this.writeHash();
  }
  private get S(): ReviewUi {
    return this.ui.getState();
  }
  private set(patch: Partial<ReviewUi>): void {
    this.ui.setState(patch);
  }
  private wall(): number {
    return this.deps.now?.() ?? Date.now() / 1000;
  }
  private clampT(ts: number): number {
    if (!this.deps.embedded) return Number.isFinite(ts) ? ts : (this.t ?? this.dayStart);
    const nextMidnight = new Date(this.dayStart * 1000);
    nextMidnight.setDate(nextMidnight.getDate() + 1);
    return Math.max(this.dayStart, Math.min(nextMidnight.getTime() / 1000 - 0.01, ts));
  }
  private allSegs(): Segment[] {
    return this.cams.flatMap(c => this.segs[c] ?? []);
  }

  /** Details remain day-local even though the media timeline holds several cached days. */
  private setTranscriptCurrent(ts: number, centre: boolean): void {
    const entries = this.deps.embedded ? this.entries : (this.chunks.get(this.S.day ?? '')?.entries ?? []);
    this.tr?.setCur(lineAt(entries, ts), centre);
  }

  private pendingAfter(ts: number, before = Number.POSITIVE_INFINITY): boolean {
    return (
      !this.deps.embedded &&
      (this.windowDays ?? []).some(day => {
        const bounds = dayBounds(day);
        return (
          bounds.t1 > ts &&
          bounds.t0 <= ts + DAY_S &&
          bounds.t0 <= before &&
          bounds.t0 <= this.wall() &&
          (this.S.chunkStatus[day] === undefined || this.S.chunkStatus[day] === 'loading')
        );
      })
    );
  }

  private waitForMore(ts: number): boolean {
    if (!this.pendingAfter(ts)) return false;
    this.awaitingMedia = ts;
    this.awaitingTarget = false;
    this.t = ts;
    this.contextAt(ts);
    this.set({ waiting: true, t: ts });
    return true;
  }

  // ---- the visible window ----
  setView(t0: number, t1: number, user = false): void {
    this.view = clampView(t0, t1, this.deps.embedded ? this.dayStart : undefined);
    if (this.loaded != null) {
      // Published once it settles: a glide, a drag, or the playhead's follow moves it every frame meanwhile.
      clearTimeout(this.rangeTimer);
      this.rangeTimer = setTimeout(() => this.publishRange(), RANGE_SETTLE_MS);
    }
    if (user) {
      this.viewIntent++;
      this.lastUser = performance.now();
      this.anim = null; // a hand on the timeline stops a zoom in progress
    }
    this.tl?.redraw();
  }
  /** Range follows the view now: unless it already says that, so a subscriber isn't woken for nothing. */
  private publishRange(): void {
    clearTimeout(this.rangeTimer);
    this.rangeTimer = undefined;
    const r = this.S.range;
    if (!r || r.t0 !== this.view.t0 || r.t1 !== this.view.t1 || this.S.rangeIntent !== this.viewIntent)
      this.set({ range: { ...this.view }, rangeIntent: this.viewIntent });
  }
  /** Glide the timeline to [t0, t1], as a hand on it would (#414): a range narrower than the closest zoom is
   *  centred at that zoom. */
  zoomTo(t0: number, t1: number): void {
    const v = clampView(Math.min(t0, t1), Math.max(t0, t1), this.deps.embedded ? this.dayStart : undefined);
    this.glideTo(v.t0, v.t1);
  }
  /** Ease the view to [t0, t1] over ZOOM_MS (stepped from the frame loop), or jump there when the viewer asks for
   *  less motion. */
  private glideTo(t0: number, t1: number): void {
    this.lastUser = performance.now(); // and don't let follow() pull it back to the playhead meanwhile
    if (reducedMotion()) {
      this.anim = null;
      this.setView(t0, t1);
      return;
    }
    this.anim = { a: { ...this.view }, b: { t0, t1 }, at: performance.now() };
  }
  private glide(now: number): void {
    if (!this.anim) return;
    const k = Math.min(1, (now - this.anim.at) / ZOOM_MS);
    const { a, b } = this.anim;
    if (k >= 1) {
      this.anim = null;
      this.setView(b.t0, b.t1);
      return;
    }
    this.setView(...glideView(a, b, 1 - (1 - k) ** 3));
  }
  /** Where the view is headed: a glide's end while one runs, else the view itself. */
  private targetView(): { t0: number; t1: number } {
    return this.anim?.b ?? this.view;
  }
  /** Bring t into view at the same span, centred on it, unless it's already where the view is headed. During a glide
   *  the glide is pointed there instead, or its next frame would pull the view back (#2597). */
  private keepInView(t: number): void {
    const { t0, t1 } = this.targetView();
    if (t >= t0 && t <= t1) return;
    const half = (t1 - t0) / 2;
    if (!this.anim) {
      this.setView(t - half, t + half);
      return;
    }
    const v = clampView(t - half, t + half, this.deps.embedded ? this.dayStart : undefined);
    this.glideTo(v.t0, v.t1);
  }
  /** Zoom to span seconds across, centred on the playhead (or the middle of the day), with a glide (#2577). */
  zoomSpan(span: number): void {
    const c = this.t ?? this.dayStart + DAY_S / 2;
    const v = clampView(c - span / 2, c + span / 2, this.deps.embedded ? this.dayStart : undefined);
    this.viewIntent++; // a hand on the timeline, as setView(..., true) counts one
    this.glideTo(v.t0, v.t1);
  }
  /** Go to where the day's newest full minute of video starts, or the end of its data. */
  latest(): void {
    const day = this.S.day;
    const chunk = !this.deps.embedded && day ? this.chunks.get(day) : null;
    const ed = chunk
      ? dataEdges(
          chunk.entries,
          chunk.segments.map(s => ({ ...s, bytes: 0 })),
          chunk.spans,
        )
      : dataEdges(this.entries, this.allSegs(), this.spans);
    if (!ed) return;
    const t = chunk && day ? this.preferredTime(day, chunk) : (latestVideo(this.segs, this.wall()) ?? ed[1] - 1);
    const span = Math.min(this.view.t1 - this.view.t0, 3600);
    this.setView(t - span * 0.7, t + span * 0.3, true);
    this.seek(t);
  }
  // Keep the playhead in view while playing, unless someone moved the view in the last few seconds.
  private follow(): void {
    const { t0, t1 } = this.view;
    const span = t1 - t0;
    if (this.t == null || performance.now() - this.lastUser < FOLLOW_MS) return;
    if (this.t > t1 - span * 0.04 || this.t < t0) this.setView(this.t - span * 0.3, this.t + span * 0.7);
  }

  // ---- days ----
  /** Show a day, from its start (or the time in the link, when start() read one). */
  pickDay(d: string): void {
    if (!DAY_RE.test(d)) return;
    this.personPlayback.cancel();
    this.closeEvent();
    this.landing = false;
    if (!this.deps.embedded && this.loaded != null) {
      const bounds = dayBounds(d);
      const data = this.chunks.get(d);
      this.dayLanding = data ? null : d;
      const target = data ? this.preferredTime(d, data) : bounds.t0;
      this.seek(target);
      this.dayLanding = data ? null : d;
      this.setView(bounds.t0, bounds.t1, true);
      this.publishRange();
      return;
    }
    this.t = null;
    this.setDay(d);
  }
  /** The days with something recorded have loaded. When Review opened on today by default and today has nothing,
   *  it moves to the latest recorded day, unless a day was picked or linked to meanwhile. Only the first list counts:
   *  later refreshes never move the page. today is the server's: when it is ahead of the browser's, its newest days
   *  still count. */
  recordedDays(days: readonly string[], today?: string): void {
    if (!this.landing || !this.S.day) return;
    this.landing = false;
    const d = landingDay(days, this.S.day, today && today > this.S.day ? today : this.S.day);
    if (d) this.pickDay(d);
  }
  private setDay(d: string): void {
    this.navigationVersion++;
    this.closeEvent();
    this.playFromStartOnLoad = false;
    this.setPlaying(false);
    this.set({ notice: null, avail: [], unplayable: [], unplayableCodecs: {}, gap: null });
    for (const c of this.cams) this.release(c);
    this.dayStart = new Date(`${d}T00:00:00`).getTime() / 1000;
    this.loaded = null;
    clearTimeout(this.preloadTimer); // a preload asked for on the last day
    clearTimeout(this.rangeTimer); // and the last day's range: none until this one loads
    this.rangeTimer = undefined;
    this.set({ range: null });
    this.anim = null; // a glide toward the last day's times
    for (const e of this.ahead.values()) if (!e.done) e.ctl.abort(); // another day's room audio
    this.ahead.clear();
    this.aheadFailed.clear();
    // The last day's chips go at once, so the new day is never drawn (or skipped through) with them.
    this.chips = [];
    this.chipsGen = null;
    this.pendingChips = null;
    this.setSel([], false);
    this.set({ day: d, gen: this.S.gen + 1 });
    this.writeHash(); // at once, so the link names the day even when it has nothing to play
  }

  /** A cursor crossing midnight changes context, never the engine or its loaded media lifecycle. */
  private contextAt(ts: number): void {
    if (this.deps.embedded) return;
    const day = isoDay(new Date(ts * 1000));
    if (day === this.S.day) return;
    this.dayStart = dayBounds(day).t0;
    this.set({ day, t: ts, notice: null });
    this.seen.cur = -2;
  }

  private preferredTime(day: string, data: DayData): number {
    const segments = data.segments.map(s => ({ ...s, bytes: 0 }));
    const grouped: Record<string, Segment[]> = Object.create(null);
    for (const s of segments) {
      grouped[s.cam] ??= [];
      grouped[s.cam]?.push(s);
    }
    const edges = dataEdges(data.entries, segments, data.spans);
    return latestVideo(grouped, this.wall()) ?? (edges ? edges[1] - 1 : dayBounds(day).t0);
  }

  /** Keep decoded timeline rows bounded to the visible window; QueryClient owns the reusable day cache. */
  setWindowDays(days: string[]): void {
    const next = [...new Set(days)].sort();
    if (this.windowDays && sameSet(next, this.windowDays)) return;
    this.windowDays = next;
    for (const day of this.chunks.keys()) if (!next.includes(day)) this.chunks.delete(day);
    for (const day of this.chunkChips.keys()) if (!next.includes(day)) this.chunkChips.delete(day);
    this.set({
      chunkStatus: Object.fromEntries(Object.entries(this.S.chunkStatus).filter(([day]) => next.includes(day))),
    });
    if (this.loaded != null) this.mergeChunks();
    this.mergeChips();
  }

  setDayStatus(day: string, status: ChunkStatus): void {
    if (this.windowDays && !this.windowDays.includes(day)) return;
    if (this.S.chunkStatus[day] === status) return;
    if (status === 'error' && this.dayLanding === day) this.dayLanding = null;
    this.set({ chunkStatus: { ...this.S.chunkStatus, [day]: status } });
    this.tl?.redraw();
  }

  /** One successful day snapshot replaces only that day; a neighboring response cannot replace the cursor. */
  upsertDayData(day: string, data: DayData): void {
    if (this.windowDays && !this.windowDays.includes(day)) return;
    const old = this.chunks.get(day);
    // A prefetched neighbor can become the selected day before the first day initializes. Its cached snapshot must
    // initialize this generation even when the source arrays haven't changed.
    if (
      this.loaded === this.S.gen &&
      old?.entries === data.entries &&
      old.segments === data.segments &&
      old.spans === data.spans
    )
      return;
    this.chunks.set(day, data);
    if (this.loaded == null) {
      if (day !== this.S.day) return;
      this.setDayData(day, this.S.gen, data);
    }
    this.mergeChunks();
    if (this.dayLanding === day) {
      this.dayLanding = null;
      this.seek(this.preferredTime(day, data));
    } else if (this.t != null && isoDay(new Date(this.t * 1000)) === day && this.awaitingMedia == null) {
      // A jump reached this chunk before its media listing. Place only cameras that have no source for that moment.
      for (const cam of this.availAt(this.t))
        if (this.P[cam]?.seg[this.P[cam]?.act ?? 0] !== segAt(this.segs[cam] ?? [], this.t))
          this.place(cam, this.t, true);
      this.followUi();
    }
  }

  private mergeChunks(): void {
    const audioLine = this.audioEntry == null ? null : this.entries[this.audioEntry];
    const entries = new Map<string, TranscriptEntry>();
    const segments = new Map<string, RawSegment>();
    const spans = new Map<string, AudioSpan>();
    for (const [day, data] of [...this.chunks].sort(([a], [b]) => a.localeCompare(b))) {
      if (this.windowDays && !this.windowDays.includes(day)) continue;
      for (const e of data.entries) entries.set(e.line ?? `${e.ts_start}:${String(e.device ?? '')}`, e);
      for (const s of data.segments) segments.set(`${s.cam}:${s.start}:${s.url}`, s);
      for (const s of data.spans) spans.set(`${s.start}:${s.end}:${s.device ?? ''}`, s);
    }
    this.entries = [...entries.values()].sort((a, b) => a.ts_start - b.ts_start);
    const joined: AudioSpan[] = [];
    for (const s of [...spans.values()].sort(
      (a, b) => (a.device ?? '').localeCompare(b.device ?? '') || a.start - b.start,
    )) {
      const last = joined.at(-1);
      if (last && last.device === s.device && s.start <= last.end) last.end = Math.max(last.end, s.end);
      else joined.push({ ...s });
    }
    this.spans = joined.sort((a, b) => a.start - b.start);
    if (audioLine) {
      const at = this.entries.findIndex(
        e =>
          (e.line && e.line === audioLine.line) || (e.ts_start === audioLine.ts_start && e.device === audioLine.device),
      );
      this.audioEntry = at < 0 ? null : at;
    }
    if (this.S.day)
      this.growSegments(
        this.S.day,
        this.S.gen,
        normalizeSegments([...segments.values()], MAX_DAY_SEGMENTS * Math.max(1, this.chunks.size)),
      );
    this.seen.cur = -2;
    this.set({ dataVersion: this.S.dataVersion + 1 });
    this.tl?.redraw();
  }

  setDayChips(day: string, chips: Chip[]): void {
    if (this.windowDays && !this.windowDays.includes(day)) return;
    if (this.chunkChips.get(day) === chips) return;
    this.chunkChips.set(day, chips);
    this.mergeChips();
  }

  private mergeChips(): void {
    const chips = new Map<string, Chip>();
    for (const values of this.chunkChips.values())
      for (const c of values) {
        const old = chips.get(c.id);
        if (!old)
          chips.set(c.id, {
            ...c,
            spans: [...c.spans],
            cams: c.cams ? { ...c.cams } : undefined,
            activities: c.activities ? [...c.activities] : undefined,
          });
        else {
          old.spans = presentSpans([{ ...c, spans: [...old.spans, ...c.spans] }], new Set([c.id]));
          old.total_s = old.total_s > 0 || c.total_s > 0 ? old.spans.reduce((n, [a, b]) => n + b - a, 0) : 0;
          old.count = old.spans.length;
          if (c.cams)
            for (const [cam, spans] of Object.entries(c.cams)) {
              old.cams ??= {};
              old.cams[cam] = presentSpans([{ ...c, spans: [...(old.cams[cam] ?? []), ...spans] }], new Set([c.id]));
            }
          if (c.activities)
            old.activities = [...new Map([...(old.activities ?? []), ...c.activities].map(a => [a.id, a])).values()];
        }
      }
    if (!this.S.day) return;
    const selected = new Set(this.S.sel);
    this.setChips(this.S.day, this.S.gen, [...chips.values()]);
    // Panning away from a selected subject must not silently remove that filter while its day is out of window.
    if (selected.size) this.setSel(new Set([...selected, ...this.S.sel]), false);
    this.set({ dataVersion: this.S.dataVersion + 1 });
  }
  /** The open day's lines, read again after a naming or a correction (#300): the same lines, under new names. The
   *  timeline redraws with them; nothing seeks or reloads. */
  relabel(day: string, gen: number, entries: TranscriptEntry[]): void {
    if (day !== this.S.day || gen !== this.S.gen || this.loaded !== gen || entries === this.entries) return;
    const chunk = this.chunks.get(day);
    if (!this.deps.embedded && chunk) {
      this.upsertDayData(day, { ...chunk, entries });
      return;
    }
    this.entries = entries;
    this.tl?.redraw();
  }
  /** The day's lines, video, and audio have loaded (or failed, as empty lists). Opens the day at the playhead from
   *  the link, else at the latest full minute of video, else at the end of its data. */
  setDayData(day: string, gen: number, data: DayData): void {
    if (day !== this.S.day || gen !== this.S.gen) return;
    if (this.loaded === gen) {
      if (!this.deps.embedded) this.upsertDayData(day, data);
      return;
    }
    this.loaded = gen;
    if (!this.deps.embedded) this.chunks.set(day, data);
    for (const c of this.cams) this.release(c);
    this.roomFailed.clear();
    this.entries = data.entries;
    this.spans = data.spans;
    this.segs = Object.create(null);
    for (const s of data.segments) {
      const seg: Segment = {
        cam: s.cam,
        start: s.start,
        end: s.end,
        bytes: 0,
        url: s.url,
        ...(s.expired ? { broken: 'expired' as const } : {}),
      };
      this.segs[s.cam] ??= [];
      this.segs[s.cam]?.push(seg);
    }
    for (const list of Object.values(this.segs)) list.sort((a, b) => a.start - b.start);
    this.seen = { cur: -2, avail: null, t: null };
    // Start on a view that fits the day's data: the whole day when it is spread out, else just the busy part.
    const ed = dataEdges(this.entries, this.allSegs(), this.spans);
    if (ed && ed[1] - ed[0] < 2 * 3600) this.setView(ed[0] - 300, ed[1] + 300);
    else this.setView(this.dayStart, dayBounds(day).t1);
    this.home = { ...this.view };
    const fromStart = this.playFromStartOnLoad;
    this.playFromStartOnLoad = false;
    const t = fromStart
      ? (ed?.[0] ?? null)
      : (this.t ?? latestVideo(this.segs, this.wall()) ?? (ed ? ed[1] - 1 : null)); // a time in the link wins
    if (t == null) {
      this.t = null;
      this.lastUi = 0;
      return;
    }
    const play = this.playOnLoad || fromStart;
    this.playOnLoad = false;
    this.seek(t, { play });
  }
  /** Open a bounded recording in an embedded player, using the same seek/play and segment lifecycle as Review. */
  openClip(start: number, segments: RawSegment[]): void {
    if (!this.deps.embedded) throw new Error('openClip requires an embedded player');
    this.pickDay(isoDay(new Date(start * 1000)));
    this.t = start;
    this.embeddedStart = start;
    this.embeddedEnded = false;
    this.playOnLoad = true;
    this.setDayData(this.S.day as string, this.S.gen, { entries: [], spans: [], segments });
  }
  /** A successful complete listing is authoritative. Reuse unchanged sources and their load failures, extend
   *  healthy growing ones, and replace vanished sources without keeping old hulls. */
  growSegments(day: string, gen: number, raw: readonly RawSegment[]): void {
    if (day !== this.S.day || gen !== this.S.gen || this.loaded !== gen) return;
    const next: Record<string, Segment[]> = {};
    const reload = new Set<string>();
    let changed = false;
    // The segments already known, by camera, stream, and start to the millisecond, so each listed one finds its old
    // self without a scan (a scan per segment was most of a listing's cost on a full day, the Review audit).
    const streamOf = (url: string) => url.split(':')[1] ?? '';
    const keyOf = (cam: string, url: string, ms: number) => `${cam}|${streamOf(url)}|${ms}`;
    const index = new Map<string, Segment[]>();
    for (const [cam, list] of Object.entries(this.segs))
      for (const v of list) {
        const k = keyOf(cam, v.url, Math.round(v.start * 1000));
        const at = index.get(k);
        if (at) at.push(v);
        else index.set(k, [v]);
      }
    const previous = (s: RawSegment): Segment | undefined => {
      const ms = Math.round(s.start * 1000);
      for (const d of [0, -1, 1]) {
        const hit = index
          .get(keyOf(s.cam, s.url, ms + d))
          ?.find(v => Math.abs(v.start - s.start) < 0.001 && (v.broken === 'expired') === (s.expired === true));
        if (hit) return hit;
      }
      return undefined;
    };
    for (const s of raw) {
      const known = previous(s);
      let segment: Segment;
      if (known && s.end >= known.end) {
        segment = known;
        if (s.end > known.end) {
          known.end = s.end;
          const p = this.P[s.cam];
          for (const i of [0, 1] as const) if (p && p.seg[i] === known) p.v[i]?.extend?.(known.end - known.start);
          changed = true;
        }
      } else {
        segment = { ...s, bytes: 0, ...(s.expired ? { broken: 'expired' as const } : {}) };
        changed = true;
      }
      const bucket = next[s.cam] ?? [];
      bucket.push(segment);
      next[s.cam] = bucket;
    }
    for (const list of Object.values(next)) list.sort((a, b) => a.start - b.start);
    for (const cam of new Set([...Object.keys(this.segs), ...Object.keys(next)])) {
      const before = this.segs[cam] ?? [],
        after = next[cam] ?? [];
      if (before.length !== after.length || before.some((s, i) => s !== after[i])) changed = true;
      if (this.P[cam]?.seg.some(s => s != null && !after.includes(s))) reload.add(cam);
    }
    if (!changed) return;
    this.segs = next;
    for (const cam of reload) {
      this.release(cam);
      if (this.t != null && segAt(next[cam] ?? [], this.t)) this.place(cam, this.t, true);
    }
    this.seen.avail = null;
    this.lastUi = 0;
    this.followUi();
    this.tl?.redraw();
  }
  /** The day's chips, first or refreshed. A refresh keeps the selection (less any chip that went away); a day picked
   *  again starts with none selected. */
  setChips(day: string, gen: number, chips: Chip[]): void {
    if (day !== this.S.day || gen !== this.S.gen) return;
    if (this.scrubbing) {
      this.pendingChips = chips;
      return;
    }
    const first = this.chipsGen !== gen;
    this.chipsGen = gen;
    this.chips = chips;
    const selected = first ? [] : [...this.S.sel].filter(id => chips.some(c => c.id === id));
    for (const name of this.S.people) {
      const chip = chips.find(c => c.kind === 'person' && c.label.toLocaleLowerCase() === name.toLocaleLowerCase());
      if (chip && !selected.includes(chip.id)) selected.push(chip.id);
    }
    this.setSel(selected, false);
  }

  // ---- video, per camera ----
  /** Attach (or detach, with null) one of a camera's two <video>s. Returns a function that detaches it. */
  bindVideo(c: string, i: 0 | 1, el: VideoLike | null): () => void {
    if (el) {
      el.defaultPlaybackRate = this.S?.speed ?? 1;
      el.playbackRate = this.S?.speed ?? 1;
    }
    this.P[c] ??= {
      v: [null, null],
      seg: [null, null],
      act: 0,
      busy: false,
      epoch: 0,
      want: null,
      last: 0,
      timer: undefined,
      safety: undefined,
      playing: [false, false],
    };
    const p = this.P[c];
    p.v[i] = el;
    if (!el) return () => {};
    el.toggleAttribute('data-standby', i !== p.act);
    // A file that won't load is either still being written (the newest segment) or was cut off by a restart.
    const onError = () => {
      const seg = p.seg[i];
      if (!seg || !el.error || seg.broken) return;
      const list = this.segs[c] ?? [];
      seg.broken = seg === list.at(-1) && this.wall() - seg.start < 600 ? 'recording' : 'cut';
      p.busy = false;
      clearTimeout(p.safety);
      this.pump(c); // a seek queued behind the failed one goes on
      this.tl?.redraw();
      this.seen.avail = null; // work out again which cameras can show this moment
      this.lastUi = 0;
    };
    const onMeta = () => {
      const seg = p.seg[i];
      if (seg && Number.isFinite(el.duration) && seg.start + el.duration < seg.end) {
        seg.end = seg.start + el.duration; // the index only knows starts; the file knows its length
        this.tl?.redraw();
      }
    };
    // The room audio starts when the lead camera's picture does, not when its own download finishes (#244).
    const onPlaying = () => {
      p.playing[i] = true;
      if (c === this.lead && i === p.act && this.room) this.startRoom(this.room, true);
    };
    const onStopped = () => {
      p.playing[i] = false;
    };
    el.addEventListener('error', onError);
    el.addEventListener('loadedmetadata', onMeta);
    el.addEventListener('playing', onPlaying);
    for (const type of ['pause', 'waiting', 'emptied']) el.addEventListener(type, onStopped);
    // A player mounted after the engine was already told where to be (a popup's, drawn in a portal after its effects
    // ran, #2423) gets that moment now: a seek before a camera had a video placed nothing, so it would stay blank.
    // place() needs both of the camera's videos, so this runs on the bind that completes the pair.
    if (p.v[0] && p.v[1] && !p.seg[0] && !p.seg[1] && this.t != null && segAt(this.segs[c] ?? [], this.t)) {
      this.place(c, this.t, true);
    }
    return () => {
      el.removeEventListener('error', onError);
      el.removeEventListener('loadedmetadata', onMeta);
      el.removeEventListener('playing', onPlaying);
      for (const type of ['pause', 'waiting', 'emptied']) el.removeEventListener(type, onStopped);
      if (p.v[i] === el) p.v[i] = null;
    };
  }
  private hold(c: string, i: 0 | 1, seg: Segment): void {
    const p = this.P[c];
    const v = p?.v[i];
    if (!p || !v || p.seg[i] === seg) return;
    p.seg[i] = seg;
    v.preload = 'auto';
    v.src = seg.url;
  }
  private swap(c: string): void {
    const p = this.P[c];
    if (!p) return;
    p.v[p.act]?.pause();
    p.v[p.act]?.toggleAttribute('data-standby', true);
    p.act = other(p.act);
    p.v[p.act]?.toggleAttribute('data-standby', false);
  }
  /** Ask camera c to show ts. Only the latest request matters; exact ones go at once and seek precisely. */
  private place(c: string, ts: number, exact: boolean): void {
    const p = this.P[c];
    if (!p?.v[0] || !p.v[1]) return;
    p.want = { ts, exact: exact || !!p.want?.exact };
    this.pump(c);
  }
  private pump(c: string): void {
    const p = this.P[c];
    if (!p || p.busy || !p.want) return;
    const wait = SEEK_MS - (performance.now() - p.last);
    if (wait > 0 && !p.want.exact) {
      clearTimeout(p.timer);
      p.timer = setTimeout(() => this.pump(c), wait);
      return;
    }
    const { ts, exact } = p.want;
    p.want = null;
    p.last = performance.now();
    const list = this.segs[c] ?? [];
    const seg = segAt(list, ts);
    if (!seg || seg.broken) {
      p.v[p.act]?.pause();
      return;
    }
    if (p.seg[p.act] !== seg) {
      if (p.seg[other(p.act)] === seg) this.swap(c);
      else this.hold(c, p.act, seg);
    }
    const v = p.v[p.act];
    if (!v) return;
    const off = Math.max(0, ts - seg.start);
    p.busy = true;
    const epoch = p.epoch;
    const done = () => {
      if (!p.busy || p.epoch !== epoch) return;
      p.busy = false;
      clearTimeout(p.safety);
      this.pump(c);
    };
    p.safety = setTimeout(() => {
      if (p.epoch !== epoch) return;
      done();
      if (this.videoMayPlay() && !p.want) v.play().catch(() => {}); // a seek that never reported back
    }, SAFETY_MS);
    const go = () => {
      if (p.epoch !== epoch) return;
      // Seek first, play once the seek lands: play() issued while a seek is pending is aborted by the browser
      // (AbortError), which left playback on with every video paused, so the next Space paused nothing (2026-10-24).
      const after = () => {
        if (p.epoch !== epoch) return;
        done();
        if (this.videoMayPlay()) {
          v.play().catch(() => {});
          // A seek within what's buffered doesn't always fire playing again, so line the room audio up here too.
          if (c === this.lead && this.room) this.startRoom(this.room, true);
        }
      };
      v.addEventListener('seeked', after, { once: true });
      try {
        if (!exact && v.fastSeek) v.fastSeek(off);
        else v.currentTime = off;
      } catch {
        after();
      }
      if (!this.videoMayPlay()) v.pause();
    };
    if (v.readyState >= 1) go();
    else v.addEventListener('loadedmetadata', go, { once: true });
    // Preload the next segment into the hidden <video> so playback rolls over without a stall.
    const nx = list[list.indexOf(seg) + 1];
    if (nx && !nx.broken && nx.start - seg.end < 5 && p.seg[other(p.act)] !== nx) this.hold(c, other(p.act), nx);
  }
  // A camera with no picture at this moment lets go of its media: a paused <video> keeps its connection (and
  // buffer), and the browser allows about six per host, so leftover ones starved other requests (2026-10-25).
  private release(c: string): void {
    const p = this.P[c];
    if (!p) return;
    p.epoch++;
    p.busy = false;
    clearTimeout(p.timer);
    clearTimeout(p.safety);
    p.v.forEach((v, i) => {
      if (!v) return;
      v.pause();
      if (v.getAttribute('src')) {
        v.removeAttribute('src');
        v.load();
      }
      p.seg[i] = null;
    });
    p.want = null;
  }
  private pauseVideo(): void {
    for (const c of this.cams) for (const v of this.P[c]?.v ?? []) v?.pause();
  }
  private videoMayPlay(): boolean {
    return this.playing && !this.scrubbing && !this.S.waiting && this.personPlayback.run?.mode !== 'audio';
  }

  // ---- the <audio>: started so the browser allows it, and never failing silently ----
  /** Play the <audio>. A refusal is logged with the browser's reason, and one the viewer can fix (no gesture yet)
   *  says so on the status line; a play() our own pause or load interrupted is normal and only logged. */
  private playAudio(audio: AudioLike, unlock = false): void {
    const token = this.audioToken;
    const queuedMoment = this.personPlayback.current;
    audio.play().then(
      () => {
        if (queuedMoment && (token !== this.audioToken || this.personPlayback.current !== queuedMoment)) return;
        // The sound plays now: take down whatever this said about it failing (and nothing else on the status line).
        if (!unlock && this.soundNotice && this.S.notice === this.soundNotice) this.set({ notice: null });
        if (!unlock) this.soundNotice = null;
      },
      (e: unknown) => {
        if (queuedMoment && (token !== this.audioToken || this.personPlayback.current !== queuedMoment)) return;
        const name = e && typeof e === 'object' && 'name' in e ? String(e.name) : 'Error';
        const message = e && typeof e === 'object' && 'message' in e ? String(e.message) : String(e);
        if (name === 'AbortError') {
          console.debug('Review: a play() of the sound was interrupted', name, message);
          return;
        }
        if (name === 'NotAllowedError') this.unlocked = false; // the next click unlocks it again
        // Still blocked: the status line already says so, and the retry every half second shouldn't flood the log.
        if (name === 'NotAllowedError' && this.S.notice === BLOCKED) return;
        console.warn('Review: the browser refused to play the sound', name, message);
        const notice = name === 'NotAllowedError' ? BLOCKED : unlock ? null : `Couldn't play the sound (${name}).`;
        if (notice) {
          this.soundNotice = notice;
          this.set({ notice });
        }
        if (!unlock && this.personPlayback.run?.mode === 'audio') this.personPlayback.stop();
      },
    );
  }
  /** Inside the click that starts playback, play silence on an <audio> that has nothing loaded yet, so a browser
   *  that wants a gesture allows the room audio or line that arrives after the download. */
  private unlockAudio(): void {
    const audio = this.audio;
    if (!audio || this.unlocked) return; // once per element: a browser that asks for a gesture asks once
    this.unlocked = true;
    if (!audio.src) {
      audio.src = SILENCE;
      this.playAudio(audio, true);
      return;
    }
    // It already holds sound (room audio preloaded while paused, #344): play and pause it inside the click, so a
    // browser that lets an element start only from a gesture (Safari) allows it later, when the picture moves.
    if (audio.paused) {
      audio.play().catch(() => {}); // the pause just below interrupts it
      audio.pause();
    }
  }

  // ---- where there's no video: play the lines one after another ----
  /** Fetch url for the <audio>, dropping the fetch before it, and hand the blob to then unless something newer was
   *  asked for meanwhile. */
  private fetchAudio(url: string, then: (blob: Blob, token: number) => void, failed: () => void): number {
    this.audioFetch?.abort();
    const ctl = new AbortController();
    this.audioFetch = ctl;
    const token = ++this.audioToken;
    (this.deps.fetchBlob ?? fetchBlob)(url, ctl.signal).then(
      blob => {
        if (token !== this.audioToken) return; // the playhead moved on while it loaded
        this.audioFetch = null;
        then(blob, token);
      },
      () => {
        if (token === this.audioToken) failed();
      },
    );
    return token;
  }
  /** Play line i's clip from the playhead's place in it (from), or from its start. */
  private playLine(i: number, from?: number): void {
    const e = this.entries[i];
    const audio = this.audio;
    if (!e || !audio) return;
    // Starting inside the line keeps the playhead where it was sent: putting it back at the line's start made
    // Skip the rest send it forward again every frame, fetching the clip each time (#182).
    const offset = Math.max(0, Math.min((from ?? e.ts_start) - e.ts_start, lineEnd(e) - e.ts_start));
    this.audioEntry = i;
    this.room = null;
    this.t = e.ts_start + offset;
    this.fetchAudio(
      `/clip?ts_start=${encodeURIComponent(e.ts_start)}`,
      (blob, token) => {
        this.loadedRoom = null;
        loadBlob(audio, blob);
        audio.addEventListener(
          'loadedmetadata',
          () => {
            if (token !== this.audioToken) return;
            if (offset > 0) audio.currentTime = offset + CLIP_LEAD_S;
            audio.playbackRate = this.S.speed; // a line plays at the chosen speed, whatever the room audio was eased to
            this.playAudio(audio);
          },
          { once: true },
        );
      },
      () => {
        this.audioEntry = null;
      },
    );
  }
  /** Play the line at ts, from ts; or the next one, when ts falls after a line has ended. */
  private playAudioAt(ts: number): void {
    let i = Math.max(0, lineAt(this.entries, ts));
    const e = this.entries[i];
    if (e && lineEnd(e) <= ts) i++;
    this.playLine(i, ts);
  }
  /** The <audio> finished: play the next line while there's still no video, or the room's audio once there is. */
  audioEnded(): void {
    if (this.audio?.src === SILENCE) return; // the unlocking silence, not a line or the room audio
    if (this.personPlayback.run?.mode === 'audio' && this.audio?.ended === false) return;
    if (this.personPlayback.audioEnded()) return;
    const next = this.audioEntry == null ? null : this.audioEntry + 1;
    if (this.avail.length) {
      // Video began while a line played: the room's audio takes over under it (#183).
      this.audioEntry = null;
      if (this.playing && this.t != null && !this.room) this.roomAt(this.t);
    } else if (this.room && this.audioEntry == null) this.afterRoom();
    else if (this.playing && next != null) {
      // On to the next line, or into kept room audio that comes first (#196), or the next video.
      const done = this.entries[next - 1];
      const end = done ? lineEnd(done) : (this.t ?? 0);
      const nt = this.keptAt(end) ? end : nextMedia(this.allSegs(), this.entries, end, this.kept());
      if (this.pendingAfter(end, nt ?? undefined)) {
        this.waitForMore(end);
        return;
      }
      if (nt == null) {
        if (!this.waitForMore(end)) this.setPlaying(false);
      } else this.seek(nt, { play: true, centre: false });
    } else this.setPlaying(false);
  }
  /** The room's kept audio ran out where there's no video: on to whatever plays next, or stop. */
  private afterRoom(): void {
    const room = this.room;
    if (!room || !this.playing) return;
    const end = room.end;
    // A kept stretch that goes on from here (or starts a moment later), or a video at its end, plays on from there.
    const next = this.kept().find(span => span.start <= end + ROOM_JOIN_S && span.end > end);
    const nt = next
      ? Math.max(end, next.start)
      : this.availAt(end).length
        ? end
        : nextMedia(this.allSegs(), this.entries, end, this.kept());
    if (this.pendingAfter(end, nt ?? undefined)) {
      this.waitForMore(end);
      return;
    }
    if (nt == null) {
      if (!this.waitForMore(end)) this.setPlaying(false);
    } else this.seek(nt, { play: true, centre: false });
  }
  private stopAudio(): void {
    if (this.personAudioError) this.audio?.removeEventListener?.('error', this.personAudioError);
    this.personAudioError = null;
    // pause() on an element that is mid-load but not playing wedged Chromium's media loader (the room audio under
    // video never reached loadedmetadata, 2026-10-24), so only pause one that is playing.
    try {
      if (this.audio && !this.audio.paused) this.audio.pause();
    } catch {
      // nothing was playing
    }
    this.audioEntry = null;
    this.room = null;
    // Whatever was on its way is no longer wanted: a clip that arrives (or finishes loading) after a pause must not
    // start playing.
    this.audioFetch?.abort();
    this.audioFetch = null;
    this.audioToken++;
  }

  // ---- the room's sound under recorded video: one stretch per video segment, kept in step with the picture ----
  private roomAt(ts: number): void {
    if (this.personPlayback.run?.mode === 'audio') return;
    const audio = this.audio;
    if (!audio) return;
    // Under video, the room audio follows the lead camera's segments; where there's none, the kept audio's own
    // stretches (#196).
    const found = segAt(this.lead ? (this.segs[this.lead] ?? []) : this.spans, ts);
    if (!found) return;
    const seg = roomWindow(found, ts, this.deps.roomWindowS);
    const same = (r: { start: number; end: number } | null) => r && r.start === seg.start && r.end === seg.end;
    if (same(this.room)) {
      if (audio.readyState >= 1) this.startRoom(this.room as { start: number; end: number });
      return;
    }
    this.audioEntry = null;
    // The <audio> still holds this segment's room audio from before a pause or a seek: play it again, rather than
    // fetching the whole stretch each time (#184).
    if (same(this.loadedRoom)) {
      const room = this.loadedRoom as { start: number; end: number };
      this.room = room;
      this.audioFetch?.abort();
      this.audioToken++;
      if (audio.readyState >= 1) this.startRoom(room);
      else audio.addEventListener('loadedmetadata', () => this.startRoom(room, true), { once: true });
      return;
    }
    const room = { start: seg.start, end: seg.end };
    this.room = room;
    this.audioFetch?.abort();
    this.audioFetch = null;
    const token = ++this.audioToken;
    this.dropAhead(room); // what's still downloading for a stretch the playhead left
    // Usually already here: fetched while paused at this spot, or ahead of the stretch before it (#344).
    this.roomBlob(room).then(
      blob => {
        if (token !== this.audioToken || this.room !== room) return;
        this.loadedRoom = room;
        loadBlob(audio, blob);
        audio.addEventListener('loadedmetadata', () => this.startRoom(room, true), { once: true });
        audio.load();
      },
      () => {
        if (token !== this.audioToken || this.room !== room) return;
        this.room = null; // no room audio here: under video, the picture plays on alone
        if (this.lead) return;
        // With no picture, this stretch's lines play one by one instead, and it isn't tried again today (#196).
        this.roomFailed.add(room.start);
        if (this.S.roomOnly) this.set({ roomOnly: false });
        if (this.playing && this.t != null) this.playAudioAt(this.t);
      },
    );
  }
  /** The room audio for stretch r, fetched once and kept for a few stretches, so Play and the next stretch start with
   *  their picture rather than a second after it (#344). */
  private roomBlob(r: Span2): Promise<Blob> {
    const key = roomKey(r);
    if (this.aheadFailed.has(key)) return Promise.reject(new Error('no room audio there'));
    let got = this.ahead.get(key);
    if (got) {
      this.ahead.delete(key); // used again: the most recent, last to go
      this.ahead.set(key, got);
    } else {
      const ctl = new AbortController();
      const entry: Ahead = { p: (this.deps.fetchBlob ?? fetchBlob)(roomUrl(r), ctl.signal), ctl, done: false };
      entry.p.then(
        () => {
          entry.done = true;
        },
        () => {
          if (this.ahead.get(key) !== entry) return;
          this.ahead.delete(key);
          // Not asked for again today: a stretch with no room audio would otherwise be fetched every frame of the
          // last AHEAD_S before it. An abort (the playhead moved on) isn't a failure.
          if (!entry.ctl.signal.aborted) this.aheadFailed.add(key);
        },
      );
      this.ahead.set(key, entry);
      // Over the limit, the least recently used finished stretch goes; never the one playing, nor one still coming.
      for (const [k, e] of this.ahead) {
        if (this.ahead.size <= AHEAD_KEEP) break;
        if (!e.done || k === key || (this.room && k === roomKey(this.room))) continue;
        this.ahead.delete(k);
      }
      got = entry;
    }
    return got.p;
  }
  /** Drop the downloads still running for stretches other than r and the one after it: the playhead moved on (#184). */
  private dropAhead(r: Span2): void {
    const keep = new Set([roomKey(r)]);
    const next = this.roomSegAt(r.end + 0.5);
    if (next) keep.add(roomKey(next));
    for (const [k, e] of this.ahead)
      if (!e.done && !keep.has(k)) {
        e.ctl.abort();
        this.ahead.delete(k);
      }
  }
  /** The stretch of room audio at ts: the lead camera's segment under video, else the kept audio's own (#196). */
  private roomSegAt(ts: number): Span2 | null {
    const lead = this.availAt(ts)[0];
    const seg = segAt(lead ? (this.segs[lead] ?? []) : this.spans, ts);
    return seg ? roomWindow(seg, ts, this.deps.roomWindowS) : null;
  }
  /** Paused at ts: have its room audio in the <audio> already, so Play starts it with the picture (#344). It waits
   *  PRELOAD_MS, so clicking from line to line while paused fetches only where it stops. */
  private preloadRoom(ts: number): void {
    clearTimeout(this.preloadTimer);
    this.preloadTimer = setTimeout(() => this.preloadNow(ts), PRELOAD_MS);
  }
  private preloadNow(ts: number): void {
    const audio = this.audio;
    const room = this.roomSegAt(ts);
    if (!audio || !room || this.playing) return;
    if (this.loadedRoom && roomKey(this.loadedRoom) === roomKey(room)) return;
    this.dropAhead(room); // a spot clicked away from while its audio still came
    const token = ++this.audioToken;
    this.roomBlob(room).then(
      blob => {
        if (token !== this.audioToken || this.playing) return; // played or moved meanwhile: roomAt has it from here
        this.loadedRoom = room;
        loadBlob(audio, blob);
        audio.load();
      },
      () => {},
    );
  }

  /** Play the room audio from the playhead's place in it, once the lead camera's picture is moving. exact lines it up
   *  even when it is only slightly off (the picture has just started). */
  private startRoom(room: { start: number; end: number }, exact = false): void {
    const audio = this.audio;
    // The <audio> must hold this stretch: until its download lands it still holds the last one (or a line's clip).
    if (!audio || !this.playing || this.room !== room || this.loadedRoom !== room || this.t == null) return;
    if (audio.readyState < 1) return;
    const lead = this.lead ? this.P[this.lead] : undefined;
    if (lead && (lead.busy || lead.want || !lead.playing[lead.act])) return; // its playing event starts it
    const at = Math.max(0, this.t - room.start);
    if (exact || Math.abs(audio.currentTime - at) > DRIFT_S) audio.currentTime = at;
    audio.playbackRate = this.S.speed;
    if (audio.paused) this.playAudio(audio);
  }
  private syncRoom(): void {
    const audio = this.audio;
    if (!audio || !this.room || !this.playing || this.t == null) return;
    // Near the end of this stretch, fetch the next one, so it starts with its picture (#344).
    if (this.room.end - this.t < AHEAD_S) {
      const next = this.roomSegAt(this.room.end + 0.5);
      if (next) void this.roomBlob(next).catch(() => {});
    }
    const at = this.t - this.room.start;
    if (at < 0 || at > this.room.end - this.room.start) {
      this.roomAt(this.t);
      return;
    }
    if (audio.readyState < 2 || this.loadedRoom !== this.room) return; // still downloading this stretch
    const lead = this.lead ? this.P[this.lead] : undefined;
    if (lead && !lead.playing[lead.act]) {
      if (!audio.paused) audio.pause(); // the picture stalled: hold the sound until its frames come again
      return;
    }
    const drift = audio.currentTime - at; // above 0: the sound is ahead of the picture
    if (Math.abs(drift) > DRIFT_S) {
      audio.currentTime = at;
      audio.playbackRate = this.S.speed;
    } else if (Math.abs(drift) < NUDGE_MIN_S) audio.playbackRate = this.S.speed;
    // Eased toward the picture around the chosen speed (#244, #327).
    else audio.playbackRate = this.S.speed * (1 - Math.max(-NUDGE_MAX, Math.min(NUDGE_MAX, drift * NUDGE_GAIN)));
    if (audio.paused) this.playAudio(audio);
  }

  /** The day's kept room audio that has loaded or not been tried (one that failed is left to the lines). */
  private keptAt(ts: number): AudioSpan | null {
    const span = segAt(this.spans, ts);
    return span && !this.roomFailed.has(span.start) ? span : null;
  }
  private kept(): AudioSpan[] {
    return this.spans.filter(span => !this.roomFailed.has(span.start));
  }

  private availAt(ts: number): string[] {
    return this.cams.filter(c => {
      const s = segAt(this.segs[c] ?? [], ts);
      return s && !s.broken;
    });
  }
  /** Why there's no video at ts when a segment covers it: 'recording', 'cut', or 'expired', else null. */
  private brokenAt(ts: number): 'recording' | 'cut' | 'expired' | null {
    for (const c of this.cams) {
      const b = segAt(this.segs[c] ?? [], ts)?.broken;
      if (b) return b;
    }
    return null;
  }

  // ---- moving through time ----
  private setPlaying(b: boolean): void {
    this.playing = b;
    if (this.S.playing !== b) this.set({ playing: b });
    if (!b && this.S.roomOnly) this.set({ roomOnly: false });
    if (!b) {
      this.awaitingMedia = null;
      this.awaitingTarget = false;
      if (this.S.waiting) this.set({ waiting: false });
      this.pauseVideo();
      this.stopAudio();
    }
  }
  /** Move the playhead to ts, playing from there when play is true (anything else stops playback). centre (the
   *  default) scrolls the transcript's line to the middle. */
  seek(ts: number, { play = false, centre = true }: { play?: boolean; centre?: boolean } = {}): void {
    if (!this.personJump) this.personPlayback.cancel();
    if (!this.personPlayback.run && this.S.skip && this.S.sel.size && this.present?.length === 0) play = false;
    this.navigationVersion++;
    this.dayLanding = null;
    this.awaitingMedia = null;
    this.awaitingTarget = false;
    if (this.S.waiting) this.set({ waiting: false });
    this.embeddedEnded = false;
    if (play) stopClip(); // one sound at a time: a clip playing from another panel stops
    this.t = this.clampT(ts);
    this.contextAt(this.t);
    this.keepInView(this.t);
    this.playing = play;
    if (this.S.playing !== play) this.set({ playing: play });
    this.stopAudio();
    // Only an <audio> that has never been loaded gets the silence, and the first seek that plays always comes from a
    // click or a key, so it runs inside that gesture.
    if (play) this.unlockAudio();
    if (play && !this.deps.embedded && !this.chunks.has(this.S.day ?? '')) {
      this.pauseVideo();
      this.avail = [];
      this.lead = null;
      this.seen.avail = null;
      this.awaitingMedia = this.t;
      this.awaitingTarget = true;
      this.set({ waiting: true, t: this.t, avail: [], roomOnly: false });
      this.writeHash();
      this.lastUi = 0;
      return;
    }
    const avail = this.availAt(this.t);
    // Lead with this moment's cameras at once: the next frame would otherwise read the old lead's video time and drag
    // the playhead back (a summary link to a moment without video snapped back to where it was).
    this.avail = avail;
    this.lead = this.personCamera(avail);
    for (const c of this.cams) {
      if (avail.includes(c)) this.place(c, this.t, true);
      else this.release(c);
    }
    // These cameras are placed for this moment: the next frame mustn't place them again (#299).
    this.seen.avail = avail.join(',');
    if (!sameSet(avail, this.S.avail)) this.set({ avail });
    const roomOnly = play && !avail.length && this.keptAt(this.t) != null;
    if (this.S.roomOnly !== roomOnly) this.set({ roomOnly });
    if (play && !avail.length) {
      // No picture: the room's kept audio when there is some here, else the lines one by one (#196).
      if (roomOnly) this.roomAt(this.t);
      else this.playAudioAt(this.t);
    } else if (play) this.roomAt(this.t);
    if (play) clearTimeout(this.preloadTimer);
    else this.preloadRoom(this.t);
    this.setTranscriptCurrent(this.t, centre);
    if (!this.deps.embedded) {
      this.set({ t: this.t });
      this.writeHash();
    }
    this.lastUi = 0; // follow at once
  }
  /** Move the playhead to ts and play from there. */
  seekPlay(ts: number): void {
    this.seek(ts, { play: true });
  }
  /** Move the playhead to ts, still playing if it was. */
  jump(ts: number): void {
    this.seek(ts, { play: this.playing });
  }
  togglePlay(): void {
    this.personPlayback.cancel();
    if (this.t == null) {
      if (this.loaded !== this.S.gen) {
        this.playFromStartOnLoad = !this.playFromStartOnLoad;
        if (this.playFromStartOnLoad) this.unlockAudio();
        return;
      }
      const first = dataEdges(this.entries, this.allSegs(), this.spans)?.[0];
      if (first != null) this.seek(first, { play: true });
      return;
    }
    if (this.playing) this.setPlaying(false);
    else this.seek(this.embeddedEnded ? (this.embeddedStart ?? this.t) : this.t, { play: true, centre: false });
  }
  pause(): void {
    this.personPlayback.cancel();
    this.setPlaying(false);
  }

  /** Queue jumps keep their immutable selection; public seeks always take over and cancel it. */
  private personSeek(ts: number, play: boolean): void {
    this.personJump = true;
    try {
      this.seek(ts, { play, centre: false });
    } finally {
      this.personJump = false;
    }
  }
  private personCamera(available: readonly string[]): string | null {
    const run = this.personPlayback.run;
    if (run?.mode === 'audio') return null;
    if (run?.mode === 'video') {
      const cam = this.personPlayback.current?.cam;
      return cam && available.includes(cam) ? cam : null;
    }
    return available[0] ?? null;
  }
  private skipPersonMoment(notice: string): void {
    const current = this.personPlayback.current;
    this.set({ notice });
    // Missing recordings can be consecutive. A microtask avoids recursion through a long unavailable queue and
    // lets a manual cancellation invalidate this transition before it runs.
    queueMicrotask(() => {
      if (current && this.personPlayback.current === current) this.personPlayback.next();
    });
  }
  private playPersonMoment(moment: PlaybackMoment, mode: PersonPlaybackMode): void {
    if (this.personPlayback.run?.index === 0) this.set({ notice: null });
    if (mode === 'audio') {
      this.personSeek(moment.start, false);
      clearTimeout(this.preloadTimer);
      stopClip();
      const audio = this.audio;
      if (!audio) {
        this.personPlayback.stop();
        this.set({ notice: 'Speech playback is unavailable.' });
        return;
      }
      this.unlockAudio();
      this.setPlaying(true);
      this.pauseVideo();
      this.audioEntry = null;
      this.room = null;
      this.fetchAudio(
        `/clip?ts_start=${encodeURIComponent(moment.start)}`,
        (blob, token) => {
          this.loadedRoom = null;
          loadBlob(audio, blob);
          audio.addEventListener(
            'loadedmetadata',
            () => {
              if (token !== this.audioToken) return;
              audio.currentTime = CLIP_LEAD_S;
              audio.playbackRate = this.S.speed;
              this.playAudio(audio);
            },
            { once: true },
          );
          this.personAudioError = () => {
            if (token === this.audioToken) this.skipPersonMoment('This speech recording is unavailable.');
          };
          audio.addEventListener('error', this.personAudioError, { once: true });
        },
        () => this.skipPersonMoment('This speech recording is unavailable.'),
      );
      return;
    }
    if (!moment.cam || !this.cams.includes(moment.cam)) {
      this.skipPersonMoment('This camera recording is unavailable.');
      return;
    }
    if (this.S.cameras.length && !this.S.cameras.includes(moment.cam)) {
      this.set({ cameras: this.cams.filter(cam => cam === moment.cam || this.S.cameras.includes(cam)) });
      this.setSel(this.S.sel, false);
    }
    this.set({ camera: moment.cam, cameraIntent: (this.S.cameraIntent ?? 0) + 1 });
    this.personVideoAt(moment, moment.start);
  }
  private personVideoAt(moment: PlaybackMoment, from: number): void {
    const day = isoDay(new Date(from * 1000));
    if (!this.deps.embedded && !this.chunks.has(day)) {
      if (this.S.chunkStatus[day] === 'error') this.skipPersonMoment('This camera recording could not be loaded.');
      else this.personSeek(from, true); // Publish the exact date/time so the window loader requests its chunk.
      return;
    }
    const segment = (this.segs[moment.cam ?? ''] ?? []).find(s => !s.broken && s.end > from && s.start < moment.end);
    if (!segment || !moment.cam || !this.cams.includes(moment.cam)) {
      // The current day's index can be complete while a continuation in a neighboring day is still unknown.
      const pending = [...new Set([calendarStep(day, 1), ...(this.windowDays ?? [])])].sort().find(d => {
        const bounds = dayBounds(d);
        return (
          !this.chunks.has(d) &&
          this.S.chunkStatus[d] !== 'error' &&
          bounds.t1 > from &&
          bounds.t0 < moment.end &&
          bounds.t0 <= this.wall()
        );
      });
      if (pending) {
        this.personSeek(Math.max(from, dayBounds(pending).t0), true);
        return;
      }
      this.skipPersonMoment('This camera recording is unavailable.');
      return;
    }
    this.personSeek(Math.max(from, segment.start), true);
  }

  // ---- chips: select people, objects, and events; mark, zoom to, and step through their time ----
  private setSel(ids: Iterable<string>, zoom = true): void {
    const sel = new Set(ids);
    const colors = chipColors(this.chips, sel);
    this.marks = this.filteredChips
      .filter(c => sel.has(c.id))
      .map(c => ({ color: colors[c.id] ?? '', spans: c.spans }));
    this.present = facetedSpans(this.filteredChips, sel, this.chips);
    this.set({
      sel,
      colors,
      filterStatus: !sel.size ? null : this.present === null ? 'unknown' : !this.present.length ? 'empty' : null,
    });
    if (!this.personPlayback.run && this.playing && this.S.skip && sel.size && this.present?.length === 0)
      this.setPlaying(false);
    this.tl?.redraw();
    if (!zoom) return;
    const first = this.present?.[0];
    const last = this.present?.at(-1);
    if (first && last) {
      const pad = Math.max(30, (last[1] - first[0]) * 0.08);
      this.glideTo(first[0] - pad, last[1] + pad);
    } else if (this.home) this.glideTo(this.home.t0, this.home.t1);
  }
  toggleChip(id: string): void {
    const next = new Set(this.S.sel);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    this.setSel(next);
    const chip = this.chips.find(c => c.id === id && c.kind === 'person' && namedPerson(c.label));
    if (chip) {
      if (this.personPlayback.run) this.pause();
      const people = this.S.people.filter(name => name.toLocaleLowerCase() !== chip.label.toLocaleLowerCase());
      if (next.has(id)) people.push(chip.label);
      this.set({ people, person: people.at(-1) ?? '' });
      this.writeHash();
    }
  }
  clearChips(): void {
    if (this.personPlayback.run) this.pause();
    this.set({ people: [], person: '' });
    this.setSel([]);
    this.writeHash();
  }
  /** Go to the next (dir 1) or previous (dir -1) stretch of the selected chips' time. */
  stepChip(dir: 1 | -1): void {
    if (!this.present?.length) return;
    const ts = this.t ?? this.view.t0;
    const nt = dir > 0 ? nextSpanStart(this.present, ts) : prevSpanStart(this.present, ts);
    if (nt != null) this.seek(nt, { play: this.playing });
  }
  /** Mute or unmute the room's sound; the video goes on playing. It holds for an element attached later too. */
  setMuted(muted: boolean): void {
    this.set({ muted });
    if (this.audio) this.audio.muted = muted;
  }

  /** Play at this speed (1, 1.5, or 2): every camera's video, and the room audio or line under it (#327). */
  setSpeed(speed: number): void {
    this.set({ speed });
    for (const p of Object.values(this.P))
      for (const v of p.v)
        if (v) {
          v.defaultPlaybackRate = speed; // kept when a new segment loads into it
          v.playbackRate = speed;
        }
    if (this.audio) this.audio.playbackRate = speed;
  }
  toggleSkip(): void {
    const skip = !this.S.skip;
    writePrefs({ ...readPrefs(), reviewSkip: skip });
    this.set({ skip });
    if (!this.personPlayback.run && skip && this.S.sel.size && this.present?.length === 0) this.setPlaying(false);
    this.tl?.redraw();
  }

  // ---- scrubbing (from the timeline) ----
  scrubStart(ts: number): void {
    this.personPlayback.cancel();
    this.scrubbing = true;
    this.wasPlaying = this.playing;
    if (this.playing) this.setPlaying(false);
    this.scrubMove(ts);
  }
  scrubMove(ts: number): void {
    this.t = this.clampT(ts);
    this.contextAt(this.t);
    for (const c of this.availAt(this.t)) this.place(c, this.t, false);
  }
  scrubEnd(ts?: number): void {
    if (!this.scrubbing) return;
    this.scrubbing = false;
    this.seek(ts ?? this.t ?? this.dayStart, { play: this.wasPlaying });
    if (this.pendingChips && this.S.day) {
      const chips = this.pendingChips;
      this.pendingChips = null;
      this.setChips(this.S.day, this.S.gen, chips);
    }
  }

  // ---- the loop ----
  /** One animation frame (exported for tests, which step time themselves). */
  frame(now: number): void {
    if (this.anim) this.glide(now);
    if (this.t != null && this.playing && !this.scrubbing) this.advance(now);
    this.tl?.frame();
    // Setting textContent replaces the text even when it's the same, which costs a layout and a paint every frame,
    // playing or not (#486): write only a new time.
    const clock = this.t != null ? hms(this.t) : '';
    if (this.clock && this.clock.textContent !== clock) this.clock.textContent = clock;
    if (now - this.lastUi >= UI_MS) {
      this.lastUi = now;
      this.followUi();
    }
  }
  private loop = (now: number): void => {
    this.raf = requestAnimationFrame(this.loop);
    this.frame(now);
  };
  /** Move the playhead along while playing: with the lead camera's video, else the line playing, else on to the next
   *  video or speech. */
  private advance(now: number): void {
    if (this.t == null) return;
    if (this.personPlayback.run?.mode === 'audio') {
      const moment = this.personPlayback.current;
      if (moment && this.audio && !this.audio.paused && this.audio.readyState >= 2)
        this.t = moment.start + Math.max(0, this.audio.currentTime - CLIP_LEAD_S);
      if (moment && this.t >= moment.end) {
        this.t = moment.end;
        this.personPlayback.atEnd(this.t);
        return;
      }
      this.follow();
      return;
    }
    if (this.personPlayback.run?.mode === 'video' && this.awaitingMedia != null) {
      const from = this.awaitingMedia;
      const day = isoDay(new Date(from * 1000));
      if (!this.chunks.has(day)) {
        if (this.S.chunkStatus[day] === 'error') this.skipPersonMoment('This camera recording could not be loaded.');
        return;
      }
      const moment = this.personPlayback.current;
      this.awaitingMedia = null;
      this.awaitingTarget = false;
      this.set({ waiting: false });
      if (moment) this.personVideoAt(moment, from);
      return;
    }
    if (this.awaitingMedia != null) {
      const from = this.awaitingMedia;
      if (this.awaitingTarget) {
        const day = isoDay(new Date(from * 1000));
        const chunk = this.chunks.get(day);
        if (!chunk) {
          if (this.S.chunkStatus[day] === 'error') this.setPlaying(false);
          return;
        }
        if (
          this.S.chunkStatus[day] === 'error' &&
          !chunk.entries.length &&
          !chunk.segments.length &&
          !chunk.spans.length
        ) {
          this.setPlaying(false);
          return;
        }
        this.seek(from, { play: true, centre: false });
        return;
      }
      if (this.pendingAfter(from)) return;
      this.awaitingMedia = null;
      this.set({ waiting: false });
      const next =
        this.availAt(from).length || this.keptAt(from)
          ? from
          : nextMedia(this.allSegs(), this.entries, from - 0.1, this.kept());
      if (next == null) this.setPlaying(false);
      else this.seek(next, { play: true, centre: false });
      return;
    }
    if (!this.personPlayback.run && this.S.skip && this.S.sel.size && this.present !== null) {
      // only the selected chips' time: jump over the rest, and stop after the last of it
      const at = presentFrom(this.present, this.t);
      if (at == null) {
        this.setPlaying(false);
        return;
      }
      if (at > this.t + 0.5) {
        this.seek(at, { play: true, centre: false });
        return;
      }
    }
    const lead = this.lead ? this.P[this.lead] : undefined;
    const v = lead?.v[lead.act];
    const seg = lead?.seg[lead.act];
    if (lead && v) {
      if (this.personPlayback.run?.mode === 'video' && (v.unplayable || seg?.broken)) {
        this.skipPersonMoment('This camera recording cannot be played.');
        return;
      }
      // While a seek is on its way, the video still shows where it was: reading its time (or its segment's end) would
      // drag the playhead back, or roll it over to the wrong segment.
      if (lead.busy || lead.want) {
        if (!lead.busy && this.lead) this.pump(this.lead); // nothing in flight: send the queued seek
        this.follow();
        return;
      }
      if (seg && v.readyState >= 2 && !v.seeking) this.t = seg.start + v.currentTime;
      if (this.personPlayback.atEnd(this.t)) return;
      if (seg && (v.ended || this.t >= seg.end - 0.03)) {
        if (this.personPlayback.run?.mode === 'video') {
          const moment = this.personPlayback.current;
          if (moment) this.personVideoAt(moment, seg.end);
          return;
        }
        let nt = nextMedia(this.allSegs(), this.entries, seg.end - 0.1);
        // The video ended, but the room's audio was kept past it: play on with that (#196).
        if (this.keptAt(seg.end) && (nt == null || nt > seg.end + ROOM_GAP_S)) nt = seg.end;
        if (this.pendingAfter(seg.end, nt ?? undefined)) {
          this.waitForMore(seg.end);
          return;
        }
        if (nt == null) {
          if (this.waitForMore(seg.end)) return;
          this.embeddedEnded = this.embeddedStart != null;
          // A clip that played to its end keeps showing its last frame: segments end exclusive, so a playhead left
          // on the end itself reads as no video there ("No video at 01:03:19", #2547).
          if (this.embeddedEnded) this.t = endHold(this.t, seg);
          this.setPlaying(false);
        } else this.seek(nt, { play: true, centre: false });
        return;
      }
      // The other cameras: start them when their video begins, and nudge them back in step now and then.
      if (now - this.lastSync > SYNC_MS) {
        this.lastSync = now;
        this.syncRoom();
        for (const c of this.avail) {
          const p = this.P[c];
          if (c === this.lead || !p) continue;
          const s = p.seg[p.act];
          const pv = p.v[p.act];
          if (!s || this.t >= s.end || this.t < s.start) this.place(c, this.t, true);
          else if (pv && (Math.abs(pv.currentTime - (this.t - s.start)) > DRIFT_S || pv.paused))
            this.place(c, this.t, true);
        }
      }
    } else if (this.personPlayback.run?.mode === 'video') {
      this.skipPersonMoment('This camera recording is unavailable.');
      return;
    } else if (this.room && this.audioEntry == null && !this.avail.length) {
      // The room's kept audio, with no picture: the playhead follows the sound (#196).
      const audio = this.audio;
      if (audio && this.loadedRoom === this.room && !audio.paused && audio.readyState >= 2) {
        this.t = this.room.start + audio.currentTime;
      } else if (audio?.paused && this.loadedRoom === this.room && now - this.lastSync > SYNC_MS) {
        this.lastSync = now; // refused, or not started yet: try again now and then, as under video
        this.startRoom(this.room);
      }
      if (this.t >= this.room.end - 0.05) this.afterRoom();
    } else if (this.audioEntry != null && this.audio && !this.audio.paused) {
      const e = this.entries[this.audioEntry];
      if (e) this.t = e.ts_start + Math.max(0, this.audio.currentTime - CLIP_LEAD_S);
    } else if (!this.avail.length && this.audioEntry == null) {
      // nothing here: on to the next video, speech, or kept audio
      const nt = nextMedia(this.allSegs(), this.entries, this.t, this.kept());
      if (this.pendingAfter(this.t, nt ?? undefined)) {
        this.waitForMore(this.t);
        return;
      }
      if (nt == null) {
        if (this.waitForMore(this.t)) return;
        this.embeddedEnded = this.embeddedStart != null;
        this.setPlaying(false);
      } else this.seek(nt, { play: true, centre: false });
    }
    this.follow();
  }
  private followUi(): void {
    if (this.loaded !== this.S.gen) return;
    if (this.t == null) {
      if (this.seen.avail !== '') {
        this.seen.avail = '';
        this.set({ avail: [], t: null, gap: null });
      }
      return;
    }
    this.contextAt(this.t);
    const avail = this.availAt(this.t);
    const key = avail.join(',');
    if (key !== this.seen.avail) {
      this.seen.avail = key;
      // A camera's video begins in a stretch of room audio alone: hold the sound until its picture moves, when
      // startRoom lines it up again, so the seconds it takes to load aren't heard twice (#196).
      if (this.playing && !this.avail.length && avail.length && this.room && this.audio && !this.audio.paused) {
        this.audio.pause();
      }
      const had = this.avail;
      this.avail = avail;
      if (!sameSet(avail, this.S.avail)) this.set({ avail });
      for (const c of this.cams) if (!avail.includes(c)) for (const v of this.P[c]?.v ?? []) v?.pause();
      // Only a camera whose video begins here: the others are already playing in step, and placing them again would
      // seek them for nothing (#299).
      if (this.videoMayPlay()) for (const c of avail) if (!had.includes(c)) this.place(c, this.t, true);
    }
    this.lead = this.personCamera(avail);
    const roomOnly = this.playing && !avail.length && this.room != null && this.audioEntry == null;
    if (this.S.roomOnly !== roomOnly) this.set({ roomOnly });
    const gap = avail.length ? null : this.brokenAt(this.t);
    if (gap !== this.S.gap) this.set({ gap });
    const unplayable = avail.filter(c => {
      const p = this.P[c];
      return !!p?.v[p.act]?.unplayable;
    });
    if (!sameSet(unplayable, this.S.unplayable)) {
      const unplayableCodecs: Record<string, string> = {};
      for (const c of unplayable) {
        const p = this.P[c];
        const codec = p?.v[p.act]?.unplayableCodec;
        if (codec) unplayableCodecs[c] = codec;
      }
      this.set({ unplayable, unplayableCodecs });
    }
    const cur = lineAt(this.entries, this.t);
    if (cur !== this.seen.cur) {
      this.seen.cur = cur;
      this.setTranscriptCurrent(this.t, false);
    }
    if (Math.abs(this.t - (this.seen.t ?? -1e9)) >= 0.5) {
      this.seen.t = this.t;
      this.set({ t: this.t });
    }
    this.loadNotes(this.t);
    if (!this.scrubbing) this.writeHash();
  }
  private loadNotes(t: number): void {
    if (this.notesForbidden) return;
    const b = Math.floor(t / BUCKET_S);
    if (b === this.noteBucket) return;
    this.noteBucket = b;
    const cached = this.notes.get(b);
    if (cached) {
      this.set({ notes: cached });
      return;
    }
    clearTimeout(this.noteTimer);
    this.noteTimer = setTimeout(
      () => {
        this.deps.fetchNotes(b).then(
          d => {
            this.notes.set(b, d);
            if (this.noteBucket === b) this.set({ notes: d });
          },
          (e: unknown) => {
            if (e instanceof HttpError && e.status === 403) {
              this.notesForbidden = true;
              this.set({ notes: { cams: {}, forbidden: true } });
            }
            // anything else: no notes for this moment; the next bucket tries again
          },
        );
      },
      this.scrubbing ? 150 : 0,
    );
  }

  // ---- the person in the link ----
  /** Select a confirmed person without discarding other facets. Empty clears the people facet. */
  pickPerson(name: string): void {
    if (name && !namedPerson(name)) return;
    if (this.personPlayback.run) this.pause();
    const people = name ? [...this.S.people.filter(n => n.toLocaleLowerCase() !== name.toLocaleLowerCase()), name] : [];
    this.selectPeople(people);
    this.writeHash();
  }
  private selectPeople(names: readonly string[]): void {
    const seen = new Set<string>();
    const people = names
      .map(n => n.trim())
      .filter(n => {
        const key = n.toLocaleLowerCase();
        if (!namedPerson(n) || seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    this.set({ people, person: people.at(-1) ?? '' });
    const selected = [...this.S.sel].filter(id => !id.startsWith('person:'));
    for (const name of people) {
      const chip = this.chips.find(
        c => c.kind === 'person' && c.label.toLocaleLowerCase() === name.toLocaleLowerCase(),
      );
      if (chip) selected.push(chip.id);
    }
    this.setSel(selected, false);
  }

  /** A detected activity names its source camera; a fresh intent overrides a previous manual tile choice. With play,
   *  it plays from its start every time, as soon as its day's data is in when that has to load first (#2259). */
  seekActivity(start: number, camera: string, { play = false }: { play?: boolean } = {}): void {
    if (this.cams.includes(camera)) this.set({ camera, cameraIntent: (this.S.cameraIntent ?? 0) + 1 });
    this.playOnLoad = play;
    this.playFromStartOnLoad = false;
    const day = isoDay(new Date(start * 1000));
    if (!this.deps.embedded && this.loaded != null) this.seek(start, { play });
    else if (day !== this.S.day) {
      this.landing = false;
      this.t = start; // Like a cross-day hash link, retain the exact target until that generation's data arrives.
      this.setDay(day);
    } else if (this.loaded == null) {
      this.t = start;
      this.setPlaying(false);
      this.writeHash();
    } else this.seek(start, { play });
  }

  /** Player's effective large camera, including manual tile swaps and fallback selection; null on unmount. */
  setDisplayedCamera(camera: string | null): void {
    this.displayedCamera = camera != null && this.cams.includes(camera) ? camera : null;
  }

  /** The camera a new event belongs to when nothing names one: the large tile, else the linked or first camera. */
  markingCamera(): string | null {
    return this.displayedCamera ?? this.S.camera ?? this.avail[0] ?? this.cams[0] ?? null;
  }

  /** Set an event boundary at the actual playhead, keeping the first boundary's camera. */
  markEventEnd(end: 'in' | 'out'): void {
    if (this.t == null || !Number.isFinite(this.t)) return;
    const cam = this.S.eventMarks?.cam ?? this.markingCamera();
    if (!cam) return;
    const marks = { cam, in: this.S.eventMarks?.in ?? null, out: this.S.eventMarks?.out ?? null, [end]: this.t };
    if (marks.in != null && marks.out != null) {
      this.markEventSpan({ cam, start: Math.min(marks.in, marks.out), end: Math.max(marks.in, marks.out) });
    } else {
      this.set({ eventMarks: marks });
      this.tl?.redraw();
    }
  }
  /** A completed range, from keys or the explicit pointer-marking tool. */
  markEventSpan(span: EventSpan): void {
    this.pause();
    this.set({ eventMarks: { cam: span.cam, in: span.start, out: span.end }, eventSpan: span });
    this.tl?.redraw();
  }
  /** A successful save clears the band, retaining the form's saved confirmation. */
  clearEventMarks(): void {
    this.set({ eventMarks: null });
    this.tl?.redraw();
  }
  /** Turn the Mark pointer tool on or off (the control bar's Mark; Escape and a finished drag turn it off). */
  setMarkingTool(on: boolean): void {
    if (this.S.markingTool !== on) this.set({ markingTool: on });
  }
  closeEvent(): void {
    this.set({ eventMarks: null, eventSpan: null });
    this.tl?.redraw();
  }

  // ---- keys ----
  // Capture Space before the shared player's listener and the focused control's native activation: a button, chip,
  // switch, or checkbox never takes it, only a text field (#2352). Its keyup is cancelled too, for a browser that
  // activates on keyup. Other shortcuts keep their bubbling order, so controls can consume them first. The shared
  // player respects preventDefault.
  private spaceDown = false;
  // A key flips the browser into :focus-visible, so a control focused by a click (a chip, a button) would light its
  // focus ring the moment a playback key is pressed. When the engine handles one, a control a pointer focused lets go
  // of focus; one reached with Tab keeps it, and its ring, for keyboard navigation (#2427).
  private pressing = false;
  private pointerFocus: Element | null = null;
  private onPointerDown = (): void => {
    this.pressing = true;
  };
  private onPointerUp = (): void => {
    this.pressing = false;
  };
  private onFocusIn = (e: FocusEvent): void => {
    this.pointerFocus = this.pressing && e.target instanceof Element ? e.target : null;
  };
  /** A playback key was handled: the control a click focused lets go, so it shows no ring. */
  private shedPointerFocus(): void {
    const el = this.pointerFocus;
    this.pointerFocus = null;
    if (el instanceof HTMLElement && document.activeElement === el) el.blur();
  }
  private onSpace = (e: KeyboardEvent): void => {
    if ((shortcut(e.key)?.id !== 'play' && e.code !== 'Space') || playbackKeyBlocked(e, { spaceOverControls: true }))
      return;
    e.preventDefault();
    this.spaceDown = true;
    this.shedPointerFocus();
    if (e.repeat) return;
    if (!toggleClip()) this.togglePlay();
  };
  private onSpaceUp = (e: KeyboardEvent): void => {
    if (!this.spaceDown || (shortcut(e.key)?.id !== 'play' && e.code !== 'Space')) return;
    this.spaceDown = false;
    e.preventDefault();
  };

  /** Review's other keys, by their id in REVIEW_KEYS (reviewKeys.ts), which the footer lists. */
  private onKey = (e: KeyboardEvent): void => {
    const target = e.target instanceof Element ? e.target : null;
    const key = shortcut(e.key);
    if (!key || key.id === 'play' || playbackKeyBlocked(e)) return;
    // Marking works even from a focused button; the rest leave a button its own keys. Escape turns the bar's Mark off
    // even before the day has a playhead.
    if (key.id === 'clear') {
      if (!this.S.eventMarks && !this.S.markingTool) return;
      this.setMarkingTool(false);
      if (this.S.eventMarks) this.closeEvent();
      e.preventDefault();
      this.shedPointerFocus();
      return;
    }
    if (this.t == null) return;
    if (key.id === 'mark') {
      if (e.shiftKey || e.repeat) return;
      this.markEventEnd(e.key === 'i' ? 'in' : 'out');
      e.preventDefault();
      this.shedPointerFocus();
      return;
    }
    if (target?.closest('button') && !target.closest('[data-review-keys]')) return;
    const t = this.t;
    // Where the view is headed, so + and - pressed during a glide halve or double where it lands (#2597).
    const headed = this.targetView();
    const span = headed.t1 - headed.t0;
    const nudge = e.shiftKey ? span / 8 : Math.max(1, Math.min(30, span / 150));
    const go = (ts: number | undefined) => {
      if (ts != null) this.seek(ts, { play: this.playing });
    };
    const run: Record<Exclude<ReviewKey['id'], 'play' | 'clear' | 'mark'>, () => void> = {
      nudge: () => go(e.key === 'ArrowLeft' ? t - nudge : t + nudge),
      ends: () => {
        const ed = dataEdges(this.entries, this.allSegs(), this.spans);
        go(e.key === 'Home' ? ed?.[0] : ed ? ed[1] - 1 : undefined);
      },
      lines: () => {
        const cur = lineAt(this.entries, t);
        if (e.key === 'k') go(this.entries[cur + 1]?.ts_start);
        else if (cur > 0) go(this.entries[cur - 1]?.ts_start);
      },
      moments: () => this.stepChip(e.key === '[' ? -1 : 1),
      skip: () => this.toggleSkip(),
      zoom: () => this.zoomSpan(e.key === '-' ? span * 2 : span / 2),
    };
    run[key.id]();
    e.preventDefault();
    this.shedPointerFocus();
  };

  // ---- the URL ----
  /** Keep the link on the page's place (#2391). A camera or day the link doesn't name yet is pushed, so Back steps
   *  through them; anything else replaces the entry. A change of the time alone (a seek or playback) is written at
   *  most once every hashTiming.everyMs, the latest one last, with t rounded to a second. The write dispatches nothing, so
   *  it never seeks. */
  private writeHash(due = false): void {
    // Only Review's own URL: behind Settings (Review keeps running under the drawer) the entry is Settings', and its
    // history state is how closing Settings gets back here.
    if (this.deps.embedded || tabOf(location.pathname) !== 'review') return;
    const hash = reviewHash({
      people: this.S.people,
      cam: this.S.camera,
      cameras: this.S.cameras,
      day: this.S.day ?? undefined,
      t: this.t == null ? undefined : Math.round(this.t),
    });
    if (hash === location.hash) return;
    const was = readHash();
    const push =
      !this.following &&
      was.day != null &&
      (was.day !== (this.S.day ?? undefined) || (was.cam ?? '') !== (this.S.camera ?? ''));
    // Only a time that moves on from one already in the link waits: the first time a link gets is written at once.
    const timeOnly = was.t != null && withoutTime(hash) === withoutTime(location.hash);
    const now = performance.now();
    // Only a started page waits; a stopped one has no playback to follow and must not write later, on another page.
    if (timeOnly && !due && this.started && now - this.lastHash < hashTiming.everyMs) {
      this.hashTimer ??= setTimeout(
        () => {
          this.hashTimer = undefined;
          this.writeHash(true);
        },
        this.lastHash + hashTiming.everyMs - now,
      );
      return;
    }
    clearTimeout(this.hashTimer);
    this.hashTimer = undefined;
    this.lastHash = now;
    setHash(hash, { push });
    this.wrote = location.hash;
  }
  /** A camera picked to show large: the link names it, and Back returns to the one before (#2391). */
  pickCamera(camera: string): void {
    if (!this.cams.includes(camera) || camera === this.S.camera) return;
    this.set({ camera });
    this.writeHash();
  }
  private pickCamerasFromHash(cameras: readonly string[]): void {
    const next = this.cams.filter(c => cameras.includes(c));
    if (sameSet(next, this.S.cameras)) return;
    this.set({ cameras: next });
    this.setSel(this.S.sel, false);
  }
  private onHash = (): void => {
    this.following = true;
    try {
      this.followHash();
    } finally {
      this.following = false;
    }
  };
  private followHash(): void {
    if (this.personPlayback.run) this.pause();
    const hp = readHash();
    this.pickCamerasFromHash(hp.cameras ?? []);
    if (hp.cam && this.cams.includes(hp.cam)) {
      this.set({ camera: hp.cam, cameraIntent: (this.S.cameraIntent ?? 0) + 1 });
    } else if (!hp.cam && this.S.camera) {
      // Back to a link from before a camera was picked: the large tile goes back to the default.
      this.set({ camera: undefined, cameraIntent: (this.S.cameraIntent ?? 0) + 1 });
    }
    if (hp.play) this.playOnLoad = true;
    if (JSON.stringify(hp.people ?? []) !== JSON.stringify(this.S.people)) this.selectPeople(hp.people ?? []);
    if (hp.day && hp.day !== this.S.day) {
      this.landing = false;
      if (!this.deps.embedded && this.loaded != null) {
        if (hp.t != null) this.seek(hp.t, { play: !!hp.play });
        else this.pickDay(hp.day);
        this.playOnLoad = false;
      } else {
        this.t = hp.t ?? null;
        this.setDay(hp.day);
      }
    } else if (hp.t != null && (hp.play || Math.abs(hp.t - (this.t ?? 0)) > 1)) {
      if (this.loaded == null) this.t = hp.t;
      else {
        this.playOnLoad = false;
        this.seek(hp.t, { play: !!hp.play });
        this.writeHash();
      }
    }
  }

  // ---- the DOM the engine drives directly ----
  setClock(el: { textContent: string | null } | null): void {
    this.clock = el;
  }
  setAudio(el: AudioLike | null): void {
    // The element let go of keeps its room audio's blob alive for as long as the page lives: let go of it too.
    const old = this.audio;
    if (old && old !== el && old.src.startsWith('blob:')) URL.revokeObjectURL(old.src);
    this.audio = el;
    if (el) el.muted = this.ui.getState().muted;
    this.loadedRoom = null; // a new element holds nothing yet
    this.unlocked = false; // nor has it been played inside a click
  }
  attachTimeline(t: TimelineHooks | null): void {
    this.tl = t;
  }
  /** Hand the transcript its hooks and mark the current line: centred when centre is true (a list just shown), else
   *  left where the list is (the same list re-rendered with new lines). */
  attachTranscript(t: TranscriptHooks | null, centre = true): void {
    this.tr = t;
    this.seen.cur = -2;
    if (t && this.t != null) this.setTranscriptCurrent(this.t, centre);
  }

  /** Read the link, open its day, and start following the playhead. A later start (Review shown again after another
   *  tab) keeps its place, unless the link now names a person, day, or time. */
  start(): void {
    if (this.started) return;
    this.started = true;
    this.following = true; // the link being read is where the page is: nothing to push
    try {
      this.begin();
    } finally {
      this.following = false;
    }
  }
  private begin(): void {
    if (this.deps.embedded) {
      this.raf = requestAnimationFrame(this.loop);
      return;
    }
    const hp = readHash();
    this.pickCamerasFromHash(hp.cameras ?? []);
    if (hp.cam && this.cams.includes(hp.cam) && (!this.begun || location.hash !== this.wrote)) {
      this.set({ camera: hp.cam, cameraIntent: (this.S.cameraIntent ?? 0) + 1 });
    }
    if (hp.play) this.playOnLoad = true;
    if (this.begun) {
      if (location.hash === this.wrote) {
        // Its own link, as it left it (a development remount): nothing new to follow
        if (this.t != null) this.seek(this.t);
      } else if (hp.day && hp.t == null) {
        // A link to a day with no moment opens it as a pick does: afresh, at its newest video.
        if (JSON.stringify(hp.people ?? []) !== JSON.stringify(this.S.people)) this.selectPeople(hp.people ?? []);
        this.deps.reload?.(hp.day);
        this.pickDay(hp.day);
      } else if (hp.person || hp.day || hp.t != null) this.onHash();
      else {
        this.writeHash();
        if (this.t != null) this.seek(this.t); // the videos were let go on stop: load the playhead's again
      }
    } else {
      this.begun = true;
      this.selectPeople(hp.people ?? []);
      if (hp.t != null) this.t = hp.t;
      // Only the first start decides: a remount (React's development double mount) starts again with the day the
      // first start wrote into the link, which nobody asked for.
      if (!hp.day && hp.t == null) this.landing = true;
      this.setDay(hp.day ?? isoDay(new Date((this.t ?? this.wall()) * 1000)));
    }
    window.addEventListener('hashchange', this.onHash);
    window.addEventListener('popstate', this.onHash);
    document.addEventListener('keydown', this.onSpace, true);
    document.addEventListener('keyup', this.onSpaceUp, true);
    document.addEventListener('keydown', this.onKey);
    document.addEventListener('pointerdown', this.onPointerDown, true);
    document.addEventListener('pointerup', this.onPointerUp, true);
    document.addEventListener('focusin', this.onFocusIn, true);
    this.raf = requestAnimationFrame(this.loop);
  }
  stop(): void {
    this.personPlayback.cancel();
    this.setPlaying(false);
    this.started = false;
    cancelAnimationFrame(this.raf);
    clearTimeout(this.noteTimer);
    clearTimeout(this.hashTimer); // the next page's URL is not Review's
    this.hashTimer = undefined;
    clearTimeout(this.preloadTimer); // a hidden tab fetches nothing
    if (this.rangeTimer !== undefined) this.publishRange(); // now, not dropped: nothing moves the view while hidden
    for (const c of this.cams) {
      const p = this.P[c];
      clearTimeout(p?.timer);
      clearTimeout(p?.safety);
      if (p) p.busy = false;
      // Let go of the media: a hidden tab keeps its <video>s in the page, and each one holding a source keeps one of
      // the browser's few connections to this host (#330).
      this.release(c);
    }
    this.playing = false;
    this.set({ playing: false });
    this.stopAudio();
    window.removeEventListener('hashchange', this.onHash);
    window.removeEventListener('popstate', this.onHash);
    document.removeEventListener('keydown', this.onSpace, true);
    document.removeEventListener('keyup', this.onSpaceUp, true);
    document.removeEventListener('keydown', this.onKey);
    document.removeEventListener('pointerdown', this.onPointerDown, true);
    document.removeEventListener('pointerup', this.onPointerUp, true);
    document.removeEventListener('focusin', this.onFocusIn, true);
    this.playFromStartOnLoad = false;
    this.closeEvent();
  }
}
