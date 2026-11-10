// The Live tab's cameras through Continuum's gateway: one live session per camera on screen, carrying the
// camera's frames (drawn by mux.ts) and its vision overlays (drawn by boxes.ts as LiveFrames, the shape v1's
// /video/live/boxes sent, so the box renderer, its smoothing, and its clicks are unchanged).
//
// A session asks the gateway for `<cam>.video` live with the overlay groups `faces` and `objects`, and `poses` while a
// view wants skeletons; a change in what's wanted sends a new `live`, which replaces the old one. A session that
// closes or fails is opened again with backoff while anything still watches the camera, and closed when nothing does.
//
// Each overlay carries only the groups that had a record at its capture time, but the box renderer takes a complete
// list per update, as v1's per-frame boxes were (a box missing from one fades). So a session keeps each group's
// newest boxes and hands the merged list over with every overlay and every video frame, until the group's hold runs
// out (HOLD_S).

import { useEffect, useState } from 'react';
import { sort } from '../player/decode';
import type { Overlay } from '../player/overlays';
import { defaultGatewayUrl, type MediaContext, type StreamSummary } from '../player/protocol';
import { type Connection, connect as connectGateway, type Handlers, type Reply } from '../player/transport';
import { holdWatch, listStreams, watchAvailable, watchStream } from '../player/watch';
import type { Pose } from '../pose/skeleton';

/** One camera frame's boxes, as v1's vision/live.py sent them on /video/live/boxes. */
export interface LiveFrame {
  cam: string;
  ts?: number;
  face_ts?: number;
  /** The camera's latest skeletons, while a view wants them and they are fresh. */
  poses?: Pose[];
  pose_ts?: number;
  objects?: { id: string | number; box: number[]; label: string }[];
  faces?: { track: string | number; box: number[]; name: string }[];
}

type Connect = (url: string, handlers: Handlers) => Promise<Connection>;

let connectImpl: Connect = connectGateway;
let urlOf = (): string =>
  defaultGatewayUrl(typeof location === 'undefined' ? { hostname: '', protocol: '' } : location);

/** For tests: the connection to use, and the gateway's address. */
export function setGatewayForTests(connect: Connect, url = 'ws://gateway/ws', watch = false): void {
  connectImpl = connect;
  urlOf = () => url;
  useWatch = watch;
}

/** Each camera's session rides the page's one watch connection (player/watch.ts); off, or on a gateway
 *  without `watch`, each camera has a connection of its own with a `live` on it. */
let useWatch = true;

const MAX_RETRY_MS = 15000;
/** How long a group's newest boxes stand after they arrived, with nothing newer. Faces and skeletons come with about
 *  every frame the vision workers process (10/s or more), so a person who leaves fades half a second later. Grounded
 *  objects come again with every caption, at least every 10 s per camera, so they stand 30 s: objects that leave the
 *  view clear within half a minute. The gateway sends a session the camera's newest grounding as it starts. */
export const HOLD_S = { faces: 0.5, poses: 0.5, objects: 30 } as const;

/** Wall-clock seconds: when held boxes arrived, and how old an object is. */
let wallNow = (): number => Date.now() / 1000;
/** For tests: the clock holds are measured by. */
export function setClockForTests(now: () => number): void {
  wallNow = now;
}

/** A group's newest boxes: their capture time and when they arrived. */
interface Held<T> {
  ts: number;
  at: number;
  list: T;
}

interface Session {
  cam: string;
  frames: Set<(jpeg: Uint8Array, capturedAt: number, flip: boolean) => void>;
  boxes: Set<(d: LiveFrame) => void>;
  video: Set<VideoListener>;
  /** Stops following the camera on the watch connection, while it does. */
  unwatch: (() => void) | null;
  generation: number;
  poseWants: number;
  conn: Connection | null;
  stream: number | null;
  retry: number;
  timer: ReturnType<typeof setTimeout> | undefined;
  /** Each group's newest boxes, carried into later updates while their hold lasts. */
  faces: Held<NonNullable<LiveFrame['faces']>> | null;
  objects: Held<NonNullable<LiveFrame['objects']>> | null;
  poses: Held<Pose[]> | null;
  /** Confirmed claims awaiting a named backend frame, scoped to this session's stable tracks and groups. */
  names: Map<string, { at: number; group: number; name: string }>;
  /** This session ended for good (nothing watches it). */
  closed: boolean;
}

/** A video access unit with its wire header and its media context when it has one; null resets the decoder. */
type VideoListener = (buffer: ArrayBuffer | null, media?: MediaContext) => void;

const sessions = new Map<string, Session>();

// Keep a single media feed. Toggling a drawing layer must not replace the video subscription.
// These are existing vision records; requesting poses does not start inference.
const LIVE_GROUPS = ['faces', 'objects', 'poses'];

/** The faces and objects of one overlay, as v1's LiveFrame has them: faces by track, named or `unknown`; objects by
 *  label. */
export function boxesOf(o: Overlay): Pick<LiveFrame, 'faces' | 'objects'> {
  return {
    faces: o.boxes
      .filter(b => b.kind === 'face')
      .map(b => ({ track: b.id ?? '', box: [...b.xyxy], name: b.label || 'unknown' })),
    objects: o.boxes
      .filter(b => b.kind === 'object')
      .map((b, i) => ({ id: `${b.label}:${i}`, box: [...b.xyxy], label: b.label })),
  };
}

/** The camera's boxes now: each group's newest while its hold lasts. Its time is the newest record's capture time
 *  (never a later frame's, so the renderer's smoothing never sees time go back when the next record comes). */
export function heldFrameOf(s: Pick<Session, 'cam' | 'faces' | 'objects' | 'poses'>, now = wallNow()): LiveFrame {
  const faces = s.faces && now - s.faces.at <= HOLD_S.faces ? s.faces : null;
  const objects = s.objects && now - s.objects.at <= HOLD_S.objects ? s.objects : null;
  const poses = s.poses && now - s.poses.at <= HOLD_S.poses ? s.poses : null;
  const times = [faces?.ts, objects?.ts, poses?.ts].filter((t): t is number => t !== undefined);
  const d: LiveFrame = { cam: s.cam, faces: faces?.list ?? [], objects: objects?.list ?? [] };
  if (faces) d.face_ts = faces.ts;
  if (times.length) d.ts = Math.max(...times);
  if (poses) {
    d.poses = poses.list;
    d.pose_ts = poses.ts;
  }
  return d;
}

function frameOf(s: Session): LiveFrame {
  const frame = heldFrameOf(s);
  frame.faces = frame.faces?.map(face => {
    const claim = s.names.get(String(face.track));
    return claim && frame.face_ts !== undefined && frame.face_ts >= claim.at && face.name === `Face ${claim.group}`
      ? { ...face, name: claim.name }
      : face;
  });
  return frame;
}

/** Read the current held boxes without opening a camera connection or extending any group's hold. */
export function currentBoxes(cam: string): LiveFrame | null {
  const s = sessions.get(cam);
  return s ? frameOf(s) : null;
}

/** Publish only a successful claim's canonical name; never relabel another camera, track, group, or known face. */
export function nameLiveFace(source: { camera: string; at: number; track: string; group: number }, name: string): void {
  const s = sessions.get(source.camera);
  if (!s || !Number.isFinite(source.at) || !name.trim()) return;
  const current = s.faces?.list.find(face => String(face.track) === source.track);
  if (current && s.faces && s.faces.ts >= source.at && current.name !== `Face ${source.group}`) return;
  s.names.delete(source.track);
  s.names.set(source.track, { at: source.at, group: source.group, name });
  while (s.names.size > 128) {
    const oldest = s.names.keys().next().value;
    if (oldest !== undefined) s.names.delete(oldest);
  }
  pushBoxes(s);
}

function open(s: Session): void {
  if (s.closed || s.conn || s.unwatch) return;
  if (useWatch) {
    const generation = ++s.generation;
    const current = () => !s.closed && s.generation === generation;
    // Every frame, with its overlays, on the page's one watch connection (which reconnects by itself).
    s.unwatch = watchStream(
      `${s.cam}.video`,
      0,
      frame => {
        if (current()) binary(s, frame.buffer, frame.media ?? undefined);
      },
      {
        overlays: LIVE_GROUPS,
        onReset: () => {
          if (current()) for (const fn of s.video) fn(null);
        },
        onUnavailable: () => {
          // An older gateway: this camera gets a connection of its own after all.
          if (!current()) return;
          s.unwatch?.();
          s.unwatch = null;
          s.generation++;
          openOwn(s);
        },
      },
    );
    return;
  }
  openOwn(s);
}

/** A connection of the camera's own, with a `live` on it. */
function openOwn(s: Session): void {
  if (s.closed || s.conn) return;
  const generation = ++s.generation;
  const current = () => !s.closed && s.generation === generation;
  const handlers: Handlers = {
    onReply: r => {
      if (current()) reply(s, r);
    },
    onBinary: b => {
      if (current()) binary(s, b);
    },
    onClose: () => {
      if (current()) lost(s);
    },
  };
  connectImpl(urlOf(), handlers).then(
    c => {
      if (!current()) c.close();
      else s.conn = c;
    },
    () => {
      if (current()) lost(s);
    },
  );
}

function reply(s: Session, r: Reply): void {
  if (r.type === 'hello') s.conn?.send({ type: 'streams' });
  else if (r.type === 'streams') {
    const found = r.streams.find(x => x.name === `${s.cam}.video`);
    if (!found) {
      lost(s); // capture hasn't registered the camera yet: try again later
      return;
    }
    s.stream = found.id;
    goLive(s);
  } else if (r.type === 'live') s.retry = 0;
  else if (r.type === 'stopped' && r.reason)
    lost(s); // the gateway ended it (the broker connection went)
  else if (r.type === 'error') lost(s);
}

function goLive(s: Session): void {
  if (!s.conn || s.stream === null) return;
  s.conn.send({ type: 'live', stream: s.stream, overlays: LIVE_GROUPS });
}

function binary(s: Session, buffer: ArrayBuffer, media?: MediaContext): void {
  const m = sort(buffer);
  if (m.kind === 'jpeg') {
    for (const fn of s.frames) fn(m.payload, m.header.captureUs / 1e6, m.header.flip);
    if (s.faces || s.objects || s.poses) pushBoxes(s); // every frame, as v1's boxes came
  } else if (m.kind === 'video') {
    for (const fn of s.video) fn(buffer, media);
    if (s.faces || s.objects || s.poses) pushBoxes(s);
  } else if (m.kind === 'overlay') {
    const ts = m.overlay.captureUs / 1e6;
    const at = wallNow();
    const { faces, objects } = boxesOf(m.overlay);
    if (faces?.length && (!s.faces || ts >= s.faces.ts)) {
      for (const face of faces) {
        const claim = s.names.get(String(face.track));
        if (claim && ts >= claim.at && !/^Face \d+$/.test(face.name) && face.name !== 'unknown')
          s.names.delete(String(face.track));
      }
      s.faces = { ts, at, list: faces };
    }
    if (objects?.length && (!s.objects || ts >= s.objects.ts)) s.objects = { ts, at, list: objects };
    if (m.overlay.skeletons.length && s.poseWants > 0) {
      const list = m.overlay.skeletons.map(k => ({
        box: [...k.box.xyxy, k.box.score],
        ...(k.box.id ? { track_id: k.box.id } : {}),
        hue: k.box.hue,
        keypoints: k.joints.map(j => [...j]),
      }));
      s.poses = { ts, at, list };
    }
    pushBoxes(s);
  }
}

function pushBoxes(s: Session): void {
  if (!s.boxes.size) return;
  const d = frameOf(s);
  for (const fn of s.boxes) fn(d);
}

function lost(s: Session): void {
  s.generation++;
  for (const fn of s.video) fn(null);
  s.conn?.close();
  s.conn = null;
  s.stream = null;
  if (s.closed) return;
  clearTimeout(s.timer);
  s.timer = setTimeout(() => openOwn(s), Math.min(MAX_RETRY_MS, 1000 * 2 ** s.retry++));
}

function session(cam: string): Session {
  let s = sessions.get(cam);
  if (!s) {
    s = {
      cam,
      frames: new Set(),
      boxes: new Set(),
      video: new Set(),
      generation: 0,
      poseWants: 0,
      conn: null,
      unwatch: null,
      stream: null,
      retry: 0,
      timer: undefined,
      faces: null,
      objects: null,
      poses: null,
      names: new Map(),
      closed: false,
    };
    sessions.set(cam, s);
    open(s);
  }
  return s;
}

function release(s: Session): void {
  if (s.frames.size || s.boxes.size || s.poseWants || s.video.size) return;
  s.closed = true;
  s.generation++;
  clearTimeout(s.timer);
  s.unwatch?.();
  s.unwatch = null;
  s.conn?.close();
  s.conn = null;
  s.names.clear();
  sessions.delete(s.cam);
}

/** Calls fn with each JPEG, its capture time in Unix seconds, and whether the camera was mounted upside down (the
 * picture is drawn turned half way round, as the Review player does); returns the function that stops. Existing
 * callbacks that accept only JPEG bytes continue to work. */
export function watchFrames(
  cam: string,
  fn: (jpeg: Uint8Array, capturedAt: number, flip: boolean) => void,
): () => void {
  const s = session(cam);
  s.frames.add(fn);
  return () => {
    s.frames.delete(fn);
    release(s);
  };
}

/** Native encoded access units, retaining their wire header. Null resets the decoder after a lost connection.
 * Consumers must copy before transferring; other subscribers still own the original gateway message. */
export function watchVideo(cam: string, fn: VideoListener): () => void {
  const s = session(cam);
  s.video.add(fn);
  return () => {
    s.video.delete(fn);
    release(s);
  };
}

/** Calls fn with each of camera cam's overlays as a LiveFrame; returns the function that stops. */
export function watchBoxes(cam: string, fn: (d: LiveFrame) => void): () => void {
  const s = session(cam);
  s.boxes.add(fn);
  return () => {
    s.boxes.delete(fn);
    release(s);
  };
}

/** Asks for camera cam's skeletons while a view wants them; returns the function that stops asking. */
export function wantPoses(cam: string): () => void {
  const s = session(cam);
  s.poseWants++;
  let done = false;
  return () => {
    if (done) return;
    done = true;
    s.poseWants--;
    if (s.poseWants === 0) s.poses = null;
    release(s);
  };
}

/** The cameras with a session, for tests and the stale ticker. */
export const watchedCams = (): string[] => [...sessions.keys()];

/** How long the gateway may take to list its streams before a poll gives up. */
const STREAMS_TIMEOUT_MS = 5000;

/** The gateway's streams, from a short connection of their own. */
export function gatewayStreams(): Promise<StreamSummary[]> {
  return new Promise((resolve, reject) => {
    let conn: Connection | null = null;
    let hello = false;
    let done = false;
    const finish = (err: Error | null, streams: StreamSummary[] = []) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      conn?.close();
      if (err) reject(err);
      else resolve(streams);
    };
    const ask = () => {
      if (conn && hello) conn.send({ type: 'streams' });
    };
    const timer = setTimeout(() => finish(new Error('the gateway took too long')), STREAMS_TIMEOUT_MS);
    connectImpl(urlOf(), {
      onReply: r => {
        if (r.type === 'hello') {
          hello = true;
          ask();
        } else if (r.type === 'streams') finish(null, r.streams as StreamSummary[]);
        else if (r.type === 'error') finish(new Error(r.message));
      },
      onBinary: () => {},
      onClose: reason => finish(new Error(`the gateway closed: ${reason}`)),
    }).then(
      c => {
        if (done) c.close();
        else {
          conn = c;
          ask();
        }
      },
      e => finish(e instanceof Error ? e : new Error(String(e))),
    );
  });
}

/** The cameras whose live video (`<cam>.video`) has a frame no older than freshS at now (Unix seconds). */
export function liveCamsOf(streams: StreamSummary[], now: number, freshS: number): string[] {
  const out: string[] = [];
  for (const s of streams) {
    const m = s.name ? /^(.+)\.video$/.exec(s.name) : null;
    if (m?.[1] && s.frames > 0 && now - s.end_us / 1e6 <= freshS) out.push(m[1]);
  }
  return out.sort();
}

/** How often the Live tab asks the gateway which cameras are live. */
export const CAMS_POLL_MS = 5000;

/** The gateway's streams for the Live tab's poll: on the page's watch connection, which the hook holds open, so a
 *  poll opens no connection; a connection of its own only on a gateway without `watch`. A hidden page keeps what it
 *  had (null). */
async function pollStreams(): Promise<StreamSummary[] | null> {
  if (!useWatch) return gatewayStreams();
  try {
    return await listStreams();
  } catch (error) {
    if (watchAvailable() === false) return gatewayStreams();
    if (typeof document !== 'undefined' && document.hidden) return null;
    throw error;
  }
}

/** The cameras with fresh live video at the gateway, asked every CAMS_POLL_MS; empty while it can't be reached. */
export function useGatewayLiveCams(freshS: number): string[] {
  const [cams, setCams] = useState<string[]>([]);
  useEffect(() => {
    let stop = false;
    const release = useWatch ? holdWatch() : () => {};
    const poll = () =>
      pollStreams().then(
        streams =>
          !stop && streams !== null && setCams(prev => same(prev, liveCamsOf(streams, Date.now() / 1000, freshS))),
        () => !stop && setCams(prev => (prev.length ? [] : prev)),
      );
    void poll();
    const timer = setInterval(poll, CAMS_POLL_MS);
    return () => {
      stop = true;
      clearInterval(timer);
      release();
    };
  }, [freshS]);
  return cams;
}

const same = (a: string[], b: string[]): string[] => (a.length === b.length && a.every((x, i) => x === b[i]) ? a : b);
