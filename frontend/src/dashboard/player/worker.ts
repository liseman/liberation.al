// The player's Web Worker: decodes what the gateway sends (WebCodecs' VideoDecoder for H.264 and HEVC access units,
// ImageDecoder for JPEG frames), keeps overlays in a map by capture time, and draws a frame with the overlay whose
// time matches onto the OffscreenCanvas the page handed over (docs/CONTINUUM-GATEWAY.md). The page
// drives time: it posts the playhead on every animation frame, and the worker draws the newest decoded frame at or
// before it (or the newest of all, live), then reports what it drew and how the overlay was picked, which is the
// sync log the test page shows. Raw camera pixels carrying flip are rotated 180 degrees before upright overlays
// are drawn. Orientation stays with each timestamp through reordered codec output and buffered pictures. Capture
// requests queue behind incoming media, flush delayed codec output, then redraw the newest frame at or before
// their playhead before reporting the request ID and actual capture time.

import { type FrameMetadata, sourceBoxToDecoded, sourceContentRect } from '../camera/geometry';
import { CopyRoom, SmallCopier, type SmallCopy } from '../enhance/small';
import { BONES, MIN_CONF } from '../pose/skeleton';
import { av1Sequence, looksLikeAv1 } from './av1';
import { avcParameterSets } from './avc';
import { type Incoming, sort } from './decode';
import { decodeHeader } from './header';
import { MAX_GAP_US, type Overlay, OverlayMap, type Pick } from './overlays';
import type { MediaContext } from './protocol';

/** From the page. */
export type ToWorker =
  | { type: 'canvas'; canvas: OffscreenCanvas; returnBitmap?: boolean }
  /** Whether each returned frame should come with its small copy (Live's Enhance view analyses that). */
  | { type: 'small'; on: boolean }
  | { type: 'media'; media: MediaContext }
  | { type: 'message'; buffer: ArrayBuffer; media?: MediaContext; epoch?: number }
  | { type: 'playhead'; us: number | 'live' }
  | { type: 'reset'; epoch?: number }
  | { type: 'capture'; requestId: number; us: number };

/** To the page. */
export type FromWorker =
  | {
      type: 'frame';
      bitmap: ImageBitmap;
      small?: SmallCopy;
      frameUs: number;
      metadata?: FrameMetadata;
      epoch?: number;
    }
  | { type: 'captured'; requestId: number; frameUs: number | null }
  | {
      type: 'drawn';
      frameUs: number;
      metadata?: FrameMetadata;
      overlay: Pick['kind'];
      overlayUs: number | null;
      matched: boolean;
    }
  | { type: 'codec'; codec: string; supported: boolean }
  | { type: 'error'; message: string }
  | { type: 'stats'; decoded: number; buffered: number; overlays: number; dropped: number };

/** The decode worker as the page holds it: a real Worker running this file, or a test's stand-in. */
export interface WorkerLike {
  postMessage(message: ToWorker, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<FromWorker>) => void) | null;
  terminate(): void;
}

/** Decoded frames behind the playhead kept for scrubbing back; frames ahead of it are kept up to MAX_FRAMES (a
 *  whole window arrives in a burst before playback reaches it) and older ones are closed as more arrive. */
const KEEP_BEHIND = 30;
const MAX_FRAMES = 600;
type Codec = 'avc' | 'hevc' | 'av1';

/** Rust's source.rs sniffs the same way: an H.264 SPS or an HEVC VPS or SPS says which codec, and that it's a keyframe.
 *  An archived AV1 unit is an OBU stream, never Annex B: its keyframe carries the sequence header. */
export function sniffCodec(annexb: Uint8Array): Codec | null {
  if (looksLikeAv1(annexb)) return av1Sequence(annexb).kind === 'none' ? null : 'av1';
  let i = 0;
  while (i + 3 < annexb.length) {
    if (annexb[i] === 0 && annexb[i + 1] === 0 && annexb[i + 2] === 1) {
      const b0 = annexb[i + 3] ?? 0;
      const b1 = annexb[i + 4] ?? 0;
      const hevcType = (b0 >> 1) & 0x3f;
      if (b1 === 1 && hevcType >= 32 && hevcType <= 34) return 'hevc';
      if ((b0 & 0x1f) === 7) return 'avc';
      i += 3;
    } else i += 1;
  }
  return null;
}

/** HEVC keeps its existing configuration; AVC is derived from its actual in-band SPS. */
const HEVC_CODEC = 'hev1.1.6.L120.B0';

type Picture = VideoFrame | ImageBitmap;

class Pipeline {
  private canvas: OffscreenCanvas | null = null;
  private returnBitmap = false;
  private small = false;
  private copier: SmallCopier | null = null;
  /** Whether there is room to make the small copy without holding the video up. */
  private room = new CopyRoom();
  private ctx: OffscreenCanvasRenderingContext2D | null = null;
  private decoder: VideoDecoder | null = null;
  private codec: Codec | null = null;
  /** The in-band parameters the decoder was configured from (AVC SPS/PPS or the AV1 sequence header). */
  private avcParameters: string | null = null;
  private readonly frames = new Map<
    number,
    { picture: Picture; flip: boolean; media?: MediaContext; epoch?: number }
  >();
  /** Orientation waiting for codec output, keyed by presentation time, not decode order. */
  private readonly pendingFlips = new Map<
    number,
    { us: number; flip: boolean; media?: MediaContext; epoch?: number }
  >();
  private nextDuplicate = -1;
  private readonly mediaContexts = new Map<number, { media: MediaContext; bytes: number }>();
  private mediaBytes = 0;
  /** Old decoder callbacks must not consume a replacement decoder's timestamp metadata. */
  private decoderGeneration = 0;
  /** Reset invalidates queued messages and JPEG decodes that were already awaiting native output. */
  private generation = 0;
  private readonly overlays = new OverlayMap();
  private shownUs = -1;
  /** The last playhead asked for: a frame that lands after it (a paused seek's) is drawn then. */
  private head: number | 'live' | null = null;
  private decoded = 0;
  private dropped = 0;
  /** Messages are handled one after another: a keyframe's configure must finish before the next frame is looked at. */
  private chain: Promise<void> = Promise.resolve();
  private readonly post: (m: FromWorker) => void;

  constructor(post: (m: FromWorker) => void) {
    this.post = post;
  }

  /** Whether to hand a small copy over with each returned frame. */
  setSmall(on: boolean): void {
    this.small = on;
    this.room = new CopyRoom();
    if (!on) this.copier = null;
  }

  /** The canvas's picture as a small copy with its luma (enhance/small.ts), made here so the page's thread doesn't
   *  have to: reading pixels back and halving a large frame are what held it. */
  private smallCopy(w: number, h: number): SmallCopy | undefined {
    if (!this.canvas) return undefined;
    try {
      this.copier ??= new SmallCopier((width, height) => new OffscreenCanvas(width, height));
      const { canvas, luma } = this.copier.copy(this.canvas, w, h, false, true);
      const picture = (canvas as OffscreenCanvas).transferToImageBitmap();
      return { picture, luma: luma.data, width: luma.width, height: luma.height };
    } catch {
      return undefined; // the page makes its own
    }
  }

  setCanvas(canvas: OffscreenCanvas, returnBitmap = false): void {
    this.canvas = canvas;
    this.returnBitmap = returnBitmap;
    this.ctx = canvas.getContext('2d');
  }

  reset(): void {
    this.generation++;
    for (const f of this.frames.values()) f.picture.close();
    this.frames.clear();
    this.mediaContexts.clear();
    this.mediaBytes = 0;
    this.overlays.clear();
    this.shownUs = -1;
    this.head = null;
    this.dropDecoder();
  }

  context(media: MediaContext): void {
    if (this.mediaContexts.has(media.context_id)) return;
    const bytes = new TextEncoder().encode(JSON.stringify(media)).byteLength;
    if (bytes > 1024 * 1024) throw new Error('Worker media context exceeds budget.');
    this.mediaContexts.set(media.context_id, { media, bytes });
    this.mediaBytes += bytes;
    while (this.mediaContexts.size > 128 || this.mediaBytes > 1024 * 1024) {
      const oldest = this.mediaContexts.keys().next().value;
      if (oldest === undefined) break;
      this.mediaBytes -= this.mediaContexts.get(oldest)?.bytes ?? 0;
      this.mediaContexts.delete(oldest);
    }
  }

  take(buffer: ArrayBuffer, media?: MediaContext, epoch?: number): Promise<void> {
    if (!media) {
      let id: number | undefined;
      try {
        id = buffer.byteLength >= 20 ? decodeHeader(buffer).contextId : undefined;
      } catch {
        /* process reports malformed headers below */
      }
      if (id !== undefined) {
        media = this.mediaContexts.get(id)?.media;
        if (!media) {
          this.post({ type: 'error', message: 'Recorded frame context is unavailable.' });
          return Promise.resolve();
        }
      }
    }

    const generation = this.generation;
    this.chain = this.chain
      .then(() => (generation === this.generation ? this.process(buffer, media, epoch) : undefined))
      .catch(e => this.post({ type: 'error', message: e instanceof Error ? e.message : String(e) }));
    return this.chain;
  }

  /** Complete a one-shot capture after every preceding access unit and delayed B-frame output. The final draw is
   *  forced even if playback already showed that frame, so its bitmap immediately precedes the completion message. */
  capture(requestId: number, us: number): Promise<void> {
    const generation = this.generation;
    this.chain = this.chain
      .then(async () => {
        if (generation !== this.generation) return;
        await this.decoder?.flush();
        if (generation !== this.generation) return;
        this.shownUs = -1;
        this.show(us);
        this.post({ type: 'captured', requestId, frameUs: this.shownUs < 0 ? null : this.shownUs });
      })
      .catch(error => {
        if (generation !== this.generation) return;
        this.dropDecoder();
        this.post({ type: 'error', message: error instanceof Error ? error.message : String(error) });
      });
    return this.chain;
  }

  private async process(buffer: ArrayBuffer, media?: MediaContext, epoch?: number): Promise<void> {
    let m: Incoming;
    try {
      m = sort(buffer);
    } catch (e) {
      this.post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
      return;
    }
    switch (m.kind) {
      case 'overlay':
        this.overlays.add(m.overlay);
        // Vision can finish after its pixels were shown, and people in one frame arrive separately.
        // Repaint the retained pixels so later people replace the partial inference snapshot.
        if (this.head !== null && this.shownUs >= 0 && Math.abs(m.overlay.captureUs - this.shownUs) <= MAX_GAP_US) {
          this.shownUs = -1;
          this.show(this.head);
        }
        break;
      case 'jpeg':
        await this.decodeJpeg(m.payload, m.header.captureUs, m.header.flip, media, epoch);
        break;
      case 'video':
        await this.decodeVideo(m.payload, m.header.captureUs, m.header.keyframe, m.header.flip, media, epoch);
        break;
      case 'arrow':
        break; // tables are for the page, not the canvas; nothing draws them yet
    }
    this.stats();
  }

  private keep(us: number, picture: Picture, flip: boolean, media?: MediaContext, epoch?: number): void {
    this.frames.get(us)?.picture.close();
    this.frames.set(us, { picture, flip, media, epoch });
    if (this.shownUs === us) this.shownUs = -1;
    this.decoded += 1;
    if (this.head === 'live') {
      // Live never scrubs backward. Keep the newest presentation time only, including reordered decoder output;
      // WebCodecs owns its encoded reference chain independently of these already-decoded display pictures.
      const newest = Math.max(...this.frames.keys());
      for (const t of this.frames.keys()) if (t !== newest) this.evict(t);
    } else {
      // Recorded playback retains its short backlog and queued future pictures for scrubbing. Behind means behind
      // the playhead, not the last frame drawn: after a seek back, every frame of the new window is older than the
      // one still on screen, and judging by it threw them all away, so nothing drew until the playhead caught up
      // with the old picture.
      const at = typeof this.head === 'number' ? this.head : this.shownUs;
      const behind = [...this.frames.keys()].filter(t => t < at).sort((a, b) => a - b);
      for (const t of behind.slice(0, Math.max(0, behind.length - KEEP_BEHIND))) this.evict(t);
      // Over the cap, the frames farthest ahead of the playhead go: each seek back leaves the old position's frames
      // ahead of it, and after a few, evicting the oldest threw away the window about to play.
      while (this.frames.size > MAX_FRAMES)
        this.evict(typeof this.head === 'number' ? Math.max(...this.frames.keys()) : Math.min(...this.frames.keys()));
    }
    if (this.head !== null) this.show(this.head); // the frame a paused seek waits for
  }

  private evict(us: number): void {
    this.frames.get(us)?.picture.close();
    this.frames.delete(us);
    this.dropped += 1;
  }

  private async decodeJpeg(
    payload: Uint8Array,
    us: number,
    flip: boolean,
    media?: MediaContext,
    epoch?: number,
  ): Promise<void> {
    const generation = this.generation;
    try {
      let picture: Picture;
      if ('ImageDecoder' in globalThis) {
        const decoder = new ImageDecoder({ data: payload, type: 'image/jpeg' });
        try {
          picture = (await decoder.decode()).image;
        } finally {
          decoder.close();
        }
      } else {
        picture = await createImageBitmap(new Blob([payload as BlobPart], { type: 'image/jpeg' }));
      }
      if (generation !== this.generation) picture.close();
      else this.keep(us, picture, flip, media, epoch);
    } catch (e) {
      if (generation === this.generation)
        this.post({ type: 'error', message: `jpeg at ${us}: ${e instanceof Error ? e.message : String(e)}` });
    }
  }

  private async decodeVideo(
    payload: Uint8Array,
    us: number,
    key: boolean,
    flip: boolean,
    media?: MediaContext,
    epoch?: number,
  ): Promise<void> {
    const nextCodec = key ? sniffCodec(payload) : null;
    const parameters =
      key && nextCodec === 'av1'
        ? av1Sequence(payload)
        : key && nextCodec !== 'hevc' && (nextCodec === 'avc' || this.codec === 'avc')
          ? avcParameterSets(payload)
          : null;
    if (parameters?.kind === 'invalid') {
      this.post({ type: 'error', message: parameters.message });
      this.dropDecoder();
      return;
    }
    const changed =
      (nextCodec !== null && nextCodec !== this.codec) ||
      (parameters?.kind === 'ready' && parameters.signature !== this.avcParameters);
    if (this.decoder && changed) {
      const generation = this.generation;
      await this.decoder.flush();
      if (generation !== this.generation) return;
      this.dropDecoder();
    }
    if (!this.decoder) {
      if (!key) return; // decoding has to start at a keyframe
      if (!nextCodec || (nextCodec !== 'hevc' && parameters?.kind !== 'ready')) {
        this.post({
          type: 'error',
          message: `keyframe at ${us} carries no complete parameter sets; can't configure the codec`,
        });
        return;
      }
      try {
        await this.configure(nextCodec, parameters?.kind === 'ready' ? parameters.codec : HEVC_CODEC);
      } catch (error) {
        this.dropDecoder();
        this.post({
          type: 'error',
          message: `configure at ${us}: ${error instanceof Error ? error.message : String(error)}`,
        });
        return;
      }
      if (!this.decoder) return;
      this.avcParameters = nextCodec !== 'hevc' && parameters?.kind === 'ready' ? parameters.signature : null;
    }
    const data = payload.slice();
    try {
      // Decoder timestamps are opaque identity tags; duplicate presentation times must not overwrite metadata.
      const tag = this.pendingFlips.has(us) ? this.nextDuplicate-- : us;
      this.pendingFlips.set(tag, { us, flip, media, epoch });
      while (this.pendingFlips.size > MAX_FRAMES) {
        const oldest = this.pendingFlips.keys().next().value;
        if (oldest !== undefined) this.pendingFlips.delete(oldest);
      }
      this.decoder.decode(new EncodedVideoChunk({ type: key ? 'key' : 'delta', timestamp: tag, data }));
    } catch (e) {
      this.post({ type: 'error', message: `decode at ${us}: ${e instanceof Error ? e.message : String(e)}` });
      this.dropDecoder();
    }
  }

  private async configure(codec: Codec, codecString: string): Promise<void> {
    const config: VideoDecoderConfig = { codec: codecString, optimizeForLatency: true };
    // AV1 has no Annex B: its units go to the decoder as the OBU stream they are.
    const annexb = codec === 'av1' ? config : ({ ...config, [codec]: { format: 'annexb' } } as VideoDecoderConfig);
    const generation = this.generation;
    const support = await VideoDecoder.isConfigSupported(annexb);
    if (generation !== this.generation) return;
    this.post({ type: 'codec', codec, supported: support.supported === true });
    if (!support.supported) return;
    this.codec = codec;
    const decoderGeneration = ++this.decoderGeneration;
    this.decoder = new VideoDecoder({
      output: frame => {
        const pending = this.pendingFlips.get(frame.timestamp);
        if (decoderGeneration !== this.decoderGeneration || pending === undefined) {
          frame.close();
          return;
        }
        this.pendingFlips.delete(frame.timestamp);
        this.keep(pending.us, frame, pending.flip, pending.media, pending.epoch);
      },
      error: e => {
        if (decoderGeneration !== this.decoderGeneration) return;
        this.post({ type: 'error', message: `decoder: ${e.message}` });
        this.dropDecoder(); // the next keyframe configures a new one
      },
    });
    this.decoder.configure(annexb);
  }

  /** Forgets a decoder that errored or was closed; decoding resumes at the next keyframe. */
  private dropDecoder(): void {
    this.decoderGeneration++;
    this.pendingFlips.clear();
    if (this.decoder && this.decoder.state !== 'closed') this.decoder.close();
    this.decoder = null;
    this.codec = null;
    this.avcParameters = null;
  }

  show(playhead: number | 'live'): void {
    this.head = playhead;
    if (!this.ctx || !this.canvas || this.frames.size === 0) return;
    let best = -1;
    for (const t of this.frames.keys()) if ((playhead === 'live' || t <= playhead) && t > best) best = t;
    if (best < 0 || best === this.shownUs) return;
    const frame = this.frames.get(best);
    if (!frame) return;
    const { picture, flip, media, epoch } = frame;
    const began = performance.now();
    this.shownUs = best;
    const w = picture instanceof ImageBitmap ? picture.width : picture.displayWidth;
    const h = picture instanceof ImageBitmap ? picture.height : picture.displayHeight;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.ctx.save();
    try {
      if (flip) {
        this.ctx.translate(w, h);
        this.ctx.scale(-1, -1);
      }
      this.ctx.drawImage(picture, 0, 0, w, h);
    } finally {
      this.ctx.restore();
    }
    const metadata: FrameMetadata = {
      capturedAtUs: best,
      width: w,
      height: h,
      flip,
      profileStatus: media?.profile_status ?? 'unsupported',
      nativeGeometry: media?.native_geometry ?? null,
    };
    // Inference overlays already use upright image coordinates, so they never inherit the pixel transform.
    const pick = this.overlays.pick(best);
    if (pick.kind !== 'none') draw(this.ctx, media ? projectedOverlay(pick.overlay, metadata) : pick.overlay, w, h);
    if (this.returnBitmap) {
      // The small copy, where there is room for it, before the canvas's picture leaves with the frame
      let small: SmallCopy | undefined;
      let copyMs: number | undefined;
      if (this.small && this.room.wanted(performance.now())) {
        const from = performance.now();
        small = this.smallCopy(w, h);
        copyMs = performance.now() - from;
      }
      this.post({
        type: 'frame',
        bitmap: this.canvas.transferToImageBitmap(),
        ...(small ? { small } : {}),
        frameUs: best,
        metadata,
        epoch,
      });
      if (this.small) this.room.took(best, performance.now() - began - (copyMs ?? 0), copyMs);
    }
    this.post({
      type: 'drawn',
      metadata,
      frameUs: best,
      overlay: pick.kind,
      overlayUs: pick.kind === 'none' ? null : pick.overlay.captureUs,
      matched: pick.kind === 'exact',
    });
  }

  private stats(): void {
    this.post({
      type: 'stats',
      decoded: this.decoded,
      buffered: this.frames.size,
      overlays: this.overlays.size,
      dropped: this.dropped,
    });
  }

  get codecName(): string | null {
    return this.codec;
  }
}

/** Gateway overlays use upright source fractions, independently of encoded letterbox padding. */
function projectedOverlay(overlay: Overlay, metadata: FrameMetadata): Overlay {
  const rect = sourceContentRect(metadata);
  if (!rect) return { ...overlay, boxes: [], skeletons: [] };
  return {
    ...overlay,
    boxes: overlay.boxes.flatMap(box => {
      const xyxy = sourceBoxToDecoded(box.xyxy, metadata);
      return xyxy ? [{ ...box, xyxy }] : [];
    }),
    skeletons: overlay.skeletons.flatMap(skeleton => {
      const xyxy = sourceBoxToDecoded(skeleton.box.xyxy, metadata);
      if (!xyxy) return [];
      const joints = skeleton.joints.map(([x, y, score]): [number, number, number] =>
        x < 0 || y < 0 || x > 1 || y > 1
          ? [-1, -1, score]
          : [(rect.x + x * rect.w) / metadata.width, (rect.y + y * rect.h) / metadata.height, score],
      );
      return [{ ...skeleton, box: { ...skeleton.box, xyxy }, joints }];
    }),
  };
}

/** Draws an overlay onto a frame of `w` by `h` pixels. Exported for the page's own canvases too. */
export function draw(
  ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
  o: Overlay,
  w: number,
  h: number,
): void {
  ctx.lineWidth = Math.max(2, w / 480);
  ctx.font = `${Math.max(12, w / 60)}px sans-serif`;
  for (const b of o.boxes) {
    const color = b.hue === null ? 'rgb(255, 200, 0)' : `hsl(${b.hue} 90% 55%)`;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    const [x1, y1, x2, y2] = b.xyxy;
    ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
    if (b.label) ctx.fillText(b.label, x1 * w + 4, y1 * h - 4);
  }
  for (const s of o.skeletons) {
    const color = s.box.hue === null ? 'rgb(0, 220, 255)' : `hsl(${s.box.hue} 90% 60%)`;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    const seen = s.joints.map(j => j[2] >= MIN_CONF && j[0] >= 0 && j[1] >= 0);
    const bones: readonly (readonly [number, number])[] =
      s.layout === 'halpe26' ? BONES : s.joints.slice(1).map((_, i) => [i, i + 1] as const);
    ctx.beginPath();
    for (const [a, b] of bones) {
      const ja = s.joints[a];
      const jb = s.joints[b];
      if (!ja || !jb || !seen[a] || !seen[b]) continue;
      ctx.moveTo(ja[0] * w, ja[1] * h);
      ctx.lineTo(jb[0] * w, jb[1] * h);
    }
    ctx.stroke();
    for (const [i, j] of s.joints.entries()) {
      if (!seen[i]) continue;
      ctx.beginPath();
      ctx.arc(j[0] * w, j[1] * h, ctx.lineWidth * 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (o.captions.length) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    const line = Math.max(16, w / 45);
    ctx.fillRect(0, h - line * o.captions.length - 8, w, line * o.captions.length + 8);
    ctx.fillStyle = 'white';
    o.captions.forEach((c, i) => {
      ctx.fillText(c.speaker ? `${c.speaker}: ${c.text}` : c.text, 8, h - 6 - line * (o.captions.length - 1 - i));
    });
  }
}

// Only when running as a worker; importing this module in a test or the page registers nothing.
if (typeof window === 'undefined' && typeof self !== 'undefined' && 'postMessage' in self) {
  const pipeline = new Pipeline(m =>
    self.postMessage(m, {
      transfer: m.type === 'frame' ? (m.small ? [m.bitmap, m.small.picture, m.small.luma.buffer] : [m.bitmap]) : [],
    }),
  );
  self.onmessage = (e: MessageEvent<ToWorker>) => {
    const m = e.data;
    switch (m.type) {
      case 'canvas':
        pipeline.setCanvas(m.canvas, m.returnBitmap);
        break;
      case 'small':
        pipeline.setSmall(m.on);
        break;
      case 'media':
        pipeline.context(m.media);
        break;
      case 'message':
        void pipeline.take(m.buffer, m.media, m.epoch);
        break;
      case 'playhead':
        pipeline.show(m.us);
        break;
      case 'reset':
        pipeline.reset();
        break;
      case 'capture':
        void pipeline.capture(m.requestId, m.us);
        break;
    }
  };
}

export { Pipeline };
