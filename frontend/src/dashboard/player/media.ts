import { decodeHeader } from './header';
import type { MediaContext } from './protocol';

const MAX_CONTEXTS = 128;
const MAX_CONTEXT_BYTES = 1024 * 1024;
const MAX_PENDING_BYTES = 64 * 1024 * 1024;
const MAX_PENDING = 600;

/** Bounded association for independently delivered WebTransport control and binary streams.
 * A later packet never passes a packet waiting for its immutable context (GOP decode order matters). */
export class MediaPackets {
  private contexts = new Map<number, { media: MediaContext; bytes: number }>();
  private queue: Array<{ buffer: ArrayBuffer; id?: number }> = [];
  private bytes = 0;
  private contextBytes = 0;

  private readonly deliver: (buffer: ArrayBuffer, media?: MediaContext) => void;
  constructor(deliver: (buffer: ArrayBuffer, media?: MediaContext) => void) {
    this.deliver = deliver;
  }

  get waiting(): boolean {
    return this.queue.length > 0;
  }

  clear(): void {
    this.contexts.clear();
    this.queue = [];
    this.bytes = 0;
    this.contextBytes = 0;
  }

  context(media: MediaContext): void {
    const bytes = new TextEncoder().encode(JSON.stringify(media)).byteLength;
    if (
      !Number.isSafeInteger(media.context_id) ||
      media.context_id <= 0 ||
      media.context_id > 0xffffffff ||
      !Number.isSafeInteger(media.stream) ||
      media.stream < 0 ||
      !['absent', 'unsupported', 'unavailable'].includes(media.profile_status) ||
      bytes > MAX_CONTEXT_BYTES
    )
      throw new Error('Invalid recorded media context.');
    const prior = this.contexts.get(media.context_id);
    if (prior) {
      if (JSON.stringify(prior.media) !== JSON.stringify(media)) throw new Error('Recorded media context changed.');
      return;
    }
    this.contexts.set(media.context_id, { media, bytes });
    this.contextBytes += bytes;
    this.drain();
    while (this.contexts.size > MAX_CONTEXTS || this.contextBytes > MAX_CONTEXT_BYTES) {
      const oldest = this.contexts.keys().next().value;
      if (oldest === undefined) break;
      this.contextBytes -= this.contexts.get(oldest)?.bytes ?? 0;
      this.contexts.delete(oldest);
    }
  }

  packet(buffer: ArrayBuffer): void {
    const header = decodeHeader(buffer);
    if (this.queue.length >= MAX_PENDING || this.bytes + buffer.byteLength > MAX_PENDING_BYTES)
      throw new Error('Recorded media context backlog exceeded.');
    this.queue.push({ buffer, id: header.contextId });
    this.bytes += buffer.byteLength;
    this.drain();
  }

  private drain(): void {
    while (this.queue.length) {
      const next = this.queue[0];
      if (!next) return;
      const context = next.id === undefined ? undefined : this.contexts.get(next.id)?.media;
      if (next.id !== undefined && !context) return;
      const header = decodeHeader(next.buffer);
      if (context && context.stream !== header.streamId) throw new Error('Recorded media stream mismatch.');
      this.queue.shift();
      this.bytes -= next.buffer.byteLength;
      this.deliver(next.buffer, context);
    }
  }
}
