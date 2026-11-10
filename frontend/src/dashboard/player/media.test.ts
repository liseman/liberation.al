import { describe, expect, it } from 'vitest';
import { encodeHeader, PayloadType, splitMessage } from './header';
import { MediaPackets } from './media';
import type { MediaContext } from './protocol';

const media = (id: number, stream = 1): MediaContext => ({
  type: 'media',
  context_id: id,
  stream,
  chunk_id: String(id),
  revision: 1,
  profile_status: 'absent',
  native_geometry: { width: 640, height: 480 },
});
const packet = (contextId?: number) => {
  const header = encodeHeader({
    streamId: 1,
    payloadType: PayloadType.jpeg,
    keyframe: true,
    live: false,
    captureUs: 123,
    contextId,
  });
  const result = new Uint8Array(header.length + 2);
  result.set(header);
  result.set([0xff, 0xd8], header.length);
  return result.buffer;
};

describe('recorded media association', () => {
  it('keeps legacy payload offsets and opted-in context offsets distinct', () => {
    expect([...splitMessage(packet()).payload]).toEqual([0xff, 0xd8]);
    const result = splitMessage(packet(7));
    expect(result.header.contextId).toBe(7);
    expect([...result.payload]).toEqual([0xff, 0xd8]);
    expect(() => splitMessage(packet(7).slice(0, 18))).toThrow('truncated');
  });
  it('waits for reordered control without letting later GOP packets pass, even at equal timestamps', () => {
    const received: number[] = [];
    const packets = new MediaPackets((_buffer, context) => received.push(context?.context_id ?? 0));
    packets.packet(packet(1));
    packets.context(media(2));
    packets.packet(packet(2));
    expect(received).toEqual([]);
    packets.context(media(1));
    expect(received).toEqual([1, 2]);
  });
  it('rejects changed snapshots, cross-stream contexts, and a bounded unresolved backlog', () => {
    const packets = new MediaPackets(() => {});
    packets.context(media(1));
    expect(() => packets.context({ ...media(1), profile_status: 'unavailable' })).toThrow('changed');
    packets.clear();
    packets.context(media(2, 3));
    expect(() => packets.packet(packet(2))).toThrow('mismatch');
    packets.clear();
    for (let i = 0; i < 600; i++) packets.packet(packet(99));
    expect(() => packets.packet(packet(99))).toThrow('backlog');
    packets.clear();
    packets.packet(packet());
  });
});
