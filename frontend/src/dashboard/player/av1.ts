// AV1 codec strings name the sequence header's profile, level, tier, and bit depth: av01.P.LLT.DD
// (https://aomediacodec.github.io/av1-isobmff/#codecsparam). The archive pass stores one temporal unit per
// frame in the low-overhead OBU format, its keyframe carrying the sequence header in band. This reads the header's
// first fields, not the whole syntax; WebCodecs still validates the actual bitstream. The pass always encodes 8-bit
// 4:2:0, so the depth is 08: a decoder takes the true depth from the in-band header either way.
export type Av1Sequence =
  | { kind: 'none' }
  | { kind: 'invalid'; message: string }
  | { kind: 'ready'; codec: string; signature: string };

const SEQUENCE_HEADER = 1;
const TEMPORAL_DELIMITER = 2;

/** An OBU stream starts with an OBU header (forbidden bit clear), never Annex B's zero bytes. */
export function looksLikeAv1(bytes: Uint8Array): boolean {
  const b0 = bytes[0];
  return b0 !== undefined && (b0 & 0x80) === 0 && ((b0 >> 3) & 0xf) === TEMPORAL_DELIMITER;
}

class Bits {
  private bit = 0;
  private readonly bytes: Uint8Array;
  constructor(bytes: Uint8Array) {
    this.bytes = bytes;
  }
  read(n: number): number {
    let out = 0;
    for (let i = 0; i < n; i++) {
      const byte = this.bytes[this.bit >> 3];
      if (byte === undefined) throw new RangeError('AV1 sequence header ends early');
      out = out * 2 + ((byte >> (7 - (this.bit & 7))) & 1);
      this.bit++;
    }
    return out;
  }
  /** uvlc(): leading zeros, then that many bits. */
  uvlc(): number {
    let zeros = 0;
    while (this.read(1) === 0) if (++zeros >= 32) throw new RangeError('AV1 uvlc too long');
    return zeros === 0 ? 0 : this.read(zeros) + 2 ** zeros - 1;
  }
}

/** Each OBU's type and payload; `null` for a malformed stream. */
function obus(bytes: Uint8Array): { type: number; payload: Uint8Array }[] | null {
  const out = [];
  let i = 0;
  while (i < bytes.length) {
    const header = bytes[i] ?? 0;
    if ((header & 0x80) !== 0) return null;
    const type = (header >> 3) & 0xf;
    const extension = (header >> 2) & 1;
    const hasSize = (header >> 1) & 1;
    i += 1 + extension;
    let size = bytes.length - i;
    if (hasSize) {
      size = 0;
      for (let k = 0; ; k++) {
        const b = bytes[i++];
        if (b === undefined || k >= 8) return null;
        size += (b & 0x7f) * 2 ** (7 * k);
        if ((b & 0x80) === 0) break;
      }
    }
    if (i + size > bytes.length) return null;
    out.push({ type, payload: bytes.subarray(i, i + size) });
    i += size;
  }
  return out;
}

const hex = (bytes: Uint8Array) => Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');

/** The codec string from a keyframe's sequence header; `none` for a unit without one (it reuses the decoder). */
export function av1Sequence(bytes: Uint8Array): Av1Sequence {
  const units = obus(bytes);
  if (units === null) return { kind: 'invalid', message: 'Malformed AV1 OBU stream' };
  const header = units.find(u => u.type === SEQUENCE_HEADER);
  if (!header) return { kind: 'none' };
  try {
    const r = new Bits(header.payload);
    const profile = r.read(3);
    r.read(1); // still_picture
    const reduced = r.read(1);
    let level: number;
    let tier = 0;
    if (reduced) level = r.read(5);
    else {
      if (r.read(1)) {
        // timing_info, then the decoder model info it may carry; the first operating point comes after both
        r.read(32);
        r.read(32);
        if (r.read(1)) r.uvlc();
        if (r.read(1)) {
          r.read(5);
          r.read(32);
          r.read(5);
          r.read(5);
        }
      }
      r.read(1); // initial_display_delay_present_flag: only the operating points' later fields read it
      r.read(5); // operating_points_cnt_minus_1: the first operating point describes the whole stream
      r.read(12); // operating_point_idc
      level = r.read(5);
      if (level > 7) tier = r.read(1);
    }
    if (profile > 2) return { kind: 'invalid', message: `Unknown AV1 profile ${profile}` };
    const codec = `av01.${profile}.${String(level).padStart(2, '0')}${tier ? 'H' : 'M'}.08`;
    return { kind: 'ready', codec, signature: hex(header.payload) };
  } catch {
    return { kind: 'invalid', message: 'Malformed AV1 sequence header' };
  }
}
