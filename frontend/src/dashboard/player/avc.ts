// AVC codec strings describe the SPS profile_idc, compatibility byte, and level_idc (RFC 6381 section 3.4).
// https://www.w3.org/TR/webcodecs-avc-codec-registration/#fully-qualified-codec-strings
// This reads parameter headers, not complete SPS/PPS syntax. WebCodecs still validates the actual bitstream.
export type AvcParameterSets =
  | { kind: 'none' }
  | { kind: 'invalid'; message: string }
  | { kind: 'ready'; codec: string; signature: string };

const MAX_PARAMETER_BYTES = 65_536;
const MAX_PARAMETER_SETS = 256;

/** Annex-B start codes may have three or four bytes, and SPS need not be the first NAL. */
function* units(bytes: Uint8Array): Generator<Uint8Array> {
  let start: number | null = null;
  for (let i = 0; i + 2 < bytes.length; i++) {
    if (bytes[i] !== 0 || bytes[i + 1] !== 0 || bytes[i + 2] !== 1) continue;
    if (start !== null) {
      let end = i;
      while (end > start && bytes[end - 1] === 0) end--;
      if (end > start) yield bytes.subarray(start, end);
    }
    start = i + 3;
    i += 2;
  }
  if (start !== null) {
    let end = bytes.length;
    while (end > start && bytes[end - 1] === 0) end--;
    if (end > start) yield bytes.subarray(start, end);
  }
}

const hex = (bytes: Uint8Array) => Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');

/** Complete in-band parameter sets are required to configure a new decoder; a bare IDR may reuse an existing one. */
export function avcParameterSets(bytes: Uint8Array): AvcParameterSets {
  let codec: string | null = null;
  let size = 0;
  let count = 0;
  let pps = false;
  const signatures = new Set<string>();
  for (const nal of units(bytes)) {
    const header = nal[0] ?? 0;
    const type = header & 31;
    if (type !== 7 && type !== 8) continue;
    size += nal.length;
    count++;
    if (size > MAX_PARAMETER_BYTES || count > MAX_PARAMETER_SETS)
      return { kind: 'invalid', message: 'H.264 parameter sets exceed their bound' };
    if ((header & 0x80) !== 0 || (header & 0x60) === 0 || nal.length < (type === 7 ? 4 : 2))
      return { kind: 'invalid', message: 'Malformed H.264 parameter header' };
    if (type === 7) {
      const profile = nal[1] ?? 0;
      const compatibility = nal[2] ?? 0;
      const level = nal[3] ?? 0;
      if (profile === 0 || level === 0 || (compatibility & 3) !== 0)
        return { kind: 'invalid', message: 'Malformed H.264 SPS profile, compatibility, or level' };
      const candidate = `avc1.${hex(nal.subarray(1, 4))}`;
      if (codec !== null && candidate !== codec)
        return { kind: 'invalid', message: 'Conflicting H.264 SPS codec headers' };
      codec = candidate;
    } else pps = true;
    signatures.add(hex(nal));
  }
  if (count === 0) return { kind: 'none' };
  if (codec === null || !pps) return { kind: 'invalid', message: 'Incomplete H.264 SPS/PPS parameter sets' };
  return { kind: 'ready', codec, signature: [...signatures].sort().join(':') };
}
