// Mono 16-bit PCM WAV encoding for recorded voice clips (the People tab's recorder and Settings' owner voice).

/** Encode Float32 sample chunks (values in -1..1) as a mono 16-bit PCM WAV file. Out-of-range samples clamp. */
export function encodeWav(chunks: readonly Float32Array[], rate: number): Uint8Array {
  const n = chunks.reduce((t, c) => t + c.length, 0);
  const view = new DataView(new ArrayBuffer(44 + n * 2));
  const str = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, 'RIFF');
  view.setUint32(4, 36 + n * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  str(36, 'data');
  view.setUint32(40, n * 2, true);
  let o = 44;
  for (const c of chunks)
    for (const x of c) {
      const s = Math.max(-1, Math.min(1, x));
      view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      o += 2;
    }
  return new Uint8Array(view.buffer);
}
