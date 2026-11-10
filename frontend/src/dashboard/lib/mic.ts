// Recording from the browser's microphone as a mono 16-bit WAV (the People tab's recorder and Settings).
import { useCallback, useEffect, useRef, useState } from 'react';
import { encodeWav } from './wav';

/** Seconds of the recording that have sound in them: 50 ms frames louder than -40 dBFS. The server checks the same. */
export function voicedSeconds(chunks: readonly Float32Array[], rate: number): number {
  const hop = Math.max(1, Math.round(rate * 0.05));
  let n = 0;
  let acc = 0;
  let cnt = 0;
  for (const c of chunks)
    for (const x of c) {
      acc += x * x;
      if (++cnt === hop) {
        if (Math.sqrt(acc / hop) > 0.01) n++;
        acc = 0;
        cnt = 0;
      }
    }
  return (n * hop) / rate;
}

export type MicPhase = 'idle' | 'rec' | 'done';

export interface MicRecorder {
  phase: MicPhase;
  secs: number; // elapsed while recording
  start: () => Promise<void>;
  stop: () => void;
  wav: Blob | null; // once stopped
  seconds: number; // its length
  voiced: number; // seconds of it with sound in them
  error: string | null;
}

interface Live {
  stream: MediaStream;
  ctx: AudioContext;
  src: MediaStreamAudioSourceNode;
  proc: ScriptProcessorNode;
  chunks: Float32Array[];
  timer: ReturnType<typeof setInterval>;
}

/** Record up to maxS seconds from the microphone, raw (no echo cancellation, noise suppression, or gain). */
export function useMicRecorder(maxS: number): MicRecorder {
  const [phase, setPhase] = useState<MicPhase>('idle');
  const [secs, setSecs] = useState(0);
  const [wav, setWav] = useState<Blob | null>(null);
  const [seconds, setSeconds] = useState(0);
  const [voiced, setVoiced] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const rec = useRef<Live | null>(null);
  const mounted = useRef(false); // false once the page that started it is gone

  // Reads only the ref, so one copy serves every render, and unmounting lets go of the microphone.
  const release = useCallback(() => {
    const r = rec.current;
    if (!r) return;
    clearInterval(r.timer);
    r.proc.disconnect();
    r.src.disconnect();
    for (const t of r.stream.getTracks()) t.stop();
    void r.ctx.close();
    rec.current = null;
  }, []);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      // Hidden mid-recording (its tab stays mounted, #330): the take is let go, so it shows as not recording
      if (rec.current) {
        setPhase('idle');
        setSecs(0);
      }
      release();
    };
  }, [release]);

  // Reads only the ref and setters, so the timer's copy of it is never stale.
  const stop = () => {
    const r = rec.current;
    if (!r) return;
    const rate = r.ctx.sampleRate;
    const chunks = r.chunks;
    release();
    setSeconds(chunks.reduce((t, c) => t + c.length, 0) / rate);
    setVoiced(voicedSeconds(chunks, rate));
    setWav(new Blob([encodeWav(chunks, rate) as BlobPart], { type: 'audio/wav' }));
    setPhase('done');
  };

  const start = async () => {
    setWav(null);
    setError(null);
    try {
      // Raw capture: this records a voice reference, and the browser's echo cancellation removes anything the
      // computer is playing, so a voice played through the speakers was recorded as silence (2026-10-26).
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
      // The browser may ask for permission first. If the page closed while it asked, let the microphone go at once
      // instead of recording into nothing until maxS.
      if (!mounted.current) {
        for (const t of stream.getTracks()) t.stop();
        return;
      }
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      // ScriptProcessor is deprecated but needs no extra worklet file and works everywhere this page runs.
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      const chunks: Float32Array[] = [];
      proc.onaudioprocess = e => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      src.connect(proc);
      proc.connect(ctx.destination); // Chrome only runs the processor when it is connected; its output is silence
      const t0 = Date.now();
      const timer = setInterval(() => {
        const s = (Date.now() - t0) / 1000;
        setSecs(s);
        if (s >= maxS) stop();
      }, 250);
      rec.current = { stream, ctx, src, proc, chunks, timer };
      setSecs(0);
      setPhase('rec');
    } catch (e) {
      const name = (e as { name?: unknown } | null)?.name || String(e); // a DOMException is not an Error everywhere
      setError(`The microphone is unavailable (${name}). Allow microphone access for this page and try again.`);
    }
  };

  return { phase, secs, start, stop, wav, seconds, voiced, error };
}
