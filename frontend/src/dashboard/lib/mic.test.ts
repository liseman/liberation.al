import { act, renderHook } from '@testing-library/react';
import { Activity, createElement, type ReactNode, useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useMicRecorder, voicedSeconds } from './mic';

let show: (on: boolean) => void = () => {};
/** A tab that can be hidden, as App shows one (#330). */
function Tab({ children }: { children: ReactNode }) {
  const [on, setOn] = useState(true);
  show = setOn;
  return createElement(Activity, { mode: on ? 'visible' : 'hidden', children });
}

describe('voicedSeconds', () => {
  it('counts the 50 ms frames louder than -40 dBFS', () => {
    const rate = 1000; // 50 samples a frame
    const loud = new Float32Array(100).fill(0.02); // two frames above 0.01 RMS
    const quiet = new Float32Array(150).fill(0.005); // three frames below it
    expect(voicedSeconds([loud, quiet], rate)).toBeCloseTo(0.1);
  });

  it('drops a partial frame at the end', () => {
    expect(voicedSeconds([new Float32Array(49).fill(1)], 1000)).toBe(0);
  });
});

describe('useMicRecorder', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('says how to fix a refused microphone', async () => {
    vi.stubGlobal('navigator', {
      mediaDevices: { getUserMedia: () => Promise.reject(new DOMException('denied', 'NotAllowedError')) },
    });
    const { result } = renderHook(() => useMicRecorder(30));
    await act(() => result.current.start());
    expect(result.current.phase).toBe('idle');
    expect(result.current.error).toBe(
      'The microphone is unavailable (NotAllowedError). Allow microphone access for this page and try again.',
    );
  });

  it('records raw audio and stops into a WAV with its length and voiced seconds', async () => {
    let onaudioprocess: ((e: { inputBuffer: { getChannelData: () => Float32Array } }) => void) | null = null;
    const track = { stop: vi.fn() };
    const getUserMedia = vi.fn(() => Promise.resolve({ getTracks: () => [track] }));
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia } });
    const node = () => ({ connect: vi.fn(), disconnect: vi.fn() });
    vi.stubGlobal(
      'AudioContext',
      class {
        sampleRate = 1000;
        destination = {};
        close = vi.fn(() => Promise.resolve());
        createMediaStreamSource = node;
        createScriptProcessor = () => ({
          ...node(),
          set onaudioprocess(f: typeof onaudioprocess) {
            onaudioprocess = f;
          },
        });
      },
    );
    const { result } = renderHook(() => useMicRecorder(30));
    await act(() => result.current.start());
    expect(getUserMedia).toHaveBeenCalledWith({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    expect(result.current.phase).toBe('rec');
    act(() => onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array(500).fill(0.5) } }));
    act(() => result.current.stop());
    expect(result.current.phase).toBe('done');
    expect(result.current.seconds).toBeCloseTo(0.5);
    expect(result.current.voiced).toBeCloseTo(0.5);
    expect(result.current.wav?.type).toBe('audio/wav');
    expect(result.current.wav?.size).toBe(44 + 500 * 2);
    expect(track.stop).toHaveBeenCalled(); // the microphone is let go
  });

  it('goes back to idle, not a dead "recording", when its tab is hidden mid-recording (#330)', async () => {
    const track = { stop: vi.fn() };
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: () => Promise.resolve({ getTracks: () => [track] }) } });
    const node = () => ({ connect: vi.fn(), disconnect: vi.fn() });
    vi.stubGlobal(
      'AudioContext',
      class {
        sampleRate = 1000;
        destination = {};
        close = vi.fn(() => Promise.resolve());
        createMediaStreamSource = node;
        createScriptProcessor = node;
      },
    );
    const { result } = renderHook(() => useMicRecorder(30), { wrapper: Tab });
    await act(() => result.current.start());
    expect(result.current.phase).toBe('rec');
    act(() => show(false)); // the tab is hidden: the microphone is let go
    act(() => show(true));
    expect(track.stop).toHaveBeenCalled();
    expect(result.current.phase).toBe('idle');
  });

  it('lets the microphone go when the page closes while the browser asks for permission', async () => {
    let grant: (stream: unknown) => void = () => {};
    const track = { stop: vi.fn() };
    vi.stubGlobal('navigator', {
      mediaDevices: { getUserMedia: () => new Promise(resolve => (grant = resolve)) },
    });
    const AudioContext = vi.fn();
    vi.stubGlobal('AudioContext', AudioContext);
    const { result, unmount } = renderHook(() => useMicRecorder(30));
    let started: Promise<void> = Promise.resolve();
    act(() => {
      started = result.current.start();
    });
    unmount(); // the form closed while the permission prompt was up
    await act(async () => {
      grant({ getTracks: () => [track] });
      await started;
    });
    expect(track.stop).toHaveBeenCalled();
    expect(AudioContext).not.toHaveBeenCalled();
  });
});
