import { render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FakeEventSource, fakeFrames } from './testFakes';
import { level, mix, Waveform } from './Waveform';

describe('mix', () => {
  it('blends two colours channel by channel', () => {
    expect(mix('#000000', '#ffffff', 0.5)).toBe('rgb(128,128,128)');
    expect(mix('#9db8ff', '#ff9a3c', 0)).toBe('rgb(157,184,255)');
    expect(mix('#9db8ff', '#ff9a3c', 1)).toBe('rgb(255,154,60)');
  });
  it('picks the nearer one when a colour is not #rrggbb', () => {
    expect(mix('rgb(1, 2, 3)', '#ffffff', 0.4)).toBe('rgb(1, 2, 3)');
    expect(mix('rgb(1, 2, 3)', '#ffffff', 0.6)).toBe('#ffffff');
  });
});

describe('level', () => {
  it('stays flat within 6 dB of the noise floor', () => {
    expect(level(-60, -60)).toBe(0);
    expect(level(-54, -60)).toBe(0);
  });
  it('is full 36 dB above it, and curved in between', () => {
    expect(level(-24, -60)).toBe(1);
    expect(level(0, -60)).toBe(1);
    expect(level(-39, -60)).toBeCloseTo(0.5 ** 1.6, 6); // halfway through the 30 dB span
  });
});

describe('Waveform', () => {
  let frames: ReturnType<typeof fakeFrames>;
  const strokes: string[] = [];
  beforeEach(() => {
    FakeEventSource.all = [];
    strokes.length = 0;
    vi.stubGlobal('EventSource', FakeEventSource);
    frames = fakeFrames();
    const ctx = {
      setTransform() {},
      clearRect() {},
      beginPath() {},
      moveTo() {},
      lineTo() {},
      stroke() {},
      set strokeStyle(v: string) {
        strokes.push(v);
      },
    };
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
  });
  afterEach(() => vi.unstubAllGlobals());

  it('listens to the levels stream and warms toward orange while someone speaks', () => {
    render(<Waveform />);
    const [stream] = FakeEventSource.all;
    expect(stream?.url).toBe('/audio/levels');
    frames.step(16);
    expect(strokes.at(-1)).toBe('rgb(157,184,255)'); // the accent fallback, silent
    stream?.send({ levels: { desk: { rms: -40, peak: -30, speech: true }, room: { rms: -70, peak: -60 } } });
    frames.step(32);
    // warm eases 15% of the way each frame: 157 + 0.15 * (255 - 157) = 171.7, and so on.
    expect(strokes.at(-1)).toBe('rgb(172,180,226)');
  });

  it('closes the stream and stops drawing when unmounted', () => {
    const { unmount } = render(<Waveform />);
    unmount();
    expect(FakeEventSource.all[0]?.closed).toBe(true);
    expect(frames.pending).toBe(0);
  });
});
