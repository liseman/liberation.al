import { act, fireEvent, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { feedTaps, useLiveNow } from '../camera/mux';
import { usePrefs } from '../lib/prefs';
import { EnhanceButton, EnhanceLayer, EnhanceLook, useEnhanceUnsupported } from './Enhance';

// jsdom has no WebGL: where a test needs frames drawn, a stand-in takes the renderer's place
const drawer = vi.hoisted(() => ({ on: false, draw: (): void => {}, drawn: 0 }));
vi.mock('./renderer', async original => {
  const real = await original<typeof import('./renderer')>();
  class StandIn {
    setLook(): void {}
    draw(): void {
      drawer.drawn += 1;
      drawer.draw();
    }
    dispose(): void {}
  }
  return {
    ...real,
    Enhancer: function Enhancer(canvas: HTMLCanvasElement) {
      return drawer.on ? new StandIn() : new real.Enhancer(canvas);
    },
  };
});

const FRAME = { width: 4, height: 2, close: () => {} } as ImageBitmap;

function reducedMotion(on: boolean): void {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    query =>
      ({
        matches: on && query.includes('prefers-reduced-motion'),
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList,
  );
}

afterEach(() => {
  localStorage.clear();
  usePrefs.setState({ enhance: undefined, enhanceLook: undefined });
  useEnhanceUnsupported.setState({ reason: undefined });
  useLiveNow.setState(Object.create(null), true);
  Object.assign(drawer, { on: false, draw: () => {}, drawn: 0 });
  vi.restoreAllMocks();
});

describe("Live's Enhance button", () => {
  it('starts off, and is remembered in this browser when turned on', () => {
    renderPage(<EnhanceButton />);
    const button = screen.getByRole('button', { name: /^Enhance: steady the picture/ });
    expect(button).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(button);
    expect(button).toHaveAttribute('aria-pressed', 'true');
    expect(JSON.parse(localStorage.getItem('liberational.prefs') ?? '{}')).toMatchObject({ enhance: true });
  });

  it('says why it is unavailable, and does nothing, where the browser cannot run it', () => {
    useEnhanceUnsupported.setState({ reason: 'This browser has no WebGL2.' });
    renderPage(<EnhanceButton />);
    const button = screen.getByRole('button', { name: /isn't available in this browser: This browser has no WebGL2/ });
    expect(button).toHaveAttribute('aria-disabled', 'true');
    fireEvent.click(button);
    expect(usePrefs.getState().enhance).toBeFalsy();
  });
});

describe('the Enhance look', () => {
  it('offers Subtle, Contrast, and Colour, Subtle first, and remembers the choice', () => {
    reducedMotion(false);
    renderPage(<EnhanceLook />);
    expect(screen.getByRole('radio', { name: 'Subtle' })).toBeChecked();
    fireEvent.click(screen.getByRole('radio', { name: 'Contrast' }));
    expect(usePrefs.getState().enhanceLook).toBe('contrast');
    expect(screen.getByRole('radio', { name: 'Colour' })).toBeEnabled();
  });

  it('keeps the pulsing Colour look off when the system asks for reduced motion', () => {
    reducedMotion(true);
    usePrefs.setState({ enhanceLook: 'colour' });
    renderPage(<EnhanceLook />);
    expect(screen.getByRole('radio', { name: 'Colour' })).toBeDisabled();
    expect(screen.getByRole('radio', { name: 'Subtle' })).toBeChecked();
  });
});

describe('the Enhance layer', () => {
  it('falls back to the plain picture, and says so, without WebGL2', () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    renderPage(<EnhanceLayer cam="cam0" />);
    expect(useEnhanceUnsupported.getState().reason).toBe('This browser has no WebGL2.');
  });

  it('shows its picture again when frames come back after they stopped', () => {
    drawer.on = true;
    useLiveNow.setState({ cam0: true });
    const { container } = renderPage(<EnhanceLayer cam="cam0" />);
    const canvas = container.querySelector('canvas.enhance-canvas');
    act(() => feedTaps('cam0', FRAME, 1));
    expect(canvas).toHaveClass('drawn');
    act(() => useLiveNow.setState({ cam0: false }));
    expect(canvas).toHaveClass('paused');
    act(() => {
      feedTaps('cam0', FRAME, 9);
      useLiveNow.setState({ cam0: true });
    });
    expect(canvas).toHaveClass('drawn');
    expect(canvas).not.toHaveClass('paused');
  });

  it('gives the plain picture back, and says so, when drawing a frame fails', () => {
    drawer.on = true;
    const { container } = renderPage(<EnhanceLayer cam="cam0" />);
    act(() => feedTaps('cam0', FRAME, 1));
    expect(container.querySelector('canvas.enhance-canvas')).toHaveClass('drawn');
    drawer.draw = () => {
      throw new Error('This browser cannot render the running averages.');
    };
    act(() => feedTaps('cam0', FRAME, 2));
    expect(useEnhanceUnsupported.getState().reason).toMatch(/stopped/);
    // Its last picture must not stay over the live one, and it stops taking frames
    expect(container.querySelector('canvas.enhance-canvas')).not.toHaveClass('drawn');
    act(() => feedTaps('cam0', FRAME, 3));
    expect(drawer.drawn).toBe(2);
  });
});
