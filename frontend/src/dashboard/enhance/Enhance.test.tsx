import { fireEvent, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { usePrefs } from '../lib/prefs';
import { EnhanceButton, EnhanceLayer, EnhanceLook, useEnhanceUnsupported } from './Enhance';

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
});
