import { fireEvent, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { usePrefs } from '../lib/prefs';
import { SkeletonButton } from './Stage';

describe("Live's Skeletons button", () => {
  afterEach(() => {
    localStorage.clear();
    usePrefs.setState({ skeletons: undefined });
  });

  it('starts on, and switches skeletons off and on for Live and Review alike', () => {
    renderPage(<SkeletonButton />);
    const button = screen.getByRole('button', { name: "Show people's skeletons" });
    expect(button).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(button);
    expect(button).toHaveAttribute('aria-pressed', 'false');
    expect(usePrefs.getState().skeletons).toBe(false); // the one choice Review's Show skeletons reads too
    fireEvent.click(button);
    expect(usePrefs.getState().skeletons).toBe(true);
  });
});
