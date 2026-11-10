import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { ReviewHelp } from './ReviewHelp';
import classes from './review.module.css';

const css = readFileSync(resolve(import.meta.dirname, 'review.module.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');

/** The classes some rule in review.module.css hides (display: none), whatever media query it sits under. */
function hiddenClasses(): string[] {
  const hidden: string[] = [];
  for (const [, selectors = '', body = ''] of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (!/display:\s*none/.test(body)) continue;
    for (const s of selectors.split(',')) if (/^\s*\.[A-Za-z]\w*\s*$/.test(s)) hidden.push(s.trim().slice(1));
  }
  return hidden;
}

// The stacked layout (a window under 4:3, 900 px wide, or 600 px tall) hides the inline hints; the "?" table of keys
// must not be caught by it, since a narrow window can still have a keyboard.
it("shows the shortcuts in the '?' popover however the window is shaped", async () => {
  renderPage(<ReviewHelp timeline />);
  await userEvent.click(screen.getByRole('button', { name: 'Shortcuts and timeline help' }));
  const table = await screen.findByRole('table', { name: 'Keyboard shortcuts' });
  expect(hiddenClasses()).toContain('hint'); // the stacked rule is still read, so this checks something
  const hiddenNames = hiddenClasses().map(name => (classes as Record<string, string>)[name]);
  for (let el: HTMLElement | null = table; el && el !== document.body; el = el.parentElement)
    for (const name of hiddenNames) expect(el.classList.contains(name ?? '')).toBe(false);
});

// The accessibility pass: the panel is read and scrolled, so it takes focus as it opens and gives it back.
it("moves focus into the '?' panel as it opens, and back to '?' when Escape closes it", async () => {
  renderPage(<ReviewHelp timeline />);
  const button = screen.getByRole('button', { name: 'Shortcuts and timeline help' });
  await userEvent.click(button);
  const panel = await screen.findByRole('region', { name: 'Shortcuts and timeline help' });
  await waitFor(() => expect(panel).toHaveFocus());
  await userEvent.keyboard('{Escape}');
  await waitFor(() => expect(button).toHaveFocus());
  expect(screen.queryByRole('region', { name: 'Shortcuts and timeline help' })).toBeNull();
});
