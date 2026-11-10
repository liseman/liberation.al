import { afterEach, expect, it } from 'vitest';
import { setChromeScheme, themeCss } from './theme';

/** The declarations of one rule of `css`, by its selector. */
function rule(css: string, selector: string): Record<string, string> {
  const body = css.split('\n').find(line => line.startsWith(`${selector}{`));
  if (!body) throw new Error(`no ${selector} rule`);
  return Object.fromEntries(
    body
      .slice(selector.length + 1, -1)
      .split(';')
      .filter(Boolean)
      .map(d => [d.slice(0, d.indexOf(':')), d.slice(d.indexOf(':') + 1)]),
  );
}

afterEach(() => setChromeScheme('light'));

it("gives the editor's page the dashboard's tokens, light on :root and dark on :root.dark", () => {
  const css = themeCss();
  const light = rule(css, ':root');
  const dark = rule(css, ':root.dark');
  // theme.ts's own values, as the dashboard has them
  expect(light['--lib-card']).toBe('#ffffff');
  expect(light['--mantine-color-text']).toBe('#192022');
  expect(dark['--lib-card']).toBe('var(--mantine-color-dark-6)');
  // and what a dark value names is there too: the dashboard's dark palette, not Mantine's default one
  expect(light['--mantine-color-dark-6']).toBe('#1d2427');
  // Mantine's own tokens that plan-editor.css maps onto, in both schemes
  for (const name of ['--mantine-color-dimmed', '--mantine-color-default-border', '--mantine-color-anchor']) {
    expect(light[name]).toBeTruthy();
    expect(dark[name]).toBeTruthy();
  }
  expect(light['--mantine-primary-color-filled']).toBeTruthy();
});

it("puts the editor's chrome in a scheme on <html> and <body>, and back", () => {
  setChromeScheme('dark');
  expect(document.documentElement.classList.contains('dark')).toBe(true);
  expect(document.body.classList.contains('dark')).toBe(true);
  setChromeScheme('light');
  expect(document.documentElement.classList.contains('dark')).toBe(false);
  expect(document.body.classList.contains('dark')).toBe(false);
});
