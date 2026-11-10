// The dashboard's theme inside the plan editor's page. The page is a document of its own (an iframe), so the
// dashboard's CSS variables don't reach it: this builds the same ones from the same source (shared/theme.ts, through
// Mantine's own resolver), so every --mantine-* and --lib-* token has the value it has on the dashboard, in light and
// dark. plan-editor.css maps the editor's own theme variables onto them, so a change to theme.ts reaches the editor
// with nothing here to touch. The scheme is a `dark` class on <html> and <body>: the editor's chrome follows that
// class, and its menus are added to <body>.
import { DEFAULT_THEME, defaultCssVariablesResolver, mergeMantineTheme } from '@mantine/core';
import { cssVariablesResolver, theme } from '../shared/theme';

type Variables = Record<string, string>;

const block = (selector: string, variables: Variables): string =>
  `${selector}{${Object.entries(variables)
    .map(([name, value]) => `${name}:${value};`)
    .join('')}}`;

/** The dashboard's CSS variables as a stylesheet: the shared ones and the light ones on :root, the dark ones on
 *  :root.dark. */
export function themeCss(): string {
  const merged = mergeMantineTheme(DEFAULT_THEME, theme);
  const mantine = defaultCssVariablesResolver(merged);
  const ours = cssVariablesResolver(merged);
  return [
    block(':root', { ...mantine.variables, ...ours.variables, ...mantine.light, ...ours.light }),
    block(':root.dark', { ...mantine.dark, ...ours.dark }),
  ].join('\n');
}

/** Put the dashboard's variables into this page, once. */
export function installTheme(): void {
  const style = document.createElement('style');
  style.dataset.theme = 'dashboard';
  style.textContent = themeCss();
  document.head.append(style);
}

/** Show the editor's chrome in a scheme: its `dark` class, on <html> and on <body> (where its menus go). */
export function setChromeScheme(scheme: 'light' | 'dark'): void {
  for (const el of [document.documentElement, document.body]) el.classList.toggle('dark', scheme === 'dark');
}
