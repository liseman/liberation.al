---
name: mantine
description: >
  How this repository's dashboard (frontend/) uses Mantine 9: the packages, the theme and its --lib-* tokens, the
  shared components to reuse before raw Mantine, light and dark, and where Mantine's own docs, skills, and MCP server
  are. Load before writing or reviewing any dashboard UI: a component, a CSS module, a hook, a loading or error state,
  or a test that renders one.
---

# Mantine in this dashboard

This file is the project's own. Mantine's vendored skills sit beside it (`mantine-combobox`, `mantine-form`, and
`mantine-custom-components`, each with a `NOTICE.md`); load one when its task comes up. Where they and this file
differ about this repository, this file wins.

## Versions and docs

- `frontend/package.json` pins `^9.7.5` for every `@mantine/*` package. Check `frontend/package-lock.json` for the
  installed version before trusting a doc page about a feature added in a later minor.
- Mantine's docs for agents follow the latest release (9.x); no older 9.x copy is published. Read the changelog page
  for a minor newer than ours before using what it added.
- Without MCP, fetch one page at a time: [mantine.dev/llms.txt](https://mantine.dev/llms.txt) is the index, and each
  page is `https://mantine.dev/llms/<group>-<name>.md` (`core-skeleton.md`, `hooks-use-media-query.md`,
  `theming-color-schemes.md`, `styles-postcss-preset.md`). [llms-full.txt](https://mantine.dev/llms-full.txt) is
  every page in one 4.6 MB file: grep a downloaded copy, never read it whole.
- With MCP, the project's `.mcp.json` runs `@mantine/mcp-server` (pinned). Its tools: `search_docs`, `get_item_doc`
  (outline, one `section`, or `full`), `get_item_props` (a component's props or a hook's signature), `list_items`, and
  `get_api`. It fetches the same static docs from mantine.dev on each lookup; offline it has nothing.

## Packages

`@mantine/core`, `hooks`, `dates`, `modals`, `notifications`, and `spotlight` (the command palette, `palette/`).
Icons are `@tabler/icons-react`. Not installed: `@mantine/form` (the forms here are small and controlled; reach for
it, with the `mantine-form` skill, only for a form with real validation, nested fields, or lists), `@mantine/charts`
(charts are our own SVG), `@mantine/dropzone`, and `@mantine/carousel`. Adding a package is fine when it replaces
hand-written code; say so in the PR.

## Theme and tokens

`frontend/src/shared/theme.ts` is the theme; `shared/Providers.tsx` wraps every page (and every test, in Mantine's
test mode, through `src/test/render.tsx`). Read the comment at the top of `theme.ts` before picking a colour.

- Primary colour `accent`, shade 6 in both schemes, `autoContrast`; `defaultRadius` md (4px); Figtree for UI,
  Cousine for monospace, `--lib-speech` (a serif) for transcript text.
- Headings: page titles come from `PageHeader` (24px); a section heading is `h2` (18/600), a heading in a section
  `h3` (16/600).
- Colours go through tokens, never a hex in a component: `--lib-ok`, `--lib-warn`, `--lib-bad`, and `--lib-live` for
  status; `--lib-ok-text` and `--lib-warn-text` when that status is small text (they meet AA on cards and the page);
  `--lib-*-tint` as a ground, `--lib-bad-row` for a whole row; `--lib-card`, `--lib-raise`, `--lib-bg2`, `--lib-dim`,
  `--lib-hi`, `--lib-track`, and `--lib-track-off` for surfaces. Mantine's own (`--mantine-color-dimmed`,
  `--mantine-color-default-border`, `--mantine-color-default-hover`, `--mantine-color-anchor`) are fine too.
- Every token has a light and a dark value already. A component needs no dark override for a token; if one seems
  needed, the token is wrong or missing: fix it in `cssVariablesResolver`.
- Component overrides in `theme.ts` are plain objects, not `Component.extend()` (that import puts the component in
  every page's entry bundle).

## Reuse before raw Mantine

`frontend/src/dashboard/lib/` holds the shared pieces. Search it before writing a new one.

- **Page frame and states** (`page.tsx`): `PageFrame` and `PageHeader` for every page; `QueryState` for a TanStack
  query's loading, error, and data; `PageLoading`, `PageEmpty`, `PageError`, and `LoadError`; `ViewMenu` to fold view
  controls on a phone; `PHONE_WIDTH` as the phone breakpoint.
- **Loading** (`loading.tsx`): skeletons shaped like what is coming: `RowsSkeleton`, `BlockSkeleton`, and
  `CountSkeleton` for a count still loading inside a line (a tab, a button, a heading), never a spinner or "...".
  `SlowLoader` and `useSlow` show a loader only once a wait runs long.
- **Messages** (`message.tsx`, `Toast.tsx`): a change's result is a `Message`, said by `StatusLine` under what it is
  about or by `Toast` (Mantine notifications, one queue, Undo support).
- **Asking first** (`confirm.tsx`): `confirm()` through `@mantine/modals`, never `window.confirm`.
- **Others**: `Disclosure` (a collapsible section), `FilterChip`, `SelectableGrid` (keyboard grid of tiles),
  `PersonNameInput` and `LazyPersonNameInput`, `PlayToggle` and `PlayClip`, and `format.ts` for times, ages, and
  plurals.
- **Status badges**: `Badge variant="light"` with `tt="none"` and `fw={500}` (Mantine's uppercase default is not used
  for status), coloured `teal`, `yellow`, `red`, or `gray`; a yellow one sets `c="var(--lib-warn-text)"` for contrast.
  `known/DeviceSections.tsx`, `journal/JournalTab.tsx`, and `settings/SettingsForm.tsx` show it.
- **Cards**: `Card` and `Paper` already sit on `--lib-card` through the theme; don't set their background.

## Light and dark

The scheme is `auto` by default and the owner can pin it, so every change is checked in both. In CSS modules, prefer
in this order: a token that already has both values; `light-dark(<light>, <dark>)`; `@mixin dark { ... }` (or
`@mixin light`) from `postcss-preset-mantine`. Never `@media (prefers-color-scheme)`: it follows the system, not the
dashboard's setting. In TSX, `useComputedColorScheme` only when a value can't be CSS (a canvas or WebGL colour).

## CSS

- CSS modules beside the component; Mantine props (`gap`, `p`, `c`, `fw`) for one-off spacing and type.
- Hover with `@mixin hover { ... }`, not `:hover`: the dashboard is used on phones, where `:hover` sticks after a tap.
  `frontend/biome.json` allows Mantine's `@mixin` at-rules; `postcss-preset-mantine` expands them at build time.
- Breakpoints: `PHONE_WIDTH` (720px) in TS; in CSS the same number, or `$mantine-breakpoint-*` (see
  `frontend/postcss.config.mjs`). In TS, `useMediaQuery(q, undefined, { getInitialValueInEffect: false })` reads the
  right value on the first render.
- Never style Mantine's private `--button-*`, `--input-*` variables; use the Styles API (`classNames`, `styles`, or
  `vars`) instead.

## Don't

- Hand-roll what Mantine or `@mantine/hooks` already has: a debounce (`useDebouncedValue`, `useDebouncedCallback`), a
  media query listener, click-outside, a hotkey (`useHotkeys`), an interval, a disclosure toggle (`useDisclosure`),
  or a dropdown (`Combobox`, `Menu`, `Popover`).
- Put a raw `<button>`, `<input>`, or `<select>` in a component when `UnstyledButton`, `TextInput`, or `Select` fits.
- Add a second notification, modal, or toast system; or a `window.confirm`, `alert`, or `prompt`.
- Use a hex colour, a hand-picked `--mantine-color-<name>-<n>` shade for a status, or a dark override for a token.
- Leave an icon-only control without an accessible name (`aria-label` on `ActionIcon`).

## Checks

From `frontend/`, through `tools/test-slot.sh`: `npx vitest related <changed files> --run`, `npm run lint` (Biome and
`tsc`), and `npm run build` (both Vite builds) when a build could break. Never `mise run frontend` in a worktree.

## Modern web guidance

`.claude/skills/modern-web-guidance` (with its `PROJECT.md`) covers HTML, CSS, browser APIs, performance, and
accessibility. Its guides are written for plain HTML and CSS. For this dashboard, this file and Mantine's docs come
first: where a guide says to use a native `<dialog>`, the Popover API, anchor positioning, or a native `<select>` and
Mantine has the component (`Modal`, `Popover`, `Menu`, `Select`, `Combobox`), use Mantine's, so focus handling,
portals, z-index, and theming stay one system. Use the guides where Mantine has no opinion: CSS layout and type
(container queries, `text-wrap`, `subgrid`), performance (INP, long tasks, `content-visibility`), media and
browser APIs, and accessibility checks.
