// Mantine themed with the old dashboard's palette (static/base.css): its accent, surfaces, and status colours, in
// light and dark. Status colours are CSS variables (--lib-ok, --lib-warn, --lib-bad, --lib-live) so any component
// can use them; --lib-speech is the serif that transcripts are set in. --lib-card, --lib-raise, --lib-bg2, --lib-dim,
// --lib-hi, and --lib-acc-ink stand in for the old pages' --card, --raise, --bg2, --dim, --hi, and --acc-ink, so a
// tab's old CSS ports by renaming them. --lib-ok-text is --lib-ok as small text, dark enough for AA contrast.
// --lib-bad-tint, --lib-warn-tint, and --lib-ok-tint are the status colours as a ground (an error box, a halo),
// --lib-bad-row a fainter one for a whole row, --lib-track the empty part of a meter, and --lib-track-off a switch
// that's off.
// The component overrides below are plain objects, not Card.extend() and the like: .extend is an identity function, and
// importing a component only to call it pulls that component's code into the entry bundle for every page.
import {
  type CSSVariablesResolver,
  createTheme,
  type MantineColorsTuple,
  type MantineTheme,
  type MantineThemeComponent,
  type TabsProps,
} from '@mantine/core';

// Filled controls use shade 6 in both schemes. Dark foreground accents stay light for contrast.
const accent: MantineColorsTuple = [
  '#eef2ff',
  '#dbe3ff',
  '#bccbff',
  '#9db8ff',
  '#7f9cf5',
  '#5a78e6',
  '#3454d1',
  '#2a45b0',
  '#22398f',
  '#1a2c70',
];

// Dark surfaces, in Mantine's order: 0 is text, 2 dimmed, 4 borders, 5 raised, 6 cards and inputs, 7 the page.
const dark: MantineColorsTuple = [
  '#e4eae7',
  '#c3ccc8',
  '#95a39f',
  '#66746f',
  '#2c363a',
  '#252e32',
  '#1d2427',
  '#161b1e',
  '#111518',
  '#0b0e10',
];

const ui = '"Figtree", system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", sans-serif';
const surface = { root: { backgroundColor: 'var(--lib-card)' } };
// A dialog's content is a Paper (so --lib-card, above), but its header isn't: give the header the same surface, so
// the title and the body read as one card instead of a strip on a block (#210).
const dialog = { header: { backgroundColor: 'var(--lib-card)' }, content: { backgroundColor: 'var(--lib-card)' } };
// Their close (x) button has only an icon, so it says what it does to a screen reader (#478).
const dialogComponent: MantineThemeComponent = {
  styles: dialog,
  defaultProps: { closeButtonProps: { 'aria-label': 'Close' } },
};

export const theme = createTheme({
  primaryColor: 'accent',
  primaryShade: { light: 6, dark: 6 },
  colors: { accent, dark },
  autoContrast: true,
  fontFamily: ui,
  fontFamilyMonospace: '"Cousine", ui-monospace, "SF Mono", Menlo, Consolas, monospace',
  // One section size everywhere (#2594): page titles are 24 through PageHeader, a section heading (h2) is 18/600, and
  // a heading inside a section (h3) is 16/600. h4 and below keep Mantine's sizes, which some titles still name.
  headings: {
    fontFamily: ui,
    fontWeight: '600',
    sizes: {
      h2: { fontSize: '18px', lineHeight: '1.45', fontWeight: '600' },
      h3: { fontSize: '16px', lineHeight: '1.5', fontWeight: '600' },
    },
  },
  radius: { xs: '1px', sm: '2px', md: '4px', lg: '8px', xl: '16px' },
  defaultRadius: 'md',
  components: {
    Card: { styles: surface },
    Paper: { styles: surface },
    Tabs: {
      vars: (_theme: MantineTheme, props: TabsProps) => ({
        root:
          props.variant === 'pills' || (props.color && props.color !== 'accent')
            ? {}
            : { '--tabs-color': 'var(--mantine-color-anchor)' },
      }),
    },
    Modal: dialogComponent,
    Drawer: dialogComponent,
  },
});

export const cssVariablesResolver: CSSVariablesResolver = () => ({
  variables: {
    '--lib-speech':
      '"Aleo", Charter, "Bitstream Charter", "Iowan Old Style", "Sitka Text", Cambria, "Noto Serif", "DejaVu Serif", Georgia, serif',
  },
  light: {
    '--mantine-color-body': '#f2f4f3',
    '--mantine-color-text': '#192022',
    '--mantine-color-dimmed': '#56625f',
    '--mantine-color-placeholder': '#8a9692',
    '--mantine-color-default-border': '#d9dfdc',
    '--lib-card': '#ffffff',
    '--lib-raise': '#f7f9f8',
    '--lib-bg2': '#e8ecea',
    '--lib-dim': '#5f6b67', // 5.0:1 on the page ground, 5.6:1 on white: AA for small text (was #8a9692, 2.8:1)
    '--lib-hi': 'rgba(52, 84, 209, 0.08)',
    '--lib-acc-ink': '#ffffff',
    '--lib-ok': '#2e8a57',
    '--lib-ok-text': '#237a35', // --lib-ok as small text: 5.4:1 on white, 4.9:1 on the page ground (AA)
    '--lib-warn': '#a86b00',
    '--lib-warn-text': '#9a6200', // --lib-warn as small text: 5.1:1 on white, 4.6:1 on the page ground (AA)
    '--lib-bad': '#c8402c',
    '--lib-live': '#d23f2a',
    '--lib-bad-tint': 'rgba(200, 64, 44, 0.08)',
    '--lib-bad-row': 'rgba(200, 64, 44, 0.035)',
    '--lib-warn-tint': 'rgba(168, 107, 0, 0.1)',
    '--lib-ok-tint': 'rgba(46, 138, 87, 0.1)',
    '--lib-track': '#e3e8e6',
    '--lib-track-off': '#c9d1cd',
  },
  dark: {
    '--lib-card': 'var(--mantine-color-dark-6)',
    '--lib-raise': 'var(--mantine-color-dark-5)',
    '--lib-bg2': 'var(--mantine-color-dark-6)',
    '--lib-dim': 'var(--mantine-color-dark-3)',
    '--lib-hi': 'rgba(157, 184, 255, 0.1)',
    '--lib-acc-ink': '#ffffff',
    '--mantine-color-anchor': 'var(--mantine-color-accent-3)',
    '--mantine-color-accent-text': 'var(--mantine-color-accent-3)',
    '--mantine-color-accent-light-color': 'var(--mantine-color-accent-3)',
    '--mantine-color-accent-outline': 'var(--mantine-color-accent-3)',
    '--lib-ok': '#7fcb9c',
    '--lib-ok-text': '#7fcb9c',
    '--lib-warn': '#e9b65a',
    '--lib-warn-text': 'var(--lib-warn)',
    '--lib-bad': '#ff7a66',
    '--lib-live': '#ff6a55',
    '--lib-bad-tint': 'rgba(255, 122, 102, 0.12)',
    '--lib-bad-row': 'rgba(255, 122, 102, 0.045)',
    '--lib-warn-tint': 'rgba(233, 182, 90, 0.1)',
    '--lib-ok-tint': 'rgba(127, 203, 156, 0.1)',
    '--lib-track': 'var(--mantine-color-dark-4)',
    '--lib-track-off': '#3a464b',
  },
});
