// The command palette's open state, shared by the top bar's button and the lazily loaded palette.
import { createSpotlight } from '@mantine/spotlight';

export const [paletteStore, palette] = createSpotlight();
