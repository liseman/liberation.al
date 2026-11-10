// The command palette's open state (#2586), shared by the top bar's button and the lazily loaded palette.
import { createSpotlight } from '@mantine/spotlight';

export const [paletteStore, palette] = createSpotlight();
