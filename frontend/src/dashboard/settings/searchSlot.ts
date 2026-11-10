// Where Settings' search goes: the drawer's header row (#2450 shell follow-up). The drawer provides the element;
// first-run setup has no drawer, so the page keeps its search atop the menu.
import { createContext } from 'react';

export const SearchSlot = createContext<HTMLElement | null>(null);
