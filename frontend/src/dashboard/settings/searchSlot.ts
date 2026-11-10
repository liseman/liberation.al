// Where Settings' search goes: the drawer's header row. The drawer provides the element;
// first-run setup has no drawer, so the page keeps its search atop the menu.
import { createContext } from 'react';

export const SearchSlot = createContext<HTMLElement | null>(null);
