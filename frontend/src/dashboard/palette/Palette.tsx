// The command palette: Ctrl+K (Cmd+K on a Mac), or the Go to button in the top bar, opens Mantine's Spotlight
// to jump anywhere: a page, a Settings section, a person's or a pet's card by name, a day in Review ("yesterday",
// "nov 3"), a common next step (what waits to be named, clips to check), or Ask. Everything else only navigates. People and
// pets are read while it's open. It loads after the page (lazily), so the first page's own reads come first.
//
// Built from Spotlight's parts rather than its all-in-one component, so that:
// - a screen reader can follow it: the dialog is named, the search box is a combobox pointing at the selected option,
//   and a polite status line says how many match and which is selected;
// - each group shows at most a few until something is typed, so people and pets don't push Settings off the list;
// - its shortcut works in a text field too, but never opens over another dialog (a confirm, Settings' drawer).
import '@mantine/spotlight/styles.css';
import { useHotkeys } from '@mantine/hooks';
import { Spotlight, useSpotlight } from '@mantine/spotlight';
import { IconSearch } from '@tabler/icons-react';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { openAskHere } from '../ask/openAsk';
import { isoDay } from '../lib/format';
import { getPeople, PEOPLE_KEY } from '../people/api';
import { getPets, PETS_KEY, SIGHTINGS_HASH } from '../people/petLinks';
import { KNOWN_TABS, navigate, TABS } from '../router';
import { SECTIONS } from '../settings/catalog';
import { dayOf } from './day';
import { palette, paletteStore } from './store';

/** One thing the palette can jump to. */
export interface PaletteAction {
  id: string;
  label: string;
  description?: string;
  keywords?: readonly string[];
  go: () => void;
}

export interface PaletteGroup {
  group: string;
  actions: PaletteAction[];
}

const to = (path: string) => () => navigate(path);

/** Settings opens as a drawer that gives focus back, when it closes, to what had it when it opened. From the palette
 *  that would be its own search box, gone by then, so focus goes to the Go to button first (as ShellMenu does). */
const toSettings = (path: string) => () => {
  document.querySelector<HTMLElement>('[data-palette-trigger]')?.focus();
  navigate(path);
};

/** A long date for a day's action ("Wednesday, November 4"). */
const longDay = (day: string): string =>
  new Date(`${day}T12:00:00`).toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });

/** Everything the palette can jump to for `query` (the typed text), in its groups, before filtering. */
export function paletteActions(
  query: string,
  people: readonly string[],
  pets: readonly string[],
  now: Date = new Date(),
): PaletteGroup[] {
  const day = dayOf(query, now);
  const groups: PaletteGroup[] = [
    ...(day
      ? [
          {
            group: 'Review',
            actions: [
              { id: `day:${day}`, label: `Review ${longDay(day)}`, keywords: [query], go: to(`/review#day=${day}`) },
            ],
          },
        ]
      : []),
    {
      group: 'Next steps',
      actions: [
        // The one Ask, as the top bar opens it (on Review, with its day as a hint)
        { id: 'do:ask', label: 'Ask', description: 'A question about your recordings', go: () => openAskHere() },
        {
          id: 'do:whos-this',
          label: "Who's this?",
          description: 'Faces and voices to name',
          go: to('/people#whos-this'),
        },
        {
          id: 'do:sightings',
          label: 'Sightings to name',
          description: 'Animals to name',
          go: to(`/pets${SIGHTINGS_HASH}`),
        },
        { id: 'do:clips', label: 'Clips to check', description: 'Activities', go: to('/activities') },
        // Today when it's clicked, named in the address: Review stays on the day it showed without one.
        {
          id: 'do:today',
          label: 'Review today',
          keywords: ['today'],
          go: () => navigate(`/review#day=${isoDay(new Date())}`),
        },
      ],
    },
    {
      group: 'Pages',
      actions: [
        ...TABS.map(t => ({ id: `page:${t.id}`, label: t.label, go: to(t.path) })),
        ...KNOWN_TABS.map(t => ({ id: `page:${t.id}`, label: t.label, description: 'Known', go: to(t.path) })),
        { id: 'page:settings', label: 'Settings', go: toSettings('/settings') },
      ],
    },
    {
      group: 'People',
      actions: people.map(name => ({
        id: `person:${name}`,
        label: name,
        go: to(`/people#card=${encodeURIComponent(name)}`),
      })),
    },
    {
      group: 'Pets',
      actions: pets.map(name => ({ id: `pet:${name}`, label: name, go: to(`/pets#pet=${encodeURIComponent(name)}`) })),
    },
    {
      group: 'Settings',
      actions: SECTIONS.map(s => ({ id: `settings:${s.id}`, label: s.title, go: toSettings(`/settings#${s.id}`) })),
    },
  ];
  return groups.filter(g => g.actions.length);
}

/** How many of a group show before anything is typed; typing shows every match. */
export const SHOWN_UNTYPED = 6;

/** The groups' actions that match `query` (every word in the label, description, or keywords), each group capped at
 *  SHOWN_UNTYPED while nothing is typed. */
export function filterGroups(groups: readonly PaletteGroup[], query: string): PaletteGroup[] {
  const words = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  return groups
    .map(g => {
      const hits = g.actions.filter(a => {
        const text = [a.label, a.description ?? '', ...(a.keywords ?? [])].join(' ').toLocaleLowerCase();
        return words.every(w => text.includes(w)) || (a.keywords ?? []).includes(query);
      });
      return { group: g.group, actions: words.length ? hits : hits.slice(0, SHOWN_UNTYPED) };
    })
    .filter(g => g.actions.length);
}

/** Whether another dialog is showing (a confirm, Settings' drawer): the palette's shortcut leaves it alone, as Ask's "/"
 *  does (AskBar.tsx), counting only one that's visible, since some drawers stay mounted while closed. */
const otherDialogOpen = () =>
  [...document.querySelectorAll<HTMLElement>('[aria-modal=true]')].some(
    el =>
      !el.closest('[data-palette]') &&
      (typeof el.checkVisibility === 'function' ? el.checkVisibility() : getComputedStyle(el).display !== 'none'),
  );

/** The results list's id: the search box's aria-controls, and where option ids are set. */
const LIST_ID = 'palette-list';

export function Palette() {
  const { opened, selected } = useSpotlight(paletteStore);
  const [query, setQuery] = useState('');
  const people = useQuery({ queryKey: PEOPLE_KEY, queryFn: ({ signal }) => getPeople({ signal }), enabled: opened });
  const pets = useQuery({ queryKey: PETS_KEY, queryFn: ({ signal }) => getPets(signal), enabled: opened });
  // Ctrl+K (Cmd+K) in a text field too (it's a chord, so it can't be typed by accident), never over another dialog.
  useHotkeys(
    [
      [
        'mod+K',
        () => {
          if (opened) palette.close();
          else if (!otherDialogOpen()) palette.open();
        },
      ],
    ],
    [],
  );
  const groups = filterGroups(
    paletteActions(
      query,
      (people.data?.people ?? []).map(p => p.name),
      (pets.data?.pets ?? []).map(p => p.name),
    ),
    query,
  );
  const flat = groups.flatMap(g => g.actions);
  const optionId = (i: number) => `palette-option-${i}`;
  const current = selected >= 0 ? flat[selected] : undefined;
  const status = !flat.length
    ? 'Nothing matches.'
    : `${flat.length === 1 ? '1 match' : `${flat.length} matches`}${current ? `. ${current.label} selected` : ''}.`;
  // Spotlight.Action keeps the id it's given to itself, so each option gets its id here, in the order the store counts
  // them (the list's [data-action] buttons), for the search box's aria-activedescendant.
  // The list mounts after the dialog's open transition, later than this component's render, so it's held in state:
  // its arrival re-renders the palette, and the effect then reaches it (a plain lookup ran before it existed).
  const [list, setList] = useState<HTMLDivElement | null>(null);
  useEffect(() => {
    list?.querySelectorAll<HTMLElement>('[data-action]').forEach((el, n) => {
      el.id = optionId(n);
    });
    list?.querySelector<HTMLElement>('[data-scrollarea-viewport]')?.setAttribute('tabindex', '0');
  });
  let i = -1;
  return (
    <Spotlight.Root
      store={paletteStore}
      query={query}
      onQueryChange={setQuery}
      shortcut={null}
      title="Go to"
      styles={{ header: { display: 'none' } }}
      scrollable
      maxHeight="min(60vh, 480px)"
      data-palette
    >
      <Spotlight.Search
        leftSection={<IconSearch size={18} aria-hidden />}
        placeholder="Go to a page, person, pet, setting, or day"
        aria-label="Go to"
        role="combobox"
        aria-expanded={opened}
        aria-controls={LIST_ID}
        aria-autocomplete="list"
        aria-activedescendant={current ? optionId(selected) : undefined}
      />
      {/* A fixed id, so the search box names its list from the first render; the viewport takes focus (its options
          are reached with the arrows), so a keyboard can also scroll it. */}
      <Spotlight.ActionsList ref={setList} id={LIST_ID} role="listbox" aria-label="Places to go">
        {groups.map(g => (
          <Spotlight.ActionsGroup key={g.group} label={g.group}>
            {g.actions.map(a => {
              i += 1;
              return (
                <Spotlight.Action
                  key={a.id}
                  role="option"
                  aria-selected={i === selected}
                  label={a.label}
                  description={a.description}
                  highlightQuery
                  onClick={a.go}
                />
              );
            })}
          </Spotlight.ActionsGroup>
        ))}
        {!flat.length && (
          <Spotlight.Empty>Nothing matches. Try a page, a name, or a day like “yesterday”.</Spotlight.Empty>
        )}
      </Spotlight.ActionsList>
      <div role="status" aria-live="polite" style={HIDDEN}>
        {opened ? status : ''}
      </div>
    </Spotlight.Root>
  );
}

/** Read by screen readers, not shown. */
const HIDDEN = {
  position: 'absolute',
  width: 1,
  height: 1,
  overflow: 'hidden',
  clipPath: 'inset(50%)',
  whiteSpace: 'nowrap',
} as const;
