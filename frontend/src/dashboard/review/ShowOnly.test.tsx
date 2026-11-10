import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readPrefs, writePrefs } from '../../shared/prefs';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { usePrefs } from '../lib/prefs';
import { spkColor } from '../lib/speakers';
import { liveKey } from '../live/feed';
import { EngineContext } from './context';
import { ReviewEngine } from './engine';
import { activityChips, UNKNOWN_FACES } from './lanes';
import { ShowOnly } from './ShowOnly';
import * as spanLogic from './spans';
import { Timeline } from './Timeline';
import type { Chip } from './types';

const DAY = '2026-10-27';
const D0 = new Date(2026, 9, 27).getTime() / 1000;
const chip = (kind: string, label: string, id = `${kind}:${label.toLowerCase()}`): Chip => ({
  id,
  kind,
  label,
  spans: [[D0 + 3600, D0 + 3700]],
  total_s: 100,
  count: 2,
});
const CHIPS: Chip[] = [
  chip('person', 'Dana'),
  chip('person', 'Eli'),
  chip('person', 'Unknown faces', UNKNOWN_FACES),
  chip('object', 'Mug'),
  chip('event', 'Lunch'),
];

function setup(chips: Chip[] = CHIPS, person = '', cams = ['cam0']) {
  writePrefs({ ...readPrefs(), reviewTimeline: 'expanded' }); // these cases exercise the labelled full lanes
  usePrefs.setState(readPrefs(), true);
  const engine = new ReviewEngine({ cams, fetchNotes: async () => ({ cams: {} }) });
  engine.pickDay(DAY);
  if (person) engine.pickPerson(person);
  const gen = engine.ui.getState().gen;
  engine.setDayData(DAY, gen, { entries: [], segments: [], spans: [] });
  engine.setChips(DAY, gen, chips);
  const page = renderPage(
    <EngineContext value={engine}>
      <ShowOnly chips={chips} />
      <Timeline />
    </EngineContext>,
  );
  return { engine, ...page };
}
const showOnly = () => within(screen.getByRole('group', { name: 'Show only' }));
const sel = (e: ReviewEngine) => [...e.ui.getState().sel];

beforeEach(() => {
  server.use(
    http.get('/speakers', () => HttpResponse.json({ speakers: [{ id: 4, label: 'Dana', n: 3 }], merged: {} })),
  );
});
afterEach(() => history.replaceState(null, '', '/'));

describe('Show only', () => {
  it('groups people and unknown faces together, followed by things and events', () => {
    setup();
    const groups = showOnly()
      .getAllByRole('group')
      .filter(g => g.getAttribute('aria-labelledby')?.includes('-g'));
    expect(groups.map(g => g.textContent?.match(/^(People|Unknown faces|Things|Events)/)?.[0])).toEqual([
      'People',
      'Things',
      'Events',
    ]);
    expect(
      within(showOnly().getByRole('group', { name: 'People' })).getByRole('button', { name: /Unknown faces/ }),
    ).toBeInTheDocument();
    expect(
      within(showOnly().getByRole('group', { name: 'People' }))
        .getAllByRole('button')
        .map(b => b.textContent),
    ).toEqual(['Dana2', 'Eli2', 'Unknown faces2']);
  });

  it('picks and unpicks with a click, and Show all clears every pick', async () => {
    const { engine } = setup();
    const dana = showOnly().getByRole('button', { name: /^Dana/ });
    const mug = showOnly().getByRole('checkbox', { name: /^Mug/ }); // things and events are pills
    await userEvent.click(dana);
    await userEvent.click(mug);
    expect(dana).toHaveAttribute('aria-pressed', 'true');
    expect(mug).toBeChecked();
    expect(sel(engine).sort()).toEqual(['object:mug', 'person:dana']);
    await userEvent.click(dana);
    expect(sel(engine)).toEqual(['object:mug']);
    await userEvent.click(showOnly().getByRole('button', { name: 'Show all' }));
    expect(sel(engine)).toEqual([]);
    expect(mug).not.toBeChecked();
    expect(showOnly().getByRole('button', { name: 'Show all' })).toBeDisabled();
  });

  it('starts with the person Review was opened for (#person=), once, so hiding them sticks', async () => {
    const { engine, unmount } = setup(CHIPS, 'dana');
    await waitFor(() => expect(sel(engine)).toEqual(['person:dana']));
    await userEvent.click(showOnly().getByRole('button', { name: /^Dana/ }));
    // Shown again, even remounted: Dana stays hidden.
    unmount();
    renderPage(
      <EngineContext value={engine}>
        <ShowOnly chips={[...CHIPS]} />
      </EngineContext>,
    );
    await screen.findByRole('group', { name: 'Show only' });
    expect(sel(engine)).toEqual([]);
  });

  it("picks the person once the engine has the day's chips, even when they arrive after Show only renders", async () => {
    // Review renders Show only with the chips first, then hands them to the engine, which clears the selection.
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.pickDay(DAY);
    engine.pickPerson('dana');
    const gen = engine.ui.getState().gen;
    engine.setDayData(DAY, gen, { entries: [], segments: [], spans: [] });
    renderPage(
      <EngineContext value={engine}>
        <ShowOnly chips={CHIPS} />
      </EngineContext>,
    );
    await screen.findByRole('group', { name: 'Show only' });
    act(() => engine.setChips(DAY, gen, CHIPS));
    await waitFor(() => expect(sel(engine)).toEqual(['person:dana']));
  });

  it('finds chips by name once there are many, and says when nothing matches', async () => {
    const many = Array.from({ length: 14 }, (_, i) => chip('object', `Thing ${i}`));
    setup(many);
    const find = showOnly().getByRole('searchbox', { name: 'Find people, things, and events' });
    await userEvent.type(find, 'thing 1');
    expect(showOnly().getAllByRole('checkbox', { name: /^Thing 1/ })).toHaveLength(5); // 1, 10, 11, 12, 13
    await userEvent.clear(find);
    await userEvent.type(find, 'zebra');
    expect(showOnly().getByText('Nothing matches “zebra”.')).toBeInTheDocument();
  });

  it('says when the day has nothing to show', () => {
    setup([]);
    expect(showOnly().getByText('No people, things, or events found for this day yet.')).toBeInTheDocument();
  });
});

describe('the timeline lanes', () => {
  it('labels Who spoke, each camera by its name, the audio, and Events', () => {
    vi.stubGlobal('NAMES', { views: { cam0: 'front camera' } });
    try {
      const { container } = setup();
      expect([...container.querySelectorAll('span[style*="top"]')].map(s => s.textContent)).toEqual([
        'Who spoke',
        'Front camera',
        'Audio kept',
        'Events',
      ]);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('a drag on the timeline', () => {
  it("ends cleanly when the day's chips change mid-drag", async () => {
    const { engine, container } = setup();
    const track = container.querySelector('[role="slider"]');
    if (!track) throw new Error('no track');
    fireEvent.pointerDown(track, { pointerId: 1, clientX: 100, clientY: 50 }); // in a lane: a scrub
    fireEvent.pointerMove(track, { pointerId: 1, clientX: 120, clientY: 50 });
    act(() => engine.setChips(DAY, engine.ui.getState().gen, [...CHIPS])); // a new layout mid-drag
    fireEvent.pointerUp(track, { pointerId: 1, clientX: 120, clientY: 50 });
    expect((engine as unknown as { scrubbing: boolean }).scrubbing).toBe(false);
  });
});

describe('a click on the timeline', () => {
  const W = 1000; // the track's width: jsdom lays nothing out, so each test gives it this
  const at = (engine: ReviewEngine, ts: number) => ((ts - engine.view.t0) / (engine.view.t1 - engine.view.t0)) * W;
  const click = (track: Element, x: number, y: number, to = x) => {
    fireEvent.pointerDown(track, { pointerId: 1, clientX: x, clientY: y });
    if (to !== x) fireEvent.pointerMove(track, { pointerId: 1, clientX: to, clientY: y });
    fireEvent.pointerUp(track, { pointerId: 1, clientX: to, clientY: y });
  };
  const crossing = {
    id: 'crossing',
    camera: 'cam0',
    start: D0 + 3600,
    end: D0 + 3600,
    count: null,
    name: null,
    kind: 'crossing',
    person: null,
    person_name: 'Dana',
    track_id: 'track-a',
    promoted: false,
  };
  function open(chips: Chip[] = [...CHIPS, ...activityChips([crossing])]) {
    const fetched: string[] = [];
    server.use(
      http.get('/history/at', ({ request }) => {
        fetched.push(new URL(request.url).pathname);
        return HttpResponse.json({ cams: {} });
      }),
      http.get('/activities', ({ request }) => {
        fetched.push(new URL(request.url).pathname);
        return HttpResponse.json({ items: [crossing], truncated: false });
      }),
    );
    const page = setup(chips);
    const track = page.container.querySelector('[role="slider"]') as HTMLElement;
    Object.defineProperty(track, 'clientWidth', { configurable: true, value: W });
    const labels = [...page.container.querySelectorAll('span[style*="top"]')] as HTMLElement[];
    const lane = (name: string) => Number.parseFloat(labels.find(l => l.textContent === name)?.style.top ?? '0');
    page.engine.setView(D0, D0 + 7200);
    return { ...page, track, lane, fetched };
  }

  // 7200 s over 1000 px: 7.2 s a pixel, so a click 5 px past a mark lands 36 s after it
  it.each([
    ['an event tick and an activity marker', 'Events', 3600],
    ['empty lane', 'Audio kept', 5000],
    ['the ruler', null, 5000],
  ])('seeks to where a click on %s lands, and opens and fetches nothing', async (_, name, offset) => {
    const { engine, track, lane, fetched } = open();
    const jump = vi.spyOn(engine, 'jump').mockImplementation(() => {});
    const seekActivity = vi.spyOn(engine, 'seekActivity');
    click(track, at(engine, D0 + offset) + 5, name ? lane(name) + 20 : 4);
    expect(jump).toHaveBeenCalledExactlyOnceWith(D0 + offset + 36);
    expect(seekActivity).not.toHaveBeenCalled();
    await new Promise(r => setTimeout(r, 20)); // a lazy fetch would have gone out by now
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(fetched).toEqual([]);
  });

  it('seeks on a mark right by the playhead, where a press starts a scrub', () => {
    const { engine, track, lane } = open();
    engine.jump(D0 + 3600); // the playhead sits on the tick
    click(track, at(engine, D0 + 3600) + 5, lane('Events') + 20);
    expect(engine.t).toBe(D0 + 3636);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('takes a press that slips 4 px as a click, seeking to where it was let go', () => {
    const { engine, track, lane } = open();
    const jump = vi.spyOn(engine, 'jump').mockImplementation(() => {});
    const x = at(engine, D0 + 3600);
    click(track, x, lane('Events') + 20, x + 4);
    expect(jump).toHaveBeenCalledExactlyOnceWith(D0 + 3600 + 4 * 7.2);
  });

  it('opens nothing on Enter', async () => {
    const { engine, track } = open();
    engine.jump(D0 + 3620);
    track.focus();
    await userEvent.keyboard('{Enter}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('does nothing for a touch the browser took back', () => {
    const { engine, track, lane } = open();
    const jump = vi.spyOn(engine, 'jump');
    const x = at(engine, D0 + 3600);
    fireEvent.pointerDown(track, { pointerId: 3, clientX: x, clientY: lane('Events') + 20 });
    fireEvent.pointerCancel(track, { pointerId: 3, clientX: x, clientY: lane('Events') + 20 });
    expect(jump).not.toHaveBeenCalled();
  });
});

it('offers one Events filter per activity kind and keeps a crossing point selectable', async () => {
  const activity = {
    camera: 'cam0',
    start: D0 + 3600,
    end: D0 + 3600,
    count: null,
    name: null,
    kind: 'crossing',
    person: null,
    promoted: false,
  };
  const chips = activityChips(
    Array.from({ length: 200 }, (_, i) => ({
      ...activity,
      id: `activity-${i}`,
      start: D0 + 3600 + i * 10,
      end: D0 + 3600 + i * 10,
    })),
  );
  const { engine } = setup(chips);
  const events = within(showOnly().getByRole('group', { name: 'Events' }));
  expect(events.getAllByRole('checkbox')).toHaveLength(1);
  await userEvent.click(events.getByRole('checkbox', { name: /Walked past/ }));
  expect(engine.ui.getState().sel.has('event:activity:crossing')).toBe(true);
  engine.stepChip(1);
  expect(engine.t).toBe(D0 + 3600);
});

it('offers camera display names, filters timeline lanes, and clears cameras with Show all', async () => {
  vi.stubGlobal('NAMES', { views: { cam0: 'front camera' } });
  history.replaceState(null, '', '/review'); // Review writes its link only on its own page
  try {
    const { engine, container } = setup();
    await userEvent.click(showOnly().getByRole('checkbox', { name: 'Front camera' }));
    expect(engine.ui.getState().cameras).toEqual(['cam0']);
    expect(location.hash).toContain('cameras=cam0');
    expect([...container.querySelectorAll('span[style*="top"]')].some(e => e.textContent === 'Front camera')).toBe(
      true,
    );
    await userEvent.click(showOnly().getByRole('button', { name: 'Show all' }));
    expect(engine.ui.getState().cameras).toEqual([]);
    expect(location.hash).not.toContain('cameras=');
  } finally {
    vi.unstubAllGlobals();
  }
});

describe('co-occurring filter alternatives', () => {
  const timed = (kind: string, label: string, spans: [number, number][], more: Partial<Chip> = {}): Chip => ({
    ...chip(kind, label),
    spans: spans.map(([a, b]) => [D0 + a, D0 + b]),
    ...more,
  });
  // People are toggle buttons; things and events are pill checkboxes.
  const option = (name: string) =>
    showOnly().queryByRole('checkbox', { name: new RegExp(`^${name}\\d`) }) ??
    showOnly().getByRole('button', { name: new RegExp(`^${name}\\d`) });

  it('updates temporal availability on real toggles, ORs same-facet alternatives, and ANDs other facets', async () => {
    const { engine } = setup([
      timed('person', 'Alice', [[10, 20]]),
      timed('person', 'Dana', [[30, 40]]),
      timed('object', 'Mug', [[15, 35]]),
      timed('object', 'Lamp', [[50, 60]]),
      timed('event', 'Lunch', [[18, 19]]),
    ]);
    await userEvent.click(option('Alice'));
    expect(option('Lamp')).toHaveAttribute('data-unrelated', 'true');
    expect(option('Dana')).not.toHaveAttribute('data-unrelated'); // No other facet restricts People yet.
    await userEvent.click(option('Mug'));
    expect(option('Dana')).not.toHaveAttribute('data-unrelated'); // Dana overlaps the chosen thing.
    await userEvent.click(option('Lunch'));
    expect(option('Dana')).toHaveAttribute('data-unrelated', 'true');
    engine.stepChip(1);
    expect(engine.t).toBe(D0 + 18); // Navigation now uses the shared cross-facet intersection.
    await userEvent.hover(option('Dana'));
    expect(
      await screen.findByText(/Never at the same time as Mug and Lunch/, { selector: '[role="tooltip"]' }),
    ).toHaveTextContent('You can still choose it.');
    await userEvent.unhover(option('Dana'));
    expect(option('Dana')).toBeEnabled();
    await userEvent.click(option('Dana'));
    expect(sel(engine)).toEqual(['person:alice', 'object:mug', 'event:lunch', 'person:dana']);
    expect(option('Dana')).toHaveAttribute('aria-pressed', 'true');
    expect(option('Dana')).not.toHaveAttribute('data-unrelated'); // Selected options can always be removed.
    await userEvent.click(option('Lunch'));
    await userEvent.click(option('Dana'));
    expect(option('Dana')).not.toHaveAttribute('data-unrelated'); // Removing the Event facet restores Dana.
  });

  it('keeps unavailable timing neutral and retains the supplied counts', async () => {
    setup([
      timed('person', 'Alice', [[10, 20]]),
      timed('object', 'Unknown', [], { count: 7 }),
      timed('object', 'Invalid', [[Number.NaN, 30]], { count: 9 }),
    ]);
    await userEvent.click(option('Alice'));
    expect(option('Unknown')).not.toHaveAttribute('data-unrelated');
    expect(option('Invalid')).not.toHaveAttribute('data-unrelated');
    expect(option('Unknown')).toHaveAccessibleName('Unknown7');
    expect(option('Invalid')).toHaveAccessibleName('Invalid9');
    await userEvent.click(option('Unknown'));
    expect(option('Invalid')).not.toHaveAttribute('data-unrelated');
  });

  it('dims camera-eliminated options without hiding them and keeps camera-less evidence available', async () => {
    const { engine } = setup(
      [
        timed(
          'person',
          'Alice',
          [
            [10, 20],
            [100, 110],
          ],
          { cams: { cam0: [[D0 + 10, D0 + 20]], cam1: [[D0 + 100, D0 + 110]] } },
        ),
        timed('object', 'Mug', [[15, 18]], { camera: 'cam0' }),
        timed('event', 'Lunch', [[105, 108]], { camera: 'cam1' }),
        timed('event', 'Speech', [[16, 17]]),
      ],
      '',
      ['cam0', 'cam1'],
    );
    await userEvent.click(option('Alice'));
    const first = showOnly().getByRole('checkbox', { name: 'cam0' });
    const second = showOnly().getByRole('checkbox', { name: 'cam1' });
    await userEvent.click(first);
    expect(option('Lunch')).toHaveAttribute('data-unrelated', 'true');
    expect(option('Speech')).not.toHaveAttribute('data-unrelated');
    expect(option('Alice')).not.toHaveAttribute('data-unrelated');
    await userEvent.click(option('Mug'));
    expect(second.closest('[data-unrelated]')).not.toBeNull();
    await userEvent.hover(second.closest('[data-unrelated]') as HTMLElement);
    expect(
      await screen.findByText(/Never at the same time as Alice and Mug/, {
        selector: '[role="tooltip"]',
      }),
    ).toBeInTheDocument();
    await userEvent.unhover(second.closest('[data-unrelated]') as HTMLElement);
    await userEvent.click(second);
    expect(engine.ui.getState().cameras).toEqual(['cam0', 'cam1']);
    expect(second).toBeChecked();
    expect(second.closest('[data-unrelated]')).toBeNull();
    await userEvent.click(first);
    await userEvent.click(second);
    expect(engine.ui.getState().cameras).toEqual([]); // No checked cameras still means all cameras.
    expect(showOnly().getByRole('status')).toHaveTextContent('all cameras');
    await userEvent.click(showOnly().getByRole('button', { name: 'Show all' }));
    expect(sel(engine)).toEqual([]);
    expect(option('Lunch')).not.toHaveAttribute('data-unrelated');
  });
  it('reuses overlap calculations while searching or speaker colors update, and refreshes after a toggle', async () => {
    const calculation = vi.spyOn(spanLogic, 'nonCooccurring');
    const { client } = setup([
      timed('person', 'Alice', [[10, 20]]),
      ...Array.from({ length: 14 }, (_, i) => timed('object', `Thing ${i}`, [[i, i + 1]])),
    ]);
    const calls = calculation.mock.calls.length;
    const search = showOnly().getByRole('searchbox', { name: 'Find people, things, and events' });
    await userEvent.type(search, 'Thing');
    expect(calculation).toHaveBeenCalledTimes(calls);
    await userEvent.clear(search);
    await act(async () => {
      client.setQueryData(liveKey('speakers'), { list: [{ id: 4, label: 'Alice', n: 3 }], merged: {} });
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    await waitFor(() => expect(option('Alice').style.getPropertyValue('--so-color')).toBe(spkColor(4)));
    expect(calculation).toHaveBeenCalledTimes(calls);
    await userEvent.click(option('Alice'));
    expect(calculation).toHaveBeenCalledTimes(calls + 1);
  });
});

it('keeps moment navigation buttons out of Show only', () => {
  setup();
  expect(showOnly().queryByRole('button', { name: 'Previous' })).not.toBeInTheDocument();
  expect(showOnly().queryByRole('button', { name: 'Next' })).not.toBeInTheDocument();
});

describe('a long group', () => {
  // 13 things, counted 1 to 13: the 10 most common are Thing 4 to Thing 13. A chip's name is its label, then its
  // count ("Thing 33" is Thing 3, seen 3 times).
  const things = Array.from({ length: 13 }, (_, i) => ({ ...chip('object', `Thing ${i + 1}`), count: i + 1 }));
  it('lists its 10 most common first, then all of them behind +N more, and back', async () => {
    setup(things);
    expect(showOnly().queryByRole('checkbox', { name: 'Thing 33' })).not.toBeInTheDocument();
    expect(showOnly().getAllByRole('checkbox', { name: /^Thing / })).toHaveLength(10);
    await userEvent.click(showOnly().getByRole('button', { name: '+3 more' }));
    expect(showOnly().getAllByRole('checkbox', { name: /^Thing / })).toHaveLength(13);
    await userEvent.click(showOnly().getByRole('button', { name: 'Show fewer' }));
    expect(showOnly().getAllByRole('checkbox', { name: /^Thing / })).toHaveLength(10);
  });

  it('keeps a picked chip in view when it is not among the most common', () => {
    const { engine } = setup(things);
    act(() => engine.toggleChip('object:thing 1'));
    expect(showOnly().getByRole('checkbox', { name: 'Thing 11' })).toBeChecked();
    expect(showOnly().getByRole('button', { name: '+2 more' })).toBeInTheDocument();
  });

  it('searches every chip, not only the ones listed', async () => {
    setup(things);
    await userEvent.type(showOnly().getByRole('searchbox', { name: 'Find people, things, and events' }), 'thing 2');
    expect(showOnly().getByRole('checkbox', { name: 'Thing 22' })).toBeInTheDocument();
    expect(showOnly().queryByRole('button', { name: /more$/ })).not.toBeInTheDocument();
  });
});

it("holds the video's boxes and Skip quiet time on Review's page", async () => {
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.pickDay(DAY);
  renderPage(
    <EngineContext value={engine}>
      <ShowOnly chips={CHIPS} video />
    </EngineContext>,
  );
  await userEvent.click(showOnly().getByRole('checkbox', { name: 'Skip quiet time' }));
  expect(engine.ui.getState().skip).toBe(true);
  expect(within(showOnly().getByRole('group', { name: 'Review markers' })).getAllByRole('checkbox')).toHaveLength(3);
});
